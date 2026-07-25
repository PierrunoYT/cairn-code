use super::process_runner::{self, with_cleanup, RunError, RunOptions};
use super::registry::Tool;
use std::process::Command;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

/// Max chars returned to the model. Prefer head+tail so summaries
/// (e.g. `cargo test` "147 passed") survive even when the middle is huge.
const MAX_OUTPUT_CHARS: usize = 12_000;
const HEAD_CHARS: usize = 6_000;
const TAIL_CHARS: usize = 4_000;
/// Wall-clock cap applied when the model does not ask for one, matching the
/// reasoning behind `git_tool::GIT_TIMEOUT`: a command that never returns —
/// a stuck network call, a server left in the foreground, a prompt waiting on
/// input that can no longer arrive — must not hold the agent forever. Long
/// builds and test suites finish well inside this, and a command that needs
/// more can still ask by passing `timeout`.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(600);
/// Ceiling on an explicit `timeout`. `process_runner` builds its deadline with
/// `Instant::now().checked_add(..)` and degrades to *no* deadline when that
/// overflows, so an absurd value — a model guessing, or a prompt injection
/// picking `u64::MAX` — would otherwise restore the unbounded run this cap
/// exists to prevent. A day is far beyond any legitimate command and leaves
/// `checked_add` no way to fail.
const MAX_TIMEOUT: Duration = Duration::from_secs(24 * 60 * 60);

pub struct ShellTool;

impl Tool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }
    fn description(&self) -> &str {
        "Execute a shell command. On Windows this uses PowerShell (-Command); \
         on Unix it uses bash (-c). For intentional PowerShell work on Windows, \
         prefer the dedicated `powershell` tool. Always check the exit code footer. \
         Bare `git commit` invocations automatically get a Co-Authored-By: cairn-code trailer."
    }
    fn needs_permission(&self) -> bool {
        true
    }

    fn input_schema(&self) -> String {
        r#"{"type":"object","properties":{"command":{"type":"string"},"timeout":{"type":"integer","description":"Wall-clock limit in milliseconds (default 600000, maximum 86400000; larger values are capped)"}},"required":["command"]}"#.into()
    }

    fn execute(&self, input: &str) -> Result<String, String> {
        // Direct/test callers get the same behavior with a token that is never set.
        self.execute_with_cancel(input, &AtomicBool::new(false))
    }

    fn execute_with_cancel(&self, input: &str, cancel: &AtomicBool) -> Result<String, String> {
        let val = crate::json::parse(input).map_err(|e| format!("invalid input: {e}"))?;
        let obj = val.as_object().ok_or("expected object")?;
        let cmd = obj
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or("command required")?;
        let timeout_ms = obj.get("timeout").and_then(|v| v.as_u64());

        // Commits via shell bypass the git tool's --trailer injection; rewrite
        // bare `git commit` so GitHub still co-attributes cairn-code.
        let cmd = if cfg!(windows) {
            crate::tools::commit_attribution::ensure_powershell_command_co_author(cmd)
        } else {
            crate::tools::commit_attribution::ensure_shell_command_co_author(cmd)
        };

        let shell = if cfg!(windows) { "powershell" } else { "bash" };
        let flag = if cfg!(windows) { "-Command" } else { "-c" };

        let mut command = Command::new(shell);
        command.arg(flag).arg(&cmd);

        let options = run_options_for(timeout_ms);
        let result = match process_runner::run(command, &options, Some(cancel)) {
            Ok(result) => result,
            Err(error) => return Err(format_run_error(error)),
        };

        let code = result.code;
        let ok = result.success;

        let stdout = normalize_cli_output(&result.stdout);
        let stderr = normalize_cli_output(&result.stderr);

        let mut body = String::new();
        if !stdout.is_empty() {
            body.push_str(&stdout);
        }
        if !stderr.is_empty() {
            if !body.is_empty() && !body.ends_with('\n') {
                body.push('\n');
            }
            // Label stderr so the model can tell streams apart.
            if !stdout.is_empty() {
                body.push_str("--- stderr ---\n");
            }
            body.push_str(&stderr);
        }

        let body = truncate_head_tail(&body, MAX_OUTPUT_CHARS, HEAD_CHARS, TAIL_CHARS);

        // Always surface exit code. Non-zero used to return a bare "exit code: N"
        // when stdout was empty (e.g. missing `tail` on Windows), and when stdout
        // was non-empty the failure was silent — both confuse the agent.
        let mut result = body;
        if !result.is_empty() && !result.ends_with('\n') {
            result.push('\n');
        }
        result.push_str(&format!("(exit code {code})"));

        if ok {
            Ok(result)
        } else {
            // Prefix so the TUI marks it red; keep full body so the model can recover.
            Err(format!("exit code {code}\n{result}"))
        }
    }
}

/// Run options for a shell invocation.
///
/// Separated from `execute_with_cancel` so the timeout policy can be asserted
/// without spawning a process: an omitted `timeout` must still carry a
/// deadline, an explicit one must win even when it exceeds the default, and
/// no input may produce a duration a deadline cannot be built from.
fn run_options_for(timeout_ms: Option<u64>) -> RunOptions {
    RunOptions {
        timeout: Some(
            timeout_ms
                .map_or(DEFAULT_TIMEOUT, Duration::from_millis)
                .min(MAX_TIMEOUT),
        ),
        head_chars: HEAD_CHARS,
        tail_chars: TAIL_CHARS,
    }
}

/// Turn a [`RunError`] into the shell tool's user-facing error string,
/// preserving the historical "timed out" / "exec error" phrasing.
///
/// `after_ms` is the deadline the run actually enforced, so a request capped
/// by [`MAX_TIMEOUT`] reports the cap rather than the number it asked for.
fn format_run_error(error: RunError) -> String {
    match error {
        RunError::Spawn(message) => format!("exec error: {message}"),
        RunError::TimedOut {
            after_ms,
            cleanup_error,
        } => with_cleanup(
            format!("command timed out after {after_ms}ms"),
            &cleanup_error,
        ),
        RunError::Cancelled { cleanup_error } => {
            with_cleanup("command cancelled".to_string(), &cleanup_error)
        }
        RunError::Wait {
            reason,
            cleanup_error,
        } => with_cleanup(format!("exec error: {reason}"), &cleanup_error),
    }
}

/// Turn CR progress rewrites (`cargo`'s `\r`) into real newlines and normalize CRLF.
fn normalize_cli_output(s: &str) -> String {
    let s = s.replace("\r\n", "\n").replace('\r', "\n");
    // Collapse huge runs of blank lines from progress spam.
    let mut out = String::with_capacity(s.len());
    let mut blank_run = 0usize;
    for line in s.split('\n') {
        if line.trim().is_empty() {
            blank_run += 1;
            if blank_run <= 2 {
                out.push('\n');
            }
        } else {
            blank_run = 0;
            out.push_str(line);
            out.push('\n');
        }
    }
    while out.ends_with('\n') {
        out.pop();
    }
    out
}

/// Keep the beginning and end of large output so status lines at the bottom
/// (test summaries, build results) are not chopped off.
fn truncate_head_tail(s: &str, max: usize, head: usize, tail: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max {
        return s.to_string();
    }
    let head = head.min(chars.len());
    let tail = tail.min(chars.len().saturating_sub(head));
    let start: String = chars[..head].iter().collect();
    let end: String = chars[chars.len() - tail..].iter().collect();
    let omitted = chars.len() - head - tail;
    format!("{start}\n... [{omitted} chars truncated] ...\n{end}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn sleep_command(seconds: u64) -> String {
        if cfg!(windows) {
            format!("Start-Sleep -Seconds {seconds}")
        } else {
            format!("sleep {seconds}")
        }
    }

    #[test]
    fn test_timeout_kills_long_running_command() {
        let tool = ShellTool;
        let cmd = sleep_command(5);
        let input = format!(r#"{{"command":"{cmd}","timeout":300}}"#);
        let err = tool.execute(&input).unwrap_err();
        assert!(err.contains("timed out"), "unexpected error: {err}");
    }

    #[test]
    fn omitted_timeout_still_runs_fast_commands() {
        let tool = ShellTool;
        let input = r#"{"command":"echo hi"}"#.to_string();
        let out = tool.execute(&input).unwrap();
        assert!(out.contains("hi"), "unexpected output: {out}");
        assert!(out.contains("(exit code 0)"), "missing exit footer: {out}");
    }

    #[test]
    fn omitting_timeout_applies_the_default_rather_than_running_unbounded() {
        // The model usually omits `timeout`. Previously that meant no deadline
        // at all, so a command that never returns held the agent until the
        // user noticed. Assert the option carries a deadline either way.
        let with_default = run_options_for(None);
        assert_eq!(with_default.timeout, Some(DEFAULT_TIMEOUT));

        // An explicit value still wins, including one longer than the default.
        assert_eq!(
            run_options_for(Some(1_500)).timeout,
            Some(Duration::from_millis(1_500))
        );
        assert_eq!(
            run_options_for(Some(3_600_000)).timeout,
            Some(Duration::from_millis(3_600_000))
        );
    }

    #[test]
    fn oversized_timeout_is_capped_so_a_deadline_always_exists() {
        // `process_runner` builds the deadline with `checked_add` and falls
        // back to "no deadline" on overflow, so an unclamped huge value would
        // run unbounded — the exact hang this tool's timeout exists to stop.
        for requested in [
            u64::MAX,
            u64::MAX / 2,
            1 << 53,
            MAX_TIMEOUT.as_millis() as u64 + 1,
        ] {
            let timeout = run_options_for(Some(requested)).timeout.unwrap();
            assert_eq!(timeout, MAX_TIMEOUT, "not capped: {requested}ms");
            assert!(
                Instant::now().checked_add(timeout).is_some(),
                "no deadline for {requested}ms"
            );
        }

        // The cap only bites above itself; a value just under it is untouched.
        let under = MAX_TIMEOUT - Duration::from_millis(1);
        assert_eq!(
            run_options_for(Some(under.as_millis() as u64)).timeout,
            Some(under)
        );
    }

    #[test]
    fn test_generous_timeout_does_not_interrupt_fast_command() {
        let tool = ShellTool;
        let input = r#"{"command":"echo hi","timeout":30000}"#;
        let out = tool.execute(input).unwrap();
        assert!(out.contains("hi"), "unexpected output: {out}");
    }

    #[test]
    fn test_large_stdout_and_stderr_do_not_deadlock() {
        let tool = ShellTool;
        let cmd = if cfg!(windows) {
            "1..20000 | ForEach-Object { Write-Output ('stdout padding ' + $_); [Console]::Error.WriteLine(('stderr padding ' + $_)) }"
        } else {
            "for i in $(seq 1 20000); do echo stdout-padding-$i; echo stderr-padding-$i >&2; done"
        };
        let input = format!(r#"{{"command":"{cmd}","timeout":20000}}"#);
        let out = tool.execute(&input).unwrap();
        assert!(
            out.contains("stdout padding 1") || out.contains("stdout-padding-1"),
            "lost stdout: {out}"
        );
        assert!(
            out.contains("stderr padding 20000") || out.contains("stderr-padding-20000"),
            "lost stderr: {out}"
        );
        assert!(out.contains("(exit code 0)"), "missing exit footer: {out}");
        assert!(out.chars().count() <= MAX_OUTPUT_CHARS + 100);
    }

    #[cfg(unix)]
    #[test]
    fn test_timeout_kills_descendants() {
        let marker = std::env::temp_dir().join(format!(
            "cairn-shell-descendant-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        // Let the shell exit immediately. The descendant keeps the inherited
        // pipes open, so timeout handling must still kill its process group.
        let command = format!("(sleep 1; printf survived > '{}') &", marker.display());
        let input = serde_json::json!({ "command": command, "timeout": 200 }).to_string();

        let err = ShellTool.execute(&input).unwrap_err();
        assert!(err.contains("timed out"), "unexpected error: {err}");
        std::thread::sleep(Duration::from_millis(1200));
        assert!(
            !marker.exists(),
            "descendant survived timeout and wrote {}",
            marker.display()
        );
    }

    #[cfg(windows)]
    #[test]
    fn test_timeout_kills_descendants() {
        let marker = std::env::temp_dir().join(format!(
            "cairn-shell-descendant-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        // PowerShell accepts forward slashes; also double-quote for LiteralPath.
        let marker_ps = marker
            .display()
            .to_string()
            .replace('\\', "/")
            .replace('\'', "''");
        let command = format!(
            "Start-Process powershell -ArgumentList @('-NoProfile','-Command','Start-Sleep -Milliseconds 1000; Set-Content -LiteralPath ''{marker_ps}'' -Value survived'); Start-Sleep -Seconds 5"
        );
        // marker_ps uses forward slashes for PowerShell -LiteralPath; keep
        // marker as PathBuf for the filesystem assertion below.
        let input = serde_json::json!({
            "command": command,
            "timeout": 200
        })
        .to_string();

        let err = ShellTool.execute(&input).unwrap_err();
        assert!(err.contains("timed out"), "unexpected error: {err}");
        std::thread::sleep(Duration::from_millis(1200));
        assert!(
            !marker.exists(),
            "descendant survived timeout and wrote {}",
            marker.display()
        );
    }

    #[test]
    fn normalize_turns_cr_into_newlines() {
        let s = normalize_cli_output("a\rb\r\nc");
        assert!(s.contains('\n'));
        assert!(!s.contains('\r'));
        assert!(s.contains('a') && s.contains('b') && s.contains('c'));
    }

    #[test]
    fn truncate_keeps_head_and_tail() {
        let s = format!("{}{}{}", "H".repeat(100), "M".repeat(5000), "T".repeat(100));
        let out = truncate_head_tail(&s, 500, 100, 100);
        assert!(out.starts_with(&"H".repeat(100)));
        assert!(out.ends_with(&"T".repeat(100)));
        assert!(out.contains("truncated"));
        assert!(!out.contains(&"M".repeat(50)));
    }

    #[test]
    fn shell_git_commit_gets_co_author_trailer() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("cairn-shell-coauthor-{nanos}"));
        std::fs::create_dir_all(&dir).unwrap();
        let dir_str = dir.to_string_lossy().replace('\\', "/");

        let setup = if cfg!(windows) {
            format!(
                "Set-Location -LiteralPath '{dir_str}'; git init -q; git config user.email t@e.com; git config user.name t; Set-Content -Path a.txt -Value a; git add a.txt; git commit -m 'shell-commit'"
            )
        } else {
            format!(
                "cd '{dir_str}' && git init -q && git config user.email t@e.com && git config user.name t && echo a > a.txt && git add a.txt && git commit -m 'shell-commit'"
            )
        };
        let input = serde_json::json!({ "command": setup, "timeout": 60_000 }).to_string();
        let out = ShellTool.execute(&input);
        // Some environments may lack git; skip soft if so.
        let body = match out {
            Ok(s) => s,
            Err(e) if e.contains("git") || e.to_ascii_lowercase().contains("not recognized") => {
                let _ = std::fs::remove_dir_all(&dir);
                return;
            }
            Err(e) => panic!("setup/commit failed: {e}"),
        };
        assert!(
            body.contains("(exit code 0)") || !body.contains("exit code"),
            "{body}"
        );

        let log_cmd = if cfg!(windows) {
            format!("Set-Location -LiteralPath '{dir_str}'; git log -1 --format=%B")
        } else {
            format!("cd '{dir_str}' && git log -1 --format=%B")
        };
        let log_input = serde_json::json!({ "command": log_cmd, "timeout": 30_000 }).to_string();
        let log = ShellTool.execute(&log_input).unwrap_or_default();
        assert!(
            log.to_ascii_lowercase().contains("co-authored-by:") && log.contains("cairn-code"),
            "expected cairn-code trailer in commit via shell, got: {log}"
        );
        assert!(log.contains("shell-commit"), "{log}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn failed_command_includes_body_not_bare_code() {
        let tool = ShellTool;
        let cmd = if cfg!(windows) {
            "Write-Output 'visible-fail-body'; exit 7"
        } else {
            "echo visible-fail-body; exit 7"
        };
        let input = format!(r#"{{"command":"{cmd}"}}"#);
        let err = tool.execute(&input).unwrap_err();
        assert!(
            err.contains("visible-fail-body"),
            "lost stdout on failure: {err}"
        );
        assert!(err.contains("exit code"), "missing exit code: {err}");
    }
}
