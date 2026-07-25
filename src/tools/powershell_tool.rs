//! Native PowerShell tool. Prefer this on Windows instead of shelling through
//! bash/git-bash style commands. Resolves `pwsh` first, then Windows PowerShell 5.1.

use std::collections::VecDeque;
use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use super::registry::Tool;

const MAX_OUTPUT_CHARS: usize = 12_000;
const HEAD_CHARS: usize = 6_000;
const TAIL_CHARS: usize = 4_000;
const DEFAULT_TIMEOUT_MS: u64 = 120_000;

pub struct PowerShellTool;

impl Tool for PowerShellTool {
    fn name(&self) -> &str {
        "powershell"
    }

    fn description(&self) -> &str {
        "Run a PowerShell command natively (pwsh if available, else Windows PowerShell). \
         Use PowerShell syntax (Select-Object, Get-ChildItem, Select-String, etc.) — not bash. \
         Prefer this over shell for Windows-native work. Check the exit code footer. \
         Bare `git commit` invocations automatically get a Co-Authored-By: cairn-code trailer."
    }

    fn needs_permission(&self) -> bool {
        true
    }

    fn input_schema(&self) -> String {
        r#"{"type":"object","properties":{"command":{"type":"string","description":"PowerShell command or script block text"},"timeout":{"type":"integer","description":"Timeout in milliseconds (default 120000)"}},"required":["command"]}"#.into()
    }

    fn execute(&self, input: &str) -> Result<String, String> {
        let val = crate::json::parse(input).map_err(|e| format!("invalid input: {e}"))?;
        let obj = val.as_object().ok_or("expected object")?;
        let command = obj
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or("command required")?;
        let timeout_ms = obj
            .get("timeout")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_TIMEOUT_MS);

        // Same co-author guarantee as the git tool when agents commit via pwsh.
        let command =
            crate::tools::commit_attribution::ensure_powershell_command_co_author(command);

        let (bin, flag) = resolve_powershell()?;
        let mut cmd = Command::new(&bin);
        // -NoProfile keeps startup fast and avoids user-profile side effects.
        cmd.arg("-NoProfile")
            .arg("-NonInteractive")
            .arg(flag)
            .arg(&command)
            // Closed, not inherited: this tool spawns directly rather than
            // through `process_runner::run`, so it needs the same guard. A
            // script calling `Read-Host` would otherwise read the terminal the
            // TUI holds in raw mode. `-NonInteractive` suppresses PowerShell's
            // own prompts but not a read the script asks for itself.
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }

        let mut child = cmd
            .spawn()
            .map_err(|e| format!("powershell exec error ({bin}): {e}"))?;

        let deadline = Instant::now()
            .checked_add(Duration::from_millis(timeout_ms))
            .ok_or_else(|| format!("timeout too large: {timeout_ms}ms"))?;

        // Drain both streams concurrently instead of waiting for exit first:
        // a command that writes more than a pipe buffer holds would otherwise
        // block on output forever, since nothing reads it until the process
        // exits, and get killed as a false-positive timeout.
        let stdout_handle = drain_bounded(
            child.stdout.take().expect("stdout is piped"),
            HEAD_CHARS,
            TAIL_CHARS,
        );
        let stderr_handle = drain_bounded(
            child.stderr.take().expect("stderr is piped"),
            HEAD_CHARS,
            TAIL_CHARS,
        );

        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) => {
                    if Instant::now() >= deadline {
                        let _ = child.kill();
                        let _ = child.wait();
                        let _ = stdout_handle.join();
                        let _ = stderr_handle.join();
                        return Err(format!("powershell timed out after {timeout_ms}ms"));
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(e) => return Err(format!("powershell exec error: {e}")),
            }
        };
        let code = status.code().unwrap_or(-1);
        let ok = status.success();

        let stdout = normalize_cli_output(&stdout_handle.join().unwrap_or_default());
        let stderr = normalize_cli_output(&stderr_handle.join().unwrap_or_default());

        let mut body = String::new();
        if !stdout.is_empty() {
            body.push_str(&stdout);
        }
        if !stderr.is_empty() {
            if !body.is_empty() && !body.ends_with('\n') {
                body.push('\n');
            }
            if !stdout.is_empty() {
                body.push_str("--- stderr ---\n");
            }
            body.push_str(&stderr);
        }

        let body = truncate_head_tail(&body, MAX_OUTPUT_CHARS, HEAD_CHARS, TAIL_CHARS);
        let mut result = body;
        if !result.is_empty() && !result.ends_with('\n') {
            result.push('\n');
        }
        result.push_str(&format!("(exit code {code})"));

        if ok {
            Ok(result)
        } else {
            Err(format!("exit code {code}\n{result}"))
        }
    }
}

/// Keeps only the first `head_max` and a rolling window of the last
/// `tail_max` chars seen, so memory use stays bounded no matter how much a
/// runaway command prints.
struct BoundedCollector {
    head: String,
    head_len: usize,
    head_max: usize,
    tail: VecDeque<char>,
    tail_max: usize,
    total_chars: usize,
}

impl BoundedCollector {
    fn new(head_max: usize, tail_max: usize) -> Self {
        BoundedCollector {
            head: String::new(),
            head_len: 0,
            head_max,
            tail: VecDeque::new(),
            tail_max,
            total_chars: 0,
        }
    }

    fn push_str(&mut self, s: &str) {
        for c in s.chars() {
            self.total_chars += 1;
            if self.head_len < self.head_max {
                self.head.push(c);
                self.head_len += 1;
            } else {
                if self.tail.len() >= self.tail_max {
                    self.tail.pop_front();
                }
                self.tail.push_back(c);
            }
        }
    }

    fn finish(self) -> String {
        let tail_len = self.tail.len();
        let tail_str: String = self.tail.into_iter().collect();
        if self.total_chars <= self.head_len + tail_len {
            format!("{}{}", self.head, tail_str)
        } else {
            let omitted = self.total_chars - self.head_len - tail_len;
            format!(
                "{}\n... [{omitted} chars truncated] ...\n{}",
                self.head, tail_str
            )
        }
    }
}

/// Reads `reader` to EOF on a background thread so it can never block the
/// child process waiting for us to notice it should be killed.
fn drain_bounded<R: Read + Send + 'static>(
    mut reader: R,
    head_max: usize,
    tail_max: usize,
) -> std::thread::JoinHandle<String> {
    std::thread::spawn(move || {
        let mut collector = BoundedCollector::new(head_max, tail_max);
        let mut buf = [0u8; 8192];
        loop {
            match reader.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => collector.push_str(&String::from_utf8_lossy(&buf[..n])),
            }
        }
        collector.finish()
    })
}

/// Prefer PowerShell 7+ (`pwsh`), then Windows PowerShell 5.1 (`powershell`).
fn resolve_powershell() -> Result<(String, &'static str), String> {
    if which_ok("pwsh") {
        // pwsh uses -Command for one-liners the same way.
        return Ok(("pwsh".into(), "-Command"));
    }
    if cfg!(windows) && which_ok("powershell") {
        return Ok(("powershell".into(), "-Command"));
    }
    if cfg!(windows) {
        Err(
            "no PowerShell on PATH (tried pwsh, powershell). Install PowerShell 7 or use Windows PowerShell."
                .into(),
        )
    } else {
        Err(
            "no pwsh on PATH. Install PowerShell 7 (https://aka.ms/powershell) to use the powershell tool on this OS."
                .into(),
        )
    }
}

fn which_ok(bin: &str) -> bool {
    // --version works for both pwsh and Windows powershell 5.1
    let mut c = Command::new(bin);
    if bin == "powershell" {
        c.args([
            "-NoProfile",
            "-Command",
            "$PSVersionTable.PSVersion.ToString()",
        ]);
    } else {
        c.arg("--version");
    }
    c.stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn normalize_cli_output(s: &str) -> String {
    let s = s.replace("\r\n", "\n").replace('\r', "\n");
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

    #[test]
    fn identity() {
        let t = PowerShellTool;
        assert_eq!(t.name(), "powershell");
        assert!(t.needs_permission());
        assert!(crate::json::parse(&t.input_schema()).is_ok());
    }

    #[test]
    fn requires_command() {
        assert!(PowerShellTool.execute("{}").is_err());
    }

    #[test]
    fn echo_or_missing_binary() {
        let input = r#"{"command":"Write-Output 'hi-from-ps'"}"#;
        if resolve_powershell().is_err() {
            let err = PowerShellTool.execute(input).unwrap_err();
            assert!(
                err.contains("no PowerShell") || err.contains("no pwsh"),
                "unexpected: {err}"
            );
            return;
        }
        let out = PowerShellTool.execute(input).unwrap();
        assert!(out.to_ascii_lowercase().contains("hi-from-ps"), "{out}");
        assert!(out.contains("(exit code 0)"), "{out}");
    }

    #[test]
    fn timeout_kills_long_sleep() {
        if resolve_powershell().is_err() {
            return;
        }
        let input = r#"{"command":"Start-Sleep -Seconds 5","timeout":300}"#;
        let err = PowerShellTool.execute(input).unwrap_err();
        assert!(err.contains("timed out"), "{err}");
    }

    #[test]
    fn resolve_finds_something_on_windows() {
        if !cfg!(windows) {
            return;
        }
        let (bin, _) = resolve_powershell().expect("Windows should have powershell");
        assert!(bin == "pwsh" || bin == "powershell", "{bin}");
    }

    /// Regression test for the deadlock this tool used to hit: reading
    /// stdout only after the process exited meant a command writing more
    /// than a pipe buffer would block forever on output, never exit, and
    /// get killed as a false-positive timeout. Generous timeout here to
    /// prove it completes at all, not to test timing.
    #[test]
    fn large_output_does_not_deadlock() {
        if resolve_powershell().is_err() {
            return;
        }
        let input = r#"{"command":"1..20000 | ForEach-Object { \"line $_ of output padding to grow past a pipe buffer\" }","timeout":20000}"#;
        let out = PowerShellTool.execute(input).unwrap();
        assert!(out.contains("(exit code 0)"), "{out}");
    }

    #[test]
    fn bounded_collector_keeps_head_and_tail_under_memory_cap() {
        let mut c = BoundedCollector::new(5, 5);
        c.push_str(&"a".repeat(5000));
        let out = c.finish();
        assert!(out.contains("truncated"));
        assert!(out.starts_with("aaaaa"));
        assert!(out.ends_with("aaaaa"));
    }

    #[test]
    fn bounded_collector_no_truncation_when_under_cap() {
        let mut c = BoundedCollector::new(100, 100);
        c.push_str("short output");
        assert_eq!(c.finish(), "short output");
    }
}
