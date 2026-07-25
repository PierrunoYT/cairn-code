//! Thin `python` tool: run a snippet (`code`) or a script file (`file` + `args`).
//! Same idea as `shell` / `go` — spawn an interpreter on PATH, return stdout/stderr.

use cap_fs_ext::DirExt;
use std::path::{Component, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use super::registry::Tool;
use super::workspace;

/// Max chars returned to the model (head + tail).
const MAX_OUTPUT_CHARS: usize = 12_000;
const HEAD_CHARS: usize = 6_000;
const TAIL_CHARS: usize = 4_000;
const DEFAULT_TIMEOUT_MS: u64 = 60_000;

pub struct PythonTool;

impl Tool for PythonTool {
    fn name(&self) -> &str {
        "python"
    }

    fn description(&self) -> &str {
        "Run Python via the system interpreter (python3 / python / py -3). \
         Pass either `code` (inline snippet) or `file` (script path under the workspace) \
         with optional `args` and `timeout` (ms). Prefer this over shell for Python so \
         quoting stays simple. Check the exit code footer in the result."
    }

    fn needs_permission(&self) -> bool {
        true
    }

    fn input_schema(&self) -> String {
        r#"{"type":"object","properties":{"code":{"type":"string","description":"Inline Python source to run with python -c"},"file":{"type":"string","description":"Path to a .py script under the workspace"},"args":{"type":"array","items":{"type":"string"},"description":"Arguments passed to the script (file mode only)"},"timeout":{"type":"integer","description":"Timeout in milliseconds (default 60000)"}},"additionalProperties":false}"#.into()
    }

    fn execute(&self, input: &str) -> Result<String, String> {
        let val = crate::json::parse(input).map_err(|e| format!("invalid input: {e}"))?;
        let obj = val.as_object().ok_or("expected object")?;

        let code = obj.get("code").and_then(|v| v.as_str());
        let file = obj.get("file").and_then(|v| v.as_str());
        let timeout_ms = obj
            .get("timeout")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_TIMEOUT_MS);

        if code.is_some() && file.is_some() {
            return Err("pass either code or file, not both".into());
        }
        if code.is_none() && file.is_none() {
            return Err("code or file required".into());
        }

        let args_list: Vec<String> = obj
            .get("args")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        if !args_list.is_empty() && file.is_none() {
            return Err("args only apply with file (not code)".into());
        }

        let (bin, prefix_args) = resolve_python()?;
        let mut cmd = Command::new(&bin);
        cmd.args(&prefix_args);

        if let Some(src) = code {
            cmd.arg("-c").arg(src);
        } else if let Some(path) = file {
            let ws = workspace::Workspace::current().map_err(|e| format!("workspace: {e}"))?;
            let abs = resolve_workspace_file(&ws, path)?;
            cmd.arg(&abs);
            cmd.args(&args_list);
        }

        // Closed, not inherited: another direct spawn, and `input()` in
        // model-supplied code is exactly the hang this closes — it would block
        // on the terminal the TUI is drawing over.
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        // Quiet child console on Windows.
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }

        let mut child = cmd
            .spawn()
            .map_err(|e| format!("python exec error ({bin}): {e}"))?;

        let deadline = Instant::now() + Duration::from_millis(timeout_ms);
        loop {
            match child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) => {
                    if Instant::now() >= deadline {
                        let _ = child.kill();
                        let _ = child.wait();
                        return Err(format!("python timed out after {timeout_ms}ms"));
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(e) => return Err(format!("python exec error: {e}")),
            }
        }

        let output = child
            .wait_with_output()
            .map_err(|e| format!("python exec error: {e}"))?;
        let code_n = output.status.code().unwrap_or(-1);
        let ok = output.status.success();

        let stdout = normalize_output(&String::from_utf8_lossy(&output.stdout));
        let stderr = normalize_output(&String::from_utf8_lossy(&output.stderr));

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
        result.push_str(&format!("(exit code {code_n})"));

        if ok {
            Ok(result)
        } else {
            Err(format!("exit code {code_n}\n{result}"))
        }
    }
}

/// Resolve `path` to an on-disk file under the workspace without ever
/// following a symlink, mirroring the walk `grep`'s single-file search uses.
/// `workspace.relative_path` alone only rejects `..`/absolute escapes
/// lexically; a symlink planted inside the workspace (e.g. by a prior `shell`
/// call) would still resolve outside it if we handed a raw std::fs path
/// straight to the interpreter.
fn resolve_workspace_file(ws: &workspace::Workspace, path: &str) -> Result<PathBuf, String> {
    let relative = ws.relative_path(path)?;
    let components: Vec<_> = relative.components().collect();
    if components.is_empty() {
        return Err(format!("file not found: {path}"));
    }

    let mut current = ws
        .dir()
        .try_clone()
        .map_err(|e| format!("file path: cannot open workspace: {e}"))?;
    for (index, component) in components.iter().enumerate() {
        let Component::Normal(name) = component else {
            return Err(format!("file path: invalid path component in {path}"));
        };
        let metadata = current
            .symlink_metadata(name)
            .map_err(|_| format!("file not found: {path}"))?;
        if metadata.file_type().is_symlink() {
            return Err(format!("refusing to follow symlink: {path}"));
        }
        let is_last = index == components.len() - 1;
        if is_last {
            if !metadata.is_file() {
                return Err(format!("file not found: {path}"));
            }
        } else {
            if !metadata.is_dir() {
                return Err(format!("file not found: {path}"));
            }
            current = current
                .open_dir_nofollow(name)
                .map_err(|e| format!("file path: {e}"))?;
        }
    }
    Ok(ws.root().join(&relative))
}

/// Pick a Python interpreter: `python3`, then `python`, then Windows `py -3`.
fn resolve_python() -> Result<(String, Vec<String>), String> {
    if which_ok("python3") {
        return Ok(("python3".into(), Vec::new()));
    }
    if which_ok("python") {
        return Ok(("python".into(), Vec::new()));
    }
    if cfg!(windows) && which_ok("py") {
        return Ok(("py".into(), vec!["-3".into()]));
    }
    Err(
        "no Python interpreter on PATH (tried python3, python, py -3). Install Python or add it to PATH."
            .into(),
    )
}

fn which_ok(bin: &str) -> bool {
    Command::new(bin)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn normalize_output(s: &str) -> String {
    s.replace("\r\n", "\n").replace('\r', "\n")
}

fn truncate_head_tail(s: &str, max_total: usize, head: usize, tail: usize) -> String {
    if s.len() <= max_total {
        return s.to_string();
    }
    let head_n = head.min(s.len());
    let tail_n = tail.min(s.len().saturating_sub(head_n));
    let head_part = &s[..head_n];
    let tail_part = &s[s.len() - tail_n..];
    let omitted = s.len() - head_n - tail_n;
    format!("{head_part}\n… ({omitted} bytes omitted) …\n{tail_part}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn identity() {
        let t = PythonTool;
        assert_eq!(t.name(), "python");
        assert!(t.needs_permission());
        assert!(crate::json::parse(&t.input_schema()).is_ok());
    }

    #[test]
    fn requires_code_or_file() {
        assert!(PythonTool.execute("{}").is_err());
        assert!(PythonTool
            .execute(r#"{"code":"x=1","file":"a.py"}"#)
            .is_err());
    }

    #[test]
    fn args_without_file_rejected() {
        let err = PythonTool
            .execute(r#"{"code":"print(1)","args":["x"]}"#)
            .unwrap_err();
        assert!(err.contains("args"), "{err}");
    }

    #[test]
    fn runs_inline_or_reports_missing_python() {
        match PythonTool.execute(r#"{"code":"print(1+1)"}"#) {
            Ok(out) => {
                assert!(out.contains('2'), "{out}");
                assert!(out.contains("exit code 0"), "{out}");
            }
            Err(e) => {
                assert!(
                    e.contains("no Python") || e.contains("python exec") || e.contains("exit code"),
                    "unexpected: {e}"
                );
            }
        }
    }

    #[test]
    fn runs_file_in_workspace_or_missing_python() {
        // Write under the repo cwd so resolve_in_workspace accepts it (no chdir).
        let name = format!("cairn_py_test_{}.py", std::process::id());
        fs::write(&name, "import sys\nprint(sys.argv[1])\n").unwrap();
        let input = format!(r#"{{"file":"{name}","args":["ok"]}}"#);
        let result = PythonTool.execute(&input);
        let _ = fs::remove_file(&name);
        match result {
            Ok(out) => {
                assert!(out.contains("ok"), "{out}");
                assert!(out.contains("exit code 0"), "{out}");
            }
            Err(e) => {
                assert!(
                    e.contains("no Python") || e.contains("python exec") || e.contains("exit code"),
                    "unexpected: {e}"
                );
            }
        }
    }

    #[test]
    fn resolve_python_returns_something_or_clear_error() {
        match resolve_python() {
            Ok((bin, _)) => assert!(!bin.is_empty()),
            Err(e) => assert!(e.contains("no Python"), "{e}"),
        }
    }

    fn temp_dir(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "cairn-python-{label}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn resolve_workspace_file_accepts_real_file() {
        let workspace_dir = temp_dir("ok");
        fs::create_dir_all(workspace_dir.join("src")).unwrap();
        fs::write(workspace_dir.join("src/main.py"), "print('hi')\n").unwrap();

        let ws = workspace::Workspace::new(&workspace_dir).unwrap();
        let resolved = resolve_workspace_file(&ws, "src/main.py").unwrap();
        assert!(resolved.is_file(), "{}", resolved.display());
        assert_eq!(resolved.file_name().unwrap(), "main.py");

        let _ = fs::remove_dir_all(&workspace_dir);
    }

    #[test]
    fn resolve_workspace_file_missing_file_errors() {
        let ws = workspace::Workspace::current().unwrap();
        let err = resolve_workspace_file(&ws, "cairn-definitely-missing-file.py").unwrap_err();
        assert!(err.contains("not found"), "{err}");
    }

    #[test]
    fn resolve_workspace_file_rejects_parent_traversal() {
        let ws = workspace::Workspace::current().unwrap();
        let err = resolve_workspace_file(&ws, "../outside.py").unwrap_err();
        assert!(err.contains("outside the workspace"), "{err}");
    }

    #[test]
    fn resolve_workspace_file_rejects_symlink_escape() {
        let workspace_dir = temp_dir("link");
        let outside = temp_dir("link-outside");
        fs::create_dir_all(&workspace_dir).unwrap();
        fs::create_dir_all(&outside).unwrap();
        fs::write(outside.join("secret.py"), "print('leaked')\n").unwrap();
        let link = workspace_dir.join("escape");
        assert!(
            create_dir_link(&outside, &link),
            "failed to create test link"
        );

        let ws = workspace::Workspace::new(&workspace_dir).unwrap();
        let err = resolve_workspace_file(&ws, "escape/secret.py").unwrap_err();
        assert!(
            err.contains("symlink") || err.contains("not found"),
            "{err}"
        );

        let _ = fs::remove_dir_all(&workspace_dir);
        let _ = fs::remove_dir_all(&outside);
    }

    #[cfg(unix)]
    fn create_dir_link(target: &std::path::Path, link: &std::path::Path) -> bool {
        std::os::unix::fs::symlink(target, link).is_ok()
    }

    #[cfg(windows)]
    fn create_dir_link(target: &std::path::Path, link: &std::path::Path) -> bool {
        std::process::Command::new("cmd")
            .args(["/C", "mklink", "/J"])
            .arg(link)
            .arg(target)
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }
}
