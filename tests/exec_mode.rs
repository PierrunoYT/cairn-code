use std::io::Write;
use std::process::{Command, Output, Stdio};

/// Runs `cairn-code exec` against an unreachable provider and returns the
/// finished process. Enough to exercise argument handling and start-up
/// behaviour without a live model.
fn run_exec(extra_args: &[&str]) -> Output {
    let mut args = vec![
        "exec",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
    ];
    args.extend_from_slice(extra_args);

    let mut child = Command::new(env!("CARGO_BIN_EXE_cairn-code"))
        .args(&args)
        .env("CAIRN_PROVIDER", "ollama")
        .env("CAIRN_MODEL", "cairn-exec-test-model")
        .env("OLLAMA_HOST", "http://127.0.0.1:1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn cairn-code exec");

    let input_msg = serde_json::json!({
        "schemaVersion": 2,
        "type": "message",
        "role": "user",
        "content": "hello cairn exec"
    })
    .to_string();

    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(format!("{input_msg}\n").as_bytes())
        .expect("write json input");

    child.wait_with_output().expect("wait for cairn-code exec")
}

/// C-1 regression: `--auto` is a valueless flag. The old parser consumed the
/// argument after it, so `--auto --init-session-id X` silently dropped the
/// session id (and, with it, any chance of resuming the right session).
#[test]
fn exec_auto_flag_does_not_swallow_the_following_argument() {
    let output = run_exec(&["--auto", "--init-session-id", "test-auto-session-456"]);
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        stdout.contains("test-auto-session-456"),
        "--auto swallowed the following argument: {stdout}"
    );
}

/// Unattended approval of every tool is a posture the operator has to choose
/// out loud, so it is announced on stderr and absent without the flag.
#[test]
fn exec_warns_on_stderr_only_when_auto_is_passed() {
    let with_auto = run_exec(&["--auto", "--init-session-id", "test-auto-warn"]);
    let stderr = String::from_utf8_lossy(&with_auto.stderr);
    assert!(
        stderr.contains("--auto approves every tool call"),
        "expected an --auto warning on stderr, got: {stderr}"
    );

    let without_auto = run_exec(&["--init-session-id", "test-no-auto-warn"]);
    let stderr = String::from_utf8_lossy(&without_auto.stderr);
    assert!(
        !stderr.contains("approves every tool call"),
        "a run without --auto must not claim auto-approval: {stderr}"
    );
}

#[test]
fn exec_mode_emits_stream_json_events() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_cairn-code"))
        .args([
            "exec",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--init-session-id",
            "test-exec-session-123",
        ])
        .env("CAIRN_PROVIDER", "ollama")
        .env("CAIRN_MODEL", "cairn-exec-test-model")
        .env("OLLAMA_HOST", "http://127.0.0.1:1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn cairn-code exec");

    let input_msg = serde_json::json!({
        "schemaVersion": 2,
        "type": "message",
        "role": "user",
        "content": "hello cairn exec"
    })
    .to_string();

    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(format!("{input_msg}\n").as_bytes())
        .expect("write json input");

    let output = child.wait_with_output().expect("wait for cairn-code exec");
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        stdout.contains("\"type\":\"run_start\""),
        "stdout missing run_start: {stdout}"
    );
    assert!(
        stdout.contains("test-exec-session-123"),
        "stdout missing session ID: {stdout}"
    );
    assert!(
        stdout.contains("\"type\":\"run_end\""),
        "stdout missing run_end: {stdout}"
    );
}
