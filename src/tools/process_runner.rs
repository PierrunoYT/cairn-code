//! Shared subprocess runner: bounded output, an optional finite timeout, an
//! optional cooperative cancellation token, and process-tree termination.
//!
//! The `shell`, `git`, and `go` tools all run child processes that can hang,
//! spawn descendants, or produce unbounded output. This module centralizes the
//! safe way to do that: output is drained on background threads into a
//! head+tail bounded buffer (so a chatty process cannot exhaust memory or
//! deadlock on a full pipe), the caller can cap wall-clock time, and the whole
//! process group / job object is killed on timeout or cancellation so no
//! descendant keeps running after the tool returns.

use std::collections::VecDeque;
use std::io::Read;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// How the caller wants a command run.
pub struct RunOptions {
    /// Maximum wall-clock time before the process tree is killed. `None`
    /// leaves the command unbounded (caller relies on cancellation only).
    pub timeout: Option<Duration>,
    /// Characters kept from the start of each stream.
    pub head_chars: usize,
    /// Characters kept from the end of each stream.
    pub tail_chars: usize,
}

/// A completed process, with each stream already bounded to head+tail.
pub struct RunResult {
    pub code: i32,
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
}

/// A completed process whose output fit within strict byte limits.
pub(crate) struct ByteLimitedOutput {
    pub status: ExitStatus,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

/// Failure from [`run_with_byte_limits`]. Unlike [`run`], this mode never
/// returns truncated output: crossing either limit terminates the process tree
/// and reports which stream overflowed.
#[derive(Debug)]
pub(crate) enum ByteLimitedRunError {
    Spawn(String),
    StdoutLimit {
        limit: usize,
        cleanup_error: Option<String>,
    },
    StderrLimit {
        limit: usize,
        cleanup_error: Option<String>,
    },
    Read {
        stream: &'static str,
        reason: String,
        cleanup_error: Option<String>,
    },
    Wait(String),
}

/// Why a run did not complete normally. Messages are raw (no tool prefix) so
/// each caller can phrase them in its own voice.
#[derive(Debug)]
pub enum RunError {
    /// Failed to set up the process group / job object, spawn, or attach.
    Spawn(String),
    /// Exceeded the configured timeout; `after_ms` is that timeout.
    TimedOut {
        after_ms: u64,
        cleanup_error: Option<String>,
    },
    /// The cancellation token was set while the command was running.
    Cancelled { cleanup_error: Option<String> },
    /// Waiting on the child failed partway through.
    Wait {
        reason: String,
        cleanup_error: Option<String>,
    },
}

/// Append a process-cleanup failure detail to an error message, matching the
/// phrasing the shell tool has always used.
pub fn with_cleanup(base: String, cleanup_error: &Option<String>) -> String {
    match cleanup_error {
        Some(error) => format!("{base}; process cleanup failed: {error}"),
        None => base,
    }
}

/// Run `command` to completion (or until timeout/cancellation), returning
/// bounded output. Stdout/stderr are forced to pipes and the process is placed
/// in its own group/job so descendants can be killed together.
pub fn run(
    mut command: Command,
    options: &RunOptions,
    cancel: Option<&AtomicBool>,
) -> Result<RunResult, RunError> {
    // stdin is closed rather than inherited. The TUI holds the terminal in raw
    // mode under an alternate screen, so a child that reads stdin — `git
    // rebase -i`, `ssh` asking for a passphrase, `npm init`, a bare `read` —
    // would compete with the composer for keystrokes while the user cannot see
    // what it is waiting for. Closed stdin turns that into an immediate EOF,
    // which every one of those commands handles by failing fast.
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    // Overflow (timeout near the end of the monotonic clock) degrades to
    // "no deadline" rather than erroring; cancellation still applies.
    let deadline = options.timeout.and_then(|d| Instant::now().checked_add(d));

    let mut child = ManagedChild::spawn(command).map_err(RunError::Spawn)?;

    let stdout_rx = drain_bounded(
        child.stdout.take().expect("stdout is piped"),
        options.head_chars,
        options.tail_chars,
    );
    let stderr_rx = drain_bounded(
        child.stderr.take().expect("stderr is piped"),
        options.head_chars,
        options.tail_chars,
    );

    let mut stdout = None;
    let mut stderr = None;

    let status = loop {
        receive_output(&stdout_rx, &mut stdout);
        receive_output(&stderr_rx, &mut stderr);

        if cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
            let mut cleanup_error = child.terminate().err();
            add_wait_error(&mut cleanup_error, child.wait().err());
            return Err(RunError::Cancelled { cleanup_error });
        }

        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            let mut cleanup_error = child.terminate().err();
            add_wait_error(&mut cleanup_error, child.wait().err());
            let after_ms = options
                .timeout
                .map(|d| d.as_millis().min(u128::from(u64::MAX)) as u64)
                .unwrap_or(0);
            return Err(RunError::TimedOut {
                after_ms,
                cleanup_error,
            });
        }

        // Keep the group leader unreaped until all pipe holders exit. Its PID
        // is also the Unix process-group ID and must not be reusable while
        // timeout/cancel cleanup may still signal the group.
        if stdout.is_some() && stderr.is_some() {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) => {}
                Err(e) => {
                    let mut cleanup_error = child.terminate().err();
                    add_wait_error(&mut cleanup_error, child.wait().err());
                    return Err(RunError::Wait {
                        reason: e.to_string(),
                        cleanup_error,
                    });
                }
            }
        }

        std::thread::sleep(Duration::from_millis(10));
    };

    Ok(RunResult {
        code: status.code().unwrap_or(-1),
        success: status.success(),
        stdout: stdout.unwrap_or_default(),
        stderr: stderr.unwrap_or_default(),
    })
}

fn add_wait_error(cleanup_error: &mut Option<String>, wait_error: Option<std::io::Error>) {
    let Some(wait_error) = wait_error else {
        return;
    };
    match cleanup_error {
        Some(cleanup) => cleanup.push_str(&format!("; wait failed: {wait_error}")),
        None => *cleanup_error = Some(format!("wait failed: {wait_error}")),
    }
}

/// Run a subprocess while retaining at most the requested number of bytes
/// from each stream. Both pipes are drained concurrently. If either limit is
/// crossed, the managed process tree is terminated and reaped, and no partial
/// output is returned to the caller. The command must enforce its own finite
/// wall-clock timeout; current callers use curl's `--max-time`.
pub(crate) fn run_with_byte_limits(
    mut command: Command,
    stdin: Option<Vec<u8>>,
    stdout_limit: usize,
    stderr_limit: usize,
) -> Result<ByteLimitedOutput, ByteLimitedRunError> {
    // Same reasoning as `run`: a child with nothing to be fed gets a closed
    // stdin, not the TUI's terminal. Callers here (curl for web_fetch and
    // web_search) do not read it today, but inheriting is the wrong default
    // to leave lying around next to a raw-mode terminal.
    match stdin {
        Some(_) => command.stdin(Stdio::piped()),
        None => command.stdin(Stdio::null()),
    };
    command.stdout(Stdio::piped()).stderr(Stdio::piped());

    let mut child = ManagedChild::spawn(command).map_err(ByteLimitedRunError::Spawn)?;
    let stdout = child.stdout.take().expect("stdout is piped");
    let stderr = child.stderr.take().expect("stderr is piped");
    let child = Mutex::new(child);

    let (stdout_result, stderr_result) = std::thread::scope(|scope| {
        let stdout_reader = scope.spawn(|| read_strict(stdout, stdout_limit, &child));
        let stderr_reader = scope.spawn(|| read_strict(stderr, stderr_limit, &child));
        let stdin_writer = stdin.map(|input| {
            let mut stdin = child
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .stdin
                .take();
            scope.spawn(move || {
                if let Some(stdin) = stdin.as_mut() {
                    let _ = std::io::Write::write_all(stdin, &input);
                }
            })
        });

        let stdout_result = stdout_reader.join().unwrap_or_else(|_| StrictRead::Read {
            reason: "stdout reader thread panicked".to_string(),
            cleanup_error: terminate_locked(&child),
        });
        let stderr_result = stderr_reader.join().unwrap_or_else(|_| StrictRead::Read {
            reason: "stderr reader thread panicked".to_string(),
            cleanup_error: terminate_locked(&child),
        });
        if let Some(writer) = stdin_writer {
            let _ = writer.join();
        }
        (stdout_result, stderr_result)
    });

    let wait_result = child
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .wait();

    let stdout = strict_output("stdout", stdout_limit, stdout_result)?;
    let stderr = strict_output("stderr", stderr_limit, stderr_result)?;
    let status = wait_result.map_err(|error| ByteLimitedRunError::Wait(error.to_string()))?;
    Ok(ByteLimitedOutput {
        status,
        stdout,
        stderr,
    })
}

enum StrictRead {
    Complete(Vec<u8>),
    Limit {
        cleanup_error: Option<String>,
    },
    Read {
        reason: String,
        cleanup_error: Option<String>,
    },
}

fn read_strict<R: Read>(mut reader: R, limit: usize, child: &Mutex<ManagedChild>) -> StrictRead {
    let mut retained = Vec::with_capacity(limit.min(8_192));
    let mut buffer = [0_u8; 8_192];
    loop {
        match reader.read(&mut buffer) {
            Ok(0) => return StrictRead::Complete(retained),
            Ok(read) => {
                let remaining = limit.saturating_sub(retained.len());
                retained.extend_from_slice(&buffer[..read.min(remaining)]);
                if read > remaining {
                    return StrictRead::Limit {
                        cleanup_error: terminate_locked(child),
                    };
                }
            }
            Err(error) => {
                return StrictRead::Read {
                    reason: error.to_string(),
                    cleanup_error: terminate_locked(child),
                }
            }
        }
    }
}

fn strict_output(
    stream: &'static str,
    limit: usize,
    result: StrictRead,
) -> Result<Vec<u8>, ByteLimitedRunError> {
    match result {
        StrictRead::Complete(output) => Ok(output),
        StrictRead::Limit { cleanup_error } if stream == "stdout" => {
            Err(ByteLimitedRunError::StdoutLimit {
                limit,
                cleanup_error,
            })
        }
        StrictRead::Limit { cleanup_error } => Err(ByteLimitedRunError::StderrLimit {
            limit,
            cleanup_error,
        }),
        StrictRead::Read {
            reason,
            cleanup_error,
        } => Err(ByteLimitedRunError::Read {
            stream,
            reason,
            cleanup_error,
        }),
    }
}

fn terminate_locked(child: &Mutex<ManagedChild>) -> Option<String> {
    child
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .terminate()
        .err()
}

fn receive_output(rx: &Receiver<String>, output: &mut Option<String>) {
    if output.is_some() {
        return;
    }
    match rx.try_recv() {
        Ok(value) => *output = Some(value),
        Err(TryRecvError::Empty) => {}
        Err(TryRecvError::Disconnected) => *output = Some(String::new()),
    }
}

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
        Self {
            head: String::new(),
            head_len: 0,
            head_max,
            tail: VecDeque::new(),
            tail_max,
            total_chars: 0,
        }
    }

    fn push_str(&mut self, value: &str) {
        for character in value.chars() {
            self.total_chars += 1;
            if self.head_len < self.head_max {
                self.head.push(character);
                self.head_len += 1;
            } else {
                if self.tail.len() >= self.tail_max {
                    self.tail.pop_front();
                }
                self.tail.push_back(character);
            }
        }
    }

    fn finish(self) -> String {
        let tail_len = self.tail.len();
        let tail: String = self.tail.into_iter().collect();
        if self.total_chars <= self.head_len + tail_len {
            format!("{}{tail}", self.head)
        } else {
            let omitted = self.total_chars - self.head_len - tail_len;
            format!("{}\n... [{omitted} chars truncated] ...\n{tail}", self.head)
        }
    }
}

fn drain_bounded<R: Read + Send + 'static>(
    mut reader: R,
    head_max: usize,
    tail_max: usize,
) -> Receiver<String> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut collector = BoundedCollector::new(head_max, tail_max);
        let mut buffer = [0u8; 8192];
        loop {
            match reader.read(&mut buffer) {
                Ok(0) | Err(_) => break,
                Ok(read) => collector.push_str(&String::from_utf8_lossy(&buffer[..read])),
            }
        }
        let _ = tx.send(collector.finish());
    });
    rx
}

/// A child placed in its own process group / job object, so all subprocess
/// call sites use the same descendant-safe termination behavior.
pub(crate) struct ManagedChild {
    child: Child,
    process_tree: ProcessTree,
}

impl ManagedChild {
    pub(crate) fn spawn(mut command: Command) -> Result<Self, String> {
        configure_process_group(&mut command);
        let process_tree = ProcessTree::new()?;
        let mut child = command.spawn().map_err(|error| error.to_string())?;
        if let Err(error) = process_tree.attach(&mut child) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(error);
        }
        Ok(Self {
            child,
            process_tree,
        })
    }

    pub(crate) fn terminate(&mut self) -> Result<(), String> {
        self.process_tree.terminate(&mut self.child)
    }
}

impl std::ops::Deref for ManagedChild {
    type Target = Child;

    fn deref(&self) -> &Self::Target {
        &self.child
    }
}

impl std::ops::DerefMut for ManagedChild {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.child
    }
}

#[cfg(unix)]
fn configure_process_group(command: &mut Command) {
    use std::os::unix::process::CommandExt;
    command.process_group(0);
}

#[cfg(windows)]
fn configure_process_group(command: &mut Command) {
    use std::os::windows::process::CommandExt;
    command.creation_flags(windows_sys::Win32::System::Threading::CREATE_SUSPENDED);
}

#[cfg(not(any(unix, windows)))]
fn configure_process_group(_command: &mut Command) {}

#[cfg(unix)]
struct ProcessTree;

#[cfg(unix)]
impl ProcessTree {
    fn new() -> Result<Self, String> {
        Ok(Self)
    }

    fn attach(&self, _child: &mut std::process::Child) -> Result<(), String> {
        Ok(())
    }

    fn terminate(&self, child: &mut std::process::Child) -> Result<(), String> {
        // The child's PID is also the process-group ID configured before spawn.
        if unsafe { libc::kill(-(child.id() as i32), libc::SIGKILL) } == 0 {
            return Ok(());
        }
        let group_error = std::io::Error::last_os_error();
        if group_error.raw_os_error() == Some(libc::ESRCH) {
            return Ok(());
        }
        match child.kill() {
            Ok(()) => Err(group_error.to_string()),
            Err(child_error) => Err(format!(
                "{group_error}; child kill also failed: {child_error}"
            )),
        }
    }
}

#[cfg(windows)]
struct ProcessTree(windows_sys::Win32::Foundation::HANDLE);

// Job-object HANDLEs are process-wide kernel objects. Readers share
// `&Mutex<ManagedChild>` across scoped threads for timeout kill; Drop still
// closes the handle once after those threads join.
#[cfg(windows)]
unsafe impl Send for ProcessTree {}

#[cfg(windows)]
impl ProcessTree {
    fn new() -> Result<Self, String> {
        use windows_sys::Win32::System::JobObjects::CreateJobObjectW;

        let job = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
        if job.is_null() {
            return Err(format!(
                "could not create job object: {}",
                std::io::Error::last_os_error()
            ));
        }
        Ok(Self(job))
    }

    fn attach(&self, child: &mut std::process::Child) -> Result<(), String> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::System::JobObjects::AssignProcessToJobObject;

        let job = self.0;
        if unsafe { AssignProcessToJobObject(job, child.as_raw_handle() as _) } == 0 {
            return Err(format!(
                "could not assign process to job object: {}",
                std::io::Error::last_os_error()
            ));
        }
        resume_process(child.id())
    }

    fn terminate(&self, child: &mut std::process::Child) -> Result<(), String> {
        if unsafe { windows_sys::Win32::System::JobObjects::TerminateJobObject(self.0, 1) } != 0 {
            return Ok(());
        }
        let job_error = std::io::Error::last_os_error();
        child
            .kill()
            .map_err(|child_error| format!("{job_error}; child kill also failed: {child_error}"))?;
        Err(job_error.to_string())
    }
}

#[cfg(windows)]
fn resume_process(process_id: u32) -> Result<(), String> {
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, Thread32First, Thread32Next, TH32CS_SNAPTHREAD, THREADENTRY32,
    };
    use windows_sys::Win32::System::Threading::{OpenThread, ResumeThread, THREAD_SUSPEND_RESUME};

    let snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0) };
    if snapshot == INVALID_HANDLE_VALUE {
        return Err(format!(
            "could not enumerate suspended process threads: {}",
            std::io::Error::last_os_error()
        ));
    }

    let mut entry = THREADENTRY32 {
        dwSize: std::mem::size_of::<THREADENTRY32>() as u32,
        ..Default::default()
    };
    // A CREATE_SUSPENDED process has only its primary thread: it cannot run
    // and create another before assignment to the job and this snapshot.
    let mut found = unsafe { Thread32First(snapshot, &mut entry) } != 0;
    let mut thread_id = None;
    while found {
        if entry.th32OwnerProcessID == process_id {
            thread_id = Some(entry.th32ThreadID);
            break;
        }
        found = unsafe { Thread32Next(snapshot, &mut entry) } != 0;
    }
    unsafe {
        CloseHandle(snapshot);
    }

    let thread_id =
        thread_id.ok_or_else(|| "could not find suspended process's primary thread".to_string())?;
    let thread = unsafe { OpenThread(THREAD_SUSPEND_RESUME, 0, thread_id) };
    if thread.is_null() {
        return Err(format!(
            "could not open suspended process thread: {}",
            std::io::Error::last_os_error()
        ));
    }
    let resumed = unsafe { ResumeThread(thread) };
    unsafe {
        CloseHandle(thread);
    }
    if resumed != 1 {
        return Err(format!(
            "suspended process had unexpected resume count {resumed}"
        ));
    }
    Ok(())
}

#[cfg(windows)]
impl Drop for ProcessTree {
    fn drop(&mut self) {
        unsafe {
            windows_sys::Win32::Foundation::CloseHandle(self.0);
        }
    }
}

#[cfg(not(any(unix, windows)))]
struct ProcessTree;

#[cfg(not(any(unix, windows)))]
impl ProcessTree {
    fn new() -> Result<Self, String> {
        Ok(Self)
    }

    fn attach(&self, _child: &mut std::process::Child) -> Result<(), String> {
        Ok(())
    }

    fn terminate(&self, child: &mut std::process::Child) -> Result<(), String> {
        child.kill().map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn echo_command(text: &str) -> Command {
        let mut command = Command::new(if cfg!(windows) { "powershell" } else { "bash" });
        if cfg!(windows) {
            command
                .arg("-Command")
                .arg(format!("Write-Output '{text}'"));
        } else {
            command.arg("-c").arg(format!("echo {text}"));
        }
        command
    }

    fn sleep_command(seconds: u64) -> Command {
        let mut command = Command::new(if cfg!(windows) { "powershell" } else { "bash" });
        if cfg!(windows) {
            command
                .arg("-Command")
                .arg(format!("Start-Sleep -Seconds {seconds}"));
        } else {
            command.arg("-c").arg(format!("sleep {seconds}"));
        }
        command
    }

    /// Reads stdin to EOF and reports how many characters it saw.
    fn read_stdin_command() -> Command {
        let mut command = Command::new(if cfg!(windows) { "powershell" } else { "bash" });
        if cfg!(windows) {
            command
                .arg("-Command")
                .arg("$i = [Console]::In.ReadToEnd(); Write-Output \"len=$($i.Length)\"");
        } else {
            command.arg("-c").arg("data=$(cat); echo \"len=${#data}\"");
        }
        command
    }

    #[test]
    fn child_stdin_is_closed_rather_than_inherited() {
        // A child that reads stdin must get EOF at once. With stdin inherited
        // it reads the terminal instead, competing with the TUI composer for
        // keystrokes while the user cannot see what it is waiting for. The
        // timeout is the real assertion: an inherited terminal stdin blocks
        // here, so reaching the length check at all is the guarantee.
        let options = RunOptions {
            timeout: Some(Duration::from_secs(20)),
            head_chars: 200,
            tail_chars: 200,
        };
        let result = run(read_stdin_command(), &options, None)
            .expect("reading stdin must not block: it should be closed, not inherited");
        assert!(
            result.stdout.contains("len=0"),
            "child saw stdin content: {:?}",
            result.stdout
        );
    }

    #[test]
    fn byte_limited_child_stdin_is_closed_when_there_is_nothing_to_send() {
        // The sibling path: `run_with_byte_limits` piped stdin only when it had
        // input, and inherited it otherwise. It has no timeout of its own, so a
        // regression shows up as a hung run rather than a failed assertion —
        // `len=0` is what proves the child reached EOF instead of the terminal.
        let output = run_with_byte_limits(read_stdin_command(), None, 1024, 1024)
            .expect("reading stdin must not block: it should be closed, not inherited");
        assert!(
            String::from_utf8_lossy(&output.stdout).contains("len=0"),
            "child saw stdin content: {:?}",
            String::from_utf8_lossy(&output.stdout)
        );
    }

    fn large_output_command(stderr: bool) -> Command {
        let mut command = Command::new(if cfg!(windows) { "powershell" } else { "bash" });
        if cfg!(windows) {
            let stream = if stderr { "Error" } else { "Out" };
            command
                .arg("-Command")
                .arg(format!("[Console]::{stream}.Write(('x' * 1000000))"));
        } else {
            let redirect = if stderr { " >&2" } else { "" };
            command
                .arg("-c")
                .arg(format!("head -c 1000000 /dev/zero{redirect}"));
        }
        command
    }

    #[test]
    fn runs_to_completion_and_captures_stdout() {
        let result = run(
            echo_command("hello-runner"),
            &RunOptions {
                timeout: Some(Duration::from_secs(30)),
                head_chars: 6_000,
                tail_chars: 4_000,
            },
            None,
        )
        .expect("command should complete");
        assert!(result.success);
        assert_eq!(result.code, 0);
        assert!(result.stdout.contains("hello-runner"), "{}", result.stdout);
    }

    #[test]
    fn timeout_reports_and_kills() {
        let err = run(
            sleep_command(5),
            &RunOptions {
                timeout: Some(Duration::from_millis(200)),
                head_chars: 100,
                tail_chars: 100,
            },
            None,
        )
        .err()
        .expect("command should time out");
        match err {
            RunError::TimedOut { after_ms, .. } => assert_eq!(after_ms, 200),
            _ => panic!("expected TimedOut"),
        }
    }

    #[test]
    fn cancellation_stops_a_running_command() {
        let cancel = AtomicBool::new(false);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                std::thread::sleep(Duration::from_millis(200));
                cancel.store(true, Ordering::Relaxed);
            });
            let err = run(
                sleep_command(30),
                &RunOptions {
                    timeout: None,
                    head_chars: 100,
                    tail_chars: 100,
                },
                Some(&cancel),
            )
            .err()
            .expect("command should be cancelled");
            assert!(matches!(err, RunError::Cancelled { .. }));
        });
    }

    #[test]
    fn byte_limited_run_accepts_normal_output() {
        let output = run_with_byte_limits(echo_command("bounded-ok"), None, 1024, 1024)
            .expect("small output should fit");
        assert!(output.status.success());
        assert!(String::from_utf8_lossy(&output.stdout).contains("bounded-ok"));
        assert!(output.stderr.is_empty());
    }

    #[test]
    fn byte_limited_run_stops_fast_oversized_stdout() {
        let started = Instant::now();
        let error = run_with_byte_limits(large_output_command(false), None, 1024, 1024)
            .err()
            .expect("oversized stdout must fail");
        assert!(matches!(
            error,
            ByteLimitedRunError::StdoutLimit { limit: 1024, .. }
        ));
        assert!(started.elapsed() < Duration::from_secs(10));
    }

    #[test]
    fn byte_limited_run_stops_fast_oversized_stderr() {
        let started = Instant::now();
        let error = run_with_byte_limits(large_output_command(true), None, 1024, 1024)
            .err()
            .expect("oversized stderr must fail");
        assert!(matches!(
            error,
            ByteLimitedRunError::StderrLimit { limit: 1024, .. }
        ));
        assert!(started.elapsed() < Duration::from_secs(10));
    }
}
