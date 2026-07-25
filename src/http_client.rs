use crate::tools::process_runner::{self, ByteLimitedOutput, ByteLimitedRunError, ManagedChild};
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::RecvTimeoutError;
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub struct HttpRequest {
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Option<String>,
}

#[derive(Debug)]
pub struct HttpResponse {
    pub body: String,
}

const MAX_RETRIES: u32 = 3;
const BASE_BACKOFF_MS: u64 = 500;
const CONNECT_TIMEOUT_SECS: &str = "10";
const POST_TIMEOUT_SECS: &str = "600";
/// Catalog endpoints (`/v1/models`) answer fast or not at all.
const GET_TIMEOUT_SECS: &str = "12";
/// Provider completions and model catalogs are normally far smaller than
/// this. Eight MiB leaves ample room for unusually large completions/tool
/// calls while bounding headers plus body before JSON parsing.
const RESPONSE_CAP_BYTES: usize = 8 * 1024 * 1024;
/// Streaming wire data has substantial JSON/SSE overhead per token, but is
/// reduced incrementally rather than retained verbatim. A separate 64 MiB cap
/// accommodates long reasoning streams while still bounding total transport.
const STREAM_RESPONSE_CAP_BYTES: usize = 64 * 1024 * 1024;
/// Upper bound on a raw form-post response (headers + body). OAuth token
/// responses are a few KB; anything larger is treated as a transport fault so
/// a hostile endpoint cannot make us buffer an unbounded reply.
const RAW_RESPONSE_CAP_BYTES: usize = 512 * 1024;
/// SSE records are normally a few KiB at most. This is deliberately generous
/// for providers that send a large tool argument in one data record.
const STREAM_EVENT_CAP_BYTES: usize = 1024 * 1024;
/// Matches the hardened web-fetch subprocess path: enough for actionable curl
/// diagnostics, but not enough for a hostile subprocess to consume memory.
const STDERR_CAP_BYTES: usize = 16 * 1024;
const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(60);
const WATCHDOG_POLL: Duration = Duration::from_millis(200);

#[derive(Debug)]
enum RequestError {
    /// A completed HTTP response with a non-2xx status.
    Status(u16, String),
    /// curl couldn't be spawned, the connection failed, or a similar
    /// transport-level problem occurred before/without a usable response.
    Transport(String),
    /// A transport failure known to have happened before the POST reached
    /// the provider, so retrying cannot duplicate the request.
    RetryableTransport(String),
}

impl RequestError {
    fn into_string(self) -> String {
        match self {
            RequestError::Status(status, body) => format_status_error(status, &body),
            RequestError::Transport(msg) | RequestError::RetryableTransport(msg) => {
                format_transport_error(&msg)
            }
        }
    }
}

/// Pull a short human-readable detail out of a provider error body (JSON or plain text).
fn extract_error_detail(body: &str) -> String {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    if let Ok(val) = crate::json::parse(trimmed) {
        if let Some(obj) = val.as_object() {
            // OpenAI / OpenRouter: {"error":{"message":"..."}}
            if let Some(err) = obj.get("error") {
                if let Some(msg) = err.get("message").and_then(|v| v.as_str()) {
                    return msg.trim().to_string();
                }
                if let Some(msg) = err.as_str() {
                    return msg.trim().to_string();
                }
            }
            // Anthropic-ish: {"type":"error","error":{"type":"...","message":"..."}}
            // Already handled above. Also bare {"message":"..."}.
            if let Some(msg) = obj.get("message").and_then(|v| v.as_str()) {
                return msg.trim().to_string();
            }
        }
    }
    // Collapse whitespace and cap length so huge HTML error pages stay readable.
    let flat: String = trimmed.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() > 280 {
        let short: String = flat.chars().take(280).collect();
        format!("{short}…")
    } else {
        flat
    }
}

/// True when an error string looks like a context-window / prompt-too-long
/// failure from a common provider. Used for reactive compaction.
pub fn is_context_limit_error(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    [
        "context length",
        "context window",
        "context_length_exceeded",
        "maximum context",
        "context limit",
        "prompt is too long",
        "too many tokens",
        "reduce the length of the messages",
        "input is too long",
        // avoid bare "max_tokens" alone: it appears in normal request logs
        "max_tokens is too large",
        "exceeds the model's",
        "exceeds model",
    ]
    .iter()
    .any(|n| lower.contains(n))
}

fn looks_like_context_limit(detail: &str) -> bool {
    is_context_limit_error(detail)
}

fn format_status_error(status: u16, body: &str) -> String {
    let detail = crate::redact::redact_secrets(&extract_error_detail(body));
    let advice = if looks_like_context_limit(&detail) {
        "Prompt exceeds the model context window. Start a new session (/clear) or continue so compaction can shrink history."
    } else {
        match status {
            401 | 403 => {
                "Authentication failed. Check your API key (env var or save one via /provider)."
            }
            404 => "Not found. Check the model id and that your provider supports it.",
            402 => "Payment required or insufficient credits on this provider.",
            429 => "Rate limited by the provider. Wait and retry, or switch model/provider.",
            500 | 502 => "Provider server error. Retry shortly, or switch provider.",
            503 | 529 => "Provider overloaded or unavailable. Retry shortly, or switch provider.",
            _ => "",
        }
    };

    if advice.is_empty() && detail.is_empty() {
        format!("HTTP {status} from provider.")
    } else if advice.is_empty() {
        format!("HTTP {status}: {detail}")
    } else if detail.is_empty() {
        format!("HTTP {status}: {advice}")
    } else {
        format!("HTTP {status}: {advice} Provider said: {detail}")
    }
}

fn format_transport_error(msg: &str) -> String {
    let m = msg.trim();
    if m.is_empty() {
        return "Network error: could not reach the provider. Check connectivity and that curl is on PATH.".into();
    }
    let lower = m.to_ascii_lowercase();
    if lower.contains("could not resolve host") || lower.contains("name or service not known") {
        return format!("Network error: DNS lookup failed ({m}). Check connectivity.");
    }
    if lower.contains("connection refused") {
        return format!(
            "Network error: connection refused ({m}). Is the provider running and reachable?"
        );
    }
    if lower.contains("timed out") || lower.contains("timeout") {
        return format!(
            "Network error: connection timed out ({m}). Retry, or check network/firewall."
        );
    }
    if lower.contains("failed to connect") || lower.contains("couldn't connect") {
        return format!(
            "Network error: could not connect ({m}). Check connectivity and provider URL."
        );
    }
    // curl missing from PATH shows up as a spawn error.
    if lower.contains("the system cannot find the file")
        || lower.contains("no such file or directory")
        || lower.contains("program not found")
        || lower.contains("not found") && lower.contains("curl")
    {
        return format!(
            "Network error: could not run curl ({m}). Install curl and ensure it is on PATH."
        );
    }
    format!("Network error: {m}")
}

fn is_retriable_status(status: u16) -> bool {
    matches!(status, 429 | 503 | 529)
}

fn backoff_delay(attempt: u32) -> Duration {
    Duration::from_millis(BASE_BACKOFF_MS * 2u64.saturating_pow(attempt.saturating_sub(1)))
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Global opt-in for [`debug_log_request`]. Off by default (H-03): request
/// URLs, headers, and bodies previously landed on disk unconditionally on
/// every provider call, with only heuristic secret redaction. Set from
/// `Config::debug_log_requests` at startup, or via `CAIRN_DEBUG_HTTP=1`.
static DEBUG_LOGGING_ENABLED: AtomicBool = AtomicBool::new(false);
static DEBUG_LOG_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);
static DEBUG_LOG_WRITE_LOCK: Mutex<()> = Mutex::new(());

/// Enables or disables writing request metadata for troubleshooting. Call
/// once at startup from the loaded config; defaults to disabled otherwise.
pub fn set_debug_logging_enabled(enabled: bool) {
    // Older versions wrote full URLs, header values, prompts, and source code
    // to this path. Remove that legacy dump on every startup before an
    // optional metadata-only replacement can be written.
    remove_legacy_debug_log(&debug_log_path());
    DEBUG_LOGGING_ENABLED.store(enabled, Ordering::Relaxed);
}

fn remove_legacy_debug_log(path: &std::path::Path) {
    let _ = std::fs::remove_file(path);
}

fn debug_logging_enabled() -> bool {
    let env_value = std::env::var("CAIRN_DEBUG_HTTP").ok();
    debug_logging_enabled_for(
        DEBUG_LOGGING_ENABLED.load(Ordering::Relaxed),
        env_value.as_deref(),
    )
}

fn debug_logging_enabled_for(config_enabled: bool, env_value: Option<&str>) -> bool {
    config_enabled
        || env_value
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
}

/// When explicitly enabled, records request *metadata* only - the URL origin
/// without userinfo or path, header names (never values), and body size - to
/// `~/.config/cairn-code/debug_request.json`. Header values and body content
/// (which can contain full prompts, source code, and credentials) are never
/// written, so there is nothing here for heuristic redaction to miss. The
/// file is overwritten (not appended) on every request, so it never
/// accumulates history beyond the most recent call; written atomically with
/// owner-only permissions where the OS supports it.
fn debug_log_request(req: &HttpRequest) {
    if !debug_logging_enabled() {
        return;
    }
    let path = debug_log_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = write_atomic_private(&path, &debug_dump_content(req));
}

fn debug_log_path() -> std::path::PathBuf {
    let dir = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());
    std::path::PathBuf::from(dir).join(".config/cairn-code/debug_request.json")
}

/// Pure formatting for [`debug_log_request`], split out so the "no header
/// values, no body content" contract is easy to test without touching disk.
fn debug_dump_content(req: &HttpRequest) -> String {
    let header_names = req
        .headers
        .iter()
        .map(|(k, _)| k.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    let body_bytes = req.body.as_ref().map(|b| b.len()).unwrap_or(0);
    format!(
        "timestamp_ms: {}\nurl: {}\nheader_names: {}\nbody_bytes: {}\n",
        now_millis(),
        sanitize_debug_url(&req.url),
        header_names,
        body_bytes,
    )
}

fn sanitize_debug_url(url: &str) -> String {
    const UNSAFE_URL: &str = "<invalid-or-unsafe-url>";

    let (scheme, remainder) = if url
        .get(..7)
        .is_some_and(|s| s.eq_ignore_ascii_case("http://"))
    {
        ("http", &url[7..])
    } else if url
        .get(..8)
        .is_some_and(|s| s.eq_ignore_ascii_case("https://"))
    {
        ("https", &url[8..])
    } else {
        return UNSAFE_URL.into();
    };
    let authority_end = remainder
        .find(|c| c == '/' || c == '\\' || c == '?' || c == '#')
        .unwrap_or(remainder.len());
    let authority = &remainder[..authority_end];
    if authority.matches('@').count() > 1 {
        return UNSAFE_URL.into();
    }
    let host = authority
        .rsplit_once('@')
        .map(|(_, host)| host)
        .unwrap_or(authority);

    if !valid_debug_authority(host) {
        return UNSAFE_URL.into();
    }

    // Paths can contain API keys (for example in a configurable proxy base
    // URL), so retain only the origin rather than attempting to redact them.
    format!("{scheme}://{host}/")
}

fn valid_debug_authority(authority: &str) -> bool {
    fn valid_port(port: &str) -> bool {
        !port.is_empty() && port.bytes().all(|b| b.is_ascii_digit()) && port.parse::<u16>().is_ok()
    }

    if let Some(ipv6) = authority.strip_prefix('[') {
        let Some(bracket) = ipv6.find(']') else {
            return false;
        };
        let address = &ipv6[..bracket];
        let suffix = &ipv6[bracket + 1..];
        return address.parse::<std::net::Ipv6Addr>().is_ok()
            && (suffix.is_empty() || suffix.strip_prefix(':').is_some_and(valid_port));
    }

    if !authority.is_ascii() || authority.contains(['[', ']', '%']) {
        return false;
    }
    let mut parts = authority.split(':');
    let hostname = parts.next().unwrap_or_default();
    let port = parts.next();
    if parts.next().is_some()
        || hostname.is_empty()
        || hostname.len() > 253
        || !hostname.split('.').all(|label| {
            !label.is_empty()
                && label.len() <= 63
                && label
                    .as_bytes()
                    .first()
                    .is_some_and(u8::is_ascii_alphanumeric)
                && label
                    .as_bytes()
                    .last()
                    .is_some_and(u8::is_ascii_alphanumeric)
                && label
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-')
        })
    {
        return false;
    }
    port.map(valid_port).unwrap_or(true)
}

/// Writes `contents` to `path` via a same-directory temp file + rename, so a
/// reader never observes a partial write, and restricts the file to
/// owner-read/write where the OS supports Unix permission bits.
fn write_atomic_private(path: &std::path::Path, contents: &str) -> std::io::Result<()> {
    let _guard = DEBUG_LOG_WRITE_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let sequence = DEBUG_LOG_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp = path.with_extension(format!("tmp-{}-{sequence}", std::process::id()));

    let result = (|| {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&tmp)?;
        file.write_all(contents.as_bytes())?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&tmp, path)
    })();

    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

/// Rejects header names/values containing control characters.
///
/// curl passes `-H` through verbatim, so an embedded CR or LF would splice
/// extra headers into the request. Values reach here from config and provider
/// code rather than from the model or the network, so this is defence in depth
/// — but it is the kind of check that is cheap now and expensive to add after
/// a new caller starts forwarding a value from somewhere less trusted.
fn check_header(name: &str, value: &str) -> Result<(), RequestError> {
    let bad = |s: &str| s.chars().any(|c| c.is_control());
    if bad(name) || bad(value) {
        return Err(RequestError::Transport(format!(
            "header {name:?} contains a control character"
        )));
    }
    Ok(())
}

/// Builds every curl invocation this module makes.
///
/// The hardening below has to hold on both the POST and the GET path, and it
/// only held on one of them until recently. One builder is what keeps them
/// from drifting apart again: a new option added here cannot land on a single
/// caller, and `check_header` cannot be skipped by a path that forgets it.
fn curl_command_for(
    method: &str,
    url: &str,
    headers: &[(String, String)],
    max_time: &str,
    body_from_stdin: bool,
) -> Result<Command, RequestError> {
    let mut cmd = Command::new("curl");
    // Must be the first argument so curl does not load settings (including
    // unsafe retry policies) from the user's curlrc.
    cmd.arg("-q");
    cmd.args([
        "-sS",
        "-i",
        "-X",
        method,
        // Confine the transport to HTTP(S) so a malformed base URL cannot
        // reach file:, scp:, or similar. `web_fetch` already does this; the
        // provider path had no equivalent.
        "--proto",
        "=http,https",
        "--connect-timeout",
        CONNECT_TIMEOUT_SECS,
        "--max-time",
        max_time,
    ]);
    for (k, v) in headers {
        check_header(k, v)?;
        cmd.arg("-H").arg(format!("{k}: {v}"));
    }
    // Disable "Expect: 100-continue" so the response is a single header block,
    // which keeps status-line parsing simple for large request bodies.
    cmd.arg("-H").arg("Expect:");
    if body_from_stdin {
        cmd.arg("--data-binary").arg("@-");
        cmd.stdin(Stdio::piped());
    }
    // The URL goes last, behind `--`: everything after that separator is a
    // URL, so it must follow every option rather than precede them.
    cmd.arg("--").arg(url);
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    Ok(cmd)
}

fn curl_command(req: &HttpRequest) -> Result<Command, RequestError> {
    curl_command_for(
        "POST",
        &req.url,
        &req.headers,
        POST_TIMEOUT_SECS,
        // The body is streamed in rather than passed as an argument: it can be
        // megabytes, and it is never safe on a command line.
        true,
    )
}

fn spawn_curl(req: &HttpRequest) -> Result<ManagedChild, String> {
    debug_log_request(req);

    let mut child = ManagedChild::spawn(curl_command(req).map_err(RequestError::into_string)?)
        .map_err(|error| format!("curl: {error}"))?;

    let body = req.body.clone().unwrap_or_default();
    let mut stdin = child.stdin.take().ok_or("curl: no stdin")?;
    std::thread::spawn(move || {
        let _ = stdin.write_all(body.as_bytes());
    });

    Ok(child)
}

fn run_curl_with_cap(
    command: Command,
    body: Option<Vec<u8>>,
    response_cap: usize,
) -> Result<ByteLimitedOutput, RequestError> {
    process_runner::run_with_byte_limits(command, body, response_cap, STDERR_CAP_BYTES)
        .map_err(byte_limited_curl_error)
}

fn byte_limited_curl_error(error: ByteLimitedRunError) -> RequestError {
    match error {
        ByteLimitedRunError::Spawn(reason) => {
            RequestError::RetryableTransport(format!("curl: {reason}"))
        }
        ByteLimitedRunError::StdoutLimit {
            limit,
            cleanup_error,
        } => RequestError::Transport(process_runner::with_cleanup(
            format!("provider response exceeded {limit} byte limit"),
            &cleanup_error,
        )),
        ByteLimitedRunError::StderrLimit {
            limit,
            cleanup_error,
        } => RequestError::Transport(process_runner::with_cleanup(
            format!("curl stderr exceeded {limit} byte limit"),
            &cleanup_error,
        )),
        ByteLimitedRunError::Read {
            stream,
            reason,
            cleanup_error,
        } => RequestError::Transport(process_runner::with_cleanup(
            format!("failed reading curl {stream}: {reason}"),
            &cleanup_error,
        )),
        ByteLimitedRunError::Wait(reason) => {
            RequestError::Transport(format!("failed waiting for curl: {reason}"))
        }
    }
}

fn curl_exit_error(status: ExitStatus, stderr: &[u8]) -> RequestError {
    let message = String::from_utf8_lossy(stderr).trim().to_string();
    if is_retryable_curl_exit_code(status.code()) {
        RequestError::RetryableTransport(message)
    } else {
        RequestError::Transport(message)
    }
}

fn is_retryable_curl_exit_code(code: Option<i32>) -> bool {
    // curl 5/6/7 are proxy resolution, host resolution, and connection
    // establishment failures. No HTTP request reached the provider.
    matches!(code, Some(5..=7))
}

fn parse_status_line(line: &str) -> u16 {
    line.split_whitespace()
        .nth(1)
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(0)
}

fn request_once(req: &HttpRequest) -> Result<HttpResponse, RequestError> {
    debug_log_request(req);
    let output = run_curl_with_cap(
        curl_command(req)?,
        Some(req.body.clone().unwrap_or_default().into_bytes()),
        RESPONSE_CAP_BYTES,
    )?;

    if !output.status.success() {
        return Err(curl_exit_error(output.status, &output.stderr));
    }

    let raw = String::from_utf8_lossy(&output.stdout);
    let split_at = raw
        .find("\r\n\r\n")
        .map(|i| i + 4)
        .or_else(|| raw.find("\n\n").map(|i| i + 2))
        .unwrap_or(0);
    let status = raw
        .get(..split_at)
        .and_then(|h| h.lines().next())
        .map(parse_status_line)
        .unwrap_or(0);
    let body = raw.get(split_at..).unwrap_or(&raw).to_string();

    if status < 200 || status >= 300 {
        return Err(RequestError::Status(status, body));
    }

    Ok(HttpResponse { body })
}

/// Sends a non-streaming request, retrying only transport errors known to
/// precede the POST. Completed HTTP responses are never safe to retry here.
pub fn request(req: &HttpRequest) -> Result<HttpResponse, String> {
    let mut attempt = 0;
    loop {
        match request_once(req) {
            Ok(resp) => return Ok(resp),
            Err(RequestError::RetryableTransport(_)) if attempt < MAX_RETRIES => {
                attempt += 1;
                std::thread::sleep(backoff_delay(attempt));
            }
            Err(e) => return Err(e.into_string()),
        }
    }
}

/// POST an `application/x-www-form-urlencoded` body and return the raw
/// `(status, body)` for any completed response.
///
/// This reuses the same hardened curl invocation as the streaming/POST paths
/// (first-argument `-q` so a user's `.curlrc` cannot redirect/proxy/trace the
/// request, plus connect/total timeouts and curl exit-status validation).
/// Unlike `request`, non-2xx responses are returned instead of folded into an
/// error string, because OAuth flows must read error bodies such as
/// `authorization_pending`. The response is size-bounded and this path does
/// not emit debug request logs, so credential-bearing bodies never touch disk.
pub fn form_post(url: &str, form_body: &str) -> Result<(u16, String), String> {
    let req = HttpRequest {
        url: url.to_string(),
        headers: vec![
            (
                "Content-Type".to_string(),
                "application/x-www-form-urlencoded".to_string(),
            ),
            ("Accept".to_string(), "application/json".to_string()),
        ],
        body: Some(form_body.to_string()),
    };
    let mut attempt = 0;
    loop {
        match request_raw_once(&req) {
            Ok(pair) => return Ok(pair),
            Err(RequestError::RetryableTransport(_)) if attempt < MAX_RETRIES => {
                attempt += 1;
                std::thread::sleep(backoff_delay(attempt));
            }
            Err(e) => return Err(e.into_string()),
        }
    }
}

/// Single attempt behind [`form_post`]. Builds the command via `curl_command`
/// directly (rather than `spawn_curl`) so credential-bearing OAuth bodies are
/// never written to the debug request log.
fn request_raw_once(req: &HttpRequest) -> Result<(u16, String), RequestError> {
    let output = run_curl_with_cap(
        curl_command(req)?,
        Some(req.body.clone().unwrap_or_default().into_bytes()),
        RAW_RESPONSE_CAP_BYTES,
    )?;
    if !output.status.success() {
        return Err(curl_exit_error(output.status, &output.stderr));
    }

    let raw = String::from_utf8_lossy(&output.stdout);
    let split_at = raw
        .find("\r\n\r\n")
        .map(|i| i + 4)
        .or_else(|| raw.find("\n\n").map(|i| i + 2))
        .unwrap_or(0);
    let status = raw
        .get(..split_at)
        .and_then(|h| h.lines().next())
        .map(parse_status_line)
        .unwrap_or(0);
    let body = raw.get(split_at..).unwrap_or(&raw).to_string();
    Ok((status, body))
}

/// GET with optional headers. Used for catalog endpoints like `/v1/models`.
/// Retries the same transient failures as [`request`]. Caps wall time via curl `--max-time`.
pub fn request_get(url: &str, headers: &[(String, String)]) -> Result<HttpResponse, String> {
    let mut attempt = 0;
    loop {
        match request_get_once(url, headers) {
            Ok(resp) => return Ok(resp),
            Err(RequestError::Status(status, _))
                if is_retriable_status(status) && attempt < MAX_RETRIES =>
            {
                attempt += 1;
                std::thread::sleep(backoff_delay(attempt));
            }
            Err(RequestError::Transport(_) | RequestError::RetryableTransport(_))
                if attempt < MAX_RETRIES =>
            {
                attempt += 1;
                std::thread::sleep(backoff_delay(attempt));
            }
            Err(e) => return Err(e.into_string()),
        }
    }
}

fn request_get_once(url: &str, headers: &[(String, String)]) -> Result<HttpResponse, RequestError> {
    let cmd = curl_command_for("GET", url, headers, GET_TIMEOUT_SECS, false)?;
    let output = run_curl_with_cap(cmd, None, RESPONSE_CAP_BYTES)?;
    if !output.status.success() {
        return Err(curl_exit_error(output.status, &output.stderr));
    }
    let raw = String::from_utf8_lossy(&output.stdout);
    let split_at = raw
        .find("\r\n\r\n")
        .map(|i| i + 4)
        .or_else(|| raw.find("\n\n").map(|i| i + 2))
        .unwrap_or(0);
    let status = raw
        .get(..split_at)
        .and_then(|h| h.lines().next())
        .map(parse_status_line)
        .unwrap_or(0);
    let body = raw.get(split_at..).unwrap_or(&raw).to_string();
    if status < 200 || status >= 300 {
        return Err(RequestError::Status(status, body));
    }
    Ok(HttpResponse { body })
}

#[derive(Debug)]
enum StreamOutcome {
    Cancelled,
    IdleTimeout,
    Other(RequestError),
}

#[derive(Debug)]
enum StreamReadFailure {
    ResponseLimit,
    EventLimit,
    Io(String),
}

enum StderrRead {
    Complete(Vec<u8>),
    Limit {
        cleanup_error: Option<String>,
    },
    Io {
        reason: String,
        cleanup_error: Option<String>,
    },
}

/// Read one line without allowing `BufRead::lines`/`read_line` to grow a
/// `String` before a limit can be checked. The total includes line terminators;
/// the per-event budget does not.
fn read_bounded_stream_line<R: BufRead>(
    reader: &mut R,
    total: &mut usize,
    total_limit: usize,
    event_limit: usize,
) -> Result<Option<String>, StreamReadFailure> {
    let mut line = Vec::with_capacity(8 * 1024);
    loop {
        let available = reader
            .fill_buf()
            .map_err(|error| StreamReadFailure::Io(error.to_string()))?;
        if available.is_empty() {
            if line.is_empty() {
                return Ok(None);
            }
            return String::from_utf8(line)
                .map(Some)
                .map_err(|error| StreamReadFailure::Io(error.to_string()));
        }

        let newline = available.iter().position(|byte| *byte == b'\n');
        let content_bytes = newline.unwrap_or(available.len());
        let consumed = content_bytes + usize::from(newline.is_some());
        if total
            .checked_add(consumed)
            .is_none_or(|next| next > total_limit)
        {
            return Err(StreamReadFailure::ResponseLimit);
        }
        if line
            .len()
            .checked_add(content_bytes)
            .is_none_or(|next| next > event_limit)
        {
            return Err(StreamReadFailure::EventLimit);
        }

        line.extend_from_slice(&available[..content_bytes]);
        reader.consume(consumed);
        *total += consumed;
        if newline.is_some() {
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            return String::from_utf8(line)
                .map(Some)
                .map_err(|error| StreamReadFailure::Io(error.to_string()));
        }
    }
}

fn terminate_stream_child(child: &Mutex<ManagedChild>) -> Option<String> {
    child
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .terminate()
        .err()
}

fn read_stream_stderr<R: Read>(mut stderr: R, child: &Mutex<ManagedChild>) -> StderrRead {
    let mut retained = Vec::with_capacity(STDERR_CAP_BYTES);
    let mut buffer = [0_u8; 8_192];
    loop {
        match stderr.read(&mut buffer) {
            Ok(0) => return StderrRead::Complete(retained),
            Ok(read) => {
                let remaining = STDERR_CAP_BYTES.saturating_sub(retained.len());
                retained.extend_from_slice(&buffer[..read.min(remaining)]);
                if read > remaining {
                    return StderrRead::Limit {
                        cleanup_error: terminate_stream_child(child),
                    };
                }
            }
            Err(error) => {
                return StderrRead::Io {
                    reason: error.to_string(),
                    cleanup_error: terminate_stream_child(child),
                }
            }
        }
    }
}

fn add_wait_cleanup(cleanup_error: &mut Option<String>, wait_error: Option<&std::io::Error>) {
    let Some(wait_error) = wait_error else {
        return;
    };
    match cleanup_error {
        Some(cleanup) => cleanup.push_str(&format!("; wait failed: {wait_error}")),
        None => *cleanup_error = Some(format!("wait failed: {wait_error}")),
    }
}

/// Runs one streaming attempt. `on_line` is called for every non-empty body
/// line of a 2xx response. Returns whether any line was emitted (needed by
/// the retry wrapper: once data has started flowing to the caller, retrying
/// from scratch would duplicate/confuse it, so retries only happen for
/// failures before the first emitted line).
fn request_streaming_attempt<F>(
    req: &HttpRequest,
    on_line: &mut F,
    cancel: Option<&AtomicBool>,
) -> Result<(), (StreamOutcome, bool)>
where
    F: FnMut(&str),
{
    let mut child = match spawn_curl(req) {
        Ok(c) => c,
        Err(e) => {
            return Err((
                StreamOutcome::Other(RequestError::RetryableTransport(e)),
                false,
            ))
        }
    };
    let stdout = child.stdout.take().expect("curl stdout is piped");
    let stderr = child.stderr.take().expect("curl stderr is piped");
    let mut reader = BufReader::with_capacity(64 * 1024, stdout);

    let child = Mutex::new(child);
    let last_activity = AtomicU64::new(now_millis());
    let timed_out = AtomicBool::new(false);
    let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();

    let mut emitted_any = false;
    let mut response_bytes = 0;
    let mut read_failure: Option<(StreamReadFailure, Option<String>)> = None;

    #[derive(PartialEq)]
    enum State {
        Status,
        Headers,
        Body,
    }

    let mut state = State::Status;
    let mut status: u16 = 0;
    let mut error_body = String::new();

    let child_ref = &child;
    let last_activity_ref = &last_activity;
    let timed_out_ref = &timed_out;

    let stderr_result = std::thread::scope(|scope| {
        let stderr_reader = scope.spawn(|| read_stream_stderr(stderr, &child));
        scope.spawn(move || loop {
            match stop_rx.recv_timeout(WATCHDOG_POLL) {
                Ok(()) => return,
                Err(RecvTimeoutError::Disconnected) => return,
                Err(RecvTimeoutError::Timeout) => {
                    let cancelled = cancel.map(|c| c.load(Ordering::Relaxed)).unwrap_or(false);
                    let idle_for =
                        now_millis().saturating_sub(last_activity_ref.load(Ordering::Relaxed));
                    if cancelled || idle_for >= STREAM_IDLE_TIMEOUT.as_millis() as u64 {
                        if !cancelled {
                            timed_out_ref.store(true, Ordering::Relaxed);
                        }
                        let _ = terminate_stream_child(child_ref);
                        return;
                    }
                }
            }
        });

        loop {
            let line = match read_bounded_stream_line(
                &mut reader,
                &mut response_bytes,
                STREAM_RESPONSE_CAP_BYTES,
                STREAM_EVENT_CAP_BYTES,
            ) {
                Ok(Some(line)) => line,
                Ok(None) => break,
                Err(failure) => {
                    let cleanup_error = terminate_stream_child(&child);
                    read_failure = Some((failure, cleanup_error));
                    break;
                }
            };
            last_activity.store(now_millis(), Ordering::Relaxed);
            match state {
                State::Status => {
                    status = parse_status_line(&line);
                    state = State::Headers;
                }
                State::Headers => {
                    if line.is_empty() || line == "\r" {
                        state = State::Body;
                    }
                }
                State::Body => {
                    if (200..300).contains(&status) {
                        if !line.is_empty() {
                            emitted_any = true;
                            on_line(&line);
                        }
                    } else if !line.is_empty() {
                        error_body.push_str(&line);
                        error_body.push('\n');
                    }
                }
            }
        }

        let stderr_result = stderr_reader.join().unwrap_or_else(|_| StderrRead::Io {
            reason: "stderr reader thread panicked".to_string(),
            cleanup_error: terminate_stream_child(&child),
        });
        let _ = stop_tx.send(());
        stderr_result
    });

    let wait_result = child
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .wait();
    let cancelled = cancel.map(|c| c.load(Ordering::Relaxed)).unwrap_or(false);
    if cancelled {
        return Err((StreamOutcome::Cancelled, emitted_any));
    }
    if timed_out.load(Ordering::Relaxed) {
        return Err((StreamOutcome::IdleTimeout, emitted_any));
    }

    if let Some((failure, mut cleanup_error)) = read_failure {
        add_wait_cleanup(&mut cleanup_error, wait_result.as_ref().err());
        let message = match failure {
            StreamReadFailure::ResponseLimit => process_runner::with_cleanup(
                format!("provider stream exceeded {STREAM_RESPONSE_CAP_BYTES} byte response limit"),
                &cleanup_error,
            ),
            StreamReadFailure::EventLimit => process_runner::with_cleanup(
                format!("provider stream event exceeded {STREAM_EVENT_CAP_BYTES} byte limit"),
                &cleanup_error,
            ),
            StreamReadFailure::Io(reason) => process_runner::with_cleanup(
                format!("failed reading provider stream: {reason}"),
                &cleanup_error,
            ),
        };
        return Err((
            StreamOutcome::Other(RequestError::Transport(message)),
            emitted_any,
        ));
    }

    let stderr = match stderr_result {
        StderrRead::Complete(stderr) => stderr,
        StderrRead::Limit { mut cleanup_error } => {
            add_wait_cleanup(&mut cleanup_error, wait_result.as_ref().err());
            return Err((
                StreamOutcome::Other(RequestError::Transport(process_runner::with_cleanup(
                    format!("curl stderr exceeded {STDERR_CAP_BYTES} byte limit"),
                    &cleanup_error,
                ))),
                emitted_any,
            ));
        }
        StderrRead::Io {
            reason,
            mut cleanup_error,
        } => {
            add_wait_cleanup(&mut cleanup_error, wait_result.as_ref().err());
            return Err((
                StreamOutcome::Other(RequestError::Transport(process_runner::with_cleanup(
                    format!("failed reading curl stderr: {reason}"),
                    &cleanup_error,
                ))),
                emitted_any,
            ));
        }
    };

    let exit = match wait_result {
        Ok(exit) => exit,
        Err(error) => {
            return Err((
                StreamOutcome::Other(RequestError::Transport(format!(
                    "failed waiting for curl: {error}"
                ))),
                emitted_any,
            ))
        }
    };

    if !exit.success() {
        return Err((
            StreamOutcome::Other(curl_exit_error(exit, &stderr)),
            emitted_any,
        ));
    }

    if status == 0 {
        return Err((
            StreamOutcome::Other(RequestError::Transport("no response".into())),
            emitted_any,
        ));
    }

    if status < 200 || status >= 300 {
        return Err((
            StreamOutcome::Other(RequestError::Status(status, error_body)),
            emitted_any,
        ));
    }

    Ok(())
}

/// Streams a request, calling `on_line` for each non-empty line of a 2xx
/// response body. A watchdog kills the underlying curl process if no data
/// arrives for 60s or if `cancel` is set. Only connection failures known to
/// precede the POST are retried, and only before any line has been emitted.
pub fn request_streaming_with_cancel<F>(
    req: &HttpRequest,
    mut on_line: F,
    cancel: Option<&AtomicBool>,
) -> Result<(), String>
where
    F: FnMut(&str),
{
    let mut attempt = 0;
    loop {
        match request_streaming_attempt(req, &mut on_line, cancel) {
            Ok(()) => return Ok(()),
            Err((StreamOutcome::Cancelled, _)) => {
                return Err("Request cancelled.".into());
            }
            Err((StreamOutcome::IdleTimeout, _)) => {
                return Err(
                    "Stream idle timeout: no data from the provider for 60s. Retry, or check network/provider status."
                        .into(),
                );
            }
            Err((StreamOutcome::Other(RequestError::RetryableTransport(_)), false))
                if attempt < MAX_RETRIES =>
            {
                attempt += 1;
                std::thread::sleep(backoff_delay(attempt));
            }
            Err((StreamOutcome::Other(e), _)) => return Err(e.into_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    /// Serves each response in order, one per accepted connection, on a
    /// fresh loopback port. Used to make curl (a real subprocess) exercise
    /// the retry path against a real, if trivial, HTTP server.
    fn start_mock_server(responses: Vec<&'static str>) -> String {
        start_owned_mock_server(responses.into_iter().map(String::from).collect())
    }

    fn start_owned_mock_server(responses: Vec<String>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            for resp in responses {
                if let Ok((mut stream, _)) = listener.accept() {
                    let mut buf = [0u8; 4096];
                    let _ = stream.read(&mut buf);
                    let _ = stream.write_all(resp.as_bytes());
                    let _ = stream.flush();
                }
            }
        });
        format!("http://{addr}")
    }

    #[test]
    fn test_is_retriable_status() {
        assert!(is_retriable_status(429));
        assert!(is_retriable_status(503));
        assert!(is_retriable_status(529));
        assert!(!is_retriable_status(500));
        assert!(!is_retriable_status(404));
    }

    #[test]
    fn test_backoff_delay_increases_exponentially() {
        assert_eq!(backoff_delay(1), Duration::from_millis(500));
        assert_eq!(backoff_delay(2), Duration::from_millis(1000));
        assert_eq!(backoff_delay(3), Duration::from_millis(2000));
    }

    #[test]
    fn test_post_does_not_retry_503() {
        let url = start_mock_server(vec![
            "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok",
        ]);
        let req = HttpRequest {
            url,
            headers: vec![],
            body: Some("{}".into()),
        };
        let err = request(&req).unwrap_err();
        assert!(err.contains("503"), "expected the original 503, got: {err}");
    }

    #[test]
    fn test_request_does_not_retry_on_400() {
        let url = start_mock_server(vec![
            "HTTP/1.1 400 Bad Request\r\nContent-Length: 6\r\nConnection: close\r\n\r\nbadreq",
        ]);
        let req = HttpRequest {
            url,
            headers: vec![],
            body: Some("{}".into()),
        };
        let err = request(&req).unwrap_err();
        assert!(
            err.contains("400"),
            "expected error to mention status 400, got: {err}"
        );
    }

    #[test]
    fn test_form_post_returns_raw_body_on_non_2xx() {
        // OAuth device polling relies on reading the error body of a 400,
        // so form_post must surface (status, body) instead of an error string.
        let url = start_mock_server(vec![
            "HTTP/1.1 400 Bad Request\r\nContent-Length: 32\r\nConnection: close\r\n\r\n{\"error\":\"authorization_pending\"}",
        ]);
        let (status, body) = form_post(&url, "grant_type=device_code").unwrap();
        assert_eq!(status, 400);
        assert!(body.contains("authorization_pending"), "got body: {body}");
    }

    #[test]
    fn test_form_post_returns_status_and_body_on_2xx() {
        let url = start_mock_server(vec![
            "HTTP/1.1 200 OK\r\nContent-Length: 20\r\nConnection: close\r\n\r\n{\"access_token\":\"x\"}",
        ]);
        let (status, body) = form_post(&url, "grant_type=refresh_token").unwrap();
        assert_eq!(status, 200);
        assert!(body.contains("access_token"), "got body: {body}");
    }

    #[test]
    fn test_curl_retry_classification_is_conservative() {
        assert!(is_retryable_curl_exit_code(Some(5)));
        assert!(is_retryable_curl_exit_code(Some(6)));
        assert!(is_retryable_curl_exit_code(Some(7)));
        assert!(!is_retryable_curl_exit_code(Some(18)));
        assert!(!is_retryable_curl_exit_code(Some(28)));
        assert!(!is_retryable_curl_exit_code(None));
    }

    #[test]
    fn test_post_curl_configures_timeouts_and_ignores_curlrc() {
        let req = HttpRequest {
            url: "https://example.com/v1/messages".into(),
            headers: vec![],
            body: Some("{}".into()),
        };
        let args: Vec<_> = curl_command(&req)
            .expect("plain headers are accepted")
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();

        assert_eq!(args.first().map(String::as_str), Some("-q"));
        assert!(args
            .windows(2)
            .any(|args| args == ["--connect-timeout", CONNECT_TIMEOUT_SECS]));
        assert!(args
            .windows(2)
            .any(|args| args == ["--max-time", POST_TIMEOUT_SECS]));
    }

    fn arg_strings(command: Command) -> Vec<String> {
        command
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect()
    }

    /// The hardening both request paths must satisfy, asserted the same way for
    /// each so neither can quietly lose it.
    fn assert_hardened(args: &[String], url: &str) {
        assert_eq!(args.first().map(String::as_str), Some("-q"), "{args:?}");
        assert!(args.windows(2).any(|a| a == ["--proto", "=http,https"]));
        assert!(args
            .windows(2)
            .any(|a| a == ["--connect-timeout", CONNECT_TIMEOUT_SECS]));
        // `--` must be the final option: everything after it is read as a URL,
        // so any option placed later would be sent as one instead.
        assert_eq!(
            args.iter().rev().take(2).cloned().collect::<Vec<_>>(),
            vec![url.to_string(), "--".to_string()],
            "URL must be last, immediately behind `--`: {args:?}"
        );
    }

    #[test]
    fn test_post_curl_restricts_protocols_and_terminates_options() {
        let url = "https://example.com/v1/messages";
        let req = HttpRequest {
            url: url.into(),
            headers: vec![("x-api-key".into(), "k".into())],
            body: Some("{}".into()),
        };
        let args = arg_strings(curl_command(&req).expect("plain headers are accepted"));

        assert_hardened(&args, url);
        assert!(args
            .windows(2)
            .any(|a| a == ["--max-time", POST_TIMEOUT_SECS]));
        assert!(args.windows(2).any(|a| a == ["--data-binary", "@-"]));
    }

    #[test]
    fn test_get_curl_restricts_protocols_and_terminates_options() {
        // The GET path had the same treatment applied by hand and no test
        // holding it there. Both paths now come from one builder; this pins
        // that the GET side really does carry the guarantees.
        let url = "https://example.com/v1/models";
        let headers = [("authorization".to_string(), "Bearer k".to_string())];
        let args = arg_strings(
            curl_command_for("GET", url, &headers, GET_TIMEOUT_SECS, false)
                .expect("plain headers are accepted"),
        );

        assert_hardened(&args, url);
        assert!(args
            .windows(2)
            .any(|a| a == ["--max-time", GET_TIMEOUT_SECS]));
        // A GET sends no body: `--data-binary @-` would make curl wait on a
        // stdin that nothing writes to.
        assert!(!args.iter().any(|a| a == "--data-binary"), "{args:?}");
    }

    #[test]
    fn test_curl_rejects_headers_containing_control_characters() {
        for (name, value) in [
            ("x-api-key", "abc\r\nX-Injected: 1"),
            ("x-api-key", "abc\ndef"),
            ("x-bad\rname", "v"),
        ] {
            let headers = vec![(name.to_string(), value.to_string())];
            // Both request paths, not just the POST one: a header smuggled
            // through the catalog request is the same injection.
            for method in ["POST", "GET"] {
                let error = curl_command_for(
                    method,
                    "https://example.com/",
                    &headers,
                    POST_TIMEOUT_SECS,
                    method == "POST",
                )
                .err()
                .unwrap_or_else(|| panic!("{method} {name}: {value:?} should be rejected"))
                .into_string();
                assert!(error.contains("control character"), "{method}: {error}");
            }
        }
    }

    #[test]
    fn test_curl_accepts_ordinary_header_values() {
        let req = HttpRequest {
            url: "https://example.com/".into(),
            headers: vec![
                ("content-type".into(), "application/json".into()),
                ("authorization".into(), "Bearer sk-abc.123-_~+/=".into()),
            ],
            body: None,
        };
        assert!(curl_command(&req).is_ok());
    }

    #[test]
    fn test_request_rejects_truncated_response() {
        let url = start_mock_server(vec![
            "HTTP/1.1 200 OK\r\nContent-Length: 10\r\nConnection: close\r\n\r\npartial",
            "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok",
        ]);
        let req = HttpRequest {
            url,
            headers: vec![],
            body: Some("{}".into()),
        };

        let result = request(&req);

        assert!(
            result.is_err(),
            "a truncated POST must not be accepted or retried"
        );
    }

    #[test]
    fn test_streaming_rejects_truncated_response_after_emitting_data() {
        let url = start_mock_server(vec![
            "HTTP/1.1 200 OK\r\nContent-Length: 20\r\nConnection: close\r\n\r\ndata: partial\n",
        ]);
        let req = HttpRequest {
            url,
            headers: vec![],
            body: Some("{}".into()),
        };
        let mut lines = Vec::new();

        let result =
            request_streaming_attempt(&req, &mut |line| lines.push(line.to_string()), None);

        assert_eq!(lines, ["data: partial"]);
        assert!(matches!(
            result,
            Err((StreamOutcome::Other(RequestError::Transport(_)), true))
        ));
    }

    #[test]
    fn bounded_stream_reader_accepts_normal_lines_and_tracks_total() {
        let input = b"data: one\r\ndata: two\n";
        let mut reader = BufReader::with_capacity(4, &input[..]);
        let mut total = 0;

        assert_eq!(
            read_bounded_stream_line(&mut reader, &mut total, 64, 32).unwrap(),
            Some("data: one".to_string())
        );
        assert_eq!(
            read_bounded_stream_line(&mut reader, &mut total, 64, 32).unwrap(),
            Some("data: two".to_string())
        );
        assert_eq!(
            read_bounded_stream_line(&mut reader, &mut total, 64, 32).unwrap(),
            None
        );
        assert_eq!(total, input.len());
    }

    #[test]
    fn bounded_stream_reader_rejects_event_and_total_overflow() {
        let mut event_reader = BufReader::with_capacity(4, &b"12345\n"[..]);
        let mut total = 0;
        assert!(matches!(
            read_bounded_stream_line(&mut event_reader, &mut total, 64, 4),
            Err(StreamReadFailure::EventLimit)
        ));

        let mut total_reader = BufReader::with_capacity(4, &b"one\ntwo\n"[..]);
        let mut total = 0;
        assert!(read_bounded_stream_line(&mut total_reader, &mut total, 7, 4).is_ok());
        assert!(matches!(
            read_bounded_stream_line(&mut total_reader, &mut total, 7, 4),
            Err(StreamReadFailure::ResponseLimit)
        ));
    }

    #[test]
    fn test_streaming_accepts_normal_response() {
        let body = "data: ok\n\n";
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let req = HttpRequest {
            url: start_owned_mock_server(vec![response]),
            headers: vec![],
            body: Some("{}".into()),
        };
        let mut lines = Vec::new();

        request_streaming_attempt(&req, &mut |line| lines.push(line.to_string()), None).unwrap();

        assert_eq!(lines, ["data: ok"]);
    }

    #[test]
    fn test_request_rejects_fast_oversized_stdout() {
        let body = "x".repeat(RESPONSE_CAP_BYTES);
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let req = HttpRequest {
            url: start_owned_mock_server(vec![response]),
            headers: vec![],
            body: Some("{}".into()),
        };

        let error = request(&req).unwrap_err();

        assert!(error.contains("response exceeded"), "{error}");
        assert!(error.contains(&RESPONSE_CAP_BYTES.to_string()), "{error}");
    }

    #[test]
    fn test_streaming_rejects_fast_oversized_event() {
        let body = format!("data: {}\n", "x".repeat(STREAM_EVENT_CAP_BYTES));
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let req = HttpRequest {
            url: start_owned_mock_server(vec![response]),
            headers: vec![],
            body: Some("{}".into()),
        };
        let mut lines = Vec::new();

        let result =
            request_streaming_attempt(&req, &mut |line| lines.push(line.to_string()), None);

        assert!(lines.is_empty());
        match result {
            Err((StreamOutcome::Other(RequestError::Transport(message)), false)) => {
                assert!(message.contains("stream event exceeded"), "{message}");
                assert!(
                    message.contains(&STREAM_EVENT_CAP_BYTES.to_string()),
                    "{message}"
                );
            }
            _ => panic!("oversized event must be rejected before emission"),
        }
    }

    #[test]
    fn test_status_error_auth_is_actionable() {
        let msg = format_status_error(401, r#"{"error":{"message":"Incorrect API key provided"}}"#);
        assert!(msg.contains("401"), "{msg}");
        assert!(
            msg.to_ascii_lowercase().contains("authentication")
                || msg.to_ascii_lowercase().contains("api key"),
            "{msg}"
        );
        assert!(msg.contains("Incorrect API key"), "{msg}");
    }

    #[test]
    fn test_status_error_rate_limit_is_actionable() {
        let msg = format_status_error(429, r#"{"error":{"message":"Rate limit exceeded"}}"#);
        assert!(msg.contains("429"), "{msg}");
        assert!(msg.to_ascii_lowercase().contains("rate"), "{msg}");
    }

    #[test]
    fn test_status_error_context_limit_detected() {
        let msg = format_status_error(
            400,
            r#"{"error":{"message":"This model's maximum context length is 128000 tokens"}}"#,
        );
        assert!(msg.to_ascii_lowercase().contains("context"), "{msg}");
        assert!(is_context_limit_error(&msg), "{msg}");
    }

    #[test]
    fn test_transport_error_dns() {
        let msg = format_transport_error("Could not resolve host: api.example.com");
        assert!(
            msg.to_ascii_lowercase().contains("dns")
                || msg.to_ascii_lowercase().contains("network"),
            "{msg}"
        );
    }

    /// H-03 regression: request logging must be off unless explicitly
    /// enabled, either via config (`set_debug_logging_enabled`) or the
    /// `CAIRN_DEBUG_HTTP` escape hatch.
    #[test]
    fn debug_logging_disabled_by_default() {
        assert!(!debug_logging_enabled_for(false, None));
    }

    #[test]
    fn debug_logging_enabled_via_config_flag() {
        assert!(debug_logging_enabled_for(true, None));
    }

    #[test]
    fn debug_logging_enabled_via_env_var() {
        assert!(debug_logging_enabled_for(false, Some("1")));
        assert!(debug_logging_enabled_for(false, Some("TRUE")));
        assert!(debug_logging_enabled_for(false, Some("True")));
        assert!(!debug_logging_enabled_for(false, Some("0")));
    }

    /// H-03: even when logging is enabled, the dump must never contain
    /// header values or body content — only metadata — so heuristic secret
    /// redaction has nothing left to fail to catch.
    #[test]
    fn debug_dump_never_contains_header_values_or_body() {
        let req = HttpRequest {
            url: "https://user:secret-url-password@api.example.com/v1/messages?api_key=secret-query#secret-fragment".into(),
            headers: vec![
                ("Authorization".into(), "Bearer sk-ant-supersecretvalue123456".into()),
                ("Content-Type".into(), "application/json".into()),
            ],
            body: Some(r#"{"prompt":"delete all my files","api_key":"sk-supersecret"}"#.into()),
        };
        let dump = debug_dump_content(&req);

        assert!(dump.contains("api.example.com"), "{dump}");
        assert!(
            !dump.contains("/v1/messages"),
            "URL paths are intentionally omitted: {dump}"
        );
        assert!(
            !dump.contains("secret-url-password"),
            "leaked URL userinfo: {dump}"
        );
        assert!(!dump.contains("secret-query"), "leaked URL query: {dump}");
        assert!(
            !dump.contains("secret-fragment"),
            "leaked URL fragment: {dump}"
        );
        assert!(
            dump.contains("Authorization"),
            "header *names* are metadata: {dump}"
        );
        assert!(dump.contains("Content-Type"), "{dump}");
        assert!(
            !dump.contains("sk-ant-supersecretvalue123456"),
            "leaked header value: {dump}"
        );
        assert!(
            !dump.contains("sk-supersecret"),
            "leaked body secret: {dump}"
        );
        assert!(
            !dump.contains("delete all my files"),
            "leaked prompt content: {dump}"
        );
        assert!(dump.contains("body_bytes"), "{dump}");
    }

    #[test]
    fn sanitize_debug_url_handles_urls_without_credentials() {
        assert_eq!(
            sanitize_debug_url("https://api.example.com/v1/messages?debug=true#response"),
            "https://api.example.com/"
        );
        assert_eq!(
            sanitize_debug_url("not-a-url?secret=value"),
            "<invalid-or-unsafe-url>"
        );
    }

    #[test]
    fn sanitize_debug_url_never_preserves_paths_or_malformed_input() {
        for (url, expected) in [
            (
                "https://proxy.example/tokens/sk-live",
                "https://proxy.example/",
            ),
            (
                "https://user:pw@[2001:db8::1]:11434/path?x#y",
                "https://[2001:db8::1]:11434/",
            ),
            (
                "HTTPS://USER:PW@example.com\\secret",
                "https://example.com/",
            ),
            ("mailto:user:secret@example.com", "<invalid-or-unsafe-url>"),
            ("//user:secret@example.com/path", "<invalid-or-unsafe-url>"),
            ("https://[invalid/path", "<invalid-or-unsafe-url>"),
            (
                "https://api.example.com:secret/path",
                "<invalid-or-unsafe-url>",
            ),
            ("https://[secret]/path", "<invalid-or-unsafe-url>"),
            (
                "https://user@extra@example.com/path",
                "<invalid-or-unsafe-url>",
            ),
            (
                "https://api.example.com:65536/path",
                "<invalid-or-unsafe-url>",
            ),
            (
                "https://api.example.com\u{2028}secret/path",
                "<invalid-or-unsafe-url>",
            ),
            ("💥https://secret.example/path", "<invalid-or-unsafe-url>"),
        ] {
            assert_eq!(sanitize_debug_url(url), expected, "input: {url}");
        }
    }

    #[test]
    fn initialization_removes_legacy_full_request_dump() {
        let dir = std::env::temp_dir().join(format!(
            "cairn-http-legacy-test-{}-{}",
            std::process::id(),
            DEBUG_LOG_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed),
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("debug_request.json");
        std::fs::write(&path, "Body:\nsecret prompt and API key").unwrap();

        remove_legacy_debug_log(&path);

        assert!(
            !path.exists(),
            "legacy secret-bearing debug log must be removed"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_atomic_private_is_atomic_and_owner_only() {
        let dir =
            std::env::temp_dir().join(format!("cairn-http-client-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("debug_request.json");

        write_atomic_private(&path, "first").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "first");

        write_atomic_private(&path, "second").unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "second",
            "second write should fully replace the first, not append or corrupt it"
        );

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(
                mode, 0o600,
                "debug log must be owner-read/write only, got {mode:o}"
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn concurrent_debug_writes_publish_only_complete_documents() {
        let dir = std::env::temp_dir().join(format!(
            "cairn-http-concurrent-test-{}-{}",
            std::process::id(),
            DEBUG_LOG_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed),
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = std::sync::Arc::new(dir.join("debug_request.json"));
        let documents: Vec<String> = (0..8)
            .map(|i| format!("document-{i}:{}", "x".repeat(4096)))
            .collect();

        let writers: Vec<_> = documents
            .iter()
            .cloned()
            .map(|document| {
                let path = path.clone();
                std::thread::spawn(move || write_atomic_private(&path, &document).unwrap())
            })
            .collect();
        for writer in writers {
            writer.join().unwrap();
            let observed = std::fs::read_to_string(path.as_ref()).unwrap();
            assert!(
                documents.contains(&observed),
                "observed a partial debug document"
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
