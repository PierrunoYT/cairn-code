use super::registry::Tool;
use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
use cap_std::{
    ambient_authority,
    fs::{Dir, File, OpenOptions},
};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

pub struct MemoryTool;
static MEMORY_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Match the aggregate byte and result caps used by grep/glob. A normal memory
/// store is much smaller, while one invocation accepts at most 1 MiB of memory
/// file content and returns at most 1,000 entries.
const MAX_MEMORY_READ_BYTES: usize = 1_048_576;
const MAX_MEMORY_RESULTS: usize = 1_000;
/// Match the grep/glob traversal cap and add a wall-clock backstop for slow
/// storage. Both limits apply to one permission-free memory invocation.
const MAX_MEMORY_VISITED_ENTRIES: usize = 100_000;
const MEMORY_READ_TIMEOUT: Duration = Duration::from_secs(2);
const MEMORY_TRUNCATED_MESSAGE: &str = "Memory scan truncated at safety limit.";

impl Tool for MemoryTool {
    fn name(&self) -> &str {
        "memory"
    }
    fn description(&self) -> &str {
        "Store and retrieve cross-session information. Use for user preferences, project conventions, and important context."
    }
    fn needs_permission(&self) -> bool {
        false
    }
    fn needs_permission_for(&self, input: &str) -> bool {
        crate::json::parse(input)
            .ok()
            .map(|value| {
                matches!(
                    value.get("action").and_then(|action| action.as_str()),
                    Some("save" | "delete")
                )
            })
            .unwrap_or(false)
    }
    fn permission_key(&self, input: &str) -> String {
        let action = crate::json::parse(input)
            .ok()
            .and_then(|value| value.get("action")?.as_str().map(str::to_owned));
        match action.as_deref() {
            Some(action @ ("save" | "delete")) => format!("memory:{action}"),
            _ => self.name().to_string(),
        }
    }

    fn input_schema(&self) -> String {
        r#"{"type":"object","properties":{"action":{"type":"string","enum":["save","recall","list","delete","search"]},"key":{"type":"string"},"content":{"type":"string"},"query":{"type":"string"}},"required":["action"]}"#.into()
    }

    fn execute(&self, input: &str) -> Result<String, String> {
        let val = crate::json::parse(input).map_err(|e| format!("invalid input: {e}"))?;
        let obj = val.as_object().ok_or("expected object")?;
        let action = obj
            .get("action")
            .and_then(|v| v.as_str())
            .ok_or("action required")?;

        match action {
            "save" => {
                let key = obj
                    .get("key")
                    .and_then(|v| v.as_str())
                    .ok_or("key required for save")?;
                let content = obj.get("content").and_then(|v| v.as_str()).unwrap_or("");
                let root =
                    open_memory_dir(true).map_err(|e| format!("open memory directory: {e}"))?;
                save_memory(&root, key, content)?;
                Ok(format!("Saved memory '{}'", key))
            }
            "recall" => {
                let key = obj
                    .get("key")
                    .and_then(|v| v.as_str())
                    .ok_or("key required for recall")?;
                let mut budget = MemoryBudget::new();
                let Some(root) = open_existing_memory_dir()? else {
                    return Err(format!("Memory '{}' not found", key));
                };
                recall_memory(&root, key, &mut budget)
            }
            "list" => {
                let query = obj.get("query").and_then(|v| v.as_str()).unwrap_or("");
                let mut budget = MemoryBudget::new();
                let Some(root) = open_existing_memory_dir()? else {
                    return Ok("No memories found.".to_string());
                };
                list_memories(&root, query, &mut budget)
            }
            "delete" => {
                let key = obj
                    .get("key")
                    .and_then(|v| v.as_str())
                    .ok_or("key required for delete")?;
                let Some(root) = open_existing_memory_dir()? else {
                    return Err(format!("Memory '{}' not found", key));
                };
                let name = memory_file_name(key)?;
                reject_symlink(&root, &name, key)?;
                root.remove_file(&name).map_err(|e| {
                    if e.kind() == std::io::ErrorKind::NotFound {
                        format!("Memory '{}' not found", key)
                    } else {
                        format!("delete: {e}")
                    }
                })?;
                Ok(format!("Deleted memory '{}'", key))
            }
            "search" => {
                let query = obj.get("query").and_then(|v| v.as_str()).unwrap_or("");
                if query.is_empty() {
                    return Err("query required for search".into());
                }
                let mut budget = MemoryBudget::new();
                let Some(root) = open_existing_memory_dir()? else {
                    return Ok("No memories match query.".to_string());
                };
                search_memories(&root, query, &mut budget)
            }
            _ => Err(format!("Unknown action: {action}")),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MemoryLimit {
    Bytes,
    Results,
    Work,
    Deadline,
}

struct MemoryBudget {
    max_bytes: usize,
    max_results: usize,
    max_visited: usize,
    deadline: Instant,
    bytes_read: usize,
    results: usize,
    visited: usize,
    limit: Option<MemoryLimit>,
}

impl MemoryBudget {
    fn new() -> Self {
        Self::with_limits(
            MAX_MEMORY_READ_BYTES,
            MAX_MEMORY_RESULTS,
            MAX_MEMORY_VISITED_ENTRIES,
            Instant::now()
                .checked_add(MEMORY_READ_TIMEOUT)
                .unwrap_or_else(Instant::now),
        )
    }

    fn with_limits(
        max_bytes: usize,
        max_results: usize,
        max_visited: usize,
        deadline: Instant,
    ) -> Self {
        Self {
            max_bytes,
            max_results,
            max_visited,
            deadline,
            bytes_read: 0,
            results: 0,
            visited: 0,
            limit: None,
        }
    }

    fn check_deadline(&mut self) -> bool {
        if Instant::now() >= self.deadline {
            self.stop(MemoryLimit::Deadline);
            false
        } else {
            true
        }
    }

    fn visit(&mut self) -> bool {
        if !self.check_deadline() {
            return false;
        }
        if self.visited >= self.max_visited {
            self.stop(MemoryLimit::Work);
            return false;
        }
        self.visited += 1;
        true
    }

    fn record_result(&mut self) -> bool {
        if !self.check_deadline() {
            return false;
        }
        if self.results >= self.max_results {
            self.stop(MemoryLimit::Results);
            return false;
        }
        self.results += 1;
        true
    }

    fn remaining_bytes(&self) -> usize {
        self.max_bytes.saturating_sub(self.bytes_read)
    }

    fn record_bytes(&mut self, bytes: usize) -> bool {
        if bytes > self.remaining_bytes() {
            self.stop(MemoryLimit::Bytes);
            return false;
        }
        self.bytes_read += bytes;
        true
    }

    fn stop(&mut self, limit: MemoryLimit) {
        if self.limit.is_none() {
            self.limit = Some(limit);
        }
    }

    fn truncated(&self) -> bool {
        self.limit.is_some()
    }

    fn recall_error(&self, key: &str) -> String {
        match self.limit {
            Some(MemoryLimit::Bytes) => format!(
                "Memory '{key}' exceeds the {} byte read limit",
                self.max_bytes
            ),
            Some(MemoryLimit::Deadline) => format!(
                "Memory '{key}' exceeded the {} ms read deadline",
                MEMORY_READ_TIMEOUT.as_millis()
            ),
            Some(MemoryLimit::Results | MemoryLimit::Work) | None => {
                format!("Memory '{key}' exceeded a safety limit")
            }
        }
    }
}

fn recall_memory(root: &Dir, key: &str, budget: &mut MemoryBudget) -> Result<String, String> {
    let Some(content) = read_memory_bounded(root, key, budget)? else {
        return Err(budget.recall_error(key));
    };
    if !budget.record_result() {
        return Err(budget.recall_error(key));
    }
    let (_, body) = parse_frontmatter(&content);
    Ok(format!("---\n{}\n{}", key, body.trim()))
}

fn list_memories(root: &Dir, query: &str, budget: &mut MemoryBudget) -> Result<String, String> {
    let mut entries: Vec<String> = Vec::new();
    for entry in root.entries().map_err(|e| format!("read dir: {e}"))? {
        if !budget.visit() {
            break;
        }
        let entry = entry.map_err(|e| format!("entry: {e}"))?;
        let name = entry.file_name().to_string_lossy().to_string();
        let Some(key) = name.strip_suffix(".md") else {
            continue;
        };
        if memory_file_name(key).is_err() {
            continue;
        }

        if query.is_empty() {
            if validate_memory_file(root, &name, key).is_err() || !budget.check_deadline() {
                continue;
            }
        } else {
            let content = match read_memory_bounded(root, key, budget) {
                Ok(Some(content)) => content,
                Ok(None) => break,
                Err(_) => continue,
            };
            if !content.contains(query) {
                continue;
            }
        }
        if !budget.record_result() {
            break;
        }
        entries.push(key.to_string());
    }

    let mut output = if entries.is_empty() {
        "No memories found.".to_string()
    } else {
        format!("Memories:\n{}", entries.join("\n"))
    };
    append_truncation(&mut output, budget);
    Ok(output)
}

fn search_memories(root: &Dir, query: &str, budget: &mut MemoryBudget) -> Result<String, String> {
    let mut results: Vec<(String, String)> = Vec::new();
    for entry in root.entries().map_err(|e| format!("read dir: {e}"))? {
        if !budget.visit() {
            break;
        }
        let entry = entry.map_err(|e| format!("entry: {e}"))?;
        let name = entry.file_name().to_string_lossy().to_string();
        let Some(key) = name.strip_suffix(".md") else {
            continue;
        };
        if memory_file_name(key).is_err() {
            continue;
        }

        let content = match read_memory_bounded(root, key, budget) {
            Ok(Some(content)) => content,
            Ok(None) => break,
            Err(_) => continue,
        };
        let (_, body) = parse_frontmatter(&content);
        if body.contains(query) || key.contains(query) {
            if !budget.record_result() {
                break;
            }
            let preview: String = body.chars().take(200).collect();
            results.push((key.to_string(), preview));
        }
    }

    let mut output = if results.is_empty() {
        "No memories match query.".to_string()
    } else {
        let out: Vec<String> = results
            .iter()
            .map(|(key, body)| format!("{key}: {body}"))
            .collect();
        format!("Search results:\n{}", out.join("\n---\n"))
    };
    append_truncation(&mut output, budget);
    Ok(output)
}

fn append_truncation(output: &mut String, budget: &MemoryBudget) {
    if budget.truncated() {
        output.push('\n');
        output.push_str(MEMORY_TRUNCATED_MESSAGE);
    }
}

fn memory_home() -> Result<PathBuf, String> {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .map_err(|_| "HOME or USERPROFILE is required for memory storage".to_string())
}

fn validate_memory_key(key: &str) -> Result<(), String> {
    if key.is_empty()
        || !key
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
    {
        return Err("memory key must contain only ASCII letters, numbers, '-' or '_'".into());
    }
    Ok(())
}

fn memory_file_name(key: &str) -> Result<String, String> {
    validate_memory_key(key)?;
    Ok(format!("{key}.md"))
}

fn open_memory_dir(create: bool) -> std::io::Result<Dir> {
    let home =
        memory_home().map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
    open_memory_dir_at(&home, create)
}

fn open_memory_dir_at(home: &Path, create: bool) -> std::io::Result<Dir> {
    let mut current = Dir::open_ambient_dir(home, ambient_authority())?;
    for component in [".config", "cairn-code", "memory"] {
        if create {
            match current.create_dir(component) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error),
            }
        }
        current = current.open_dir_nofollow(component)?;
    }
    Ok(current)
}

fn open_existing_memory_dir() -> Result<Option<Dir>, String> {
    match open_memory_dir(false) {
        Ok(dir) => Ok(Some(dir)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(format!("open memory directory: {error}")),
    }
}

fn save_memory(root: &Dir, key: &str, content: &str) -> Result<(), String> {
    let name = memory_file_name(key)?;
    let now = timestamp();
    let (created, existing_content) = match read_memory_file(root, &name, key) {
        Ok(existing) => parse_frontmatter(&existing),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => (now.clone(), String::new()),
        Err(error) => return Err(format!("read existing memory: {error}")),
    };
    let body = if content.is_empty() {
        &existing_content
    } else {
        content
    };
    let output =
        format!("---\nkey: {key}\ncreated_at: {created}\nupdated_at: {now}\n---\n\n{body}");
    write_memory_file_atomic(root, &name, output.as_bytes()).map_err(|e| format!("write: {e}"))
}

fn write_memory_file_atomic(root: &Dir, name: &str, contents: &[u8]) -> std::io::Result<()> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);

    for _ in 0..16 {
        let sequence = MEMORY_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let temp_name = format!(".memory-{}-{sequence}.tmp", std::process::id());
        let mut file = match open_memory_file(root, &temp_name, &temp_name, &options) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        };

        let result = (|| {
            file.write_all(contents)?;
            file.sync_all()?;
            drop(file);
            root.rename(&temp_name, root, name)
        })();
        if result.is_err() {
            let _ = root.remove_file(&temp_name);
        }
        return result;
    }

    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        "could not allocate a temporary memory file",
    ))
}

fn reject_symlink(root: &Dir, name: &str, key: &str) -> Result<(), String> {
    match root.symlink_metadata(name) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() {
                return Err(format!("refusing to follow symlink for memory '{key}'"));
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(format!("inspect memory path: {error}")),
    }
    Ok(())
}

fn open_memory_file(
    root: &Dir,
    name: &str,
    key: &str,
    options: &OpenOptions,
) -> std::io::Result<File> {
    let mut options = options.clone();
    options.follow(FollowSymlinks::No);
    root.open_with(name, &options).map_err(|error| {
        if root
            .symlink_metadata(name)
            .map(|metadata| metadata.file_type().is_symlink())
            .unwrap_or(false)
        {
            std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                format!("refusing to follow symlink for memory '{key}'"),
            )
        } else {
            error
        }
    })
}

fn read_memory_file(root: &Dir, name: &str, key: &str) -> std::io::Result<String> {
    let mut options = OpenOptions::new();
    options.read(true);
    let mut file = open_memory_file(root, name, key, &options)?;
    let mut content = String::new();
    file.read_to_string(&mut content)?;
    Ok(content)
}

fn read_memory_file_bounded(
    root: &Dir,
    name: &str,
    key: &str,
    budget: &mut MemoryBudget,
) -> std::io::Result<Option<String>> {
    if !budget.check_deadline() {
        return Ok(None);
    }

    let mut options = OpenOptions::new();
    options.read(true);
    let mut file = open_memory_file(root, name, key, &options)?;
    let remaining = budget.remaining_bytes();
    let mut bytes = Vec::new();
    Read::by_ref(&mut file)
        .take(remaining.saturating_add(1) as u64)
        .read_to_end(&mut bytes)?;
    if !budget.record_bytes(bytes.len()) || !budget.check_deadline() {
        return Ok(None);
    }

    String::from_utf8(bytes).map(Some).map_err(|error| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("stream did not contain valid UTF-8: {error}"),
        )
    })
}

fn validate_memory_file(root: &Dir, name: &str, key: &str) -> std::io::Result<()> {
    let mut options = OpenOptions::new();
    options.read(true);
    open_memory_file(root, name, key, &options).map(|_| ())
}

fn read_memory_bounded(
    root: &Dir,
    key: &str,
    budget: &mut MemoryBudget,
) -> Result<Option<String>, String> {
    let name = memory_file_name(key)?;
    read_memory_file_bounded(root, &name, key, budget).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            format!("Memory '{}' not found", key)
        } else {
            format!("read: {e}")
        }
    })
}

/// Days since the Unix epoch (1970-01-01) to a proleptic Gregorian (year,
/// month, day). Howard Hinnant's `civil_from_days`:
/// <http://howardhinnant.github.io/date_algorithms.html>
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

fn timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    let nanos = dur.subsec_nanos();
    let days = (secs / 86400) as i64;
    let time_secs = secs % 86400;
    let hours = time_secs / 3600;
    let mins = (time_secs % 3600) / 60;
    let sec = time_secs % 60;
    let (year, month, day) = civil_from_days(days);
    format!("{year:04}-{month:02}-{day:02}T{hours:02}:{mins:02}:{sec:02}.{nanos:06}Z")
}

fn parse_frontmatter(content: &str) -> (String, String) {
    let trimmed = content.trim_start();
    if !trimmed.starts_with("---") {
        return (String::new(), trimmed.to_string());
    }
    let after_delim = trimmed.trim_start_matches("---").trim_start();
    if let Some(end) = after_delim.find("\n---") {
        let front = &after_delim[..end];
        let body = after_delim[end + 4..].trim_start().to_string();
        let mut created = String::new();
        for line in front.lines() {
            if let Some(val) = line.strip_prefix("created_at:") {
                created = val.trim().trim_matches('"').to_string();
            }
        }
        (created, body)
    } else {
        (String::new(), trimmed.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_path(label: &str) -> PathBuf {
        use std::time::{SystemTime, UNIX_EPOCH};

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("cairn-memory-{label}-{nanos}"))
    }

    fn test_budget(max_bytes: usize, max_results: usize, max_visited: usize) -> MemoryBudget {
        MemoryBudget::with_limits(
            max_bytes,
            max_results,
            max_visited,
            Instant::now() + Duration::from_secs(30),
        )
    }

    fn write_test_memory(base: &Path, key: &str, content: &str) {
        fs::write(
            base.join(format!(".config/cairn-code/memory/{key}.md")),
            content,
        )
        .unwrap();
    }

    #[test]
    fn recall_enforces_the_byte_boundary() {
        let base = temp_path("recall-byte-budget");
        fs::create_dir_all(&base).unwrap();
        let root = open_memory_dir_at(&base, true).unwrap();
        write_test_memory(&base, "bounded", "123456");

        let mut exact = test_budget(6, 1, 1);
        assert_eq!(
            recall_memory(&root, "bounded", &mut exact).unwrap(),
            "---\nbounded\n123456"
        );
        assert!(!exact.truncated());

        let mut too_small = test_budget(5, 1, 1);
        let error = recall_memory(&root, "bounded", &mut too_small).unwrap_err();
        assert!(error.contains("5 byte read limit"), "{error}");
        assert_eq!(too_small.limit, Some(MemoryLimit::Bytes));

        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn search_enforces_the_cumulative_byte_boundary() {
        let base = temp_path("search-byte-budget");
        fs::create_dir_all(&base).unwrap();
        let root = open_memory_dir_at(&base, true).unwrap();
        write_test_memory(&base, "first", "needle");
        write_test_memory(&base, "second", "needle");

        let mut exact = test_budget(12, 10, 10);
        let output = search_memories(&root, "needle", &mut exact).unwrap();
        assert_eq!(output.matches(": needle").count(), 2, "{output}");
        assert!(!output.contains(MEMORY_TRUNCATED_MESSAGE), "{output}");

        let mut too_small = test_budget(11, 10, 10);
        let output = search_memories(&root, "needle", &mut too_small).unwrap();
        assert_eq!(output.matches(": needle").count(), 1, "{output}");
        assert!(output.contains(MEMORY_TRUNCATED_MESSAGE), "{output}");
        assert_eq!(too_small.limit, Some(MemoryLimit::Bytes));

        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn list_enforces_the_result_count_boundary() {
        let base = temp_path("list-result-budget");
        fs::create_dir_all(&base).unwrap();
        let root = open_memory_dir_at(&base, true).unwrap();
        write_test_memory(&base, "first", "one");
        write_test_memory(&base, "second", "two");

        let mut exact = test_budget(100, 2, 10);
        let output = list_memories(&root, "", &mut exact).unwrap();
        assert!(output.contains("first"), "{output}");
        assert!(output.contains("second"), "{output}");
        assert!(!output.contains(MEMORY_TRUNCATED_MESSAGE), "{output}");

        let mut capped = test_budget(100, 1, 10);
        let output = list_memories(&root, "", &mut capped).unwrap();
        assert_eq!(capped.results, 1);
        assert!(output.contains(MEMORY_TRUNCATED_MESSAGE), "{output}");
        assert_eq!(capped.limit, Some(MemoryLimit::Results));

        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn list_enforces_the_visited_entry_boundary() {
        let base = temp_path("list-work-budget");
        fs::create_dir_all(&base).unwrap();
        let root = open_memory_dir_at(&base, true).unwrap();
        write_test_memory(&base, "first", "one");
        write_test_memory(&base, "second", "two");

        let mut exact = test_budget(100, 10, 2);
        let output = list_memories(&root, "", &mut exact).unwrap();
        assert!(!output.contains(MEMORY_TRUNCATED_MESSAGE), "{output}");

        let mut capped = test_budget(100, 10, 1);
        let output = list_memories(&root, "", &mut capped).unwrap();
        assert_eq!(capped.visited, 1);
        assert!(output.contains(MEMORY_TRUNCATED_MESSAGE), "{output}");
        assert_eq!(capped.limit, Some(MemoryLimit::Work));

        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn recall_enforces_the_deadline_boundary() {
        let base = temp_path("recall-deadline-budget");
        fs::create_dir_all(&base).unwrap();
        let root = open_memory_dir_at(&base, true).unwrap();
        write_test_memory(&base, "bounded", "content");
        let mut budget = MemoryBudget::with_limits(100, 1, 1, Instant::now());

        let error = recall_memory(&root, "bounded", &mut budget).unwrap_err();
        assert!(error.contains("read deadline"), "{error}");
        assert_eq!(budget.limit, Some(MemoryLimit::Deadline));

        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn civil_from_days_matches_known_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(31), (1970, 2, 1));
        assert_eq!(civil_from_days(59), (1970, 3, 1)); // 1970 is not a leap year
        assert_eq!(civil_from_days(365), (1971, 1, 1));
        assert_eq!(civil_from_days(10_957), (2000, 1, 1));
        assert_eq!(civil_from_days(11_016), (2000, 2, 29)); // leap day
        assert_eq!(civil_from_days(11_017), (2000, 3, 1));
    }

    #[test]
    fn civil_from_days_covers_every_day_of_a_leap_year() {
        // Regression: the old day/month math (`1 + days % 28`) could never
        // produce day 29, 30, or 31 for any month.
        let start = 10_957; // 2000-01-01
        let mut saw_31 = false;
        for offset in 0..366 {
            let (year, month, day) = civil_from_days(start + offset);
            assert_eq!(year, 2000);
            assert!((1..=12).contains(&month), "month {month} out of range");
            let max_day = match month {
                1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
                4 | 6 | 9 | 11 => 30,
                2 => 29,
                _ => unreachable!(),
            };
            assert!((1..=max_day).contains(&day), "invalid date: {month}/{day}");
            saw_31 |= day == 31;
        }
        assert!(saw_31, "a 31-day month should produce day 31");
    }

    #[test]
    fn timestamp_produces_a_plausible_date() {
        let stamp = timestamp();
        let year: u32 = stamp[0..4].parse().unwrap();
        let month: u32 = stamp[5..7].parse().unwrap();
        let day: u32 = stamp[8..10].parse().unwrap();
        assert!(year >= 1970, "{stamp}");
        assert!((1..=12).contains(&month), "{stamp}");
        assert!((1..=31).contains(&day), "{stamp}");
    }

    #[test]
    fn test_parse_frontmatter_basic() {
        let input = "---\nkey: test\ncreated_at: 2026-06-28T12:00:00Z\n---\n\nHello world";
        let (created, body) = parse_frontmatter(input);
        assert_eq!(created, "2026-06-28T12:00:00Z");
        assert_eq!(body.trim(), "Hello world");
    }

    #[test]
    fn test_parse_frontmatter_no_frontmatter() {
        let (created, body) = parse_frontmatter("Just plain text");
        assert_eq!(created, "");
        assert_eq!(body, "Just plain text");
    }

    #[test]
    fn test_parse_frontmatter_missing_delim() {
        let input = "---\nkey: test\nno closing delim";
        let (_created, body) = parse_frontmatter(input);
        assert!(body.contains("key: test"));
    }

    #[test]
    fn test_parse_frontmatter_empty() {
        let (created, body) = parse_frontmatter("");
        assert_eq!(created, "");
        assert_eq!(body, "");
    }

    #[test]
    fn test_tool_name_and_description() {
        let tool = MemoryTool;
        assert_eq!(tool.name(), "memory");
        assert!(tool.description().contains("cross-session"));
    }

    #[test]
    fn test_mutating_actions_need_permission() {
        let tool = MemoryTool;
        assert!(tool.needs_permission_for(r#"{"action":"save","key":"test"}"#));
        assert!(tool.needs_permission_for(r#"{"action":"delete","key":"test"}"#));
        assert_eq!(
            tool.permission_key(r#"{"action":"save","key":"test"}"#),
            "memory:save"
        );
        assert_eq!(
            tool.permission_key(r#"{"action":"delete","key":"test"}"#),
            "memory:delete"
        );
        assert!(!tool.needs_permission_for(r#"{"action":"recall","key":"test"}"#));
        assert_eq!(
            tool.permission_key(r#"{"action":"recall","key":"test"}"#),
            "memory"
        );
        assert!(!tool.needs_permission_for(r#"{"action":"list"}"#));
        assert!(!tool.needs_permission_for("invalid"));
    }

    #[test]
    fn test_memory_file_name_rejects_unsafe_keys() {
        for key in [
            "",
            "../secret",
            "..\\secret",
            "nested/key",
            "nested\\key",
            ".",
            "two words",
        ] {
            assert!(memory_file_name(key).is_err(), "accepted unsafe key: {key}");
        }
        assert_eq!(memory_file_name("safe-key_123").unwrap(), "safe-key_123.md");
    }

    #[cfg(unix)]
    #[test]
    fn test_memory_path_rejects_symlink() {
        use std::os::unix::fs::symlink;

        let base = temp_path("symlink-test");
        let outside = base.join("outside.md");
        fs::create_dir_all(&base).unwrap();
        let root = open_memory_dir_at(&base, true).unwrap();
        fs::write(&outside, "secret").unwrap();
        symlink(&outside, base.join(".config/cairn-code/memory/linked.md")).unwrap();

        let error = read_memory_file(&root, "linked.md", "linked").unwrap_err();
        assert!(
            error.to_string().contains("symlink"),
            "unexpected error: {error}"
        );

        let _ = fs::remove_dir_all(base);
    }

    #[cfg(unix)]
    #[test]
    fn test_directory_capability_blocks_symlink_swap() {
        use std::os::unix::fs::symlink;

        let base = temp_path("symlink-swap-test");
        let outside = base.join("outside.md");
        fs::create_dir_all(&base).unwrap();
        let root = open_memory_dir_at(&base, true).unwrap();
        fs::write(&outside, "secret").unwrap();

        reject_symlink(&root, "linked.md", "linked").unwrap();
        symlink(&outside, base.join(".config/cairn-code/memory/linked.md")).unwrap();

        assert!(read_memory_file(&root, "linked.md", "linked").is_err());

        let _ = fs::remove_dir_all(base);
    }

    #[cfg(unix)]
    #[test]
    fn test_memory_root_rejects_symlinked_parent() {
        use std::os::unix::fs::symlink;

        let base = temp_path("parent-symlink-test");
        let home = base.join("home");
        let outside = base.join("outside");
        fs::create_dir_all(&home).unwrap();
        fs::create_dir_all(outside.join("cairn-code/memory")).unwrap();
        fs::write(
            outside.join("cairn-code/memory/sentinel.md"),
            "outside secret",
        )
        .unwrap();
        symlink(&outside, home.join(".config")).unwrap();

        assert!(open_memory_dir_at(&home, false).is_err());
        assert_eq!(
            fs::read_to_string(outside.join("cairn-code/memory/sentinel.md")).unwrap(),
            "outside secret"
        );

        let _ = fs::remove_dir_all(base);
    }

    #[cfg(windows)]
    #[test]
    fn test_memory_root_rejects_junctioned_parent() {
        use std::process::Command;

        let base = temp_path("junction-test");
        let home = base.join("home");
        let outside = base.join("outside");
        let junction = home.join(".config");
        fs::create_dir_all(&home).unwrap();
        fs::create_dir_all(outside.join("cairn-code/memory")).unwrap();
        fs::write(
            outside.join("cairn-code/memory/sentinel.md"),
            "outside secret",
        )
        .unwrap();

        let status = Command::new("cmd")
            .args(["/C", "mklink", "/J"])
            .arg(&junction)
            .arg(&outside)
            .status()
            .unwrap();
        assert!(status.success(), "failed to create test junction");

        assert!(open_memory_dir_at(&home, false).is_err());
        assert_eq!(
            fs::read_to_string(outside.join("cairn-code/memory/sentinel.md")).unwrap(),
            "outside secret"
        );

        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn test_save_replaces_hard_link_without_modifying_outside_file() {
        let base = temp_path("hard-link-test");
        let outside = base.join("outside.md");
        let memory_path = base.join(".config/cairn-code/memory/linked.md");
        fs::create_dir_all(&base).unwrap();
        let root = open_memory_dir_at(&base, true).unwrap();
        fs::write(&outside, "outside content").unwrap();
        fs::hard_link(&outside, &memory_path).unwrap();

        save_memory(&root, "linked", "new memory").unwrap();

        assert_eq!(fs::read_to_string(&outside).unwrap(), "outside content");
        assert!(fs::read_to_string(&memory_path)
            .unwrap()
            .contains("new memory"));
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn test_save_read_error_preserves_existing_file() {
        let base = temp_path("read-error-test");
        let memory_path = base.join(".config/cairn-code/memory/binary.md");
        fs::create_dir_all(&base).unwrap();
        let root = open_memory_dir_at(&base, true).unwrap();
        let original = [0xff, 0xfe, 0xfd];
        fs::write(&memory_path, original).unwrap();

        let error = save_memory(&root, "binary", "replacement").unwrap_err();

        assert!(
            error.contains("read existing memory"),
            "unexpected error: {error}"
        );
        assert_eq!(fs::read(&memory_path).unwrap(), original);
        let _ = fs::remove_dir_all(base);
    }

    #[test]
    fn test_input_schema_is_valid_json() {
        let tool = MemoryTool;
        let schema = tool.input_schema();
        let parsed = crate::json::parse(&schema);
        assert!(
            parsed.is_ok(),
            "Schema should be valid JSON: {:?}",
            parsed.err()
        );
        let obj = parsed.unwrap();
        let props = obj.get("properties").and_then(|v| v.as_object());
        assert!(props.is_some(), "Schema should have properties");
        assert!(
            props.unwrap().contains_key("action"),
            "Schema should have action property"
        );
    }

    #[test]
    fn test_execute_unknown_action() {
        let tool = MemoryTool;
        let result = tool.execute(r#"{"action":"invalid"}"#);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown action"));
    }

    #[test]
    fn test_execute_missing_action() {
        let tool = MemoryTool;
        let result = tool.execute(r#"{}"#);
        assert!(result.is_err());
    }

    #[test]
    fn test_execute_invalid_json() {
        let tool = MemoryTool;
        let result = tool.execute("not json");
        assert!(result.is_err());
    }
}
