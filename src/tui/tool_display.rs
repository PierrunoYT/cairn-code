//! Compact tool-call and permission-prompt formatting.

use super::*;

pub(super) fn permission_risk_warning(tool_name: &str) -> Option<&'static str> {
    match tool_name {
        "git" => Some(
            "Shell-equivalent risk: Git may execute aliases, hooks, helpers, and configured commands.",
        ),
        _ => None,
    }
}

/// Max display columns for a single permission-preview field value.
const PERM_VALUE_MAX_COLS: usize = 96;
/// Hard cap on preview rows so a multi-KB `file_edit` cannot fill the chrome.
const PERM_PREVIEW_MAX_LINES: usize = 8;

/// Collapse newlines and truncate so permission chrome stays readable.
fn truncate_perm_value(s: &str, max_cols: usize) -> String {
    let one_line = s
        .replace("\r\n", "\n")
        .replace('\n', "↵")
        .replace('\r', "↵");
    let dw = display_width(&one_line);
    if dw <= max_cols {
        return one_line;
    }
    // Leave room for the ellipsis glyph.
    let budget = max_cols.saturating_sub(1).max(1);
    let mut out = String::new();
    let mut used = 0;
    for c in one_line.chars() {
        let cw = char_width(c);
        if used + cw > budget {
            break;
        }
        out.push(c);
        used += cw;
    }
    out.push('…');
    out
}

/// Structured, truncated multi-line preview for the permission prompt.
/// Never dumps raw multi-KB JSON: that previously wrapped across the full
/// chrome and made the TUI look corrupted on large `file_edit` payloads.
pub(super) fn format_permission_tool_input(input: &str) -> Vec<String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    if let Ok(val) = crate::json::parse(trimmed) {
        if let Some(obj) = val.as_object() {
            let mut lines = Vec::new();
            // Prefer a stable field order for common tools.
            for key in [
                "file_path",
                "path",
                "command",
                "query",
                "url",
                "pattern",
                "old_string",
                "new_string",
                "content",
                "replace_all",
                "args",
            ] {
                if lines.len() >= PERM_PREVIEW_MAX_LINES {
                    break;
                }
                let Some(v) = obj.get(key) else {
                    continue;
                };
                let shown = if let Some(s) = v.as_str() {
                    truncate_perm_value(s, PERM_VALUE_MAX_COLS)
                } else if let Some(b) = v.as_bool() {
                    b.to_string()
                } else if let Some(n) = v.as_u64() {
                    n.to_string()
                } else if let Some(arr) = v.as_array() {
                    let joined = arr
                        .iter()
                        .filter_map(|x| x.as_str())
                        .map(|x| format!("{x:?}"))
                        .collect::<Vec<_>>()
                        .join(" ");
                    truncate_perm_value(&joined, PERM_VALUE_MAX_COLS)
                } else {
                    continue;
                };
                lines.push(format!("{key}: {shown}"));
            }
            if !lines.is_empty() {
                // Note remaining keys if we hit the line cap or skipped unknowns.
                let shown_keys: usize = [
                    "file_path",
                    "path",
                    "command",
                    "query",
                    "url",
                    "pattern",
                    "old_string",
                    "new_string",
                    "content",
                    "replace_all",
                    "args",
                ]
                .iter()
                .filter(|k| obj.get(**k).is_some())
                .count();
                if shown_keys > lines.len() {
                    let extra = shown_keys - lines.len();
                    lines.push(format!("… (+{extra} more field(s))"));
                }
                return lines;
            }
        }
    }

    // Non-JSON or unrecognized shape: single compact line, never the full blob.
    let hint = compact_tool_arg_hint(trimmed);
    if hint.is_empty() {
        Vec::new()
    } else {
        vec![hint]
    }
}

/// One-line arg preview for tool_use rows (avoid dumping pretty JSON).
pub(super) fn compact_tool_arg_hint(input: &str) -> String {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    // Prefer a few common fields when the arg is a JSON object.
    if let Ok(val) = crate::json::parse(trimmed) {
        if let Some(obj) = val.as_object() {
            for key in [
                "pattern",
                "file_path",
                "path",
                "command",
                "query",
                "url",
                "old_string",
                "args",
            ] {
                if let Some(v) = obj.get(key).and_then(|x| x.as_str()) {
                    let v = v.replace('\n', " ");
                    let shown: String = v.chars().take(64).collect();
                    let ellipsis = if v.chars().count() > 64 { "…" } else { "" };
                    return format!("{key}={shown}{ellipsis}");
                }
            }
            if let Some(args) = obj.get("args").and_then(|value| value.as_array()) {
                let args = args
                    .iter()
                    .filter_map(|value| value.as_str())
                    .map(|value| format!("{value:?}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                let shown: String = args.chars().take(64).collect();
                let ellipsis = if args.chars().count() > 64 { "…" } else { "" };
                return format!("args={shown}{ellipsis}");
            }
        }
    }
    let one_line = trimmed.replace('\n', " ");
    let shown: String = one_line.chars().take(72).collect();
    if one_line.chars().count() > 72 {
        format!("{shown}…")
    } else {
        shown
    }
}

/// Resolve display kind from stored name, or sniff content when name was lost
/// (older resumes used a generic `"tool"` label).
pub(super) fn infer_tool_display_kind<'a>(tool_name: &'a str, content: &str) -> &'a str {
    if tool_name != "tool" && !tool_name.is_empty() {
        return tool_name;
    }
    if content.contains("(showing lines") {
        return "file_read";
    }
    if content.contains("result(s)") || content.contains(" more (") && content.contains(" total)") {
        return "glob";
    }
    if content.contains("(exit code") {
        return "shell";
    }
    if content
        .lines()
        .take(5)
        .any(|l| l.contains(':') && !l.starts_with('{'))
        && content.lines().count() > 3
        && !content.contains("(showing lines")
    {
        // Heuristic: path:line:text style matches.
        if content
            .lines()
            .take(8)
            .filter(|l| l.matches(':').count() >= 2)
            .count()
            >= 2
        {
            return "grep";
        }
    }
    tool_name
}

/// Summary-first transcript body for a tool result. Full payload stays with the agent.
pub(super) fn compact_tool_result_display(kind: &str, content: &str) -> String {
    let content = content.trim();
    if content.is_empty() {
        return String::new();
    }
    match kind {
        "file_read" => {
            if let Some(summary) = content.lines().rev().find(|l| l.contains("(showing lines")) {
                return summary.to_string();
            }
            let n = content.lines().filter(|l| !l.trim().is_empty()).count();
            format!("({n} lines read)")
        }
        "glob" => {
            if content.contains("No matches") {
                return "No matches found.".into();
            }
            if let Some(summary) = content.lines().rev().find(|l| {
                let t = l.trim();
                t.contains("result(s)") || t.contains(" total)") || t.starts_with('…')
            }) {
                // Pull a clean count when the summary is "… and N more (M total)"
                // or "M result(s)".
                let s = summary.trim();
                if let Some(rest) = s.strip_suffix(" result(s)") {
                    if rest.chars().all(|c| c.is_ascii_digit()) {
                        return format!("{rest} matches");
                    }
                }
                if let Some(i) = s.rfind('(') {
                    if let Some(j) = s.rfind(" total)") {
                        if j > i {
                            let n = s[i + 1..j].trim();
                            if n.chars().all(|c| c.is_ascii_digit()) {
                                return format!("{n} matches");
                            }
                        }
                    }
                }
                return s.to_string();
            }
            let n = content.lines().filter(|l| !l.trim().is_empty()).count();
            format!("{n} matches")
        }
        "grep" => {
            let hits: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();
            if hits.is_empty() {
                return "No matches.".into();
            }
            if hits.len() == 1 {
                return hits[0].chars().take(100).collect();
            }
            format!("{} matches", hits.len())
        }
        // Keep a little shell context so test summaries / exit codes remain visible.
        "shell" => truncate_display(content, 2, 3),
        _ => {
            let n = content.lines().filter(|l| !l.trim().is_empty()).count();
            if n <= 2 {
                return content.to_string();
            }
            // One-line summary for unknown tools.
            let first = content.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
            let first: String = first.chars().take(80).collect();
            format!("{first}… ({n} lines)")
        }
    }
}

#[cfg(test)]
mod tool_display_tests {
    use super::*;

    #[test]
    fn compact_file_read_is_summary_only() {
        let mut body = String::new();
        for i in 151..=188 {
            body.push_str(&format!("{i}:line {i}\n"));
        }
        body.push_str("\nREADME.md:151 (showing lines 151-188 of 188)");
        let out = compact_tool_result_display("file_read", &body);
        assert_eq!(out, "README.md:151 (showing lines 151-188 of 188)");
        assert_eq!(out.lines().count(), 1);
    }

    #[test]
    fn compact_glob_is_match_count() {
        let mut body = String::new();
        for i in 0..15 {
            body.push_str(&format!("src/f{i}.rs\n"));
        }
        body.push_str("… and 24 more (39 total)");
        let out = compact_tool_result_display("glob", &body);
        assert_eq!(out, "39 matches");
    }

    #[test]
    fn infer_kind_from_content_when_name_lost() {
        let body = "1:x\n\nfoo.rs:1 (showing lines 1-1 of 10)";
        assert_eq!(infer_tool_display_kind("tool", body), "file_read");
        assert_eq!(
            infer_tool_display_kind("tool", "a.rs\nb.rs\n2 result(s)"),
            "glob"
        );
    }

    #[test]
    fn compact_tool_arg_hint_extracts_pattern() {
        let h = compact_tool_arg_hint(r#"{"pattern":"src/**/*.rs"}"#);
        assert!(h.contains("pattern=src/**/*.rs"), "{h}");
    }

    #[test]
    fn compact_tool_arg_hint_preserves_array_boundaries() {
        let hint = compact_tool_arg_hint(r#"{"args":["status","path with spaces",""]}"#);
        assert_eq!(hint, r#"args="status" "path with spaces" """#);
    }

    #[test]
    fn git_permission_warning_classifies_shell_equivalent_risk() {
        let warning = permission_risk_warning("git").unwrap();
        assert!(warning.contains("Shell-equivalent risk"));
        assert!(permission_risk_warning("go").is_none());
    }

    #[test]
    fn permission_preview_shows_structured_fields() {
        let input = r#"{"file_path":"src/tools/file_edit.rs","old_string":"a","new_string":"b","replace_all":true}"#;
        let lines = format_permission_tool_input(input);
        assert_eq!(
            lines,
            vec![
                "file_path: src/tools/file_edit.rs".to_string(),
                "old_string: a".to_string(),
                "new_string: b".to_string(),
                "replace_all: true".to_string(),
            ]
        );
    }

    #[test]
    fn permission_preview_truncates_large_file_edit_payload() {
        let big = "x".repeat(400);
        let input = format!(
            r#"{{"file_path":"src/tools/file_edit.rs","old_string":"keep","new_string":"{big}"}}"#
        );
        let lines = format_permission_tool_input(&input);
        assert!(
            lines.iter().any(|l| l.starts_with("file_path:")),
            "{lines:?}"
        );
        let new_line = lines
            .iter()
            .find(|l| l.starts_with("new_string:"))
            .expect("new_string row");
        assert!(
            new_line.ends_with('…'),
            "expected truncated new_string, got {new_line}"
        );
        assert!(
            display_width(new_line) <= "new_string: ".len() + PERM_VALUE_MAX_COLS + 2,
            "preview line too wide: {} cols ({new_line})",
            display_width(new_line)
        );
        // Full payload must never appear as a single raw JSON dump.
        let joined = lines.join("\n");
        assert!(
            !joined.contains(&big),
            "raw multi-KB value leaked into chrome"
        );
        assert!(lines.len() <= PERM_PREVIEW_MAX_LINES + 1);
    }

    #[test]
    fn permission_preview_collapses_newlines_in_values() {
        let input = r#"{"file_path":"a.rs","old_string":"line1\nline2","new_string":"x"}"#;
        let lines = format_permission_tool_input(input);
        let old = lines
            .iter()
            .find(|l| l.starts_with("old_string:"))
            .expect("old_string");
        assert!(old.contains('↵'), "{old}");
        assert!(!old.contains('\n'), "{old}");
    }
}
