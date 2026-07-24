//! Terminal text measurement, truncation, and sanitizing.

use super::*;

/// Claude Code-style short label for a completed think phase.
pub(crate) fn format_thought_label(elapsed: Option<Duration>) -> String {
    let Some(d) = elapsed else {
        return "Thought".into();
    };
    let secs = d.as_secs();
    if secs == 0 {
        // Sub-second thinks still get a readable marker.
        let ms = d.as_millis();
        if ms < 100 {
            return "Thought briefly".into();
        }
        return "Thought for <1s".into();
    }
    if secs < 60 {
        return format!("Thought for {secs}s");
    }
    let m = secs / 60;
    let s = secs % 60;
    if s == 0 {
        format!("Thought for {m}m")
    } else {
        format!("Thought for {m}m {s}s")
    }
}

/// Clean pasted text for the composer: drop CSI/OSC and other C0/C1 controls
/// while keeping Unicode (including emoji), newlines, and tabs.
pub(crate) fn sanitize_paste_for_composer(input: &str) -> String {
    // Reuse the shared control-sequence stripper; it already preserves \n/\t and
    // normal Unicode scalar values (emoji, CJK, combining marks).
    sanitize_terminal_output(input)
}

/// Strip terminal controls before writing untrusted text directly to a terminal.
///
/// This is intentionally used only at raw stdout boundaries. Ratatui should
/// continue receiving the original text so its normal rendering is unchanged.
pub fn sanitize_terminal_output(input: &str) -> String {
    #[derive(Clone, Copy)]
    enum State {
        Text,
        Escape,
        EscapeIntermediate,
        Csi,
        ControlString { osc: bool },
        ControlStringEscape { osc: bool },
    }

    let mut output = String::with_capacity(input.len());
    let mut state = State::Text;

    for character in input.chars() {
        state = match state {
            State::Text => match character {
                '\n' | '\t' => {
                    output.push(character);
                    State::Text
                }
                '\u{001b}' => State::Escape,
                '\u{009b}' => State::Csi,
                '\u{0090}' | '\u{0098}' | '\u{009d}' | '\u{009e}' | '\u{009f}' => {
                    State::ControlString {
                        osc: character == '\u{009d}',
                    }
                }
                '\u{0000}'..='\u{001f}' | '\u{007f}'..='\u{009f}' => State::Text,
                _ => {
                    output.push(character);
                    State::Text
                }
            },
            State::Escape => match character {
                '[' => State::Csi,
                ']' => State::ControlString { osc: true },
                'P' | 'X' | '^' | '_' => State::ControlString { osc: false },
                '\u{001b}' => State::Escape,
                '\u{009b}' => State::Csi,
                '\u{0090}' | '\u{0098}' | '\u{009d}' | '\u{009e}' | '\u{009f}' => {
                    State::ControlString {
                        osc: character == '\u{009d}',
                    }
                }
                '\n' | '\t' => {
                    output.push(character);
                    State::Text
                }
                '\u{0020}'..='\u{002f}' => State::EscapeIntermediate,
                '\u{0030}'..='\u{007e}' | '\u{0000}'..='\u{001f}' | '\u{007f}'..='\u{009f}' => {
                    State::Text
                }
                _ => {
                    output.push(character);
                    State::Text
                }
            },
            State::EscapeIntermediate => match character {
                '\u{0020}'..='\u{002f}' => State::EscapeIntermediate,
                '\u{0030}'..='\u{007e}' => State::Text,
                '\u{001b}' => State::Escape,
                '\n' | '\t' => {
                    output.push(character);
                    State::Text
                }
                '\u{0000}'..='\u{001f}' | '\u{007f}'..='\u{009f}' => State::Text,
                _ => {
                    output.push(character);
                    State::Text
                }
            },
            State::Csi => match character {
                '\u{0040}'..='\u{007e}' => State::Text,
                '\u{001b}' => State::Escape,
                '\u{009b}' => State::Csi,
                '\u{0090}' | '\u{0098}' | '\u{009d}' | '\u{009e}' | '\u{009f}' => {
                    State::ControlString {
                        osc: character == '\u{009d}',
                    }
                }
                '\u{009c}' => State::Text,
                _ => State::Csi,
            },
            State::ControlString { osc } => match character {
                '\u{009c}' => State::Text,
                '\u{0007}' if osc => State::Text,
                '\u{001b}' => State::ControlStringEscape { osc },
                _ => State::ControlString { osc },
            },
            State::ControlStringEscape { osc } => match character {
                '\\' | '\u{009c}' => State::Text,
                '\u{0007}' if osc => State::Text,
                '\u{001b}' => State::ControlStringEscape { osc },
                _ => State::ControlString { osc },
            },
        };
    }

    output
}

/// Code point ranges rendered two columns wide by terminals.
///
/// Ordered by first code point. Previously an if/else-if chain, which tripped
/// `manual_range_contains` and `if_same_then_else` once per arm — around 54 of
/// the crate's clippy warnings came from this one function.
const WIDE_RANGES: &[(u32, u32)] = &[
    (0x1100, 0x115F),
    (0x2329, 0x232A),
    (0x2600, 0x27BF), // misc symbols + dingbats
    (0x2E80, 0x303E),
    (0x3040, 0x3096),
    (0x3099, 0x30FF),
    (0x3105, 0x312F),
    (0x3131, 0x318E),
    (0x3190, 0x31E3),
    (0x31F0, 0x321E),
    (0x3220, 0x3247),
    (0x3250, 0x4DBF),
    (0x4E00, 0xA4CF),
    (0xA960, 0xA97C),
    (0xAC00, 0xD7A3),
    (0xF900, 0xFAFF),
    (0xFE10, 0xFE19),
    (0xFE30, 0xFE6F),
    (0xFF01, 0xFF60),
    (0xFFE0, 0xFFE6),
    (0x1B000, 0x1B0FF),
    (0x1B100, 0x1B12F),
    (0x1F000, 0x1F02F), // mahjong tiles
    (0x1F0A0, 0x1F0FF), // playing cards
    (0x1F100, 0x1F1FF), // enclosed alphanumerics / regional indicators
    (0x1F200, 0x1F2FF),
    (0x1F300, 0x1F9FF), // pictographs, emoticons, transport, supplemental
    (0x1FA00, 0x1FAFF), // chess symbols, pictographs extended-A
    (0x20000, 0x2FFFD),
    (0x30000, 0x3FFFD),
];

/// Code point ranges that occupy no columns: combining marks, variation
/// selectors, and the joiners used to build emoji sequences.
const ZERO_WIDTH_RANGES: &[(u32, u32)] = &[
    (0x0300, 0x036F),
    (0x1AB0, 0x1AFF),
    (0x1DC0, 0x1DFF),
    (0x200D, 0x200D), // zero-width joiner
    (0x20D0, 0x20FF),
    (0xFE00, 0xFE0F),
    (0xFEFF, 0xFEFF), // zero-width no-break space
    (0xE0100, 0xE01EF),
];

fn in_ranges(cp: u32, ranges: &[(u32, u32)]) -> bool {
    ranges.iter().any(|(lo, hi)| (*lo..=*hi).contains(&cp))
}

pub(super) fn char_width(c: char) -> usize {
    let cp = c as u32;
    if in_ranges(cp, ZERO_WIDTH_RANGES) {
        return 0;
    }
    if cp < 0x1100 {
        return 1;
    }
    if in_ranges(cp, WIDE_RANGES) {
        2
    } else {
        1
    }
}

pub(super) fn line_is_blank(line: &Line<'_>) -> bool {
    line.spans.is_empty() || line.spans.iter().all(|s| s.content.trim().is_empty())
}

pub(super) fn display_width(s: &str) -> usize {
    s.chars().map(char_width).sum()
}

/// Pad or truncate `s` to exactly `width` terminal columns (display width).
pub(super) fn pad_to_display_width(s: &str, width: usize) -> String {
    let dw = display_width(s);
    if dw < width {
        return format!("{}{}", s, " ".repeat(width - dw));
    }
    if dw == width {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut w_used = 0;
    for c in s.chars() {
        let cw = char_width(c);
        if w_used + cw > width {
            break;
        }
        out.push(c);
        w_used += cw;
    }
    if w_used < width {
        out.push_str(&" ".repeat(width - w_used));
    }
    out
}

pub(super) fn truncate_summary(summary: &str, max_chars: usize) -> String {
    if summary.chars().count() > max_chars {
        format!("{}…", summary.chars().take(max_chars).collect::<String>())
    } else {
        summary.to_string()
    }
}

/// Compact elapsed time for spinner / footer (Claude Code style: `3s`, `1m 12s`).
pub(crate) fn format_elapsed_compact(d: Duration) -> String {
    let secs = d.as_secs();
    if secs < 60 {
        format!("{secs}s")
    } else {
        let m = secs / 60;
        let s = secs % 60;
        format!("{m}m {s:02}s")
    }
}

pub(super) fn terminal_height() -> Option<usize> {
    std::env::var("LINES")
        .ok()
        .and_then(|v| v.parse().ok())
        .or(Some(24))
}

pub(super) fn format_timestamp(ts: u64) -> String {
    // `ts` is absolute unix seconds; show relative age.
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let secs = now.saturating_sub(ts);
    let days = secs / 86400;
    let hours = (secs % 86400) / 3600;
    let mins = (secs % 3600) / 60;
    if days > 0 {
        format!("{days}d {hours}h ago")
    } else if hours > 0 {
        format!("{hours}h {mins}m ago")
    } else if mins > 0 {
        format!("{mins}m ago")
    } else {
        "just now".into()
    }
}

pub(super) fn total_wrapped(lines: &[Line], width: usize) -> usize {
    let w = width.max(1);
    lines
        .iter()
        .map(|l| {
            let line_w: usize = l.spans.iter().map(|s| display_width(&s.content)).sum();
            if line_w == 0 {
                1
            } else {
                (line_w + w - 1) / w
            }
        })
        .sum()
}

/// Limit on-screen tool output by line count (head + tail) so long shell
/// dumps stay readable and keep the trailing summary / exit code.
pub(super) fn truncate_display(s: &str, head_lines: usize, tail_lines: usize) -> String {
    let lines: Vec<&str> = s.lines().collect();
    if lines.len() <= head_lines + tail_lines {
        return s.to_string();
    }
    let mut out = String::new();
    for line in &lines[..head_lines] {
        out.push_str(line);
        out.push('\n');
    }
    let omitted = lines.len() - head_lines - tail_lines;
    out.push_str(&format!("… ({omitted} lines omitted) …\n"));
    for line in &lines[lines.len() - tail_lines..] {
        out.push_str(line);
        out.push('\n');
    }
    while out.ends_with('\n') {
        out.pop();
    }
    out
}

#[cfg(test)]
mod terminal_output_tests {
    use super::sanitize_terminal_output;

    #[test]
    fn strips_osc_clipboard_title_and_hyperlink_payloads() {
        let input = concat!(
            "before",
            "\u{001b}]52;c;YXR0YWNrZXItY29udHJvbGxlZA==\u{0007}",
            "\u{001b}]0;forged title\u{001b}\\",
            "\u{001b}]8;;https://evil.example/\u{001b}\\link text\u{001b}]8;;\u{001b}\\",
            "after"
        );

        assert_eq!(sanitize_terminal_output(input), "beforelink textafter");
        assert_eq!(
            sanitize_terminal_output("safe\u{001b}]52;c;unterminated payload"),
            "safe"
        );
    }

    #[test]
    fn strips_csi_other_escape_sequences_and_their_payloads() {
        let input = concat!(
            "plain ",
            "\u{001b}[31mred\u{001b}[0m",
            "\u{009b}2J",
            " visible",
            "\u{001b}P1;2|dcs payload\u{001b}\\",
            " end"
        );

        assert_eq!(sanitize_terminal_output(input), "plain red visible end");
    }

    #[test]
    fn strips_c0_c1_controls_and_eight_bit_control_strings() {
        let input = "a\u{0000}b\u{0007}c\u{0008}d\r e\u{007f}f\u{0085}g\u{001b}";
        assert_eq!(sanitize_terminal_output(input), "abcd efg");
        assert_eq!(
            sanitize_terminal_output("left\u{009d}52;c;secret\u{009c}right"),
            "leftright"
        );
    }

    #[test]
    fn preserves_normal_unicode_newlines_and_tabs() {
        let input = "Grüße from 東京 🏔️\n\tcafé\n";
        assert_eq!(sanitize_terminal_output(input), input);
    }
}

#[cfg(test)]
mod summary_truncation_tests {
    use super::*;

    fn assert_unicode_boundaries(max_chars: usize) {
        for boundary_char in ['🙂', '界', 'é'] {
            let prefix = "a".repeat(max_chars - 1);
            let summary = format!("{prefix}{boundary_char}tail");
            assert_eq!(
                truncate_summary(&summary, max_chars),
                format!("{prefix}{boundary_char}…")
            );
        }
    }

    #[test]
    fn list_summary_truncates_unicode_at_60_characters() {
        assert_unicode_boundaries(60);
    }

    #[test]
    fn picker_summary_truncates_unicode_at_50_characters() {
        assert_unicode_boundaries(50);
    }
}
