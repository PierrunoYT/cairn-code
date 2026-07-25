//! Frame rendering and stream flushing.

use super::*;

impl Tui {
    pub(super) fn render(&mut self, f: &mut Frame) {
        let area = f.area();
        let dim = self.theme.muted;
        let bright = self.theme.accent;
        let bold_dim = self.theme.faintest;
        let orange = self.theme.accent;
        let white = self.theme.ink;
        let red = self.theme.red;
        let green = self.theme.green;
        let orange_fg = self.theme.accent_fg;
        let selected = self.theme.selected;

        let mut lines: Vec<Line> = Vec::new();

        // Welcome box (Claude Code style): full terminal width rounded frame.
        // Inner content width is terminal cols minus the two border glyphs.
        // Emoji (e.g. 🪨) must use display_width=2 or the right border drops off.
        let pw = (area.width as usize).saturating_sub(2).max(1);
        let pad = |s: &str| pad_to_display_width(s, pw);
        let box_style = orange;
        let box_row = |inner: String, style: Style| {
            Line::from(vec![
                Span::styled("│", box_style),
                Span::styled(inner, style),
                Span::styled("│", box_style),
            ])
        };

        lines.push(Line::from(Span::styled(
            format!("╭{}╮", "─".repeat(pw)),
            box_style,
        )));
        lines.push(box_row(
            pad(&format!("  🪨 Cairn Code v{}", self.version)),
            bright,
        ));
        lines.push(box_row(pad("  open terminal coding agent  ·  /help"), dim));
        lines.push(Line::from(Span::styled(
            format!("├{}┤", "─".repeat(pw)),
            box_style,
        )));
        lines.push(box_row(
            pad(&format!("  Model   {} / {}", self.provider, self.model)),
            dim,
        ));
        lines.push(box_row(pad(&format!("  Path    {}", self.work_dir)), dim));
        lines.push(Line::from(Span::styled(
            format!("╰{}╯", "─".repeat(pw)),
            box_style,
        )));
        lines.push(Line::from(""));

        // Output
        for line in &self.output_lines {
            match line.type_.as_str() {
                "user" => {
                    // Claude Code: past user turns use a quieter marker than the live ❯.
                    lines.push(Line::from(vec![
                        Span::styled("> ", orange),
                        Span::styled(line.content.as_str(), white),
                    ]));
                    lines.push(Line::from(""));
                }
                "text" => {
                    // Interim assistant narration (plans / status between tools):
                    // blank line above and below + accent colour so it separates
                    // from dense tool chrome without looking like user prompts.
                    if lines.last().is_none_or(|l| !line_is_blank(l)) {
                        lines.push(Line::from(""));
                    }
                    lines.extend(crate::markdown::render_with_body(
                        &line.content,
                        &self.theme,
                        orange_fg,
                    ));
                    if lines.last().is_none_or(|l| !line_is_blank(l)) {
                        lines.push(Line::from(""));
                    }
                }
                "tool_use" => {
                    // One line: name + short arg hint (no multi-line JSON dump).
                    let hint = compact_tool_arg_hint(&line.content);
                    let label = if hint.is_empty() {
                        line.tool_name.clone()
                    } else {
                        format!("{}  {}", line.tool_name, hint)
                    };
                    lines.push(Line::from(vec![
                        Span::styled("● ", white),
                        Span::styled(label, dim),
                    ]));
                }
                "tool_result" => {
                    let is_err = line.content.starts_with("Error:")
                        || line.content.contains("exit code")
                            && !line.content.contains("(exit code 0)");
                    let color = if is_err { red } else { green };
                    let dur = if line.duration.is_empty() {
                        String::new()
                    } else {
                        format!(" ({})", line.duration)
                    };
                    // Summary-first body. Agent still has the full tool payload.
                    let kind = infer_tool_display_kind(&line.tool_name, &line.content);
                    let display = compact_tool_result_display(kind, &line.content);
                    let name = if line.tool_name == "tool" || line.tool_name.is_empty() {
                        kind
                    } else {
                        line.tool_name.as_str()
                    };
                    let header = if display.lines().count() <= 1 && !display.is_empty() {
                        // Fold single-line summaries onto the status row.
                        format!("{name}{dur}  {display}")
                    } else {
                        format!("{name}{dur}")
                    };
                    lines.push(Line::from(vec![
                        Span::styled("● ", color),
                        Span::styled(header, dim),
                    ]));
                    if display.lines().count() > 1 {
                        for part in display.split('\n') {
                            if part.is_empty() {
                                continue;
                            }
                            lines.push(Line::from(vec![Span::styled(format!("  {part}"), dim)]));
                        }
                    }
                }
                "error" => {
                    for (i, part) in line.content.split('\n').enumerate() {
                        if i == 0 {
                            lines.push(Line::from(vec![Span::styled(format!("● {part}"), red)]));
                        } else {
                            lines.push(Line::from(vec![Span::styled(format!("  {part}"), red)]));
                        }
                    }
                }
                "system" => {
                    for part in line.content.split('\n') {
                        lines.push(Line::from(vec![Span::styled(part, dim)]));
                    }
                }
                "thinking" => {
                    // Full preserved thinking (only written when show_thinking is on).
                    lines.push(Line::from(vec![Span::styled("── Thinking ──", bold_dim)]));
                    for part in line.content.split('\n') {
                        lines.push(Line::from(vec![Span::styled(part, dim)]));
                    }
                }
                "thinking_summary" => {
                    // Claude Code default: short marker, no body.
                    let label = if line.content.is_empty() {
                        "Thought".to_string()
                    } else {
                        line.content.clone()
                    };
                    lines.push(Line::from(vec![Span::styled(format!("✦ {label}"), dim)]));
                }
                _ => {
                    for part in line.content.split('\n') {
                        lines.push(Line::from(Span::raw(part)));
                    }
                }
            }
        }

        // Streaming thinking: full body only when toggled on; off-mode uses spinner only.
        if self.show_thinking && !self.stream_thinking.is_empty() {
            lines.push(Line::from(vec![Span::styled("── Thinking ──", bold_dim)]));
            for part in self.stream_thinking.split('\n') {
                lines.push(Line::from(vec![Span::styled(part, dim)]));
            }
        }
        if !self.streaming_text.is_empty() {
            // Match finished assistant text: accent body + breathing room.
            if lines.last().is_none_or(|l| !line_is_blank(l)) {
                lines.push(Line::from(""));
            }
            lines.extend(crate::markdown::render_with_body(
                &self.streaming_text,
                &self.theme,
                orange_fg,
            ));
        }
        // Spinner while waiting / thinking without answer text. Skip when full
        // thinking body is already on screen (show_thinking on).
        // OpenClaude-style: glyph + rotating verb + elapsed seconds.
        let show_spin = matches!(self.state, State::Running)
            && self.streaming_text.is_empty()
            && !(self.show_thinking && !self.stream_thinking.is_empty());
        if show_spin {
            let spin = SPINNER_CHARS[self.spinner_idx % SPINNER_CHARS.len()];
            let verb = SPINNER_VERBS[self.spinner_verb_idx % SPINNER_VERBS.len()];
            let elapsed = self
                .running_started
                .map(|t| format_elapsed_compact(t.elapsed()))
                .unwrap_or_default();
            let mut spin_spans = vec![
                Span::styled(spin, orange),
                Span::styled(format!(" {verb}…"), dim),
            ];
            if !elapsed.is_empty() {
                spin_spans.push(Span::styled(format!(" {elapsed}"), bold_dim));
            }
            lines.push(Line::from(spin_spans));
        }

        // Composer / pickers live in a fixed bottom chrome region so typing never
        // steals viewport rows from the transcript above (single-scroll layout used to
        // push the last LLM line off-screen as soon as the prompt grew).
        let mut chrome: Vec<Line> = Vec::new();
        // Cursor: (x offset within chrome width, logical line index in chrome).
        let mut cursor_pos: Option<(u16, usize)> = None;

        if let Some(name) = &self.confirm_remove_provider {
            chrome.push(Line::from(vec![Span::styled(
                format!("Remove saved API key for '{name}'?"),
                white,
            )]));
            chrome.push(Line::from(vec![Span::styled(
                "This only deletes the key from the config file.",
                dim,
            )]));
            chrome.push(Line::from(""));
            let options = ["Cancel", "Remove"];
            let mut option_spans = Vec::new();
            for (i, opt) in options.iter().enumerate() {
                if i > 0 {
                    option_spans.push(Span::raw("  "));
                }
                let is_sel = i == self.confirm_remove_sel;
                let open = if is_sel { "[" } else { " " };
                let close = if is_sel { "]" } else { " " };
                option_spans.push(Span::styled(
                    format!("{open}{opt}{close}"),
                    if is_sel {
                        orange_fg.add_modifier(Modifier::BOLD)
                    } else {
                        dim
                    },
                ));
            }
            chrome.push(Line::from(option_spans));
            chrome.push(Line::from(vec![Span::styled(
                "(← → navigate  Enter confirm  Esc cancel)",
                dim,
            )]));
        } else if let Some(name) = &self.confirm_history_provider {
            chrome.push(Line::from(vec![Span::styled(
                format!("Send existing conversation to '{name}'?"),
                white,
            )]));
            chrome.push(Line::from(vec![
                Span::styled(
                    "Existing prompts, source excerpts, and tool results will be sent to this provider.",
                    dim,
                ),
            ]));
            chrome.push(Line::from(""));
            let options = ["Cancel", "Continue"];
            let mut option_spans = Vec::new();
            for (i, opt) in options.iter().enumerate() {
                if i > 0 {
                    option_spans.push(Span::raw("  "));
                }
                let is_sel = i == self.confirm_history_sel;
                let open = if is_sel { "[" } else { " " };
                let close = if is_sel { "]" } else { " " };
                option_spans.push(Span::styled(
                    format!("{open}{opt}{close}"),
                    if is_sel {
                        orange_fg.add_modifier(Modifier::BOLD)
                    } else {
                        dim
                    },
                ));
            }
            chrome.push(Line::from(option_spans));
            chrome.push(Line::from(vec![Span::styled(
                "(← → navigate  Enter confirm  Esc cancel)",
                dim,
            )]));
        } else if self.show_help {
            chrome.push(Line::from(vec![
                Span::styled("── Help ", orange),
                Span::styled("(Esc or Enter close) ──", bold_dim),
            ]));
            for (keys, desc) in HELP_ROWS {
                if keys.is_empty() && desc.is_empty() {
                    chrome.push(Line::from(""));
                    continue;
                }
                if desc.is_empty() {
                    // Section header
                    chrome.push(Line::from(vec![Span::styled(
                        format!("  {keys}"),
                        orange_fg.add_modifier(Modifier::BOLD),
                    )]));
                    continue;
                }
                chrome.push(Line::from(vec![
                    Span::styled(format!("  {keys:<28}"), orange_fg),
                    Span::styled(*desc, dim),
                ]));
            }
        } else if self.show_shortcuts {
            // Matches the footer hint "? for shortcuts" on an empty idle prompt.
            chrome.push(Line::from(vec![
                Span::styled("── Shortcuts ", orange),
                Span::styled("(? or Esc close) ──", bold_dim),
            ]));
            for (keys, desc) in SHORTCUT_ROWS {
                chrome.push(Line::from(vec![
                    Span::styled(format!("  {keys:<22}"), orange_fg),
                    Span::styled(*desc, dim),
                ]));
            }
            chrome.push(Line::from(""));
            chrome.push(Line::from(vec![
                Span::styled("  /help", orange_fg),
                Span::styled("                  slash commands and more", dim),
            ]));
        } else if self.show_provider_picker {
            chrome.push(Line::from(vec![
                Span::styled("── Provider ", orange),
                Span::styled(
                    "(↑↓ navigate  Enter select  Del remove key  Esc cancel) ──",
                    bold_dim,
                ),
            ]));
            for (i, name) in self.provider_picker_list.iter().enumerate() {
                let is_sel = i == self.provider_picker_sel;
                let is_cur = *name == self.provider;
                let cur_mark = if is_cur { "  (current)" } else { "" };
                let key_mark = if self.provider_picker_keys.get(i).copied().unwrap_or(false) {
                    "  [signed in]"
                } else if crate::oauth::supports_oauth(name) {
                    "  [browser login]"
                } else {
                    ""
                };
                let prefix = if is_sel { "▸ " } else { "  " };
                chrome.push(Line::from(vec![Span::styled(
                    format!("{prefix}{name}{key_mark}{cur_mark}"),
                    if is_sel { selected } else { dim },
                )]));
            }
        } else if self.show_model_picker {
            let visible = self.picker_visible_height();
            let end = (self.picker_scrl + visible).min(self.picker_models.len());
            let num = self.picker_models.len();

            chrome.push(Line::from(vec![
                Span::styled("── Model ", orange),
                Span::styled("(↑↓ navigate  Enter select  Esc cancel) ──", bold_dim),
            ]));
            if num > visible {
                chrome.push(Line::from(vec![Span::styled(
                    format!("  … {}/{}  ↑↓ scroll", self.picker_sel + 1, num),
                    dim,
                )]));
            }
            for i in self.picker_scrl..end {
                let m = &self.picker_models[i];
                let is_sel = i == self.picker_sel;
                let is_cur = m.id == self.model;
                let ctx = if m.max_ctx > 0 {
                    format!(" ({}K context)", m.max_ctx / 1000)
                } else {
                    String::new()
                };
                let check = if is_cur { "  ✓" } else { "" };
                let prefix = if is_sel { "▸ " } else { "  " };
                chrome.push(Line::from(vec![Span::styled(
                    format!("{prefix}{}  {}{ctx}{check}", m.name, m.id),
                    if is_sel { selected } else { dim },
                )]));
            }
        } else if self.show_theme_picker {
            chrome.push(Line::from(vec![
                Span::styled("── Theme ", orange),
                Span::styled("(↑↓ live-preview  Enter apply  Esc cancel) ──", bold_dim),
            ]));
            for (i, t) in self.theme_picker_list.iter().enumerate() {
                let is_sel = i == self.theme_picker_sel;
                let is_cur = t.name == self.theme.name;
                let cur_mark = if is_cur { "  ✓" } else { "" };
                let prefix = if is_sel { "▸ " } else { "  " };
                chrome.push(Line::from(vec![Span::styled(
                    format!("{prefix}{} ({}){cur_mark}", t.label, t.name),
                    if is_sel { selected } else { dim },
                )]));
            }
        } else if self.show_session_picker {
            let visible = 10usize;
            let end = (self.picker_session_scrl + visible).min(self.picker_sessions.len());
            let num = self.picker_sessions.len();

            let title = if self.session_picker_delete {
                "── Delete Session "
            } else {
                "── Resume Session "
            };
            let hint = if self.session_picker_delete {
                "(↑↓ navigate  Enter delete  Esc cancel) ──"
            } else {
                "(↑↓ navigate  Enter select  Esc cancel) ──"
            };
            chrome.push(Line::from(vec![
                Span::styled(title, orange),
                Span::styled(hint, bold_dim),
            ]));
            if num > visible {
                chrome.push(Line::from(vec![Span::styled(
                    format!("  … {}/{}  ↑↓ scroll", self.picker_session_sel + 1, num),
                    dim,
                )]));
            }
            for i in self.picker_session_scrl..end {
                let s = &self.picker_sessions[i];
                let is_sel = i == self.picker_session_sel;
                let prefix = if is_sel { "▸ " } else { "  " };
                let summary = truncate_summary(&s.summary, 50);
                let time_str = format_timestamp(s.updated_at);
                chrome.push(Line::from(vec![Span::styled(
                    format!(
                        "{prefix}{}  {}  {} msgs  {time_str}",
                        s.id.get(..8).unwrap_or(s.id.as_str()),
                        s.model,
                        s.msg_count
                    ),
                    if is_sel { selected } else { dim },
                )]));
                if !summary.is_empty() && is_sel {
                    chrome.push(Line::from(vec![Span::styled(
                        format!("   {summary}"),
                        if is_sel { selected } else { dim },
                    )]));
                }
            }
        } else if self.show_permission_prompt {
            chrome.push(Line::from(vec![Span::styled(
                format!("Tool '{}' wants to run:", self.perm_tool_name),
                white,
            )]));
            if let Some(warning) = permission_risk_warning(&self.perm_tool_name) {
                chrome.push(Line::from(vec![Span::styled(
                    format!("  {warning}"),
                    orange_fg,
                )]));
            }
            for preview in format_permission_tool_input(&self.perm_tool_input) {
                chrome.push(Line::from(vec![Span::styled(format!("  {preview}"), dim)]));
            }
            chrome.push(Line::from(""));
            // Claude Code / OpenClaude style: numbered vertical options.
            // Only option 4 (Discuss) has an inline feedback field.
            let options = ["1. Yes", "2. Yes, always allow", "3. No", "4. Discuss"];
            let mut discuss_cursor_pos: Option<(u16, usize)> = None;
            for (i, opt) in options.iter().enumerate() {
                let is_sel = i == self.perm_selection;
                let prefix = if is_sel { "❯ " } else { "  " };
                chrome.push(Line::from(vec![Span::styled(
                    format!("{prefix}{opt}"),
                    if is_sel {
                        orange_fg.add_modifier(Modifier::BOLD)
                    } else {
                        dim
                    },
                )]));
                // Inline field appears only while Discuss is selected.
                if i == 3 && is_sel {
                    let cursor = self.perm_discuss_cursor.min(self.perm_discuss_buf.len());
                    let (before, after) = self.perm_discuss_buf.split_at(cursor);
                    let label = "     ";
                    let prompt_line_idx = chrome.len();
                    let mut spans = vec![Span::styled(label, dim)];
                    if before.is_empty() && after.is_empty() {
                        spans.push(Span::styled("▋", orange_fg));
                        spans.push(Span::styled(" optional feedback · Enter to send", bold_dim));
                    } else {
                        spans.push(Span::styled(before, white));
                        spans.push(Span::styled("▋", orange_fg));
                        spans.push(Span::styled(after, white));
                    }
                    chrome.push(Line::from(spans));
                    discuss_cursor_pos = Some((
                        display_width(label) as u16 + display_width(before) as u16,
                        prompt_line_idx,
                    ));
                }
            }
            chrome.push(Line::from(""));
            chrome.push(Line::from(vec![Span::styled(
                "(1 yes  ·  2 always  ·  3 no  ·  4 discuss  ·  Esc cancel)",
                dim,
            )]));
            cursor_pos = discuss_cursor_pos.or(Some((0, 0)));
        } else if self.show_recovery_prompt {
            chrome.push(Line::from(vec![Span::styled(
                format!(
                    "LLM failed ({}/{}). Switch and retry your request:",
                    self.provider, self.model
                ),
                white,
            )]));
            chrome.push(Line::from(""));
            let options = ["Switch model (m)", "Switch provider (p)", "Dismiss (d)"];
            let mut option_spans = Vec::new();
            for (i, opt) in options.iter().enumerate() {
                if i > 0 {
                    option_spans.push(Span::raw("  "));
                }
                let is_sel = i == self.recovery_selection;
                let open = if is_sel { "[" } else { " " };
                let close = if is_sel { "]" } else { " " };
                option_spans.push(Span::styled(
                    format!("{open}{opt}{close}"),
                    if is_sel {
                        orange_fg.add_modifier(Modifier::BOLD)
                    } else {
                        dim
                    },
                ));
            }
            chrome.push(Line::from(option_spans));
            chrome.push(Line::from(vec![Span::styled(
                "(← → navigate  Enter confirm  Esc dismiss)",
                dim,
            )]));
            let cursor = self.cursor.min(self.input_buf.len());
            let (before, after) = self.input_buf.split_at(cursor);
            let prompt_line_idx = chrome.len();
            chrome.push(Line::from(vec![
                Span::styled("❯ ", orange_fg),
                Span::raw(before),
                Span::styled("▋", orange_fg),
                Span::raw(after),
            ]));
            cursor_pos = Some((
                display_width("❯ ") as u16 + display_width(before) as u16,
                prompt_line_idx,
            ));
        } else if self.awaiting_api_key {
            let target = self.api_key_target.as_deref().unwrap_or(&self.provider);
            let env_hint = crate::config::env_var_name(target).unwrap_or("API_KEY");
            let label = format!("{target} API key ({env_hint}) > ");
            let cursor_chars = self.input_buf[..self.cursor.min(self.input_buf.len())]
                .chars()
                .count();
            let masked = crate::config::mask_secret_display(&self.input_buf, 4);
            let masked_chars: Vec<char> = masked.chars().collect();
            let before: String = masked_chars.iter().take(cursor_chars).collect();
            let after: String = masked_chars.iter().skip(cursor_chars).collect();
            chrome.push(Line::from(vec![
                Span::styled(format!("{label}{before}"), orange_fg),
                Span::styled("▋", orange_fg),
                Span::styled(after, orange_fg),
            ]));
            chrome.push(Line::from(vec![Span::styled(
                "Hidden as you type (last 4 characters shown). Enter to save  ·  Esc to cancel",
                dim,
            )]));
            cursor_pos = Some((
                display_width(&label) as u16 + display_width(&before) as u16,
                0,
            ));
        } else {
            // Normal composer is drawn later with a ratatui Block (reliable full-width
            // borders). Here we only build what sits under it: the command list
            // while a `/…` is being typed, then the status byline.
            if !self.cmd_picker_filtered.is_empty() {
                const VISIBLE: usize = 8;
                let total = self.cmd_picker_filtered.len();
                let sel = self.cmd_picker_sel.min(total - 1);
                // Scroll only once the selection walks past the window.
                let start = sel.saturating_sub(VISIBLE - 1);
                let end = (start + VISIBLE).min(total);
                let rows = &self.cmd_picker_filtered[start..end];

                chrome.push(Line::from(vec![
                    Span::styled("── Commands ", orange),
                    Span::styled("(↑↓ select  Tab complete  Esc dismiss) ──", bold_dim),
                ]));
                if total > VISIBLE {
                    chrome.push(Line::from(vec![Span::styled(
                        format!("  … {}/{}  ↑↓ scroll", sel + 1, total),
                        dim,
                    )]));
                }
                let name_w = rows.iter().map(|c| display_width(c)).max().unwrap_or(0);
                for (i, cmd) in rows.iter().enumerate() {
                    let is_sel = start + i == sel;
                    let prefix = if is_sel { "▸ " } else { "  " };
                    let gap = " ".repeat(name_w.saturating_sub(display_width(cmd)));
                    let mut row = vec![Span::styled(
                        format!("{prefix}{cmd}{gap}"),
                        if is_sel { selected } else { dim },
                    )];
                    if let Some(help) = slash_completion_help(cmd) {
                        row.push(Span::styled(format!("  {help}"), bold_dim));
                    }
                    chrome.push(Line::from(row));
                }
            }

            let mut status = Vec::new();
            if self.ctrl_c_exit_armed {
                status.push(Span::styled(
                    "Press Ctrl+C again to exit",
                    orange_fg.add_modifier(Modifier::BOLD),
                ));
            } else if matches!(self.state, State::Running) {
                status.push(Span::styled("esc to interrupt", bold_dim));
                if let Some(started) = self.running_started {
                    status.push(Span::styled(" · ", bold_dim));
                    status.push(Span::styled(format_elapsed_compact(started.elapsed()), dim));
                }
                status.push(Span::styled(" · ", bold_dim));
                status.push(Span::styled(
                    format!("{}/{}", self.provider, self.model),
                    dim,
                ));
            } else {
                if !self.pending_images.is_empty() {
                    let n = self.pending_images.len();
                    status.push(Span::styled(
                        if n == 1 {
                            "1 image attached".to_string()
                        } else {
                            format!("{n} images attached")
                        },
                        orange_fg.add_modifier(Modifier::BOLD),
                    ));
                    status.push(Span::styled(" · ", bold_dim));
                }
                status.push(Span::styled(
                    format!("{}/{}", self.provider, self.model),
                    dim,
                ));
                let path = self.work_dir.as_str();
                let short_path = path
                    .rsplit(['/', '\\'])
                    .next()
                    .filter(|s| !s.is_empty())
                    .unwrap_or(path);
                status.push(Span::styled(" · ", bold_dim));
                status.push(Span::styled(short_path, dim));
                if self.total_usage.input_tokens > 0 || self.total_usage.output_tokens > 0 {
                    let est = crate::cost::estimate_cost(&self.model, &self.total_usage);
                    let cost_str = crate::cost::format_cost(est);
                    status.push(Span::styled(" · ", bold_dim));
                    status.push(Span::styled(
                        format!(
                            "{}↓ {}↑",
                            self.total_usage.input_tokens, self.total_usage.output_tokens
                        ),
                        dim,
                    ));
                    if est > 0.0 {
                        status.push(Span::styled(" · ", bold_dim));
                        status.push(Span::styled(cost_str, dim));
                    }
                }
                if self.input_buf.is_empty() {
                    status.push(Span::styled(" · ", bold_dim));
                    status.push(Span::styled("? for shortcuts", bold_dim));
                }
            }
            chrome.push(Line::from(status));
            // Signal: paint Block composer instead of line-drawn box in chrome.
            cursor_pos = Some((u16::MAX, usize::MAX));
        }

        let width = area.width as usize;
        let body_wrapped = total_wrapped(&lines, width);
        // Normal composer uses a separate 3-row Block + status line in chrome.
        let use_block_composer = cursor_pos == Some((u16::MAX, usize::MAX));
        let status_h = if use_block_composer {
            total_wrapped(&chrome, width).max(1) as u16
        } else {
            0
        };
        let chrome_wrapped = if use_block_composer {
            // status only (composer is separate)
            status_h as usize
        } else {
            total_wrapped(&chrome, width).max(1)
        };
        // Keep room for transcript; cap chrome so pickers cannot hide all output.
        let composer_h: u16 = if use_block_composer { 3 } else { 0 };
        let max_chrome = (area.height as usize)
            .saturating_sub(3)
            .min((area.height as usize).saturating_mul(2) / 3)
            .max(1);
        let chrome_h = if use_block_composer {
            status_h.min(max_chrome as u16).max(1)
        } else {
            chrome_wrapped.min(max_chrome) as u16
        };
        let chrome_scroll = if use_block_composer {
            0
        } else {
            chrome_wrapped.saturating_sub(chrome_h as usize)
        };

        // Claude Code style: pin composer/status to the bottom.
        let (body_area, composer_area, chrome_area) = if use_block_composer {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Min(1),
                    Constraint::Length(composer_h),
                    Constraint::Length(chrome_h),
                ])
                .split(area);
            (chunks[0], Some(chunks[1]), chunks[2])
        } else {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(chrome_h)])
                .split(area);
            (chunks[0], None, chunks[1])
        };

        // videre-style rowoff: free scroll when not following; pin to bottom when following.
        let body_h = body_area.height as usize;
        let max_off = body_wrapped.saturating_sub(body_h.max(1));
        self.last_body_h = body_h;
        self.last_body_wrapped = body_wrapped;
        let body_scroll = if self.transcript_follow {
            self.transcript_rowoff = max_off;
            max_off
        } else {
            let off = self.transcript_rowoff.min(max_off);
            self.transcript_rowoff = off;
            if off >= max_off {
                self.transcript_follow = true;
            }
            off
        };

        f.render_widget(
            Paragraph::new(Text::from(lines))
                .wrap(Wrap { trim: false })
                .scroll((body_scroll as u16, 0)),
            body_area,
        );

        if let Some(composer_area) = composer_area {
            // Reliable full-width rounded box via Block (avoids manual │ padding wrap bugs).
            let cursor = self.cursor.min(self.input_buf.len());
            let (before, after) = self.input_buf.split_at(cursor);
            // ASCII ">" is always width-1; ❯ can be ambiguous across fonts/terminals.
            let mark = "> ";
            let show_idle_hint = self.input_buf.is_empty()
                && matches!(self.state, State::Idle)
                && self.idle_suggestion.is_some()
                && !self.show_permission_prompt
                && !self.show_recovery_prompt
                && !self.awaiting_api_key;

            let mut spans = vec![Span::styled(mark, orange_fg)];
            if show_idle_hint {
                spans.push(Span::styled(
                    self.idle_suggestion.as_deref().unwrap_or(""),
                    bold_dim,
                ));
            } else {
                // Typed text uses ink (bright), not muted/dim like suggestions.
                spans.push(Span::styled(before, white));
                spans.push(Span::styled(after, white));
                if cursor >= self.input_buf.len() {
                    if let Some(cmd) = self.selected_slash_completion() {
                        if let Some(suffix) = slash_ghost_suffix(&self.input_buf, cmd) {
                            spans.push(Span::styled(suffix, bold_dim));
                        }
                    }
                }
            }

            let block = Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(orange_fg);
            f.render_widget(
                Paragraph::new(Line::from(spans)).block(block),
                composer_area,
            );
            f.render_widget(
                Paragraph::new(Text::from(chrome)).wrap(Wrap { trim: false }),
                chrome_area,
            );

            // Caret inside the block content area (one cell in from borders).
            let x = composer_area.x.saturating_add(1).saturating_add(
                (display_width(mark) + display_width(before))
                    .min(composer_area.width.saturating_sub(3) as usize) as u16,
            );
            let y = composer_area.y.saturating_add(1);
            f.set_cursor_position(Position { x, y });
        } else {
            f.render_widget(
                Paragraph::new(Text::from(chrome.clone()))
                    .wrap(Wrap { trim: false })
                    .scroll((chrome_scroll as u16, 0)),
                chrome_area,
            );
            if let Some((x_off, line_idx)) = cursor_pos {
                let line_idx = line_idx.min(chrome.len().saturating_sub(1));
                let wrapped_before = total_wrapped(&chrome[..line_idx], width);
                let y =
                    (chrome_area.y as usize + wrapped_before).saturating_sub(chrome_scroll) as u16;
                let y = y.min(
                    chrome_area
                        .y
                        .saturating_add(chrome_area.height.saturating_sub(1)),
                );
                let x = chrome_area
                    .x
                    .saturating_add(x_off.min(chrome_area.width.saturating_sub(1)));
                f.set_cursor_position(Position { x, y });
            }
        }

        // Scroll position hint when not pinned to bottom (videre shows %).
        // Must fully reset cell style (incl. BOLD from the welcome box border):
        // ratatui patches styles, so a dim-only overlay over accent BOLD keeps
        // bold and looks heavier / glitched where the % chip touches the frame.
        if !self.transcript_follow && max_off > 0 {
            let pct = (body_scroll * 100) / max_off;
            let hint = format!(" ↑ {pct}% · PgUp/PgDn · wheel · Ctrl+U/D ");
            let hint_w = display_width(&hint) as u16;
            let hx = body_area
                .x
                .saturating_add(body_area.width.saturating_sub(hint_w.saturating_add(1)));
            let hy = body_area.y;
            if body_area.width > 8 && hint_w > 0 {
                // Style::reset clears bold/colors from underlying cells; patch
                // muted fg so the chip matches the footer without inheriting
                // the orange box border weight.
                let hint_style = Style::reset().patch(dim);
                f.render_widget(
                    Paragraph::new(Span::styled(hint, hint_style)),
                    ratatui::layout::Rect {
                        x: hx,
                        y: hy,
                        width: hint_w.min(
                            body_area
                                .width
                                .saturating_sub(hx.saturating_sub(body_area.x)),
                        ),
                        height: 1,
                    },
                );
            }
        }
    }

    pub(super) fn flush_streaming(&mut self) {
        if self.streaming_text.is_empty() && self.stream_thinking.is_empty() {
            return;
        }
        // Finish the think phase before answer text so order matches Claude Code.
        if !self.stream_thinking.is_empty() {
            let elapsed = self.thinking_started.take().map(|t| t.elapsed());
            if self.show_thinking {
                self.output_lines.push(OutputLine {
                    type_: "thinking".into(),
                    content: self.stream_thinking.clone(),
                    tool_name: String::new(),
                    duration: String::new(),
                });
            } else {
                self.output_lines.push(OutputLine {
                    type_: "thinking_summary".into(),
                    content: format_thought_label(elapsed),
                    tool_name: String::new(),
                    duration: String::new(),
                });
            }
            self.stream_thinking.clear();
        } else {
            self.thinking_started = None;
        }
        if !self.streaming_text.is_empty() {
            self.output_lines.push(OutputLine {
                type_: "text".into(),
                content: self.streaming_text.clone(),
                tool_name: String::new(),
                duration: String::new(),
            });
            self.streaming_text.clear();
        }
    }

    pub(super) fn picker_visible_height(&self) -> usize {
        let h = terminal_height().unwrap_or(24).saturating_sub(10);
        if h < 3 {
            3
        } else {
            h.min(self.picker_models.len())
        }
    }
}
