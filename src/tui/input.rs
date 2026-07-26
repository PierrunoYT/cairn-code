//! Key, mouse, and paste handling for the composer and overlays.

use super::*;

impl Tui {
    /// True when a key should reach the hidden composer buffer rather than an
    /// overlay (model/provider/theme/session picker, help, or shortcuts).
    fn composer_has_focus(&self) -> bool {
        self.awaiting_api_key
            || (!self.show_model_picker
                && !self.show_provider_picker
                && !self.show_theme_picker
                && !self.show_session_picker
                && !self.show_help
                && !self.show_shortcuts)
    }

    /// Insert clipboard / bracketed-paste text into the composer at the cursor.
    /// Returns true if the buffer changed (needs redraw).
    pub(super) fn handle_paste(&mut self, data: &str) -> bool {
        // Strip CSI/OSC noise if someone pastes styled terminal output; keep
        // Unicode (emoji), newlines, and tabs for normal prompts.
        let cleaned = sanitize_paste_for_composer(data);
        if cleaned.is_empty() {
            return false;
        }
        // Only Discuss (option 4) accepts inline paste/feedback.
        if self.permission_discuss_active() {
            self.ctrl_c_exit_armed = false;
            self.perm_discuss_buf
                .insert_str(self.perm_discuss_cursor, &cleaned);
            self.perm_discuss_cursor += cleaned.len();
            return true;
        }
        // Same gates as KeyCode::Char: don't hijack list pickers or overlays.
        if !self.composer_has_focus() || self.show_permission_prompt {
            return false;
        }
        self.ctrl_c_exit_armed = false;
        self.idle_suggestion = None;
        self.input_buf.insert_str(self.cursor, &cleaned);
        self.cursor += cleaned.len();
        if !self.awaiting_api_key {
            self.update_cmd_picker();
        }
        true
    }

    /// Send a permission decision and clear the prompt.
    fn submit_permission(&mut self, decision: &str) {
        self.show_permission_prompt = false;
        self.perm_discuss_buf.clear();
        self.perm_discuss_cursor = 0;
        if let Some(tx) = &self.perm_tx {
            let _ = tx.send(decision.to_string());
        }
        self.refresh_idle_suggestion();
    }

    /// True when the Discuss option (key 4) owns the permission chrome.
    fn permission_discuss_active(&self) -> bool {
        self.show_permission_prompt && self.perm_selection == 3
    }

    /// Move the permission highlight. Inline feedback exists only on Discuss;
    /// leaving it drops any draft text so Yes/Always/No never carry a field.
    fn set_perm_selection(&mut self, sel: usize) {
        let sel = sel.min(3);
        if sel != 3 {
            self.perm_discuss_buf.clear();
            self.perm_discuss_cursor = 0;
        }
        self.perm_selection = sel;
    }

    /// Confirm the currently highlighted permission option (Enter path).
    fn confirm_permission_selection(&mut self) {
        match self.perm_selection {
            0 => self.submit_permission("allow"),
            1 => self.submit_permission("always_allow"),
            2 => self.submit_permission("deny"),
            3 => {
                // Only Discuss accepts optional inline feedback.
                let feedback = self.perm_discuss_buf.trim();
                if feedback.is_empty() {
                    self.submit_permission("discuss");
                } else {
                    self.submit_permission(&format!("discuss:{feedback}"));
                }
            }
            _ => self.submit_permission("deny"),
        }
    }

    /// Returns true if the event was actually acted on (a scroll), so callers
    /// can skip a redraw for pointer movement or other unhandled events.
    pub(super) fn handle_mouse(&mut self, kind: MouseEventKind) -> bool {
        // Ignore mouse while list pickers own the chrome (wheel should not fight them).
        if self.show_model_picker
            || self.show_provider_picker
            || self.show_theme_picker
            || self.show_session_picker
            || self.show_command_picker
        {
            return false;
        }
        match kind {
            MouseEventKind::ScrollUp => {
                self.scroll_transcript(-3);
                true
            }
            MouseEventKind::ScrollDown => {
                self.scroll_transcript(3);
                true
            }
            _ => false,
        }
    }

    /// Max transcript scroll offset from the last paint (0 = everything fits).
    fn transcript_max_off(&self) -> usize {
        self.last_body_wrapped
            .saturating_sub(self.last_body_h.max(1))
    }

    /// Scroll the transcript by `delta` wrapped lines (negative = up / older).
    /// Mirrors videre's rowoff model: free scroll until the bottom, then re-follow.
    fn scroll_transcript(&mut self, delta: isize) {
        let max_off = self.transcript_max_off();
        if max_off == 0 {
            self.transcript_rowoff = 0;
            self.transcript_follow = true;
            return;
        }
        let cur = if self.transcript_follow {
            max_off
        } else {
            self.transcript_rowoff.min(max_off)
        };
        let next = if delta < 0 {
            cur.saturating_sub((-delta) as usize)
        } else {
            cur.saturating_add(delta as usize).min(max_off)
        };
        self.transcript_rowoff = next;
        self.transcript_follow = next >= max_off;
    }

    /// Previous entry in the local prompt history (shell-style).
    fn history_prev(&mut self) {
        if self.hist_idx > 0 {
            self.hist_idx -= 1;
            self.input_buf = self.history[self.hist_idx].clone();
            self.cursor = self.input_buf.len();
        }
    }

    /// Next entry in the local prompt history (shell-style).
    fn history_next(&mut self) {
        if self.hist_idx < self.history.len().saturating_sub(1) {
            self.hist_idx += 1;
            self.input_buf = self.history[self.hist_idx].clone();
        } else {
            self.hist_idx = self.history.len();
            self.input_buf.clear();
        }
        self.cursor = self.input_buf.len();
    }

    fn scroll_page(&mut self, down: bool) {
        let page = self.last_body_h.max(1).saturating_sub(1);
        self.scroll_transcript(if down {
            page as isize
        } else {
            -(page as isize)
        });
    }

    fn scroll_half_page(&mut self, down: bool) {
        // videre: half = screen_rows / 2
        let half = (self.last_body_h.max(1) / 2).max(1);
        self.scroll_transcript(if down {
            half as isize
        } else {
            -(half as isize)
        });
    }

    pub(super) fn handle_key(&mut self, key: ratatui::crossterm::event::KeyEvent) -> bool {
        if key.kind != KeyEventKind::Press {
            return true;
        }

        let is_ctrl_c = matches!(key.code, KeyCode::Char('c') | KeyCode::Char('C'))
            && key.modifiers.contains(KeyModifiers::CONTROL);
        // Claude Code: any key other than Ctrl+C disarms the "press again to exit" arm.
        if !is_ctrl_c {
            self.ctrl_c_exit_armed = false;
        }
        if is_ctrl_c {
            return self.handle_ctrl_c();
        }

        if self.is_image_paste_chord(key) {
            self.paste_clipboard_image();
            return true;
        }

        // Transcript scroll (videre-style Page / Ctrl-U/D + arrows with Ctrl).
        // Skip when a list picker owns navigation keys. Slash ghost completion is
        // inline (not a multi-line list), so page/wheel scroll still works while
        // typing `/…`. Bare Up/Down still cycle slash candidates first.
        let picker_nav = self.show_model_picker
            || self.show_provider_picker
            || self.show_theme_picker
            || self.show_session_picker
            || self.show_permission_prompt
            || self.show_help
            || self.show_shortcuts
            || self.confirm_remove_provider.is_some()
            || self.confirm_history_provider.is_some();
        if self.show_help {
            match key.code {
                KeyCode::Esc | KeyCode::Enter | KeyCode::Char('?') => {
                    self.show_help = false;
                }
                KeyCode::PageUp => self.scroll_page(false),
                KeyCode::PageDown => self.scroll_page(true),
                KeyCode::Up if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.scroll_transcript(-1);
                }
                KeyCode::Down if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.scroll_transcript(1);
                }
                KeyCode::Up => self.scroll_transcript(-3),
                KeyCode::Down => self.scroll_transcript(3),
                _ => {}
            }
            return true;
        }
        if !picker_nav {
            match key.code {
                KeyCode::PageUp => {
                    self.scroll_page(false);
                    return true;
                }
                KeyCode::PageDown => {
                    self.scroll_page(true);
                    return true;
                }
                KeyCode::Char('u') | KeyCode::Char('U')
                    if key.modifiers.contains(KeyModifiers::CONTROL) =>
                {
                    self.scroll_half_page(false);
                    return true;
                }
                KeyCode::Char('d') | KeyCode::Char('D')
                    if key.modifiers.contains(KeyModifiers::CONTROL) =>
                {
                    self.scroll_half_page(true);
                    return true;
                }
                // Readline-style history that works even when Up/Down scroll the chat.
                KeyCode::Char('p') | KeyCode::Char('P')
                    if key.modifiers.contains(KeyModifiers::CONTROL) =>
                {
                    self.history_prev();
                    return true;
                }
                KeyCode::Char('n') | KeyCode::Char('N')
                    if key.modifiers.contains(KeyModifiers::CONTROL) =>
                {
                    self.history_next();
                    return true;
                }
                KeyCode::Up if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.scroll_transcript(-1);
                    return true;
                }
                KeyCode::Down if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.scroll_transcript(1);
                    return true;
                }
                KeyCode::Home if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.transcript_follow = false;
                    self.transcript_rowoff = 0;
                    return true;
                }
                KeyCode::End if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.transcript_follow = true;
                    return true;
                }
                // Copy last assistant message to the system clipboard.
                KeyCode::Char('y') | KeyCode::Char('Y')
                    if key.modifiers.contains(KeyModifiers::CONTROL) =>
                {
                    self.copy_last_assistant_to_clipboard();
                    return true;
                }
                // Plain-text select mode (reliable drag-select on Windows Terminal).
                KeyCode::Char('o') | KeyCode::Char('O')
                    if key.modifiers.contains(KeyModifiers::CONTROL) =>
                {
                    self.pending_select = Some(SelectDump::LastAssistant);
                    return true;
                }
                _ => {}
            }
        }

        if self.show_shortcuts {
            match key.code {
                KeyCode::Esc | KeyCode::Char('?') => {
                    self.show_shortcuts = false;
                }
                // Keep scroll available while the help panel is open.
                KeyCode::PageUp => self.scroll_page(false),
                KeyCode::PageDown => self.scroll_page(true),
                KeyCode::Char('u') | KeyCode::Char('U')
                    if key.modifiers.contains(KeyModifiers::CONTROL) =>
                {
                    self.scroll_half_page(false);
                }
                KeyCode::Char('d') | KeyCode::Char('D')
                    if key.modifiers.contains(KeyModifiers::CONTROL) =>
                {
                    self.scroll_half_page(true);
                }
                KeyCode::Up if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.scroll_transcript(-1);
                }
                KeyCode::Down if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.scroll_transcript(1);
                }
                KeyCode::Up => self.scroll_transcript(-3),
                KeyCode::Down => self.scroll_transcript(3),
                _ => {}
            }
            return true;
        }

        if self.show_permission_prompt {
            match key.code {
                // Claude Code / OpenClaude: 1/2/3 confirm immediately (no text field).
                // 4 focuses Discuss, the only option with optional inline feedback.
                KeyCode::Char('1') => self.submit_permission("allow"),
                KeyCode::Char('2') => self.submit_permission("always_allow"),
                KeyCode::Char('3') => self.submit_permission("deny"),
                KeyCode::Char('4') => self.set_perm_selection(3),
                KeyCode::Up => {
                    self.set_perm_selection(self.perm_selection.saturating_sub(1));
                }
                KeyCode::Down => {
                    if self.perm_selection < 3 {
                        self.set_perm_selection(self.perm_selection + 1);
                    }
                }
                // Discuss-only: Left/Right move the feedback cursor.
                KeyCode::Left
                    if self.permission_discuss_active() && self.perm_discuss_cursor > 0 =>
                {
                    let previous = self.perm_discuss_buf[..self.perm_discuss_cursor]
                        .char_indices()
                        .next_back()
                        .map(|(index, _)| index)
                        .unwrap_or(0);
                    self.perm_discuss_cursor = previous;
                }
                KeyCode::Left => {
                    self.set_perm_selection(self.perm_selection.saturating_sub(1));
                }
                KeyCode::Right if self.permission_discuss_active() => {
                    if self.perm_discuss_cursor < self.perm_discuss_buf.len() {
                        let next = self.perm_discuss_buf[self.perm_discuss_cursor..]
                            .chars()
                            .next()
                            .map(|c| c.len_utf8())
                            .unwrap_or(0);
                        self.perm_discuss_cursor += next;
                    }
                }
                KeyCode::Right => {
                    if self.perm_selection < 3 {
                        self.set_perm_selection(self.perm_selection + 1);
                    }
                }
                KeyCode::Home if self.permission_discuss_active() => {
                    self.perm_discuss_cursor = 0;
                }
                KeyCode::End if self.permission_discuss_active() => {
                    self.perm_discuss_cursor = self.perm_discuss_buf.len();
                }
                KeyCode::Enter => self.confirm_permission_selection(),
                KeyCode::Esc => self.submit_permission("deny"),
                KeyCode::Backspace if self.permission_discuss_active() => {
                    if self.perm_discuss_cursor > 0 {
                        let previous = self.perm_discuss_buf[..self.perm_discuss_cursor]
                            .char_indices()
                            .next_back()
                            .map(|(index, _)| index)
                            .unwrap_or(0);
                        self.perm_discuss_buf.remove(previous);
                        self.perm_discuss_cursor = previous;
                    }
                }
                KeyCode::Delete if self.permission_discuss_active() => {
                    if self.perm_discuss_cursor < self.perm_discuss_buf.len() {
                        self.perm_discuss_buf.remove(self.perm_discuss_cursor);
                    }
                }
                // Printable text only while Discuss is selected (not Yes/Always/No).
                KeyCode::Char(ch) if self.permission_discuss_active() && !ch.is_control() => {
                    // Digits 1-4 already matched above as global shortcuts.
                    self.perm_discuss_buf.insert(self.perm_discuss_cursor, ch);
                    self.perm_discuss_cursor += ch.len_utf8();
                }
                _ => {}
            }
            return true;
        }

        if self.show_recovery_prompt {
            match key.code {
                KeyCode::Left => {
                    self.recovery_selection = self.recovery_selection.saturating_sub(1);
                }
                KeyCode::Right => {
                    if self.recovery_selection < 2 {
                        self.recovery_selection += 1;
                    }
                }
                KeyCode::Char('m') | KeyCode::Char('M') => {
                    self.show_recovery_prompt = false;
                    self.open_model_picker();
                }
                KeyCode::Char('p') | KeyCode::Char('P') => {
                    self.show_recovery_prompt = false;
                    self.open_provider_picker();
                }
                KeyCode::Char('d') | KeyCode::Char('D') | KeyCode::Esc => {
                    self.show_recovery_prompt = false;
                    self.refresh_idle_suggestion();
                }
                KeyCode::Enter => {
                    let sel = self.recovery_selection;
                    self.show_recovery_prompt = false;
                    match sel {
                        0 => self.open_model_picker(),
                        1 => self.open_provider_picker(),
                        _ => self.refresh_idle_suggestion(),
                    }
                }
                _ => {}
            }
            return true;
        }

        if let Some(name) = self.confirm_remove_provider.clone() {
            match key.code {
                KeyCode::Left => {
                    self.confirm_remove_sel = 0;
                }
                KeyCode::Right => {
                    self.confirm_remove_sel = 1;
                }
                KeyCode::Esc => {
                    self.confirm_remove_provider = None;
                }
                KeyCode::Enter => {
                    let remove = self.confirm_remove_sel == 1;
                    self.confirm_remove_provider = None;
                    if remove {
                        let (type_, content) = match crate::config::remove_api_key(&name) {
                            Ok(true) => {
                                let env_note = match crate::config::env_key_for(&name) {
                                    Some(_) => format!(" Note: {} is still set in the environment and will still be used.",
                                        crate::config::env_var_name(&name).unwrap_or("its environment variable")),
                                    None => String::new(),
                                };
                                (
                                    "system",
                                    format!("Removed saved API key for {name}.{env_note}"),
                                )
                            }
                            Ok(false) => ("system", format!("No saved API key for {name}.")),
                            Err(e) => {
                                ("error", format!("Failed to remove API key for {name}: {e}"))
                            }
                        };
                        self.output_lines.push(OutputLine {
                            type_: type_.into(),
                            content,
                            tool_name: String::new(),
                            duration: String::new(),
                        });
                        self.provider_picker_keys = self
                            .provider_picker_list
                            .iter()
                            .map(|n| crate::config::has_usable_credential(n))
                            .collect();
                    }
                }
                _ => {}
            }
            return true;
        }

        if let Some(name) = self.confirm_history_provider.clone() {
            match key.code {
                KeyCode::Left => {
                    self.confirm_history_sel = 0;
                }
                KeyCode::Right => {
                    self.confirm_history_sel = 1;
                }
                KeyCode::Esc => {
                    self.confirm_history_provider = None;
                }
                KeyCode::Enter => {
                    let proceed = self.confirm_history_sel == 1;
                    self.confirm_history_provider = None;
                    if proceed {
                        self.begin_provider_selection(name);
                    }
                }
                _ => {}
            }
            return true;
        }

        match key.code {
            KeyCode::Up => {
                if self.show_session_picker {
                    if self.picker_session_sel > 0 {
                        self.picker_session_sel -= 1;
                    }
                    return true;
                }
                if self.show_theme_picker {
                    if self.theme_picker_sel > 0 {
                        self.theme_picker_sel -= 1;
                        self.theme = self.theme_picker_list[self.theme_picker_sel].clone();
                    }
                    return true;
                }
                // Cycle slash ghost candidates without a multi-line picker.
                if !self.cmd_picker_filtered.is_empty() {
                    if self.cmd_picker_sel > 0 {
                        self.cmd_picker_sel -= 1;
                    }
                    return true;
                }
                if self.show_provider_picker {
                    if self.provider_picker_sel > 0 {
                        self.provider_picker_sel -= 1;
                    }
                    return true;
                }
                if self.show_model_picker {
                    if self.picker_sel > 0 {
                        self.picker_sel -= 1;
                        if self.picker_sel < self.picker_scrl {
                            self.picker_scrl = self.picker_sel;
                        }
                    }
                    return true;
                }
                // Windows Terminal (and others) on the alt screen often send the
                // mouse wheel as bare Up/Down. Prefer transcript scroll whenever
                // the chat overflows so wheel does not walk prompt history.
                if self.transcript_max_off() > 0 {
                    self.scroll_transcript(-3);
                    return true;
                }
                self.history_prev();
                true
            }
            KeyCode::Down => {
                if self.show_session_picker {
                    if self.picker_session_sel + 1 < self.picker_sessions.len() {
                        self.picker_session_sel += 1;
                    }
                    return true;
                }
                if self.show_theme_picker {
                    if self.theme_picker_sel + 1 < self.theme_picker_list.len() {
                        self.theme_picker_sel += 1;
                        self.theme = self.theme_picker_list[self.theme_picker_sel].clone();
                    }
                    return true;
                }
                if !self.cmd_picker_filtered.is_empty() {
                    if self.cmd_picker_sel + 1 < self.cmd_picker_filtered.len() {
                        self.cmd_picker_sel += 1;
                    }
                    return true;
                }
                if self.show_provider_picker {
                    if self.provider_picker_sel + 1 < self.provider_picker_list.len() {
                        self.provider_picker_sel += 1;
                    }
                    return true;
                }
                if self.show_model_picker {
                    if self.picker_sel + 1 < self.picker_models.len() {
                        self.picker_sel += 1;
                        let vh = self.picker_visible_height();
                        if self.picker_sel >= self.picker_scrl + vh {
                            self.picker_scrl = self.picker_sel - vh + 1;
                        }
                    }
                    return true;
                }
                if self.transcript_max_off() > 0 {
                    self.scroll_transcript(3);
                    return true;
                }
                self.history_next();
                true
            }
            KeyCode::Left => {
                if self.cursor > 0 {
                    self.cursor = self.input_buf[..self.cursor]
                        .char_indices()
                        .next_back()
                        .map(|(index, _)| index)
                        .unwrap_or(0);
                }
                true
            }
            KeyCode::Right => {
                // Right arrow at end: accept slash ghost, else empty-composer idle hint.
                if self.cursor >= self.input_buf.len() && !self.awaiting_api_key {
                    if let Some(cmd) = self.selected_slash_completion() {
                        if slash_ghost_suffix(&self.input_buf, cmd).is_some() {
                            let cmd = cmd.to_string();
                            self.apply_slash_completion(&cmd);
                            return true;
                        }
                    }
                    if self.input_buf.is_empty() {
                        if let Some(hint) = self.idle_suggestion.clone() {
                            self.input_buf = hint;
                            self.cursor = self.input_buf.len();
                            self.idle_suggestion = None;
                            return true;
                        }
                    }
                }
                if self.cursor < self.input_buf.len() {
                    self.cursor += self.input_buf[self.cursor..]
                        .chars()
                        .next()
                        .map(char::len_utf8)
                        .unwrap_or(0);
                }
                true
            }
            KeyCode::Tab => {
                // Empty composer: accept grayed ready-to-send prompt.
                if self.input_buf.is_empty()
                    && !self.awaiting_api_key
                    && !self.show_model_picker
                    && !self.show_provider_picker
                    && !self.show_theme_picker
                    && !self.show_session_picker
                {
                    if let Some(hint) = self.idle_suggestion.clone() {
                        self.input_buf = hint;
                        self.cursor = self.input_buf.len();
                        self.idle_suggestion = None;
                        return true;
                    }
                }
                // Slash ghost: Tab fills the selected completion (then next-arg candidates).
                if self.input_buf.starts_with('/') {
                    self.update_cmd_picker();
                }
                if let Some(cmd) = self.selected_slash_completion().map(|s| s.to_string()) {
                    self.apply_slash_completion(&cmd);
                }
                true
            }
            KeyCode::Esc => {
                if self.awaiting_api_key {
                    self.awaiting_api_key = false;
                    self.cancel_pending_provider_selection();
                    self.api_key_target = None;
                    self.input_buf.clear();
                    self.cursor = 0;
                    self.output_lines.push(OutputLine {
                        type_: "system".into(),
                        content: "API key entry cancelled.".into(),
                        tool_name: String::new(),
                        duration: String::new(),
                    });
                } else if self.show_help {
                    self.show_help = false;
                } else if self.show_shortcuts {
                    self.show_shortcuts = false;
                } else if self.show_command_picker || !self.cmd_picker_filtered.is_empty() {
                    self.show_command_picker = false;
                    self.cmd_picker_filtered.clear();
                    self.cmd_picker_sel = 0;
                } else if self.show_provider_picker {
                    self.show_provider_picker = false;
                } else if self.show_model_picker {
                    self.show_model_picker = false;
                    self.cancel_pending_provider_selection();
                } else if self.show_session_picker {
                    self.show_session_picker = false;
                    self.session_picker_delete = false;
                } else if self.show_theme_picker {
                    self.show_theme_picker = false;
                    if let Some(prev) = self.theme_before_picker.take() {
                        self.theme = theme::lookup(&prev);
                    }
                } else if matches!(self.state, State::Running) {
                    if let Some(flag) = &self.cancel_flag {
                        flag.store(true, Ordering::Relaxed);
                    }
                    self.cancel_pending_provider_selection();
                    self.flush_streaming();
                    self.output_lines.push(OutputLine {
                        type_: "system".into(),
                        content: "Cancelled.".into(),
                        tool_name: String::new(),
                        duration: String::new(),
                    });
                }
                true
            }
            KeyCode::Enter => {
                // Enter always submits the typed buffer (Tab/Right accept the ghost first).
                // Clear any leftover slash-completion state so it does not steal focus.
                if self.show_command_picker || !self.cmd_picker_filtered.is_empty() {
                    self.show_command_picker = false;
                    self.cmd_picker_filtered.clear();
                    self.cmd_picker_sel = 0;
                }
                if self.show_provider_picker {
                    if self.provider_picker_sel < self.provider_picker_list.len() {
                        let name = self.provider_picker_list[self.provider_picker_sel].clone();
                        self.show_provider_picker = false;
                        if self.provider_switch_needs_confirmation(&name) {
                            self.confirm_history_provider = Some(name);
                            self.confirm_history_sel = 0;
                        } else {
                            self.begin_provider_selection(name);
                        }
                    } else {
                        self.show_provider_picker = false;
                    }
                    return true;
                }
                if self.show_model_picker {
                    if self.picker_sel < self.picker_models.len() {
                        let selected_model = self.picker_models[self.picker_sel].id.clone();
                        self.show_model_picker = false;
                        let provider = self
                            .pending_provider_selection
                            .take()
                            .unwrap_or_else(|| self.provider.clone());
                        self.finish_provider_model_selection(provider, selected_model);
                    } else {
                        self.show_model_picker = false;
                        self.cancel_pending_provider_selection();
                    }
                    return true;
                }

                if self.show_session_picker {
                    if self.picker_session_sel < self.picker_sessions.len() {
                        let id = self.picker_sessions[self.picker_session_sel].id.clone();
                        let deleting = self.session_picker_delete;
                        self.show_session_picker = false;
                        self.session_picker_delete = false;
                        if deleting {
                            self.delete_session(&id);
                        } else {
                            self.resume_session(&id);
                        }
                    } else {
                        self.show_session_picker = false;
                        self.session_picker_delete = false;
                    }
                    return true;
                }

                if self.show_theme_picker {
                    if self.theme_picker_sel < self.theme_picker_list.len() {
                        let selected = self.theme_picker_list[self.theme_picker_sel].clone();
                        let name = selected.name.to_string();
                        let previous = self
                            .theme_before_picker
                            .as_deref()
                            .map(theme::lookup)
                            .unwrap_or_else(|| self.theme.clone());
                        self.apply_theme_preference(
                            selected,
                            previous,
                            crate::config::save_theme(&name),
                        );
                    }
                    self.show_theme_picker = false;
                    self.theme_before_picker = None;
                    return true;
                }

                if self.awaiting_api_key {
                    let key = self.input_buf.trim().to_string();
                    self.input_buf.clear();
                    self.cursor = 0;
                    if key.is_empty() {
                        return true;
                    }
                    self.awaiting_api_key = false;
                    let target = self
                        .api_key_target
                        .take()
                        .unwrap_or_else(|| self.provider.clone());
                    if self.pending_model_after_auth {
                        // Provider switch path: save key, then open model list (live catalog).
                        match crate::config::save_api_key(&target, &key) {
                            Ok(()) => {
                                self.pending_model_after_auth = false;
                                crate::config::apply_key_to_env(&target, &key);
                                let tail = crate::config::mask_secret_display(&key, 4);
                                self.output_lines.push(OutputLine {
                                    type_: "system".into(),
                                    content: format!(
                                        "API key saved for {target} ({tail}). Choose a model."
                                    ),
                                    tool_name: String::new(),
                                    duration: String::new(),
                                });
                                self.open_model_picker();
                            }
                            Err(error) => {
                                self.cancel_pending_provider_selection();
                                self.output_lines.push(OutputLine {
                                    type_: "error".into(),
                                    content: format!(
                                        "Failed to save API key for {target}: {error}"
                                    ),
                                    tool_name: String::new(),
                                    duration: String::new(),
                                });
                            }
                        }
                    } else {
                        // `/auth key` saves credentials only; it must not bypass
                        // cross-provider history-transfer confirmation.
                        let (type_, content) = match crate::config::save_api_key(&target, &key) {
                            Ok(()) => {
                                crate::config::apply_key_to_env(&target, &key);
                                (
                                    "system",
                                    format!(
                                        "API key saved for {target} ({}).",
                                        crate::config::mask_secret_display(&key, 4)
                                    ),
                                )
                            }
                            Err(error) => (
                                "error",
                                format!("Failed to save API key for {target}: {error}"),
                            ),
                        };
                        self.output_lines.push(OutputLine {
                            type_: type_.into(),
                            content,
                            tool_name: String::new(),
                            duration: String::new(),
                        });
                    }
                    return true;
                }

                if !matches!(self.state, State::Idle) {
                    return true;
                }

                let input = self.input_buf.trim().to_string();
                if input.is_empty() && self.pending_images.is_empty() {
                    return true;
                }

                if !input.is_empty() && input.starts_with('/') && self.pending_images.is_empty() {
                    self.input_buf.clear();
                    self.cursor = 0;
                    return self.handle_command(&input);
                }

                self.input_buf.clear();
                self.cursor = 0;
                if !input.is_empty() {
                    self.history.push(input.clone());
                    self.hist_idx = self.history.len();
                }
                self.show_recovery_prompt = false;
                // New turn: pin transcript to bottom (follow latest output).
                self.transcript_follow = true;
                self.expect_turn_notify = true;
                self.idle_suggestion = None;
                // Stable id for the whole conversation so mid-turn checkpoints
                // and the final save share one file from the first prompt.
                self.ensure_session_identity();
                let images = std::mem::take(&mut self.pending_images);
                let label = if images.is_empty() {
                    input.clone()
                } else {
                    llm::UserBlocks {
                        text: input.clone(),
                        images: images.clone(),
                    }
                    .display_label()
                };
                self.output_lines.push(OutputLine {
                    type_: "user".into(),
                    content: label,
                    tool_name: String::new(),
                    duration: String::new(),
                });
                self.begin_running();
                if let Some(tx) = &self.agent_tx {
                    if images.is_empty() {
                        let _ = tx.send(input);
                    } else {
                        let _ = tx.send(encode_user_json_cmd(&input, &images));
                    }
                }
                true
            }
            KeyCode::Backspace => {
                if self.cursor > 0 && self.composer_has_focus() {
                    let previous = self.input_buf[..self.cursor]
                        .char_indices()
                        .next_back()
                        .map(|(index, _)| index)
                        .unwrap_or(0);
                    self.input_buf.remove(previous);
                    self.cursor = previous;
                    self.update_cmd_picker();
                    if self.input_buf.is_empty() {
                        self.refresh_idle_suggestion();
                    }
                }
                true
            }
            KeyCode::Delete => {
                if self.show_provider_picker {
                    if self.provider_picker_sel < self.provider_picker_list.len() {
                        let name = self.provider_picker_list[self.provider_picker_sel].clone();
                        if crate::config::config_has_api_key(&name) {
                            self.confirm_remove_provider = Some(name);
                            self.confirm_remove_sel = 0;
                        } else {
                            self.output_lines.push(OutputLine {
                                type_: "system".into(),
                                content: format!("No saved API key for {name} in the config file."),
                                tool_name: String::new(),
                                duration: String::new(),
                            });
                        }
                    }
                    return true;
                }
                if self.cursor < self.input_buf.len() && self.composer_has_focus() {
                    self.input_buf.remove(self.cursor);
                }
                true
            }
            KeyCode::Char(ch) => {
                // Footer advertises "? for shortcuts" when the composer is empty.
                // Toggle the panel instead of inserting `?` into the prompt.
                if ch == '?'
                    && !self.awaiting_api_key
                    && self.input_buf.is_empty()
                    && matches!(self.state, State::Idle)
                    && !self.show_model_picker
                    && !self.show_provider_picker
                    && !self.show_theme_picker
                    && !self.show_session_picker
                    && !self.show_permission_prompt
                    && !self.show_recovery_prompt
                    && self.confirm_remove_provider.is_none()
                    && self.confirm_history_provider.is_none()
                {
                    self.show_shortcuts = !self.show_shortcuts;
                    return true;
                }
                // Allow typing into the API-key prompt and the normal input
                // (but not while a list picker is focused).
                if self.composer_has_focus() {
                    self.input_buf.insert(self.cursor, ch);
                    self.cursor += ch.len_utf8();
                    if !self.awaiting_api_key {
                        self.update_cmd_picker();
                    }
                }
                true
            }
            _ => true,
        }
    }

    /// Claude Code Ctrl+C: interrupt running work; clear prompt when idle with
    /// text; on empty idle prompt, arm exit then quit on a second press.
    /// Returns `false` to exit the TUI.
    pub(super) fn handle_ctrl_c(&mut self) -> bool {
        // Close overlays / cancel key entry first (same spirit as Esc).
        if self.awaiting_api_key {
            self.awaiting_api_key = false;
            self.cancel_pending_provider_selection();
            self.api_key_target = None;
            self.input_buf.clear();
            self.cursor = 0;
            self.ctrl_c_exit_armed = false;
            self.output_lines.push(OutputLine {
                type_: "system".into(),
                content: "API key entry cancelled.".into(),
                tool_name: String::new(),
                duration: String::new(),
            });
            return true;
        }
        if self.confirm_remove_provider.is_some() {
            self.confirm_remove_provider = None;
            self.ctrl_c_exit_armed = false;
            return true;
        }
        if self.confirm_history_provider.is_some() {
            self.confirm_history_provider = None;
            self.ctrl_c_exit_armed = false;
            return true;
        }
        if self.show_command_picker {
            self.show_command_picker = false;
            self.ctrl_c_exit_armed = false;
            return true;
        }
        if self.show_provider_picker {
            self.show_provider_picker = false;
            self.ctrl_c_exit_armed = false;
            return true;
        }
        if self.show_model_picker {
            self.show_model_picker = false;
            self.cancel_pending_provider_selection();
            self.ctrl_c_exit_armed = false;
            return true;
        }
        if self.show_help {
            self.show_help = false;
            self.ctrl_c_exit_armed = false;
            return true;
        }
        if self.show_shortcuts {
            self.show_shortcuts = false;
            self.ctrl_c_exit_armed = false;
            return true;
        }
        if self.show_session_picker {
            self.show_session_picker = false;
            self.session_picker_delete = false;
            self.ctrl_c_exit_armed = false;
            return true;
        }
        if self.show_theme_picker {
            self.show_theme_picker = false;
            if let Some(prev) = self.theme_before_picker.take() {
                self.theme = theme::lookup(&prev);
            }
            self.ctrl_c_exit_armed = false;
            return true;
        }
        if self.show_recovery_prompt {
            self.show_recovery_prompt = false;
            self.ctrl_c_exit_armed = false;
            return true;
        }
        if self.show_permission_prompt {
            self.submit_permission("deny");
            self.ctrl_c_exit_armed = false;
            return true;
        }

        // Interrupt a running agent turn.
        if matches!(self.state, State::Running) {
            if let Some(flag) = &self.cancel_flag {
                flag.store(true, Ordering::Relaxed);
            }
            self.cancel_pending_provider_selection();
            self.flush_streaming();
            self.ctrl_c_exit_armed = false;
            self.output_lines.push(OutputLine {
                type_: "system".into(),
                content: "Interrupted.".into(),
                tool_name: String::new(),
                duration: String::new(),
            });
            return true;
        }

        // Idle: clear non-empty prompt (first action).
        if !self.input_buf.is_empty() {
            self.input_buf.clear();
            self.cursor = 0;
            self.show_command_picker = false;
            self.cmd_picker_filtered.clear();
            self.ctrl_c_exit_armed = false;
            return true;
        }

        // Empty idle prompt: second Ctrl+C exits; first arms confirmation.
        // Claude Code shows the hint in the footer chrome, not as a transcript line.
        if self.ctrl_c_exit_armed {
            return false;
        }
        self.ctrl_c_exit_armed = true;
        true
    }
}

#[cfg(test)]
mod shortcuts_panel_tests {
    use super::*;
    use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    #[test]
    fn question_mark_toggles_shortcuts_when_composer_empty() {
        let mut tui = Tui::new("test", "model", "provider", ".");
        assert!(!tui.show_shortcuts);

        tui.handle_key(KeyEvent::new(KeyCode::Char('?'), KeyModifiers::NONE));
        assert!(tui.show_shortcuts);
        assert!(tui.input_buf.is_empty(), "must not type ? into the prompt");

        tui.handle_key(KeyEvent::new(KeyCode::Char('?'), KeyModifiers::NONE));
        assert!(!tui.show_shortcuts);
    }

    #[test]
    fn question_mark_types_when_composer_has_text() {
        let mut tui = Tui::new("test", "model", "provider", ".");
        tui.input_buf = "what".into();
        tui.cursor = tui.input_buf.len();

        tui.handle_key(KeyEvent::new(KeyCode::Char('?'), KeyModifiers::NONE));

        assert!(!tui.show_shortcuts);
        assert_eq!(tui.input_buf, "what?");
    }

    #[test]
    fn esc_closes_shortcuts_panel() {
        let mut tui = Tui::new("test", "model", "provider", ".");
        tui.show_shortcuts = true;

        tui.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));

        assert!(!tui.show_shortcuts);
    }
}

#[cfg(test)]
mod scroll_history_tests {
    use super::*;
    use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    #[test]
    fn bare_up_scrolls_when_transcript_overflows_instead_of_history() {
        let mut tui = Tui::new("test", "model", "provider", ".");
        tui.history = vec!["first".into(), "second".into()];
        tui.hist_idx = tui.history.len();
        // Simulate a painted frame where the body is taller than the viewport.
        tui.last_body_wrapped = 100;
        tui.last_body_h = 20;
        tui.transcript_follow = true;
        tui.transcript_rowoff = 80;

        tui.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));

        assert!(
            !tui.transcript_follow,
            "Up should leave follow mode when content overflows"
        );
        assert!(
            tui.transcript_rowoff < 80,
            "Up should move rowoff toward older content"
        );
        assert!(
            tui.input_buf.is_empty(),
            "Up must not load prompt history while chat can scroll"
        );
    }

    #[test]
    fn bare_up_uses_history_when_transcript_fits() {
        let mut tui = Tui::new("test", "model", "provider", ".");
        tui.history = vec!["prior prompt".into()];
        tui.hist_idx = tui.history.len();
        tui.last_body_wrapped = 10;
        tui.last_body_h = 40;
        tui.transcript_follow = true;

        tui.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));

        assert_eq!(tui.input_buf, "prior prompt");
        assert_eq!(tui.hist_idx, 0);
    }

    #[test]
    fn ctrl_p_always_walks_prompt_history() {
        let mut tui = Tui::new("test", "model", "provider", ".");
        tui.history = vec!["a".into(), "b".into()];
        tui.hist_idx = tui.history.len();
        tui.last_body_wrapped = 100;
        tui.last_body_h = 20;
        tui.transcript_follow = true;

        tui.handle_key(KeyEvent::new(KeyCode::Char('p'), KeyModifiers::CONTROL));

        assert_eq!(tui.input_buf, "b");
        assert_eq!(tui.hist_idx, 1);
    }
}

#[cfg(test)]
mod unicode_input_tests {
    use super::*;
    use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    pub(super) fn press(tui: &mut Tui, code: KeyCode) {
        tui.handle_key(KeyEvent::new(code, KeyModifiers::NONE));
    }

    #[test]
    fn cursor_and_edits_follow_unicode_character_boundaries() {
        let mut tui = Tui::new("test", "test-model", "test-provider", ".");
        tui.input_buf = "é界🙂".into();
        tui.cursor = tui.input_buf.len();

        press(&mut tui, KeyCode::Left);
        assert_eq!(tui.cursor, "é界".len());
        press(&mut tui, KeyCode::Left);
        assert_eq!(tui.cursor, "é".len());
        press(&mut tui, KeyCode::Right);
        assert_eq!(tui.cursor, "é界".len());

        press(&mut tui, KeyCode::Backspace);
        assert_eq!(tui.input_buf, "é🙂");
        assert_eq!(tui.cursor, "é".len());
        press(&mut tui, KeyCode::Delete);
        assert_eq!(tui.input_buf, "é");

        press(&mut tui, KeyCode::Char('界'));
        press(&mut tui, KeyCode::Char('🙂'));
        assert_eq!(tui.input_buf, "é界🙂");
        assert_eq!(tui.cursor, tui.input_buf.len());
    }
}

#[cfg(test)]
mod claude_chrome_tests {
    use super::*;
    use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    #[test]
    fn format_elapsed_compact_units() {
        assert_eq!(format_elapsed_compact(Duration::from_secs(0)), "0s");
        assert_eq!(format_elapsed_compact(Duration::from_secs(9)), "9s");
        assert_eq!(format_elapsed_compact(Duration::from_secs(65)), "1m 05s");
        assert_eq!(format_elapsed_compact(Duration::from_secs(600)), "10m 00s");
    }

    #[test]
    fn rock_emoji_is_double_width_for_welcome_pad() {
        // U+1FAA8 ROCK must count as 2 cols or the welcome right border slips.
        assert_eq!(char_width('🪨'), 2);
        // "  " (2) + rock (2) + " Cairn" (6) = 10
        assert_eq!(display_width("  🪨 Cairn"), 10);
        let padded = pad_to_display_width("  🪨 Cairn Code v0.1.0", 40);
        assert_eq!(display_width(&padded), 40);
    }

    #[test]
    fn ctrl_c_arms_exit_without_transcript_noise() {
        let mut tui = Tui::new("test", "model", "provider", ".");
        assert!(tui.input_buf.is_empty());
        assert!(tui.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)));
        assert!(tui.ctrl_c_exit_armed);
        assert!(
            !tui.output_lines
                .iter()
                .any(|l| l.content.contains("Ctrl+C")),
            "exit hint belongs in the footer, not the transcript"
        );
    }
}

#[cfg(test)]
mod permission_prompt_tests {
    use super::*;
    use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use std::sync::mpsc;

    fn prompt_tui() -> (Tui, mpsc::Receiver<String>) {
        let mut tui = Tui::new("test", "model", "provider", ".");
        let (tx, rx) = mpsc::channel();
        tui.set_perm_tx(tx);
        tui.show_permission_prompt = true;
        tui.perm_tool_name = "shell".into();
        tui.perm_tool_input = r#"{"command":"echo hi"}"#.into();
        tui.perm_selection = 0;
        (tui, rx)
    }

    #[test]
    fn number_keys_one_two_three_confirm_without_text_field() {
        for (key, expected) in [
            (KeyCode::Char('1'), "allow"),
            (KeyCode::Char('2'), "always_allow"),
            (KeyCode::Char('3'), "deny"),
        ] {
            let (mut tui, rx) = prompt_tui();
            // Typing on Yes/Always/No must not populate discuss feedback.
            tui.handle_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE));
            assert!(
                tui.perm_discuss_buf.is_empty(),
                "options 1-3 must not accept inline text"
            );
            tui.handle_key(KeyEvent::new(key, KeyModifiers::NONE));
            assert!(!tui.show_permission_prompt);
            assert_eq!(rx.try_recv().unwrap(), expected);
        }
    }

    #[test]
    fn key_four_selects_discuss_and_allows_inline_feedback() {
        let (mut tui, rx) = prompt_tui();
        tui.handle_key(KeyEvent::new(KeyCode::Char('4'), KeyModifiers::NONE));
        assert!(tui.show_permission_prompt);
        assert_eq!(tui.perm_selection, 3);
        assert!(tui.permission_discuss_active());

        for ch in ['u', 's', 'e', ' ', 'g', 'i', 't'] {
            tui.handle_key(KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE));
        }
        assert_eq!(tui.perm_discuss_buf, "use git");

        tui.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert!(!tui.show_permission_prompt);
        assert_eq!(rx.try_recv().unwrap(), "discuss:use git");
    }

    #[test]
    fn discuss_enter_without_text_sends_bare_discuss() {
        let (mut tui, rx) = prompt_tui();
        tui.handle_key(KeyEvent::new(KeyCode::Char('4'), KeyModifiers::NONE));
        tui.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(rx.try_recv().unwrap(), "discuss");
    }

    #[test]
    fn leaving_discuss_clears_draft_feedback() {
        let (mut tui, _rx) = prompt_tui();
        tui.handle_key(KeyEvent::new(KeyCode::Char('4'), KeyModifiers::NONE));
        tui.handle_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE));
        assert_eq!(tui.perm_discuss_buf, "x");
        tui.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(tui.perm_selection, 2);
        assert!(
            tui.perm_discuss_buf.is_empty(),
            "draft must not linger on No/Yes options"
        );
        assert!(!tui.permission_discuss_active());
    }

    #[test]
    fn paste_only_into_discuss_not_other_options() {
        let (mut tui, _rx) = prompt_tui();
        assert!(!tui.handle_paste("nope"));
        assert!(tui.perm_discuss_buf.is_empty());
        tui.set_perm_selection(3);
        assert!(tui.handle_paste("hello"));
        assert_eq!(tui.perm_discuss_buf, "hello");
    }
}

#[cfg(test)]
mod paste_tests {
    use super::*;

    #[test]
    fn paste_inserts_emoji_and_unicode_at_cursor() {
        let mut tui = Tui::new("test", "model", "provider", ".");
        tui.input_buf = "hi ".into();
        tui.cursor = tui.input_buf.len();
        assert!(tui.handle_paste("🪨 world 🙂"));
        assert_eq!(tui.input_buf, "hi 🪨 world 🙂");
        assert_eq!(tui.cursor, tui.input_buf.len());
    }

    #[test]
    fn paste_mid_buffer_and_multiline() {
        let mut tui = Tui::new("test", "model", "provider", ".");
        tui.input_buf = "ab".into();
        tui.cursor = 1; // between a and b
        assert!(tui.handle_paste("X\nY"));
        assert_eq!(tui.input_buf, "aX\nYb");
        assert_eq!(tui.cursor, "aX\nY".len());
    }

    #[test]
    fn paste_strips_csi_keeps_text() {
        let cleaned = sanitize_paste_for_composer("hello\x1b[31mred\x1b[0m 🪨");
        assert!(cleaned.contains("hello"));
        assert!(cleaned.contains("red"));
        assert!(cleaned.contains('🪨'));
        assert!(!cleaned.contains('\u{001b}'));
    }

    #[test]
    fn paste_ignored_while_model_picker_open() {
        let mut tui = Tui::new("test", "model", "provider", ".");
        tui.show_model_picker = true;
        assert!(!tui.handle_paste("should not land"));
        assert!(tui.input_buf.is_empty());
    }
}
