//! Clipboard, OSC 52, and plain-text select mode.

use super::*;

impl Tui {
    /// Copy the most recent assistant text to the OS clipboard.
    pub(super) fn copy_last_assistant_to_clipboard(&mut self) {
        let owned = self.last_assistant_text();
        let Some(text) = owned.as_deref().filter(|s| !s.trim().is_empty()) else {
            self.output_lines.push(OutputLine {
                type_: "system".into(),
                content: "Nothing to copy (no assistant message yet). Try /select to open plain-text view."
                    .into(),
                tool_name: String::new(),
                duration: String::new(),
            });
            return;
        };
        match copy_text_to_clipboard(text) {
            Ok(how) => {
                let n = text.chars().count();
                self.output_lines.push(OutputLine {
                    type_: "system".into(),
                    content: format!(
                        "Copied last assistant message ({n} chars via {how}). Tip: Shift+drag selects in the TUI; /select (Ctrl+O) for plain-text view."
                    ),
                    tool_name: String::new(),
                    duration: String::new(),
                });
            }
            Err(e) => {
                self.output_lines.push(OutputLine {
                    type_: "system".into(),
                    content: format!(
                        "Copy failed: {e}. Try Shift+drag to select, or /select (Ctrl+O)."
                    ),
                    tool_name: String::new(),
                    duration: String::new(),
                });
            }
        }
    }

    fn last_assistant_text(&self) -> Option<String> {
        if let Some(l) = self
            .output_lines
            .iter()
            .rev()
            .find(|l| l.type_ == "text" && !l.content.trim().is_empty())
        {
            return Some(l.content.clone());
        }
        if !self.streaming_text.trim().is_empty() {
            return Some(self.streaming_text.clone());
        }
        None
    }

    pub(super) fn select_dump_text(&self, kind: SelectDump) -> String {
        match kind {
            SelectDump::LastAssistant => self
                .last_assistant_text()
                .unwrap_or_else(|| "(no assistant message yet)".into()),
            SelectDump::FullTranscript => {
                let mut out = String::new();
                for line in &self.output_lines {
                    match line.type_.as_str() {
                        "user" => {
                            out.push_str("› ");
                            out.push_str(&line.content);
                            out.push_str("\n\n");
                        }
                        "text" => {
                            out.push_str(&line.content);
                            out.push_str("\n\n");
                        }
                        "thinking" => {
                            out.push_str("── Thinking ──\n");
                            out.push_str(&line.content);
                            out.push_str("\n\n");
                        }
                        "thinking_summary" => {
                            out.push_str("✦ ");
                            out.push_str(&line.content);
                            out.push('\n');
                        }
                        "tool_use" => {
                            out.push_str(&format!("● {} {}\n", line.tool_name, line.content));
                        }
                        "tool_result" => {
                            out.push_str(&format!(
                                "● {} result:\n{}\n",
                                line.tool_name, line.content
                            ));
                        }
                        "error" => {
                            out.push_str("Error: ");
                            out.push_str(&line.content);
                            out.push('\n');
                        }
                        "system" => {
                            out.push_str(&line.content);
                            out.push('\n');
                        }
                        _ => {
                            out.push_str(&line.content);
                            out.push('\n');
                        }
                    }
                }
                if !self.streaming_text.is_empty() {
                    out.push_str(&self.streaming_text);
                    out.push('\n');
                }
                if out.trim().is_empty() {
                    "(empty session)".into()
                } else {
                    out
                }
            }
        }
    }

    /// Platform paste chord for clipboard images:
    /// - Windows: Alt+V
    /// - Linux / other Unix: Ctrl+V
    /// - macOS: Cmd (SUPER)+V
    pub(super) fn is_image_paste_chord(&self, key: ratatui::crossterm::event::KeyEvent) -> bool {
        if !matches!(key.code, KeyCode::Char('v') | KeyCode::Char('V')) {
            return false;
        }
        #[cfg(windows)]
        {
            return key.modifiers.contains(KeyModifiers::ALT)
                && !key.modifiers.contains(KeyModifiers::CONTROL);
        }
        #[cfg(target_os = "macos")]
        {
            return key.modifiers.contains(KeyModifiers::SUPER);
        }
        #[cfg(all(unix, not(target_os = "macos")))]
        {
            return key.modifiers.contains(KeyModifiers::CONTROL)
                && !key.modifiers.contains(KeyModifiers::ALT);
        }
        #[cfg(not(any(windows, unix)))]
        {
            let _ = key;
            false
        }
    }

    pub(super) fn paste_clipboard_image(&mut self) {
        if self.awaiting_api_key
            || self.show_model_picker
            || self.show_provider_picker
            || self.show_theme_picker
            || self.show_session_picker
            || self.show_permission_prompt
            || self.show_recovery_prompt
            || self.show_help
            || self.show_shortcuts
            || self.confirm_remove_provider.is_some()
            || self.confirm_history_provider.is_some()
        {
            return;
        }
        if !matches!(self.state, State::Idle) {
            self.output_lines.push(OutputLine {
                type_: "system".into(),
                content: "Wait for the current turn to finish before pasting an image.".into(),
                tool_name: String::new(),
                duration: String::new(),
            });
            return;
        }
        const MAX_PENDING_IMAGES: usize = 4;
        if self.pending_images.len() >= MAX_PENDING_IMAGES {
            self.output_lines.push(OutputLine {
                type_: "error".into(),
                content: format!(
                    "At most {MAX_PENDING_IMAGES} images can be attached to one message."
                ),
                tool_name: String::new(),
                duration: String::new(),
            });
            return;
        }
        match crate::clipboard_image::read_clipboard_image() {
            Ok(img) => {
                let media = img.media_type.clone();
                let bytes = img.bytes.len();
                self.pending_images.push(img.into_block());
                self.output_lines.push(OutputLine {
                    type_: "system".into(),
                    content: format!(
                        "Attached clipboard image ({media}, {bytes} bytes). Add a caption in the composer and press Enter - or paste more images."
                    ),
                    tool_name: String::new(),
                    duration: String::new(),
                });
            }
            Err(e) => {
                self.output_lines.push(OutputLine {
                    type_: "error".into(),
                    content: format!("Clipboard image paste failed: {e}"),
                    tool_name: String::new(),
                    duration: String::new(),
                });
            }
        }
    }
}

/// Leave the ratatui alt-screen and print plain text so the host terminal can
/// drag-select (Windows Terminal does not reliably select inside a redrawing TUI).
pub(super) fn enter_plain_select_mode(
    terminal: &mut DefaultTerminal,
    text: &str,
) -> Result<(), String> {
    // Drop mouse capture / paste mode and leave alt-screen / raw mode.
    let _ = execute!(stdout(), DisableMouseCapture);
    let _ = execute!(stdout(), DisableBracketedPaste);
    // restore() disables raw mode and leaves the alternate screen.
    ratatui::restore();

    let mut out = stdout();
    let _ = writeln!(
        out,
        "\n======== Cairn select mode ========\n\
         Drag to highlight, then copy (Ctrl+Shift+C in Windows Terminal).\n\
         Press Enter to return to Cairn.\n\
         ==================================\n"
    );
    let text = sanitize_terminal_output(text);
    let _ = writeln!(out, "{text}");
    let _ = writeln!(out, "\n======== end - press Enter to return ========\n");
    let _ = out.flush();

    // stdin is cooked again after disable_raw_mode; block until Enter.
    let mut line = String::new();
    let _ = io::stdin().read_line(&mut line);

    *terminal = ratatui::init();
    terminal.clear().map_err(|e| e.to_string())?;
    Ok(())
}

/// Best-effort clipboard write: Windows PowerShell first, then OSC 52.
fn copy_text_to_clipboard(text: &str) -> Result<&'static str, String> {
    #[cfg(windows)]
    {
        if copy_text_windows_clipboard(text).is_ok() {
            return Ok("Windows clipboard");
        }
    }
    copy_text_osc52(text)?;
    Ok("OSC 52")
}

#[cfg(windows)]
fn copy_text_windows_clipboard(text: &str) -> Result<(), String> {
    use std::process::{Command, Stdio};
    // Temp UTF-8 file avoids command-line length limits and quoting issues.
    let path = std::env::temp_dir().join(format!("cairn-clip-{}.txt", std::process::id()));
    std::fs::write(&path, text).map_err(|e| format!("write temp clip file: {e}"))?;
    let path_str = path.to_string_lossy().replace('\'', "''");
    let status = Command::new("powershell.exe")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            &format!("Get-Content -LiteralPath '{path_str}' -Raw -Encoding utf8 | Set-Clipboard"),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .status()
        .map_err(|e| format!("powershell: {e}"))?;
    let _ = std::fs::remove_file(&path);
    if status.success() {
        Ok(())
    } else {
        Err(format!("Set-Clipboard exited {status}"))
    }
}

/// OSC 52 clipboard write (base64 payload). No external crate.
fn copy_text_osc52(text: &str) -> Result<(), String> {
    let b64 = base64_encode(text.as_bytes());
    // BEL-terminated form is widely supported (Windows Terminal, iTerm2, kitty, …).
    let seq = format!("\x1b]52;c;{b64}\x07");
    let mut out = stdout();
    out.write_all(seq.as_bytes())
        .map_err(|e| format!("write OSC 52: {e}"))?;
    out.flush().map_err(|e| format!("flush OSC 52: {e}"))
}

fn base64_encode(data: &[u8]) -> String {
    const T: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    let mut i = 0;
    while i + 3 <= data.len() {
        let n = ((data[i] as u32) << 16) | ((data[i + 1] as u32) << 8) | (data[i + 2] as u32);
        out.push(T[((n >> 18) & 63) as usize] as char);
        out.push(T[((n >> 12) & 63) as usize] as char);
        out.push(T[((n >> 6) & 63) as usize] as char);
        out.push(T[(n & 63) as usize] as char);
        i += 3;
    }
    match data.len() - i {
        1 => {
            let n = (data[i] as u32) << 16;
            out.push(T[((n >> 18) & 63) as usize] as char);
            out.push(T[((n >> 12) & 63) as usize] as char);
            out.push('=');
            out.push('=');
        }
        2 => {
            let n = ((data[i] as u32) << 16) | ((data[i + 1] as u32) << 8);
            out.push(T[((n >> 18) & 63) as usize] as char);
            out.push(T[((n >> 12) & 63) as usize] as char);
            out.push(T[((n >> 6) & 63) as usize] as char);
            out.push('=');
        }
        _ => {}
    }
    out
}

#[cfg(test)]
mod clipboard_tests {
    use super::*;

    #[test]
    fn base64_encode_known_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn encode_user_json_cmd_roundtrips_shape() {
        let cmd = encode_user_json_cmd(
            "see this",
            &[llm::ImageBlock {
                media_type: "image/png".into(),
                data_base64: "Zm9v".into(),
            }],
        );
        assert!(cmd.starts_with("__user_json__:"));
        let json = cmd.trim_start_matches("__user_json__:");
        let v: serde_json::Value = serde_json::from_str(json).unwrap();
        assert_eq!(v["text"], "see this");
        assert_eq!(v["images"][0]["media_type"], "image/png");
        assert_eq!(v["images"][0]["data"], "Zm9v");
    }
}
