//! Slash-command dispatch, pickers, and preference persistence.

use super::*;

impl Tui {
    pub(super) fn handle_command(&mut self, cmd: &str) -> bool {
        let parts: Vec<&str> = cmd.split_whitespace().collect();
        if parts.is_empty() {
            return true;
        }

        match parts[0] {
            "/clear" => {
                if !matches!(self.state, State::Idle) {
                    self.output_lines.push(OutputLine {
                        type_: "system".into(),
                        content: "Wait for the current turn to finish before clearing.".into(),
                        tool_name: String::new(),
                        duration: String::new(),
                    });
                    return true;
                }
                self.autosave_session(false);
                self.output_lines.clear();
                self.streaming_text.clear();
                self.stream_thinking.clear();
                self.thinking_started = None;
                self.current_session_id = None;
                self.session_created_at = 0;
                self.pending_images.clear();
                self.last_checkpoint_save = None;
                self.total_usage = llm::Usage::default();
                if let Some(tx) = &self.agent_tx {
                    let _ = tx.send("__clear__".to_string());
                }
                if let Some(mirror) = &self.live_mirror {
                    if let Ok(mut g) = mirror.lock() {
                        g.messages.clear();
                        g.tokens_in = 0;
                        g.tokens_out = 0;
                    }
                }
                self.output_lines.push(OutputLine {
                    type_: "system".into(),
                    content: "Cleared conversation and session state.".into(),
                    tool_name: String::new(),
                    duration: String::new(),
                });
            }
            "/thinking" => {
                let arg = parts.get(1).map(|s| s.to_ascii_lowercase());
                let next = match arg.as_deref() {
                    Some("on" | "true" | "1" | "show") => true,
                    Some("off" | "false" | "0" | "hide") => false,
                    Some(other) => {
                        self.output_lines.push(OutputLine {
                            type_: "system".into(),
                            content: format!(
                                "Unknown /thinking option '{other}'. Use /thinking, /thinking on, or /thinking off."
                            ),
                            tool_name: String::new(),
                            duration: String::new(),
                        });
                        return true;
                    }
                    None => !self.show_thinking,
                };
                self.apply_thinking_preference(next, crate::config::save_show_thinking(next));
            }
            "/suggestions" => {
                let arg = parts.get(1).map(|s| s.to_ascii_lowercase());
                let next = match arg.as_deref() {
                    Some("on" | "true" | "1" | "show" | "enable") => true,
                    Some("off" | "false" | "0" | "hide" | "disable") => false,
                    Some(other) => {
                        self.output_lines.push(OutputLine {
                            type_: "system".into(),
                            content: format!(
                                "Unknown /suggestions option '{other}'. Use /suggestions, /suggestions on, or /suggestions off."
                            ),
                            tool_name: String::new(),
                            duration: String::new(),
                        });
                        return true;
                    }
                    None => !self.show_suggestions,
                };
                self.apply_suggestions_preference(next, crate::config::save_show_suggestions(next));
            }
            "/mouse" => {
                let arg = parts.get(1).map(|s| s.to_ascii_lowercase());
                let next = match arg.as_deref() {
                    Some("on" | "true" | "1" | "enable") => true,
                    Some("off" | "false" | "0" | "disable") => false,
                    Some(other) => {
                        self.output_lines.push(OutputLine {
                            type_: "system".into(),
                            content: format!(
                                "Unknown /mouse option '{other}'. Use /mouse, /mouse on, or /mouse off."
                            ),
                            tool_name: String::new(),
                            duration: String::new(),
                        });
                        return true;
                    }
                    None => !self.mouse_capture,
                };
                self.set_mouse_capture(next);
                let state = if self.mouse_capture { "on" } else { "off" };
                let detail = if self.mouse_capture {
                    "Wheel scrolls the transcript. Shift+drag to select and copy (terminal-native)."
                } else {
                    "Mouse capture off. Select with a normal drag if the host allows it; scroll with PgUp/PgDn or Ctrl+U/D."
                };
                self.output_lines.push(OutputLine {
                    type_: "system".into(),
                    content: format!("Mouse capture: {state}. {detail}"),
                    tool_name: String::new(),
                    duration: String::new(),
                });
            }
            "/copy" => {
                self.copy_last_assistant_to_clipboard();
            }
            "/select" => {
                let kind = match parts.get(1).map(|s| s.to_ascii_lowercase()).as_deref() {
                    Some("all" | "full" | "session") => SelectDump::FullTranscript,
                    Some("last" | "reply" | "assistant") | None => SelectDump::LastAssistant,
                    Some(other) => {
                        self.output_lines.push(OutputLine {
                            type_: "system".into(),
                            content: format!(
                                "Unknown /select option '{other}'. Use /select, /select last, or /select all."
                            ),
                            tool_name: String::new(),
                            duration: String::new(),
                        });
                        return true;
                    }
                };
                self.pending_select = Some(kind);
            }
            "/model" => {
                if parts.len() > 1 {
                    self.finish_provider_model_selection(
                        self.provider.clone(),
                        parts[1..].join(" "),
                    );
                } else {
                    self.open_model_picker();
                }
            }
            "/cost" => {
                let est = crate::cost::estimate_cost(&self.model, &self.total_usage);
                let cost_str = crate::cost::format_cost(est);
                self.output_lines.push(OutputLine {
                    type_: "system".into(),
                    content: format!(
                        "Tokens: {} in, {} out  •  {}\nModel: {}\nEstimated cost: {}",
                        self.total_usage.input_tokens,
                        self.total_usage.output_tokens,
                        self.total_usage.cache_read + self.total_usage.cache_create,
                        self.model,
                        cost_str
                    ),
                    tool_name: String::new(),
                    duration: String::new(),
                });
            }
            "/provider" => {
                if parts.len() > 1 {
                    let name = parts[1].to_ascii_lowercase();
                    let providers = crate::llm::default_providers();
                    if providers.contains_key(&name) {
                        if self.provider_switch_needs_confirmation(&name) {
                            self.confirm_history_provider = Some(name);
                            self.confirm_history_sel = 0;
                        } else {
                            self.begin_provider_selection(name);
                        }
                    } else {
                        self.output_lines.push(OutputLine {
                            type_: "system".into(),
                            content: format!(
                                "Unknown provider '{name}'. Use /provider to pick from the list."
                            ),
                            tool_name: String::new(),
                            duration: String::new(),
                        });
                    }
                } else {
                    self.open_provider_picker();
                }
            }
            "/help" => {
                // Model-picker style overlay in bottom chrome; Esc / Enter dismisses.
                // Do not dump into the transcript (stays until /clear).
                self.show_shortcuts = false;
                self.show_help = true;
            }
            "/skills" => {
                let cfg = match crate::config::Config::load() {
                    Ok(cfg) => cfg,
                    Err(error) => {
                        self.output_lines.push(OutputLine {
                            type_: "error".into(),
                            content: format!("Error loading configuration: {error}"),
                            tool_name: String::new(),
                            duration: String::new(),
                        });
                        return true;
                    }
                };
                let dir = cfg
                    .skills_dir
                    .as_ref()
                    .map(std::path::PathBuf::from)
                    .unwrap_or_else(crate::skills::default_skills_dir);
                let mut roots = vec![dir.clone()];
                if let Some(a) = crate::skills::agents_skills_dir() {
                    roots.push(a);
                }
                let list = crate::skills::load_from_roots(&roots);
                if list.is_empty() {
                    self.output_lines.push(OutputLine {
                        type_: "system".into(),
                        content: format!(
                            "No skills found. Add packs as {}/<name>/SKILL.md (or set CAIRN_SKILLS_DIR).",
                            dir.display()
                        ),
                        tool_name: String::new(),
                        duration: String::new(),
                    });
                } else {
                    let mut body = format!("Skills ({}) from {}:\n", list.len(), dir.display());
                    for s in &list {
                        body.push_str(&format!("  {} - {}\n", s.name, s.description));
                    }
                    body.push_str("Load in-chat with the skill tool: {\"name\":\"...\"}");
                    self.output_lines.push(OutputLine {
                        type_: "system".into(),
                        content: body,
                        tool_name: String::new(),
                        duration: String::new(),
                    });
                }
            }
            "/mcp" => {
                let cfg = match crate::config::Config::load() {
                    Ok(cfg) => cfg,
                    Err(error) => {
                        self.output_lines.push(OutputLine {
                            type_: "error".into(),
                            content: format!("Error loading configuration: {error}"),
                            tool_name: String::new(),
                            duration: String::new(),
                        });
                        return true;
                    }
                };
                if cfg.mcp.servers.is_empty() {
                    self.output_lines.push(OutputLine {
                        type_: "system".into(),
                        content: format!(
                            "No MCP servers in config. Add mcp.servers (or mcpServers) to {}.",
                            crate::config::config_path()
                                .map(|p| p.display().to_string())
                                .unwrap_or_else(|| "~/.config/cairn-code/config.json".into())
                        ),
                        tool_name: String::new(),
                        duration: String::new(),
                    });
                } else {
                    let mut body = String::from("Configured MCP servers:\n");
                    let mut names: Vec<_> = cfg.mcp.servers.keys().cloned().collect();
                    names.sort();
                    for n in names {
                        let s = &cfg.mcp.servers[&n];
                        let state = if s.disabled { "disabled" } else { "enabled" };
                        let args_str = if s.args.is_empty() {
                            String::new()
                        } else {
                            format!(" {}", crate::redact::redact_secrets(&s.args.join(" ")))
                        };
                        body.push_str(&format!("  {n} [{state}] - {}{args_str}\n", s.command));
                    }
                    body.push_str(
                        "Tools register at startup as mcp_<server>_<tool> (permission required).",
                    );
                    self.output_lines.push(OutputLine {
                        type_: "system".into(),
                        content: body,
                        tool_name: String::new(),
                        duration: String::new(),
                    });
                }
            }
            "/auth" => {
                let sub = parts.get(1).copied().unwrap_or("status");
                match sub {
                    "login" => {
                        let provider = parts.get(2).copied().unwrap_or("xai").to_ascii_lowercase();
                        self.begin_oauth_login(&provider, false);
                    }
                    "key" => {
                        // Escape hatch: paste API key instead of OAuth (xAI / others).
                        let provider = parts.get(2).copied().unwrap_or("xai").to_ascii_lowercase();
                        if crate::config::provider_requires_api_key(&provider) {
                            self.begin_api_key_prompt(&provider);
                        } else {
                            self.output_lines.push(OutputLine {
                                type_: "system".into(),
                                content: format!(
                                    "Provider '{provider}' does not use a cloud API key."
                                ),
                                tool_name: String::new(),
                                duration: String::new(),
                            });
                        }
                    }
                    "logout" => {
                        let provider = parts.get(2).copied().unwrap_or("xai").to_ascii_lowercase();
                        if self.agent_tx.is_some() {
                            self.begin_running();
                            if let Some(tx) = &self.agent_tx {
                                let _ = tx.send(format!("__auth_logout__:{provider}"));
                            }
                        }
                    }
                    "status" | _ => {
                        if self.agent_tx.is_some() {
                            self.begin_running();
                            if let Some(tx) = &self.agent_tx {
                                let _ = tx.send("__auth_status__".into());
                            }
                        }
                    }
                }
            }
            "/theme" => {
                if parts.get(1) == Some(&"list") {
                    let names = theme::theme_names().join(", ");
                    self.output_lines.push(OutputLine {
                        type_: "system".into(),
                        content: format!("Active theme: {}\nThemes: {names}", self.theme.name),
                        tool_name: String::new(),
                        duration: String::new(),
                    });
                } else if parts.len() > 1 {
                    let name = parts[1..].join("-");
                    match theme::lookup_exact(&name) {
                        Some(t) => {
                            let applied = t.name.to_string();
                            self.apply_theme_preference(
                                t,
                                self.theme.clone(),
                                crate::config::save_theme(&applied),
                            );
                        }
                        None => {
                            let names = theme::theme_names().join(", ");
                            self.output_lines.push(OutputLine {
                                type_: "error".into(),
                                content: format!("Unknown theme '{name}'. Themes: {names}"),
                                tool_name: String::new(),
                                duration: String::new(),
                            });
                        }
                    }
                } else {
                    self.open_theme_picker();
                }
            }
            "/reset" => {
                // ChatGPT subscription banked rate-limit resets (Codex-compatible API).
                // Only works with OpenAI ChatGPT OAuth (Codex auth.json or oauth:openai),
                // not with plain API keys.
                let args: Vec<&str> = parts.iter().skip(1).copied().collect();
                match crate::openai_reset::run_reset_command(&args) {
                    Ok(msg) => {
                        self.output_lines.push(OutputLine {
                            type_: "system".into(),
                            content: msg,
                            tool_name: String::new(),
                            duration: String::new(),
                        });
                    }
                    Err(e) => {
                        self.output_lines.push(OutputLine {
                            type_: "error".into(),
                            content: e,
                            tool_name: String::new(),
                            duration: String::new(),
                        });
                    }
                }
            }
            "/compact" => {
                if !matches!(self.state, State::Idle) {
                    self.output_lines.push(OutputLine {
                        type_: "system".into(),
                        content: "Wait for the current turn to finish before compacting.".into(),
                        tool_name: String::new(),
                        duration: String::new(),
                    });
                } else if self.agent_tx.is_some() {
                    self.begin_running();
                    if let Some(tx) = &self.agent_tx {
                        let _ = tx.send("__compact__".into());
                    }
                }
            }
            "/save" => {
                self.save_session();
            }
            "/sessions" => {
                self.list_sessions();
            }
            "/delete" => {
                if parts.len() > 1 {
                    let query = parts[1..].join(" ");
                    match session::resolve_id(&self.sessions_dir(), &query) {
                        Ok(id) => self.delete_session(&id),
                        Err(e) => {
                            self.output_lines.push(OutputLine {
                                type_: "error".into(),
                                content: e,
                                tool_name: String::new(),
                                duration: String::new(),
                            });
                        }
                    }
                } else {
                    let sessions = session::list(&self.sessions_dir()).unwrap_or_default();
                    if sessions.is_empty() {
                        self.output_lines.push(OutputLine {
                            type_: "system".into(),
                            content: "No saved sessions to delete.".into(),
                            tool_name: String::new(),
                            duration: String::new(),
                        });
                    } else {
                        self.show_session_picker = true;
                        self.session_picker_delete = true;
                        self.picker_sessions = sessions;
                        self.picker_session_sel = 0;
                        self.picker_session_scrl = 0;
                    }
                }
            }
            "/resume" => {
                let sessions = session::list(&self.sessions_dir()).unwrap_or_default();
                if sessions.is_empty() {
                    self.output_lines.push(OutputLine {
                        type_: "system".into(),
                        content:
                            "No saved sessions. Use /save to save the current conversation first."
                                .into(),
                        tool_name: String::new(),
                        duration: String::new(),
                    });
                } else {
                    self.show_session_picker = true;
                    self.session_picker_delete = false;
                    self.picker_sessions = sessions;
                    self.picker_session_sel = 0;
                    self.picker_session_scrl = 0;
                }
            }
            "/exit" | "/quit" | "/q" => {
                return false;
            }
            _ => {
                self.output_lines.push(OutputLine {
                    type_: "error".into(),
                    content: format!("Unknown command: {} (type /help)", parts[0]),
                    tool_name: String::new(),
                    duration: String::new(),
                });
            }
        }
        true
    }

    pub(super) fn sessions_dir(&self) -> String {
        crate::config::sessions_dir()
    }

    pub(super) fn open_model_picker(&mut self) {
        let providers = crate::llm::default_providers();
        let provider_name = self
            .pending_provider_selection
            .as_deref()
            .unwrap_or(&self.provider);
        if let Some(p) = providers.get(provider_name) {
            self.picker_models = p.available_models();
            let selected_model = if self.pending_provider_selection.is_some() {
                if provider_name == "openrouter" {
                    "gpt-5-mini"
                } else {
                    p.default_model()
                }
            } else {
                &self.model
            };
            self.picker_sel = self
                .picker_models
                .iter()
                .position(|m| m.id == selected_model)
                .unwrap_or(0);
        }
        self.show_model_picker = true;
        let vh = self.picker_visible_height();
        self.picker_scrl = self.picker_sel.saturating_sub(vh.saturating_sub(1));
    }

    pub(super) fn open_provider_picker(&mut self) {
        let providers = crate::llm::default_providers();
        let mut names: Vec<String> = providers.into_keys().collect();
        names.sort();
        // Current provider first, matching render order so the
        // selection index always points at the displayed row.
        names.sort_by_key(|n| usize::from(*n != self.provider));
        self.provider_picker_keys = names
            .iter()
            .map(|n| crate::config::has_usable_credential(n))
            .collect();
        self.provider_picker_list = names;
        self.provider_picker_sel = 0;
        self.show_provider_picker = true;
    }

    pub(super) fn provider_switch_needs_confirmation(&self, provider: &str) -> bool {
        if provider == self.provider {
            return false;
        }
        if let Some(mirror) = &self.live_mirror {
            if let Ok(snapshot) = mirror.lock() {
                if !snapshot.messages.is_empty() {
                    return true;
                }
            }
        }
        self.output_lines.iter().any(|line| {
            matches!(
                line.type_.as_str(),
                "user" | "text" | "tool_use" | "tool_result"
            )
        })
    }

    pub(super) fn begin_provider_selection(&mut self, name: String) {
        let providers = crate::llm::default_providers();
        if !providers.contains_key(&name) {
            return;
        }
        self.pending_provider_selection = (name != self.provider).then(|| name.clone());
        // Auth first (browser OAuth or API key), then model list. Live
        // catalogs need credentials; model-before-login was backwards.
        if crate::config::needs_credential(&name) {
            self.pending_model_after_auth = true;
            if crate::oauth::supports_oauth(&name) {
                self.begin_oauth_login(&name, true);
            } else {
                self.begin_api_key_prompt(&name);
            }
        } else {
            self.open_model_picker();
        }
    }

    pub(super) fn cancel_pending_provider_selection(&mut self) {
        self.pending_model_after_auth = false;
        self.pending_provider_selection = None;
    }

    fn open_theme_picker(&mut self) {
        self.theme_before_picker = Some(self.theme.name.to_string());
        self.theme_picker_list = theme::all_themes();
        self.theme_picker_sel = self
            .theme_picker_list
            .iter()
            .position(|t| t.name == self.theme.name)
            .unwrap_or(0);
        // Live-preview current selection immediately
        if let Some(t) = self.theme_picker_list.get(self.theme_picker_sel) {
            self.theme = t.clone();
        }
        self.show_theme_picker = true;
    }

    pub(super) fn begin_api_key_prompt(&mut self, provider: &str) {
        self.awaiting_api_key = true;
        self.api_key_target = Some(provider.to_string());
        self.input_buf.clear();
        self.cursor = 0;
        let env = crate::config::env_var_name(provider).unwrap_or("API_KEY");
        let oauth_hint = if crate::oauth::supports_oauth(provider) {
            " Prefer browser login: Esc, then `/auth login xai`."
        } else {
            ""
        };
        self.output_lines.push(OutputLine {
            type_: "system".into(),
            content: format!(
                "Enter API key for {provider} (saved to OS keyring, env {env}). Input is masked.{oauth_hint}"
            ),
            tool_name: String::new(),
            duration: String::new(),
        });
    }

    /// Start device-code OAuth (browser) for a provider, like zero / Grok Build.
    /// When `then_model_picker` is true (provider path), successful login opens
    /// the model list; on failure, falls back to API key paste then model list.
    fn begin_oauth_login(&mut self, provider: &str, then_model_picker: bool) {
        if !crate::oauth::supports_oauth(provider) {
            self.output_lines.push(OutputLine {
                type_: "system".into(),
                content: format!(
                    "OAuth login is not implemented for '{provider}'. Supported: xai. Use an API key via /auth key {provider}."
                ),
                tool_name: String::new(),
                duration: String::new(),
            });
            return;
        }
        if !matches!(self.state, State::Idle) {
            self.output_lines.push(OutputLine {
                type_: "system".into(),
                content: "Wait for the current turn to finish before logging in.".into(),
                tool_name: String::new(),
                duration: String::new(),
            });
            return;
        }
        if self.agent_tx.is_none() {
            self.output_lines.push(OutputLine {
                type_: "error".into(),
                content: "Agent channel not ready; cannot start OAuth.".into(),
                tool_name: String::new(),
                duration: String::new(),
            });
            return;
        }
        if let Some(flag) = &self.cancel_flag {
            flag.store(false, Ordering::Relaxed);
        }
        self.pending_model_after_auth = then_model_picker;
        self.begin_running();
        self.output_lines.push(OutputLine {
            type_: "system".into(),
            content: "Starting xAI browser OAuth (device code)… A browser window should open. Approve the code shown next, or open the URL manually.".into(),
            tool_name: String::new(),
            duration: String::new(),
        });
        if let Some(tx) = &self.agent_tx {
            let _ = tx.send(format!("__auth_login__:{provider}"));
        }
    }

    /// Finish provider/model selection and synchronize the Agent.
    pub(super) fn finish_provider_model_selection(&mut self, provider: String, model: String) {
        let result = Self::persist_provider_model(&provider, &model);
        self.apply_provider_model_selection(provider, model, result);
    }

    fn apply_provider_model_selection(
        &mut self,
        provider: String,
        model: String,
        result: Result<(), String>,
    ) {
        if !self.report_config_save(
            result,
            format!("Provider set to: {provider}\nModel set to: {model}"),
            "provider and model selection",
        ) {
            return;
        }
        self.provider = provider;
        self.model = model;
        if let Some(tx) = &self.agent_tx {
            let _ = tx.send(format!("__switch__:{}:{}", self.provider, self.model));
        }
    }

    #[cfg(not(test))]
    fn persist_provider_model(provider: &str, model: &str) -> Result<(), String> {
        crate::config::save_config(provider, model, None)
    }

    #[cfg(test)]
    fn persist_provider_model(_provider: &str, _model: &str) -> Result<(), String> {
        Ok(())
    }

    fn report_config_save(
        &mut self,
        result: Result<(), String>,
        success: String,
        context: &str,
    ) -> bool {
        let (type_, content, saved) = match result {
            Ok(()) => ("system", success, true),
            Err(error) => ("error", format!("Failed to save {context}: {error}"), false),
        };
        self.output_lines.push(OutputLine {
            type_: type_.into(),
            content,
            tool_name: String::new(),
            duration: String::new(),
        });
        saved
    }

    pub(super) fn apply_theme_preference(
        &mut self,
        selected: Theme,
        previous: Theme,
        result: Result<(), String>,
    ) {
        let name = selected.name;
        let label = selected.label;
        if self.report_config_save(
            result,
            format!("Theme set to: {label} ({name})"),
            "theme preference",
        ) {
            self.theme = selected;
        } else {
            self.theme = previous;
        }
    }

    fn apply_thinking_preference(&mut self, next: bool, result: Result<(), String>) {
        let state = if next { "on" } else { "off" };
        let detail = if next {
            "Full thinking streams and is kept in the transcript."
        } else {
            "Thinking is hidden; a short \"Thought for …\" line is kept (Claude Code default)."
        };
        if self.report_config_save(
            result,
            format!("Thinking display: {state}. {detail}"),
            "thinking preference",
        ) {
            self.show_thinking = next;
        }
    }

    fn apply_suggestions_preference(&mut self, next: bool, result: Result<(), String>) {
        let state = if next { "on" } else { "off" };
        let detail = if next {
            "Grayed ready-to-send prompts appear when the composer is empty (Tab/→ to accept)."
        } else {
            "Idle composer stays blank (default)."
        };
        if self.report_config_save(
            result,
            format!("Suggestions: {state}. {detail}"),
            "suggestions preference",
        ) {
            self.set_show_suggestions(next);
        }
    }
}

#[cfg(test)]
mod provider_privacy_tests {
    use super::*;
    use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn tui_with_history() -> Tui {
        let mut tui = Tui::new("test", "claude", "anthropic", ".");
        tui.output_lines.push(OutputLine {
            type_: "user".into(),
            content: "inspect private source".into(),
            tool_name: String::new(),
            duration: String::new(),
        });
        tui
    }

    #[test]
    fn cross_provider_switch_requires_confirmation_when_history_exists() {
        let mut tui = tui_with_history();
        tui.show_provider_picker = true;
        tui.provider_picker_list = vec!["ollama".into()];
        tui.provider_picker_sel = 0;

        tui.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));

        assert_eq!(tui.provider, "anthropic");
        assert_eq!(tui.confirm_history_provider.as_deref(), Some("ollama"));
        assert_eq!(
            tui.confirm_history_sel, 0,
            "confirmation must default to cancel"
        );

        tui.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(
            tui.provider, "anthropic",
            "default confirmation action must cancel"
        );
    }

    #[test]
    fn provider_command_uses_the_same_confirmation() {
        let mut tui = tui_with_history();

        tui.handle_command("/provider ollama");

        assert_eq!(tui.provider, "anthropic");
        assert_eq!(tui.confirm_history_provider.as_deref(), Some("ollama"));
    }

    #[test]
    fn confirmed_cross_provider_switch_proceeds() {
        let mut tui = tui_with_history();
        tui.show_provider_picker = true;
        tui.provider_picker_list = vec!["ollama".into()];
        tui.provider_picker_sel = 0;

        tui.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        tui.handle_key(KeyEvent::new(KeyCode::Right, KeyModifiers::NONE));
        tui.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));

        assert_eq!(
            tui.provider, "anthropic",
            "provider stays committed until model selection"
        );
        assert_eq!(tui.pending_provider_selection.as_deref(), Some("ollama"));
        assert!(tui.show_model_picker);

        tui.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert_eq!(tui.provider, "ollama");
        assert!(tui.pending_provider_selection.is_none());
    }

    #[test]
    fn same_provider_or_empty_history_does_not_require_confirmation() {
        let tui = tui_with_history();
        assert!(!tui.provider_switch_needs_confirmation("anthropic"));

        let empty = Tui::new("test", "claude", "anthropic", ".");
        assert!(!empty.provider_switch_needs_confirmation("ollama"));
    }

    #[test]
    fn selecting_current_provider_preserves_current_model_selection() {
        let mut tui = Tui::new("test", "mistral", "ollama", ".");

        tui.begin_provider_selection("ollama".into());

        assert!(tui.pending_provider_selection.is_none());
        assert!(tui.show_model_picker);
        assert_eq!(tui.picker_models[tui.picker_sel].id, "mistral");
    }

    #[test]
    fn cancelling_provider_model_picker_keeps_committed_provider_and_model() {
        let mut tui = tui_with_history();
        tui.show_provider_picker = true;
        tui.provider_picker_list = vec!["ollama".into()];

        tui.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        tui.handle_key(KeyEvent::new(KeyCode::Right, KeyModifiers::NONE));
        tui.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert!(tui.show_model_picker);

        tui.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));

        assert_eq!(tui.provider, "anthropic");
        assert_eq!(tui.model, "claude");
        assert!(tui.pending_provider_selection.is_none());
    }

    #[test]
    fn direct_model_command_switches_agent_model() {
        let mut tui = Tui::new("test", "old-model", "ollama", ".");
        let (tx, rx) = mpsc::channel();
        tui.agent_tx = Some(tx);

        tui.handle_command("/model new-model");

        assert_eq!(tui.model, "new-model");
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            "__switch__:ollama:new-model"
        );
    }

    #[test]
    fn provider_model_write_failure_does_not_switch_or_claim_success() {
        let mut tui = Tui::new("test", "old-model", "ollama", ".");
        let (tx, rx) = mpsc::channel();
        tui.agent_tx = Some(tx);

        tui.apply_provider_model_selection(
            "openai".into(),
            "new-model".into(),
            Err("disk full".into()),
        );

        assert_eq!(tui.provider, "ollama");
        assert_eq!(tui.model, "old-model");
        assert!(rx.try_recv().is_err());
        let line = tui.output_lines.last().unwrap();
        assert_eq!(line.type_, "error");
        assert!(line.content.contains("provider and model selection"));
        assert!(line.content.contains("disk full"));
        assert!(!line.content.contains("Provider set to"));
    }

    #[test]
    fn interrupt_waits_for_worker_done_before_becoming_idle() {
        let mut tui = Tui::new("test", "llama3.2", "ollama", ".");
        let cancelled = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel();
        tui.cancel_flag = Some(cancelled.clone());
        tui.agent_tx = Some(tx);
        tui.state = State::Running;

        assert!(tui.handle_ctrl_c());

        assert!(cancelled.load(Ordering::Relaxed));
        assert!(matches!(tui.state, State::Running));
        assert!(
            rx.try_recv().is_err(),
            "cancellation must not leave a stale command queued"
        );
    }
}

#[cfg(test)]
mod theme_command_tests {
    use super::*;

    #[test]
    fn unknown_theme_name_is_rejected_without_changing_the_active_theme() {
        let mut tui = Tui::new("test", "model", "provider", ".");
        let previous = tui.theme.name.to_string();

        tui.handle_command("/theme totally-bogus-theme");

        assert_eq!(tui.theme.name, previous);
        let line = tui.output_lines.last().unwrap();
        assert_eq!(line.type_, "error");
        assert!(line.content.contains("Unknown theme"), "{}", line.content);
    }
}

#[cfg(test)]
mod config_persistence_error_tests {
    use super::*;

    fn assert_last_save_error(tui: &Tui, context: &str) {
        let line = tui.output_lines.last().unwrap();
        assert_eq!(line.type_, "error");
        assert!(line.content.contains(context), "{}", line.content);
        assert!(
            line.content.contains("permission denied"),
            "{}",
            line.content
        );
        assert!(!line.content.contains(" set to:"));
    }

    #[test]
    fn theme_write_failure_restores_previous_theme_and_reports_error() {
        let mut tui = Tui::new("test", "model", "provider", ".");
        let previous = tui.theme.clone();
        let selected = theme::lookup("dune");
        tui.theme = selected.clone();

        tui.apply_theme_preference(selected, previous.clone(), Err("permission denied".into()));

        assert_eq!(tui.theme.name, previous.name);
        assert_last_save_error(&tui, "theme preference");
    }

    #[test]
    fn thinking_write_failure_preserves_state_and_reports_error() {
        let mut tui = Tui::new("test", "model", "provider", ".");
        assert!(!tui.show_thinking);

        tui.apply_thinking_preference(true, Err("permission denied".into()));

        assert!(!tui.show_thinking);
        assert_last_save_error(&tui, "thinking preference");
    }

    #[test]
    fn suggestions_write_failure_preserves_state_and_reports_error() {
        let mut tui = Tui::new("test", "model", "provider", ".");
        assert!(!tui.show_suggestions);

        tui.apply_suggestions_preference(true, Err("permission denied".into()));

        assert!(!tui.show_suggestions);
        assert_last_save_error(&tui, "suggestions preference");
    }
}

#[cfg(test)]
mod exit_tests {
    use super::*;
    use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    #[test]
    fn exit_commands_stop_the_event_loop() {
        for command in ["/exit", "/quit", "/q"] {
            let mut tui = Tui::new("test", "test-model", "test-provider", ".");
            tui.input_buf = command.into();
            tui.cursor = tui.input_buf.len();

            assert!(
                !tui.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
                "{command} should request normal event-loop termination"
            );
        }
    }
}

#[cfg(test)]
mod help_overlay_tests {
    use super::*;
    use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    #[test]
    fn help_command_opens_overlay_without_transcript_dump() {
        let mut tui = Tui::new("test", "model", "provider", ".");
        let before = tui.output_lines.len();
        tui.handle_command("/help");
        assert!(tui.show_help);
        assert_eq!(
            tui.output_lines.len(),
            before,
            "/help must not dump into the transcript"
        );
    }

    #[test]
    fn esc_and_enter_close_help_overlay() {
        let mut tui = Tui::new("test", "model", "provider", ".");
        tui.show_help = true;
        tui.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
        assert!(!tui.show_help);

        tui.show_help = true;
        tui.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert!(!tui.show_help);
    }
}
