//! Slash-command table, fuzzy matching, and composer completions.

use super::*;

impl Tui {
    pub(super) fn update_cmd_picker(&mut self) {
        if self.awaiting_api_key || !self.input_buf.starts_with('/') {
            self.show_command_picker = false;
            self.cmd_picker_filtered.clear();
            return;
        }

        let models: Vec<String> = {
            let providers = crate::llm::default_providers();
            providers
                .get(&self.provider)
                .map(|p| p.available_models().into_iter().map(|m| m.id).collect())
                .unwrap_or_default()
        };
        let mut provider_names: Vec<String> = crate::llm::default_providers().into_keys().collect();
        provider_names.sort();
        let themes: Vec<String> = theme::theme_names()
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        let sessions: Vec<String> = session::list(&self.sessions_dir())
            .unwrap_or_default()
            .into_iter()
            .map(|s| s.id.get(..8).unwrap_or(s.id.as_str()).to_string())
            .collect();

        self.cmd_picker_filtered = slash_completions(
            &self.input_buf,
            &self.cmd_picker_list,
            &models,
            &provider_names,
            &themes,
            &sessions,
        );
        if self.cmd_picker_sel >= self.cmd_picker_filtered.len() {
            self.cmd_picker_sel = self.cmd_picker_filtered.len().saturating_sub(1);
        }
        // Also gates mouse-wheel scrolling and Esc/Ctrl+C dismiss for the
        // visible Commands overlay rendered under the composer.
        self.show_command_picker = !self.cmd_picker_filtered.is_empty();
    }

    /// Selected slash completion string, if any.
    pub(super) fn selected_slash_completion(&self) -> Option<&str> {
        self.cmd_picker_filtered
            .get(self.cmd_picker_sel)
            .map(|s| s.as_str())
    }

    /// Refresh the grayed ready-to-send prompt in the empty composer.
    pub(super) fn refresh_idle_suggestion(&mut self) {
        if !self.show_suggestions {
            self.idle_suggestion = None;
            return;
        }
        if !self.input_buf.is_empty() || !matches!(self.state, State::Idle) {
            return;
        }
        // Permission/recovery own the chrome; no ghost prompt until they clear.
        if self.show_permission_prompt || self.show_recovery_prompt {
            self.idle_suggestion = None;
            return;
        }
        self.idle_suggestion = Some(compute_idle_suggestion(&self.output_lines));
    }

    /// Insert a completion into the composer. Adds a trailing space when more
    /// arguments are expected so the next Tab level can open immediately.
    pub(super) fn apply_slash_completion(&mut self, completion: &str) {
        let text = if completion_wants_trailing_space(completion) {
            format!("{completion} ")
        } else {
            completion.to_string()
        };
        self.input_buf = text;
        self.cursor = self.input_buf.len();
        self.update_cmd_picker();
    }
}

/// Gray ghost text after the typed prefix: `/e` + `/exit` → `xit`.
/// Returns `None` when the completion does not extend the current input.
pub(crate) fn slash_ghost_suffix(input: &str, completion: &str) -> Option<String> {
    if completion.len() <= input.len() {
        return None;
    }
    if !completion
        .to_ascii_lowercase()
        .starts_with(&input.to_ascii_lowercase())
    {
        return None;
    }
    Some(completion[input.len()..].to_string())
}

fn default_empty_suggestion() -> &'static str {
    "Give me an overview of this codebase and how it is organized"
}

/// Pick a short ready-to-send prompt from recent transcript context.
/// Tab/→ inserts it into the composer as a real user message.
pub(crate) fn compute_idle_suggestion(lines: &[OutputLine]) -> String {
    // Walk recent transcript for a contextual ready prompt.
    let mut last_user: Option<&str> = None;
    let mut last_assistant: Option<&str> = None;
    let mut last_tool: Option<&str> = None;
    let mut saw_error = false;
    for line in lines.iter().rev().take(40) {
        match line.type_.as_str() {
            "user" if last_user.is_none() => last_user = Some(line.content.as_str()),
            "text" if last_assistant.is_none() => last_assistant = Some(line.content.as_str()),
            "tool_use" if last_tool.is_none() => last_tool = Some(line.tool_name.as_str()),
            "error" => saw_error = true,
            _ => {}
        }
    }
    if saw_error {
        return "Retry the last step and fix any errors you hit".into();
    }
    if let Some(tool) = last_tool {
        if tool == "shell" {
            return "Summarize the command output and what we should do next".into();
        }
        if tool == "file_read" || tool == "grep" || tool == "glob" {
            return format!(
                "Based on the {tool} results, explain what you found and recommend next steps"
            );
        }
        return format!("Summarize the {tool} results and continue with the next step");
    }
    if let Some(u) = last_user {
        let lower = u.to_ascii_lowercase();
        if lower.contains("test") {
            return "Run the tests, fix any failures, and report the final result".into();
        }
        if lower.contains("commit") || lower.contains("push") {
            return "Review the git status and diff, then commit and push if the changes look good"
                .into();
        }
        if lower.contains("fix") || lower.contains("bug") {
            return "Verify the fix works end-to-end and check for related regressions".into();
        }
        if lower.contains("refactor") {
            return "Continue the refactor and keep behavior covered by tests".into();
        }
    }
    if last_assistant.is_some() {
        return "Continue with the next step from your previous plan".into();
    }
    default_empty_suggestion().into()
}

/// Commands that still need another argument after Tab.
fn completion_wants_trailing_space(completion: &str) -> bool {
    let c = completion.trim_end();
    matches!(
        c,
        "/auth"
            | "/auth login"
            | "/auth logout"
            | "/auth key"
            | "/reset"
            | "/theme"
            | "/model"
            | "/provider"
            | "/resume"
            | "/delete"
            | "/thinking"
            | "/suggestions"
            | "/mouse"
    )
}

/// Base slash commands paired with the one-line help shown in the composer
/// picker. This is the source of truth for `cmd_picker_list`, so a new command
/// only has to be added here to show up in completions and in the list.
pub(crate) const SLASH_COMMANDS: &[(&str, &str)] = &[
    ("/auth", "Manage provider credentials"),
    ("/clear", "Clear the conversation"),
    ("/compact", "Summarize the conversation to free context"),
    ("/copy", "Copy the last reply to the clipboard"),
    ("/cost", "Show token usage and estimated cost"),
    ("/delete", "Delete a saved session"),
    ("/exit", "Exit Cairn"),
    ("/help", "List commands and keybindings"),
    ("/mcp", "List configured MCP servers"),
    ("/model", "Switch the active model"),
    ("/mouse", "Toggle mouse capture"),
    ("/provider", "Switch the active provider"),
    ("/q", "Exit Cairn"),
    ("/quit", "Exit Cairn"),
    ("/reset", "Show ChatGPT rate-limit reset times"),
    ("/resume", "Resume a saved session"),
    ("/save", "Save the current session"),
    ("/select", "Plain-text view for terminal selection"),
    ("/sessions", "List saved sessions"),
    ("/skills", "List available skills"),
    ("/suggestions", "Toggle idle prompt suggestions"),
    ("/theme", "Change the color theme"),
    ("/thinking", "Toggle thinking output"),
];

/// One-line help for a picker row: the command's own description for a bare
/// command, otherwise a label for the argument being completed.
pub(crate) fn slash_completion_help(completion: &str) -> Option<&'static str> {
    let parts: Vec<&str> = completion.split_whitespace().collect();
    match parts.as_slice() {
        [root] => SLASH_COMMANDS
            .iter()
            .find(|(c, _)| *c == *root)
            .map(|(_, help)| *help),
        ["/auth", "login"] => Some("Sign in to a provider"),
        ["/auth", "logout"] => Some("Remove stored credentials"),
        ["/auth", "status"] => Some("Show credential status"),
        ["/auth", "key"] => Some("Paste an API key"),
        ["/theme", "list"] => Some("List theme names"),
        ["/reset", "list"] => Some("List banked rate-limit resets"),
        ["/reset", "apply"] => Some("Apply a banked rate-limit reset"),
        ["/reset", "status"] => Some("Show rate-limit reset status"),
        [_, "on"] => Some("Enable"),
        [_, "off"] => Some("Disable"),
        ["/auth", _, _] | ["/provider", _] => Some("provider"),
        ["/model", ..] => Some("model"),
        ["/theme", _] => Some("theme"),
        ["/resume", _] | ["/delete", _] => Some("session id"),
        _ => None,
    }
}

/// Fuzzy-match `query` against `candidate`, case-insensitively.
///
/// Returns `None` unless every character of `query` appears in `candidate` in
/// order, so `/mdl` finds `/model` but `/xyz` finds nothing. The score only has
/// to order candidates against each other, not mean anything on its own: real
/// prefixes rank above scattered matches, and consecutive or word-boundary hits
/// rank above hits buried mid-word.
pub(crate) fn fuzzy_score(candidate: &str, query: &str) -> Option<i32> {
    let cand: Vec<char> = candidate.to_ascii_lowercase().chars().collect();
    let q: Vec<char> = query.to_ascii_lowercase().chars().collect();
    if q.is_empty() {
        return Some(0);
    }
    if q.len() > cand.len() {
        return None;
    }

    let mut score = 0i32;
    let mut next = 0usize;
    let mut prev_hit: Option<usize> = None;
    for &qc in &q {
        let idx = (next..cand.len()).find(|&i| cand[i] == qc)?;
        next = idx + 1;
        // Runs of adjacent characters are what "typing the start of a word"
        // looks like, so they weigh most.
        if prev_hit.is_some_and(|p| p + 1 == idx) {
            score += 8;
        }
        // Segment starts (`/auth`, `grok-4.5`, `claude_x`) are strong anchors.
        if idx == 0 || matches!(cand[idx - 1], '/' | '-' | '_' | '.' | ':' | ' ') {
            score += 6;
        }
        // Early hits beat late ones, but never by enough to outweigh an anchor.
        score -= (idx as i32).min(10);
        prev_hit = Some(idx);
    }
    // What the user typed verbatim is almost always what they meant.
    if cand.starts_with(&q) {
        score += 40;
    }
    // Between two matches the tighter one is the better guess - but only once
    // the query says something. A bare `/` matches everything equally well, and
    // length is then the only differing term, which would silently re-sort the
    // whole command list by name length.
    if q.len() > 1 {
        score -= ((cand.len() - q.len()) as i32).min(20);
    }
    Some(score)
}

/// Rank `candidates` by fuzzy match against `query`, dropping non-matches.
///
/// Equal scores keep the caller's original order, so a bare `/` still lists the
/// commands in the order they were declared.
fn fuzzy_rank<F: Fn(&str) -> String>(candidates: &[String], query: &str, format: F) -> Vec<String> {
    let mut scored: Vec<(i32, usize, &String)> = candidates
        .iter()
        .enumerate()
        .filter_map(|(i, c)| fuzzy_score(c, query).map(|s| (s, i, c)))
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut out: Vec<String> = scored.into_iter().map(|(_, _, c)| format(c)).collect();
    out.dedup();
    out
}

/// Contextual slash-command completions for the composer.
///
/// Supports base commands, `/auth` subcommands + providers, `/theme` names,
/// `/model` ids for the active provider, `/provider` names, and short session
/// ids for `/resume` / `/delete`. Matching is fuzzy throughout, best match first.
pub(crate) fn slash_completions(
    input: &str,
    base_commands: &[String],
    models: &[String],
    providers: &[String],
    themes: &[String],
    session_ids: &[String],
) -> Vec<String> {
    if !input.starts_with('/') {
        return Vec::new();
    }
    let ends_with_space = input.ends_with(' ') || input.ends_with('\t');
    let parts: Vec<&str> = input.split_whitespace().collect();
    if parts.is_empty() {
        return base_commands.to_vec();
    }

    let cmd = parts[0].to_ascii_lowercase();

    // Still typing the root command: `/mo` → `/model`, `/mdl` → `/model`
    if parts.len() == 1 && !ends_with_space {
        return fuzzy_rank(base_commands, &cmd, |c| c.to_string());
    }

    let rank_match = |candidates: &[String], typed: &str, format: &dyn Fn(&str) -> String| {
        fuzzy_rank(candidates, typed, format)
    };

    match cmd.as_str() {
        "/thinking" | "/suggestions" | "/mouse" => {
            let root = cmd.as_str();
            let opts = ["on", "off"];
            if parts.len() == 1 && ends_with_space {
                return opts.iter().map(|s| format!("{root} {s}")).collect();
            }
            if parts.len() == 2 && !ends_with_space {
                let p = parts[1].to_ascii_lowercase();
                return opts
                    .iter()
                    .filter(|s| s.starts_with(&p))
                    .map(|s| format!("{root} {s}"))
                    .collect();
            }
            Vec::new()
        }
        "/reset" => {
            let opts = ["list", "apply", "status"];
            if parts.len() == 1 && ends_with_space {
                return opts.iter().map(|s| format!("/reset {s}")).collect();
            }
            if parts.len() == 2 && !ends_with_space {
                let p = parts[1].to_ascii_lowercase();
                return opts
                    .iter()
                    .filter(|s| s.starts_with(&p))
                    .map(|s| format!("/reset {s}"))
                    .collect();
            }
            Vec::new()
        }
        "/auth" => {
            let subs = ["login", "logout", "status", "key"];
            if parts.len() == 1 && ends_with_space {
                return subs.iter().map(|s| format!("/auth {s}")).collect();
            }
            if parts.len() == 2 && !ends_with_space {
                let p = parts[1].to_ascii_lowercase();
                return subs
                    .iter()
                    .filter(|s| s.starts_with(&p))
                    .map(|s| format!("/auth {s}"))
                    .collect();
            }
            if parts.len() >= 2 {
                let action = parts[1].to_ascii_lowercase();
                if matches!(action.as_str(), "login" | "logout" | "key") {
                    if parts.len() == 2 && ends_with_space {
                        return providers
                            .iter()
                            .map(|p| format!("/auth {action} {p}"))
                            .collect();
                    }
                    if parts.len() == 3 && !ends_with_space {
                        return rank_match(providers, parts[2], &|p| format!("/auth {action} {p}"));
                    }
                }
            }
            Vec::new()
        }
        "/theme" => {
            if parts.len() == 1 && ends_with_space {
                let mut v: Vec<String> = themes.iter().map(|t| format!("/theme {t}")).collect();
                v.insert(0, "/theme list".into());
                return v;
            }
            if parts.len() == 2 && !ends_with_space {
                let mut v = rank_match(themes, parts[1], &|t| format!("/theme {t}"));
                if "list".starts_with(&parts[1].to_ascii_lowercase()) {
                    v.insert(0, "/theme list".into());
                }
                v.dedup();
                return v;
            }
            Vec::new()
        }
        "/model" => {
            if parts.len() == 1 && ends_with_space {
                return models.iter().map(|m| format!("/model {m}")).collect();
            }
            if parts.len() >= 2 && !ends_with_space {
                let typed = parts[1..].join(" ");
                return rank_match(models, &typed, &|m| format!("/model {m}"));
            }
            Vec::new()
        }
        "/provider" => {
            if parts.len() == 1 && ends_with_space {
                return providers.iter().map(|p| format!("/provider {p}")).collect();
            }
            if parts.len() == 2 && !ends_with_space {
                return rank_match(providers, parts[1], &|p| format!("/provider {p}"));
            }
            Vec::new()
        }
        "/resume" | "/delete" => {
            if parts.len() == 1 && ends_with_space {
                return session_ids.iter().map(|id| format!("{cmd} {id}")).collect();
            }
            if parts.len() == 2 && !ends_with_space {
                return rank_match(session_ids, parts[1], &|id| format!("{cmd} {id}"));
            }
            Vec::new()
        }
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod suggestion_tests {
    use super::*;

    pub(super) fn line(type_: &str, content: &str, tool: &str) -> OutputLine {
        OutputLine {
            type_: type_.into(),
            content: content.into(),
            tool_name: tool.into(),
            duration: String::new(),
        }
    }

    #[test]
    fn suggestion_from_recent_user_and_tools() {
        let lines = vec![
            line("user", "please run the tests", ""),
            line("tool_use", r#"{"command":"cargo test"}"#, "shell"),
            line("tool_result", "ok", "shell"),
        ];
        let s = compute_idle_suggestion(&lines);
        assert!(
            s.to_ascii_lowercase().contains("command output"),
            "expected ready prompt about shell output, got {s}"
        );
        // Ready prompt, not a meta instruction to the user.
        assert!(!s.to_ascii_lowercase().starts_with("inspect"));
        assert!(!s.to_ascii_lowercase().starts_with("review"));
        assert!(!s.to_ascii_lowercase().starts_with("try again"));
    }

    #[test]
    fn suggestion_default_when_empty() {
        let s = compute_idle_suggestion(&[]);
        assert_eq!(s, default_empty_suggestion());
        assert!(
            s.chars()
                .next()
                .map(|c| c.is_ascii_uppercase())
                .unwrap_or(false),
            "ready prompts should read as imperative agent requests"
        );
    }

    #[test]
    fn suggestion_after_error_is_sendable_retry() {
        let lines = vec![line("error", "LLM error: boom", "")];
        let s = compute_idle_suggestion(&lines);
        assert!(s.to_ascii_lowercase().contains("retry"));
        assert!(!s.contains("/provider"));
    }
}

#[cfg(test)]
mod completion_tests {
    use super::*;

    fn base() -> Vec<String> {
        vec![
            "/auth".into(),
            "/clear".into(),
            "/help".into(),
            "/model".into(),
            "/provider".into(),
            "/reset".into(),
            "/resume".into(),
            "/delete".into(),
            "/suggestions".into(),
            "/theme".into(),
            "/thinking".into(),
            "/mouse".into(),
            "/copy".into(),
            "/select".into(),
        ]
    }

    pub(super) fn all() -> Vec<String> {
        SLASH_COMMANDS.iter().map(|(c, _)| (*c).into()).collect()
    }

    #[test]
    fn fuzzy_matches_skipped_characters() {
        // The whole point: `/mdl` is not a prefix of anything.
        let c = slash_completions("/mdl", &base(), &[], &[], &[], &[]);
        assert_eq!(c, vec!["/model".to_string()]);
        let c = slash_completions("/thnk", &base(), &[], &[], &[], &[]);
        assert_eq!(c, vec!["/thinking".to_string()]);
    }

    #[test]
    fn fuzzy_rejects_out_of_order_and_absent_characters() {
        assert!(slash_completions("/ldom", &base(), &[], &[], &[], &[]).is_empty());
        assert!(slash_completions("/zzz", &base(), &[], &[], &[], &[]).is_empty());
    }

    #[test]
    fn tighter_matches_rank_first() {
        // /compact, /copy and /cost all prefix-match; the shortest wins.
        let c = slash_completions("/co", &all(), &[], &[], &[], &[]);
        assert_eq!(c.first().map(String::as_str), Some("/copy"), "{c:?}");
        assert!(c.contains(&"/compact".to_string()), "{c:?}");
    }

    #[test]
    fn fuzzy_ranks_arguments_too() {
        let models = vec!["claude-opus-5".into(), "grok-4.5".into()];
        let c = slash_completions("/model g45", &base(), &models, &[], &[], &[]);
        assert_eq!(c, vec!["/model grok-4.5".to_string()]);
    }

    #[test]
    fn fuzzy_score_orders_prefix_above_subsequence() {
        let prefix = fuzzy_score("/model", "/mod").unwrap();
        let scattered = fuzzy_score("/model", "/mdl").unwrap();
        assert!(prefix > scattered, "{prefix} vs {scattered}");
        assert_eq!(fuzzy_score("/model", "/xyz"), None);
        // An empty query matches everything, so a bare `/` opens the full list.
        assert_eq!(fuzzy_score("/model", ""), Some(0));
    }

    #[test]
    fn bare_slash_lists_every_command_in_declared_order() {
        let c = slash_completions("/", &all(), &[], &[], &[], &[]);
        assert_eq!(c, all());
    }

    #[test]
    fn every_command_has_help_text() {
        for (cmd, help) in SLASH_COMMANDS {
            assert!(!help.is_empty(), "{cmd} has no help text");
            assert_eq!(slash_completion_help(cmd), Some(*help));
        }
    }

    #[test]
    fn help_describes_argument_completions() {
        assert_eq!(
            slash_completion_help("/auth login"),
            Some("Sign in to a provider")
        );
        assert_eq!(slash_completion_help("/auth login xai"), Some("provider"));
        assert_eq!(slash_completion_help("/model grok-4.5"), Some("model"));
        assert_eq!(slash_completion_help("/theme dune"), Some("theme"));
        assert_eq!(
            slash_completion_help("/resume abc12345"),
            Some("session id")
        );
        assert_eq!(slash_completion_help("/thinking on"), Some("Enable"));
        assert_eq!(slash_completion_help("/mouse off"), Some("Disable"));
        assert_eq!(
            slash_completion_help("/reset list"),
            Some("List banked rate-limit resets")
        );
        assert_eq!(
            slash_completion_help("/reset apply"),
            Some("Apply a banked rate-limit reset")
        );
        assert_eq!(
            slash_completion_help("/reset status"),
            Some("Show rate-limit reset status")
        );
        assert_eq!(slash_completion_help("not a command"), None);
    }

    #[test]
    fn completes_suggestions_toggle() {
        let c = slash_completions("/sugg", &base(), &[], &[], &[], &[]);
        assert_eq!(c, vec!["/suggestions".to_string()]);
        let c = slash_completions("/suggestions ", &base(), &[], &[], &[], &[]);
        assert!(c.iter().any(|x| x == "/suggestions on"));
        assert!(c.iter().any(|x| x == "/suggestions off"));
    }

    #[test]
    fn completes_mouse_toggle() {
        let c = slash_completions("/mo", &base(), &[], &[], &[], &[]);
        // /model and /mouse both match /mo
        assert!(
            c.iter().any(|x| x == "/mouse") || c.iter().any(|x| x == "/model"),
            "{c:?}"
        );
        let c = slash_completions("/mouse ", &base(), &[], &[], &[], &[]);
        assert!(c.iter().any(|x| x == "/mouse on"));
        assert!(c.iter().any(|x| x == "/mouse off"));
    }

    #[test]
    fn thought_label_formats_duration() {
        assert_eq!(format_thought_label(None), "Thought");
        assert_eq!(
            format_thought_label(Some(Duration::from_millis(40))),
            "Thought briefly"
        );
        assert_eq!(
            format_thought_label(Some(Duration::from_secs(3))),
            "Thought for 3s"
        );
        assert_eq!(
            format_thought_label(Some(Duration::from_secs(65))),
            "Thought for 1m 5s"
        );
    }

    #[test]
    fn completes_thinking_toggle() {
        let c = slash_completions("/thin", &base(), &[], &[], &[], &[]);
        assert_eq!(c, vec!["/thinking".to_string()]);
        let c = slash_completions("/thinking ", &base(), &[], &[], &[], &[]);
        assert!(c.iter().any(|x| x == "/thinking on"));
        assert!(c.iter().any(|x| x == "/thinking off"));
    }

    #[test]
    fn completes_root_command_prefix() {
        let c = slash_completions("/mo", &base(), &[], &[], &[], &[]);
        assert!(c.contains(&"/model".to_string()), "{c:?}");
        assert!(c.contains(&"/mouse".to_string()), "{c:?}");
        let c = slash_completions("/mod", &base(), &[], &[], &[], &[]);
        assert_eq!(c, vec!["/model".to_string()]);
    }

    #[test]
    fn ghost_suffix_for_partial_command() {
        assert_eq!(slash_ghost_suffix("/e", "/exit").as_deref(), Some("xit"));
        assert_eq!(
            slash_ghost_suffix("/auth lo", "/auth login").as_deref(),
            Some("gin")
        );
        assert_eq!(slash_ghost_suffix("/exit", "/exit"), None);
        assert_eq!(slash_ghost_suffix("/z", "/exit"), None);
    }

    #[test]
    fn completes_auth_subcommands() {
        let c = slash_completions("/auth ", &base(), &[], &[], &[], &[]);
        assert!(c.iter().any(|x| x == "/auth login"));
        assert!(c.iter().any(|x| x == "/auth status"));
        let c = slash_completions("/auth lo", &base(), &[], &[], &[], &[]);
        assert_eq!(
            c,
            vec!["/auth login".to_string(), "/auth logout".to_string()]
        );
    }

    #[test]
    fn completes_auth_login_provider() {
        let providers = vec!["anthropic".into(), "xai".into()];
        let c = slash_completions("/auth login ", &base(), &[], &providers, &[], &[]);
        assert!(c.iter().any(|x| x == "/auth login xai"));
        let c = slash_completions("/auth login x", &base(), &[], &providers, &[], &[]);
        assert_eq!(c, vec!["/auth login xai".to_string()]);
    }

    #[test]
    fn completes_reset_subcommands() {
        let c = slash_completions("/re", &base(), &[], &[], &[], &[]);
        assert!(c.contains(&"/reset".to_string()), "{c:?}");
        let c = slash_completions("/reset ", &base(), &[], &[], &[], &[]);
        assert!(c.iter().any(|x| x == "/reset list"));
        assert!(c.iter().any(|x| x == "/reset apply"));
        let c = slash_completions("/reset a", &base(), &[], &[], &[], &[]);
        assert_eq!(c, vec!["/reset apply".to_string()]);
    }

    #[test]
    fn completes_models_and_themes() {
        let models = vec!["grok-4.5:high".into(), "grok-4.3".into()];
        let c = slash_completions("/model grok-4.5", &base(), &models, &[], &[], &[]);
        assert_eq!(c, vec!["/model grok-4.5:high".to_string()]);
        let themes = vec!["dark".into(), "dune".into()];
        let c = slash_completions("/theme d", &base(), &[], &[], &themes, &[]);
        assert!(c.iter().any(|x| x == "/theme dark"));
        assert!(c.iter().any(|x| x == "/theme dune"));
    }

    #[test]
    fn completes_session_ids_for_resume() {
        let sessions = vec!["abcdef12".into(), "abcdef99".into(), "deadbeef".into()];
        let c = slash_completions("/resume abc", &base(), &[], &[], &[], &sessions);
        assert_eq!(
            c,
            vec![
                "/resume abcdef12".to_string(),
                "/resume abcdef99".to_string()
            ]
        );
    }

    #[test]
    fn trailing_space_helpers() {
        assert!(completion_wants_trailing_space("/auth login"));
        assert!(!completion_wants_trailing_space("/auth login xai"));
        assert!(!completion_wants_trailing_space("/help"));
    }
}
