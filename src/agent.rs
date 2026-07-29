use std::fs;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;

use crate::config::Config;
use crate::llm;
use crate::llm::Usage;
use crate::tools::registry::Registry;

#[allow(dead_code)]
pub enum AgentEvent {
    Text(String),
    Thinking(String),
    ToolUse(String, String),
    ToolResult(String, String, String),
    Error(String),
    TurnEnd(llm::Usage),
    PermissionRequest(String, String),
    Compacted(usize),
    /// Transcript advanced far enough that the TUI should durable-checkpoint.
    /// Emitted after user input, assistant messages, tool results, and compact.
    Checkpoint,
    Done,
}

/// Above this fraction of a model's context window, compact proactively
/// before sending the next turn.
const COMPACT_THRESHOLD: f64 = 0.7;
/// Number of trailing messages compaction always keeps verbatim (the actual
/// kept tail may be a bit longer, since the split walks left to an assistant
/// boundary so the suffix never starts mid tool-call).
const KEEP_RECENT_MESSAGES: usize = 8;
/// Don't bother compacting for a handful of messages.
const MIN_MESSAGES_TO_COMPACT: usize = KEEP_RECENT_MESSAGES + 2;
const SUMMARY_MAX_TOKENS: usize = 1024;

const SUMMARY_SYSTEM_PROMPT: &str = "You are compacting an in-progress coding session's history so it fits in a smaller context window. Summarize the conversation below concisely but completely: preserve file paths touched, decisions made, and any outstanding/unfinished tasks. Write plain prose, not a transcript. Do not use tools.";

/// How a run answers a tool call that the deny/ask/`auto_allow` policy says
/// needs approval.
///
/// This is the run's *authorization posture*, deliberately separate from
/// whether the caller happens to hold a permission channel. A caller that owns
/// an event channel for streaming (`exec`) is not thereby a user who can
/// consent, so the two must not be inferred from one another.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ApprovalMode {
    /// Ask the human: emit `PermissionRequest` and wait for the answer.
    #[default]
    Interactive,
    /// No human to ask, so approval-requiring tools fail closed. An explicit
    /// `auto_allow` entry in config is the only way to permit one.
    FailClosed,
    /// Approve every tool call without asking. Only ever set from an explicit
    /// operator opt-in (`exec --auto`), never inferred.
    AutoApproveAll,
}

pub struct Agent {
    provider: Box<dyn llm::Provider>,
    model: String,
    messages: Vec<llm::Message>,
    tools: Registry,
    config: Config,
    usage: Usage,
    last_input_tokens: u64,
    /// Mirrors full message history for session autosave (shared with TUI).
    live_mirror: Option<crate::session::LiveMirror>,
    /// Skill catalog (names/descriptions); bodies loaded via the `skill` tool.
    skills: Vec<crate::skills::Skill>,
    /// Authorization posture for tool calls that require approval.
    approval: ApprovalMode,
}

impl Agent {
    #[allow(dead_code)]
    pub fn new(
        provider: Box<dyn llm::Provider>,
        model: String,
        tools: Registry,
        config: Config,
    ) -> Self {
        Self::new_with_skills(provider, model, tools, config, Vec::new())
    }

    pub fn new_with_skills(
        provider: Box<dyn llm::Provider>,
        model: String,
        tools: Registry,
        config: Config,
        skills: Vec<crate::skills::Skill>,
    ) -> Self {
        Agent {
            provider,
            model,
            messages: Vec::new(),
            tools,
            config,
            usage: Usage::default(),
            last_input_tokens: 0,
            live_mirror: None,
            skills,
            approval: ApprovalMode::default(),
        }
    }

    /// Sets the authorization posture for this run. Callers with no user to
    /// prompt must set [`ApprovalMode::FailClosed`]; only an explicit operator
    /// opt-in may set [`ApprovalMode::AutoApproveAll`].
    pub fn set_approval_mode(&mut self, mode: ApprovalMode) {
        self.approval = mode;
    }

    pub fn set_live_mirror(&mut self, mirror: crate::session::LiveMirror) {
        self.live_mirror = Some(mirror);
        self.sync_live_mirror();
    }

    fn sync_live_mirror(&self) {
        let Some(mirror) = &self.live_mirror else {
            return;
        };
        if let Ok(mut g) = mirror.lock() {
            g.messages = self.messages.clone();
            g.tokens_in = self.usage.input_tokens;
            g.tokens_out = self.usage.output_tokens;
        }
    }

    /// Push the live transcript and ask the TUI to flush it to disk.
    fn checkpoint(&self, tx: Option<&mpsc::Sender<AgentEvent>>) {
        self.sync_live_mirror();
        if let Some(tx) = tx {
            let _ = tx.send(AgentEvent::Checkpoint);
        }
    }

    #[allow(dead_code)]
    pub fn model(&self) -> &str {
        &self.model
    }
    #[allow(dead_code)]
    pub fn provider_name(&self) -> &str {
        self.provider.name()
    }
    #[allow(dead_code)]
    pub fn available_models(&self) -> Vec<llm::ModelInfo> {
        self.provider.available_models()
    }

    pub fn switch_provider(&mut self, provider_name: &str, model: &str) -> Result<(), String> {
        let mut providers = crate::llm::provider::default_providers();
        let provider = providers
            .remove(provider_name)
            .ok_or_else(|| format!("Unknown provider: {provider_name}"))?;
        self.provider = provider;
        self.model = model.to_string();
        self.last_input_tokens = 0;
        self.sync_live_mirror();
        Ok(())
    }

    pub fn set_state(&mut self, messages: Vec<llm::Message>, usage: Usage) {
        self.messages = messages;
        self.usage = usage;
        self.last_input_tokens = 0;
        self.sync_live_mirror();
    }

    pub fn reset_state(&mut self) {
        self.messages.clear();
        self.usage = Usage::default();
        self.last_input_tokens = 0;
        self.sync_live_mirror();
    }

    #[allow(dead_code)]
    pub fn messages(&self) -> &[llm::Message] {
        &self.messages
    }

    #[allow(dead_code)]
    pub fn usage(&self) -> &Usage {
        &self.usage
    }

    fn should_proactively_compact(&self) -> bool {
        if self.last_input_tokens == 0 || self.messages.len() < MIN_MESSAGES_TO_COMPACT {
            return false;
        }
        let max_ctx = self
            .provider
            .available_models()
            .iter()
            .find(|m| m.id == self.model)
            .map(|m| m.max_ctx);
        match max_ctx {
            Some(max_ctx) if max_ctx > 0 => {
                self.last_input_tokens as f64 >= max_ctx as f64 * COMPACT_THRESHOLD
            }
            _ => false,
        }
    }

    /// Summarizes the oldest messages (up to a safe split point) into one
    /// message, keeping the most recent turns verbatim. Returns the number
    /// of original messages folded into the summary, or 0 if it skipped
    /// (no safe split point, too few messages, or the summarization call
    /// itself failed). When `tx` is `Some`, emits usage and Compacted events.
    fn compact_history(&mut self, tx: Option<&mpsc::Sender<AgentEvent>>) -> usize {
        let Some(split) = find_safe_split_point(&self.messages) else {
            return 0;
        };

        let transcript = render_transcript(&self.messages[..split]);
        let request = vec![llm::Message {
            role: "user".into(),
            content: llm::Content::Text(transcript),
        }];

        let (summary_msgs, summary_usage) = match self.provider.complete(
            &request,
            &[],
            SUMMARY_SYSTEM_PROMPT,
            &self.model,
            SUMMARY_MAX_TOKENS,
        ) {
            Ok(r) => r,
            Err(_) => return 0,
        };

        let summary = summary_msgs.iter().find_map(|m| match &m.content {
            llm::Content::Text(t) if !t.is_empty() => Some(t.clone()),
            _ => None,
        });
        let Some(summary) = summary else {
            return 0;
        };

        let mut compacted = vec![llm::Message {
            role: "user".into(),
            content: llm::Content::Text(format!("[Earlier conversation summary]\n{summary}")),
        }];
        compacted.extend_from_slice(&self.messages[split..]);
        self.messages = compacted;

        // Reset so the next turn re-measures against the shrunk history instead
        // of immediately re-triggering on the pre-compact input token count.
        self.last_input_tokens = 0;

        self.usage.input_tokens += summary_usage.input_tokens;
        self.usage.output_tokens += summary_usage.output_tokens;
        if let Some(tx) = tx {
            let _ = tx.send(AgentEvent::TurnEnd(llm::Usage {
                input_tokens: summary_usage.input_tokens,
                output_tokens: summary_usage.output_tokens,
                cache_read: summary_usage.cache_read,
                cache_create: summary_usage.cache_create,
            }));
            let _ = tx.send(AgentEvent::Compacted(split));
        }

        // Durable checkpoint so a crash right after compact still keeps the
        // shortened history (and does not leave an older, longer file only).
        self.checkpoint(tx);

        split
    }

    /// Manual `/compact`: fold history now. Emits Compacted on success.
    pub fn compact_now(&mut self, tx: &mpsc::Sender<AgentEvent>) -> Result<usize, String> {
        let n = self.compact_history(Some(tx));
        if n == 0 {
            return Err(
                "Could not compact history (need more messages, a safe split point, or a working summarizer)."
                    .into(),
            );
        }
        Ok(n)
    }

    fn format_llm_err(e: String) -> String {
        let e = crate::redact::redact_secrets(&e);
        if e.starts_with("LLM error:") {
            e
        } else {
            format!("LLM error: {e}")
        }
    }

    fn max_turns_error(max_turns: usize) -> String {
        format!("Agent reached the maximum of {max_turns} turns before completing")
    }

    pub fn run(
        &mut self,
        input: &str,
        tx: mpsc::Sender<AgentEvent>,
        cancel: &AtomicBool,
        perm_rx: &mpsc::Receiver<String>,
    ) -> Result<(), String> {
        self.run_user(llm::UserBlocks::text_only(input), tx, cancel, perm_rx)
    }

    /// Run a user turn that may include pasted clipboard images.
    pub fn run_user(
        &mut self,
        user: llm::UserBlocks,
        tx: mpsc::Sender<AgentEvent>,
        cancel: &AtomicBool,
        perm_rx: &mpsc::Receiver<String>,
    ) -> Result<(), String> {
        if user.is_empty() {
            return Err("empty user message".into());
        }
        self.messages.push(llm::Message::user_blocks(user));
        // Persist the user prompt immediately so a crash during the first LLM
        // call does not lose the start of the turn.
        self.checkpoint(Some(&tx));

        let system = load_system_prompt(&self.config.system_prompt_file, &self.skills);
        // At most one reactive compact-and-retry per user turn so a provider
        // that keeps returning context errors cannot loop forever.
        let mut reactive_compact_attempted = false;

        let result = (|| -> Result<(), String> {
            for _turn in 0..self.config.max_turns {
                if cancel.load(Ordering::Relaxed) {
                    return Ok(());
                }

                if self.should_proactively_compact() {
                    self.compact_history(Some(&tx));
                }

                let tool_defs = self.tools.definitions();

                let tx_clone = tx.clone();
                let stream_result = self.provider.stream_complete(
                    &self.messages,
                    &tool_defs,
                    &system,
                    &self.model,
                    self.config.max_tokens,
                    Box::new(move |chunk, chunk_type| {
                        let _ = tx_clone.send(match chunk_type {
                            "thinking" => AgentEvent::Thinking(chunk.to_string()),
                            _ => AgentEvent::Text(chunk.to_string()),
                        });
                    }),
                    cancel,
                );

                let (new_msgs, usage) = match stream_result {
                    Ok(r) => r,
                    Err(e) => {
                        let err = Self::format_llm_err(e);
                        if !reactive_compact_attempted
                            && crate::http_client::is_context_limit_error(&err)
                        {
                            reactive_compact_attempted = true;
                            if self.compact_history(Some(&tx)) > 0 {
                                continue;
                            }
                        }
                        return Err(err);
                    }
                };

                if cancel.load(Ordering::Relaxed) {
                    return Ok(());
                }

                self.usage.input_tokens += usage.input_tokens;
                self.usage.output_tokens += usage.output_tokens;
                self.usage.cache_read += usage.cache_read;
                self.usage.cache_create += usage.cache_create;
                self.last_input_tokens = usage.input_tokens;

                let _ = tx.send(AgentEvent::TurnEnd(llm::Usage {
                    input_tokens: usage.input_tokens,
                    output_tokens: usage.output_tokens,
                    cache_read: usage.cache_read,
                    cache_create: usage.cache_create,
                }));

                let tool_uses: Vec<llm::ToolUse> = new_msgs
                    .iter()
                    .filter_map(|m| match &m.content {
                        llm::Content::ToolUse(tu) => Some(tu.clone()),
                        _ => None,
                    })
                    .collect();

                for msg in &new_msgs {
                    self.messages.push(msg.clone());
                }
                // Assistant text / tool_use is on the mirror before tools run.
                self.checkpoint(Some(&tx));

                if tool_uses.is_empty() {
                    return Ok(());
                }

                for tu in &tool_uses {
                    if cancel.load(Ordering::Relaxed) {
                        return Ok(());
                    }

                    let _ = tx.send(AgentEvent::ToolUse(tu.name.clone(), tu.input.clone()));

                    let result = self.execute_tool_with_policy(tu, Some((&tx, cancel, perm_rx)));

                    let result_str = match &result {
                        Ok(s) => s.clone(),
                        Err(e) => format!("Error: {e}"),
                    };

                    let _ = tx.send(AgentEvent::ToolResult(
                        tu.name.clone(),
                        tu.input.clone(),
                        result_str.clone(),
                    ));

                    self.messages.push(llm::Message {
                        role: "user".into(),
                        content: llm::Content::ToolResult(llm::ToolResult {
                            tool_use_id: tu.id.clone(),
                            content: result_str,
                        }),
                    });
                    // Checkpoint after each tool so long multi-tool turns
                    // survive process crashes mid-loop.
                    self.checkpoint(Some(&tx));
                }
            }

            Err(Self::max_turns_error(self.config.max_turns))
        })();

        if let Err(ref e) = result {
            let _ = tx.send(AgentEvent::Error(e.clone()));
        }
        self.sync_live_mirror();
        let _ = tx.send(AgentEvent::Done);
        result
    }

    fn dispatch_tool(&self, tu: &llm::ToolUse, cancel: &AtomicBool) -> Result<String, String> {
        match self.tools.get(&tu.name) {
            Some(tool) => tool.execute_with_cancel(&tu.input, cancel),
            None => Err(format!("Unknown tool: {}", tu.name)),
        }
    }

    /// Applies the deny/ask/auto_allow policy to a tool call and, if
    /// permitted, executes it. This is the single authorization path shared
    /// by the interactive TUI loop and print (`--print`) mode, so a tool
    /// cannot bypass deny lists or `needs_permission_for()` by going through one
    /// path instead of the other.
    ///
    /// `interactive` carries the channels needed to prompt a human for
    /// approval (`tx` to emit `PermissionRequest`, `cancel` to abort the
    /// wait, `perm_rx` to receive the answer). When `None` (non-interactive
    /// callers like `--print`, which have no user to ask) any tool that
    /// would otherwise require approval fails closed instead of running
    /// unattended; the only way to permit it is an explicit `auto_allow`
    /// entry in config.
    ///
    /// [`Agent::set_approval_mode`] overrides that per run, and it is the
    /// authority: holding the channels does not make a run interactive. A
    /// caller that streams events but has no human behind them (`exec`) sets
    /// [`ApprovalMode::FailClosed`] and the channels are ignored here, so
    /// approval cannot be granted by whatever happens to be feeding
    /// `perm_rx`.
    fn execute_tool_with_policy(
        &mut self,
        tu: &llm::ToolUse,
        interactive: Option<(
            &mpsc::Sender<AgentEvent>,
            &AtomicBool,
            &mpsc::Receiver<String>,
        )>,
    ) -> Result<String, String> {
        let tool = self.tools.get(&tu.name);
        let wants_permission = tool
            .map(|t| t.needs_permission_for(&tu.input))
            .unwrap_or(false);
        let needs_ask = wants_permission || self.config.ask.iter().any(|t| t == &tu.name);
        let permission_key = tool
            .map(|t| t.permission_key(&tu.input))
            .unwrap_or_else(|| tu.name.clone());
        let always_allowed = self.config.auto_allow.iter().any(|t| t == &permission_key);
        let denied = self.config.is_tool_denied(&tu.name);

        if denied {
            return Err(format!("Tool '{}' is denied by config", tu.name));
        }

        // Cancellation reaches synchronous tool execution through this token.
        // Any caller that supplied one gets it, regardless of approval posture:
        // `exec` cannot approve tools but can still abort a running one.
        // Callers with no token at all (e.g. --print) never cancel.
        static NEVER_CANCEL: AtomicBool = AtomicBool::new(false);
        let cancel_flag: &AtomicBool = match interactive {
            Some((_, cancel, _)) => cancel,
            None => &NEVER_CANCEL,
        };

        // The approval posture decides how (and whether) a human is consulted,
        // and it is what makes the permission channel usable. `deny` is checked
        // above, so `--auto` widens approval but never overrides a denial.
        let auto_approve_all = self.approval == ApprovalMode::AutoApproveAll;
        let approval_channels = match self.approval {
            ApprovalMode::Interactive => interactive,
            ApprovalMode::FailClosed | ApprovalMode::AutoApproveAll => None,
        };

        if !needs_ask || always_allowed || auto_approve_all {
            return self.dispatch_tool(tu, cancel_flag);
        }

        let Some((tx, cancel, perm_rx)) = approval_channels else {
            return Err(format!(
                "Tool '{}' requires approval and there is no user to prompt in this mode. \
                 Add it to auto_allow in the config to permit it non-interactively.",
                tu.name
            ));
        };

        let _ = tx.send(AgentEvent::PermissionRequest(
            tu.name.clone(),
            tu.input.clone(),
        ));
        let response = loop {
            match perm_rx.recv_timeout(std::time::Duration::from_millis(200)) {
                Ok(resp) => break resp,
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if cancel.load(Ordering::Relaxed) {
                        break "deny".to_string();
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break "deny".to_string(),
            }
        };
        match permission_decision(&response) {
            PermissionDecision::AlwaysAllow => {
                if !always_allowed {
                    self.config.auto_allow.push(permission_key);
                }
                let _ = crate::config::save_permissions(&self.config);
                self.dispatch_tool(tu, cancel_flag)
            }
            PermissionDecision::Allow => self.dispatch_tool(tu, cancel_flag),
            PermissionDecision::Discuss(feedback) => {
                let detail = match feedback {
                    Some(msg) if !msg.is_empty() => {
                        format!("User declined and said: {msg}")
                    }
                    _ => "User wants to discuss this action before it runs.".into(),
                };
                Err(format!(
                    "Permission denied by user for tool '{}'. {detail}",
                    tu.name
                ))
            }
            PermissionDecision::Deny => {
                Err(format!("Permission denied by user for tool '{}'", tu.name))
            }
        }
    }

    pub fn run_simple(&mut self, input: &str) -> Result<String, String> {
        self.messages.push(llm::Message {
            role: "user".into(),
            content: llm::Content::Text(input.to_string()),
        });

        let system = load_system_prompt(&self.config.system_prompt_file, &self.skills);
        let mut reactive_compact_attempted = false;
        let mut output = String::new();

        let result = (|| -> Result<String, String> {
            for _turn in 0..self.config.max_turns {
                if self.should_proactively_compact() {
                    self.compact_history(None);
                }

                let tool_defs = self.tools.definitions();
                let (new_msgs, usage) = match self.provider.complete(
                    &self.messages,
                    &tool_defs,
                    &system,
                    &self.model,
                    self.config.max_tokens,
                ) {
                    Ok(r) => r,
                    Err(e) => {
                        let err = Self::format_llm_err(e);
                        if !reactive_compact_attempted
                            && crate::http_client::is_context_limit_error(&err)
                        {
                            reactive_compact_attempted = true;
                            if self.compact_history(None) > 0 {
                                continue;
                            }
                        }
                        return Err(err);
                    }
                };

                self.usage.input_tokens += usage.input_tokens;
                self.usage.output_tokens += usage.output_tokens;
                self.usage.cache_read += usage.cache_read;
                self.usage.cache_create += usage.cache_create;
                self.last_input_tokens = usage.input_tokens;

                let tool_uses: Vec<llm::ToolUse> = new_msgs
                    .iter()
                    .filter_map(|msg| match &msg.content {
                        llm::Content::ToolUse(tu) => Some(tu.clone()),
                        _ => None,
                    })
                    .collect();

                for msg in &new_msgs {
                    self.messages.push(msg.clone());
                    if let llm::Content::Text(text) = &msg.content {
                        output.push_str(text);
                    }
                }

                if tool_uses.is_empty() {
                    return Ok(output);
                }

                for tu in &tool_uses {
                    // Print mode has no user to prompt, so approval-required
                    // tools fail closed unless config explicitly auto-allows them.
                    let tool_result = self
                        .execute_tool_with_policy(tu, None)
                        .unwrap_or_else(|e| format!("Error: {e}"));
                    output.push_str(&format!("\n[{}({})]: {}", tu.name, tu.input, tool_result));
                    self.messages.push(llm::Message {
                        role: "user".into(),
                        content: llm::Content::ToolResult(llm::ToolResult {
                            tool_use_id: tu.id.clone(),
                            content: tool_result,
                        }),
                    });
                }
            }

            Err(Self::max_turns_error(self.config.max_turns))
        })();

        self.sync_live_mirror();
        result
    }
}

/// How the interactive permission prompt resolved a tool approval request.
#[derive(Debug, Clone, PartialEq, Eq)]
enum PermissionDecision {
    Allow,
    AlwaysAllow,
    Deny,
    /// Do not run the tool; feed optional free-text feedback to the model.
    Discuss(Option<String>),
}

/// Parse the TUI/permission-channel response string.
///
/// Wire values: `allow`, `always_allow`, `deny`, `discuss`, or `discuss:<msg>`.
fn permission_decision(response: &str) -> PermissionDecision {
    if response == "allow" {
        return PermissionDecision::Allow;
    }
    if response == "always_allow" {
        return PermissionDecision::AlwaysAllow;
    }
    if response == "discuss" {
        return PermissionDecision::Discuss(None);
    }
    if let Some(msg) = response.strip_prefix("discuss:") {
        return PermissionDecision::Discuss(Some(msg.to_string()));
    }
    PermissionDecision::Deny
}

fn load_system_prompt(path: &str, skills: &[crate::skills::Skill]) -> String {
    let mut content = fs::read_to_string(path).unwrap_or_default();
    let catalog = crate::skills::catalog_prompt(skills);
    if !catalog.is_empty() {
        if !content.is_empty() && !content.ends_with('\n') {
            content.push('\n');
        }
        if !content.is_empty() {
            content.push('\n');
        }
        content.push_str(&catalog);
    }
    content
}

/// Index at which to split history for compaction: `messages[..split]` is
/// summarized, `messages[split..]` is kept. Walks the naive keep-window start
/// leftward so the preserved suffix begins on an assistant message (providers
/// reject a dangling tool_result without its preceding tool_use, and a
/// user-led suffix would create consecutive user turns after the injected
/// summary). Returns `None` when there is nothing safe to fold.
fn find_safe_split_point(messages: &[llm::Message]) -> Option<usize> {
    if messages.len() < MIN_MESSAGES_TO_COMPACT {
        return None;
    }
    let mut split = messages.len() - KEEP_RECENT_MESSAGES;
    while split > 0 && messages[split].role != "assistant" {
        split -= 1;
    }
    if split == 0 {
        return None;
    }
    Some(split)
}

/// Flatten messages into plain text for the summarizer.
fn render_transcript(messages: &[llm::Message]) -> String {
    messages
        .iter()
        .map(|m| match &m.content {
            llm::Content::Text(t) => format!("{}: {t}", m.role),
            llm::Content::User(blocks) => format!("{}: {}", m.role, blocks.display_label()),
            llm::Content::Thinking(t) => format!("{} [thinking]: {t}", m.role),
            llm::Content::ToolUse(tu) => {
                format!("assistant [tool call]: {}({})", tu.name, tu.input)
            }
            llm::Content::ToolResult(tr) => format!("tool result: {}", tr.content),
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{Content, Message, ToolResult, ToolUse};

    fn text(role: &str, body: &str) -> Message {
        Message {
            role: role.into(),
            content: Content::Text(body.into()),
        }
    }

    fn tool_use(name: &str, input: &str) -> Message {
        Message {
            role: "assistant".into(),
            content: Content::ToolUse(ToolUse {
                id: "1".into(),
                name: name.into(),
                input: input.into(),
            }),
        }
    }

    fn tool_result(content: &str) -> Message {
        Message {
            role: "user".into(),
            content: Content::ToolResult(ToolResult {
                tool_use_id: "1".into(),
                content: content.into(),
            }),
        }
    }

    #[test]
    fn find_safe_split_skips_short_histories() {
        let msgs = vec![text("user", "a"), text("assistant", "b")];
        assert!(find_safe_split_point(&msgs).is_none());
    }

    #[test]
    fn find_safe_split_keeps_recent_and_starts_on_assistant() {
        // 12 messages: u a u a u a u a u a u a
        let mut msgs = Vec::new();
        for i in 0..6 {
            msgs.push(text("user", &format!("u{i}")));
            msgs.push(text("assistant", &format!("a{i}")));
        }
        let split = find_safe_split_point(&msgs).expect("should split");
        // Naive keep window is last 8; walk left to assistant if needed.
        assert!(split <= msgs.len() - KEEP_RECENT_MESSAGES);
        assert_eq!(msgs[split].role, "assistant");
        assert!(msgs.len() - split >= KEEP_RECENT_MESSAGES);
    }

    #[test]
    fn find_safe_split_does_not_start_on_tool_result() {
        // Build enough messages, ending the keep window on a tool_result.
        let mut msgs = vec![
            text("user", "start"),
            text("assistant", "ok"),
            text("user", "do work"),
            tool_use("read", r#"{"path":"a"}"#),
            tool_result("file a"),
            text("assistant", "done a"),
            text("user", "more"),
            tool_use("read", r#"{"path":"b"}"#),
            tool_result("file b"),
            text("assistant", "done b"),
            text("user", "again"),
            text("assistant", "final"),
        ];
        // Pad if under min
        while msgs.len() < MIN_MESSAGES_TO_COMPACT {
            msgs.insert(0, text("user", "pad"));
            msgs.insert(1, text("assistant", "pad"));
        }
        let split = find_safe_split_point(&msgs).expect("should split");
        assert_eq!(msgs[split].role, "assistant");
        assert!(!matches!(msgs[split].content, Content::ToolResult(_)));
    }

    #[test]
    fn render_transcript_covers_content_kinds() {
        let msgs = vec![
            text("user", "hello"),
            Message {
                role: "user".into(),
                content: Content::User(crate::llm::UserBlocks {
                    text: "cap".into(),
                    images: vec![crate::llm::ImageBlock {
                        media_type: "image/png".into(),
                        data_base64: "x".into(),
                    }],
                }),
            },
            text("assistant", "hi"),
            tool_use("grep", r#"{"q":"x"}"#),
            tool_result("matches"),
        ];
        let t = render_transcript(&msgs);
        assert!(t.contains("user: hello"));
        assert!(t.contains("user: cap\n[image]"));
        assert!(t.contains("assistant: hi"));
        assert!(t.contains("assistant [tool call]: grep("));
        assert!(t.contains("tool result: matches"));
    }

    #[test]
    fn provider_and_model_switches_preserve_history_usage_and_live_mirror() {
        let mut agent = Agent::new(
            Box::new(crate::llm::ollama::OllamaProvider::new()),
            "llama3.2".into(),
            crate::tools::registry::Registry::new(),
            crate::config::Config::default(),
        );
        let history = vec![
            text("user", "inspect src/main.rs"),
            tool_use("read", r#"{"path":"src/main.rs"}"#),
            tool_result("fn main() {}"),
            text("assistant", "done"),
        ];
        let usage = Usage {
            input_tokens: 120,
            output_tokens: 30,
            cache_read: 10,
            cache_create: 5,
        };
        let mirror = crate::session::new_live_mirror();
        agent.set_live_mirror(mirror.clone());
        agent.set_state(history, usage);

        agent.switch_provider("openai", "gpt-5-mini").unwrap();

        assert_eq!(agent.provider_name(), "openai");
        assert_eq!(agent.model(), "gpt-5-mini");
        assert_eq!(agent.messages().len(), 4);
        assert!(
            matches!(&agent.messages()[0].content, Content::Text(t) if t == "inspect src/main.rs")
        );
        assert!(matches!(&agent.messages()[1].content, Content::ToolUse(tu) if tu.name == "read"));
        assert!(
            matches!(&agent.messages()[2].content, Content::ToolResult(tr) if tr.content == "fn main() {}")
        );
        assert!(matches!(&agent.messages()[3].content, Content::Text(t) if t == "done"));
        assert_eq!(agent.usage().input_tokens, 120);
        assert_eq!(agent.usage().output_tokens, 30);
        assert_eq!(agent.usage().cache_read, 10);
        assert_eq!(agent.usage().cache_create, 5);

        let snapshot = mirror.lock().unwrap();
        assert_eq!(snapshot.messages.len(), 4);
        assert_eq!(snapshot.tokens_in, 120);
        assert_eq!(snapshot.tokens_out, 30);
        drop(snapshot);

        agent.switch_provider("openai", "gpt-4.1-mini").unwrap();

        assert_eq!(agent.model(), "gpt-4.1-mini");
        assert_eq!(agent.messages().len(), 4);
        assert_eq!(agent.usage().input_tokens, 120);
        assert_eq!(mirror.lock().unwrap().messages.len(), 4);
    }

    /// Live-exercises proactive compaction without a network call: first stream
    /// reports a high input-token count, `complete` returns a summary, second
    /// stream should see the compacted history.
    struct SharedMock {
        stream_calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        complete_calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        last_stream_message_count: std::sync::Arc<std::sync::Mutex<usize>>,
    }

    impl llm::Provider for SharedMock {
        fn name(&self) -> &str {
            "mock"
        }
        fn default_model(&self) -> &str {
            "mock-model"
        }
        fn available_models(&self) -> Vec<llm::ModelInfo> {
            vec![llm::ModelInfo {
                id: "mock-model".into(),
                name: "Mock".into(),
                max_ctx: 1000,
            }]
        }
        fn stream_complete(
            &self,
            messages: &[llm::Message],
            _tools: &[llm::ToolDefinition],
            _system: &str,
            _model: &str,
            _max_tokens: usize,
            mut on_chunk: llm::StreamingCallback,
            _cancel: &AtomicBool,
        ) -> Result<(Vec<llm::Message>, Usage), String> {
            let n = self.stream_calls.fetch_add(1, Ordering::SeqCst);
            *self.last_stream_message_count.lock().unwrap() = messages.len();
            on_chunk("ok", "text");
            Ok((
                vec![llm::Message {
                    role: "assistant".into(),
                    content: llm::Content::Text(format!("reply-{n}")),
                }],
                Usage {
                    input_tokens: if n == 0 { 800 } else { 120 },
                    output_tokens: 4,
                    cache_read: 0,
                    cache_create: 0,
                },
            ))
        }
        fn complete(
            &self,
            _messages: &[llm::Message],
            _tools: &[llm::ToolDefinition],
            system: &str,
            _model: &str,
            _max_tokens: usize,
        ) -> Result<(Vec<llm::Message>, Usage), String> {
            self.complete_calls.fetch_add(1, Ordering::SeqCst);
            assert!(
                system.contains("compacting"),
                "summarizer should use compaction system prompt, got: {system}"
            );
            Ok((
                vec![llm::Message {
                    role: "assistant".into(),
                    content: llm::Content::Text("Prior work on foo.rs and bar.rs.".into()),
                }],
                Usage {
                    input_tokens: 40,
                    output_tokens: 12,
                    cache_read: 0,
                    cache_create: 0,
                },
            ))
        }
    }

    #[test]
    fn reset_state_clears_agent_and_live_session_state() {
        let mut agent = Agent::new(
            Box::new(SharedMock {
                stream_calls: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                complete_calls: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                last_stream_message_count: std::sync::Arc::new(std::sync::Mutex::new(0)),
            }),
            "mock-model".into(),
            crate::tools::registry::Registry::new(),
            crate::config::Config::default(),
        );
        let mirror = crate::session::new_live_mirror();
        agent.set_live_mirror(mirror.clone());
        agent.set_state(
            vec![text("user", "remember this")],
            Usage {
                input_tokens: 100,
                output_tokens: 20,
                cache_read: 30,
                cache_create: 40,
            },
        );
        agent.last_input_tokens = 100;

        agent.reset_state();

        assert!(agent.messages().is_empty());
        assert_eq!(agent.usage().input_tokens, 0);
        assert_eq!(agent.usage().output_tokens, 0);
        assert_eq!(agent.usage().cache_read, 0);
        assert_eq!(agent.usage().cache_create, 0);
        assert_eq!(agent.last_input_tokens, 0);
        let snapshot = mirror.lock().unwrap();
        assert!(snapshot.messages.is_empty());
        assert_eq!(snapshot.tokens_in, 0);
        assert_eq!(snapshot.tokens_out, 0);
    }

    #[test]
    fn proactive_compaction_runs_before_next_turn() {
        let stream_calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let complete_calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let last_count = std::sync::Arc::new(std::sync::Mutex::new(0usize));

        let mut seed = Vec::new();
        for i in 0..8 {
            seed.push(text("user", &format!("user-{i}")));
            seed.push(text("assistant", &format!("assistant-{i}")));
        }

        let mut agent = Agent::new(
            Box::new(SharedMock {
                stream_calls: stream_calls.clone(),
                complete_calls: complete_calls.clone(),
                last_stream_message_count: last_count.clone(),
            }),
            "mock-model".into(),
            crate::tools::registry::Registry::new(),
            crate::config::Config::default(),
        );
        agent.set_state(seed, Usage::default());

        let (tx, rx) = mpsc::channel();
        let (_perm_tx, perm_rx) = mpsc::channel();
        let cancel = AtomicBool::new(false);

        agent.run("first", tx.clone(), &cancel, &perm_rx).unwrap();
        assert_eq!(stream_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            complete_calls.load(Ordering::SeqCst),
            0,
            "no compact on first turn"
        );

        while rx.try_recv().is_ok() {}

        agent.run("second", tx, &cancel, &perm_rx).unwrap();
        assert_eq!(
            complete_calls.load(Ordering::SeqCst),
            1,
            "summarizer should run before second stream"
        );
        assert_eq!(stream_calls.load(Ordering::SeqCst), 2);

        let events: Vec<_> = rx.try_iter().collect();
        let compacted = events.iter().any(|e| matches!(e, AgentEvent::Compacted(_)));
        assert!(
            compacted,
            "expected Compacted event among {} events",
            events.len()
        );

        let msgs = agent.messages();
        assert!(
            msgs.iter().any(|m| matches!(&m.content, llm::Content::Text(t) if t.contains("[Earlier conversation summary]"))),
            "history should start with a summary message"
        );
        // Pre-compact peak is seed(16)+user+asst+user ≈ 19; compacted should be smaller.
        let seen = *last_count.lock().unwrap();
        assert!(
            seen < 20,
            "second stream should see a compacted history, got {seen} messages"
        );
    }

    /// First stream fails with a context-limit error; after compact, second stream succeeds.
    struct ReactiveMock {
        stream_calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        complete_calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl llm::Provider for ReactiveMock {
        fn name(&self) -> &str {
            "mock"
        }
        fn default_model(&self) -> &str {
            "mock-model"
        }
        fn available_models(&self) -> Vec<llm::ModelInfo> {
            vec![llm::ModelInfo {
                id: "mock-model".into(),
                name: "Mock".into(),
                max_ctx: 1000,
            }]
        }
        fn stream_complete(
            &self,
            _messages: &[llm::Message],
            _tools: &[llm::ToolDefinition],
            _system: &str,
            _model: &str,
            _max_tokens: usize,
            mut on_chunk: llm::StreamingCallback,
            _cancel: &AtomicBool,
        ) -> Result<(Vec<llm::Message>, Usage), String> {
            let n = self.stream_calls.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                return Err("prompt is too long: exceeds the model context window".into());
            }
            on_chunk("ok", "text");
            Ok((
                vec![llm::Message {
                    role: "assistant".into(),
                    content: llm::Content::Text("recovered".into()),
                }],
                Usage {
                    input_tokens: 50,
                    output_tokens: 3,
                    cache_read: 0,
                    cache_create: 0,
                },
            ))
        }
        fn complete(
            &self,
            _messages: &[llm::Message],
            _tools: &[llm::ToolDefinition],
            _system: &str,
            _model: &str,
            _max_tokens: usize,
        ) -> Result<(Vec<llm::Message>, Usage), String> {
            self.complete_calls.fetch_add(1, Ordering::SeqCst);
            Ok((
                vec![llm::Message {
                    role: "assistant".into(),
                    content: llm::Content::Text("summary".into()),
                }],
                Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                    cache_read: 0,
                    cache_create: 0,
                },
            ))
        }
    }

    #[test]
    fn reactive_compaction_retries_after_context_limit() {
        let stream_calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let complete_calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut seed = Vec::new();
        for i in 0..8 {
            seed.push(text("user", &format!("u{i}")));
            seed.push(text("assistant", &format!("a{i}")));
        }
        let mut agent = Agent::new(
            Box::new(ReactiveMock {
                stream_calls: stream_calls.clone(),
                complete_calls: complete_calls.clone(),
            }),
            "mock-model".into(),
            crate::tools::registry::Registry::new(),
            crate::config::Config::default(),
        );
        agent.set_state(seed, Usage::default());

        let (tx, rx) = mpsc::channel();
        let (_perm_tx, perm_rx) = mpsc::channel();
        let cancel = AtomicBool::new(false);
        agent.run("big prompt", tx, &cancel, &perm_rx).unwrap();

        assert_eq!(stream_calls.load(Ordering::SeqCst), 2, "fail then retry");
        assert_eq!(complete_calls.load(Ordering::SeqCst), 1, "summarizer once");
        let events: Vec<_> = rx.try_iter().collect();
        assert!(events.iter().any(|e| matches!(e, AgentEvent::Compacted(_))));
        assert!(agent.messages().iter().any(|m| {
            matches!(&m.content, llm::Content::Text(t) if t.contains("[Earlier conversation summary]"))
        }));
    }

    #[test]
    fn compact_now_returns_error_when_too_short() {
        let stream_calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let complete_calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut agent = Agent::new(
            Box::new(SharedMock {
                stream_calls,
                complete_calls,
                last_stream_message_count: std::sync::Arc::new(std::sync::Mutex::new(0)),
            }),
            "mock-model".into(),
            crate::tools::registry::Registry::new(),
            crate::config::Config::default(),
        );
        let (tx, _rx) = mpsc::channel();
        let err = agent.compact_now(&tx).unwrap_err();
        assert!(
            err.to_ascii_lowercase().contains("could not compact"),
            "{err}"
        );
    }

    /// Returns one tool call, then a final response after receiving its result.
    struct ToolCallMock {
        tool_name: String,
        tool_input: String,
    }

    struct RecordingTool {
        name: String,
        needs_permission: bool,
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl crate::tools::registry::Tool for RecordingTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn description(&self) -> &str {
            "Records executions for authorization tests"
        }
        fn input_schema(&self) -> String {
            r#"{"type":"object"}"#.into()
        }
        fn needs_permission(&self) -> bool {
            self.needs_permission
        }
        fn execute(&self, _input: &str) -> Result<String, String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok("recorded execution".into())
        }
    }

    impl llm::Provider for ToolCallMock {
        fn name(&self) -> &str {
            "mock"
        }
        fn default_model(&self) -> &str {
            "mock-model"
        }
        fn available_models(&self) -> Vec<llm::ModelInfo> {
            vec![llm::ModelInfo {
                id: "mock-model".into(),
                name: "Mock".into(),
                max_ctx: 1000,
            }]
        }
        fn stream_complete(
            &self,
            _messages: &[llm::Message],
            _tools: &[llm::ToolDefinition],
            _system: &str,
            _model: &str,
            _max_tokens: usize,
            _on_chunk: llm::StreamingCallback,
            _cancel: &AtomicBool,
        ) -> Result<(Vec<llm::Message>, Usage), String> {
            unimplemented!("run_simple does not stream")
        }
        fn complete(
            &self,
            messages: &[llm::Message],
            _tools: &[llm::ToolDefinition],
            _system: &str,
            _model: &str,
            _max_tokens: usize,
        ) -> Result<(Vec<llm::Message>, Usage), String> {
            if messages
                .iter()
                .any(|message| matches!(&message.content, llm::Content::ToolResult(_)))
            {
                return Ok((
                    vec![llm::Message {
                        role: "assistant".into(),
                        content: llm::Content::Text("tool complete".into()),
                    }],
                    Usage::default(),
                ));
            }
            Ok((
                vec![llm::Message {
                    role: "assistant".into(),
                    content: llm::Content::ToolUse(llm::ToolUse {
                        id: "1".into(),
                        name: self.tool_name.clone(),
                        input: self.tool_input.clone(),
                    }),
                }],
                Usage::default(),
            ))
        }
    }

    fn agent_with_tool_call(
        tool_name: &str,
        needs_permission: bool,
        config: Config,
    ) -> (Agent, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut registry = crate::tools::registry::Registry::new();
        registry.register(Box::new(RecordingTool {
            name: tool_name.into(),
            needs_permission,
            calls: calls.clone(),
        }));
        (
            Agent::new(
                Box::new(ToolCallMock {
                    tool_name: tool_name.into(),
                    tool_input: "{}".into(),
                }),
                "mock-model".into(),
                registry,
                config,
            ),
            calls,
        )
    }

    fn test_tool_use(name: &str) -> llm::ToolUse {
        llm::ToolUse {
            id: "1".into(),
            name: name.into(),
            input: "{}".into(),
        }
    }

    /// C-01 regression: print mode (`run_simple`) must not execute a tool
    /// that requires approval just because there is no user to ask. It has
    /// to fail closed exactly like a denied tool would, not silently run.
    #[test]
    fn run_simple_denies_approval_required_tool_by_default() {
        let mut config = Config::default();
        config.ask.clear();
        config.auto_allow.clear();
        let (mut agent, calls) = agent_with_tool_call("dangerous", true, config);
        let output = agent.run_simple("do something").unwrap();
        assert!(
            output.to_ascii_lowercase().contains("error") && output.contains("approval"),
            "expected run_simple to fail closed on an approval-required tool, got: {output}"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn run_simple_denies_permission_free_tool_in_ask_list() {
        let mut config = Config::default();
        config.auto_allow.clear();
        config.ask = vec!["safe".into()];
        let (mut agent, calls) = agent_with_tool_call("safe", false, config);
        let output = agent.run_simple("do something").unwrap();
        assert!(output.contains("requires approval"), "{output}");
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    /// The explicit `auto_allow` opt-in the issue calls for should still let
    /// a would-be-approval-required tool run in non-interactive mode.
    #[test]
    fn run_simple_runs_tool_explicitly_auto_allowed() {
        let mut config = Config::default();
        config.auto_allow.push("dangerous".into());
        let (mut agent, calls) = agent_with_tool_call("dangerous", true, config);
        let output = agent.run_simple("do something").unwrap();
        assert!(output.contains("recorded execution"), "{output}");
        assert!(output.contains("tool complete"), "{output}");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn run_simple_submits_tool_result_and_continues_to_final_response() {
        let mut config = Config::default();
        config.ask.clear();
        let (mut agent, calls) = agent_with_tool_call("safe", false, config);

        let output = agent.run_simple("use the tool").unwrap();

        assert!(output.contains("recorded execution"), "{output}");
        assert!(output.ends_with("tool complete"), "{output}");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(matches!(
            &agent.messages()[2].content,
            llm::Content::ToolResult(result)
                if result.tool_use_id == "1" && result.content == "recorded execution"
        ));
        assert!(matches!(
            &agent.messages()[3].content,
            llm::Content::Text(text) if text == "tool complete"
        ));
    }

    /// Denied tools must stay denied in print mode too, matching the
    /// interactive loop's policy.
    #[test]
    fn run_simple_respects_deny_list() {
        let mut config = Config::default();
        config.auto_allow.push("dangerous".into());
        config.deny.push("dangerous".into());
        let (mut agent, calls) = agent_with_tool_call("dangerous", true, config);
        let output = agent.run_simple("do something").unwrap();
        assert!(output.contains("denied by config"), "{output}");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "deny must override auto_allow"
        );
    }

    /// Tools that don't need permission should keep working unattended.
    #[test]
    fn run_simple_runs_tool_that_does_not_need_permission() {
        let mut config = Config::default();
        config.ask.clear();
        config.auto_allow.clear();
        let (mut agent, calls) = agent_with_tool_call("safe", false, config);
        let output = agent.run_simple("do something").unwrap();
        assert!(output.contains("recorded execution"), "{output}");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn interactive_policy_prompts_and_executes_when_allowed() {
        let mut config = Config::default();
        config.ask.clear();
        config.auto_allow.clear();
        let (mut agent, calls) = agent_with_tool_call("dangerous", true, config);
        let (event_tx, event_rx) = mpsc::channel();
        let (perm_tx, perm_rx) = mpsc::channel();
        perm_tx.send("allow".into()).unwrap();

        let result = agent.execute_tool_with_policy(
            &test_tool_use("dangerous"),
            Some((&event_tx, &AtomicBool::new(false), &perm_rx)),
        );

        assert_eq!(result.unwrap(), "recorded execution");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(matches!(
            event_rx.recv().unwrap(),
            AgentEvent::PermissionRequest(name, _) if name == "dangerous"
        ));
    }

    #[test]
    fn interactive_policy_does_not_execute_when_denied() {
        let mut config = Config::default();
        config.ask.clear();
        config.auto_allow.clear();
        let (mut agent, calls) = agent_with_tool_call("dangerous", true, config);
        let (event_tx, event_rx) = mpsc::channel();
        let (perm_tx, perm_rx) = mpsc::channel();
        perm_tx.send("deny".into()).unwrap();

        let result = agent.execute_tool_with_policy(
            &test_tool_use("dangerous"),
            Some((&event_tx, &AtomicBool::new(false), &perm_rx)),
        );

        assert!(result.unwrap_err().contains("Permission denied"));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(matches!(
            event_rx.recv().unwrap(),
            AgentEvent::PermissionRequest(_, _)
        ));
    }

    #[test]
    fn interactive_policy_discuss_includes_feedback_without_running() {
        let mut config = Config::default();
        config.ask.clear();
        config.auto_allow.clear();
        let (mut agent, calls) = agent_with_tool_call("dangerous", true, config);
        let (event_tx, _event_rx) = mpsc::channel();
        let (perm_tx, perm_rx) = mpsc::channel();
        perm_tx
            .send("discuss:use a safer alternative".into())
            .unwrap();

        let err = agent
            .execute_tool_with_policy(
                &test_tool_use("dangerous"),
                Some((&event_tx, &AtomicBool::new(false), &perm_rx)),
            )
            .unwrap_err();

        assert!(err.contains("Permission denied"), "{err}");
        assert!(err.contains("use a safer alternative"), "{err}");
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    /// C-1 regression: `exec` mode used to spawn a thread that pumped "allow"
    /// into the permission channel 20x/second, so every approval-gated tool
    /// ran unattended. A run with no user behind it must fail closed even
    /// while something is feeding `perm_rx` — holding the channels is not
    /// consent.
    #[test]
    fn fail_closed_mode_ignores_a_permission_channel_answering_allow() {
        let mut config = Config::default();
        config.ask.clear();
        config.auto_allow.clear();
        let (mut agent, calls) = agent_with_tool_call("dangerous", true, config);
        agent.set_approval_mode(ApprovalMode::FailClosed);
        let (event_tx, event_rx) = mpsc::channel();
        let (perm_tx, perm_rx) = mpsc::channel();
        // Exactly what the deleted auto-allow thread did.
        for _ in 0..8 {
            perm_tx.send("allow".into()).unwrap();
        }

        let err = agent
            .execute_tool_with_policy(
                &test_tool_use("dangerous"),
                Some((&event_tx, &AtomicBool::new(false), &perm_rx)),
            )
            .unwrap_err();

        assert!(err.contains("approval"), "{err}");
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(
            event_rx.try_recv().is_err(),
            "fail-closed runs must not emit a prompt nobody can answer"
        );
    }

    /// The explicit `auto_allow` opt-in still works in a fail-closed run, so
    /// automation can permit specific tools without opening everything.
    #[test]
    fn fail_closed_mode_runs_tool_explicitly_auto_allowed() {
        let mut config = Config::default();
        config.ask.clear();
        config.auto_allow = vec!["dangerous".into()];
        let (mut agent, calls) = agent_with_tool_call("dangerous", true, config);
        agent.set_approval_mode(ApprovalMode::FailClosed);
        let (event_tx, _event_rx) = mpsc::channel();
        let (_perm_tx, perm_rx) = mpsc::channel();

        let result = agent.execute_tool_with_policy(
            &test_tool_use("dangerous"),
            Some((&event_tx, &AtomicBool::new(false), &perm_rx)),
        );

        assert_eq!(result.unwrap(), "recorded execution");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    /// `exec --auto` is the opt-in that restores unattended execution.
    #[test]
    fn auto_approve_all_mode_runs_approval_required_tool_without_prompting() {
        let mut config = Config::default();
        config.ask.clear();
        config.auto_allow.clear();
        let (mut agent, calls) = agent_with_tool_call("dangerous", true, config);
        agent.set_approval_mode(ApprovalMode::AutoApproveAll);
        let (event_tx, event_rx) = mpsc::channel();
        let (_perm_tx, perm_rx) = mpsc::channel();

        let result = agent.execute_tool_with_policy(
            &test_tool_use("dangerous"),
            Some((&event_tx, &AtomicBool::new(false), &perm_rx)),
        );

        assert_eq!(result.unwrap(), "recorded execution");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(event_rx.try_recv().is_err(), "--auto must not prompt");
    }

    /// `--auto` widens approval; it must not override an explicit denial.
    #[test]
    fn auto_approve_all_mode_still_respects_the_deny_list() {
        let mut config = Config::default();
        config.ask.clear();
        config.deny.push("dangerous".into());
        let (mut agent, calls) = agent_with_tool_call("dangerous", true, config);
        agent.set_approval_mode(ApprovalMode::AutoApproveAll);
        let (event_tx, _event_rx) = mpsc::channel();
        let (_perm_tx, perm_rx) = mpsc::channel();

        let err = agent
            .execute_tool_with_policy(
                &test_tool_use("dangerous"),
                Some((&event_tx, &AtomicBool::new(false), &perm_rx)),
            )
            .unwrap_err();

        assert!(err.contains("denied by config"), "{err}");
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    /// The TUI path must be untouched: default mode still prompts.
    #[test]
    fn default_approval_mode_is_interactive() {
        let config = Config::default();
        let (agent, _calls) = agent_with_tool_call("dangerous", true, config);
        assert_eq!(agent.approval, ApprovalMode::Interactive);
    }

    #[test]
    fn permission_decision_parses_wire_values() {
        assert_eq!(permission_decision("allow"), PermissionDecision::Allow);
        assert_eq!(
            permission_decision("always_allow"),
            PermissionDecision::AlwaysAllow
        );
        assert_eq!(permission_decision("deny"), PermissionDecision::Deny);
        assert_eq!(
            permission_decision("discuss"),
            PermissionDecision::Discuss(None)
        );
        assert_eq!(
            permission_decision("discuss:do it another way"),
            PermissionDecision::Discuss(Some("do it another way".into()))
        );
        assert_eq!(
            permission_decision("discuss:note: colon ok"),
            PermissionDecision::Discuss(Some("note: colon ok".into()))
        );
        assert_eq!(permission_decision("nope"), PermissionDecision::Deny);
    }

    #[test]
    fn interactive_policy_fails_closed_when_permission_channel_disconnects() {
        let mut config = Config::default();
        config.ask.clear();
        config.auto_allow.clear();
        let (mut agent, calls) = agent_with_tool_call("dangerous", true, config);
        let (event_tx, _event_rx) = mpsc::channel();
        let (perm_tx, perm_rx) = mpsc::channel();
        drop(perm_tx);

        let result = agent.execute_tool_with_policy(
            &test_tool_use("dangerous"),
            Some((&event_tx, &AtomicBool::new(false), &perm_rx)),
        );

        assert!(result.unwrap_err().contains("Permission denied"));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    struct InteractiveRunMock {
        use_tools: bool,
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl llm::Provider for InteractiveRunMock {
        fn name(&self) -> &str {
            "mock"
        }

        fn default_model(&self) -> &str {
            "mock-model"
        }

        fn available_models(&self) -> Vec<llm::ModelInfo> {
            vec![llm::ModelInfo {
                id: "mock-model".into(),
                name: "Mock".into(),
                max_ctx: 1000,
            }]
        }

        fn stream_complete(
            &self,
            _messages: &[llm::Message],
            _tools: &[llm::ToolDefinition],
            _system: &str,
            _model: &str,
            _max_tokens: usize,
            _on_chunk: llm::StreamingCallback,
            _cancel: &AtomicBool,
        ) -> Result<(Vec<llm::Message>, Usage), String> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            let content = if self.use_tools {
                llm::Content::ToolUse(llm::ToolUse {
                    id: format!("tool-{call}"),
                    name: "missing-tool".into(),
                    input: "{}".into(),
                })
            } else {
                llm::Content::Text("complete".into())
            };
            Ok((
                vec![llm::Message {
                    role: "assistant".into(),
                    content,
                }],
                Usage::default(),
            ))
        }

        fn complete(
            &self,
            _messages: &[llm::Message],
            _tools: &[llm::ToolDefinition],
            _system: &str,
            _model: &str,
            _max_tokens: usize,
        ) -> Result<(Vec<llm::Message>, Usage), String> {
            unreachable!("interactive tests only use streaming completion")
        }
    }

    fn run_interactive_with_max_turns(
        max_turns: usize,
        use_tools: bool,
    ) -> (Result<(), String>, Vec<AgentEvent>, usize) {
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut config = Config::default();
        config.max_turns = max_turns;
        let mut agent = Agent::new(
            Box::new(InteractiveRunMock {
                use_tools,
                calls: calls.clone(),
            }),
            "mock-model".into(),
            crate::tools::registry::Registry::new(),
            config,
        );
        let (event_tx, event_rx) = mpsc::channel();
        let (_perm_tx, perm_rx) = mpsc::channel();

        let result = agent.run("prompt", event_tx, &AtomicBool::new(false), &perm_rx);
        let events = event_rx.try_iter().collect();
        (result, events, calls.load(Ordering::SeqCst))
    }

    #[test]
    fn interactive_run_errors_when_tool_calls_exhaust_turns() {
        let (result, events, calls) = run_interactive_with_max_turns(2, true);
        let expected = "Agent reached the maximum of 2 turns before completing";

        assert_eq!(result.unwrap_err(), expected);
        assert_eq!(calls, 2);
        assert!(events
            .iter()
            .any(|event| matches!(event, AgentEvent::Error(error) if error == expected)));
    }

    #[test]
    fn interactive_run_completes_and_emits_done_without_tool_calls() {
        let (result, events, calls) = run_interactive_with_max_turns(1, false);

        result.unwrap();
        assert_eq!(calls, 1);
        assert!(!events
            .iter()
            .any(|event| matches!(event, AgentEvent::Error(_))));
        assert!(matches!(events.last(), Some(AgentEvent::Done)));
    }

    #[test]
    fn interactive_run_defines_zero_turns_as_immediate_exhaustion() {
        let (result, events, calls) = run_interactive_with_max_turns(0, false);
        let expected = "Agent reached the maximum of 0 turns before completing";

        assert_eq!(result.unwrap_err(), expected);
        assert_eq!(calls, 0);
        assert!(events
            .iter()
            .any(|event| matches!(event, AgentEvent::Error(error) if error == expected)));
    }
}
