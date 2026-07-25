use std::io::{self, stdout, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ratatui::{
    crossterm::{
        event::{
            DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
            Event, KeyCode, KeyEventKind, KeyModifiers, MouseEventKind,
        },
        execute,
    },
    layout::{Constraint, Direction, Layout, Position},
    style::{Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, BorderType, Borders, Paragraph, Wrap},
    DefaultTerminal, Frame,
};

/// What to dump in plain-text select mode (outside the alternate screen).
#[derive(Clone, Copy)]
enum SelectDump {
    /// Most recent assistant message (or in-progress stream).
    LastAssistant,
    /// Full session transcript as plain text.
    FullTranscript,
}

use crate::agent::AgentEvent;
use crate::llm;
use crate::session;
use crate::theme::{self, Theme};

pub struct OutputLine {
    pub type_: String,
    pub content: String,
    pub tool_name: String,
    pub duration: String,
}

enum State {
    Idle,
    Running,
}

// Same frames as charmbracelet MiniDot (Grok Build / zero).
const SPINNER_CHARS: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
// Claude Code / OpenClaude-style loading verbs (subset of openclaude spinnerVerbs).
const SPINNER_VERBS: &[&str] = &[
    "Thinking",
    "Brewing",
    "Composing",
    "Crafting",
    "Crunching",
    "Deciphering",
    "Exploring",
    "Figuring",
    "Forging",
    "Generating",
    "Mulling",
    "Noodling",
    "Pondering",
    "Reasoning",
    "Sculpting",
    "Synthesizing",
    "Unraveling",
    "Working",
    "Architecting",
    "Bootstrapping",
    "Calculating",
    "Cogitating",
    "Considering",
    "Contemplating",
    "Cooking",
    "Creating",
    "Crystallizing",
    "Deliberating",
    "Determining",
    "Envisioning",
    "Herding",
    "Incubating",
    "Manifesting",
    "Marinating",
    "Moseying",
    "Percolating",
    "Reticulating",
    "Ruminating",
    "Scheming",
    "Simmering",
    "Spelunking",
    "Transmuting",
    "Wrangling",
];
/// Rows for the `/help` chrome overlay (dismiss with Esc / Enter / ?).
/// Section headers use an empty keys column; detail rows are key + description.
const HELP_ROWS: &[(&str, &str)] = &[
    ("Commands", ""),
    ("/auth", "login · logout · status · key  (xAI OAuth)"),
    ("/model", "pick or set model"),
    ("/provider", "pick or set provider"),
    ("/clear", "clear conversation"),
    ("/compact", "summarize older history now"),
    ("/cost", "token usage and estimated cost"),
    ("/theme", "TUI theme picker"),
    ("/thinking", "on|off full thinking blocks"),
    ("/suggestions", "on|off idle ready-to-send hints"),
    ("/mouse", "on|off wheel capture"),
    ("/copy", "copy last assistant message (Ctrl+Y)"),
    ("/select", "plain-text select mode (Ctrl+O)"),
    (
        "/save · /sessions · /resume · /delete",
        "session management",
    ),
    ("/skills · /mcp", "list skills and MCP servers"),
    (
        "/reset · /reset apply",
        "ChatGPT banked rate-limit resets (OpenAI OAuth)",
    ),
    ("/exit · /quit · /q", "exit Cairn"),
    ("", ""),
    ("Keys", ""),
    ("Enter", "send message"),
    ("Tab / →", "accept slash ghost or idle suggestion"),
    ("Up / Down", "scroll chat when it overflows · else history"),
    ("Ctrl+P / Ctrl+N", "prompt history"),
    ("PgUp/PgDn · Ctrl+U/D", "page / half-page scroll"),
    ("Ctrl+Home / End", "jump to top / bottom"),
    ("Wheel", "scroll transcript"),
    ("Ctrl+C", "interrupt · press again to exit when idle"),
    ("paste image", "Win Alt+V · Linux Ctrl+V · macOS Cmd+V"),
    ("Esc", "cancel pickers / close this help"),
    ("?", "shortcuts (when composer empty)"),
    ("", ""),
    ("Tips", ""),
    ("Sounds", "CAIRN_SOUND=0 to mute"),
    ("Skills", "packs as <dir>/<name>/SKILL.md"),
    ("MCP", "stdio servers in config · tools need permission"),
];
/// Rows for the `?` shortcuts panel (keys must match real bindings in handle_key).
const SHORTCUT_ROWS: &[(&str, &str)] = &[
    ("Ctrl+C", "interrupt turn · press again to exit when idle"),
    ("Enter", "send message"),
    ("Alt+V / Ctrl+V / Cmd+V", "paste clipboard image (platform)"),
    ("Tab / →", "accept slash ghost or idle suggestion"),
    (
        "Up / Down",
        "scroll chat when it overflows · else prompt history",
    ),
    ("Ctrl+P / Ctrl+N", "previous / next prompt history"),
    ("PgUp / PgDn", "page scroll"),
    ("Ctrl+U / Ctrl+D", "half-page scroll"),
    ("Ctrl+Home / End", "jump to top / bottom of chat"),
    ("Wheel", "scroll transcript"),
    ("Ctrl+Y", "copy last assistant message"),
    ("Ctrl+O", "plain-text select mode"),
    ("/", "slash commands (↑↓ select · Tab completes)"),
];
// MiniDot FPS is time.Second/12 (~83ms). Faster ticks look like flicker; slower feels sticky.
const SPINNER_INTERVAL: Duration = Duration::from_nanos(1_000_000_000 / 12);
// Cap full-frame redraws while the agent runs. Zero coalesces stream text to ~16ms
// (60fps); without this, token-rate dirty redraws thrash the terminal around the spinner.
const MIN_FRAME: Duration = Duration::from_millis(16);

struct TerminalGuard;

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        ratatui::restore();
    }
}

pub struct Tui {
    output_lines: Vec<OutputLine>,
    input_buf: String,
    cursor: usize,
    history: Vec<String>,
    hist_idx: usize,
    spinner_idx: usize,
    /// Index into SPINNER_VERBS for the current agent turn (Claude Code-style).
    spinner_verb_idx: usize,
    streaming_text: String,
    stream_thinking: String,
    state: State,
    total_usage: llm::Usage,
    version: String,
    model: String,
    provider: String,
    work_dir: String,
    show_model_picker: bool,
    picker_models: Vec<llm::ModelInfo>,
    picker_sel: usize,
    picker_scrl: usize,
    show_provider_picker: bool,
    provider_picker_list: Vec<String>,
    provider_picker_sel: usize,
    /// Whether each provider in provider_picker_list has an API key saved in the config file.
    provider_picker_keys: Vec<bool>,
    /// When Some, a confirmation prompt to remove this provider's saved API key is shown.
    confirm_remove_provider: Option<String>,
    confirm_remove_sel: usize,
    /// When Some, ask before sending the existing conversation to this provider.
    confirm_history_provider: Option<String>,
    confirm_history_sel: usize,
    awaiting_api_key: bool,
    /// After successful OAuth or API key entry from the provider picker, open the model list.
    /// Auth comes first so live model catalogs (xAI, Anthropic, …) can load.
    pending_model_after_auth: bool,
    /// Cross-provider selection stays pending until a model is confirmed, so
    /// cancelling auth or either picker cannot desynchronize the TUI and Agent.
    pending_provider_selection: Option<String>,
    /// Provider name to capture an API key for (e.g. "openrouter", "opengateway").
    /// When Some, the awaiting_api_key flow stores the key under this provider.
    api_key_target: Option<String>,
    show_command_picker: bool,
    cmd_picker_list: Vec<String>,
    cmd_picker_filtered: Vec<String>,
    cmd_picker_sel: usize,
    show_session_picker: bool,
    /// When true, Enter on the session picker deletes instead of resuming.
    session_picker_delete: bool,
    picker_sessions: Vec<session::SessionSummary>,
    picker_session_sel: usize,
    picker_session_scrl: usize,
    agent_tx: Option<mpsc::Sender<String>>,
    perm_tx: Option<mpsc::Sender<String>>,
    cancel_flag: Option<Arc<AtomicBool>>,
    dirty: bool,
    show_permission_prompt: bool,
    perm_tool_name: String,
    perm_tool_input: String,
    /// 0=Yes, 1=Yes always, 2=No, 3=Discuss (Claude Code numbered options).
    perm_selection: usize,
    /// Free-text feedback for option 4 (Discuss) only. Cleared when leaving Discuss.
    perm_discuss_buf: String,
    /// Byte cursor into `perm_discuss_buf` (Discuss only).
    perm_discuss_cursor: usize,
    /// After an LLM/provider failure: offer Switch model / Switch provider / Dismiss.
    show_recovery_prompt: bool,
    recovery_selection: usize,
    theme: Theme,
    show_theme_picker: bool,
    theme_picker_list: Vec<Theme>,
    theme_picker_sel: usize,
    /// Theme name before opening the picker (restored on Esc).
    theme_before_picker: Option<String>,
    /// Claude Code-style exit: first Ctrl+C on empty idle prompt arms this;
    /// second Ctrl+C exits. Disarmed by any other key or action.
    ctrl_c_exit_armed: bool,
    /// Transcript vertical offset (videre-style `rowoff`): first visible wrapped
    /// line of the body. When `transcript_follow` is true, view sticks to bottom.
    transcript_rowoff: usize,
    /// When true, keep the transcript pinned to the latest content (auto-scroll).
    transcript_follow: bool,
    /// Last body pane height / content height from render (for page sizes).
    last_body_h: usize,
    last_body_wrapped: usize,
    /// Active session id for autosave / resume (None until first save or resume).
    current_session_id: Option<String>,
    /// created_at for the active session (preserved across autosaves).
    session_created_at: u64,
    /// Throttle mid-turn checkpoint disk writes (wall clock).
    last_checkpoint_save: Option<Instant>,
    /// Full agent transcript (tools included) for session files.
    live_mirror: Option<session::LiveMirror>,
    /// Clipboard images attached to the next user send (cleared on send/clear).
    pending_images: Vec<llm::ImageBlock>,
    /// When true, the next `Done` event is a finished agent turn (play sound + refresh hint).
    expect_turn_notify: bool,
    /// Grayed-out ready-to-send prompt shown when the composer is empty (Tab/→ accepts).
    idle_suggestion: Option<String>,
    /// When true, stream + keep full thinking blocks. When false (default), only
    /// a short "Thought for …" marker is kept after each think phase.
    show_thinking: bool,
    /// When true, show grayed idle ready-to-send prompts. Default off.
    show_suggestions: bool,
    /// When true, terminal mouse capture is on so the wheel scrolls the
    /// transcript. Shift+drag where supported still selects text; hosts with
    /// different native-selection gestures (e.g. iTerm's Option-based
    /// selection) can use /mouse off or /select instead. Default on.
    mouse_capture: bool,
    /// Leave the TUI and print plain text so the terminal can select/copy freely.
    pending_select: Option<SelectDump>,
    /// Wall clock for the current in-flight thinking stream (for duration labels).
    thinking_started: Option<Instant>,
    /// When the current agent turn started (Claude-style spinner elapsed time).
    running_started: Option<Instant>,
    /// Bottom chrome shows `/help` overlay (like the model picker; Esc closes).
    show_help: bool,
    /// Bottom chrome shows keyboard shortcuts (`?` when the composer is empty).
    show_shortcuts: bool,
}

impl Tui {
    pub fn new(version: &str, model: &str, provider: &str, work_dir: &str) -> Self {
        Tui {
            output_lines: Vec::new(),
            input_buf: String::new(),
            cursor: 0,
            history: Vec::new(),
            hist_idx: 0,
            spinner_idx: 0,
            spinner_verb_idx: 0,
            streaming_text: String::new(),
            stream_thinking: String::new(),
            state: State::Idle,
            total_usage: llm::Usage::default(),
            version: version.to_string(),
            model: model.to_string(),
            provider: provider.to_string(),
            work_dir: work_dir.to_string(),
            show_model_picker: false,
            picker_models: Vec::new(),
            picker_sel: 0,
            picker_scrl: 0,
            show_provider_picker: false,
            provider_picker_list: Vec::new(),
            provider_picker_sel: 0,
            provider_picker_keys: Vec::new(),
            confirm_remove_provider: None,
            confirm_remove_sel: 0,
            confirm_history_provider: None,
            confirm_history_sel: 0,
            awaiting_api_key: false,
            pending_model_after_auth: false,
            pending_provider_selection: None,
            api_key_target: None,
            show_command_picker: false,
            cmd_picker_list: SLASH_COMMANDS.iter().map(|(c, _)| (*c).into()).collect(),
            cmd_picker_filtered: Vec::new(),
            cmd_picker_sel: 0,
            show_session_picker: false,
            session_picker_delete: false,
            picker_sessions: Vec::new(),
            picker_session_sel: 0,
            picker_session_scrl: 0,
            agent_tx: None,
            perm_tx: None,
            cancel_flag: None,
            dirty: false,
            show_permission_prompt: false,
            perm_tool_name: String::new(),
            perm_tool_input: String::new(),
            perm_selection: 0,
            perm_discuss_buf: String::new(),
            perm_discuss_cursor: 0,
            show_recovery_prompt: false,
            recovery_selection: 0,
            theme: theme::default_theme(),
            show_theme_picker: false,
            theme_picker_list: theme::all_themes(),
            theme_picker_sel: 0,
            theme_before_picker: None,
            ctrl_c_exit_armed: false,
            transcript_rowoff: 0,
            transcript_follow: true,
            last_body_h: 0,
            last_body_wrapped: 0,
            current_session_id: None,
            session_created_at: 0,
            last_checkpoint_save: None,
            live_mirror: None,
            pending_images: Vec::new(),
            expect_turn_notify: false,
            idle_suggestion: None,
            show_thinking: false,
            show_suggestions: false,
            mouse_capture: true,
            pending_select: None,
            thinking_started: None,
            running_started: None,
            show_help: false,
            show_shortcuts: false,
        }
    }

    pub fn set_live_mirror(&mut self, mirror: session::LiveMirror) {
        self.live_mirror = Some(mirror);
    }

    pub fn set_theme_name(&mut self, name: &str) {
        self.theme = theme::lookup(name);
        self.theme_picker_list = theme::all_themes();
        self.theme_picker_sel = self
            .theme_picker_list
            .iter()
            .position(|t| t.name == self.theme.name)
            .unwrap_or(0);
    }

    fn begin_running(&mut self) {
        self.state = State::Running;
        self.running_started = Some(Instant::now());
        // Allow the first mid-turn Checkpoint of this run to hit disk even if
        // we just saved at the previous Done (throttle would otherwise skip it).
        self.last_checkpoint_save = None;
        // New verb each turn (Claude Code / OpenClaude spinner style).
        let seed = self
            .spinner_idx
            .wrapping_add(self.total_usage.input_tokens as usize)
            .wrapping_add(self.output_lines.len())
            .wrapping_add(1);
        self.spinner_verb_idx = seed % SPINNER_VERBS.len();
    }

    pub fn set_show_thinking(&mut self, show: bool) {
        self.show_thinking = show;
    }

    pub fn set_show_suggestions(&mut self, show: bool) {
        self.show_suggestions = show;
        if show {
            self.refresh_idle_suggestion();
        } else {
            self.idle_suggestion = None;
        }
    }

    fn set_mouse_capture(&mut self, on: bool) {
        if on == self.mouse_capture {
            return;
        }
        if on {
            if execute!(stdout(), EnableMouseCapture).is_ok() {
                self.mouse_capture = true;
            }
        } else if execute!(stdout(), DisableMouseCapture).is_ok() {
            self.mouse_capture = false;
        }
    }

    /// Copy the most recent assistant text to the OS clipboard.
    fn copy_last_assistant_to_clipboard(&mut self) {
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

    fn select_dump_text(&self, kind: SelectDump) -> String {
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

    pub fn set_agent_tx(&mut self, tx: mpsc::Sender<String>) {
        self.agent_tx = Some(tx);
    }

    pub fn set_perm_tx(&mut self, tx: mpsc::Sender<String>) {
        self.perm_tx = Some(tx);
    }

    pub fn set_cancel_flag(&mut self, flag: Arc<AtomicBool>) {
        self.cancel_flag = Some(flag);
    }

    pub fn run(&mut self, rx: mpsc::Receiver<AgentEvent>) -> Result<(), String> {
        let mut terminal = ratatui::init();
        let _terminal_guard = TerminalGuard;
        terminal.clear().map_err(|e| e.to_string())?;
        // Wheel scroll for transcript history. Shift+drag where supported is
        // handled by the terminal host (selects text without sending events
        // to the app); hosts with different native-selection gestures (e.g.
        // iTerm's Option-based selection) can use /mouse off or /select instead.
        if execute!(stdout(), EnableMouseCapture).is_ok() {
            self.mouse_capture = true;
        }
        // Bracketed paste: terminals deliver pasted blobs as Event::Paste instead of
        // fake keystrokes, so emoji / multi-byte Unicode and multi-line text work.
        let _ = execute!(stdout(), EnableBracketedPaste);

        let mut result = Ok(());
        let mut last_spinner_update = std::time::Instant::now();
        let mut last_draw = std::time::Instant::now();
        let mut needs_rebuild = false;
        self.dirty = true;

        'outer: loop {
            if matches!(self.state, State::Running) {
                let mut got_event = false;
                while let Ok(event) = rx.try_recv() {
                    got_event = true;
                    match event {
                        AgentEvent::Text(t) => {
                            self.streaming_text.push_str(&t);
                        }
                        AgentEvent::Thinking(t) => {
                            if self.thinking_started.is_none() {
                                self.thinking_started = Some(Instant::now());
                            }
                            self.stream_thinking.push_str(&t);
                        }
                        AgentEvent::ToolUse(name, input) => {
                            self.flush_streaming();
                            self.output_lines.push(OutputLine {
                                type_: "tool_use".into(),
                                content: input,
                                tool_name: name,
                                duration: String::new(),
                            });
                        }
                        AgentEvent::ToolResult(name, _inp, out) => {
                            self.output_lines.push(OutputLine {
                                type_: "tool_result".into(),
                                content: out,
                                tool_name: name,
                                duration: String::new(),
                            });
                        }
                        AgentEvent::Error(e) => {
                            self.flush_streaming();
                            let is_llm = e.starts_with("LLM error:");
                            self.output_lines.push(OutputLine {
                                type_: "error".into(),
                                content: crate::redact::redact_secrets(&e),
                                tool_name: String::new(),
                                duration: String::new(),
                            });
                            // Offer a manual model/provider switch after LLM failures only
                            // (not for compact/session errors). Never silent multi-provider fallback.
                            if is_llm {
                                self.show_recovery_prompt = true;
                                self.recovery_selection = 0;
                                crate::notify::play(crate::notify::Kind::Attention);
                                self.refresh_idle_suggestion();
                            }
                        }
                        AgentEvent::PermissionRequest(name, input) => {
                            self.flush_streaming();
                            self.show_permission_prompt = true;
                            self.perm_tool_name = name;
                            self.perm_tool_input = input;
                            self.perm_selection = 0;
                            self.perm_discuss_buf.clear();
                            self.perm_discuss_cursor = 0;
                            crate::notify::play(crate::notify::Kind::Attention);
                            self.refresh_idle_suggestion();
                        }
                        AgentEvent::TurnEnd(u) => {
                            self.total_usage.input_tokens += u.input_tokens;
                            self.total_usage.output_tokens += u.output_tokens;
                        }
                        AgentEvent::Compacted(n) => {
                            self.flush_streaming();
                            self.output_lines.push(OutputLine {
                                type_: "system".into(),
                                content: format!("Compacted {n} earlier messages into a summary."),
                                tool_name: String::new(),
                                duration: String::new(),
                            });
                        }
                        AgentEvent::Checkpoint => {
                            // Mid-turn durable flush (throttled) so crashes do not
                            // drop everything since the previous Done/exit save.
                            self.checkpoint_session();
                        }
                        AgentEvent::Done => {
                            self.flush_streaming();
                            self.state = State::Idle;
                            self.running_started = None;
                            // Always flush on turn end (bypass throttle).
                            self.autosave_session(false);
                            if self.expect_turn_notify {
                                self.expect_turn_notify = false;
                                // Permission/recovery already beeped Attention; skip double tone.
                                if !self.show_permission_prompt && !self.show_recovery_prompt {
                                    crate::notify::play(crate::notify::Kind::Done);
                                }
                            }
                            self.refresh_idle_suggestion();
                            if self.pending_model_after_auth {
                                let target = self
                                    .pending_provider_selection
                                    .clone()
                                    .unwrap_or_else(|| self.provider.clone());
                                if crate::config::has_usable_credential(&target) {
                                    // Signed in - now pick a model (live catalog available).
                                    self.pending_model_after_auth = false;
                                    self.output_lines.push(OutputLine {
                                        type_: "system".into(),
                                        content: format!(
                                            "Signed in to {}. Choose a model.",
                                            target
                                        ),
                                        tool_name: String::new(),
                                        duration: String::new(),
                                    });
                                    self.open_model_picker();
                                } else {
                                    // OAuth failed; keep pending so API key still continues to model picker.
                                    self.output_lines.push(OutputLine {
                                        type_: "system".into(),
                                        content: format!(
                                            "OAuth for {} did not complete. Paste an API key below, or run `/auth login {}` again.",
                                            target, target
                                        ),
                                        tool_name: String::new(),
                                        duration: String::new(),
                                    });
                                    self.begin_api_key_prompt(&target);
                                }
                            }
                        }
                    }
                }
                if got_event {
                    self.dirty = true;
                }
            }

            if matches!(self.state, State::Idle) {
                // Paint pending frames before blocking on input. Otherwise the
                // first frame (startup), post-Done UI, and any other idle dirty
                // never appear until the user scrolls or types.
                if !self.dirty && !needs_rebuild {
                    match ratatui::crossterm::event::read() {
                        Ok(Event::Key(key)) => {
                            if !self.handle_key(key) {
                                break 'outer;
                            } else {
                                self.dirty = true;
                            }
                        }
                        Ok(Event::Paste(data)) => {
                            if self.handle_paste(&data) {
                                self.dirty = true;
                            }
                        }
                        Ok(Event::Mouse(m)) => {
                            if self.handle_mouse(m.kind) {
                                self.dirty = true;
                            }
                        }
                        Ok(Event::Resize(_, _)) => {
                            needs_rebuild = true;
                            self.dirty = true;
                        }
                        Err(e) => {
                            result = Err(format!("Event error: {e}"));
                            break 'outer;
                        }
                        _ => {}
                    }
                }
            } else {
                // Advance the MiniDot frame on its own clock (not on every stream dirty).
                // Re-issuing a tick every event-loop lap is what makes the glyph flash.
                if last_spinner_update.elapsed() >= SPINNER_INTERVAL {
                    self.spinner_idx = self.spinner_idx.wrapping_add(1);
                    last_spinner_update = std::time::Instant::now();
                    self.dirty = true;
                }

                // Sleep until the next useful wake. Cap at MIN_FRAME so stream
                // chunks in mpsc still drain at ~60fps without a 1ms busy-poll.
                let until_spinner = SPINNER_INTERVAL
                    .checked_sub(last_spinner_update.elapsed())
                    .unwrap_or(Duration::ZERO);
                let poll_for = if self.dirty {
                    MIN_FRAME
                        .checked_sub(last_draw.elapsed())
                        .unwrap_or(Duration::ZERO)
                } else {
                    until_spinner.min(MIN_FRAME)
                };
                let event_avail = ratatui::crossterm::event::poll(poll_for).unwrap_or(false);
                if event_avail {
                    match ratatui::crossterm::event::read() {
                        Ok(Event::Key(key)) => {
                            if key.kind == KeyEventKind::Press {
                                self.handle_key(key);
                                self.dirty = true;
                            }
                        }
                        Ok(Event::Paste(data)) => {
                            if self.handle_paste(&data) {
                                self.dirty = true;
                            }
                        }
                        Ok(Event::Mouse(m)) => {
                            if self.handle_mouse(m.kind) {
                                self.dirty = true;
                            }
                        }
                        Ok(Event::Resize(_, _)) => {
                            needs_rebuild = true;
                            self.dirty = true;
                        }
                        _ => {}
                    }
                }
            }

            // Plain-text select mode: leave alt-screen so Windows Terminal (and
            // others) can drag-select freely, then re-enter the TUI.
            if let Some(kind) = self.pending_select.take() {
                let text = self.select_dump_text(kind);
                if let Err(e) = enter_plain_select_mode(&mut terminal, &text) {
                    result = Err(e);
                    break 'outer;
                }
                if self.mouse_capture {
                    let _ = execute!(stdout(), EnableMouseCapture);
                } else {
                    let _ = execute!(stdout(), DisableMouseCapture);
                }
                let _ = execute!(stdout(), EnableBracketedPaste);
                needs_rebuild = true;
                self.dirty = true;
            }

            if needs_rebuild {
                let _ = terminal.clear();
                needs_rebuild = false;
            }

            if self.dirty {
                // While Running, coalesce stream-driven dirties to ~60fps so token
                // rate cannot thrash the terminal around the spinner. Idle always
                // paints immediately so keystrokes stay snappy.
                if matches!(self.state, State::Running) && last_draw.elapsed() < MIN_FRAME {
                    continue;
                }
                if let Err(e) = terminal.draw(|f| self.render(f)) {
                    result = Err(format!("Render error: {e}"));
                    break 'outer;
                }
                last_draw = std::time::Instant::now();
                self.dirty = false;
            }
        }

        // Persist conversation on clean exit so the last session is not lost.
        self.autosave_session(false);
        if self.mouse_capture {
            let _ = execute!(stdout(), DisableMouseCapture);
        }
        let _ = execute!(stdout(), DisableBracketedPaste);
        // TerminalGuard drop restores ratatui/raw mode.
        result
    }

    /// Insert clipboard / bracketed-paste text into the composer at the cursor.
    /// Returns true if the buffer changed (needs redraw).
    fn handle_paste(&mut self, data: &str) -> bool {
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
        let allow = self.awaiting_api_key
            || (!self.show_model_picker
                && !self.show_provider_picker
                && !self.show_theme_picker
                && !self.show_session_picker
                && !self.show_help
                && !self.show_shortcuts
                && !self.show_permission_prompt);
        if !allow {
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
    fn handle_mouse(&mut self, kind: MouseEventKind) -> bool {
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

    fn handle_key(&mut self, key: ratatui::crossterm::event::KeyEvent) -> bool {
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
                if self.cursor > 0 && !self.show_model_picker && !self.show_provider_picker {
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
                if self.cursor < self.input_buf.len() && !self.show_model_picker {
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
                if self.awaiting_api_key
                    || (!self.show_model_picker
                        && !self.show_provider_picker
                        && !self.show_theme_picker
                        && !self.show_session_picker
                        && !self.show_help
                        && !self.show_shortcuts)
                {
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

    fn handle_command(&mut self, cmd: &str) -> bool {
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
                if let Some(ref d) = cfg.skills_dir {
                    std::env::set_var("CAIRN_SKILLS_DIR", d);
                }
                let list = crate::skills::load_skills();
                let dir = cfg
                    .skills_dir
                    .as_ref()
                    .map(std::path::PathBuf::from)
                    .unwrap_or_else(crate::skills::default_skills_dir);
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
                    let t = theme::lookup(&name);
                    let applied = t.name.to_string();
                    self.apply_theme_preference(
                        t,
                        self.theme.clone(),
                        crate::config::save_theme(&applied),
                    );
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

    fn sessions_dir(&self) -> String {
        crate::config::sessions_dir()
    }

    fn open_model_picker(&mut self) {
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

    fn open_provider_picker(&mut self) {
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

    fn provider_switch_needs_confirmation(&self, provider: &str) -> bool {
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

    fn begin_provider_selection(&mut self, name: String) {
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

    fn cancel_pending_provider_selection(&mut self) {
        self.pending_model_after_auth = false;
        self.pending_provider_selection = None;
    }

    /// Claude Code Ctrl+C: interrupt running work; clear prompt when idle with
    /// text; on empty idle prompt, arm exit then quit on a second press.
    /// Returns `false` to exit the TUI.
    fn handle_ctrl_c(&mut self) -> bool {
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

    fn begin_api_key_prompt(&mut self, provider: &str) {
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
    fn finish_provider_model_selection(&mut self, provider: String, model: String) {
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

    fn apply_theme_preference(
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

    /// Prefer the agent's full transcript (tools included); fall back to TUI lines.
    fn session_snapshot(&self) -> (Vec<llm::Message>, u64, u64) {
        if let Some(mirror) = &self.live_mirror {
            if let Ok(g) = mirror.lock() {
                if !g.messages.is_empty() {
                    return (g.messages.clone(), g.tokens_in, g.tokens_out);
                }
            }
        }
        let messages = self
            .output_lines
            .iter()
            .filter_map(|l| {
                if l.type_ == "user" {
                    Some(llm::Message {
                        role: "user".into(),
                        content: llm::Content::Text(l.content.clone()),
                    })
                } else if l.type_ == "text" {
                    Some(llm::Message {
                        role: "assistant".into(),
                        content: llm::Content::Text(l.content.clone()),
                    })
                } else if l.type_ == "tool_use" {
                    Some(llm::Message {
                        role: "assistant".into(),
                        content: llm::Content::ToolUse(llm::ToolUse {
                            id: String::new(),
                            name: l.tool_name.clone(),
                            input: l.content.clone(),
                        }),
                    })
                } else if l.type_ == "tool_result" {
                    Some(llm::Message {
                        role: "user".into(),
                        content: llm::Content::ToolResult(llm::ToolResult {
                            tool_use_id: String::new(),
                            content: l.content.clone(),
                        }),
                    })
                } else {
                    None
                }
            })
            .collect();
        (
            messages,
            self.total_usage.input_tokens,
            self.total_usage.output_tokens,
        )
    }

    /// Platform paste chord for clipboard images:
    /// - Windows: Alt+V
    /// - Linux / other Unix: Ctrl+V
    /// - macOS: Cmd (SUPER)+V
    fn is_image_paste_chord(&self, key: ratatui::crossterm::event::KeyEvent) -> bool {
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

    fn paste_clipboard_image(&mut self) {
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

    /// Mid-turn durable checkpoint. Throttled so rapid tool loops do not hammer
    /// the disk, but always writes at least every few seconds while history grows.
    fn checkpoint_session(&mut self) {
        const MIN_CHECKPOINT_INTERVAL: Duration = Duration::from_secs(2);
        if let Some(last) = self.last_checkpoint_save {
            if last.elapsed() < MIN_CHECKPOINT_INTERVAL {
                return;
            }
        }
        self.autosave_session(false);
    }

    /// Ensure we have a stable session id before the first disk write so mid-turn
    /// checkpoints and the final Done save land on the same file.
    fn ensure_session_identity(&mut self) {
        if self.current_session_id.is_none() {
            self.current_session_id = Some(session::new_id());
        }
        if self.session_created_at == 0 {
            self.session_created_at = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
        }
    }

    /// Save (or update) the current session. When `announce` is true, print a
    /// system line (manual `/save`). Autosave stays quiet unless it fails.
    fn autosave_session(&mut self, announce: bool) {
        let (messages, tokens_in, tokens_out) = self.session_snapshot();
        if messages.is_empty() {
            if announce {
                self.output_lines.push(OutputLine {
                    type_: "system".into(),
                    content: "Nothing to save - no conversation yet.".into(),
                    tool_name: String::new(),
                    duration: String::new(),
                });
            }
            return;
        }

        self.ensure_session_identity();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let id = self
            .current_session_id
            .clone()
            .unwrap_or_else(session::new_id);
        let created_at = if self.session_created_at > 0 {
            self.session_created_at
        } else {
            now
        };
        let msg_count = messages.len();
        let sess = session::Session {
            id: id.clone(),
            model: self.model.clone(),
            provider: self.provider.clone(),
            messages,
            tokens_in,
            tokens_out,
            created_at,
            updated_at: now,
        };
        match session::save(&self.sessions_dir(), &sess) {
            Ok(()) => {
                self.current_session_id = Some(id.clone());
                self.session_created_at = created_at;
                self.last_checkpoint_save = Some(Instant::now());
                if announce {
                    let short = if id.len() >= 8 { &id[..8] } else { id.as_str() };
                    self.output_lines.push(OutputLine {
                        type_: "system".into(),
                        content: format!(
                            "Session saved: {short} ({msg_count} msgs) → {}",
                            self.sessions_dir()
                        ),
                        tool_name: String::new(),
                        duration: String::new(),
                    });
                }
            }
            Err(e) => {
                // Always surface write failures (including silent autosave).
                self.output_lines.push(OutputLine {
                    type_: "error".into(),
                    content: format!("Failed to save session: {e}"),
                    tool_name: String::new(),
                    duration: String::new(),
                });
            }
        }
    }

    fn save_session(&mut self) {
        self.autosave_session(true);
    }

    fn list_sessions(&mut self) {
        let sessions = session::list(&self.sessions_dir()).unwrap_or_default();
        if sessions.is_empty() {
            self.output_lines.push(OutputLine {
                type_: "system".into(),
                content: "No saved sessions.".into(),
                tool_name: String::new(),
                duration: String::new(),
            });
            return;
        }
        let mut msg = String::from("Saved sessions:\n");
        for s in &sessions {
            let time_str = format_timestamp(s.updated_at);
            let summary = truncate_summary(&s.summary, 60);
            msg.push_str(&format!(
                "  {}  {}  {} msgs  {}\n",
                &s.id[..8],
                s.model,
                s.msg_count,
                time_str
            ));
            if !summary.is_empty() {
                msg.push_str(&format!("    {summary}\n"));
            }
        }
        self.output_lines.push(OutputLine {
            type_: "system".into(),
            content: msg.trim_end().to_string(),
            tool_name: String::new(),
            duration: String::new(),
        });
    }

    fn delete_session(&mut self, id: &str) {
        let short = if id.len() >= 8 { &id[..8] } else { id };
        match session::delete(&self.sessions_dir(), id) {
            Ok(()) => {
                self.output_lines.push(OutputLine {
                    type_: "system".into(),
                    content: format!("Deleted session {short}."),
                    tool_name: String::new(),
                    duration: String::new(),
                });
            }
            Err(e) => {
                self.output_lines.push(OutputLine {
                    type_: "error".into(),
                    content: format!("Failed to delete session: {e}"),
                    tool_name: String::new(),
                    duration: String::new(),
                });
            }
        }
    }

    fn resume_session(&mut self, id: &str) {
        match session::load(&self.sessions_dir(), id) {
            Ok(sess) => {
                // Rebuild TUI transcript including tool calls/results for continuity.
                let mut lines = Vec::new();
                // Pair each tool_result with the preceding tool_use name so compact
                // display rules still apply after /resume (results alone have no name).
                let mut pending_tool_name = String::new();
                for msg in &sess.messages {
                    match &msg.content {
                        llm::Content::Text(t) => {
                            lines.push(OutputLine {
                                type_: if msg.role == "user" {
                                    "user".into()
                                } else {
                                    "text".into()
                                },
                                content: t.clone(),
                                tool_name: String::new(),
                                duration: String::new(),
                            });
                        }
                        llm::Content::User(blocks) => {
                            lines.push(OutputLine {
                                type_: "user".into(),
                                content: blocks.display_label(),
                                tool_name: String::new(),
                                duration: String::new(),
                            });
                        }
                        llm::Content::Thinking(t) => {
                            if self.show_thinking {
                                lines.push(OutputLine {
                                    type_: "thinking".into(),
                                    content: t.clone(),
                                    tool_name: String::new(),
                                    duration: String::new(),
                                });
                            } else if !t.trim().is_empty() {
                                // Hidden mode: keep a Claude Code-style marker, not the body.
                                lines.push(OutputLine {
                                    type_: "thinking_summary".into(),
                                    content: "Thought".into(),
                                    tool_name: String::new(),
                                    duration: String::new(),
                                });
                            }
                        }
                        llm::Content::ToolUse(tu) => {
                            pending_tool_name = tu.name.clone();
                            lines.push(OutputLine {
                                type_: "tool_use".into(),
                                content: tu.input.clone(),
                                tool_name: tu.name.clone(),
                                duration: String::new(),
                            });
                        }
                        llm::Content::ToolResult(tr) => {
                            let name = if pending_tool_name.is_empty() {
                                "tool".into()
                            } else {
                                std::mem::take(&mut pending_tool_name)
                            };
                            lines.push(OutputLine {
                                type_: "tool_result".into(),
                                content: tr.content.clone(),
                                tool_name: name,
                                duration: String::new(),
                            });
                        }
                    }
                }
                self.output_lines = lines;
                self.total_usage = llm::Usage {
                    input_tokens: sess.tokens_in,
                    output_tokens: sess.tokens_out,
                    cache_read: 0,
                    cache_create: 0,
                };
                // Seed the live mirror so the next autosave keeps full history.
                if let Some(mirror) = &self.live_mirror {
                    if let Ok(mut g) = mirror.lock() {
                        g.messages = sess.messages.clone();
                        g.tokens_in = sess.tokens_in;
                        g.tokens_out = sess.tokens_out;
                    }
                }
                self.model = sess.model.clone();
                self.provider = sess.provider.clone();
                self.current_session_id = Some(sess.id.clone());
                self.session_created_at = if sess.created_at > 0 {
                    sess.created_at
                } else {
                    sess.updated_at
                };

                if let Some(tx) = &self.agent_tx {
                    let _ = tx.send(format!("__switch__:{}:{}", sess.provider, sess.model));
                    let _ = tx.send(format!("__load_session__:{}", sess.id));
                }
                let short = if sess.id.len() >= 8 {
                    &sess.id[..8]
                } else {
                    sess.id.as_str()
                };
                self.output_lines.push(OutputLine {
                    type_: "system".into(),
                    content: format!(
                        "Resumed session {short} (model: {}, messages: {})",
                        sess.model,
                        sess.messages.len()
                    ),
                    tool_name: String::new(),
                    duration: String::new(),
                });
            }
            Err(e) => {
                self.output_lines.push(OutputLine {
                    type_: "error".into(),
                    content: format!("Failed to load session: {e}"),
                    tool_name: String::new(),
                    duration: String::new(),
                });
            }
        }
    }

    fn render(&mut self, f: &mut Frame) {
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
                        &s.id[..8],
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

    fn flush_streaming(&mut self) {
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

    fn picker_visible_height(&self) -> usize {
        let h = terminal_height().unwrap_or(24).saturating_sub(10);
        if h < 3 {
            3
        } else {
            h.min(self.picker_models.len())
        }
    }

    pub fn set_picker_models(&mut self, models: Vec<llm::ModelInfo>) {
        self.picker_models = models;
    }

    pub fn add_output_line(&mut self, line: OutputLine) {
        self.output_lines.push(line);
    }

    fn update_cmd_picker(&mut self) {
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
            .map(|s| {
                if s.id.len() >= 8 {
                    s.id[..8].to_string()
                } else {
                    s.id
                }
            })
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
    fn selected_slash_completion(&self) -> Option<&str> {
        self.cmd_picker_filtered
            .get(self.cmd_picker_sel)
            .map(|s| s.as_str())
    }

    /// Refresh the grayed ready-to-send prompt in the empty composer.
    fn refresh_idle_suggestion(&mut self) {
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
    fn apply_slash_completion(&mut self, completion: &str) {
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

/// Leave the ratatui alt-screen and print plain text so the host terminal can
/// drag-select (Windows Terminal does not reliably select inside a redrawing TUI).
fn enter_plain_select_mode(terminal: &mut DefaultTerminal, text: &str) -> Result<(), String> {
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

/// Wire format for multimodal user turns on the agent command channel.
fn encode_user_json_cmd(text: &str, images: &[llm::ImageBlock]) -> String {
    let mut imgs = String::from("[");
    for (i, img) in images.iter().enumerate() {
        if i > 0 {
            imgs.push(',');
        }
        imgs.push_str(&format!(
            "{{\"media_type\":\"{}\",\"data\":\"{}\"}}",
            escape_json_simple(&img.media_type),
            escape_json_simple(&img.data_base64)
        ));
    }
    imgs.push(']');
    format!(
        "__user_json__:{{\"text\":\"{}\",\"images\":{imgs}}}",
        escape_json_simple(text)
    )
}

fn escape_json_simple(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
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

#[cfg(test)]
mod user_blocks_label_tests {
    use super::*;

    #[test]
    fn display_label_covers_text_and_images() {
        assert_eq!(llm::UserBlocks::text_only("hi").display_label(), "hi");
        assert_eq!(
            llm::UserBlocks {
                text: String::new(),
                images: vec![llm::ImageBlock {
                    media_type: "image/png".into(),
                    data_base64: "x".into(),
                }],
            }
            .display_label(),
            "[image]"
        );
        assert_eq!(
            llm::UserBlocks {
                text: "look".into(),
                images: vec![
                    llm::ImageBlock {
                        media_type: "image/png".into(),
                        data_base64: "a".into(),
                    },
                    llm::ImageBlock {
                        media_type: "image/jpeg".into(),
                        data_base64: "b".into(),
                    },
                ],
            }
            .display_label(),
            "look\n[2 images]"
        );
    }
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

    fn line(type_: &str, content: &str, tool: &str) -> OutputLine {
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

    fn press(tui: &mut Tui, code: KeyCode) {
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

    fn all() -> Vec<String> {
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

fn char_width(c: char) -> usize {
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

fn line_is_blank(line: &Line<'_>) -> bool {
    line.spans.is_empty() || line.spans.iter().all(|s| s.content.trim().is_empty())
}

fn display_width(s: &str) -> usize {
    s.chars().map(char_width).sum()
}

/// Pad or truncate `s` to exactly `width` terminal columns (display width).
fn pad_to_display_width(s: &str, width: usize) -> String {
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

fn truncate_summary(summary: &str, max_chars: usize) -> String {
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

fn terminal_height() -> Option<usize> {
    std::env::var("LINES")
        .ok()
        .and_then(|v| v.parse().ok())
        .or(Some(24))
}

fn format_timestamp(ts: u64) -> String {
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

fn total_wrapped(lines: &[Line], width: usize) -> usize {
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
fn truncate_display(s: &str, head_lines: usize, tail_lines: usize) -> String {
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

fn permission_risk_warning(tool_name: &str) -> Option<&'static str> {
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
fn format_permission_tool_input(input: &str) -> Vec<String> {
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
fn compact_tool_arg_hint(input: &str) -> String {
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
fn infer_tool_display_kind<'a>(tool_name: &'a str, content: &str) -> &'a str {
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
fn compact_tool_result_display(kind: &str, content: &str) -> String {
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
