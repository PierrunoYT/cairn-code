//! The terminal UI: the [`Tui`] event loop plus the modules it delegates to
//! for input, rendering, commands, and session handling.

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

use crate::agent::AgentEvent;
use crate::llm;
use crate::session;
use crate::theme::{self, Theme};

mod clipboard;
mod commands;
mod completion;
mod input;
mod render;
mod session_ops;
mod text;
mod tool_display;

// The leaf helper modules hold free functions the rest of the TUI calls
// constantly; re-exporting them here means every submodule picks them up
// through its own `use super::*`.
use clipboard::*;
use completion::*;
use text::*;
use tool_display::*;

pub use text::sanitize_terminal_output;

/// What to dump in plain-text select mode (outside the alternate screen).
#[derive(Clone, Copy)]
enum SelectDump {
    /// Most recent assistant message (or in-progress stream).
    LastAssistant,
    /// Full session transcript as plain text.
    FullTranscript,
}

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

    pub fn set_picker_models(&mut self, models: Vec<llm::ModelInfo>) {
        self.picker_models = models;
    }

    pub fn add_output_line(&mut self, line: OutputLine) {
        self.output_lines.push(line);
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
