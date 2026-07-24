//! Session snapshot, autosave, and resume.

use super::*;

impl Tui {
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

    /// Mid-turn durable checkpoint. Throttled so rapid tool loops do not hammer
    /// the disk, but always writes at least every few seconds while history grows.
    pub(super) fn checkpoint_session(&mut self) {
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
    pub(super) fn ensure_session_identity(&mut self) {
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
    pub(super) fn autosave_session(&mut self, announce: bool) {
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

    pub(super) fn save_session(&mut self) {
        self.autosave_session(true);
    }

    pub(super) fn list_sessions(&mut self) {
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

    pub(super) fn delete_session(&mut self, id: &str) {
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

    pub(super) fn resume_session(&mut self, id: &str) {
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
}
