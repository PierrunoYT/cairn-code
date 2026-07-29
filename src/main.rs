use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::{io::Read, process::ExitCode};

use cairn_code::agent::{Agent, AgentEvent, ApprovalMode};
use cairn_code::config::{self, Config};
use cairn_code::llm::provider;
use cairn_code::{http_client, llm, oauth, session, skills, tools, tui};

fn main() -> ExitCode {
    let version = env!("CARGO_PKG_VERSION");
    let mut is_print_mode = false;
    let mut initial_prompt: Option<String> = None;

    let raw_args: Vec<String> = std::env::args().collect();
    let is_exec_mode = raw_args.len() > 1 && raw_args[1] == "exec";
    let mut exec_cwd: Option<String> = None;
    let mut exec_resume: Option<String> = None;
    let mut exec_init_session_id: Option<String> = None;
    let mut exec_auto_approve = false;

    if is_exec_mode {
        let mut i = 2;
        while i < raw_args.len() {
            match raw_args[i].as_str() {
                "-C" | "--cwd" => {
                    if i + 1 < raw_args.len() {
                        exec_cwd = Some(raw_args[i + 1].clone());
                        i += 1;
                    }
                }
                "--resume" => {
                    if i + 1 < raw_args.len() {
                        exec_resume = Some(raw_args[i + 1].clone());
                        i += 1;
                    }
                }
                "--init-session-id" => {
                    if i + 1 < raw_args.len() {
                        exec_init_session_id = Some(raw_args[i + 1].clone());
                        i += 1;
                    }
                }
                "--input-format" | "--output-format" => {
                    if i + 1 < raw_args.len() {
                        i += 1;
                    }
                }
                // Valueless: consuming the next argument here used to swallow
                // whatever followed (`exec --auto --resume ID` lost --resume).
                "--auto" => exec_auto_approve = true,
                "--no-completion-gate" => {}
                _ => {}
            }
            i += 1;
        }
        if let Some(ref dir) = exec_cwd {
            let _ = std::env::set_current_dir(dir);
        }
    } else {
        for arg in std::env::args().skip(1) {
            match arg.as_str() {
                "-p" | "--print" => is_print_mode = true,
                "-h" | "--help" => {
                    print_help(version);
                    return ExitCode::SUCCESS;
                }
                "-v" | "--version" => {
                    println!("cairn-code {version}");
                    return ExitCode::SUCCESS;
                }
                arg if !arg.starts_with('-') => {
                    if initial_prompt.is_some() {
                        eprintln!("Error: unexpected positional argument '{arg}'");
                        eprintln!(
                            "Pass the prompt as one quoted argument. See '--help' for usage."
                        );
                        return ExitCode::FAILURE;
                    }
                    initial_prompt = Some(arg.to_string());
                }
                _ => {
                    eprintln!("Error: unknown option '{arg}'");
                    eprintln!("See '--help' for usage.");
                    return ExitCode::FAILURE;
                }
            }
        }
    }

    let cfg = match Config::load() {
        Ok(cfg) => cfg,
        Err(error) => {
            eprintln!("Error loading configuration: {error}");
            std::process::exit(1);
        }
    };
    // Off by default (H-03): only write request metadata to disk when the
    // user has explicitly opted in via config or CAIRN_DEBUG_HTTP=1.
    http_client::set_debug_logging_enabled(cfg.debug_log_requests);
    let provider_name = std::env::var("CAIRN_PROVIDER").unwrap_or(cfg.default_provider.clone());
    let model_name = std::env::var("CAIRN_MODEL").unwrap_or(cfg.default_model.clone());

    let mut providers = provider::default_providers();
    let (p_name, p_model) = if let Some(p) = providers.get(&provider_name) {
        let model = if model_name.is_empty() {
            p.default_model().to_string()
        } else {
            model_name
        };
        (provider_name, model)
    } else {
        (
            "anthropic".to_string(),
            "claude-sonnet-4-20250514".to_string(),
        )
    };

    let chosen_provider = providers
        .remove(&p_name)
        .unwrap_or_else(|| provider::default_providers().into_values().next().unwrap());

    let models = chosen_provider.available_models();
    let provider_name_str = chosen_provider.name().to_string();

    // Skills: optional dir override from config, else CAIRN_SKILLS_DIR / defaults.
    if let Some(dir) = cfg.skills_dir.clone() {
        std::env::set_var("CAIRN_SKILLS_DIR", dir);
    }
    let skills = skills::load_skills();
    let skills_for_agent = skills.clone();
    let (tool_registry, mcp_runtime) = tools::registry::build_registry(skills, &cfg.mcp);
    let mcp_warnings = mcp_runtime.warnings.clone();
    let mcp_tool_count = mcp_runtime.tool_names.len();
    let skill_count = skills_for_agent.len();
    // Keep MCP processes alive for the agent thread lifetime.
    let _mcp_keepalive = mcp_runtime;

    let work_dir = std::env::current_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| ".".to_string());

    if is_exec_mode {
        let mut input_line = String::new();
        if let Err(error) = std::io::stdin().read_line(&mut input_line) {
            eprintln!("Error: failed to read input event from stdin: {error}");
            return ExitCode::FAILURE;
        }
        let prompt = match serde_json::from_str::<serde_json::Value>(&input_line) {
            Ok(v) => v
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or(&input_line)
                .to_string(),
            Err(_) => input_line.trim().to_string(),
        };

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let session_id = exec_resume
            .or(exec_init_session_id)
            .unwrap_or_else(|| format!("cairn-session-{now_ms}"));
        let run_id = format!("run-{now_ms}");

        println!(
            "{}",
            serde_json::json!({
                "schemaVersion": 2,
                "type": "run_start",
                "runId": run_id,
                "sessionId": session_id,
            })
        );

        let (event_tx, event_rx) = mpsc::channel::<AgentEvent>();
        // exec has no user behind it, so nothing may answer an approval
        // prompt. The channel exists only to satisfy `run`'s signature; the
        // agent's approval mode below is what actually decides authorization.
        let (_perm_tx, perm_rx) = mpsc::channel::<String>();
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel2 = cancel.clone();

        let mut agent = Agent::new_with_skills(
            chosen_provider,
            p_model,
            tool_registry,
            cfg,
            skills_for_agent,
        );
        // Without --auto, exec fails closed exactly like --print: a tool that
        // needs approval is refused rather than run unattended, and explicit
        // `auto_allow` config is the only way to permit one.
        agent.set_approval_mode(if exec_auto_approve {
            eprintln!(
                "warning: --auto approves every tool call in this run without prompting; \
                 only the config deny list still applies"
            );
            ApprovalMode::AutoApproveAll
        } else {
            ApprovalMode::FailClosed
        });

        thread::spawn(move || {
            let _ = agent.run(&prompt, event_tx, &cancel2, &perm_rx);
        });

        let mut final_text = String::new();
        while let Ok(event) = event_rx.recv() {
            match event {
                AgentEvent::Text(t) | AgentEvent::Thinking(t) => {
                    final_text.push_str(&t);
                    println!(
                        "{}",
                        serde_json::json!({
                            "schemaVersion": 2,
                            "type": "text",
                            "runId": run_id,
                            "sessionId": session_id,
                            "text": t,
                        })
                    );
                }
                AgentEvent::ToolUse(name, input) => {
                    println!(
                        "{}",
                        serde_json::json!({
                            "schemaVersion": 2,
                            "type": "tool_use",
                            "runId": run_id,
                            "sessionId": session_id,
                            "name": name,
                            "text": input,
                        })
                    );
                }
                AgentEvent::ToolResult(_id, name, res) => {
                    println!(
                        "{}",
                        serde_json::json!({
                            "schemaVersion": 2,
                            "type": "tool_result",
                            "runId": run_id,
                            "sessionId": session_id,
                            "name": name,
                            "text": res,
                        })
                    );
                }
                AgentEvent::Error(err) => {
                    println!(
                        "{}",
                        serde_json::json!({
                            "schemaVersion": 2,
                            "type": "error",
                            "runId": run_id,
                            "sessionId": session_id,
                            "message": err,
                        })
                    );
                }
                AgentEvent::Done => break,
                _ => {}
            }
        }

        cancel.store(true, Ordering::Relaxed);

        println!(
            "{}",
            serde_json::json!({
                "schemaVersion": 2,
                "type": "final",
                "runId": run_id,
                "sessionId": session_id,
                "text": final_text,
            })
        );

        println!(
            "{}",
            serde_json::json!({
                "schemaVersion": 2,
                "type": "run_end",
                "runId": run_id,
                "sessionId": session_id,
                "status": "completed",
                "exitCode": 0,
            })
        );

        return ExitCode::SUCCESS;
    }

    if is_print_mode {
        let prompt = match initial_prompt {
            Some(p) => p,
            None => {
                let mut input = String::new();
                if let Err(error) = std::io::stdin().read_to_string(&mut input) {
                    eprintln!("Error: failed to read prompt from stdin: {error}");
                    return ExitCode::FAILURE;
                }
                input.trim().to_string()
            }
        };

        let mut agent = Agent::new_with_skills(
            chosen_provider,
            p_model.clone(),
            tool_registry,
            cfg,
            skills_for_agent,
        );
        return match agent.run_simple(&prompt) {
            Ok(output) => {
                println!("{}", tui::sanitize_terminal_output(&output));
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("Error: {error}");
                ExitCode::FAILURE
            }
        };
    }
    let theme_name = cfg.theme.clone();
    let show_thinking = cfg.show_thinking;
    let show_suggestions = cfg.show_suggestions;

    let (event_tx, event_rx) = mpsc::channel::<AgentEvent>();
    let (cmd_tx, cmd_rx) = mpsc::channel::<String>();
    let (perm_tx, perm_rx) = mpsc::channel::<String>();
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel2 = cancel.clone();
    let live_mirror = session::new_live_mirror();
    let live_mirror_agent = live_mirror.clone();

    let p_model_for_agent = p_model.clone();
    let p_model_for_print = p_model.clone();

    thread::spawn(move || {
        let mut agent = Agent::new_with_skills(
            chosen_provider,
            p_model_for_agent,
            tool_registry,
            cfg,
            skills_for_agent,
        );
        agent.set_live_mirror(live_mirror_agent);
        loop {
            match cmd_rx.recv() {
                Ok(cmd) if cmd.starts_with("__switch__:") => {
                    let rest = cmd.trim_start_matches("__switch__:");
                    if let Some((prov, modl)) = rest.split_once(':') {
                        let _ = agent.switch_provider(prov, modl);
                    }
                }
                Ok(cmd) if cmd.starts_with("__load_session__:") => {
                    let id = cmd.trim_start_matches("__load_session__:");
                    if !id.is_empty() {
                        let sessions_dir = config::sessions_dir();
                        if let Ok(session) = session::load(&sessions_dir, id) {
                            let usage = llm::Usage {
                                input_tokens: session.tokens_in,
                                output_tokens: session.tokens_out,
                                cache_read: 0,
                                cache_create: 0,
                            };
                            agent.set_state(session.messages, usage);
                        }
                    }
                }
                Ok(cmd) if cmd == "__compact__" => {
                    match agent.compact_now(&event_tx) {
                        Ok(_) => {}
                        Err(e) => {
                            let _ = event_tx.send(AgentEvent::Error(e));
                        }
                    }
                    let _ = event_tx.send(AgentEvent::Done);
                }
                Ok(cmd) if cmd == "__clear__" => {
                    agent.reset_state();
                }
                Ok(cmd) if cmd.starts_with("__auth_login__:") => {
                    let provider = cmd
                        .trim_start_matches("__auth_login__:")
                        .to_ascii_lowercase();
                    let msg = if provider == "xai" {
                        // Surface the user code before the blocking poll via a system-style error-free path:
                        // request device code first, emit as text event, then poll.
                        match oauth::request_xai_device_code() {
                            Ok(auth) => {
                                let uri = if !auth.verification_uri_complete.is_empty() {
                                    auth.verification_uri_complete.clone()
                                } else {
                                    auth.verification_uri.clone()
                                };
                                let notice = format!(
                                    "xAI device login\n1. Open: {uri}\n2. Enter code: {}\nWaiting for approval…",
                                    auth.user_code
                                );
                                let _ = event_tx.send(AgentEvent::Text(notice));
                                oauth::open_url(&uri);
                                match oauth::poll_xai_device_token(&auth, &cancel2) {
                                    Ok(token) => match oauth::save_token("xai", &token) {
                                        Ok(()) => {
                                            "xAI OAuth login saved to the OS keyring. You can select provider xai now.".to_string()
                                        }
                                        Err(e) => {
                                            format!("Login succeeded but failed to save token: {e}")
                                        }
                                    },
                                    Err(e) => format!("xAI OAuth failed: {e}"),
                                }
                            }
                            Err(e) => format!("xAI OAuth failed: {e}"),
                        }
                    } else {
                        format!("OAuth login is not implemented for '{provider}'. Supported: xai")
                    };
                    let _ = event_tx.send(AgentEvent::Text(format!("\n{msg}\n")));
                    let _ = event_tx.send(AgentEvent::Done);
                }
                Ok(cmd) if cmd.starts_with("__auth_logout__:") => {
                    let provider = cmd
                        .trim_start_matches("__auth_logout__:")
                        .to_ascii_lowercase();
                    let msg = match oauth::delete_token(&provider) {
                        Ok(true) => format!("Removed OAuth login for {provider}."),
                        Ok(false) => format!("No OAuth login stored for {provider}."),
                        Err(e) => format!("Logout failed: {e}"),
                    };
                    let _ = event_tx.send(AgentEvent::Text(format!("\n{msg}\n")));
                    let _ = event_tx.send(AgentEvent::Done);
                }
                Ok(cmd) if cmd == "__auth_status__" => {
                    let lines = [oauth::status_line("xai")];
                    let _ = event_tx.send(AgentEvent::Text(format!("\n{}\n", lines.join("\n"))));
                    let _ = event_tx.send(AgentEvent::Done);
                }
                Ok(cmd) if cmd.starts_with("__user_json__:") => {
                    cancel2.store(false, Ordering::Relaxed);
                    let json = cmd.trim_start_matches("__user_json__:");
                    match parse_user_blocks_cmd(json) {
                        Ok(user) => {
                            let _ = agent.run_user(user, event_tx.clone(), &cancel2, &perm_rx);
                        }
                        Err(e) => {
                            let _ = event_tx.send(AgentEvent::Error(format!(
                                "Invalid multimodal user message: {e}"
                            )));
                            let _ = event_tx.send(AgentEvent::Done);
                        }
                    }
                }
                Ok(prompt) => {
                    cancel2.store(false, Ordering::Relaxed);
                    let _ = agent.run(&prompt, event_tx.clone(), &cancel2, &perm_rx);
                }
                Err(_) => break,
            }
        }
    });

    config::hydrate_env_from_keyring();

    let mut tui = tui::Tui::new(version, &p_model_for_print, &provider_name_str, &work_dir);
    tui.set_theme_name(&theme_name);
    tui.set_show_thinking(show_thinking);
    tui.set_show_suggestions(show_suggestions);
    tui.set_agent_tx(cmd_tx.clone());
    tui.set_perm_tx(perm_tx);
    tui.set_cancel_flag(cancel);
    tui.set_live_mirror(live_mirror);
    tui.set_picker_models(models);

    if skill_count > 0 {
        tui.add_output_line(tui::OutputLine {
            type_: "system".into(),
            content: format!(
                "Loaded {skill_count} skill(s) from {}. Use the skill tool or /skills.",
                skills::default_skills_dir().display()
            ),
            tool_name: String::new(),
            duration: String::new(),
        });
    }
    if mcp_tool_count > 0 {
        tui.add_output_line(tui::OutputLine {
            type_: "system".into(),
            content: format!("Loaded {mcp_tool_count} MCP tool(s). Use /mcp to list."),
            tool_name: String::new(),
            duration: String::new(),
        });
    }
    for w in mcp_warnings {
        tui.add_output_line(tui::OutputLine {
            type_: "system".into(),
            content: w,
            tool_name: String::new(),
            duration: String::new(),
        });
    }

    if let Some(prompt) = initial_prompt {
        tui.add_output_line(tui::OutputLine {
            type_: "user".into(),
            content: prompt.clone(),
            tool_name: String::new(),
            duration: String::new(),
        });
        let _ = cmd_tx.send(prompt);
    }

    match tui.run(event_rx) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn print_help(version: &str) {
    println!("cairn-code {version}");
    println!("An AI coding agent for your terminal.");
    println!();
    println!("USAGE:");
    println!("    cairn-code [OPTIONS] [PROMPT]");
    println!();
    println!("ARGUMENTS:");
    println!("    PROMPT             Optional initial prompt; quote prompts containing spaces");
    println!();
    println!("OPTIONS:");
    println!("    -p, --print       Run PROMPT once non-interactively, print the result, and exit");
    println!("    -h, --help        Print this help message and exit");
    println!("    -v, --version     Print version information and exit");
    println!();
    println!("ENV:");
    println!("    CAIRN_PROVIDER    Override the configured default provider");
    println!("    CAIRN_MODEL       Override the configured default model");
    println!();
    println!("With no PROMPT, cairn-code starts the interactive TUI.");
    println!("With PROMPT but no -p, it starts the TUI and sends PROMPT as the first message.");
    println!("With PROMPT and -p, it runs once non-interactively and exits.");
}

/// Decode a multimodal user payload from the TUI (`__user_json__:` command).
fn parse_user_blocks_cmd(json: &str) -> Result<llm::UserBlocks, String> {
    let val = serde_json::from_str::<serde_json::Value>(json).map_err(|e| format!("json: {e}"))?;
    let obj = val
        .as_object()
        .ok_or_else(|| "expected object".to_string())?;
    let text = obj
        .get("text")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let mut images = Vec::new();
    if let Some(arr) = obj.get("images").and_then(|v| v.as_array()) {
        for img in arr {
            let o = img
                .as_object()
                .ok_or_else(|| "image not object".to_string())?;
            let media_type = o
                .get("media_type")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "image missing media_type".to_string())?
                .to_string();
            let data = o
                .get("data")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "image missing data".to_string())?
                .to_string();
            if data.is_empty() {
                return Err("image data empty".into());
            }
            images.push(llm::ImageBlock {
                media_type,
                data_base64: data,
            });
        }
    }
    let blocks = llm::UserBlocks { text, images };
    if blocks.is_empty() {
        return Err("empty user message".into());
    }
    Ok(blocks)
}
