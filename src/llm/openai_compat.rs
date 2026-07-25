use super::provider::*;
use crate::json;
use std::collections::HashMap;

/// Shared parsing/serialization for OpenAI-compatible chat-completions APIs
/// (openai, ollama, openrouter, and opengateway all speak this dialect).

/// Escapes a string for use inside a JSON string literal.
///
/// RFC 8259 requires every character below U+0020 to be escaped, not just the
/// five with short forms. Emitting a raw control byte produces a body that no
/// parser will accept, and these strings carry tool output: ANSI colour
/// sequences (`ESC`, U+001B) come back from anything that writes coloured
/// output, and `shell`'s `normalize_cli_output` only rewrites line endings, so
/// they reach here intact.
pub fn escape_json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            // Everything else below U+0020 has no short form and needs \uXXXX.
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

pub fn build_messages_json(messages: &[Message], system: &str) -> String {
    let mut body = String::from("[");
    let mut first = true;
    if !system.is_empty() {
        let escaped = escape_json_str(system);
        body.push_str(&format!(
            "{{\"role\":\"system\",\"content\":\"{escaped}\"}}"
        ));
        first = false;
    }
    let mut index = 0;
    while index < messages.len() {
        let msg = &messages[index];
        if !first {
            body.push(',');
        }
        first = false;
        match &msg.content {
            Content::Text(t) => {
                let escaped = escape_json_str(t);
                body.push_str(&format!(
                    "{{\"role\":\"{}\",\"content\":\"{escaped}\"}}",
                    msg.role
                ));
            }
            Content::User(blocks) => {
                // OpenAI-compatible multimodal: content is an array of parts.
                body.push_str(&format!("{{\"role\":\"{}\",\"content\":[", msg.role));
                let mut first_part = true;
                if !blocks.text.is_empty() {
                    let escaped = escape_json_str(&blocks.text);
                    body.push_str(&format!("{{\"type\":\"text\",\"text\":\"{escaped}\"}}"));
                    first_part = false;
                }
                for img in &blocks.images {
                    if !first_part {
                        body.push(',');
                    }
                    first_part = false;
                    let mt = escape_json_str(&img.media_type);
                    // data_base64 is already base64 alphabet — no further escape needed.
                    body.push_str(&format!(
                        "{{\"type\":\"image_url\",\"image_url\":{{\"url\":\"data:{mt};base64,{}\"}}}}",
                        img.data_base64
                    ));
                }
                if first_part {
                    // Empty multimodal → empty text part so the message stays valid.
                    body.push_str("{\"type\":\"text\",\"text\":\"\"}");
                }
                body.push_str("]}");
            }
            Content::ToolUse(_) => {
                // Providers return parallel calls as consecutive ToolUse
                // messages. OpenAI-compatible APIs require those calls in one
                // assistant message before any corresponding tool results.
                body.push_str("{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[");
                let mut first_call = true;
                while index < messages.len() {
                    let Content::ToolUse(tu) = &messages[index].content else {
                        break;
                    };
                    if !first_call {
                        body.push(',');
                    }
                    first_call = false;
                    let id = escape_json_str(&tu.id);
                    let name = escape_json_str(&tu.name);
                    let args = escape_json_str(&tu.input);
                    body.push_str(&format!(
                        "{{\"id\":\"{id}\",\"type\":\"function\",\"function\":{{\"name\":\"{name}\",\"arguments\":\"{args}\"}}}}"
                    ));
                    index += 1;
                }
                body.push_str("]}");
                continue;
            }
            Content::ToolResult(tr) => {
                let escaped = escape_json_str(&tr.content);
                body.push_str(&format!(
                    "{{\"role\":\"tool\",\"tool_call_id\":\"{}\",\"content\":\"{escaped}\"}}",
                    tr.tool_use_id
                ));
            }
            Content::Thinking(t) => {
                let escaped = escape_json_str(t);
                body.push_str(&format!(
                    "{{\"role\":\"assistant\",\"content\":\"{escaped}\"}}"
                ));
            }
        }
        index += 1;
    }
    body.push(']');
    body
}

/// Returns an empty string when there are no tools, otherwise a
/// `,"tools":[...]` fragment ready to be appended to a request body.
pub fn build_tools_json(tools: &[ToolDefinition]) -> String {
    if tools.is_empty() {
        return String::new();
    }
    let mut body = String::from(",\"tools\":[");
    for (i, tool) in tools.iter().enumerate() {
        if i > 0 {
            body.push(',');
        }
        let name_esc = escape_json_str(&tool.name);
        let desc_esc = escape_json_str(&tool.description);
        body.push_str(&format!(
            "{{\"type\":\"function\",\"function\":{{\"name\":\"{name_esc}\",\"description\":\"{desc_esc}\",\"parameters\":{}}}}}",
            tool.input_schema
        ));
    }
    body.push(']');
    body
}

/// Largest byte index at or below `index` that starts a character.
///
/// `str::floor_char_boundary` is still unstable, and the parser's byte offsets
/// routinely land mid-character on non-ASCII payloads.
fn char_boundary_floor(s: &str, index: usize) -> usize {
    let mut index = index.min(s.len());
    while index > 0 && !s.is_char_boundary(index) {
        index -= 1;
    }
    index
}

/// Smallest byte index at or above `index` that starts a character.
fn char_boundary_ceil(s: &str, index: usize) -> usize {
    let mut index = index.min(s.len());
    while index < s.len() && !s.is_char_boundary(index) {
        index += 1;
    }
    index
}

/// Sanity-checks a hand-built request body and returns a diagnostic error
/// (with byte offset and surrounding context) if it isn't valid JSON.
pub fn validate_json_body(body: String) -> Result<String, String> {
    if let Err(e) = json::parse(&body) {
        // The error offset is a byte index, so the surrounding window has to be
        // snapped outward to character boundaries before slicing: `&body[a..b]`
        // panics when either end splits a multi-byte character. Any non-ASCII
        // text near the fault would otherwise turn a reportable bad body into a
        // crash — in the code whose job is explaining the bad body.
        let pos = e.pos.min(body.len());
        let start = char_boundary_floor(&body, pos.saturating_sub(20));
        let end = char_boundary_ceil(&body, pos.saturating_add(20).min(body.len()));
        let context = &body[start..end];
        // Deliberately the raw byte, not the decoded character: `pos` may be
        // mid-character, and the offending byte is what makes the parse fail.
        let byte = body.as_bytes().get(pos).copied();
        let shown = byte.map(char::from).unwrap_or('?');
        return Err(format!(
            "Invalid JSON body: {e}\nByte at pos {pos}: '{shown}' (0x{:02X})\nContext: ...{context}...",
            byte.unwrap_or(0)
        ));
    }
    Ok(body)
}

/// Incremental reducer for OpenAI-compatible SSE data lines. It retains only
/// state that becomes part of the returned messages/usage, rather than a
/// second copy of the complete wire transcript.
#[derive(Default)]
pub struct StreamingResponse {
    usage: Usage,
    collected: String,
    tool_calls: HashMap<u64, (String, String, String)>,
    saw_done: bool,
    lines_seen: usize,
    error: Option<String>,
}

impl StreamingResponse {
    pub fn push_line<F>(&mut self, line: &str, on_chunk: &mut F, emit_reasoning: bool)
    where
        F: FnMut(&str, &str),
    {
        self.lines_seen = self.lines_seen.saturating_add(1);
        if self.error.is_some() || self.saw_done {
            return;
        }
        if let Err(error) = self.process_line(line, on_chunk, emit_reasoning) {
            self.error = Some(error);
        }
    }

    fn process_line<F>(
        &mut self,
        line: &str,
        on_chunk: &mut F,
        emit_reasoning: bool,
    ) -> Result<(), String>
    where
        F: FnMut(&str, &str),
    {
        let Some(data) = line.strip_prefix("data: ") else {
            return Ok(());
        };
        if data == "[DONE]" {
            self.saw_done = true;
            return Ok(());
        }

        let val = json::parse(data).map_err(|error| {
            format!(
                "Malformed OpenAI-compatible stream event on line {}: {error}",
                self.lines_seen
            )
        })?;
        if val.as_object().is_none() {
            return Err(format!(
                "Malformed OpenAI-compatible stream event on line {}: expected a JSON object",
                self.lines_seen
            ));
        }
        if let Some(error) = val.get("error") {
            let message = error
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("provider returned an error event");
            return Err(format!("OpenAI-compatible API stream error: {message}"));
        }
        if let Some(choices) = val.get("choices").and_then(|v| v.as_array()) {
            if let Some(choice) = choices.first() {
                if let Some(delta) = choice.get("delta") {
                    if let Some(text) = delta.get("content").and_then(|v| v.as_str()) {
                        self.collected.push_str(text);
                        on_chunk(text, "text");
                    }
                    if emit_reasoning {
                        if let Some(text) = delta.get("reasoning_content").and_then(|v| v.as_str())
                        {
                            if !text.is_empty() {
                                on_chunk(text, "thinking");
                            }
                        }
                    }
                    if let Some(tc_arr) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                        for tc in tc_arr {
                            let idx = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
                            let entry = self
                                .tool_calls
                                .entry(idx)
                                .or_insert_with(|| (String::new(), String::new(), String::new()));
                            if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                                if !id.is_empty() {
                                    entry.0 = id.to_string();
                                }
                            }
                            if let Some(func) = tc.get("function") {
                                if let Some(name) = func.get("name").and_then(|v| v.as_str()) {
                                    if !name.is_empty() {
                                        entry.1 = name.to_string();
                                    }
                                }
                                if let Some(args) = func.get("arguments").and_then(|v| v.as_str()) {
                                    entry.2.push_str(args);
                                }
                            }
                        }
                    }
                }
            }
        }
        if let Some(usage) = val.get("usage") {
            self.usage.input_tokens = usage
                .get("prompt_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            self.usage.output_tokens = usage
                .get("completion_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
        }
        Ok(())
    }

    pub fn finish(self) -> Result<(Vec<Message>, Usage), String> {
        if let Some(error) = self.error {
            return Err(error);
        }
        if !self.saw_done {
            return Err(
                "Incomplete OpenAI-compatible stream: missing [DONE] completion marker".into(),
            );
        }

        let mut messages = Vec::new();
        if !self.collected.is_empty() {
            messages.push(Message {
                role: "assistant".into(),
                content: Content::Text(self.collected),
            });
        }
        if !self.tool_calls.is_empty() {
            let mut calls: Vec<_> = self.tool_calls.into_iter().collect();
            calls.sort_by_key(|(idx, _)| *idx);
            for (_, (id, name, args)) in calls {
                let tool_use = ToolUse {
                    id,
                    name,
                    input: args,
                };
                tool_use
                    .validate()
                    .map_err(|error| format!("Invalid OpenAI-compatible stream: {error}"))?;
                messages.push(Message {
                    role: "assistant".into(),
                    content: Content::ToolUse(tool_use),
                });
            }
        }
        Ok((messages, self.usage))
    }
}

pub fn parse_streaming_response(raw: &str) -> Result<(Vec<Message>, Usage), String> {
    let mut response = StreamingResponse::default();
    let mut ignore_chunk = |_: &str, _: &str| {};
    for line in raw.lines() {
        response.push_line(line, &mut ignore_chunk, false);
    }
    response.finish()
}

pub fn parse_complete_response(raw: &str) -> Result<(Vec<Message>, Usage), String> {
    let mut messages = Vec::new();
    let mut usage = Usage::default();

    let val = json::parse(raw).map_err(|e| format!("Failed to parse response: {e}"))?;

    if let Some(choices) = val.get("choices").and_then(|v| v.as_array()) {
        if let Some(choice) = choices.first() {
            if let Some(msg) = choice.get("message") {
                let role = msg
                    .get("role")
                    .and_then(|v| v.as_str())
                    .unwrap_or("assistant")
                    .to_string();
                if let Some(text) = msg.get("content").and_then(|v| v.as_str()) {
                    if !text.is_empty() {
                        messages.push(Message {
                            role: role.clone(),
                            content: Content::Text(text.to_string()),
                        });
                    }
                }
                if let Some(tc_arr) = msg.get("tool_calls").and_then(|v| v.as_array()) {
                    for tc in tc_arr {
                        let tool_use = ToolUse {
                            id: tc
                                .get("id")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                            name: tc
                                .get("function")
                                .and_then(|f| f.get("name"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                            input: tc
                                .get("function")
                                .and_then(|f| f.get("arguments"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                        };
                        tool_use
                            .validate()
                            .map_err(|e| format!("Invalid OpenAI-compatible response: {e}"))?;
                        messages.push(Message {
                            role: role.clone(),
                            content: Content::ToolUse(tool_use),
                        });
                    }
                }
                if messages.is_empty() {
                    messages.push(Message {
                        role,
                        content: Content::Text(String::new()),
                    });
                }
            }
        }
    }
    if let Some(u) = val.get("usage") {
        usage.input_tokens = u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
        usage.output_tokens = u
            .get("completion_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
    }
    Ok((messages, usage))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_json_body_reports_multibyte_context_without_panicking() {
        // Whether a fixed window splits a character depends on how the fault
        // aligns against the filler, so sweep the padding: a single length can
        // land on a boundary by luck and hide the bug. Each filler width is
        // swept across a full character so every alignment is exercised.
        for filler in ["é", "中", "😀", "ü"] {
            for pad in 0..8 {
                let body = format!("{{\"k\":\"{}{}\" BAD}}", "x".repeat(pad), filler.repeat(20));
                let error = validate_json_body(body).unwrap_err();
                assert!(
                    error.contains("Context:"),
                    "filler {filler:?} pad {pad}: {error}"
                );
            }
        }
    }

    #[test]
    fn validate_json_body_handles_faults_at_the_edges() {
        // Empty and single-byte bodies drive the window past both ends.
        for body in ["", "{", "\u{1b}", "é"] {
            let error = validate_json_body(body.to_string()).unwrap_err();
            assert!(error.contains("Invalid JSON body:"), "{body:?}: {error}");
        }
    }

    #[test]
    fn validate_json_body_passes_valid_bodies_through_unchanged() {
        let body = r#"{"model":"m","messages":[{"role":"user","content":"héllo 😀"}]}"#;
        assert_eq!(validate_json_body(body.to_string()).unwrap(), body);
    }

    #[test]
    fn char_boundary_helpers_snap_outward() {
        let s = "aé😀b"; // 1 + 2 + 4 + 1 bytes
        assert_eq!(char_boundary_floor(s, 2), 1); // inside 'é'
        assert_eq!(char_boundary_ceil(s, 2), 3); // inside 'é'
        assert_eq!(char_boundary_floor(s, 5), 3); // inside '😀'
        assert_eq!(char_boundary_ceil(s, 5), 7); // inside '😀'
                                                 // Out-of-range indices clamp to the string rather than panicking.
        assert_eq!(char_boundary_floor(s, 999), s.len());
        assert_eq!(char_boundary_ceil(s, 999), s.len());
    }

    #[test]
    fn test_streaming_multiple_tool_calls_all_survive() {
        let raw = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"glob\",\"arguments\":\"{}\"}}]}}]}\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"call_2\",\"function\":{\"name\":\"grep\",\"arguments\":\"{}\"}}]}}]}\n",
            "data: [DONE]\n",
        );
        let (msgs, _usage) = parse_streaming_response(raw).unwrap();
        assert_eq!(msgs.len(), 2);
        match &msgs[0].content {
            Content::ToolUse(tu) => assert_eq!(tu.name, "glob"),
            _ => panic!("expected ToolUse"),
        }
        match &msgs[1].content {
            Content::ToolUse(tu) => assert_eq!(tu.name, "grep"),
            _ => panic!("expected ToolUse"),
        }
    }

    #[test]
    fn test_streaming_text_and_tool_calls_both_survive() {
        let raw = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"looking...\"}}]}\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"glob\",\"arguments\":\"{}\"}}]}}]}\n",
            "data: [DONE]\n",
        );
        let (msgs, _usage) = parse_streaming_response(raw).unwrap();
        assert_eq!(msgs.len(), 2);
        match &msgs[0].content {
            Content::Text(t) => assert_eq!(t, "looking..."),
            _ => panic!("expected Text"),
        }
        match &msgs[1].content {
            Content::ToolUse(tu) => assert_eq!(tu.name, "glob"),
            _ => panic!("expected ToolUse"),
        }
    }

    #[test]
    fn test_streaming_rejects_malformed_event() {
        let raw = "data: not-json\ndata: [DONE]\n";
        let error = parse_streaming_response(raw).unwrap_err();

        assert!(error.contains("Malformed OpenAI-compatible stream event"));
    }

    #[test]
    fn test_streaming_surfaces_provider_error_event() {
        let raw = "data: {\"error\":{\"message\":\"rate limited\"}}\n";
        let error = parse_streaming_response(raw).unwrap_err();

        assert!(error.contains("rate limited"), "{error}");
    }

    #[test]
    fn test_streaming_rejects_truncated_event_at_eof() {
        let raw = r#"data: {"choices":[{"delta":{"content":"partial"}}]
"#;
        let error = parse_streaming_response(raw).unwrap_err();

        assert!(error.contains("Malformed OpenAI-compatible stream event"));
    }

    #[test]
    fn test_streaming_requires_done_marker() {
        let raw = "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n";
        let error = parse_streaming_response(raw).unwrap_err();

        assert!(error.contains("missing [DONE] completion marker"));
    }

    #[test]
    fn test_streaming_rejects_tool_calls_missing_id_or_name() {
        let missing_id = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"glob\",\"arguments\":\"{}\"}}]}}]}\n",
            "data: [DONE]\n",
        );
        let missing_name = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"arguments\":\"{}\"}}]}}]}\n",
            "data: [DONE]\n",
        );

        let id_error = parse_streaming_response(missing_id).unwrap_err();
        assert!(id_error.contains("missing an id"), "{id_error}");
        let name_error = parse_streaming_response(missing_name).unwrap_err();
        assert!(name_error.contains("missing a name"), "{name_error}");
    }

    #[test]
    fn test_streaming_rejects_tool_call_missing_arguments() {
        let raw = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"glob\"}}]}}]}\n",
            "data: [DONE]\n",
        );
        let error = parse_streaming_response(raw).unwrap_err();

        assert!(error.contains("invalid JSON arguments"), "{error}");
    }

    #[test]
    fn test_streaming_rejects_invalid_tool_arguments() {
        let raw = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"glob","arguments":"{\"pattern\":"}}]}}]}
data: [DONE]
"#;
        let error = parse_streaming_response(raw).unwrap_err();

        assert!(error.contains("invalid JSON arguments"), "{error}");
    }

    #[test]
    fn test_complete_response_multiple_tool_calls_all_survive() {
        let raw = r#"{"choices":[{"message":{"role":"assistant","tool_calls":[
            {"id":"call_1","function":{"name":"glob","arguments":"{\"pattern\":\"*.rs\"}"}},
            {"id":"call_2","function":{"name":"grep","arguments":"{\"pattern\":\"foo\"}"}}
        ]}}]}"#;
        let (msgs, _usage) = parse_complete_response(raw).unwrap();
        assert_eq!(msgs.len(), 2);
        match &msgs[0].content {
            Content::ToolUse(tu) => assert_eq!(tu.id, "call_1"),
            _ => panic!("expected ToolUse"),
        }
        match &msgs[1].content {
            Content::ToolUse(tu) => assert_eq!(tu.id, "call_2"),
            _ => panic!("expected ToolUse"),
        }
    }

    #[test]
    fn test_complete_response_rejects_invalid_tool_call() {
        let raw = r#"{"choices":[{"message":{"role":"assistant","tool_calls":[
            {"id":"call_1","function":{"name":"glob","arguments":"not-json"}}
        ]}}]}"#;
        let error = parse_complete_response(raw).unwrap_err();

        assert!(error.contains("invalid JSON arguments"), "{error}");
    }

    #[test]
    fn test_build_messages_json_user_blocks_multimodal() {
        let msgs = vec![Message {
            role: "user".into(),
            content: Content::User(crate::llm::UserBlocks {
                text: "describe".into(),
                images: vec![crate::llm::ImageBlock {
                    media_type: "image/png".into(),
                    data_base64: "Zm9v".into(),
                }],
            }),
        }];
        let body = build_messages_json(&msgs, "");
        let v = crate::json::parse(&body).unwrap();
        let arr = v.as_array().unwrap();
        let content = arr[0]
            .as_object()
            .unwrap()
            .get("content")
            .unwrap()
            .as_array()
            .unwrap();
        assert_eq!(content.len(), 2);
        let p0 = content[0].as_object().unwrap();
        assert_eq!(p0.get("type").and_then(|x| x.as_str()), Some("text"));
        assert_eq!(p0.get("text").and_then(|x| x.as_str()), Some("describe"));
        let p1 = content[1].as_object().unwrap();
        assert_eq!(p1.get("type").and_then(|x| x.as_str()), Some("image_url"));
        let url = p1
            .get("image_url")
            .and_then(|x| x.as_object())
            .and_then(|o| o.get("url"))
            .and_then(|x| x.as_str())
            .unwrap();
        assert!(url.starts_with("data:image/png;base64,Zm9v"));
    }

    #[test]
    fn test_build_messages_json_tool_result_is_role_tool_not_content_block() {
        let msgs = vec![Message {
            role: "user".into(),
            content: Content::ToolResult(ToolResult {
                tool_use_id: "call_1".into(),
                content: "line1\nline2".into(),
            }),
        }];
        let body = build_messages_json(&msgs, "");
        let v = crate::json::parse(&body).unwrap();
        let arr = v.as_array().unwrap();
        let obj = arr[0].as_object().unwrap();
        assert_eq!(obj.get("role").and_then(|v| v.as_str()), Some("tool"));
        assert_eq!(
            obj.get("tool_call_id").and_then(|v| v.as_str()),
            Some("call_1")
        );
        assert_eq!(
            obj.get("content").and_then(|v| v.as_str()),
            Some("line1\nline2")
        );
    }

    #[test]
    fn test_build_messages_json_tool_use_has_tool_calls_field() {
        let msgs = vec![Message {
            role: "assistant".into(),
            content: Content::ToolUse(ToolUse {
                id: "call_1".into(),
                name: "glob".into(),
                input: "{\"pattern\":\"*.rs\"}".into(),
            }),
        }];
        let body = build_messages_json(&msgs, "");
        let v = crate::json::parse(&body).unwrap();
        let arr = v.as_array().unwrap();
        let obj = arr[0].as_object().unwrap();
        assert_eq!(obj.get("role").and_then(|v| v.as_str()), Some("assistant"));
        let tool_calls = obj.get("tool_calls").and_then(|v| v.as_array()).unwrap();
        assert_eq!(tool_calls.len(), 1);
    }

    #[test]
    fn test_build_messages_json_coalesces_parallel_tool_calls() {
        let msgs = vec![
            Message {
                role: "assistant".into(),
                content: Content::ToolUse(ToolUse {
                    id: "call_1".into(),
                    name: "glob".into(),
                    input: "{}".into(),
                }),
            },
            Message {
                role: "assistant".into(),
                content: Content::ToolUse(ToolUse {
                    id: "call_2".into(),
                    name: "grep".into(),
                    input: "{}".into(),
                }),
            },
            Message {
                role: "user".into(),
                content: Content::ToolResult(ToolResult {
                    tool_use_id: "call_1".into(),
                    content: "a.rs".into(),
                }),
            },
            Message {
                role: "user".into(),
                content: Content::ToolResult(ToolResult {
                    tool_use_id: "call_2".into(),
                    content: "match".into(),
                }),
            },
        ];

        let parsed = crate::json::parse(&build_messages_json(&msgs, "")).unwrap();
        let messages = parsed.as_array().unwrap();
        assert_eq!(
            messages.len(),
            3,
            "one assistant turn plus two tool results"
        );
        let calls = messages[0]
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(
            messages[1].get("tool_call_id").and_then(|v| v.as_str()),
            Some("call_1")
        );
        assert_eq!(
            messages[2].get("tool_call_id").and_then(|v| v.as_str()),
            Some("call_2")
        );
    }

    #[test]
    fn test_escape_handles_newlines() {
        let escaped = escape_json_str("line1\nline2\ttabbed\r");
        assert_eq!(escaped, "line1\\nline2\\ttabbed\\r");
    }

    #[test]
    fn escape_json_str_covers_every_c0_control() {
        // The whole C0 range must survive a round trip through a real parser.
        // Anything emitted raw produces a body no parser accepts.
        for code in 0u32..0x20 {
            let raw = format!("a{}b", char::from_u32(code).unwrap());
            let body = format!("{{\"k\":\"{}\"}}", escape_json_str(&raw));
            let parsed = json::parse(&body)
                .unwrap_or_else(|e| panic!("U+{code:04X} produced unparseable JSON: {e}"));
            let back = parsed
                .as_object()
                .and_then(|o| o.get("k"))
                .and_then(|v| v.as_str())
                .unwrap_or_else(|| panic!("U+{code:04X} did not round trip"));
            assert_eq!(back, raw, "U+{code:04X} changed value");
        }
    }

    #[test]
    fn escape_json_str_uses_short_forms_where_they_exist() {
        assert_eq!(escape_json_str("\u{8}"), "\\b");
        assert_eq!(escape_json_str("\u{c}"), "\\f");
        // No short form: \uXXXX, lower-case hex, always four digits.
        assert_eq!(escape_json_str("\u{1b}"), "\\u001b");
        assert_eq!(escape_json_str("\u{0}"), "\\u0000");
        assert_eq!(escape_json_str("\u{1f}"), "\\u001f");
    }

    #[test]
    fn escape_json_str_survives_ansi_coloured_tool_output() {
        // The realistic path: coloured build output arriving as a tool result.
        let raw = "\u{1b}[32mok\u{1b}[0m \u{1b}[31mfail\u{1b}[0m\u{7}";
        let body = format!("{{\"content\":\"{}\"}}", escape_json_str(raw));
        let parsed = json::parse(&body).expect("ANSI output must produce valid JSON");
        assert_eq!(
            parsed
                .as_object()
                .and_then(|o| o.get("content"))
                .and_then(|v| v.as_str()),
            Some(raw)
        );
    }

    #[test]
    fn escape_json_str_leaves_printable_and_multibyte_alone() {
        // DEL (U+007F) is not in C0 and JSON does not require escaping it.
        let raw = "héllo 😀 中文 \u{7f} ~";
        assert_eq!(escape_json_str(raw), raw);
    }

    #[test]
    fn build_messages_json_stays_valid_with_control_characters() {
        // End to end: a tool result carrying ESC must not break the body.
        let messages = vec![Message {
            role: "user".into(),
            content: Content::ToolResult(ToolResult {
                tool_use_id: "call_1".into(),
                content: "\u{1b}[1mbuilding\u{1b}[0m\u{c}done".into(),
            }),
        }];
        let body = format!("{{\"messages\":{}}}", build_messages_json(&messages, ""));
        assert!(json::parse(&body).is_ok(), "{body}");
    }
}
