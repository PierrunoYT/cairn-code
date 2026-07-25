//! Anthropic (Claude) provider.
//!
//! Model picker prefers a live `GET /v1/models` catalog when an API key is
//! available (5-minute cache), falling back to a curated list.

use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::provider::*;
use crate::http_client;
use crate::json;

const MESSAGES_URL: &str = "https://api.anthropic.com/v1/messages";
const MODELS_URL: &str = "https://api.anthropic.com/v1/models";
const API_VERSION: &str = "2023-06-01";
const DEFAULT_MODEL: &str = "claude-sonnet-5";
const CACHE_TTL: Duration = Duration::from_secs(5 * 60);

static MODELS_CACHE: OnceLock<Mutex<Option<(Instant, Vec<ModelInfo>)>>> = OnceLock::new();

pub struct AnthropicProvider {
    api_key: String,
}

impl AnthropicProvider {
    pub fn new() -> Self {
        AnthropicProvider {
            api_key: String::new(),
        }
    }

    fn get_key(&self) -> String {
        if !self.api_key.is_empty() {
            return self.api_key.clone();
        }
        if let Ok(k) = std::env::var("ANTHROPIC_API_KEY") {
            if !k.is_empty() {
                return k;
            }
        }
        crate::config::config_get_api_key("anthropic").unwrap_or_default()
    }
}

impl Provider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }
    fn default_model(&self) -> &str {
        DEFAULT_MODEL
    }

    fn available_models(&self) -> Vec<ModelInfo> {
        let key = self.get_key();
        if key.is_empty() {
            return curated_models();
        }
        if let Some(cached) = cached_models() {
            return cached;
        }
        match fetch_remote_models(&key) {
            Ok(models) if !models.is_empty() => {
                store_cache(models.clone());
                models
            }
            _ => curated_models(),
        }
    }

    fn stream_complete(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        system: &str,
        model: &str,
        max_tokens: usize,
        mut on_chunk: StreamingCallback,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<(Vec<Message>, Usage), String> {
        let key = self.get_key();
        if key.is_empty() {
            return Err(missing_api_key("ANTHROPIC_API_KEY"));
        }
        let body = build_request_body(messages, tools, system, model, max_tokens, true)?;
        let req = http_client::HttpRequest {
            url: MESSAGES_URL.into(),
            headers: vec![
                ("x-api-key".into(), key),
                ("anthropic-version".into(), API_VERSION.into()),
                ("content-type".into(), "application/json".into()),
            ],
            body: Some(body),
        };
        let mut response = AnthropicStreamingResponse::default();
        http_client::request_streaming_with_cancel(
            &req,
            |line| response.push_line(line, &mut on_chunk),
            Some(cancel),
        )?;
        response.finish()
    }

    fn complete(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        system: &str,
        model: &str,
        max_tokens: usize,
    ) -> Result<(Vec<Message>, Usage), String> {
        let key = self.get_key();
        if key.is_empty() {
            return Err(missing_api_key("ANTHROPIC_API_KEY"));
        }
        let body = build_request_body(messages, tools, system, model, max_tokens, false)?;
        let req = http_client::HttpRequest {
            url: MESSAGES_URL.into(),
            headers: vec![
                ("x-api-key".into(), key),
                ("anthropic-version".into(), API_VERSION.into()),
                ("content-type".into(), "application/json".into()),
            ],
            body: Some(body),
        };
        let resp = http_client::request(&req)?;
        parse_anthropic_complete_response(&resp.body)
    }
}

fn curated_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            id: "claude-sonnet-5".into(),
            name: "Claude Sonnet 5".into(),
            max_ctx: 200_000,
        },
        ModelInfo {
            id: "claude-opus-4-8".into(),
            name: "Claude Opus 4.8".into(),
            max_ctx: 1_000_000,
        },
        ModelInfo {
            id: "claude-haiku-4-5".into(),
            name: "Claude Haiku 4.5".into(),
            max_ctx: 200_000,
        },
        ModelInfo {
            id: "claude-sonnet-4-6".into(),
            name: "Claude Sonnet 4.6".into(),
            max_ctx: 1_000_000,
        },
        ModelInfo {
            id: "claude-opus-4-7".into(),
            name: "Claude Opus 4.7".into(),
            max_ctx: 1_000_000,
        },
        ModelInfo {
            id: "claude-sonnet-4-20250514".into(),
            name: "Claude Sonnet 4".into(),
            max_ctx: 200_000,
        },
        ModelInfo {
            id: "claude-opus-4-20250514".into(),
            name: "Claude Opus 4".into(),
            max_ctx: 200_000,
        },
    ]
}

fn models_cache() -> &'static Mutex<Option<(Instant, Vec<ModelInfo>)>> {
    MODELS_CACHE.get_or_init(|| Mutex::new(None))
}

fn cached_models() -> Option<Vec<ModelInfo>> {
    let guard = models_cache().lock().ok()?;
    let (at, models) = guard.as_ref()?;
    if at.elapsed() > CACHE_TTL {
        return None;
    }
    Some(models.clone())
}

fn store_cache(models: Vec<ModelInfo>) {
    if let Ok(mut guard) = models_cache().lock() {
        *guard = Some((Instant::now(), models));
    }
}

fn anthropic_headers(api_key: &str) -> Vec<(String, String)> {
    vec![
        ("x-api-key".into(), api_key.to_string()),
        ("anthropic-version".into(), API_VERSION.into()),
        ("Accept".into(), "application/json".into()),
    ]
}

fn fetch_remote_models(api_key: &str) -> Result<Vec<ModelInfo>, String> {
    let headers = anthropic_headers(api_key);
    let mut out = Vec::new();
    let mut after: Option<String> = None;
    // Cap pages so a weird API response cannot hang the picker.
    for _ in 0..20 {
        let url = match &after {
            Some(id) => format!("{MODELS_URL}?limit=100&after_id={id}"),
            None => format!("{MODELS_URL}?limit=100"),
        };
        let resp = http_client::request_get(&url, &headers)?;
        let page = parse_models_page(&resp.body)?;
        out.extend(page.models);
        if !page.has_more {
            break;
        }
        let Some(last) = page.last_id else {
            break;
        };
        if last.is_empty() {
            break;
        }
        after = Some(last);
    }
    if out.is_empty() {
        return Err("models list empty".into());
    }
    // API already returns newest first; keep that order.
    Ok(out)
}

struct ModelsPage {
    models: Vec<ModelInfo>,
    has_more: bool,
    last_id: Option<String>,
}

fn parse_models_page(body: &str) -> Result<ModelsPage, String> {
    let val = json::parse(body).map_err(|e| format!("models list: {e}"))?;
    let arr = val
        .get("data")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "models list: missing data array".to_string())?;
    let mut models = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for item in arr {
        let Some(id) = item.get("id").and_then(|v| v.as_str()) else {
            continue;
        };
        let id = id.trim();
        if id.is_empty() || !seen.insert(id.to_ascii_lowercase()) {
            continue;
        }
        // Skip non-chat / non-Claude entries if any appear.
        let lower = id.to_ascii_lowercase();
        if !lower.starts_with("claude") {
            continue;
        }
        let name = item
            .get("display_name")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .unwrap_or_else(|| display_name_fallback(id));
        let max_ctx = item
            .get("max_input_tokens")
            .and_then(|v| v.as_u64())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| context_for_id(id));
        models.push(ModelInfo {
            id: id.to_string(),
            name,
            max_ctx,
        });
    }
    let has_more = val
        .get("has_more")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let last_id = val
        .get("last_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| models.last().map(|m| m.id.clone()));
    Ok(ModelsPage {
        models,
        has_more,
        last_id,
    })
}

fn display_name_fallback(id: &str) -> String {
    // claude-opus-4-8 → Claude Opus 4.8 (best-effort)
    let rest = id.strip_prefix("claude-").unwrap_or(id).replace('-', " ");
    let mut out = String::from("Claude");
    for word in rest.split_whitespace() {
        out.push(' ');
        let mut chars = word.chars();
        if let Some(c) = chars.next() {
            out.push(c.to_ascii_uppercase());
            out.push_str(chars.as_str());
        }
    }
    out
}

fn context_for_id(id: &str) -> u64 {
    let lower = id.to_ascii_lowercase();
    // Newer Opus / Sonnet 4.6+ often ship with 1M context; fall back to 200k.
    if lower.contains("opus-4-") || lower.contains("sonnet-4-6") || lower.contains("sonnet-5") {
        1_000_000
    } else if lower.contains("haiku") {
        200_000
    } else {
        200_000
    }
}

fn build_request_body(
    messages: &[Message],
    tools: &[ToolDefinition],
    system: &str,
    model: &str,
    max_tokens: usize,
    stream: bool,
) -> Result<String, String> {
    let messages = messages
        .iter()
        .map(|msg| {
            let content = match &msg.content {
                Content::Text(text) | Content::Thinking(text) => {
                    serde_json::Value::String(text.clone())
                }
                Content::User(blocks) => {
                    let mut parts = Vec::new();
                    if !blocks.text.is_empty() {
                        parts.push(serde_json::json!({
                            "type": "text",
                            "text": blocks.text,
                        }));
                    }
                    for img in &blocks.images {
                        parts.push(serde_json::json!({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": img.media_type,
                                "data": img.data_base64,
                            }
                        }));
                    }
                    if parts.is_empty() {
                        serde_json::Value::String(String::new())
                    } else {
                        serde_json::Value::Array(parts)
                    }
                }
                Content::ToolUse(tool_use) => {
                    let input = serde_json::from_str::<serde_json::Value>(&tool_use.input)
                        .map_err(|e| format!("Invalid input for tool '{}': {e}", tool_use.name))?;
                    serde_json::json!([{
                        "type": "tool_use",
                        "id": tool_use.id,
                        "name": tool_use.name,
                        "input": input,
                    }])
                }
                Content::ToolResult(tool_result) => serde_json::json!([{
                    "type": "tool_result",
                    "tool_use_id": tool_result.tool_use_id,
                    "content": tool_result.content,
                }]),
            };
            Ok(serde_json::json!({
                "role": msg.role,
                "content": content,
            }))
        })
        .collect::<Result<Vec<_>, String>>()?;

    let mut body = serde_json::json!({
        "model": model,
        "max_tokens": max_tokens,
        "stream": stream,
        "messages": messages,
    });
    if !system.is_empty() {
        body["system"] = serde_json::Value::String(system.to_string());
    }
    if !tools.is_empty() {
        body["tools"] = serde_json::Value::Array(
            tools
                .iter()
                .map(|tool| {
                    let input_schema = serde_json::from_str::<serde_json::Value>(
                        &tool.input_schema,
                    )
                    .map_err(|e| format!("Invalid input schema for tool '{}': {e}", tool.name))?;
                    Ok(serde_json::json!({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": input_schema,
                    }))
                })
                .collect::<Result<Vec<_>, String>>()?,
        );
    }
    serde_json::to_string(&body).map_err(|e| format!("Failed to serialize Anthropic request: {e}"))
}

#[derive(Debug)]
struct AnthropicResponse {
    role: String,
    content: Vec<AnthropicContentBlock>,
    usage: AnthropicUsage,
}

#[derive(Debug)]
enum AnthropicContentBlock {
    Text(String),
    ToolUse {
        id: String,
        name: String,
        input: String,
    },
    Thinking(String),
}

#[derive(Debug, Default)]
struct AnthropicUsage {
    input_tokens: u64,
    output_tokens: u64,
    cache_read_input_tokens: u64,
    cache_creation_input_tokens: u64,
}

fn parse_anthropic_complete_response(raw: &str) -> Result<(Vec<Message>, Usage), String> {
    let val = json::parse(raw).map_err(|e| format!("Failed to parse Anthropic response: {e}"))?;
    let response = parse_anthropic_response_value(&val)?;

    let messages = response
        .content
        .into_iter()
        .map(|block| Message {
            role: response.role.clone(),
            content: match block {
                AnthropicContentBlock::Text(text) => Content::Text(text),
                AnthropicContentBlock::ToolUse { id, name, input } => {
                    Content::ToolUse(ToolUse { id, name, input })
                }
                AnthropicContentBlock::Thinking(thinking) => Content::Thinking(thinking),
            },
        })
        .collect();
    let usage = Usage {
        input_tokens: response.usage.input_tokens,
        output_tokens: response.usage.output_tokens,
        cache_read: response.usage.cache_read_input_tokens,
        cache_create: response.usage.cache_creation_input_tokens,
    };

    Ok((messages, usage))
}

fn parse_anthropic_response_value(val: &json::JsonValue) -> Result<AnthropicResponse, String> {
    let obj = val
        .as_object()
        .ok_or_else(|| "Invalid Anthropic response: expected an object".to_string())?;

    if obj.get("type").and_then(|v| v.as_str()) == Some("error") {
        let error = obj.get("error");
        let kind = error
            .and_then(|e| e.get("type"))
            .and_then(|v| v.as_str())
            .unwrap_or("api_error");
        let message = error
            .and_then(|e| e.get("message"))
            .and_then(|v| v.as_str())
            .unwrap_or("Anthropic returned an error response");
        return Err(format!("Anthropic API error ({kind}): {message}"));
    }

    let role = obj
        .get("role")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "Invalid Anthropic response: missing role".to_string())?
        .to_string();
    let blocks = obj
        .get("content")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "Invalid Anthropic response: missing content array".to_string())?;
    let mut content = Vec::with_capacity(blocks.len());
    for block in blocks {
        let block_type = block
            .get("type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Invalid Anthropic response: content block missing type".to_string())?;
        match block_type {
            "text" => {
                let text = block.get("text").and_then(|v| v.as_str()).ok_or_else(|| {
                    "Invalid Anthropic response: text block missing text".to_string()
                })?;
                content.push(AnthropicContentBlock::Text(text.to_string()));
            }
            "tool_use" => {
                let id = block.get("id").and_then(|v| v.as_str()).ok_or_else(|| {
                    "Invalid Anthropic response: tool_use block missing id".to_string()
                })?;
                let name = block.get("name").and_then(|v| v.as_str()).ok_or_else(|| {
                    "Invalid Anthropic response: tool_use block missing name".to_string()
                })?;
                let input = block
                    .get("input")
                    .filter(|value| value.as_object().is_some())
                    .ok_or_else(|| {
                        "Invalid Anthropic response: tool_use input must be an object".to_string()
                    })?;
                let tool_use = ToolUse {
                    id: id.to_string(),
                    name: name.to_string(),
                    input: json::serialize(input),
                };
                tool_use
                    .validate()
                    .map_err(|e| format!("Invalid Anthropic response: {e}"))?;
                content.push(AnthropicContentBlock::ToolUse {
                    id: tool_use.id,
                    name: tool_use.name,
                    input: tool_use.input,
                });
            }
            "thinking" => {
                let thinking = block
                    .get("thinking")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        "Invalid Anthropic response: thinking block missing thinking".to_string()
                    })?;
                content.push(AnthropicContentBlock::Thinking(thinking.to_string()));
            }
            _ => {}
        }
    }

    let usage = obj.get("usage");
    Ok(AnthropicResponse {
        role,
        content,
        usage: AnthropicUsage {
            input_tokens: usage
                .and_then(|value| value.get("input_tokens"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            output_tokens: usage
                .and_then(|value| value.get("output_tokens"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            cache_read_input_tokens: usage
                .and_then(|value| value.get("cache_read_input_tokens"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            cache_creation_input_tokens: usage
                .and_then(|value| value.get("cache_creation_input_tokens"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
        },
    })
}

/// Incremental Anthropic SSE reducer. Only final message state is retained;
/// processed event JSON is released after each line.
#[derive(Default)]
struct AnthropicStreamingResponse {
    messages: Vec<Message>,
    usage: Usage,
    current_tool_use: Option<(u64, ToolUse)>,
    tool_input_accum: String,
    text_accum: String,
    saw_message_stop: bool,
    lines_seen: usize,
    error: Option<String>,
}

impl AnthropicStreamingResponse {
    fn push_line<F>(&mut self, line: &str, on_chunk: &mut F)
    where
        F: FnMut(&str, &str),
    {
        self.lines_seen = self.lines_seen.saturating_add(1);
        if self.error.is_some() || self.saw_message_stop {
            return;
        }
        if let Err(error) = self.process_line(line, on_chunk) {
            self.error = Some(error);
        }
    }

    /// Emits any buffered text as an assistant message and clears the buffer.
    ///
    /// Called at every point a text block can end — its `content_block_stop`,
    /// the start of the next block, and end of stream — so the buffer is never
    /// carried across a boundary where it would be overwritten.
    fn flush_text(&mut self) {
        if !self.text_accum.is_empty() {
            self.messages.push(Message {
                role: "assistant".into(),
                content: Content::Text(std::mem::take(&mut self.text_accum)),
            });
        }
    }

    fn process_line<F>(&mut self, line: &str, on_chunk: &mut F) -> Result<(), String>
    where
        F: FnMut(&str, &str),
    {
        let Some(data) = line.strip_prefix("data: ") else {
            return Ok(());
        };
        if data == "[DONE]" {
            return Ok(());
        }

        let val = json::parse(data).map_err(|error| {
            format!(
                "Malformed Anthropic stream event on line {}: {error}",
                self.lines_seen
            )
        })?;
        let obj = val.as_object().ok_or_else(|| {
            format!(
                "Malformed Anthropic stream event on line {}: expected a JSON object",
                self.lines_seen
            )
        })?;
        match obj.get("type").and_then(|v| v.as_str()) {
            Some("content_block_start") => {
                if self.current_tool_use.is_some() {
                    return Err(
                        "Invalid Anthropic stream: new content block started before the tool call completed"
                            .into(),
                    );
                }
                if let Some(block) = obj.get("content_block") {
                    match block.get("type").and_then(|v| v.as_str()) {
                        Some("text") => {
                            // `text_accum` is assigned below, not appended, so
                            // anything still buffered has to be emitted first.
                            // content_block_stop normally does that; flushing
                            // here as well means a stream that omits the stop
                            // still cannot silently drop the previous block.
                            self.flush_text();
                            if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                                self.text_accum = text.to_string();
                                on_chunk(text, "text");
                            }
                        }
                        Some("tool_use") => {
                            let index =
                                obj.get("index").and_then(|v| v.as_u64()).ok_or_else(|| {
                                    "Invalid Anthropic stream: tool call start missing block index"
                                        .to_string()
                                })?;
                            self.flush_text();
                            self.tool_input_accum.clear();
                            self.current_tool_use = Some((
                                index,
                                ToolUse {
                                    name: block
                                        .get("name")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .to_string(),
                                    id: block
                                        .get("id")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .to_string(),
                                    // In real streaming responses content_block_start's `input` is
                                    // always `{}` — the actual arguments arrive as input_json_delta
                                    // fragments below and get assembled at content_block_stop.
                                    input: block
                                        .get("input")
                                        .map(json::serialize)
                                        .unwrap_or_default(),
                                },
                            ));
                        }
                        _ => {}
                    }
                    if let Some(thinking) = block.get("thinking").and_then(|v| v.as_str()) {
                        on_chunk(thinking, "thinking");
                    }
                }
            }
            Some("content_block_delta") => {
                if let Some(delta) = obj.get("delta") {
                    if let Some(text) = delta.get("text").and_then(|v| v.as_str()) {
                        self.text_accum.push_str(text);
                        on_chunk(text, "text");
                    }
                    if let Some(thinking) = delta.get("thinking").and_then(|v| v.as_str()) {
                        on_chunk(thinking, "thinking");
                    }
                    if delta.get("type").and_then(|v| v.as_str()) == Some("input_json_delta") {
                        let index = obj.get("index").and_then(|v| v.as_u64()).ok_or_else(|| {
                            "Invalid Anthropic stream: tool call delta missing block index"
                                .to_string()
                        })?;
                        match self.current_tool_use.as_ref() {
                            Some((tool_index, _)) if *tool_index == index => {}
                            Some(_) => {
                                return Err(
                                    "Invalid Anthropic stream: tool call delta block index does not match the open tool call"
                                        .into(),
                                );
                            }
                            None => {
                                return Err(
                                    "Invalid Anthropic stream: tool call delta has no open tool call"
                                        .into(),
                                );
                            }
                        }
                        let partial = delta
                            .get("partial_json")
                            .and_then(|v| v.as_str())
                            .ok_or_else(|| {
                                "Invalid Anthropic stream: tool call delta missing partial_json"
                                    .to_string()
                            })?;
                        self.tool_input_accum.push_str(partial);
                    }
                }
            }
            Some("content_block_stop") => {
                if let Some((tool_index, _)) = self.current_tool_use.as_ref() {
                    let index = obj.get("index").and_then(|v| v.as_u64()).ok_or_else(|| {
                        "Invalid Anthropic stream: tool call stop missing block index".to_string()
                    })?;
                    if *tool_index != index {
                        return Err(
                            "Invalid Anthropic stream: tool call stop block index does not match the open tool call"
                                .into(),
                        );
                    }
                    let (_, mut tool_use) = self.current_tool_use.take().unwrap();
                    if !self.tool_input_accum.is_empty() {
                        tool_use.input = self.tool_input_accum.clone();
                    }
                    self.tool_input_accum.clear();
                    tool_use
                        .validate()
                        .map_err(|error| format!("Invalid Anthropic stream: {error}"))?;
                    self.messages.push(Message {
                        role: "assistant".into(),
                        content: Content::ToolUse(tool_use),
                    });
                } else {
                    // A text (or thinking) block just closed. Emit its text now
                    // rather than carrying it to end of stream: the next
                    // content_block_start assigns over the buffer, so two
                    // consecutive text blocks used to drop the first — after
                    // on_chunk had already painted it on screen.
                    self.flush_text();
                }
            }
            Some("message_start") | Some("message_delta") => {
                if let Some(usage) = obj.get("usage") {
                    self.usage.input_tokens += usage
                        .get("input_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    self.usage.output_tokens += usage
                        .get("output_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                }
            }
            Some("message_stop") => self.saw_message_stop = true,
            Some("error") => {
                let error = obj.get("error");
                let kind = error
                    .and_then(|e| e.get("type"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("api_error");
                let message = error
                    .and_then(|e| e.get("message"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("Anthropic returned an error event");
                return Err(format!("Anthropic API stream error ({kind}): {message}"));
            }
            _ => {}
        }
        Ok(())
    }

    fn finish(mut self) -> Result<(Vec<Message>, Usage), String> {
        if let Some(error) = self.error {
            return Err(error);
        }
        if !self.saw_message_stop {
            return Err(
                "Incomplete Anthropic stream: missing message_stop completion marker".into(),
            );
        }
        if self.current_tool_use.is_some() {
            return Err("Invalid Anthropic stream: tool call did not complete".into());
        }
        // Backstop for a stream whose last text block never closed.
        self.flush_text();
        Ok((self.messages, self.usage))
    }
}

#[cfg(test)]
fn parse_anthropic_streaming_response(raw: &str) -> Result<(Vec<Message>, Usage), String> {
    let mut response = AnthropicStreamingResponse::default();
    let mut ignore_chunk = |_: &str, _: &str| {};
    for line in raw.lines() {
        response.push_line(line, &mut ignore_chunk);
    }
    response.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_body_escapes_model_name() {
        let model = "model\",\"injected\":true,\"tail";
        let body = build_request_body(&[], &[], "", model, 100, false).unwrap();
        let value = crate::json::parse(&body).unwrap();
        let obj = value.as_object().unwrap();
        assert_eq!(obj.get("model").and_then(|v| v.as_str()), Some(model));
        assert!(obj.get("injected").is_none());
    }

    #[test]
    fn provider_identity_and_fallback_catalog() {
        let p = AnthropicProvider::new();
        assert_eq!(p.name(), "anthropic");
        assert_eq!(p.default_model(), "claude-sonnet-5");
        let models = p.available_models();
        assert!(models.iter().any(|m| m.id.contains("sonnet")));
        assert!(models.len() >= 3);
    }

    #[test]
    fn parse_models_page_basic() {
        let body = r#"{
            "data": [
                {"id":"claude-opus-4-8","display_name":"Claude Opus 4.8","max_input_tokens":1000000,"type":"model"},
                {"id":"claude-sonnet-5","display_name":"Claude Sonnet 5","max_input_tokens":200000,"type":"model"},
                {"id":"not-claude","display_name":"Skip","type":"model"}
            ],
            "has_more": false,
            "last_id": "claude-sonnet-5"
        }"#;
        let page = parse_models_page(body).unwrap();
        assert_eq!(page.models.len(), 2);
        assert_eq!(page.models[0].id, "claude-opus-4-8");
        assert_eq!(page.models[0].name, "Claude Opus 4.8");
        assert_eq!(page.models[0].max_ctx, 1_000_000);
        assert_eq!(page.models[1].id, "claude-sonnet-5");
        assert!(!page.has_more);
    }

    #[test]
    fn parse_models_page_missing_max_uses_fallback() {
        let body = r#"{"data":[{"id":"claude-haiku-4-5","display_name":"Haiku","type":"model"}],"has_more":false}"#;
        let page = parse_models_page(body).unwrap();
        assert_eq!(page.models[0].max_ctx, 200_000);
    }

    #[test]
    fn request_body_serializes_special_characters_and_structured_tool_input() {
        let messages = vec![
            Message {
                role: "user\"role".into(),
                content: Content::Text("line 1\nline 2\r\t\0 café 🎉".into()),
            },
            Message {
                role: "assistant".into(),
                content: Content::ToolUse(ToolUse {
                    id: "tool\"id".into(),
                    name: "tool\nname".into(),
                    input: r#"{"text":"quoted \"value\"","items":[1,true]}"#.into(),
                }),
            },
            Message {
                role: "user".into(),
                content: Content::ToolResult(ToolResult {
                    tool_use_id: "tool\"id".into(),
                    content: "result\nwith\u{0008} control".into(),
                }),
            },
        ];
        let tools = vec![ToolDefinition {
            name: "tool\nname".into(),
            description: "description with \"quotes\" and café".into(),
            input_schema: r#"{"type":"object","properties":{"quoted\"key":{"type":"string"}}}"#
                .into(),
        }];

        let body = build_request_body(
            &messages,
            &tools,
            "system\nwith\tcontrols and 日本語",
            "claude\"model",
            1024,
            true,
        )
        .unwrap();
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();

        assert_eq!(value["model"], "claude\"model");
        assert_eq!(value["system"], "system\nwith\tcontrols and 日本語");
        assert_eq!(value["messages"][0]["role"], "user\"role");
        assert_eq!(
            value["messages"][0]["content"],
            "line 1\nline 2\r\t\0 café 🎉"
        );
        assert_eq!(value["messages"][1]["content"][0]["id"], "tool\"id");
        assert_eq!(
            value["messages"][1]["content"][0]["input"]["text"],
            "quoted \"value\""
        );
        assert_eq!(
            value["tools"][0]["input_schema"]["properties"]["quoted\"key"]["type"],
            "string"
        );
    }

    #[test]
    fn request_body_rejects_malformed_tool_input() {
        let messages = vec![Message {
            role: "assistant".into(),
            content: Content::ToolUse(ToolUse {
                id: "toolu_1".into(),
                name: "shell".into(),
                input: r#"{"command":"cargo test"},"role":"user""#.into(),
            }),
        }];

        let error =
            build_request_body(&messages, &[], "", "claude-sonnet-5", 1024, false).unwrap_err();

        assert!(error.contains("Invalid input for tool 'shell'"), "{error}");
    }

    #[test]
    fn request_body_rejects_malformed_tool_schema() {
        let tools = vec![ToolDefinition {
            name: "broken".into(),
            description: String::new(),
            input_schema: r#"{"type":"object"#.into(),
        }];

        let error =
            build_request_body(&[], &tools, "", "claude-sonnet-5", 1024, false).unwrap_err();

        assert!(
            error.contains("Invalid input schema for tool 'broken'"),
            "{error}"
        );
    }

    #[test]
    fn test_streaming_rejects_malformed_event() {
        let raw = "data: not-json\ndata: {\"type\":\"message_stop\"}\n";
        let error = parse_anthropic_streaming_response(raw).unwrap_err();

        assert!(error.contains("Malformed Anthropic stream event"));
    }

    #[test]
    fn test_streaming_surfaces_provider_error_event() {
        let raw = "data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n";
        let error = parse_anthropic_streaming_response(raw).unwrap_err();

        assert!(error.contains("overloaded_error"), "{error}");
        assert!(error.contains("Overloaded"), "{error}");
    }

    #[test]
    fn test_streaming_rejects_truncated_event_at_eof() {
        let raw = r#"data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"partial"}
"#;
        let error = parse_anthropic_streaming_response(raw).unwrap_err();

        assert!(error.contains("Malformed Anthropic stream event"));
    }

    fn texts_of(messages: &[Message]) -> Vec<String> {
        messages
            .iter()
            .filter_map(|m| match &m.content {
                Content::Text(t) => Some(t.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn test_streaming_keeps_both_consecutive_text_blocks() {
        let raw = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"FIRST\"}}\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"SECOND\"}}\n",
            "data: {\"type\":\"content_block_stop\",\"index\":1}\n",
            "data: {\"type\":\"message_stop\"}\n",
        );
        let (messages, _) = parse_anthropic_streaming_response(raw).unwrap();
        assert_eq!(texts_of(&messages), vec!["FIRST", "SECOND"]);
    }

    #[test]
    fn test_streaming_keeps_text_blocks_without_a_closing_stop() {
        // A stream that omits content_block_stop must still not lose the first
        // block, since the next start assigns over the buffer.
        let raw = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"ALPHA\"}}\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"BETA\"}}\n",
            "data: {\"type\":\"message_stop\"}\n",
        );
        let (messages, _) = parse_anthropic_streaming_response(raw).unwrap();
        assert_eq!(texts_of(&messages), vec!["ALPHA", "BETA"]);
    }

    #[test]
    fn test_streaming_text_deltas_still_coalesce_within_a_block() {
        // Flushing per block must not split a single block's deltas apart.
        let raw = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"par\"}}\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ti\"}}\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"al\"}}\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n",
            "data: {\"type\":\"message_stop\"}\n",
        );
        let (messages, _) = parse_anthropic_streaming_response(raw).unwrap();
        assert_eq!(texts_of(&messages), vec!["partial"]);
    }

    #[test]
    fn test_streaming_text_around_a_tool_call_keeps_order() {
        let raw = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"before\"}}\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"glob\",\"input\":{}}}\n",
            "data: {\"type\":\"content_block_stop\",\"index\":1}\n",
            "data: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"text\",\"text\":\"after\"}}\n",
            "data: {\"type\":\"content_block_stop\",\"index\":2}\n",
            "data: {\"type\":\"message_stop\"}\n",
        );
        let (messages, _) = parse_anthropic_streaming_response(raw).unwrap();
        assert_eq!(texts_of(&messages), vec!["before", "after"]);
        assert_eq!(messages.len(), 3, "tool call must stay between the texts");
        assert!(matches!(messages[1].content, Content::ToolUse(_)));
    }

    #[test]
    fn test_streaming_requires_message_stop_marker() {
        let raw = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"part\"}}\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ial\"}}\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n",
        );
        let error = parse_anthropic_streaming_response(raw).unwrap_err();

        assert!(error.contains("missing message_stop completion marker"));
    }

    #[test]
    fn test_streaming_rejects_tool_call_without_block_stop() {
        let raw = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"glob\",\"input\":{}}}\n",
            "data: {\"type\":\"message_stop\"}\n",
        );
        let error = parse_anthropic_streaming_response(raw).unwrap_err();

        assert!(error.contains("tool call did not complete"), "{error}");
    }

    #[test]
    fn test_streaming_rejects_tool_calls_missing_id_or_name() {
        let missing_id = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"name\":\"glob\",\"input\":{}}}\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n",
            "data: {\"type\":\"message_stop\"}\n",
        );
        let missing_name = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"input\":{}}}\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n",
            "data: {\"type\":\"message_stop\"}\n",
        );

        let id_error = parse_anthropic_streaming_response(missing_id).unwrap_err();
        assert!(id_error.contains("missing an id"), "{id_error}");
        let name_error = parse_anthropic_streaming_response(missing_name).unwrap_err();
        assert!(name_error.contains("missing a name"), "{name_error}");
    }

    #[test]
    fn test_streaming_rejects_tool_call_missing_input() {
        let raw = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"glob\"}}\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n",
            "data: {\"type\":\"message_stop\"}\n",
        );
        let error = parse_anthropic_streaming_response(raw).unwrap_err();

        assert!(error.contains("invalid JSON arguments"), "{error}");
    }

    #[test]
    fn test_streaming_rejects_mismatched_tool_block_stop() {
        let raw = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"glob\",\"input\":{}}}\n",
            "data: {\"type\":\"content_block_stop\",\"index\":1}\n",
            "data: {\"type\":\"message_stop\"}\n",
        );
        let error = parse_anthropic_streaming_response(raw).unwrap_err();

        assert!(error.contains("block index does not match"), "{error}");
    }

    #[test]
    fn test_streaming_rejects_invalid_tool_arguments() {
        let raw = r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"glob","input":{}}}
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"pattern\":"}}
data: {"type":"content_block_stop","index":0}
data: {"type":"message_stop"}
"#;
        let error = parse_anthropic_streaming_response(raw).unwrap_err();

        assert!(error.contains("invalid JSON arguments"), "{error}");
    }

    #[test]
    fn test_streaming_valid_incremental_text_requires_message_stop() {
        let raw = concat!(
            "data: {\"type\":\"message_start\",\"usage\":{\"input_tokens\":3}}\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"hel\"}}\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n",
            "data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":2}}\n",
            "data: {\"type\":\"message_stop\"}\n",
        );
        let (messages, usage) = parse_anthropic_streaming_response(raw).unwrap();

        assert_eq!(messages.len(), 1);
        assert!(matches!(&messages[0].content, Content::Text(text) if text == "hello"));
        assert_eq!(usage.input_tokens, 3);
        assert_eq!(usage.output_tokens, 2);
    }

    #[test]
    fn test_streaming_tool_use_reconstructs_args_from_input_json_delta() {
        // Mirrors what Anthropic actually sends: content_block_start's `input`
        // is `{}`, then the real arguments arrive as input_json_delta fragments.
        let raw = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"glob\",\"input\":{}}}\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"pattern\\\":\"}}\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"*.rs\\\"}\"}}\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n",
            "data: {\"type\":\"message_stop\"}\n",
        );
        let (msgs, _usage) = parse_anthropic_streaming_response(raw).unwrap();
        assert_eq!(msgs.len(), 1);
        match &msgs[0].content {
            Content::ToolUse(tu) => {
                assert_eq!(tu.id, "toolu_1");
                assert_eq!(tu.name, "glob");
                assert_eq!(tu.input, r#"{"pattern":"*.rs"}"#);
            }
            _ => panic!("expected ToolUse content"),
        }
    }

    #[test]
    fn test_streaming_multiple_tool_uses_each_get_own_args() {
        let raw = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"glob\",\"input\":{}}}\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{}\"}}\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_2\",\"name\":\"grep\",\"input\":{}}}\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"pattern\\\":\\\"foo\\\"}\"}}\n",
            "data: {\"type\":\"content_block_stop\",\"index\":1}\n",
            "data: {\"type\":\"message_stop\"}\n",
        );
        let (msgs, _usage) = parse_anthropic_streaming_response(raw).unwrap();
        assert_eq!(msgs.len(), 2);
        match &msgs[0].content {
            Content::ToolUse(tu) => assert_eq!(tu.input, "{}"),
            _ => panic!("expected ToolUse content"),
        }
        match &msgs[1].content {
            Content::ToolUse(tu) => assert_eq!(tu.input, r#"{"pattern":"foo"}"#),
            _ => panic!("expected ToolUse content"),
        }
    }

    #[test]
    fn test_streaming_tool_use_uses_content_block_input_without_deltas() {
        let raw = "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"glob\",\"input\":{\"pattern\":\"*.rs\"}}}\ndata: {\"type\":\"content_block_stop\",\"index\":0}\ndata: {\"type\":\"message_stop\"}\n";
        let (msgs, _usage) = parse_anthropic_streaming_response(raw).unwrap();
        assert_eq!(msgs.len(), 1);
        match &msgs[0].content {
            Content::ToolUse(tu) => assert_eq!(tu.input, r#"{"pattern":"*.rs"}"#),
            _ => panic!("expected ToolUse content"),
        }
    }

    #[test]
    fn test_complete_response_parses_text_for_non_streaming_callers() {
        let raw = r#"{
            "id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-6",
            "content":[{"type":"text","text":"The compacted summary or print output."}],
            "stop_reason":"end_turn","usage":{"input_tokens":12,"output_tokens":7}
        }"#;
        let (msgs, _usage) = parse_anthropic_complete_response(raw).unwrap();

        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, "assistant");
        match &msgs[0].content {
            Content::Text(text) => {
                assert_eq!(text, "The compacted summary or print output.");
            }
            _ => panic!("expected Text content"),
        }
    }

    #[test]
    fn test_complete_response_parses_tool_use_input() {
        let raw = r#"{
            "type":"message","role":"assistant",
            "content":[{"type":"tool_use","id":"toolu_1","name":"glob","input":{"pattern":"*.rs"}}],
            "usage":{"input_tokens":18,"output_tokens":9}
        }"#;
        let (msgs, _usage) = parse_anthropic_complete_response(raw).unwrap();

        assert_eq!(msgs.len(), 1);
        match &msgs[0].content {
            Content::ToolUse(tool_use) => {
                assert_eq!(tool_use.id, "toolu_1");
                assert_eq!(tool_use.name, "glob");
                assert_eq!(tool_use.input, r#"{"pattern":"*.rs"}"#);
            }
            _ => panic!("expected ToolUse content"),
        }
    }

    #[test]
    fn test_complete_response_preserves_escaped_tool_input() {
        let raw = r#"{
            "type":"message","role":"assistant",
            "content":[{"type":"tool_use","id":"toolu_1","name":"shell","input":{
                "command":"printf \"hello\"\n","path":"C:\\tmp","control":"tab\tand\u0001"
            }}]
        }"#;
        let (messages, _) = parse_anthropic_complete_response(raw).unwrap();
        let Content::ToolUse(tool_use) = &messages[0].content else {
            panic!("expected ToolUse content");
        };

        let input = json::parse(&tool_use.input).unwrap();
        assert_eq!(
            input.get("command").and_then(|v| v.as_str()),
            Some("printf \"hello\"\n")
        );
        assert_eq!(input.get("path").and_then(|v| v.as_str()), Some("C:\\tmp"));
        assert_eq!(
            input.get("control").and_then(|v| v.as_str()),
            Some("tab\tand\u{0001}")
        );
    }

    #[test]
    fn test_complete_response_rejects_non_object_tool_input() {
        for input in ["null", "\"bad\"", "[]"] {
            let raw = format!(
                r#"{{"type":"message","role":"assistant","content":[{{"type":"tool_use","id":"toolu_1","name":"shell","input":{input}}}]}}"#
            );
            let error = parse_anthropic_complete_response(&raw).unwrap_err();
            assert!(
                error.contains("input must be an object"),
                "unexpected error: {error}"
            );
        }
    }

    #[test]
    fn test_complete_response_parses_usage_including_cache_tokens() {
        let raw = r#"{
            "type":"message","role":"assistant","content":[],
            "usage":{
                "input_tokens":100,"output_tokens":25,
                "cache_read_input_tokens":80,"cache_creation_input_tokens":10
            }
        }"#;
        let (_msgs, usage) = parse_anthropic_complete_response(raw).unwrap();

        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 25);
        assert_eq!(usage.cache_read, 80);
        assert_eq!(usage.cache_create, 10);
    }

    #[test]
    fn test_complete_response_defaults_missing_usage() {
        let raw = r#"{
            "type":"message","role":"assistant",
            "content":[{"type":"text","text":"usable"}]
        }"#;
        let (messages, usage) = parse_anthropic_complete_response(raw).unwrap();

        assert_eq!(messages.len(), 1);
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.output_tokens, 0);
        assert_eq!(usage.cache_read, 0);
        assert_eq!(usage.cache_create, 0);
    }

    #[test]
    fn test_complete_response_defaults_invalid_usage_fields_independently() {
        let raw = r#"{
            "type":"message","role":"assistant","content":[],
            "usage":{
                "input_tokens":"unknown","output_tokens":7,
                "cache_read_input_tokens":-1,"cache_creation_input_tokens":3
            }
        }"#;
        let (_, usage) = parse_anthropic_complete_response(raw).unwrap();

        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.output_tokens, 7);
        assert_eq!(usage.cache_read, 0);
        assert_eq!(usage.cache_create, 3);
    }

    #[test]
    fn test_complete_response_returns_anthropic_api_error() {
        let raw = r#"{
            "type":"error",
            "error":{"type":"invalid_request_error","message":"max_tokens: must be positive"}
        }"#;
        let err = parse_anthropic_complete_response(raw).unwrap_err();

        assert!(err.contains("invalid_request_error"), "{err}");
        assert!(err.contains("max_tokens: must be positive"), "{err}");
    }
}
