use std::any::Any;
use std::collections::{HashMap, VecDeque};
use std::env;
use std::pin::Pin;

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue, USER_AGENT};
use reqwest::{Client, Url};
use serde_json::{Map, Value, json};
use uuid::Uuid;

use crate::chat_provider::{
    ChatProvider, ChatProviderError, ChatProviderErrorKind, StreamedMessage, ThinkingEffort,
    TokenUsage,
};
use crate::message::{
    ContentPart, Message, StreamedMessagePart, TextPart, ThinkPart, ToolCall, ToolCallFunction,
    ToolCallPart,
};
use crate::tooling::Tool;

type ByteStream = Pin<Box<dyn futures::Stream<Item = Result<Bytes, reqwest::Error>> + Send>>;
type ParsedOpenAIResponse = (Vec<StreamedMessagePart>, Option<String>, Option<TokenUsage>);

#[derive(Clone)]
pub struct OpenAILegacy {
    provider_name: String,
    model: String,
    api_key: String,
    base_url: Url,
    stream: bool,
    client: Client,
    generation_kwargs: Map<String, Value>,
    reasoning_key: Option<String>,
}

impl OpenAILegacy {
    pub fn new(
        model: impl Into<String>,
        api_key: Option<String>,
        base_url: Option<String>,
        default_headers: Option<HeaderMap>,
    ) -> Result<Self, ChatProviderError> {
        let api_key = api_key
            .filter(|value| !value.is_empty())
            .or_else(|| env::var("OPENAI_API_KEY").ok())
            .unwrap_or_default();
        let mut base_url = base_url
            .filter(|value| !value.is_empty())
            .or_else(|| env::var("OPENAI_BASE_URL").ok())
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
        if !base_url.ends_with('/') {
            base_url.push('/');
        }
        let base_url = Url::parse(&base_url).map_err(|err| {
            ChatProviderError::new(
                ChatProviderErrorKind::Other,
                format!("Invalid base URL: {err}"),
            )
        })?;

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(USER_AGENT, HeaderValue::from_static("KimiCLI"));
        if let Some(extra) = default_headers {
            for (k, v) in extra.iter() {
                if let Ok(value) = v.to_str() {
                    headers.insert(
                        k,
                        HeaderValue::from_str(value).unwrap_or_else(|_| v.clone()),
                    );
                } else {
                    headers.insert(k, v.clone());
                }
            }
        }

        let client = Client::builder()
            .default_headers(headers)
            .build()
            .map_err(|err| ChatProviderError::new(ChatProviderErrorKind::Other, err.to_string()))?;

        Ok(Self {
            provider_name: "openai".to_string(),
            model: model.into(),
            api_key,
            base_url,
            stream: true,
            client,
            generation_kwargs: Map::new(),
            reasoning_key: None,
        })
    }

    pub fn with_provider_name(mut self, provider_name: impl Into<String>) -> Self {
        self.provider_name = provider_name.into();
        self
    }

    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    pub fn with_reasoning_key(mut self, reasoning_key: Option<String>) -> Self {
        self.reasoning_key = reasoning_key;
        self
    }

    pub fn with_generation_kwargs(mut self, kwargs: Map<String, Value>) -> Self {
        for (k, v) in kwargs {
            self.generation_kwargs.insert(k, v);
        }
        self
    }

    pub fn model_parameters(&self) -> Map<String, Value> {
        let mut params = Map::new();
        params.insert(
            "base_url".to_string(),
            Value::String(self.base_url.to_string()),
        );
        for (k, v) in &self.generation_kwargs {
            params.insert(k.clone(), v.clone());
        }
        params
    }
}

#[async_trait]
impl ChatProvider for OpenAILegacy {
    fn name(&self) -> &str {
        &self.provider_name
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn thinking_effort(&self) -> Option<ThinkingEffort> {
        match self.generation_kwargs.get("reasoning_effort") {
            Some(Value::String(value)) => match value.as_str() {
                "low" | "minimal" => Some(ThinkingEffort::Low),
                "medium" => Some(ThinkingEffort::Medium),
                "high" => Some(ThinkingEffort::High),
                "xhigh" => Some(ThinkingEffort::XHigh),
                _ => Some(ThinkingEffort::Off),
            },
            Some(Value::Null) => Some(ThinkingEffort::Off),
            _ => None,
        }
    }

    async fn generate(
        &self,
        system_prompt: &str,
        tools: &[Tool],
        history: &[Message],
    ) -> Result<Box<dyn StreamedMessage>, ChatProviderError> {
        let mut messages = Vec::new();
        if !system_prompt.is_empty() {
            messages.push(json!({"role": "system", "content": system_prompt}));
        }
        for message in history {
            messages.push(convert_message(message, self.reasoning_key.as_deref())?);
        }

        let mut tool_defs = Vec::new();
        for tool in tools {
            tool_defs.push(convert_tool(tool));
        }

        let mut body = Map::new();
        body.insert("model".to_string(), Value::String(self.model.clone()));
        body.insert("messages".to_string(), Value::Array(messages));
        body.insert("tools".to_string(), Value::Array(tool_defs));
        body.insert("stream".to_string(), Value::Bool(self.stream));
        if self.stream {
            body.insert("stream_options".to_string(), json!({"include_usage": true}));
        }
        for (k, v) in &self.generation_kwargs {
            body.insert(k.clone(), v.clone());
        }

        let url = self
            .base_url
            .join("chat/completions")
            .map_err(|err| ChatProviderError::new(ChatProviderErrorKind::Other, err.to_string()))?;

        let mut request = self.client.post(url);
        if !self.api_key.is_empty() {
            request = request.header(AUTHORIZATION, format!("Bearer {}", self.api_key));
        }

        let resp = request
            .json(&body)
            .send()
            .await
            .map_err(map_reqwest_error)?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(ChatProviderError::new(
                ChatProviderErrorKind::Status(status.as_u16()),
                format!("OpenAI API error ({status}): {text}"),
            ));
        }

        if self.stream {
            Ok(Box::new(OpenAIStreamedMessage::new_stream(
                resp,
                self.reasoning_key.clone(),
            )))
        } else {
            let value: Value = resp.json().await.map_err(map_reqwest_error)?;
            let (parts, message_id, usage) =
                parse_non_stream_response(&value, self.reasoning_key.as_deref())?;
            Ok(Box::new(OpenAIStreamedMessage::new_parts(
                parts, message_id, usage,
            )))
        }
    }

    fn with_thinking(&self, effort: ThinkingEffort) -> Box<dyn ChatProvider> {
        let mut kwargs = Map::new();
        let reasoning_effort = match effort {
            ThinkingEffort::Off => None,
            ThinkingEffort::Low => Some("low"),
            ThinkingEffort::Medium => Some("medium"),
            ThinkingEffort::High => Some("high"),
            ThinkingEffort::XHigh => Some("xhigh"),
        };
        if let Some(value) = reasoning_effort {
            kwargs.insert(
                "reasoning_effort".to_string(),
                Value::String(value.to_string()),
            );
        } else {
            kwargs.insert("reasoning_effort".to_string(), Value::Null);
        }

        let provider = self.clone().with_generation_kwargs(kwargs);
        Box::new(provider)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub struct OpenAIStreamedMessage {
    stream: Option<ByteStream>,
    buffer: String,
    parts: VecDeque<StreamedMessagePart>,
    id: Option<String>,
    usage: Option<TokenUsage>,
    reasoning_key: Option<String>,
    tool_call_id_by_index: HashMap<i64, String>,
}

impl OpenAIStreamedMessage {
    pub fn new_stream(resp: reqwest::Response, reasoning_key: Option<String>) -> Self {
        let stream = resp.bytes_stream();
        Self {
            stream: Some(Box::pin(stream)),
            buffer: String::new(),
            parts: VecDeque::new(),
            id: None,
            usage: None,
            reasoning_key,
            tool_call_id_by_index: HashMap::new(),
        }
    }

    pub fn new_parts(
        parts: Vec<StreamedMessagePart>,
        id: Option<String>,
        usage: Option<TokenUsage>,
    ) -> Self {
        Self {
            stream: None,
            buffer: String::new(),
            parts: parts.into(),
            id,
            usage,
            reasoning_key: None,
            tool_call_id_by_index: HashMap::new(),
        }
    }

    fn ingest_chunk(&mut self, value: &Value) {
        if let Some(id) = value.get("id").and_then(|v| v.as_str()) {
            self.id = Some(id.to_string());
        }
        let usage_value = value.get("usage").or_else(|| {
            value
                .get("choices")
                .and_then(|v| v.as_array())
                .and_then(|choices| choices.first())
                .and_then(|choice| choice.get("usage"))
        });
        if let Some(usage) = usage_value
            && let Some(parsed) = parse_usage(usage)
        {
            self.usage = Some(parsed);
        }
        if let Some(choices) = value.get("choices").and_then(|v| v.as_array()) {
            for choice in choices {
                if let Some(delta) = choice.get("delta") {
                    ingest_delta(
                        delta,
                        self.reasoning_key.as_deref(),
                        &mut self.tool_call_id_by_index,
                        &mut self.parts,
                    );
                }
            }
        }
    }
}

#[async_trait]
impl StreamedMessage for OpenAIStreamedMessage {
    async fn next_part(&mut self) -> Result<Option<StreamedMessagePart>, ChatProviderError> {
        loop {
            if let Some(part) = self.parts.pop_front() {
                return Ok(Some(part));
            }
            let stream = match &mut self.stream {
                Some(stream) => stream,
                None => return Ok(None),
            };
            match stream.next().await {
                Some(Ok(bytes)) => {
                    let chunk = String::from_utf8_lossy(&bytes);
                    self.buffer.push_str(&chunk);
                    while let Some(pos) = self.buffer.find('\n') {
                        let line = self.buffer[..pos].trim().to_string();
                        self.buffer = self.buffer[pos + 1..].to_string();
                        if line.is_empty() {
                            continue;
                        }
                        if let Some(data) = line.strip_prefix("data: ") {
                            if data.trim() == "[DONE]" {
                                self.stream = None;
                                break;
                            }
                            let value: Value = serde_json::from_str(data).map_err(|err| {
                                ChatProviderError::new(
                                    ChatProviderErrorKind::Other,
                                    err.to_string(),
                                )
                            })?;
                            self.ingest_chunk(&value);
                        }
                    }
                }
                Some(Err(err)) => return Err(map_reqwest_error(err)),
                None => {
                    self.stream = None;
                    return Ok(None);
                }
            }
        }
    }

    fn id(&self) -> Option<String> {
        self.id.clone()
    }

    fn usage(&self) -> Option<TokenUsage> {
        self.usage.clone()
    }
}

fn convert_message(
    message: &Message,
    reasoning_key: Option<&str>,
) -> Result<Value, ChatProviderError> {
    let mut reasoning_content = String::new();
    let mut content_parts = Vec::new();
    for part in &message.content {
        match part {
            ContentPart::Think(think) => {
                if reasoning_key.is_some() {
                    reasoning_content.push_str(&think.think);
                }
            }
            _ => content_parts.push(part.clone()),
        }
    }

    let payload = serde_json::to_value(Message {
        role: message.role.clone(),
        content: content_parts,
        name: message.name.clone(),
        tool_calls: message.tool_calls.clone(),
        tool_call_id: message.tool_call_id.clone(),
        partial: message.partial,
    })
    .map_err(|err| ChatProviderError::new(ChatProviderErrorKind::Other, err.to_string()))?;

    let mut payload = strip_nulls(payload);
    if !reasoning_content.is_empty()
        && let Some(reasoning_key) = reasoning_key
        && let Value::Object(map) = &mut payload
    {
        map.insert(reasoning_key.to_string(), Value::String(reasoning_content));
    }
    Ok(payload)
}

fn strip_nulls(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut cleaned = serde_json::Map::new();
            for (key, val) in map {
                if val.is_null() {
                    continue;
                }
                cleaned.insert(key, strip_nulls(val));
            }
            Value::Object(cleaned)
        }
        Value::Array(items) => Value::Array(items.into_iter().map(strip_nulls).collect()),
        other => other,
    }
}

fn convert_tool(tool: &Tool) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
    })
}

fn parse_non_stream_response(
    value: &Value,
    reasoning_key: Option<&str>,
) -> Result<ParsedOpenAIResponse, ChatProviderError> {
    let message_id = value
        .get("id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let usage = value.get("usage").and_then(parse_usage);

    let choices = value
        .get("choices")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            ChatProviderError::new(ChatProviderErrorKind::Other, "Missing choices in response")
        })?;
    if choices.is_empty() {
        return Err(ChatProviderError::new(
            ChatProviderErrorKind::EmptyResponse,
            "The API returned an empty response.",
        ));
    }
    let message = choices[0].get("message").ok_or_else(|| {
        ChatProviderError::new(ChatProviderErrorKind::Other, "Missing message in response")
    })?;

    let mut parts = Vec::new();
    if let Some(reasoning_key) = reasoning_key
        && let Some(reasoning) = message
            .get(reasoning_key)
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
    {
        parts.push(StreamedMessagePart::Content(ContentPart::Think(
            ThinkPart {
                kind: "think".to_string(),
                think: reasoning.to_string(),
                encrypted: None,
            },
        )));
    }

    if let Some(content) = message.get("content") {
        ingest_content_value(content, &mut parts);
    }
    if let Some(refusal) = message.get("refusal").and_then(refusal_text) {
        parts.push(StreamedMessagePart::Content(ContentPart::Text(
            TextPart::new(refusal),
        )));
    }

    if let Some(tool_calls) = message.get("tool_calls").and_then(|v| v.as_array()) {
        for tool_call in tool_calls {
            if let Some(call) = parse_tool_call(tool_call) {
                parts.push(StreamedMessagePart::ToolCall(call));
            }
        }
    }

    Ok((parts, message_id, usage))
}

fn ingest_delta(
    delta: &Value,
    reasoning_key: Option<&str>,
    tool_call_id_by_index: &mut HashMap<i64, String>,
    parts: &mut VecDeque<StreamedMessagePart>,
) {
    if let Some(reasoning_key) = reasoning_key
        && let Some(reasoning) = delta
            .get(reasoning_key)
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
    {
        parts.push_back(StreamedMessagePart::Content(ContentPart::Think(
            ThinkPart {
                kind: "think".to_string(),
                think: reasoning.to_string(),
                encrypted: None,
            },
        )));
    }

    if let Some(content) = delta.get("content") {
        let mut content_parts = Vec::new();
        ingest_content_value(content, &mut content_parts);
        for part in content_parts {
            parts.push_back(part);
        }
    }
    if let Some(refusal) = delta.get("refusal").and_then(refusal_text) {
        parts.push_back(StreamedMessagePart::Content(ContentPart::Text(
            TextPart::new(refusal),
        )));
    }

    if let Some(tool_calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
        for tool_call in tool_calls {
            if let Some(part) = parse_tool_call_delta(tool_call, tool_call_id_by_index) {
                parts.push_back(part);
            }
        }
    }
}

fn ingest_content_value(content: &Value, parts: &mut Vec<StreamedMessagePart>) {
    if let Some(text) = content.as_str().filter(|s| !s.is_empty()) {
        parts.push(StreamedMessagePart::Content(ContentPart::Text(
            TextPart::new(text),
        )));
        return;
    }

    if let Some(items) = content.as_array() {
        for item in items {
            if let Some(text) = item
                .get("text")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
            {
                parts.push(StreamedMessagePart::Content(ContentPart::Text(
                    TextPart::new(text),
                )));
            }
        }
    }
}

fn refusal_text(value: &Value) -> Option<String> {
    if let Some(text) = value.as_str().filter(|s| !s.is_empty()) {
        return Some(text.to_string());
    }

    if let Some(text) = value
        .get("refusal")
        .or_else(|| value.get("text"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
    {
        return Some(text.to_string());
    }

    if let Some(items) = value.as_array() {
        let texts: Vec<String> = items
            .iter()
            .filter_map(|item| {
                item.get("refusal")
                    .or_else(|| item.get("text"))
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(ToString::to_string)
            })
            .collect();
        if !texts.is_empty() {
            return Some(texts.join("\n"));
        }
    }

    None
}

fn parse_tool_call(tool_call: &Value) -> Option<ToolCall> {
    let function = tool_call.get("function")?;
    let name = function.get("name")?.as_str()?.to_string();
    let arguments = function.get("arguments").and_then(stringify_json_value);
    let id = tool_call
        .get("id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    Some(ToolCall {
        kind: "function".to_string(),
        id,
        function: ToolCallFunction { name, arguments },
        extras: None,
    })
}

fn parse_tool_call_delta(
    tool_call: &Value,
    tool_call_id_by_index: &mut HashMap<i64, String>,
) -> Option<StreamedMessagePart> {
    let function = tool_call.get("function")?;
    let tool_call_id = stream_tool_call_id(tool_call, tool_call_id_by_index);
    let name = function
        .get("name")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty());
    let arguments = function
        .get("arguments")
        .and_then(stringify_json_value)
        .filter(|s| !s.is_empty());
    if let Some(name) = name {
        let call = ToolCall {
            kind: "function".to_string(),
            id: tool_call_id.clone(),
            function: ToolCallFunction {
                name: name.to_string(),
                arguments,
            },
            extras: None,
        };
        return Some(StreamedMessagePart::ToolCall(call));
    }
    if let Some(arguments) = arguments {
        let part = ToolCallPart {
            arguments_part: Some(arguments),
            tool_call_id: Some(tool_call_id),
        };
        return Some(StreamedMessagePart::ToolCallPart(part));
    }
    None
}

fn stream_tool_call_id(
    tool_call: &Value,
    tool_call_id_by_index: &mut HashMap<i64, String>,
) -> String {
    let index = tool_call.get("index").and_then(|v| v.as_i64());

    if let Some(index) = index
        && let Some(existing) = tool_call_id_by_index.get(&index)
    {
        return existing.clone();
    }

    if let Some(id) = tool_call
        .get("id")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
    {
        if let Some(index) = index {
            tool_call_id_by_index.insert(index, id.clone());
        }
        return id;
    }

    if let Some(index) = index {
        let generated = format!("tool_call_index_{index}");
        tool_call_id_by_index.insert(index, generated.clone());
        return generated;
    }

    Uuid::new_v4().to_string()
}

fn stringify_json_value(value: &Value) -> Option<String> {
    if let Some(text) = value.as_str() {
        return Some(text.to_string());
    }
    if value.is_null() {
        return None;
    }
    serde_json::to_string(value).ok()
}

fn parse_usage(value: &Value) -> Option<TokenUsage> {
    let prompt_tokens = value.get("prompt_tokens")?.as_i64()?;
    let completion_tokens = value.get("completion_tokens")?.as_i64()?;
    let mut cached = 0i64;
    if let Some(cached_tokens) = value.get("cached_tokens").and_then(|v| v.as_i64()) {
        cached = cached_tokens;
    } else if let Some(details) = value.get("prompt_tokens_details")
        && let Some(cached_tokens) = details.get("cached_tokens").and_then(|v| v.as_i64())
    {
        cached = cached_tokens;
    }
    let input_other = if prompt_tokens >= cached {
        prompt_tokens - cached
    } else {
        0
    };
    Some(TokenUsage {
        input_other,
        output: completion_tokens,
        input_cache_read: cached,
        input_cache_creation: 0,
    })
}

fn map_reqwest_error(err: reqwest::Error) -> ChatProviderError {
    if err.is_timeout() {
        ChatProviderError::new(ChatProviderErrorKind::Timeout, err.to_string())
    } else if err.is_connect() {
        ChatProviderError::new(ChatProviderErrorKind::Connection, err.to_string())
    } else {
        ChatProviderError::new(ChatProviderErrorKind::Other, err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_non_stream_response_with_reasoning_and_tool_call() {
        let value = json!({
            "id": "chatcmpl-test",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "prompt_tokens_details": { "cached_tokens": 2 }
            },
            "choices": [
                {
                    "message": {
                        "reasoning": "step by step",
                        "content": "hello",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {
                                    "name": "sum",
                                    "arguments": "{\"a\":1}"
                                }
                            }
                        ]
                    }
                }
            ]
        });

        let (parts, id, usage) = parse_non_stream_response(&value, Some("reasoning")).unwrap();
        assert_eq!(id.as_deref(), Some("chatcmpl-test"));
        assert_eq!(usage.expect("usage").input_cache_read, 2);
        assert_eq!(parts.len(), 3);
    }
}
