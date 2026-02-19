use std::collections::{HashMap, HashSet, VecDeque};
use std::pin::Pin;

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use serde_json::Value;

use crate::chat_provider::{ChatProviderError, ChatProviderErrorKind, StreamedMessage, TokenUsage};
use crate::message::{ContentPart, StreamedMessagePart, TextPart, ThinkPart, ToolCallPart};

use super::response::{
    map_reqwest_error, parse_parts_from_response, parse_reasoning_parts_from_item,
    parse_tool_call_item, parse_usage,
};

type ByteStream = Pin<Box<dyn futures::Stream<Item = Result<Bytes, reqwest::Error>> + Send>>;

pub struct OpenAIResponsesStreamedMessage {
    stream: Option<ByteStream>,
    buffer: String,
    parts: VecDeque<StreamedMessagePart>,
    id: Option<String>,
    usage: Option<TokenUsage>,
    saw_emittable_parts: bool,
    tool_call_by_item_id: HashMap<String, String>,
    tool_call_by_output_index: HashMap<i64, String>,
    emitted_tool_call_ids: HashSet<String>,
    tool_calls_with_argument_deltas: HashSet<String>,
    tool_calls_with_completion_arguments: HashSet<String>,
    tool_calls_with_inline_arguments: HashSet<String>,
    reasoning_delta_keys: HashSet<String>,
    reasoning_items_with_text_by_id: HashSet<String>,
    reasoning_items_with_text_by_output_index: HashSet<i64>,
}

impl OpenAIResponsesStreamedMessage {
    pub fn new_stream(resp: reqwest::Response) -> Self {
        let stream = resp.bytes_stream();
        Self {
            stream: Some(Box::pin(stream)),
            buffer: String::new(),
            parts: VecDeque::new(),
            id: None,
            usage: None,
            saw_emittable_parts: false,
            tool_call_by_item_id: HashMap::new(),
            tool_call_by_output_index: HashMap::new(),
            emitted_tool_call_ids: HashSet::new(),
            tool_calls_with_argument_deltas: HashSet::new(),
            tool_calls_with_completion_arguments: HashSet::new(),
            tool_calls_with_inline_arguments: HashSet::new(),
            reasoning_delta_keys: HashSet::new(),
            reasoning_items_with_text_by_id: HashSet::new(),
            reasoning_items_with_text_by_output_index: HashSet::new(),
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
            saw_emittable_parts: false,
            tool_call_by_item_id: HashMap::new(),
            tool_call_by_output_index: HashMap::new(),
            emitted_tool_call_ids: HashSet::new(),
            tool_calls_with_argument_deltas: HashSet::new(),
            tool_calls_with_completion_arguments: HashSet::new(),
            tool_calls_with_inline_arguments: HashSet::new(),
            reasoning_delta_keys: HashSet::new(),
            reasoning_items_with_text_by_id: HashSet::new(),
            reasoning_items_with_text_by_output_index: HashSet::new(),
        }
    }

    fn ingest_event(&mut self, value: &Value) {
        if let Some(event_type) = value.get("type").and_then(|v| v.as_str()) {
            match event_type {
                "response.created" => {
                    if let Some(id) = value
                        .get("response")
                        .and_then(|response| response.get("id"))
                        .and_then(|id| id.as_str())
                    {
                        self.id = Some(id.to_string());
                    }
                }
                "response.output_text.delta" => {
                    if let Some(delta) = value.get("delta").and_then(|v| v.as_str())
                        && !delta.is_empty()
                    {
                        self.parts
                            .push_back(StreamedMessagePart::Content(ContentPart::Text(
                                TextPart::new(delta),
                            )));
                        self.saw_emittable_parts = true;
                    }
                }
                "response.refusal.delta" => {
                    if let Some(delta) = value
                        .get("delta")
                        .or_else(|| value.get("refusal"))
                        .and_then(|v| v.as_str())
                        .filter(|v| !v.is_empty())
                    {
                        self.parts
                            .push_back(StreamedMessagePart::Content(ContentPart::Text(
                                TextPart::new(delta),
                            )));
                        self.saw_emittable_parts = true;
                    }
                }
                "response.reasoning_summary_text.delta"
                | "response.reasoning_text.delta"
                | "response.reasoning.delta"
                | "response.reasoning_summary_text.done"
                | "response.reasoning_text.done"
                | "response.reasoning.done" => {
                    let is_done_event = event_type.ends_with(".done");
                    let text = value
                        .get("delta")
                        .and_then(|v| v.as_str())
                        .or_else(|| value.get("text").and_then(|v| v.as_str()));
                    if let Some(text) = text
                        && !text.is_empty()
                        && (!is_done_event || self.should_emit_reasoning_done_text(value))
                    {
                        if !is_done_event && let Some(key) = self.reasoning_event_key(value) {
                            self.reasoning_delta_keys.insert(key);
                        }
                        self.parts
                            .push_back(StreamedMessagePart::Content(ContentPart::Think(
                                ThinkPart {
                                    kind: "think".to_string(),
                                    think: text.to_string(),
                                    encrypted: None,
                                },
                            )));
                        self.mark_reasoning_item_text_emitted(value);
                        self.saw_emittable_parts = true;
                    }
                }
                "response.function_call_arguments.delta" => {
                    if let Some(delta) = value.get("delta").and_then(|v| v.as_str())
                        && !delta.is_empty()
                    {
                        let tool_call_id = self.resolve_tool_call_id(value);
                        if let Some(tool_call_id) = &tool_call_id {
                            self.tool_calls_with_argument_deltas
                                .insert(tool_call_id.clone());
                        }
                        self.parts
                            .push_back(StreamedMessagePart::ToolCallPart(ToolCallPart {
                                arguments_part: Some(delta.to_string()),
                                tool_call_id,
                            }));
                    }
                }
                "response.function_call_arguments.done" => {
                    let tool_call_id = self.resolve_tool_call_id(value);
                    let arguments = value
                        .get("arguments")
                        .and_then(|v| v.as_str())
                        .filter(|v| !v.is_empty())
                        .map(ToString::to_string);
                    if let Some(arguments) = arguments
                        && self.should_emit_completion_arguments(tool_call_id.as_deref())
                    {
                        if let Some(tool_call_id) = &tool_call_id {
                            self.tool_calls_with_completion_arguments
                                .insert(tool_call_id.clone());
                        }
                        self.parts
                            .push_back(StreamedMessagePart::ToolCallPart(ToolCallPart {
                                arguments_part: Some(arguments),
                                tool_call_id,
                            }));
                    }
                }
                "response.output_item.added" => {
                    if let Some(item) = value.get("item")
                        && let Some(tool_call) = parse_tool_call_item(item)
                    {
                        self.remember_tool_call_route(item, value, &tool_call.id);
                        self.emitted_tool_call_ids.insert(tool_call.id.clone());
                        if tool_call
                            .function
                            .arguments
                            .as_deref()
                            .is_some_and(|args| !args.is_empty())
                        {
                            self.tool_calls_with_inline_arguments
                                .insert(tool_call.id.clone());
                        }
                        self.parts
                            .push_back(StreamedMessagePart::ToolCall(tool_call));
                        self.saw_emittable_parts = true;
                    }
                }
                "response.output_item.done" => {
                    if let Some(item) = value.get("item") {
                        match item.get("type").and_then(|v| v.as_str()) {
                            Some("reasoning") => {
                                let item_id = item.get("id").and_then(|v| v.as_str());
                                let output_index =
                                    value.get("output_index").and_then(|v| v.as_i64());

                                let mut emitted_reasoning_text = false;
                                if !self.has_streamed_reasoning_text(item_id, output_index) {
                                    for think in parse_reasoning_parts_from_item(item) {
                                        self.parts.push_back(StreamedMessagePart::Content(
                                            ContentPart::Think(think),
                                        ));
                                        emitted_reasoning_text = true;
                                    }
                                }

                                if emitted_reasoning_text {
                                    if let Some(item_id) = item_id {
                                        self.reasoning_items_with_text_by_id
                                            .insert(item_id.to_string());
                                    }
                                    if let Some(output_index) = output_index {
                                        self.reasoning_items_with_text_by_output_index
                                            .insert(output_index);
                                    }
                                    self.saw_emittable_parts = true;
                                }

                                if !emitted_reasoning_text
                                    && let Some(encrypted) = item
                                        .get("encrypted_content")
                                        .and_then(|v| v.as_str())
                                        .filter(|v| !v.is_empty())
                                {
                                    self.parts.push_back(StreamedMessagePart::Content(
                                        ContentPart::Think(ThinkPart {
                                            kind: "think".to_string(),
                                            think: String::new(),
                                            encrypted: Some(encrypted.to_string()),
                                        }),
                                    ));
                                    self.saw_emittable_parts = true;
                                }
                            }
                            Some("function_call") => {
                                if let Some(tool_call) = parse_tool_call_item(item) {
                                    self.remember_tool_call_route(item, value, &tool_call.id);
                                    let tool_call_id = tool_call.id.clone();

                                    let already_emitted =
                                        self.emitted_tool_call_ids.contains(&tool_call_id);
                                    if !already_emitted {
                                        if tool_call
                                            .function
                                            .arguments
                                            .as_deref()
                                            .is_some_and(|args| !args.is_empty())
                                        {
                                            self.tool_calls_with_inline_arguments
                                                .insert(tool_call_id.clone());
                                        }
                                        self.parts.push_back(StreamedMessagePart::ToolCall(
                                            tool_call.clone(),
                                        ));
                                        self.emitted_tool_call_ids.insert(tool_call_id.clone());
                                        self.saw_emittable_parts = true;
                                    } else if let Some(arguments) =
                                        tool_call.function.arguments.filter(|args| !args.is_empty())
                                        && self
                                            .should_emit_completion_arguments(Some(&tool_call_id))
                                    {
                                        self.tool_calls_with_completion_arguments
                                            .insert(tool_call_id.clone());
                                        self.parts.push_back(StreamedMessagePart::ToolCallPart(
                                            ToolCallPart {
                                                arguments_part: Some(arguments),
                                                tool_call_id: Some(tool_call_id),
                                            },
                                        ));
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                "response.completed" => {
                    if let Some(response) = value.get("response") {
                        if let Some(id) = response.get("id").and_then(|v| v.as_str()) {
                            self.id = Some(id.to_string());
                        }
                        if let Some(usage) = response.get("usage") {
                            self.usage = parse_usage(usage);
                        }
                        if !self.saw_emittable_parts {
                            let mut final_parts = parse_parts_from_response(response);
                            if !final_parts.is_empty() {
                                self.saw_emittable_parts = true;
                            }
                            for part in final_parts.drain(..) {
                                self.parts.push_back(part);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        if let Some(response) = value.get("response") {
            if let Some(usage) = response.get("usage") {
                self.usage = parse_usage(usage);
            }
        } else if let Some(usage) = value.get("usage") {
            self.usage = parse_usage(usage);
        }
    }

    fn remember_tool_call_route(&mut self, item: &Value, event: &Value, call_id: &str) {
        if let Some(item_id) = item.get("id").and_then(|v| v.as_str()) {
            self.tool_call_by_item_id
                .insert(item_id.to_string(), call_id.to_string());
        }
        if let Some(output_index) = event.get("output_index").and_then(|v| v.as_i64()) {
            self.tool_call_by_output_index
                .insert(output_index, call_id.to_string());
        }
    }

    fn resolve_tool_call_id(&self, event: &Value) -> Option<String> {
        if let Some(call_id) = event.get("call_id").and_then(|v| v.as_str()) {
            return Some(call_id.to_string());
        }
        if let Some(call_id) = event
            .get("item")
            .and_then(|item| item.get("call_id"))
            .and_then(|v| v.as_str())
        {
            return Some(call_id.to_string());
        }
        if let Some(item_id) = event.get("item_id").and_then(|v| v.as_str())
            && let Some(call_id) = self.tool_call_by_item_id.get(item_id)
        {
            return Some(call_id.clone());
        }
        if let Some(item_id) = event
            .get("item")
            .and_then(|item| item.get("id"))
            .and_then(|v| v.as_str())
            && let Some(call_id) = self.tool_call_by_item_id.get(item_id)
        {
            return Some(call_id.clone());
        }
        if let Some(output_index) = event.get("output_index").and_then(|v| v.as_i64())
            && let Some(call_id) = self.tool_call_by_output_index.get(&output_index)
        {
            return Some(call_id.clone());
        }
        None
    }

    fn should_emit_completion_arguments(&self, tool_call_id: Option<&str>) -> bool {
        let Some(tool_call_id) = tool_call_id else {
            return true;
        };
        !self.tool_calls_with_argument_deltas.contains(tool_call_id)
            && !self
                .tool_calls_with_completion_arguments
                .contains(tool_call_id)
            && !self.tool_calls_with_inline_arguments.contains(tool_call_id)
    }

    fn reasoning_event_key(&self, event: &Value) -> Option<String> {
        if let Some(summary_index) = event.get("summary_index").and_then(|v| v.as_i64()) {
            if let Some(item_id) = event.get("item_id").and_then(|v| v.as_str()) {
                return Some(format!("item:{item_id}:summary:{summary_index}"));
            }
            if let Some(output_index) = event.get("output_index").and_then(|v| v.as_i64()) {
                return Some(format!("output:{output_index}:summary:{summary_index}"));
            }
            return Some(format!("summary:{summary_index}"));
        }

        let content_index = event.get("content_index").and_then(|v| v.as_i64())?;
        if let Some(item_id) = event.get("item_id").and_then(|v| v.as_str()) {
            return Some(format!("item:{item_id}:content:{content_index}"));
        }
        if let Some(output_index) = event.get("output_index").and_then(|v| v.as_i64()) {
            return Some(format!("output:{output_index}:content:{content_index}"));
        }
        Some(format!("content:{content_index}"))
    }

    fn should_emit_reasoning_done_text(&self, event: &Value) -> bool {
        if let Some(key) = self.reasoning_event_key(event) {
            return !self.reasoning_delta_keys.contains(&key);
        }
        true
    }

    fn mark_reasoning_item_text_emitted(&mut self, event: &Value) {
        if let Some(item_id) = event.get("item_id").and_then(|v| v.as_str()) {
            self.reasoning_items_with_text_by_id
                .insert(item_id.to_string());
        }
        if let Some(output_index) = event.get("output_index").and_then(|v| v.as_i64()) {
            self.reasoning_items_with_text_by_output_index
                .insert(output_index);
        }
    }

    fn has_streamed_reasoning_text(
        &self,
        item_id: Option<&str>,
        output_index: Option<i64>,
    ) -> bool {
        item_id
            .map(|id| self.reasoning_items_with_text_by_id.contains(id))
            .unwrap_or(false)
            || output_index
                .map(|idx| {
                    self.reasoning_items_with_text_by_output_index
                        .contains(&idx)
                })
                .unwrap_or(false)
    }
}

#[async_trait]
impl StreamedMessage for OpenAIResponsesStreamedMessage {
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
                            self.ingest_event(&value);
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_reasoning_summary_done_is_deduplicated_after_delta() {
        let mut stream = OpenAIResponsesStreamedMessage::new_parts(Vec::new(), None, None);

        stream.ingest_event(&json!({
            "type": "response.reasoning_summary_text.delta",
            "item_id": "rs_1",
            "output_index": 0,
            "summary_index": 0,
            "delta": "summary"
        }));
        stream.ingest_event(&json!({
            "type": "response.reasoning_summary_text.done",
            "item_id": "rs_1",
            "output_index": 0,
            "summary_index": 0,
            "text": "summary"
        }));

        assert_eq!(stream.parts.len(), 1);
    }
}
