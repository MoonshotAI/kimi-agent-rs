use std::collections::{HashMap, VecDeque};
use std::pin::Pin;

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use serde_json::Value;

use crate::chat_provider::{ChatProviderError, ChatProviderErrorKind, StreamedMessage, TokenUsage};
use crate::message::StreamedMessagePart;

use super::request::map_reqwest_error;
use super::response::{ingest_content_block_delta, ingest_content_block_start, merge_usage};

type ByteStream = Pin<Box<dyn futures::Stream<Item = Result<Bytes, reqwest::Error>> + Send>>;

pub struct AnthropicStreamedMessage {
    stream: Option<ByteStream>,
    buffer: String,
    event_lines: Vec<String>,
    parts: VecDeque<StreamedMessagePart>,
    id: Option<String>,
    usage: Option<TokenUsage>,
    tool_call_id_by_content_block_index: HashMap<i64, String>,
}

impl AnthropicStreamedMessage {
    pub fn new_stream(resp: reqwest::Response) -> Self {
        let stream = resp.bytes_stream();
        Self {
            stream: Some(Box::pin(stream)),
            buffer: String::new(),
            event_lines: Vec::new(),
            parts: VecDeque::new(),
            id: None,
            usage: None,
            tool_call_id_by_content_block_index: HashMap::new(),
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
            event_lines: Vec::new(),
            parts: parts.into(),
            id,
            usage,
            tool_call_id_by_content_block_index: HashMap::new(),
        }
    }

    fn process_event_block(&mut self) -> Result<bool, ChatProviderError> {
        if self.event_lines.is_empty() {
            return Ok(false);
        }

        let mut data_lines = Vec::new();
        for line in &self.event_lines {
            if let Some(data) = line.strip_prefix("data:") {
                data_lines.push(data.trim().to_string());
            }
        }
        self.event_lines.clear();

        if data_lines.is_empty() {
            return Ok(false);
        }

        let data = data_lines.join("\n");
        if data.trim() == "[DONE]" {
            return Ok(true);
        }

        let value: Value = serde_json::from_str(&data)
            .map_err(|err| ChatProviderError::new(ChatProviderErrorKind::Other, err.to_string()))?;

        if let Some(err_value) = value.get("error") {
            let message = err_value
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("Anthropic stream error");
            return Err(ChatProviderError::new(
                ChatProviderErrorKind::Other,
                message.to_string(),
            ));
        }

        self.ingest_event(&value)
    }

    fn ingest_event(&mut self, value: &Value) -> Result<bool, ChatProviderError> {
        let event_type = value
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or_default();

        match event_type {
            "message_start" => {
                if let Some(message) = value.get("message") {
                    if let Some(id) = message.get("id").and_then(|v| v.as_str()) {
                        self.id = Some(id.to_string());
                    }
                    if let Some(usage) = message.get("usage") {
                        merge_usage(&mut self.usage, usage);
                    }
                }
            }
            "message_delta" => {
                if let Some(usage) = value.get("usage") {
                    merge_usage(&mut self.usage, usage);
                }
            }
            "content_block_start" => {
                if let Some(block) = value.get("content_block") {
                    let block_index = value.get("index").and_then(|v| v.as_i64());
                    ingest_content_block_start(
                        block,
                        block_index,
                        &mut self.tool_call_id_by_content_block_index,
                        &mut self.parts,
                    )?;
                }
            }
            "content_block_delta" => {
                if let Some(delta) = value.get("delta") {
                    let block_index = value.get("index").and_then(|v| v.as_i64());
                    ingest_content_block_delta(
                        delta,
                        block_index,
                        &self.tool_call_id_by_content_block_index,
                        &mut self.parts,
                    );
                }
            }
            "message_stop" => {
                return Ok(true);
            }
            _ => {}
        }

        Ok(false)
    }
}

#[async_trait]
impl StreamedMessage for AnthropicStreamedMessage {
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
                        let line = self.buffer[..pos].trim_end_matches('\r').to_string();
                        self.buffer = self.buffer[pos + 1..].to_string();

                        if line.is_empty() {
                            if self.process_event_block()? {
                                self.stream = None;
                                break;
                            }
                            continue;
                        }

                        self.event_lines.push(line);
                    }
                }
                Some(Err(err)) => return Err(map_reqwest_error(err)),
                None => {
                    let _ = self.process_event_block()?;
                    self.stream = None;
                    continue;
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
