use std::collections::{HashMap, VecDeque};

use serde_json::Value;

use crate::chat_provider::{ChatProviderError, TokenUsage};
use crate::message::{
    ContentPart, StreamedMessagePart, TextPart, ThinkPart, ToolCall, ToolCallFunction, ToolCallPart,
};

pub(super) type ParsedAnthropicResponse =
    (Vec<StreamedMessagePart>, Option<String>, Option<TokenUsage>);

pub(super) fn parse_non_stream_response(
    value: &Value,
) -> Result<ParsedAnthropicResponse, ChatProviderError> {
    let message_id = value
        .get("id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let usage = value.get("usage").and_then(parse_usage);

    let mut parts = Vec::new();
    if let Some(content) = value.get("content").and_then(|v| v.as_array()) {
        for block in content {
            let kind = block
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            match kind {
                "text" => {
                    if let Some(text) = block
                        .get("text")
                        .and_then(|v| v.as_str())
                        .filter(|v| !v.is_empty())
                    {
                        parts.push(StreamedMessagePart::Content(ContentPart::Text(
                            TextPart::new(text),
                        )));
                    }
                }
                "thinking" => {
                    if let Some(thinking) = block
                        .get("thinking")
                        .and_then(|v| v.as_str())
                        .filter(|v| !v.is_empty())
                    {
                        parts.push(StreamedMessagePart::Content(ContentPart::Think(
                            ThinkPart {
                                kind: "think".to_string(),
                                think: thinking.to_string(),
                                encrypted: block
                                    .get("signature")
                                    .and_then(|v| v.as_str())
                                    .map(|v| v.to_string()),
                            },
                        )));
                    }
                }
                "redacted_thinking" => {
                    if let Some(signature) = block
                        .get("data")
                        .and_then(|v| v.as_str())
                        .filter(|v| !v.is_empty())
                    {
                        parts.push(StreamedMessagePart::Content(ContentPart::Think(
                            ThinkPart {
                                kind: "think".to_string(),
                                think: String::new(),
                                encrypted: Some(signature.to_string()),
                            },
                        )));
                    }
                }
                "tool_use" => {
                    if let Some(tool_call) = parse_tool_use_block(block) {
                        parts.push(StreamedMessagePart::ToolCall(tool_call));
                    }
                }
                _ => {}
            }
        }
    }

    Ok((parts, message_id, usage))
}

pub(super) fn ingest_content_block_start(
    block: &Value,
    block_index: Option<i64>,
    tool_call_id_by_content_block_index: &mut HashMap<i64, String>,
    parts: &mut VecDeque<StreamedMessagePart>,
) -> Result<(), ChatProviderError> {
    let kind = block
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or_default();

    match kind {
        "text" => {
            if let Some(text) = block
                .get("text")
                .and_then(|v| v.as_str())
                .filter(|v| !v.is_empty())
            {
                parts.push_back(StreamedMessagePart::Content(ContentPart::Text(
                    TextPart::new(text),
                )));
            }
        }
        "thinking" => {
            if let Some(thinking) = block
                .get("thinking")
                .and_then(|v| v.as_str())
                .filter(|v| !v.is_empty())
            {
                parts.push_back(StreamedMessagePart::Content(ContentPart::Think(
                    ThinkPart {
                        kind: "think".to_string(),
                        think: thinking.to_string(),
                        encrypted: block
                            .get("signature")
                            .and_then(|v| v.as_str())
                            .map(|v| v.to_string()),
                    },
                )));
            }
        }
        "redacted_thinking" => {
            if let Some(signature) = block
                .get("data")
                .and_then(|v| v.as_str())
                .filter(|v| !v.is_empty())
            {
                parts.push_back(StreamedMessagePart::Content(ContentPart::Think(
                    ThinkPart {
                        kind: "think".to_string(),
                        think: String::new(),
                        encrypted: Some(signature.to_string()),
                    },
                )));
            }
        }
        "tool_use" => {
            if let Some(tool_call) = parse_tool_use_block_stream_start(block) {
                if let Some(block_index) = block_index {
                    tool_call_id_by_content_block_index.insert(block_index, tool_call.id.clone());
                }
                parts.push_back(StreamedMessagePart::ToolCall(tool_call));
            }
        }
        _ => {}
    }

    Ok(())
}

pub(super) fn ingest_content_block_delta(
    delta: &Value,
    block_index: Option<i64>,
    tool_call_id_by_content_block_index: &HashMap<i64, String>,
    parts: &mut VecDeque<StreamedMessagePart>,
) {
    let delta_type = delta
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or_default();

    match delta_type {
        "text_delta" => {
            if let Some(text) = delta
                .get("text")
                .and_then(|v| v.as_str())
                .filter(|v| !v.is_empty())
            {
                parts.push_back(StreamedMessagePart::Content(ContentPart::Text(
                    TextPart::new(text),
                )));
            }
        }
        "thinking_delta" => {
            if let Some(thinking) = delta
                .get("thinking")
                .and_then(|v| v.as_str())
                .filter(|v| !v.is_empty())
            {
                parts.push_back(StreamedMessagePart::Content(ContentPart::Think(
                    ThinkPart {
                        kind: "think".to_string(),
                        think: thinking.to_string(),
                        encrypted: None,
                    },
                )));
            }
        }
        "signature_delta" => {
            if let Some(signature) = delta
                .get("signature")
                .and_then(|v| v.as_str())
                .filter(|v| !v.is_empty())
            {
                parts.push_back(StreamedMessagePart::Content(ContentPart::Think(
                    ThinkPart {
                        kind: "think".to_string(),
                        think: String::new(),
                        encrypted: Some(signature.to_string()),
                    },
                )));
            }
        }
        "input_json_delta" => {
            if let Some(partial_json) = delta
                .get("partial_json")
                .and_then(|v| v.as_str())
                .filter(|v| !v.is_empty())
            {
                let tool_call_id = block_index
                    .and_then(|index| tool_call_id_by_content_block_index.get(&index).cloned());
                parts.push_back(StreamedMessagePart::ToolCallPart(ToolCallPart {
                    arguments_part: Some(partial_json.to_string()),
                    tool_call_id,
                }));
            }
        }
        _ => {}
    }
}

fn parse_tool_use_block(block: &Value) -> Option<ToolCall> {
    let id = block.get("id").and_then(|v| v.as_str())?.to_string();
    let name = block.get("name").and_then(|v| v.as_str())?.to_string();

    let arguments = match block.get("input") {
        Some(input) => serde_json::to_string(input).ok(),
        None => Some("{}".to_string()),
    };

    Some(ToolCall {
        kind: "function".to_string(),
        id,
        function: ToolCallFunction { name, arguments },
        extras: None,
    })
}

fn parse_tool_use_block_stream_start(block: &Value) -> Option<ToolCall> {
    let id = block.get("id").and_then(|v| v.as_str())?.to_string();
    let name = block.get("name").and_then(|v| v.as_str())?.to_string();

    Some(ToolCall {
        kind: "function".to_string(),
        id,
        function: ToolCallFunction {
            name,
            arguments: None,
        },
        extras: None,
    })
}

fn parse_usage(value: &Value) -> Option<TokenUsage> {
    Some(TokenUsage {
        input_other: value
            .get("input_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0),
        output: value
            .get("output_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0),
        input_cache_read: value
            .get("cache_read_input_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0),
        input_cache_creation: value
            .get("cache_creation_input_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0),
    })
}

pub(super) fn merge_usage(current: &mut Option<TokenUsage>, delta: &Value) {
    let input_other = delta.get("input_tokens").and_then(|v| v.as_i64());
    let output = delta.get("output_tokens").and_then(|v| v.as_i64());
    let input_cache_read = delta
        .get("cache_read_input_tokens")
        .and_then(|v| v.as_i64());
    let input_cache_creation = delta
        .get("cache_creation_input_tokens")
        .and_then(|v| v.as_i64());

    if input_other.is_none()
        && output.is_none()
        && input_cache_read.is_none()
        && input_cache_creation.is_none()
    {
        return;
    }

    let usage = current.get_or_insert(TokenUsage {
        input_other: 0,
        output: 0,
        input_cache_read: 0,
        input_cache_creation: 0,
    });

    if let Some(value) = input_other {
        usage.input_other = value;
    }
    if let Some(value) = output {
        usage.output = value;
    }
    if let Some(value) = input_cache_read {
        usage.input_cache_read = value;
    }
    if let Some(value) = input_cache_creation {
        usage.input_cache_creation = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_parse_non_stream_response_extracts_content_and_tool_call() {
        let value = json!({
            "id": "msg-1",
            "usage": {
                "input_tokens": 30,
                "output_tokens": 11,
                "cache_read_input_tokens": 4,
                "cache_creation_input_tokens": 2
            },
            "content": [
                { "type": "text", "text": "hello" },
                { "type": "thinking", "thinking": "reasoning", "signature": "enc" },
                { "type": "tool_use", "id": "call-1", "name": "sum", "input": { "a": 1 } }
            ]
        });

        let (parts, id, usage) = parse_non_stream_response(&value).unwrap();
        assert_eq!(id.as_deref(), Some("msg-1"));
        assert_eq!(usage.expect("usage").input_cache_read, 4);
        assert_eq!(parts.len(), 3);
    }

    #[test]
    fn test_ingest_content_block_delta_supports_signature_and_tool_args() {
        let mut parts = VecDeque::new();
        let mut tool_call_id_by_content_block_index = HashMap::new();
        tool_call_id_by_content_block_index.insert(2, "call-1".to_string());
        ingest_content_block_delta(
            &json!({"type": "signature_delta", "signature": "enc"}),
            None,
            &tool_call_id_by_content_block_index,
            &mut parts,
        );
        ingest_content_block_delta(
            &json!({"type": "input_json_delta", "partial_json": "{\"a\":"}),
            Some(2),
            &tool_call_id_by_content_block_index,
            &mut parts,
        );
        assert_eq!(parts.len(), 2);
        if let Some(StreamedMessagePart::ToolCallPart(part)) = parts.get(1) {
            assert_eq!(part.tool_call_id.as_deref(), Some("call-1"));
        } else {
            panic!("expected second part to be ToolCallPart");
        }
    }

    #[test]
    fn test_merge_usage_preserves_existing_fields_for_partial_delta() {
        let mut usage = parse_usage(&json!({
            "input_tokens": 30,
            "output_tokens": 1,
            "cache_read_input_tokens": 4,
            "cache_creation_input_tokens": 2
        }));

        merge_usage(&mut usage, &json!({"output_tokens": 15}));

        let usage = usage.expect("usage");
        assert_eq!(usage.input_other, 30);
        assert_eq!(usage.output, 15);
        assert_eq!(usage.input_cache_read, 4);
        assert_eq!(usage.input_cache_creation, 2);
    }
}
