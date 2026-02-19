use serde_json::Value;
use uuid::Uuid;

use crate::chat_provider::{ChatProviderError, ChatProviderErrorKind, TokenUsage};
use crate::message::{
    ContentPart, StreamedMessagePart, TextPart, ThinkPart, ToolCall, ToolCallFunction,
};

pub(super) type ParsedResponses = (Vec<StreamedMessagePart>, Option<String>, Option<TokenUsage>);

pub(super) fn parse_non_stream_response(value: &Value) -> ParsedResponses {
    let id = value
        .get("id")
        .and_then(|v| v.as_str())
        .map(|v| v.to_string());
    let usage = value.get("usage").and_then(parse_usage);
    let parts = parse_parts_from_response(value);
    (parts, id, usage)
}

pub(super) fn parse_parts_from_response(response: &Value) -> Vec<StreamedMessagePart> {
    let mut parts = Vec::new();
    let mut saw_message_text = false;

    if let Some(output) = response.get("output").and_then(|v| v.as_array()) {
        for item in output {
            let item_type = item
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            match item_type {
                "message" => {
                    if let Some(content) = item.get("content").and_then(|v| v.as_array()) {
                        for block in content {
                            let block_type = block
                                .get("type")
                                .and_then(|v| v.as_str())
                                .unwrap_or_default();
                            if matches!(block_type, "output_text" | "text" | "refusal")
                                && let Some(text) = block
                                    .get("text")
                                    .or_else(|| block.get("refusal"))
                                    .and_then(|v| v.as_str())
                                    .filter(|v| !v.is_empty())
                            {
                                saw_message_text = true;
                                parts.push(StreamedMessagePart::Content(ContentPart::Text(
                                    TextPart::new(text),
                                )));
                            }
                        }
                    }
                }
                "reasoning" => {
                    for think in parse_reasoning_parts_from_item(item) {
                        parts.push(StreamedMessagePart::Content(ContentPart::Think(think)));
                    }
                }
                "function_call" => {
                    if let Some(tool_call) = parse_tool_call_item(item) {
                        parts.push(StreamedMessagePart::ToolCall(tool_call));
                    }
                }
                _ => {}
            }
        }
    }

    // `output_text` is a convenience aggregate of message output text. Use it only
    // when structured output did not yield any message text.
    if !saw_message_text
        && let Some(output_text) = response
            .get("output_text")
            .and_then(|v| v.as_str())
            .filter(|v| !v.is_empty())
    {
        parts.push(StreamedMessagePart::Content(ContentPart::Text(
            TextPart::new(output_text),
        )));
    }

    parts
}

pub(super) fn parse_reasoning_parts_from_item(item: &Value) -> Vec<ThinkPart> {
    let encrypted = item
        .get("encrypted_content")
        .and_then(|v| v.as_str())
        .filter(|v| !v.is_empty())
        .map(|v| v.to_string());

    let content_texts = item
        .get("content")
        .and_then(|v| v.as_array())
        .map(|blocks| parse_reasoning_text_blocks(blocks, true))
        .unwrap_or_default();
    let summary_texts = item
        .get("summary")
        .and_then(|v| v.as_array())
        .map(|blocks| parse_reasoning_text_blocks(blocks, false))
        .unwrap_or_default();

    let texts = if !content_texts.is_empty() {
        content_texts
    } else {
        summary_texts
    };

    texts
        .into_iter()
        .map(|text| ThinkPart {
            kind: "think".to_string(),
            think: text,
            encrypted: encrypted.clone(),
        })
        .collect()
}

fn parse_reasoning_text_blocks(blocks: &[Value], prefer_content_types: bool) -> Vec<String> {
    let mut texts = Vec::new();
    for block in blocks {
        let block_type = block
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let allowed = if prefer_content_types {
            matches!(block_type, "" | "reasoning_text" | "text")
        } else {
            matches!(block_type, "" | "summary_text" | "text")
        };
        if !allowed {
            continue;
        }
        if let Some(text) = block
            .get("text")
            .and_then(|v| v.as_str())
            .filter(|v| !v.is_empty())
        {
            texts.push(text.to_string());
        }
    }
    texts
}

pub(super) fn parse_tool_call_item(item: &Value) -> Option<ToolCall> {
    let item_type = item.get("type").and_then(|v| v.as_str())?;
    if item_type != "function_call" {
        return None;
    }

    let id = item
        .get("call_id")
        .or_else(|| item.get("id"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| Uuid::new_v4().to_string());

    let name = item
        .get("name")
        .or_else(|| item.get("function").and_then(|v| v.get("name")))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())?;

    let arguments = item
        .get("arguments")
        .or_else(|| item.get("function").and_then(|v| v.get("arguments")))
        .and_then(stringify_json_value);

    Some(ToolCall {
        kind: "function".to_string(),
        id,
        function: ToolCallFunction { name, arguments },
        extras: None,
    })
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

pub(super) fn parse_usage(value: &Value) -> Option<TokenUsage> {
    let input_tokens = value
        .get("input_tokens")
        .and_then(|v| v.as_i64())
        .or_else(|| value.get("prompt_tokens").and_then(|v| v.as_i64()));
    let output_tokens = value
        .get("output_tokens")
        .and_then(|v| v.as_i64())
        .or_else(|| value.get("completion_tokens").and_then(|v| v.as_i64()));
    if input_tokens.is_none() && output_tokens.is_none() {
        return None;
    }

    let input_tokens = input_tokens.unwrap_or(0);
    let output_tokens = output_tokens.unwrap_or(0);

    let details = value.get("input_tokens_details");
    let cache_read = details
        .and_then(|v| v.get("cached_tokens"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let cache_creation = details
        .and_then(|v| v.get("cache_creation_tokens"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let input_other = (input_tokens - cache_read - cache_creation).max(0);

    Some(TokenUsage {
        input_other,
        output: output_tokens,
        input_cache_read: cache_read,
        input_cache_creation: cache_creation,
    })
}

pub(super) fn map_reqwest_error(err: reqwest::Error) -> ChatProviderError {
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
    use serde_json::json;

    #[test]
    fn test_parse_non_stream_response_extracts_text_usage_and_tool_call() {
        let value = json!({
            "id": "resp-1",
            "output_text": "hello",
            "usage": {
                "input_tokens": 20,
                "output_tokens": 10,
                "input_tokens_details": {
                    "cached_tokens": 5,
                    "cache_creation_tokens": 3
                }
            },
            "output": [
                {
                    "type": "message",
                    "content": [
                        { "type": "output_text", "text": "hello" }
                    ]
                },
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "sum",
                    "arguments": "{\"a\":1}"
                }
            ]
        });

        let (parts, id, usage) = parse_non_stream_response(&value);
        assert_eq!(id.as_deref(), Some("resp-1"));
        let usage = usage.expect("usage");
        assert_eq!(usage.input_other, 12);
        assert_eq!(parts.len(), 2);
    }

    #[test]
    fn test_parse_usage_returns_none_for_null_or_empty_payload() {
        assert_eq!(parse_usage(&Value::Null), None);
        assert_eq!(parse_usage(&json!({})), None);
    }

    #[test]
    fn test_parse_non_stream_response_falls_back_to_output_text() {
        let value = json!({
            "id": "resp-2",
            "output_text": "fallback-text",
            "output": []
        });

        let (parts, id, usage) = parse_non_stream_response(&value);
        assert_eq!(id.as_deref(), Some("resp-2"));
        assert!(usage.is_none());
        assert_eq!(
            parts,
            vec![StreamedMessagePart::Content(ContentPart::Text(
                TextPart::new("fallback-text")
            ))]
        );
    }

    #[test]
    fn test_parse_non_stream_response_prefers_reasoning_content() {
        let value = json!({
            "output": [
                {
                    "type": "reasoning",
                    "encrypted_content": "enc-1",
                    "content": [
                        { "type": "reasoning_text", "text": "full-trace" }
                    ],
                    "summary": [
                        { "type": "summary_text", "text": "summary-trace" }
                    ]
                }
            ]
        });

        let (parts, _, _) = parse_non_stream_response(&value);
        assert_eq!(parts.len(), 1);
        assert_eq!(
            parts,
            vec![StreamedMessagePart::Content(ContentPart::Think(
                ThinkPart {
                    kind: "think".to_string(),
                    think: "full-trace".to_string(),
                    encrypted: Some("enc-1".to_string()),
                }
            ))]
        );
    }
}
