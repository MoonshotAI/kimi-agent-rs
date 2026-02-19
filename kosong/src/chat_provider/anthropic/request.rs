use serde_json::{Value, json};

use crate::chat_provider::{ChatProviderError, ChatProviderErrorKind};
use crate::message::{ContentPart, Message, Role};
use crate::tooling::Tool;

pub(super) fn convert_message(message: &Message) -> Result<Value, ChatProviderError> {
    match message.role {
        Role::System => Ok(json!({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": format!("<system>{}</system>", message.extract_text("\n")),
                }
            ]
        })),
        Role::Tool => {
            let tool_use_id = message.tool_call_id.clone().ok_or_else(|| {
                ChatProviderError::new(
                    ChatProviderErrorKind::Other,
                    "Tool message missing tool_call_id",
                )
            })?;
            let content = convert_tool_result_content(&message.content)?;
            Ok(json!({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": content,
                        "is_error": false,
                    }
                ]
            }))
        }
        Role::User | Role::Assistant => {
            let role = if matches!(message.role, Role::User) {
                "user"
            } else {
                "assistant"
            };

            let mut content = Vec::new();
            for part in &message.content {
                match part {
                    ContentPart::Text(text) => {
                        content.push(json!({
                            "type": "text",
                            "text": text.text,
                        }));
                    }
                    ContentPart::ImageUrl(image) => {
                        let source = image_url_to_anthropic_source(&image.image_url.url)?;
                        content.push(json!({
                            "type": "image",
                            "source": source,
                        }));
                    }
                    ContentPart::Think(think) => {
                        if let Some(signature) = &think.encrypted {
                            content.push(json!({
                                "type": "thinking",
                                "thinking": think.think,
                                "signature": signature,
                            }));
                        }
                    }
                    ContentPart::AudioUrl(_) | ContentPart::VideoUrl(_) => {}
                }
            }

            for tool_call in message.tool_calls.clone().unwrap_or_default() {
                let input = match tool_call.function.arguments {
                    Some(arguments) if !arguments.is_empty() => {
                        let parsed: Value = serde_json::from_str(&arguments).map_err(|_| {
                            ChatProviderError::new(
                                ChatProviderErrorKind::Other,
                                "Tool call arguments must be valid JSON",
                            )
                        })?;
                        if !parsed.is_object() {
                            return Err(ChatProviderError::new(
                                ChatProviderErrorKind::Other,
                                "Tool call arguments must be a JSON object",
                            ));
                        }
                        parsed
                    }
                    _ => json!({}),
                };

                content.push(json!({
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "input": input,
                }));
            }

            Ok(json!({
                "role": role,
                "content": content,
            }))
        }
    }
}

fn image_url_to_anthropic_source(url: &str) -> Result<Value, ChatProviderError> {
    if let Some(data) = url.strip_prefix("data:") {
        let (media_type, payload) = data.split_once(";base64,").ok_or_else(|| {
            ChatProviderError::new(
                ChatProviderErrorKind::Other,
                format!("Invalid data URL for image: {url}"),
            )
        })?;

        if !matches!(
            media_type,
            "image/png" | "image/jpeg" | "image/gif" | "image/webp"
        ) {
            return Err(ChatProviderError::new(
                ChatProviderErrorKind::Other,
                format!("Unsupported media type for base64 image: {media_type}, url: {url}"),
            ));
        }

        return Ok(json!({
            "type": "base64",
            "media_type": media_type,
            "data": payload,
        }));
    }

    Ok(json!({
        "type": "url",
        "url": url,
    }))
}

fn convert_tool_result_content(parts: &[ContentPart]) -> Result<Value, ChatProviderError> {
    let mut blocks = Vec::new();
    for part in parts {
        match part {
            ContentPart::Text(text) => {
                if !text.text.is_empty() {
                    blocks.push(json!({
                        "type": "text",
                        "text": text.text.clone(),
                    }));
                }
            }
            ContentPart::ImageUrl(image) => {
                let source = image_url_to_anthropic_source(&image.image_url.url)?;
                blocks.push(json!({
                    "type": "image",
                    "source": source,
                }));
            }
            other => {
                return Err(ChatProviderError::new(
                    ChatProviderErrorKind::Other,
                    format!("Anthropic API does not support {other:?} in tool result"),
                ));
            }
        }
    }
    Ok(Value::Array(blocks))
}

pub(super) fn convert_tool(tool: &Tool) -> Value {
    json!({
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters,
    })
}
pub(super) fn extract_beta_header(beta_features: Option<Value>) -> Option<String> {
    match beta_features {
        Some(Value::String(value)) => {
            if value.is_empty() {
                None
            } else {
                Some(value)
            }
        }
        Some(Value::Array(values)) => {
            let mut features = Vec::new();
            for value in values {
                match value {
                    Value::String(value) if !value.is_empty() => features.push(value),
                    Value::Null => {}
                    other => features.push(other.to_string()),
                }
            }
            if features.is_empty() {
                None
            } else {
                Some(features.join(","))
            }
        }
        _ => None,
    }
}

pub(super) fn apply_extra_headers(
    mut request: reqwest::RequestBuilder,
    extra_headers: Option<Value>,
) -> reqwest::RequestBuilder {
    if let Some(Value::Object(headers)) = extra_headers {
        for (name, value) in headers {
            let header_value = match value {
                Value::String(text) => text,
                Value::Null => continue,
                other => other.to_string(),
            };
            if header_value.is_empty() {
                continue;
            }
            request = request.header(name, header_value);
        }
    }
    request
}

pub(super) fn mark_last_cacheable_message_block(messages: &mut [Value]) {
    let Some(last_message) = messages.last_mut() else {
        return;
    };
    let Some(content_blocks) = last_message
        .get_mut("content")
        .and_then(Value::as_array_mut)
    else {
        return;
    };
    let Some(last_block) = content_blocks.last_mut() else {
        return;
    };
    let Some(block_type) = last_block.get("type").and_then(Value::as_str) else {
        return;
    };
    if !supports_anthropic_cache_control(block_type) {
        return;
    }
    if let Some(block) = last_block.as_object_mut() {
        block.insert(
            "cache_control".to_string(),
            json!({
                "type": "ephemeral",
            }),
        );
    }
}

pub(super) fn mark_last_tool_definition(tool_defs: &mut [Value]) {
    let Some(last_tool) = tool_defs.last_mut() else {
        return;
    };
    if let Some(tool) = last_tool.as_object_mut() {
        tool.insert(
            "cache_control".to_string(),
            json!({
                "type": "ephemeral",
            }),
        );
    }
}

fn supports_anthropic_cache_control(block_type: &str) -> bool {
    matches!(
        block_type,
        "text"
            | "image"
            | "document"
            | "search_result"
            | "tool_use"
            | "tool_result"
            | "server_tool_use"
            | "web_search_tool_result"
    )
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

    #[test]
    fn test_mark_last_cacheable_message_block_and_tool_definition() {
        let mut messages = vec![
            json!({
                "role": "user",
                "content": [{"type": "text", "text": "first"}]
            }),
            json!({
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "call_1", "name": "sum", "input": {}}]
            }),
        ];
        mark_last_cacheable_message_block(&mut messages);
        assert_eq!(
            messages[1]["content"][0]["cache_control"],
            json!({"type": "ephemeral"})
        );

        let mut tools = vec![
            json!({"name": "a", "description": "a", "input_schema": {}}),
            json!({"name": "b", "description": "b", "input_schema": {}}),
        ];
        mark_last_tool_definition(&mut tools);
        assert_eq!(tools[1]["cache_control"], json!({"type": "ephemeral"}));
    }
}
