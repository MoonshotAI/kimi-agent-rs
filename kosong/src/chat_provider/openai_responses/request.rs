use serde_json::{Map, Value, json};

use crate::message::{ContentPart, Message};

pub(super) fn convert_history_message(
    message: &Message,
    input: &mut Vec<Value>,
    use_developer_role: bool,
) {
    if matches!(message.role, crate::message::Role::Tool) {
        let output = convert_tool_message_output(&message.content);
        input.push(json!({
            "type": "function_call_output",
            "call_id": message.tool_call_id.clone().unwrap_or_default(),
            "output": output,
        }));
        return;
    }

    let role = match message.role {
        crate::message::Role::System => {
            if use_developer_role {
                "developer"
            } else {
                "system"
            }
        }
        crate::message::Role::User => "user",
        crate::message::Role::Assistant => "assistant",
        crate::message::Role::Tool => "tool",
    };

    convert_non_tool_message_content(role, &message.content, input);

    if let Some(tool_calls) = &message.tool_calls {
        for tool_call in tool_calls {
            input.push(json!({
                "type": "function_call",
                "call_id": tool_call.id,
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments.clone().unwrap_or_else(|| "{}".to_string()),
            }));
        }
    }
}

fn convert_non_tool_message_content(role: &str, content: &[ContentPart], input: &mut Vec<Value>) {
    if content.is_empty() {
        return;
    }

    let mut pending_parts = Vec::new();
    let mut index = 0usize;
    while index < content.len() {
        match &content[index] {
            ContentPart::Think(think) => {
                flush_pending_message_parts(role, &mut pending_parts, input);

                let encrypted = think.encrypted.clone();
                let mut summary = vec![json!({
                    "type": "summary_text",
                    "text": think.think.clone(),
                })];
                index += 1;
                while index < content.len() {
                    match &content[index] {
                        ContentPart::Think(next) if next.encrypted == encrypted => {
                            summary.push(json!({
                                "type": "summary_text",
                                "text": next.think.clone(),
                            }));
                            index += 1;
                        }
                        _ => break,
                    }
                }

                let mut reasoning_item = Map::new();
                reasoning_item.insert("type".to_string(), Value::String("reasoning".to_string()));
                reasoning_item.insert("summary".to_string(), Value::Array(summary));
                if let Some(encrypted_content) = encrypted {
                    reasoning_item.insert(
                        "encrypted_content".to_string(),
                        Value::String(encrypted_content),
                    );
                }
                input.push(Value::Object(reasoning_item));
            }
            other => {
                pending_parts.push(other.clone());
                index += 1;
            }
        }
    }

    flush_pending_message_parts(role, &mut pending_parts, input);
}

fn flush_pending_message_parts(
    role: &str,
    pending_parts: &mut Vec<ContentPart>,
    input: &mut Vec<Value>,
) {
    if pending_parts.is_empty() {
        return;
    }

    let content = if role == "assistant" {
        convert_content_parts_to_output_items(pending_parts)
    } else {
        convert_content_parts_to_input_items(pending_parts)
    };
    pending_parts.clear();

    if content.is_empty() {
        return;
    }

    input.push(json!({
        "type": "message",
        "role": role,
        "content": content,
    }));
}

fn convert_tool_message_output(parts: &[ContentPart]) -> Value {
    let output_items = convert_content_parts_to_function_output_items(parts);
    if output_items.is_empty() {
        Value::String(extract_text_from_parts(parts, "\n"))
    } else {
        Value::Array(output_items)
    }
}

fn convert_content_parts_to_input_items(parts: &[ContentPart]) -> Vec<Value> {
    let mut converted = Vec::new();
    for part in parts {
        match part {
            ContentPart::Text(text) if !text.text.is_empty() => {
                converted.push(json!({
                    "type": "input_text",
                    "text": text.text.clone(),
                }));
            }
            ContentPart::ImageUrl(image) => {
                converted.push(json!({
                    "type": "input_image",
                    "detail": "auto",
                    "image_url": image.image_url.url.clone(),
                }));
            }
            ContentPart::AudioUrl(audio) => {
                if let Some(mapped) = map_audio_url_to_input_item(&audio.audio_url.url) {
                    converted.push(mapped);
                }
            }
            ContentPart::Text(_) => {}
            ContentPart::Think(_) | ContentPart::VideoUrl(_) => {}
        }
    }
    converted
}

fn convert_content_parts_to_output_items(parts: &[ContentPart]) -> Vec<Value> {
    let mut converted = Vec::new();
    for part in parts {
        if let ContentPart::Text(text) = part
            && !text.text.is_empty()
        {
            converted.push(json!({
                "type": "output_text",
                "text": text.text.clone(),
                "annotations": [],
            }));
        }
    }
    converted
}

fn convert_content_parts_to_function_output_items(parts: &[ContentPart]) -> Vec<Value> {
    let mut converted = Vec::new();
    for part in parts {
        match part {
            ContentPart::Text(text) if !text.text.is_empty() => {
                converted.push(json!({
                    "type": "input_text",
                    "text": text.text.clone(),
                }));
            }
            ContentPart::ImageUrl(image) => {
                converted.push(json!({
                    "type": "input_image",
                    "image_url": image.image_url.url.clone(),
                }));
            }
            ContentPart::AudioUrl(audio) => {
                if let Some(mapped) = map_audio_url_to_file_content_item(&audio.audio_url.url) {
                    converted.push(mapped);
                }
            }
            ContentPart::Text(_) => {}
            ContentPart::Think(_) | ContentPart::VideoUrl(_) => {}
        }
    }
    converted
}

fn map_audio_url_to_input_item(url: &str) -> Option<Value> {
    if let Some((media_type, data)) = parse_data_url(url) {
        if !media_type.starts_with("audio/") {
            return None;
        }
        let subtype = media_type
            .split('/')
            .nth(1)?
            .split(';')
            .next()?
            .to_ascii_lowercase();
        let ext = match subtype.as_str() {
            "mp3" | "mpeg" => "mp3",
            "wav" => "wav",
            _ => return None,
        };
        return Some(json!({
            "type": "input_file",
            "file_data": data,
            "filename": format!("inline.{ext}"),
        }));
    }

    if url.starts_with("http://") || url.starts_with("https://") {
        return Some(json!({
            "type": "input_file",
            "file_url": url,
        }));
    }

    None
}

fn map_audio_url_to_file_content_item(url: &str) -> Option<Value> {
    if let Some((media_type, data)) = parse_data_url(url) {
        if !media_type.starts_with("audio/") {
            return None;
        }
        return Some(json!({
            "type": "input_file",
            "file_data": data,
        }));
    }

    if url.starts_with("http://") || url.starts_with("https://") {
        return Some(json!({
            "type": "input_file",
            "file_url": url,
        }));
    }

    None
}

fn parse_data_url(url: &str) -> Option<(&str, &str)> {
    let data = url.strip_prefix("data:")?;
    data.split_once(',')
}

pub(super) fn is_openai_model(model: &str) -> bool {
    let model = model.trim().to_ascii_lowercase();
    if model.is_empty() || model.contains('/') {
        return false;
    }

    model.starts_with("gpt-")
        || model.starts_with("chatgpt-")
        || model.starts_with("codex-")
        || model == "o1"
        || model.starts_with("o1-")
        || model == "o3"
        || model.starts_with("o3-")
        || model == "o4"
        || model.starts_with("o4-")
}

fn extract_text_from_parts(parts: &[ContentPart], sep: &str) -> String {
    let mut text_parts = Vec::new();
    for part in parts {
        if let ContentPart::Text(text) = part {
            text_parts.push(text.text.clone());
        }
    }
    text_parts.join(sep)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_openai_model_for_role_mapping() {
        assert!(is_openai_model("gpt-5-codex"));
        assert!(is_openai_model("o3"));
        assert!(is_openai_model("chatgpt-4o-latest"));
        assert!(!is_openai_model("claude-sonnet-4"));
        assert!(!is_openai_model("openai/gpt-5"));
    }
}
