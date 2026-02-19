use crate::chat_provider::{ChatProvider, ChatProviderError, ChatProviderErrorKind, TokenUsage};
use crate::message::{Message, Role, StreamedMessagePart, ToolCall, ToolCallPart};
use crate::tooling::Tool;
use std::collections::HashMap;
use tokio::sync::mpsc;
use tracing::trace;

pub struct GenerateResult {
    pub id: Option<String>,
    pub message: Message,
    pub usage: Option<TokenUsage>,
}

pub async fn generate(
    chat_provider: &dyn ChatProvider,
    system_prompt: &str,
    tools: Vec<Tool>,
    history: &[Message],
    message_part_tx: Option<mpsc::UnboundedSender<StreamedMessagePart>>,
    on_tool_call: Option<&mut dyn FnMut(ToolCall)>,
) -> Result<GenerateResult, ChatProviderError> {
    let mut message = Message::new(Role::Assistant, Vec::new());
    let mut pending: Option<StreamedMessagePart> = None;
    let mut pending_tool_call_parts: HashMap<String, ToolCallPart> = HashMap::new();

    trace!("Generating with history: {:?}", history);
    let mut stream = chat_provider
        .generate(system_prompt, &tools, history)
        .await?;

    loop {
        let part = stream.next_part().await?;
        if part.is_none() {
            break;
        }
        let part = part.unwrap();
        trace!("Received part: {:?}", part);
        if let Some(tx) = message_part_tx.as_ref() {
            let _ = tx.send(part.clone());
        }

        if pending.is_none() {
            pending = Some(part);
            continue;
        }
        let mut current = pending.take().unwrap();
        if current.merge_in_place(&part) {
            pending = Some(current);
        } else {
            append_part(&mut message, current, &mut pending_tool_call_parts);
            pending = Some(part);
        }
    }

    if let Some(final_part) = pending {
        append_part(&mut message, final_part, &mut pending_tool_call_parts);
    }

    if let Some(cb) = on_tool_call
        && let Some(tool_calls) = message.tool_calls.clone()
    {
        for tool_call in tool_calls {
            cb(tool_call);
        }
    }

    if message.content.is_empty()
        && message
            .tool_calls
            .as_ref()
            .map(|v| v.is_empty())
            .unwrap_or(true)
    {
        return Err(ChatProviderError::new(
            ChatProviderErrorKind::EmptyResponse,
            "The API returned an empty response.",
        ));
    }

    Ok(GenerateResult {
        id: stream.id(),
        message,
        usage: stream.usage(),
    })
}

fn append_part(
    message: &mut Message,
    part: StreamedMessagePart,
    pending_tool_call_parts: &mut HashMap<String, ToolCallPart>,
) {
    match part {
        StreamedMessagePart::Content(content) => {
            message.content.push(content);
        }
        StreamedMessagePart::ToolCall(mut tool_call) => {
            if let Some(orphan_part) = pending_tool_call_parts.remove(&tool_call.id) {
                merge_orphan_tool_call_part(&mut tool_call, orphan_part);
            }
            if message.tool_calls.is_none() {
                message.tool_calls = Some(Vec::new());
            }
            if let Some(list) = &mut message.tool_calls {
                list.push(tool_call);
            }
        }
        StreamedMessagePart::ToolCallPart(tool_call_part) => {
            if merge_tool_call_part(message, &tool_call_part) {
                return;
            }
            if let Some(tool_call_id) = tool_call_part.tool_call_id.clone() {
                if let Some(existing) = pending_tool_call_parts.get_mut(&tool_call_id) {
                    let _ = existing.merge_in_place(&tool_call_part);
                } else {
                    pending_tool_call_parts.insert(tool_call_id, tool_call_part);
                }
            }
        }
    }
}

fn merge_orphan_tool_call_part(tool_call: &mut ToolCall, orphan_part: ToolCallPart) {
    if let Some(orphan_tool_call_id) = orphan_part.tool_call_id.as_deref()
        && orphan_tool_call_id != tool_call.id.as_str()
    {
        return;
    }

    let Some(orphan_arguments) = orphan_part.arguments_part else {
        return;
    };

    match tool_call.function.arguments.as_mut() {
        None => {
            tool_call.function.arguments = Some(orphan_arguments);
        }
        Some(current_arguments) => {
            if current_arguments.starts_with(&orphan_arguments) {
                return;
            }
            if orphan_arguments.starts_with(current_arguments.as_str()) {
                *current_arguments = orphan_arguments;
                return;
            }

            let mut merged_arguments = orphan_arguments;
            merged_arguments.push_str(current_arguments);
            *current_arguments = merged_arguments;
        }
    }
}

fn merge_tool_call_part(message: &mut Message, tool_call_part: &ToolCallPart) -> bool {
    let Some(tool_calls) = &mut message.tool_calls else {
        return false;
    };

    if let Some(tool_call_id) = &tool_call_part.tool_call_id {
        if let Some(call) = tool_calls
            .iter_mut()
            .rev()
            .find(|call| &call.id == tool_call_id)
        {
            return call.merge_in_place(tool_call_part);
        }
        return false;
    }

    if let Some(call) = tool_calls.last_mut() {
        return call.merge_in_place(tool_call_part);
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_orphan_tool_call_part_keeps_prefix_order() {
        let mut message = Message::new(Role::Assistant, Vec::new());
        let mut pending_tool_call_parts: HashMap<String, ToolCallPart> = HashMap::new();

        append_part(
            &mut message,
            StreamedMessagePart::ToolCallPart(ToolCallPart {
                arguments_part: Some("{\"a\":".to_string()),
                tool_call_id: Some("call_1".to_string()),
            }),
            &mut pending_tool_call_parts,
        );
        append_part(
            &mut message,
            StreamedMessagePart::ToolCall(ToolCall {
                kind: "function".to_string(),
                id: "call_1".to_string(),
                function: crate::message::ToolCallFunction {
                    name: "sum".to_string(),
                    arguments: Some("1}".to_string()),
                },
                extras: None,
            }),
            &mut pending_tool_call_parts,
        );

        let tool_calls = message.tool_calls.expect("tool_calls");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_1");
        assert_eq!(
            tool_calls[0].function.arguments.as_deref(),
            Some("{\"a\":1}")
        );
    }
}
