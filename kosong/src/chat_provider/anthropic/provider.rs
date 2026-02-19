use std::any::Any;
use std::env;

use async_trait::async_trait;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue, USER_AGENT};
use reqwest::{Client, Url};
use serde_json::{Map, Value, json};

use crate::chat_provider::{
    ChatProvider, ChatProviderError, ChatProviderErrorKind, StreamedMessage, ThinkingEffort,
};
use crate::message::Message;
use crate::tooling::Tool;

use super::request::{
    apply_extra_headers, convert_message, convert_tool, extract_beta_header, map_reqwest_error,
    mark_last_cacheable_message_block, mark_last_tool_definition,
};
use super::response::parse_non_stream_response;
use super::stream::AnthropicStreamedMessage;

const INTERLEAVED_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";
const DEFAULT_MAX_TOKENS: i64 = 50_000;

#[derive(Clone)]
pub struct Anthropic {
    model: String,
    api_key: String,
    base_url: Url,
    stream: bool,
    client: Client,
    generation_kwargs: Map<String, Value>,
}

impl Anthropic {
    pub fn new(
        model: impl Into<String>,
        api_key: Option<String>,
        base_url: Option<String>,
        default_headers: Option<HeaderMap>,
    ) -> Result<Self, ChatProviderError> {
        let api_key = api_key
            .filter(|value| !value.is_empty())
            .or_else(|| env::var("ANTHROPIC_API_KEY").ok())
            .unwrap_or_default();
        let mut base_url = base_url
            .filter(|value| !value.is_empty())
            .or_else(|| env::var("ANTHROPIC_BASE_URL").ok())
            .unwrap_or_else(|| "https://api.anthropic.com".to_string());
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
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
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
            model: model.into(),
            api_key,
            base_url,
            stream: true,
            client,
            generation_kwargs: {
                let mut kwargs = Map::new();
                kwargs.insert("max_tokens".to_string(), Value::from(DEFAULT_MAX_TOKENS));
                kwargs.insert(
                    "beta_features".to_string(),
                    Value::Array(vec![Value::String(INTERLEAVED_THINKING_BETA.to_string())]),
                );
                kwargs
            },
        })
    }

    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = stream;
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
impl ChatProvider for Anthropic {
    fn name(&self) -> &str {
        "anthropic"
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn thinking_effort(&self) -> Option<ThinkingEffort> {
        let thinking = self.generation_kwargs.get("thinking")?;
        let kind = thinking.get("type").and_then(|v| v.as_str())?;
        if kind == "disabled" {
            return Some(ThinkingEffort::Off);
        }
        let budget = thinking
            .get("budget_tokens")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        if budget <= 1024 {
            Some(ThinkingEffort::Low)
        } else if budget <= 4096 {
            Some(ThinkingEffort::Medium)
        } else if budget <= 32000 {
            Some(ThinkingEffort::High)
        } else {
            Some(ThinkingEffort::XHigh)
        }
    }

    async fn generate(
        &self,
        system_prompt: &str,
        tools: &[Tool],
        history: &[Message],
    ) -> Result<Box<dyn StreamedMessage>, ChatProviderError> {
        let mut messages = Vec::new();
        for message in history {
            messages.push(convert_message(message)?);
        }
        mark_last_cacheable_message_block(&mut messages);

        let mut tool_defs = Vec::new();
        for tool in tools {
            tool_defs.push(convert_tool(tool));
        }
        mark_last_tool_definition(&mut tool_defs);

        let mut body = Map::new();
        body.insert("model".to_string(), Value::String(self.model.clone()));
        body.insert("messages".to_string(), Value::Array(messages));
        body.insert("tools".to_string(), Value::Array(tool_defs));
        body.insert("stream".to_string(), Value::Bool(self.stream));

        if !system_prompt.is_empty() {
            body.insert(
                "system".to_string(),
                json!([
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {
                            "type": "ephemeral",
                        }
                    }
                ]),
            );
        }

        let mut generation_kwargs = self.generation_kwargs.clone();
        let beta_header = extract_beta_header(generation_kwargs.remove("beta_features"));
        let extra_headers = generation_kwargs.remove("extra_headers");
        for (k, v) in generation_kwargs {
            body.insert(k, v);
        }

        let url = self
            .base_url
            .join("v1/messages")
            .map_err(|err| ChatProviderError::new(ChatProviderErrorKind::Other, err.to_string()))?;

        let mut request = self.client.post(url);
        if !self.api_key.is_empty() {
            request = request.header("x-api-key", self.api_key.clone());
        }
        if let Some(beta_header) = beta_header {
            request = request.header("anthropic-beta", beta_header);
        }
        request = apply_extra_headers(request, extra_headers);

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
                format!("Anthropic API error ({status}): {text}"),
            ));
        }

        if self.stream {
            Ok(Box::new(AnthropicStreamedMessage::new_stream(resp)))
        } else {
            let value: Value = resp.json().await.map_err(map_reqwest_error)?;
            let (parts, message_id, usage) = parse_non_stream_response(&value)?;
            Ok(Box::new(AnthropicStreamedMessage::new_parts(
                parts, message_id, usage,
            )))
        }
    }

    fn with_thinking(&self, effort: ThinkingEffort) -> Box<dyn ChatProvider> {
        let thinking = match effort {
            ThinkingEffort::Off => json!({"type": "disabled"}),
            ThinkingEffort::Low => json!({"type": "enabled", "budget_tokens": 1024}),
            ThinkingEffort::Medium => json!({"type": "enabled", "budget_tokens": 4096}),
            ThinkingEffort::High => json!({"type": "enabled", "budget_tokens": 32000}),
            ThinkingEffort::XHigh => json!({"type": "enabled", "budget_tokens": 64000}),
        };

        let mut kwargs = Map::new();
        kwargs.insert("thinking".to_string(), thinking);

        let provider = self.clone().with_generation_kwargs(kwargs);
        Box::new(provider)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_generation_kwargs_include_interleaved_thinking_beta() {
        let provider = Anthropic::new("claude-sonnet-4", None, None, None).expect("provider");
        let params = provider.model_parameters();
        assert_eq!(
            params.get("beta_features"),
            Some(&json!(["interleaved-thinking-2025-05-14"]))
        );
    }
}
