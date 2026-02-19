use std::any::Any;
use std::env;

use async_trait::async_trait;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue, USER_AGENT};
use reqwest::{Client, Url};
use serde_json::{Map, Value, json};

use crate::chat_provider::{
    ChatProvider, ChatProviderError, ChatProviderErrorKind, StreamedMessage, ThinkingEffort,
};
use crate::message::Message;
use crate::tooling::Tool;

use super::request::{convert_history_message, is_openai_model};
use super::response::{map_reqwest_error, parse_non_stream_response};
use super::stream::OpenAIResponsesStreamedMessage;

#[derive(Clone)]
pub struct OpenAIResponses {
    model: String,
    api_key: String,
    base_url: Url,
    stream: bool,
    client: Client,
    generation_kwargs: Map<String, Value>,
}

impl OpenAIResponses {
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
            model: model.into(),
            api_key,
            base_url,
            stream: true,
            client,
            generation_kwargs: Map::new(),
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

    fn use_developer_role(&self) -> bool {
        is_openai_model(&self.model)
    }
}

#[async_trait]
impl ChatProvider for OpenAIResponses {
    fn name(&self) -> &str {
        "openai-responses"
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
        let mut input = Vec::new();
        let use_developer_role = self.use_developer_role();
        if !system_prompt.is_empty() {
            input.push(json!({
                "role": if use_developer_role {
                    "developer"
                } else {
                    "system"
                },
                "content": system_prompt,
            }));
        }
        for message in history {
            convert_history_message(message, &mut input, use_developer_role);
        }

        let mut tool_defs = Vec::new();
        for tool in tools {
            tool_defs.push(json!({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "strict": false,
            }));
        }

        let mut body = Map::new();
        body.insert("model".to_string(), Value::String(self.model.clone()));
        body.insert("input".to_string(), Value::Array(input));
        body.insert("tools".to_string(), Value::Array(tool_defs));
        body.insert("stream".to_string(), Value::Bool(self.stream));
        body.insert("store".to_string(), Value::Bool(false));

        let mut generation_kwargs = self.generation_kwargs.clone();
        let reasoning_effort = generation_kwargs.remove("reasoning_effort");
        generation_kwargs.remove("reasoning");

        let mut include = vec![Value::String("reasoning.encrypted_content".to_string())];
        if let Some(Value::Array(values)) = generation_kwargs.remove("include") {
            for value in values {
                if value.as_str() == Some("reasoning.encrypted_content") {
                    continue;
                }
                include.push(value);
            }
        } else {
            generation_kwargs.remove("include");
        }
        for (k, v) in generation_kwargs {
            body.insert(k, v);
        }
        body.insert(
            "reasoning".to_string(),
            json!({
                "effort": reasoning_effort.unwrap_or(Value::Null),
                "summary": "auto",
            }),
        );
        body.insert("include".to_string(), Value::Array(include));

        let url = self
            .base_url
            .join("responses")
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
                format!("OpenAI Responses API error ({status}): {text}"),
            ));
        }

        if self.stream {
            Ok(Box::new(OpenAIResponsesStreamedMessage::new_stream(resp)))
        } else {
            let value: Value = resp.json().await.map_err(map_reqwest_error)?;
            let (parts, message_id, usage) = parse_non_stream_response(&value);
            Ok(Box::new(OpenAIResponsesStreamedMessage::new_parts(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_with_thinking_supports_xhigh_effort() {
        let provider = OpenAIResponses::new("gpt-5", Some("k".to_string()), None, None)
            .expect("provider")
            .with_generation_kwargs({
                let mut kwargs = Map::new();
                kwargs.insert(
                    "reasoning_effort".to_string(),
                    Value::String("xhigh".to_string()),
                );
                kwargs
            });
        assert_eq!(provider.thinking_effort(), Some(ThinkingEffort::XHigh));

        let xhigh = provider.with_thinking(ThinkingEffort::XHigh);
        let typed = xhigh
            .as_any()
            .downcast_ref::<OpenAIResponses>()
            .expect("openai responses");
        assert_eq!(
            typed.generation_kwargs.get("reasoning_effort"),
            Some(&Value::String("xhigh".to_string()))
        );
    }
}
