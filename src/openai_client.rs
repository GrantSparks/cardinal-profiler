use crate::error::ProfilerError;
use async_trait::async_trait;
use rand::Rng;
use reqwest::{Client, StatusCode};
use serde::Deserialize;
use serde_json;
use serde_json::json;
use std::time::Duration;
use tokio::time::sleep;

#[async_trait]
pub trait LanguageApi: Send + Sync + Clone {
    async fn process_prompt(&self, prompt: &str) -> Result<String, ProfilerError>;

    /// JSON-mode / function-calling helper.
    /// By default we just fall back to the plain prompt call so test-stubs
    /// don’t have to implement it.
    async fn process_prompt_fc(
        &self,
        prompt: &str,
        _func_name: &str,
        _schema: &serde_json::Value,
    ) -> Result<String, ProfilerError> {
        self.process_prompt(prompt).await
    }
}

#[derive(Clone)]
pub struct OpenAIClient {
    http: Client,
    api_key: String,
    model: String,
    base_url: String,
}

impl OpenAIClient {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            http: Client::builder()
                .timeout(Duration::from_secs(45))
                .build()
                .unwrap(),
            api_key,
            model,
            base_url: "https://api.openai.com".into(),
        }
    }
    pub fn with_base_url(mut self, url: String) -> Self {
        self.base_url = url;
        self
    }
}

#[derive(Deserialize)]
struct Choice {
    message: Message,
}

#[derive(Deserialize)]
struct Message {
    /// Content can be null when function_call is returned
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    function_call: Option<FunctionCall>,
}

#[derive(Deserialize)]
struct FunctionCall {
    arguments: String,
}

#[derive(Deserialize)]
struct ApiResp {
    choices: Vec<Choice>,
}

const MAX_RETRIES: usize = 2; // fewer retries – JSON‑mode is deterministic

#[async_trait]
impl LanguageApi for OpenAIClient {
    async fn process_prompt(&self, prompt: &str) -> Result<String, ProfilerError> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = json!({ "model": self.model, "messages": [{"role":"user","content": prompt }] });
        let mut last_err = None;
        for attempt in 1..=MAX_RETRIES {
            let resp = self
                .http
                .post(&url)
                .bearer_auth(&self.api_key)
                .json(&body)
                .send()
                .await;
            match resp {
                Ok(r) if r.status() == StatusCode::OK => {
                    let parsed: ApiResp = r.json().await?;
                    if let Some(choice) = parsed.choices.into_iter().next() {
                        return Ok(choice.message.content.unwrap_or_default());
                    } else {
                        return Err(ProfilerError::OpenAI("empty choices array".into()));
                    }
                }
                Ok(r) => {
                    // take status *before* consuming r
                    let status = r.status();
                    let txt = r.text().await.unwrap_or_default();
                    last_err = Some(ProfilerError::OpenAI(format!("HTTP {} – {}", status, txt)));
                }
                Err(e) => last_err = Some(ProfilerError::Http(e)),
            }
            if attempt < MAX_RETRIES {
                let base = 1u64 << attempt;
                let jitter = rand::thread_rng().gen_range(0..base);
                sleep(Duration::from_secs(base + jitter)).await;
            }
        }
        Err(last_err.unwrap_or_else(|| ProfilerError::Other("unclassified error".into())))
    }

    async fn process_prompt_fc(
        &self,
        prompt: &str,
        func_name: &str,
        schema: &serde_json::Value,
    ) -> Result<String, ProfilerError> {
        let url = format!("{}/v1/chat/completions", self.base_url);

        // Remove `response_format`—it isn't supported by the API
        let body = json!({
            "model": self.model,
            "messages": [{"role":"user","content": prompt }],
            "functions": [{"name":func_name, "parameters": schema}],
            "function_call": {"name": func_name}
        });

        let mut last_err = None;
        for attempt in 1..=MAX_RETRIES {
            let resp = self
                .http
                .post(&url)
                .bearer_auth(&self.api_key)
                .json(&body)
                .send()
                .await;
            match resp {
                Ok(r) if r.status() == StatusCode::OK => {
                    // 1) Read the full text so you can see exactly what came back
                    let txt = r.text().await.unwrap_or_else(|e| {
                        eprintln!("failed to read OpenAI body: {}", e);
                        String::new()
                    });
                    // 3) Parse from the raw text, with richer error on failure
                    let parsed: ApiResp = serde_json::from_str(&txt).map_err(|e| {
                        ProfilerError::Other(format!("response not valid JSON ({}):\n{}", e, txt))
                    })?;
                    if let Some(choice) = parsed.choices.into_iter().next() {
                        // prefer function‑call because json_mode returns it there
                        if let Some(fc) = choice.message.function_call {
                            return Ok(fc.arguments);
                        }
                        return Ok(choice.message.content.unwrap_or_default());
                    } else {
                        return Err(ProfilerError::OpenAI("empty choices array".into()));
                    }
                }
                Ok(r) => {
                    // take status *before* consuming r
                    let status = r.status();
                    let txt = r.text().await.unwrap_or_default();
                    last_err = Some(ProfilerError::OpenAI(format!("HTTP {} – {}", status, txt)));
                }
                Err(e) => last_err = Some(ProfilerError::Http(e)),
            }
            if attempt < MAX_RETRIES {
                let base = 1u64 << attempt;
                let jitter = rand::thread_rng().gen_range(0..base);
                sleep(Duration::from_secs(base + jitter)).await;
            }
        }
        Err(last_err.unwrap_or_else(|| ProfilerError::Other("unclassified error".into())))
    }
}
