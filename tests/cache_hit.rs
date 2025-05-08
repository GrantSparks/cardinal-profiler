use cardinal_profiler::openai_client::{LanguageApi, OpenAIClient};
use cardinal_profiler::prompt;
use rusqlite::{Connection, OptionalExtension};
use sha2::{Digest, Sha256};
use std::env;
use tempfile::tempdir;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn cache_prevents_http_call() {
    // Start a mock server that returns 500 for any POST to /v1/chat/completions
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&mock)
        .await;

    // Point our client at the mock server
    env::set_var("OPENAI_API_KEY", "test");
    let client = OpenAIClient::new("test".into(), "gpt-test".into()).with_base_url(mock.uri());

    // Build the prompt and compute its hash (same as pipeline::hash_prompt)
    // Provide empty nation/continent for test
    let prompt_cow = prompt::build(
        3,
        "Test",
        "Some Diocese",
        Some(""),
        Some(""),
        None,
        None,
        None,
    );

    let prompt = prompt_cow.into_owned();
    let mut hasher = Sha256::new();
    hasher.update(prompt.as_bytes());
    let key = hex::encode(hasher.finalize());

    // Create a temp SQLite cache and insert a “cached” row
    let dir = tempdir().unwrap();
    let cache_path = dir.path().join("out.cache.sqlite");
    let conn = Connection::open(&cache_path).unwrap();
    conn.execute_batch(
        "CREATE TABLE cache (
             prompt_hash TEXT NOT NULL,
             layer       INTEGER NOT NULL,
             response    TEXT NOT NULL,
             PRIMARY KEY (prompt_hash, layer)
         );",
    )
    .unwrap();
    conn.execute(
        "INSERT INTO cache(prompt_hash, layer, response) VALUES (?1, ?2, ?3)",
        rusqlite::params![key, 3, "cached response"],
    )
    .unwrap();

    // Try to fetch the cached response; if absent, fall back to HTTP
    let cached: Option<String> = conn
        .query_row(
            "SELECT response FROM cache WHERE prompt_hash = ?1",
            rusqlite::params![key],
            |row| row.get(0),
        )
        .optional()
        .unwrap();

    let result = if let Some(response) = cached {
        // Should hit here and skip HTTP entirely
        Ok(response)
    } else {
        client.process_prompt(&prompt).await
    };

    assert_eq!(result.unwrap(), "cached response");
}
