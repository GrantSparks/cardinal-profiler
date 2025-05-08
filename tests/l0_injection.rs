use async_trait::async_trait;
use cardinal_profiler::error::ProfilerError;
use cardinal_profiler::openai_client::LanguageApi;
use cardinal_profiler::{config::Config, pipeline};
use std::fs::{read_to_string, File};
use std::io::Write;
use tempfile::tempdir;

#[derive(Clone)]
struct StubClient;

#[async_trait]
impl LanguageApi for StubClient {
    async fn process_prompt(&self, prompt: &str) -> Result<String, ProfilerError> {
        if prompt.starts_with("You are **CARDINAL-PROFILER L0**") {
            Ok(r#"{"country":"Italy","region":"Europe"}"#.into())
        } else {
            Err(ProfilerError::Other("unexpected layer".into()))
        }
    }
}

#[tokio::test]
async fn l0_injection_writes_country_region() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("in.csv");
    let mut f = File::create(&input).unwrap();
    writeln!(f, "Name,Office").unwrap();
    writeln!(f, "Test,Bar").unwrap();
    let output = dir.path().join("out.csv");

    // Configure minimal valid paths and layers
    let cfg = Config {
        input: input.clone(),
        output: output.clone(),
        layer_start: 0,
        layer_stop: 0,
        ..Default::default()
    };
    let sd = cardinal_profiler::shutdown::Shutdown::new();

    pipeline::run(&cfg, StubClient, sd).await.unwrap();

    let csv = read_to_string(output).unwrap();
    assert!(csv.contains("Italy"));
    assert!(csv.contains("Europe"));
}
