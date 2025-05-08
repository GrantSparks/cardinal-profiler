use async_trait::async_trait;
use cardinal_profiler::error::ProfilerError;
use cardinal_profiler::openai_client::LanguageApi;
use cardinal_profiler::{config::Config, pipeline};
use std::fs::{read_to_string, File};
use std::io::Write;
use tempfile::tempdir;

#[derive(Clone)]
struct DummyClient;

#[async_trait]
impl LanguageApi for DummyClient {
    async fn process_prompt(&self, prompt: &str) -> Result<String, ProfilerError> {
        if prompt.starts_with("You are **CARDINAL-PROFILER L0**") {
            Ok(r#"{"nation":"N","region":"C"}"#.into())
        } else if prompt.contains("CARDINAL-PROFILER L2") {
            Ok(r#"{"tags":["INST_CUR_PREF"],"scores":[1.0],"evidence":["e"]}"#.into())
        } else if prompt.contains("L3") {
            Ok(r#"{"tags":[],"scores":[],"evidence":[],"paragraph":""}"#.into())
        } else {
            Ok(r#"{"tags":["EU_WEST_NORDIC"],"scores":[1.0],"evidence":["e"]}"#.into())
        }
    }
}

#[tokio::test]
async fn audit_log_line_count() {
    let dir = tempdir().unwrap();
    // Create a minimal v2-format input CSV
    let input = dir.path().join("in.csv");
    let mut f = File::create(&input).unwrap();
    writeln!(f, "Name,Office").unwrap();
    writeln!(f, "Test,Diocese").unwrap();

    let output = dir.path().join("out.csv");
    let cfg = Config {
        input: input.clone(),
        output: output.clone(),
        model: "dummy".into(),
        parallel: 1,
        delimiter: None,
        jobs: 1,
        verbose: false,
        no_progress: false,
        no_emoji: true,
        no_cache: true,
        refresh_cache: false,
        layer_start: 0,
        layer_stop: 2,
        api_parallel: 1,
        min_score: 0.0,
        audit: true,
    };

    pipeline::run(
        &cfg,
        DummyClient,
        cardinal_profiler::shutdown::Shutdown::new(),
    )
    .await
    .unwrap();

    // Read and check audit JSONL
    let mut audit = output.clone();
    let stem = audit.file_stem().unwrap().to_string_lossy();
    audit.set_file_name(format!("{stem}_audit.jsonl"));
    let contents = read_to_string(&audit).unwrap();
    let lines: Vec<_> = contents.lines().collect();
    assert_eq!(lines.len(), 2); // layers 1 and 2 only

    for line in lines {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        assert!(v.get("row").is_some());
        assert!(v.get("layer").is_some());
        assert!(v.get("tags").is_some());
        assert!(v.get("scores").is_some());
        assert!(v.get("evidence").is_some());
    }
}
