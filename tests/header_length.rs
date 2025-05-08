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
        // ---------- L0 ----------
        if prompt.starts_with("You are **CARDINAL-PROFILER L0**") {
            return Ok(r#"{"nation":"N","region":"C"}"#.into());
        }

        // ---------- L1 ----------
        if prompt.contains("CARDINAL-PROFILER L1") {
            return Ok(r#"{"tags":["EUROPE"],"scores":[1.0],"evidence":["e"]}"#.into());
        }

        // ---------- L2 ----------
        if prompt.contains("CARDINAL-PROFILER L2") {
            return Ok(r#"{"tags":["INST_CUR_PREF"],"scores":[1.0],"evidence":["e"]}"#.into());
        }

        // ---------- L3 ----------
        if prompt.contains("CARDINAL-PROFILER L3") {
            return Ok(
                r#"{"tags":["REF_MOD"],"scores":[0.9],"evidence":["E"],"paragraph":""}"#.into(),
            );
        }

        // ---------- L4 ----------
        if prompt.contains("CARDINAL-PROFILER L4") {
            return Ok(r#"{"tags":["NET_JES"],"scores":[1.0],"evidence":["e"]}"#.into());
        }

        // ---------- L5 ----------
        if prompt.contains("CARDINAL-PROFILER L5") {
            return Ok(r#"{"tags":["FAC_TEAM_FRANCIS"],"scores":[1.0],"evidence":["e"]}"#.into());
        }

        Err(ProfilerError::Other("unmatched prompt".into()))
    }
}

#[tokio::test]
async fn header_length_includes_18_profiler_cols() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("in.csv");
    let mut f = File::create(&input).unwrap();
    writeln!(f, "Name,Office,Extra").unwrap();
    writeln!(f, "Foo,Bar,Baz").unwrap();

    let output = dir.path().join("out.csv");
    let cfg = Config {
        input: input.clone(),
        output: output.clone(),
        model: "x".into(),
        parallel: 1,
        delimiter: None,
        jobs: 1,
        verbose: false,
        no_progress: false,
        no_emoji: true,
        no_cache: true,
        refresh_cache: false,
        layer_start: 0,
        layer_stop: 5,
        api_parallel: 1,
        min_score: 0.0,
        audit: false,
    };

    pipeline::run(
        &cfg,
        DummyClient,
        cardinal_profiler::shutdown::Shutdown::new(),
    )
    .await
    .unwrap();

    let hdr = read_to_string(&output)
        .unwrap()
        .lines()
        .next()
        .unwrap()
        .to_string();
    let cols: Vec<_> = hdr.split(',').collect();
    // user had 3 columns, plus profiler columns
    assert_eq!(
        cols.len(),
        3 + cardinal_profiler::output::PROFILER_COLS.len()
    );
    assert_eq!(cols[3], "Country");
    // 3 user cols + 2 (L0) + 2 (L1) + 2 (L2)  -> "L3_Tags" now at index 9
    assert_eq!(cols[9], "L3_Tags");
    // immediately following comes Scores
    assert_eq!(cols[10], "L3_Scores");
}
