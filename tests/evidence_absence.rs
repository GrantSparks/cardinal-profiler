use async_trait::async_trait;
use cardinal_profiler::error::ProfilerError;
use cardinal_profiler::openai_client::LanguageApi;

use cardinal_profiler::config::Config;
use cardinal_profiler::pipeline;
use std::fs::{read_to_string, File};
use std::io::Write;
use tempfile::tempdir;

#[derive(Clone)]
struct Stub;

#[async_trait]
impl LanguageApi for Stub {
    async fn process_prompt(&self, prompt: &str) -> Result<String, ProfilerError> {
        if prompt.starts_with("You are **CARDINAL-PROFILER L0**") {
            return Ok(r#"{"nation":"X","region":"Y"}"#.into());
        }

        // ----- layer-specific responses -----
        if prompt.contains("CARDINAL-PROFILER L1") {
            return Ok(r#"{"tags":["EUROPE"],"scores":[0.9],"evidence":["E"]}"#.into());
        }
        if prompt.contains("CARDINAL-PROFILER L2") {
            return Ok(r#"{"tags":["INST_CUR_PREF"],"scores":[0.9],"evidence":["E"]}"#.into());
        }
        if prompt.contains("CARDINAL-PROFILER L3") {
            return Ok(
                r#"{"tags":["REF_MOD"],"scores":[0.9],"evidence":["E"],"paragraph":""}"#.into(),
            );
        }
        if prompt.contains("CARDINAL-PROFILER L4") {
            return Ok(r#"{"tags":["NET_JES"],"scores":[0.9],"evidence":["E"]}"#.into());
        }
        if prompt.contains("CARDINAL-PROFILER L5") {
            return Ok(r#"{"tags":["FAC_TEAM_FRANCIS"],"scores":[0.9],"evidence":["E"]}"#.into());
        }

        Err(ProfilerError::Other("unmatched prompt".into()))
    }
}

#[tokio::test]
async fn csv_excludes_evidence_column() {
    let dir = tempdir().unwrap();
    let in_csv = dir.path().join("in.csv");
    let mut f = File::create(&in_csv).unwrap();
    writeln!(f, "Name,Office").unwrap();
    writeln!(f, "Foo,Bar").unwrap();
    let out_csv = dir.path().join("out.csv");
    let cfg = Config {
        input: in_csv,
        output: out_csv.clone(),
        ..Default::default()
    };
    pipeline::run(&cfg, Stub, cardinal_profiler::shutdown::Shutdown::new())
        .await
        .unwrap();
    let hdr = read_to_string(out_csv)
        .unwrap()
        .lines()
        .next()
        .unwrap()
        .to_string();
    assert!(
        !hdr.contains("L1_Evidence"),
        "`L1_Evidence` should not be in the output header any more"
    );
}
