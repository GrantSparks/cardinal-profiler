use async_trait::async_trait;
use cardinal_profiler::error::ProfilerError;
use cardinal_profiler::openai_client::LanguageApi;
use cardinal_profiler::{config::Config, pipeline};
use std::fs::File;
use std::io::Write;
use tempfile::tempdir;

#[derive(Clone)]
struct Dummy;

#[async_trait]
impl LanguageApi for Dummy {
    async fn process_prompt(&self, _: &str) -> Result<String, ProfilerError> {
        Ok(r#"{"tags":[],"scores":[],"evidence":[]}"#.into())
    }
}

#[tokio::test]
async fn no_cache_creates_no_file() {
    let dir = tempdir().unwrap();
    let in_csv = dir.path().join("in.csv");
    let mut f = File::create(&in_csv).unwrap();
    writeln!(f, "Name,Office").unwrap();
    writeln!(f, "Foo,Bar").unwrap();
    let out_csv = dir.path().join("out.csv");
    let cfg = Config {
        input: in_csv.clone(),
        output: out_csv.clone(),
        no_cache: true,
        layer_stop: 2, // Limit profiling to layers 0-2
        ..Default::default()
    };
    pipeline::run(&cfg, Dummy, cardinal_profiler::shutdown::Shutdown::new())
        .await
        .unwrap();
    assert!(!dir.path().join("out.cache.sqlite").exists());
}
