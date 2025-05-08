use async_trait::async_trait;
use cardinal_profiler::error::ProfilerError;
use cardinal_profiler::openai_client::LanguageApi;
use cardinal_profiler::{config::Config, pipeline};
use std::fs::{read_to_string, File};
use std::io::Write;
use tempfile::tempdir;

#[derive(Clone)]
struct RegionStub;

// A stub LLM that echoes the CSV assignment back as the "region".
#[async_trait]
impl LanguageApi for RegionStub {
    async fn process_prompt(&self, prompt: &str) -> Result<String, ProfilerError> {
        if prompt.starts_with("You are **CARDINAL-PROFILER L0**") {
            // Extract the assignment line
            let region = prompt
                .lines()
                .find_map(|l| l.strip_prefix("Assignment: "))
                .unwrap_or("")
                .trim();
            // Return it as the region
            let json = format!(r#"{{"nation":"X","region":"{}"}}"#, region);
            return Ok(json);
        }
        // We only care about layer 0 for this smoke test:
        Err(ProfilerError::Other("unexpected layer".into()))
    }
}

#[tokio::test]
async fn smoke_test_all_regions() {
    // The canonical 12-region IDs (must match assets/taxonomy.toml).
    let regions = [
        "AFRICA",
        "EUROPE",
        "USA_CANADA",
        "CENTRAL_AMERICA",
        "SOUTH_AMERICA",
        "CARIBBEAN",
        "MENA",
        "SOUTH_ASIA",
        "SOUTH_EAST_ASIA",
        "EAST_ASIA",
        "OCEANIA",
    ];

    // Build a temporary CSV with one row per region
    let dir = tempdir().unwrap();
    let in_path = dir.path().join("regions.csv");
    {
        let mut f = File::create(&in_path).unwrap();
        writeln!(f, "Name,Office").unwrap();
        for (i, reg) in regions.iter().enumerate() {
            writeln!(f, "Name{}{},{}", i, reg, reg).unwrap();
        }
    }

    // Run only L0 (populate Nation+Region)
    let out_path = dir.path().join("out.csv");
    let cfg = Config {
        input: in_path.clone(),
        output: out_path.clone(),
        layer_start: 0,
        layer_stop: 0,
        ..Default::default()
    };

    pipeline::run(
        &cfg,
        RegionStub,
        cardinal_profiler::shutdown::Shutdown::new(),
    )
    .await
    .unwrap();

    // Read the output and make sure each region string appears in the Region column
    let csv = read_to_string(out_path).unwrap();
    for reg in &regions {
        // each region should appear at least once
        assert!(
            csv.contains(&format!(",{}", reg)),
            "expected Region column to contain `{}`",
            reg
        );
    }
}
