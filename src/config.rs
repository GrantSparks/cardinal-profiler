use crate::cli::CliArgs;
use crate::error::ProfilerError;
use crate::system_info::{heuristic_api_parallel, heuristic_parallel, parse_auto_or_number};
use std::fs;
use std::io;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Config {
    pub input: PathBuf,
    pub output: PathBuf,
    pub model: String,
    pub parallel: usize,
    pub delimiter: Option<u8>,
    pub jobs: usize,
    pub verbose: bool,
    pub no_progress: bool,
    pub no_emoji: bool,
    pub no_cache: bool,
    pub refresh_cache: bool,
    /// Minimum layer to process (0–5; L0 always runs)
    pub layer_start: usize,
    /// Maximum layer to process (0–5)
    pub layer_stop: usize,
    /// Concurrent OpenAI calls
    pub api_parallel: usize,
    /// Drop tags with mean score below this threshold (0.0–1.0)
    pub min_score: f32,
    /// Write evidence JSONL sidecar
    pub audit: bool,
}

impl Config {
    pub fn from_cli(args: CliArgs) -> Result<Self, ProfilerError> {
        // Validate layer bounds (0–5)
        if args.layer_start > 5 {
            return Err(ProfilerError::Other(
                "--layer-start must be between 0 and 5".into(),
            ));
        }
        if args.layer_stop < args.layer_start || args.layer_stop > 5 {
            return Err(ProfilerError::Other(
                "--layer-stop must be between --layer-start and 5".into(),
            ));
        }
        if args.min_score < 0.0 || args.min_score > 1.0 {
            return Err(ProfilerError::Other(
                "--min-score must be between 0.0 and 1.0".into(),
            ));
        }
        if !args.input.exists() {
            return Err(io::Error::new(io::ErrorKind::NotFound, "input file missing").into());
        }
        let output = args.output.unwrap_or_else(|| {
            let mut p = args.input.clone();
            let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
            p.set_file_name(format!("{stem}_profiled.csv"));
            p
        });
        if let Some(parent) = output.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                fs::create_dir_all(parent)?;
            }
        }
        // Parse parallel parameter – accepts "auto" or numeric literal
        let parallel = parse_auto_or_number(&args.parallel, heuristic_parallel)
            .map_err(ProfilerError::Other)?;

        // Parse api_parallel parameter – same semantics as above
        let api_parallel =
            parse_auto_or_number(&args.api_parallel, || heuristic_api_parallel(parallel))
                .map_err(ProfilerError::Other)?;

        Ok(Self {
            input: args.input,
            output,
            model: args.model,
            parallel,
            delimiter: args.delimiter.map(|c| c as u8),
            jobs: args.jobs,
            verbose: args.verbose,
            no_progress: args.no_progress,
            no_emoji: args.no_emoji,
            no_cache: args.no_cache,
            refresh_cache: args.refresh_cache,
            layer_start: args.layer_start,
            layer_stop: args.layer_stop,
            api_parallel,
            min_score: args.min_score,
            audit: args.audit,
        })
    }
}

// Convenient default Config for tests / examples
impl Default for Config {
    fn default() -> Self {
        Config {
            input: PathBuf::new(),
            output: PathBuf::new(),
            model: "o3".to_string(),
            parallel: 1,
            delimiter: None,
            jobs: 1,
            verbose: false,
            no_progress: false,
            no_emoji: false,
            no_cache: false,
            refresh_cache: false,
            layer_start: 0,
            layer_stop: 5,
            api_parallel: 1,
            min_score: 0.0,
            audit: false,
        }
    }
}
