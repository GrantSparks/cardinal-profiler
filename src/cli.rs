use clap::Parser;
use std::path::PathBuf;

/// Command‑line flags parsed with **clap**.
#[derive(Parser, Debug)]
#[command(
    name = "cardinal-profiler",
    version,
    about = "Enrich a CSV of cardinals with theological profiles",
    // Long help trimmed; `--help` shows full details in README.
)]
pub struct CliArgs {
    #[arg(value_name = "INPUT")]
    pub input: PathBuf,
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    #[arg(long, default_value = "o3")]
    pub model: String,
    /// End‑to‑end concurrency (rows in flight) – "auto" uses system heuristics
    #[arg(long, default_value = "auto")]
    pub parallel: String,
    #[arg(long)]
    pub delimiter: Option<char>,
    #[arg(long, default_value_t = num_cpus::get())]
    pub jobs: usize,
    #[arg(long)]
    pub verbose: bool,
    /// Disable progress bars (always off in non‑TTY)
    #[arg(long)]
    pub no_progress: bool,
    #[arg(long)]
    pub no_emoji: bool,
    #[arg(long)]
    pub no_cache: bool,
    #[arg(long)]
    pub refresh_cache: bool,
    /// Minimum layer to process (0–5; 0 = only L0)
    #[arg(long, default_value_t = 0)]
    pub layer_start: usize,
    /// Maximum layer to process (0–5; must be ≥ layer-start)
    #[arg(long, default_value_t = 5)]
    pub layer_stop: usize,
    /// Concurrent OpenAI calls, or "auto" to determine based on system specs
    #[arg(long, default_value = "auto")]
    pub api_parallel: String,
    /// Drop tags with mean score below this threshold (0.0–1.0)
    #[arg(long, default_value_t = 0.0)]
    pub min_score: f32,
    /// Write evidence JSONL sidecar
    #[arg(long)]
    pub audit: bool,
}
