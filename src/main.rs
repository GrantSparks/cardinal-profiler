use cardinal_profiler::{error, openai_client, pipeline, shutdown};
use clap::Parser;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // initialize structured logging
    tracing_subscriber::fmt::init();
    // Parse & validate CLI first so we can use cfg.jobs
    let args = cardinal_profiler::cli::CliArgs::parse();
    let cfg = cardinal_profiler::config::Config::from_cli(args)?;

    // Build a Tokio runtime with user-specified worker threads
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(cfg.jobs)
        .enable_all()
        .build()?;

    // -------- run --------
    let code = rt.block_on(async {
        match inner(cfg).await {
            Ok(_) => 0,
            Err(cardinal_profiler::error::ProfilerError::Other(s)) if s == "cancelled" => 130,
            Err(cardinal_profiler::error::ProfilerError::Other(s)) if s == "terminated" => 143,
            Err(e) => {
                eprintln!("{e}");
                1
            }
        }
    });

    // give all still-alive tasks up to 3 seconds to notice the cancel token
    rt.shutdown_timeout(std::time::Duration::from_secs(3));
    std::process::exit(code);
}

async fn inner(cfg: cardinal_profiler::config::Config) -> Result<(), error::ProfilerError> {
    if cfg.verbose {
        eprintln!("Config: {:?}", cfg);
    }
    let key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| error::ProfilerError::Other("OPENAI_API_KEY not set".into()))?;
    let client = openai_client::OpenAIClient::new(key, cfg.model.clone());
    let sd = shutdown::Shutdown::new();

    // SIGINT
    {
        let sd = sd.clone();
        tokio::spawn(async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                eprintln!("Interrupted – cleaning up…");
                sd.shutdown();
            }
        });
    }
    // SIGTERM (Unix)
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let sd = sd.clone();
        tokio::spawn(async move {
            if let Ok(mut term) = signal(SignalKind::terminate()) {
                term.recv().await;
                eprintln!("Terminated – cleaning up…");
                sd.shutdown();
            }
        });
    }
    pipeline::run(&cfg, client, sd).await
}
