use std::io;
use thiserror::Error;

/// Application-wide error enum.
#[derive(Error, Debug)]
pub enum ProfilerError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    #[error("CSV parsing error: {0}")]
    Csv(#[from] csv::Error),

    /// Missing one of the required CSV headers (e.g. "Name")
    #[error("Missing required CSV header: {0}")]
    MissingHeader(String),

    #[error("input header '{0}' conflicts with reserved profiler column")]
    HeaderClash(String),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("OpenAI API error: {0}")]
    OpenAI(String),

    #[error("Other error: {0}")]
    Other(String),
}

// Allow `?` on r2d2 pool operations without verbose `map_err` chains
impl From<r2d2::Error> for ProfilerError {
    fn from(e: r2d2::Error) -> Self {
        ProfilerError::Other(format!("db-pool error: {e}"))
    }
}
