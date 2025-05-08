use crate::error::ProfilerError;
use csv::{ReaderBuilder, WriterBuilder};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug)]
/// In‑memory representation of the CSV with metadata.
pub struct RawCsv {
    pub headers: Vec<String>,
    pub idx_name: usize,
    pub idx_assign: usize,
    pub rows: Vec<RawRecord>,
}

#[derive(Debug)]
/// A single row of user data, allowing pass‑through and optional L0 fields.
pub struct RawRecord {
    pub cols: Vec<String>,

    // Optional user‑supplied structured columns
    pub country: Option<String>,    // new
    pub born: Option<String>,       // new (string – leave age maths to LLM)
    pub order: Option<String>,      // new (CB / CP / CD)
    pub consistory: Option<String>, // new (free‑text)

    // L0 return values (back‑compat)
    pub region: Option<String>,
}

impl std::ops::Deref for RawRecord {
    type Target = Vec<String>;
    fn deref(&self) -> &Vec<String> {
        &self.cols
    }
}

impl Clone for RawRecord {
    fn clone(&self) -> Self {
        RawRecord {
            cols: self.cols.clone(),
            country: self.country.clone(),
            born: self.born.clone(),
            order: self.order.clone(),
            consistory: self.consistory.clone(),
            region: self.region.clone(),
        }
    }
}

/// CSV reader – sniff delimiter, validate headers, buffer rows.
pub struct CsvReader {
    path: String,
    delimiter: Option<u8>,
}

impl CsvReader {
    /// Create a new reader for the given file path and optional delimiter override.
    pub fn new(path: String, delimiter: Option<u8>) -> Self {
        CsvReader { path, delimiter }
    }

    /// Read the CSV, enforce required headers, detect collisions, and collect rows.
    pub fn read(&self) -> Result<RawCsv, ProfilerError> {
        let path = Path::new(&self.path);
        let (mut rdr, _delim) = reader(path, self.delimiter)?;

        // Extract and clone headers
        let hdrs = rdr.headers()?.clone();
        let headers: Vec<String> = hdrs.iter().map(|s| s.to_string()).collect();

        // Locate mandatory columns --------------------------------------
        let idx_name = headers
            .iter()
            .position(|h| h == "Name")
            .ok_or_else(|| ProfilerError::MissingHeader("Name".into()))?;

        let idx_office = headers
            .iter()
            .position(|h| h == "Office")
            .ok_or_else(|| ProfilerError::MissingHeader("Office".into()))?;

        // Optional structured columns -----------------------------------
        let idx_country = headers.iter().position(|h| h == "Country");
        let idx_born = headers.iter().position(|h| h == "Born");
        let idx_order = headers.iter().position(|h| h == "Order");
        let idx_consistory = headers.iter().position(|h| h == "Consistory");

        // Buffer all records --------------------------------------------
        let mut rows = Vec::new();
        for result in rdr.records() {
            let record = result?;
            let cols = record.iter().map(|s| s.to_string()).collect();
            rows.push(RawRecord {
                region: None,

                country: idx_country
                    .map(|i| record.get(i).unwrap_or("").to_string())
                    .filter(|s| !s.is_empty()),
                born: idx_born
                    .map(|i| record.get(i).unwrap_or("").to_string())
                    .filter(|s| !s.is_empty()),
                order: idx_order
                    .map(|i| record.get(i).unwrap_or("").to_string())
                    .filter(|s| !s.is_empty()),
                consistory: idx_consistory
                    .map(|i| record.get(i).unwrap_or("").to_string())
                    .filter(|s| !s.is_empty()),

                cols,
            });
        }

        Ok(RawCsv {
            headers,
            idx_name,
            idx_assign: idx_office,
            rows,
        })
    }
}

/// Sniff or apply the delimiter, then return a CSV reader plus the delimiter used.
pub fn reader(
    path: &Path,
    delimiter: Option<u8>,
) -> Result<(csv::Reader<File>, u8), ProfilerError> {
    // Read one line to sniff
    let file1 = File::open(path)?;
    let mut buf = BufReader::new(file1);
    let mut first = String::new();
    buf.read_line(&mut first)?;

    let delim = if let Some(d) = delimiter {
        d
    } else {
        let comma = first.matches(',').count();
        let semi = first.matches(';').count();
        let tab = first.matches('\t').count();
        if semi > comma && semi > tab {
            b';'
        } else if tab > comma && tab > semi {
            b'\t'
        } else {
            b','
        }
    };

    // Re-open for full parsing
    let file2 = File::open(path)?;
    let rdr = ReaderBuilder::new().delimiter(delim).from_reader(file2);
    Ok((rdr, delim))
}

/// Writer factory matching the sniffed/overridden delimiter.
pub fn writer(path: &Path, delimiter: u8) -> Result<csv::Writer<File>, ProfilerError> {
    let file = File::create(path)?;
    let wtr = WriterBuilder::new().delimiter(delimiter).from_writer(file);
    Ok(wtr)
}
