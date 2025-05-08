use std::fs;
use std::io;
use std::path::Path;

/// Reserved profiler columns appended after user columns
pub const PROFILER_COLS: [&str; 13] = [
    "Country",
    "Region",
    // layer-1
    "L1_Tags",
    "L1_Scores",
    // layer-2
    "L2_Tags",
    "L2_Scores",
    // layer-3
    "L3_Tags",
    "L3_Scores",
    // layer-4
    "L4_Tags",
    "L4_Scores",
    // layer-5
    "L5_Tags",
    "L5_Scores",
    // summary
    "Paragraph",
];

/// Atomically rename the fully-written temporary file into place.
pub fn atomic_rename(tmp: &Path, final_path: &Path) -> io::Result<()> {
    if final_path.exists() {
        fs::remove_file(final_path)?;
    }
    fs::rename(tmp, final_path)
}
