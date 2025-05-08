use crate::error::ProfilerError;
use serde_json::Value;

// ── Type aliases to reduce signature noise ────────────────────────────────
/// Result type for L3 parsing: (tags, scores, evidence, paragraph)
type L3ParseResult = Result<(Vec<String>, Vec<f32>, Vec<String>, String), ProfilerError>;

/// Result type for L4/L5 parsing: (tags, scores, evidence)
type L4ParseResult = Result<(Vec<String>, Vec<f32>, Vec<String>), ProfilerError>;

/// Format a float with up to three decimals, trimming trailing zeros.
fn fmt_score(f: f32) -> String {
    let mut s = format!("{:.3}", f);
    if s.contains('.') {
        while s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
    }
    if s.is_empty() {
        s = "0".into();
    }
    s
}

/// Parse L3 JSON: { tags: [str], scores: [num], evidence: [str], paragraph: str }
pub fn parse_l3(v: &Value) -> L3ParseResult {
    let arr = |key: &str| -> Result<&Vec<Value>, ProfilerError> {
        v.get(key)
            .and_then(Value::as_array)
            .ok_or_else(|| ProfilerError::Other(format!("Missing or invalid '{}'", key)))
    };

    let tags: Vec<String> = arr("tags")?
        .iter()
        .map(|x| {
            x.as_str()
                .map(|s| s.to_string())
                .ok_or_else(|| ProfilerError::Other("Non-string in tags".into()))
        })
        .collect::<Result<_, _>>()?;

    let scores: Vec<f32> = arr("scores")?
        .iter()
        .map(|x| {
            x.as_f64()
                .map(|n| n as f32)
                .ok_or_else(|| ProfilerError::Other("Non-number in scores".into()))
        })
        .collect::<Result<_, _>>()?;

    if tags.len() != scores.len() {
        return Err(ProfilerError::Other("tags/scores length mismatch".into()));
    }

    let evidence: Vec<String> = arr("evidence")?
        .iter()
        .map(|x| {
            x.as_str()
                .map(|s| s.to_string())
                .ok_or_else(|| ProfilerError::Other("Non-string in evidence".into()))
        })
        .collect::<Result<_, _>>()?;

    // Extract and enforce 100-word max
    let paragraph = v
        .get("paragraph")
        .and_then(Value::as_str)
        .map(|s| s.to_string())
        .ok_or_else(|| ProfilerError::Other("Missing 'paragraph'".into()))?;

    Ok((tags, scores, evidence, paragraph))
}

/// Parse L4 JSON: { tags: [str], scores: [num], evidence?: [str] }
pub fn parse_l4(v: &Value) -> L4ParseResult {
    let arr = |key: &str| -> Result<&Vec<Value>, ProfilerError> {
        v.get(key)
            .and_then(Value::as_array)
            .ok_or_else(|| ProfilerError::Other(format!("Missing or invalid '{}'", key)))
    };

    let tags: Vec<String> = arr("tags")?
        .iter()
        .map(|x| {
            x.as_str()
                .map(|s| s.to_string())
                .ok_or_else(|| ProfilerError::Other("Non-string in tags".into()))
        })
        .collect::<Result<_, _>>()?;

    let scores: Vec<f32> = arr("scores")?
        .iter()
        .map(|x| {
            x.as_f64()
                .map(|n| n as f32)
                .ok_or_else(|| ProfilerError::Other("Non-number in scores".into()))
        })
        .collect::<Result<_, _>>()?;

    let evidence: Vec<String> = v
        .get("evidence")
        .and_then(Value::as_array)
        .map(|ev| {
            ev.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();

    Ok((tags, scores, evidence))
}

// L5 has the same shape as L4
pub use parse_l4 as parse_l5;

/// Helpers to join for CSV
pub fn join_tags(tags: &[String]) -> String {
    tags.join(";")
}

pub fn join_scores(scores: &[f32]) -> String {
    scores
        .iter()
        .map(|&f| fmt_score(f))
        .collect::<Vec<_>>()
        .join(";")
}

/// Join evidence items for CSV
pub fn join_evidence(evidence: &[String]) -> String {
    evidence.join(";")
}
