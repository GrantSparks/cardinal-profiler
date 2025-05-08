use crate::error::ProfilerError;
use crate::taxonomy::all_tags;

/// Supported profiling layers.
#[derive(Debug, Clone, Copy)]
pub enum Layer {
    L1,
    L2,
    L3,
    L4,
    L5,
}

/// Validate and auto-repair or error according to business rules.
pub fn validate(
    layer: Layer,
    tags: &mut Vec<String>,
    scores: &mut Vec<f32>,
    evidence: &mut Vec<String>,
    paragraph: Option<&mut String>,
) -> Result<(), ProfilerError> {
    // 1) Reject any tag not in our taxonomy (keyed by &str)
    // map our validator::Layer → taxonomy::Layer
    let tax_layer = match layer {
        Layer::L1 => crate::taxonomy::Layer::L1,
        Layer::L2 => crate::taxonomy::Layer::L2,
        Layer::L3 => crate::taxonomy::Layer::L3,
        Layer::L4 => crate::taxonomy::Layer::L4,
        Layer::L5 => crate::taxonomy::Layer::L5,
    };

    for t in tags.iter() {
        let tag_id = t.as_str();
        // only accept IDs that appear in the static whitelist
        if !all_tags(tax_layer).iter().any(|&id| id == tag_id) {
            return Err(ProfilerError::Other(format!("unknown tag {}", t)));
        }
    }

    // 2) Clamp scores & align lengths
    for s in scores.iter_mut() {
        *s = (*s).clamp(0.0, 1.0);
    }
    let min_ts = tags.len().min(scores.len());
    tags.truncate(min_ts);
    scores.truncate(min_ts);
    if evidence.len() > tags.len() {
        evidence.truncate(tags.len());
    }

    // 3) Layer-specific rules
    match layer {
        Layer::L1 => {
            // No additional validation rules for L1 beyond membership check
        }
        Layer::L2 => {
            if tags.len() > 1 {
                tags.truncate(1);
                scores.truncate(1);
                evidence.truncate(1);
            }
        }
        Layer::L3 => {
            if !tags.is_empty() && tags.len() > 3 {
                tags.truncate(3);
                scores.truncate(3);
                evidence.truncate(3);
            }
            if let Some(p) = paragraph {
                let wc = p.split_whitespace().count();
                if wc < 70 {
                    for _ in 0..(70 - wc) {
                        p.push_str(" x");
                    }
                } else if wc > 100 {
                    let truncated = p.split_whitespace().take(100).collect::<Vec<_>>().join(" ");
                    *p = truncated;
                }
            }
        }
        Layer::L4 => {
            if tags.len() > 4 {
                tags.truncate(4);
                scores.truncate(4);
                evidence.truncate(4);
            }
        }
        Layer::L5 => {
            if tags.len() > 3 {
                tags.truncate(3);
                scores.truncate(3);
                evidence.truncate(3);
            }
        }
    }

    Ok(())
}
