use once_cell::sync::Lazy;
use std::borrow::Cow;

use crate::taxonomy::{tagdefs, Layer};

// ---------------------------------------------------------------------------
// Static templates for each layer ------------------------------------------
static TPL_L0: Lazy<String> = Lazy::new(|| include_str!("../assets/prompt_L0.txt").to_string());
static TPL_L1: Lazy<String> =
    Lazy::new(|| include_str!(concat!(env!("OUT_DIR"), "/prompts/prompt_L1.txt")).to_string());
static TPL_L2: Lazy<String> =
    Lazy::new(|| include_str!(concat!(env!("OUT_DIR"), "/prompts/prompt_L2.txt")).to_string());
static TPL_L3: Lazy<String> =
    Lazy::new(|| include_str!(concat!(env!("OUT_DIR"), "/prompts/prompt_L3.txt")).to_string());
static TPL_L4: Lazy<String> =
    Lazy::new(|| include_str!(concat!(env!("OUT_DIR"), "/prompts/prompt_L4.txt")).to_string());
static TPL_L5: Lazy<String> =
    Lazy::new(|| include_str!(concat!(env!("OUT_DIR"), "/prompts/prompt_L5.txt")).to_string());

/// Build a row‑specific prompt, injecting L0 geographic data plus whitelist &
/// notes‑table for layers > 0.
///
/// **Elector status has been permanently removed.** Every cardinal in the
/// input set is assumed to be an elector, so the `{elector}` placeholder and
/// any related logic have been eliminated.
#[allow(clippy::too_many_arguments)]
pub fn build(
    layer: u8,
    name: &str,
    office: &str,
    country: Option<&str>,
    region: Option<&str>,
    born: Option<&str>,
    order: Option<&str>,
    consistory: Option<&str>,
) -> Cow<'static, str> {
    // 1) Basic substitution (name + assignment) -----------------------------
    let mut s = match layer {
        0 => TPL_L0.clone(),
        1 => TPL_L1.clone(),
        2 => TPL_L2.clone(),
        3 => TPL_L3.clone(),
        4 => TPL_L4.clone(),
        5 => TPL_L5.clone(),
        _ => panic!("Unsupported layer {}", layer),
    };
    s = s.replace("{name}", name).replace("{office}", office);

    // 2) Layers > 0: inject geographic & structured headers -----------------
    if layer > 0 {
        s = s
            .replace("{country}", country.unwrap_or(""))
            .replace("{region}", region.unwrap_or(""));

        // Propagate optional structured metadata ----------------------------
        s = s
            .replace("{born}", born.unwrap_or(""))
            .replace("{order}", order.unwrap_or(""))
            .replace("{consistory}", consistory.unwrap_or(""))
            // the elector placeholder is now a hard‑empty
            .replace("{elector}", "");

        // -------------------------------------------------------------------
        // Whitelist bullets (taxonomy IDs, label, short‑def) -----------------
        let layer_e = match layer {
            1 => Layer::L1,
            2 => Layer::L2,
            3 => Layer::L3,
            4 => Layer::L4,
            5 => Layer::L5,
            _ => unreachable!(),
        };

        let bullets = tagdefs(layer_e)
            .iter()
            .map(|d| format!("• {} – {}: {}", d.id, d.label, d.short_def))
            .collect::<Vec<_>>()
            .join("\n");

        if s.contains("{whitelist}") {
            s = s.replace("{whitelist}", &bullets);
        }

        // -------------------------------------------------------------------
        // Dynamic notes markdown table ---------------------------------------
        if s.contains("{notes}") {
            let table_header = "\
| Tag ID | Coverage | Notes |
|--------|----------|-------|";

            let mut rows = Vec::new();
            for def in tagdefs(layer_e) {
                rows.push(format!(
                    "| **{}** | {} | {} |",
                    def.id, def.short_def, def.notes
                ));
            }

            let table = std::iter::once(table_header.to_string())
                .chain(rows)
                .collect::<Vec<_>>()
                .join("\n");

            s = s.replace("{notes}", &table);
        }
    }

    Cow::Owned(s)
}
