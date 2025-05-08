use once_cell::sync::Lazy;
use serde_json::json;

pub fn layer_schema(layer: u8) -> (&'static str, &'static serde_json::Value) {
    match layer {
        0 => ("profile_layer0", &*L0),
        1 => ("profile_layer1", &*L1),
        2 => ("profile_layer2", &*L2),
        3 => ("profile_layer3", &*L3),
        4 => ("profile_layer4", &*L4),
        5 => ("profile_layer5", &*L5),
        _ => unreachable!(),
    }
}

// L0 – nation / region
static L0: Lazy<serde_json::Value> = Lazy::new(|| {
    json!({
        "type": "object",
        "properties": {
            "country":  { "type": "string" },
            "region":   { "type": "string" }
        },
        "required": ["country", "region"]
    })
});

static L1: Lazy<serde_json::Value> = Lazy::new(|| {
    json!({
        "type":"object",
        "properties":{
            "tags":{"type":"array","items":{"type":"string"}},
            "scores":{"type":"array","items":{"type":"number"}},
            "evidence":{"type":"array","items":{"type":"string"}}
        },
        "required":["tags","scores","evidence"]
    })
});

// L2 – identical signature
static L2: Lazy<serde_json::Value> = Lazy::new(|| L1.clone());

// L3 – adds paragraph
static L3: Lazy<serde_json::Value> = Lazy::new(|| {
    json!({
        "type":"object",
        "properties":{
            "tags":{"type":"array","items":{"type":"string"}},
            "scores":{"type":"array","items":{"type":"number"}},
            "evidence":{"type":"array","items":{"type":"string"}},
            "paragraph":{"type":"string"}
        },
        "required":["tags","scores","evidence","paragraph"]
    })
});

// L4 & L5 – same signature as L1
static L4: Lazy<serde_json::Value> = Lazy::new(|| L1.clone());
static L5: Lazy<serde_json::Value> = Lazy::new(|| L1.clone());
