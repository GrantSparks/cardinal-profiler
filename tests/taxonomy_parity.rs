use cardinal_profiler::taxonomy::get_tagdef;
use cardinal_profiler::taxonomy::{all_tags, Layer};
use std::fs;
use toml::Value;

#[test]
fn taxonomy_parity() {
    // Read and parse the source TOML
    let toml_str =
        fs::read_to_string("assets/taxonomy.toml").expect("could not read assets/taxonomy.toml");
    let doc = toml::from_str::<Value>(&toml_str).expect("TOML parse failed");

    let layers = doc
        .get("layer")
        .and_then(Value::as_table)
        .expect("missing [layer]");

    for (lname, table) in layers {
        // map string "L1"/"l1" → Layer enum
        let layer_enum = match lname.to_ascii_uppercase().as_str() {
            "L1" => Layer::L1,
            "L2" => Layer::L2,
            "L3" => Layer::L3,
            "L4" => Layer::L4,
            "L5" => Layer::L5,
            _ => continue,
        };

        let tags_table = table
            .get("tag")
            .and_then(Value::as_table)
            .expect(&format!("missing tag table for {}", lname));

        // Expected IDs from TOML
        let expected: Vec<_> = tags_table.keys().cloned().collect();
        // Actual from generated helper
        let actual: Vec<_> = all_tags(layer_enum).iter().map(|s| s.to_string()).collect();

        assert_eq!(
            expected.len(),
            actual.len(),
            "Layer {:?}: count mismatch",
            layer_enum
        );
        for id in expected {
            assert!(
                actual.contains(&id),
                "Layer {:?}: missing tag {}",
                layer_enum,
                id
            );
            // verify our lookup helper returns Some
            assert!(
                get_tagdef(layer_enum, &id).is_some(),
                "TAXONOMY missing entry for tag {} in layer {:?}",
                id,
                layer_enum
            );
        }
    }
}
