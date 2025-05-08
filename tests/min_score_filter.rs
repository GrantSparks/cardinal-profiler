use cardinal_profiler::parse;
use serde_json::Value;

#[test]
fn min_score_filters_out_low_scores() {
    let json = r#"{"tags":["A","B","C"],"scores":[0.2,0.5,0.8],"evidence":["e1","e2","e3"]}"#;
    let val: Value = serde_json::from_str(json).unwrap();
    let (tags, scores, _ev) = parse::parse_l4(&val).expect("parse_l4 failed");
    let threshold = 0.5;
    let filtered: Vec<_> = tags
        .iter()
        .zip(&scores)
        .filter_map(|(t, &s)| {
            if s >= threshold {
                Some(t.clone())
            } else {
                None
            }
        })
        .collect();
    assert_eq!(filtered, vec!["B".to_string(), "C".to_string()]);
}
