use cardinal_profiler::error::ProfilerError;
use cardinal_profiler::parse;
use serde_json::json;

#[test]
fn parse_l3_malformed_json() {
    // Build a structurally-wrong Value (missing 'tags')
    let bad_val = json!({"foo":"bar"});
    match parse::parse_l3(&bad_val) {
        Err(ProfilerError::Other(msg)) => assert!(msg.contains("Missing or invalid 'tags'")),
        _ => panic!("Expected shape validation error"),
    }
}

#[test]
fn parse_l4_malformed_json() {
    let bad_val = json!({"tags":[], "scores":"oops"});
    match parse::parse_l4(&bad_val) {
        Err(ProfilerError::Other(msg)) => assert!(msg.contains("Missing or invalid 'scores'")),
        _ => panic!("Expected shape validation error"),
    }
}
