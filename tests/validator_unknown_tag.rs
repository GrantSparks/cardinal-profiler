use cardinal_profiler::error::ProfilerError;
use cardinal_profiler::validator::{validate, Layer};

#[test]
fn validator_rejects_unknown_tag() {
    let mut tags = vec!["NON_EXISTENT".to_string()];
    let mut scores = vec![0.5];
    let mut evidence = vec!["ev".to_string()];

    let err = validate(Layer::L1, &mut tags, &mut scores, &mut evidence, None)
        .expect_err("expected error on unknown tag");

    match err {
        ProfilerError::Other(msg) => {
            assert!(msg.contains("unknown tag NON_EXISTENT"));
        }
        _ => panic!("expected ProfilerError::Other"),
    }
}
