use cardinal_profiler::validator::{validate, Layer};

#[test]
fn evidence_longer_than_tags_trimmed() {
    let mut tags = vec!["INST_CUR_PREF".into()];
    let mut scores = vec![0.8];
    let mut evidence = vec!["e1".into(), "e2".into(), "e3".into()];
    validate(Layer::L2, &mut tags, &mut scores, &mut evidence, None).unwrap();
    assert_eq!(tags.len(), 1);
    assert_eq!(evidence.len(), 1);
}
