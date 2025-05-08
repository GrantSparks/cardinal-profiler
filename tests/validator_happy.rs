use cardinal_profiler::validator::{validate, Layer};

#[test]
fn too_many_tags_truncated() {
    let mut tags = vec![
        "NET_SANT".into(),
        "NET_JES".into(),
        "NET_CL".into(),
        "NET_FRAN".into(),
        "NET_DOM".into(),
    ];
    let mut scores = vec![0.5, 0.6, 0.7, 0.8, 0.9];
    let mut evidence = vec![
        "e1".into(),
        "e2".into(),
        "e3".into(),
        "e4".into(),
        "e5".into(),
    ];
    // L4 allows max 4 tags
    validate(Layer::L4, &mut tags, &mut scores, &mut evidence, None).unwrap();
    assert_eq!(tags.len(), 4);
    assert_eq!(scores.len(), 4);
    assert_eq!(evidence.len(), 4);
}
