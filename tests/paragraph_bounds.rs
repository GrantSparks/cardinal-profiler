use cardinal_profiler::validator::{validate, Layer};

fn make_paragraph(words: usize) -> String {
    (0..words)
        .map(|i| format!("w{}", i))
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn paragraph_trimmed_to_100() {
    let mut tags = vec!["REF_STATUS_QUO".into()];
    let mut scores = vec![0.9];
    let mut evidence = vec!["E".into()];
    let mut paragraph = make_paragraph(150);
    let para_opt = Some(&mut paragraph);
    validate(Layer::L3, &mut tags, &mut scores, &mut evidence, para_opt).unwrap();
    let words: Vec<_> = paragraph.split_whitespace().collect();
    assert!(words.len() <= 100);
}

#[test]
fn paragraph_padded_to_70() {
    let mut tags = vec!["REF_STATUS_QUO".into()];
    let mut scores = vec![0.9];
    let mut evidence = vec!["E".into()];
    let mut paragraph = make_paragraph(50);
    let para_opt = Some(&mut paragraph);
    validate(Layer::L3, &mut tags, &mut scores, &mut evidence, para_opt).unwrap();
    let new_count = paragraph.split_whitespace().count();
    assert!(new_count >= 70);
    assert!(paragraph.starts_with(&make_paragraph(50)));
}
