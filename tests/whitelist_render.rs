use cardinal_profiler::prompt;

#[test]
fn whitelist_render_first_lines() {
    // Render first 2 bullets for layer 1 as a smoke test
    let prompt = prompt::build(
        1,
        "Name",
        "Office",
        Some("Nat"),
        Some("Cont"),
        None,
        None,
        None,
    );
    let section = prompt
        .lines()
        .skip_while(|l| !l.starts_with("• "))
        .take(2)
        .collect::<Vec<_>>();
    assert!(section[0].starts_with("•"));
    assert!(section[1].starts_with("•"));
}
