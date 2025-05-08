use cardinal_profiler::csv_io::reader;
use std::fs::File;
use std::io::Write;
use tempfile::tempdir;

#[test]
fn sniff_semicolon() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("semi.csv");
    let mut f = File::create(&path).unwrap();
    writeln!(f, "a;b;c").unwrap();
    writeln!(f, "1;2;3").unwrap();
    let (_rdr, delim) = reader(&path, None).unwrap();
    assert_eq!(delim, b';');
}

#[test]
fn sniff_tab() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("tab.csv");
    let mut f = File::create(&path).unwrap();
    writeln!(f, "a\tb\tc").unwrap();
    writeln!(f, "1\t2\t3").unwrap();
    let (_rdr, delim) = reader(&path, None).unwrap();
    assert_eq!(delim, b'\t');
}

#[test]
fn override_comma() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("mix.csv");
    let mut f = File::create(&path).unwrap();
    writeln!(f, "a,b,c").unwrap();
    let (_rdr, delim) = reader(&path, Some(b',')).unwrap();
    assert_eq!(delim, b',');
}
