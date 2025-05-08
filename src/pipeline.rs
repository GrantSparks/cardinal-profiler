// Asynchronous CSV-processing pipeline

use crate::config::Config;
use crate::csv_io::{self, CsvReader};
use crate::error::ProfilerError;
use crate::functions;
use crate::openai_client::LanguageApi;
use crate::output;
use crate::output::PROFILER_COLS;
use crate::parse;
use crate::prompt;
use crate::shutdown::Shutdown;
use crate::validator::{self, Layer};

use futures::{future::join_all, StreamExt};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::{params, OptionalExtension};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, Semaphore};
use tokio_stream::{iter, wrappers::ReceiverStream};

// ---------------------------------------------------------------------------
// Helper structs & utils
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct RowBuf {
    idx: usize,
    name: String,
    office: String,

    // new optional structured inputs
    country: Option<String>,
    born: Option<String>,
    order: Option<String>,
    consistory: Option<String>,

    region: Option<String>,
    row: Vec<String>,
}

type DbPool = Pool<SqliteConnectionManager>;

fn layer_to_validator(layer: u8) -> Layer {
    match layer {
        1 => Layer::L1,
        2 => Layer::L2,
        3 => Layer::L3,
        4 => Layer::L4,
        5 => Layer::L5,
        _ => unreachable!(),
    }
}

fn filter_min_score(cfg: &Config, t: &mut Vec<String>, s: &mut Vec<f32>, e: &mut Vec<String>) {
    if cfg.min_score == 0.0 {
        return;
    }
    let keep: Vec<usize> = s
        .iter()
        .enumerate()
        .filter(|&(_, &sc)| sc >= cfg.min_score)
        .map(|(i, _)| i)
        .collect();
    let mut nt = Vec::new();
    let mut ns = Vec::new();
    let mut ne = Vec::new();
    for i in keep {
        nt.push(t[i].clone());
        ns.push(s[i]);
        if i < e.len() {
            ne.push(e[i].clone());
        }
    }
    *t = nt;
    *s = ns;
    *e = ne;
}

fn hash_prompt(p: &str) -> String {
    hex::encode(Sha256::digest(p.as_bytes()))
}

fn cache_get(pool: &Arc<DbPool>, key: &str, layer: u8) -> Result<Option<String>, ProfilerError> {
    let conn = pool.get().map_err(ProfilerError::from)?;
    conn.query_row(
        "SELECT response FROM cache WHERE prompt_hash=?1 AND layer=?2",
        params![key, layer as i32],
        |r| r.get::<_, String>(0),
    )
    .optional()
    .map_err(ProfilerError::from)
}

// ---------------------------------------------------------------------------
// Cache open helper
// ---------------------------------------------------------------------------

fn open_cache(cfg: &Config) -> Result<Option<Arc<DbPool>>, ProfilerError> {
    if cfg.no_cache {
        return Ok(None);
    }
    let mut path = cfg.output.clone();
    path.set_extension("cache.sqlite");

    // ── STEP 1 ── run migration *and* switch to WAL once, with an exclusive connection
    {
        use rusqlite::Connection;
        let conn = Connection::open(&path)?; // single writer
        conn.execute_batch(include_str!("../schema.sql"))?; // creates/updates schema
        conn.pragma_update(None, "journal_mode", "WAL")?; // ← set WAL once
    } // ← connection drops here, releasing the write-lock

    // ── STEP 2 ── build the r2d2 pool; per-conn init only sets pragmas —────────
    fn init_pragmas(c: &mut rusqlite::Connection) -> rusqlite::Result<()> {
        // No journal_mode setting here - done once above
        c.pragma_update(None, "busy_timeout", 30_000)?;
        Ok(())
    }

    let mgr = SqliteConnectionManager::file(&path).with_init(init_pragmas);

    let pool = Pool::builder()
        .max_size(8)
        .build(mgr)
        .map_err(|e| ProfilerError::Other(format!("failed to create pool: {e}")))?;

    let v: i32 = pool
        .get()?
        .query_row("SELECT version FROM meta", [], |r| r.get(0))?;
    if v != 2 {
        return Err(ProfilerError::Other("cache schema mismatch".into()));
    }
    Ok(Some(Arc::new(pool)))
}

// ---------------------------------------------------------------------------
// Entry-point
// ---------------------------------------------------------------------------

pub async fn run<A: LanguageApi + 'static>(
    cfg: &Config,
    client: A,
    sd: Shutdown,
) -> Result<(), ProfilerError> {
    // Get cancellation token early to distribute to all tasks
    let cancel = sd.subscribe();

    // 1. CSV reader
    let rdr = CsvReader::new(cfg.input.to_string_lossy().into_owned(), cfg.delimiter);
    let raw = rdr.read()?;

    if cfg.layer_start > 0
        && (!raw.headers.contains(&"Country".to_string())
            || !raw.headers.contains(&"Region".to_string()))
    {
        return Err(ProfilerError::Other(
            "--layer-start > 0 needs Country/Region cols (run L0 first)".into(),
        ));
    }

    let total = raw.rows.len() as u64;
    let user_len = raw.headers.len();

    // 2. CSV writer
    // build our output header once
    let mut header = raw.headers.clone();
    header.extend(PROFILER_COLS.iter().map(|s| s.to_string()));

    let delim = cfg.delimiter.unwrap_or(b',');
    let tmp_path = cfg.output.with_extension("tmp");

    // ── 3. Shared state ---------------------------------------------------
    let cache = open_cache(cfg)?;
    let llm_sem = Arc::new(Semaphore::new(cfg.api_parallel));

    let audit: Option<Arc<Mutex<File>>> = if cfg.audit {
        let mut p = cfg.output.clone();
        let stem = p.file_stem().unwrap().to_string_lossy();
        p.set_file_name(format!("{stem}_audit.jsonl"));
        Some(Arc::new(Mutex::new(File::create(p)?)))
    } else {
        None
    };

    // ── 4. Progress bars --------------------------------------------------
    let mprog = MultiProgress::new();
    let hide = !atty::is(atty::Stream::Stderr) || cfg.no_progress || cfg.verbose;
    let layers_req: Vec<u8> = (cfg.layer_start as u8..=cfg.layer_stop as u8).collect();
    let mut bar_vec: Vec<ProgressBar> = (0..=5).map(|_| ProgressBar::hidden()).collect();

    for &l in &layers_req {
        // if `hide`→ just use a detached hidden bar (no draw thread involved)
        let pb = if hide {
            ProgressBar::hidden()
        } else {
            mprog.add(ProgressBar::new(total))
        };
        let label = format!("L{l}:");
        let tpl = if hide {
            "{pos}/{len}".to_string()
        } else if cfg.no_emoji {
            format!("[{label} {{wide_bar}} {{pos}}/{{len}} ({{percent}}%)]")
        } else {
            format!("[{label} {{bar:40.cyan/blue}} {{pos}}/{{len}} ({{percent}}%)]")
        };
        pb.set_style(ProgressStyle::with_template(&tpl).unwrap());
        bar_vec[l as usize] = pb;
    }
    let bars = Arc::new(bar_vec);

    let idx_name = raw.idx_name;
    let idx_assign = raw.idx_assign;
    let idx_region = raw.headers.iter().position(|h| h == "Region");

    // ── 5. Channels -------------------------------------------------------
    let (tx_a, rx_a) = mpsc::channel::<RowBuf>(cfg.parallel * 4);
    let (tx_res, rx_res) = mpsc::channel::<(usize, csv::StringRecord)>(cfg.parallel * 4);

    // ── 6. Writer task ------------------------------------------------------

    // ──────────────────────────────────────────────────────────────────────
    // Writer task – owns its buffers, header row, temp-file path, receiver
    // ──────────────────────────────────────────────────────────────────────
    let writer_handle = {
        let final_path = cfg.output.clone(); // move into task
        let tmp_path = tmp_path.clone(); // move into task
        let header = header.clone(); // move into task
        let mut rx_res = rx_res; // move + make mutable
        let cancel = cancel.clone();

        tokio::spawn(async move {
            // If we were already cancelled before the task starts, bail early
            if cancel.is_cancelled() {
                return Ok::<(), ProfilerError>(());
            }

            // Create CSV writer and emit header immediately
            let mut wtr = csv_io::writer(&tmp_path, delim)?;
            wtr.write_record(&header)?;
            wtr.flush()?;

            // Re-ordering buffer so rows are written in original input order
            let mut next_idx: usize = 0;
            let mut buf: BTreeMap<usize, csv::StringRecord> = BTreeMap::new();

            loop {
                tokio::select! {
                    // Global cancellation: flush & break
                    _ = cancel.cancelled() => {
                        wtr.flush()?;
                        break;
                    }

                    // Normal stream of results from Stage-B
                    maybe = rx_res.recv() => {
                        match maybe {
                            Some((idx, rec)) => {
                                buf.insert(idx, rec);
                                while let Some(r) = buf.remove(&next_idx) {
                                    wtr.write_record(&r)?;
                                    if next_idx % 100 == 0 { wtr.flush()?; }
                                    next_idx += 1;
                                }
                            }
                            None => break,        // all senders are closed
                        }
                    }
                }
            }

            // Finalise file and atomically move it into place
            wtr.flush()?;
            output::atomic_rename(&tmp_path, &final_path)?;
            Ok::<(), ProfilerError>(())
        })
    };
    // Make a clone for stage B to use
    let tx_res_b = tx_res.clone();
    // Drop the original sender so writer task knows when all senders are done
    drop(tx_res);

    // ── 7. Stage B task ---------------------------------------------------
    {
        let cfg_b = cfg.clone();
        let cache_b = cache.clone();
        let bars_b = bars.clone();
        let llm_sem_b = llm_sem.clone();
        let audit_b = audit.clone();
        let client_b = client.clone();
        tokio::spawn(async move {
            ReceiverStream::new(rx_a)
                .for_each_concurrent(cfg_b.parallel, move |mut rb| {
                    let cfg = cfg_b.clone();
                    let cache = cache_b.clone();
                    let bars = bars_b.clone();
                    let llm_sem = llm_sem_b.clone();
                    let audit = audit_b.clone();
                    let client = client_b.clone();
                    let tx_res = tx_res_b.clone();
                    async move {
                        let base = user_len;
                        let start = (cfg.layer_start as u8).max(1);
                        let stop = cfg.layer_stop as u8;

                        if start > stop {
                            // Nothing to do – forward immediately
                            let _ = tx_res.send((rb.idx, csv::StringRecord::from(rb.row))).await;
                            return;
                        }

                        // Fan-out futures ------------------------------------------------
                        let mut layer_futs = Vec::new();
                        for layer_id in start..=stop {
                            let prom = prompt::build(
                                layer_id,
                                &rb.name,
                                &rb.office,
                                // prefer the user-supplied Country; fall back to the nation alias
                                rb.country.as_deref(),
                                rb.region.as_deref(),
                                rb.born.as_deref(),
                                rb.order.as_deref(),
                                rb.consistory.as_deref(),
                            );
                            let (func, schema) = functions::layer_schema(layer_id);
                            let key = hash_prompt(&prom);
                            let cache = cache.clone();
                            let client = client.clone();
                            let sem = llm_sem.clone();
                            let cfg_fan = cfg.clone();
                            layer_futs.push(async move {
                                // Try cache first
                                if !cfg_fan.refresh_cache {
                                    if let Some(pool) = &cache {
                                        if let Some(hit) = cache_get(pool, &key, layer_id)? {
                                            return Ok::<(u8, String), ProfilerError>((layer_id, hit));
                                        }
                                    }
                                }

                                let _p = sem.acquire_owned().await.unwrap();
                                let resp = client.process_prompt_fc(&prom, func, schema).await?;
                                drop(_p);

                                if let Some(pool) = &cache {
                                    let _ = pool.get().map(|c| c.execute(
                                            "INSERT OR REPLACE INTO cache VALUES (?1,?2,?3)",
                                            params![key, layer_id as i32, resp],
                                        ));
                                }
                                Ok::<(u8, String), ProfilerError>((layer_id, resp))
                            });
                        }

                        let results = join_all(layer_futs).await;
                        for res in results {
                            let (layer_id, raw_json) = match res {
                                Ok(t) => t,
                                Err(e) => {
                                    eprintln!("⚠️ row {} layer-fan-out error: {}", rb.idx, e);
                                    continue;
                                }
                            };

                            let val: serde_json::Value = match serde_json::from_str(&raw_json) {
                                Ok(v) => v,
                                Err(e) => {
                                    eprintln!(
                                        "⚠️ row {} L{}: JSON parse failed – {}",
                                        rb.idx, layer_id, e
                                    );
                                    continue;
                                }
                            };

                            match layer_id {
                                1 | 2 | 4 | 5 => {
                                    match parse::parse_l4(&val) {
                                        Ok((mut t, mut s, mut ev)) => {
                                            filter_min_score(&cfg, &mut t, &mut s, &mut ev);
                                            if validator::validate(
                                                layer_to_validator(layer_id),
                                                &mut t,
                                                &mut s,
                                                &mut ev,
                                                None,
                                            )
                                            .is_err()
                                            {
                                                continue;
                                            }
                                            let blk = (layer_id - 1) as usize * 2;   // 2 cols per layer now
                                            let tag_i = base + 2 + blk;                // skip Country + Region
                                            rb.row[tag_i] = parse::join_tags(&t);
                                            rb.row[tag_i + 1] = parse::join_scores(&s);
                                            // Evidence line removed as requested
                                            if let Some(a) = &audit {
                                                let mut fh = a.lock().unwrap();
                                                let _ = writeln!(
                                                    fh,
                                                    "{}",
                                                    json!({"row": rb.idx,"layer":layer_id,"tags":t,"scores":s,"evidence":ev})
                                                );
                                            }
                                        }
                                        Err(e) => eprintln!("⚠️ row {} L{}: {}", rb.idx, layer_id, e),
                                    }
                                }
                                3 => {
                                    match parse::parse_l3(&val) {
                                        Ok((mut t, mut s, mut ev, mut para)) => {
                                            filter_min_score(&cfg, &mut t, &mut s, &mut ev);
                                            if validator::validate(
                                                Layer::L3,
                                                &mut t,
                                                &mut s,
                                                &mut ev,
                                                Some(&mut para),
                                            )
                                            .is_err()
                                            {
                                                continue;
                                            }
                                            let blk = (layer_id - 1) as usize * 2;
                                            let tag_i = base + 2 + blk;
                                            rb.row[tag_i] = parse::join_tags(&t);
                                            rb.row[tag_i + 1] = parse::join_scores(&s);
                                            // Evidence line removed as requested
                                            rb.row[base + PROFILER_COLS.len() - 1] = para;
                                            if let Some(a) = &audit {
                                                let mut fh = a.lock().unwrap();
                                                let _ = writeln!(
                                                    fh,
                                                    "{}",
                                                    json!({"row": rb.idx,"layer":layer_id,"tags":t,"scores":s,"evidence":ev})
                                                );
                                            }
                                        }
                                        Err(e) => eprintln!("⚠️ row {} L3: {}", rb.idx, e),
                                    }
                                }
                                _ => unreachable!(),
                            }

                            // bump bar
                            if (layer_id as usize) < bars.len() {
                                bars[layer_id as usize].inc(1);
                            }
                        }

                        let _ = tx_res.send((rb.idx, csv::StringRecord::from(rb.row))).await;
                    }
                })
                .await;
        });
    }

    // ── 8. Stage A producer ----------------------------------------------
    let stage_a_futs = raw.rows.into_iter().enumerate().map(|(idx, rec)| {
        let cfg = cfg.clone();
        let cache = cache.clone();
        let bars = bars.clone();
        let llm_sem = llm_sem.clone();
        let client = client.clone();
        let tx_a = tx_a.clone();
        async move {
            let mut row_vec = rec.cols.clone();
            row_vec.resize(user_len + PROFILER_COLS.len(), String::new());
            let name = rec.get(idx_name).unwrap_or(&String::new()).clone();
            let office = rec.get(idx_assign).unwrap_or(&String::new()).clone();

            // pull through any user-supplied structured cols
            let mut country_val = rec.country.clone();
            let born = rec.born.clone();
            let order = rec.order.clone();
            let consistory = rec.consistory.clone();

            let mut region = idx_region.and_then(|i| rec.get(i)).map(|s| s.to_string());

            if cfg.layer_start == 0 {
                let p0 = prompt::build(0, &name, &office, None, None, None, None, None);
                let (func0, schema0) = functions::layer_schema(0);
                let key0 = hash_prompt(&p0);

                let mut json_out = None;
                if !cfg.refresh_cache {
                    if let Some(pool) = &cache {
                        match cache_get(pool, &key0, 0) {
                            Ok(Some(hit)) => json_out = Some(hit),
                            Err(e) => {
                                eprintln!("⚠️ row {} cache lookup error: {}", idx, e);
                            }
                            _ => {}
                        }
                    }
                }

                if json_out.is_none() {
                    let _p = llm_sem.acquire_owned().await.unwrap();
                    let resp_res = client.process_prompt_fc(&p0, func0, schema0).await;
                    if let Ok(resp) = resp_res {
                        json_out = Some(resp);
                    } else if let Err(e) = resp_res {
                        eprintln!("⚠️ row {} L0: {}", idx, e);
                        // Continue with blank Nation/Region fields
                    }
                    drop(_p);
                    if let (Some(pool), Some(ref txt)) = (&cache, &json_out) {
                        let _ = pool.get().map(|c| {
                            c.execute(
                                "INSERT OR REPLACE INTO cache VALUES (?1,0,?2)",
                                params![key0, txt],
                            )
                        });
                    }
                }

                if let Some(txt) = json_out {
                    let val: serde_json::Value = serde_json::from_str(&txt).unwrap_or_default();
                    country_val = val
                        .get("country")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    region = val
                        .get("region")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                }

                row_vec[user_len] = country_val.clone().unwrap_or_default();
                row_vec[user_len + 1] = region.clone().unwrap_or_default();
                if !bars[0].is_hidden() {
                    bars[0].inc(1);
                }
            }

            let rb = RowBuf {
                idx,
                name,
                office,
                country: country_val,
                born,
                order,
                consistory,
                region,
                row: row_vec,
            };
            let _ = tx_a.send(rb).await;
        }
    });

    iter(stage_a_futs)
        .buffer_unordered(cfg.parallel)
        .for_each(|_| async {})
        .await;
    drop(tx_a); // close channel so Stage B can finish
                // ── 9. Wait for either a shutdown signal or writer completion -------
    let shutdown_rx = sd.subscribe();
    let res = tokio::select! {
        // If CTRL-C or SIGTERM fires...
        _ = shutdown_rx.cancelled() => {
            eprintln!("🔴 Shutdown received—aborting pipeline early");
            Err(ProfilerError::Other("cancelled".into()))
        }
        // Otherwise wait for the writer to finish
        writer = writer_handle => {
            match writer {
                // The task itself ran to completion
                Ok(Ok(())) => {
                    // normal—writer flushed and renamed successfully
                    Ok(())
                }
                // The writer task returned an Err(ProfilerError)
                Ok(Err(e)) => {
                    Err(e)
                }
                // The writer task panicked or was aborted
                Err(join_err) => {
                    Err(ProfilerError::Other(format!(
                        "Writer task failed to join: {}",
                        join_err
                    )))
                }
            }
        }
    };

    // mark active progress-bars as done so MultiProgress can shut down cleanly
    if !hide {
        for pb in bars.iter() {
            pb.finish_and_clear();
        }
    }

    res
}
