use gedik::board::Position;
use gedik::engine::Engine;
use gedik::heuristics::{candidate_moves, Rng};
use gedik::notation::to_fen;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let total_games: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20000);
    let iters: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let num_threads: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8)
    });
    let out_path: String = args.get(4).cloned().unwrap_or_else(|| "selfplay_v2.csv".to_string());

    println!("=== AlphaZero İkinci Nesil Self-Play Üretici ===");
    println!("  Toplam Oyun:   {}", total_games);
    println!("  MCTS İterasyon:{}", iters);
    println!("  İş Parçacığı:  {} thread", num_threads);
    println!("  Çıktı Dosyası: {}", out_path);
    println!("================================================");

    let (tx, rx) = mpsc::sync_channel::<Vec<String>>(1000);
    let done_games = Arc::new(AtomicUsize::new(0));

    let out_file = std::fs::File::create(&out_path).expect("Cikti dosyasi olusturulamadi");
    let mut writer = BufWriter::with_capacity(4 * 1024 * 1024, out_file);
    writeln!(writer, "fen;label;ply;pi").unwrap();

    let writer_handle = thread::spawn(move || {
        let mut total_rows = 0usize;
        for batch in rx {
            for line in batch {
                let _ = writeln!(writer, "{}", line);
                total_rows += 1;
            }
        }
        writer.flush().unwrap();
        total_rows
    });

    let games_per_thread = (total_games + num_threads - 1) / num_threads;
    let t0 = Instant::now();
    let mut handles = Vec::new();

    for t_id in 0..num_threads {
        let tx_clone = tx.clone();
        let done_clone = Arc::clone(&done_games);

        let handle = thread::spawn(move || {
            let mut rng = Rng::new(100_000 + (t_id as u64 * 7919));
            let start_g = t_id * games_per_thread;
            let end_g = (start_g + games_per_thread).min(total_games);

            for g in start_g..end_g {
                let open_plies = 2 + (g % 7) * 2;
                let mut pos = Position::start();
                let mut ok = true;
                for _ in 0..open_plies {
                    let mv = candidate_moves(&pos);
                    if mv.is_empty() || pos.winner().is_some() {
                        ok = false;
                        break;
                    }
                    pos.make(mv[rng.below(mv.len())]);
                }
                if !ok || pos.winner().is_some() {
                    done_clone.fetch_add(1, Ordering::Relaxed);
                    continue;
                }

                // (fen, ply, side, pi_str)
                let mut rows: Vec<(String, u16, usize, String)> = Vec::with_capacity(64);

                while pos.winner().is_none() && pos.ply < 300 {
                    let is_full = rng.unit() < 0.35;
                    let current_iters = if is_full { iters } else { (iters / 4).max(300) };

                    let mut engine = Engine::parse(
                        &format!("mcts:{current_iters}:gumbel=1,nn=1"),
                        rng.next_u64(),
                    );
                    let c = engine.choose(&pos);
                    let Some(mv) = c.mv else { break };

                    let total: u32 = c.top.iter().map(|t| t.1).sum();
                    let pi_str = if is_full && total > 0 {
                        let pi: Vec<String> = c
                            .top
                            .iter()
                            .filter(|t| t.1 > 0)
                            .map(|t| format!("{}:{:.4}", t.0.action_id(), t.1 as f32 / total as f32))
                            .collect();
                        pi.join(" ")
                    } else {
                        String::new()
                    };

                    rows.push((to_fen(&pos), pos.ply, pos.side as usize, pi_str));
                    pos.make(mv);
                }

                if let Some(w) = pos.winner() {
                    let total_plies = pos.ply;
                    let d0 = pos.distance(0).unwrap_or(99) as i32;
                    let d1 = pos.distance(1).unwrap_or(99) as i32;

                    let mut batch = Vec::with_capacity(rows.len());
                    for (fen, ply, side, pi) in rows {
                        let label = if side == w as usize { 1 } else { 0 };
                        let rem_plies = total_plies.saturating_sub(ply);
                        let final_delta = if side == 0 { d1 - d0 } else { d0 - d1 };
                        batch.push(format!("{fen};{label};{ply};{rem_plies};{final_delta};{pi}"));
                    }
                    let _ = tx_clone.send(batch);
                }

                done_clone.fetch_add(1, Ordering::Relaxed);
            }
        });
        handles.push(handle);
    }

    drop(tx);

    while done_games.load(Ordering::Relaxed) < total_games {
        thread::sleep(std::time::Duration::from_millis(1500));
        let done = done_games.load(Ordering::Relaxed);
        let elapsed = t0.elapsed().as_secs_f64();
        let gps = done as f64 / elapsed.max(0.01);
        let eta = (total_games.saturating_sub(done)) as f64 / gps.max(0.01);
        let pct = (done as f64 / total_games as f64) * 100.0;
        print!(
            "\r  [İlerleme: {:>5}/{} | %{:<4.1} | {:<4.1} oyun/sn | Kalan: {:<4.1} dk]    ",
            done, total_games, pct, gps, eta / 60.0
        );
        std::io::stdout().flush().ok();
    }
    println!();

    for h in handles {
        h.join().unwrap();
    }
    let total_rows = writer_handle.join().unwrap();
    let total_sec = t0.elapsed().as_secs_f64();

    println!(
        "\n✅ Self-Play Tamamlandı: {} oyun, {} satır veri üretildi ({:.1} sn, {:.1} oyun/sn).",
        total_games, total_rows, total_sec, total_games as f64 / total_sec
    );
}

