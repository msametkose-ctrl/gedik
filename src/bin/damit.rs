//! Damıtma verisi: ResNet öğretmenin bilgisini hızlı MLP'ye aktarmak için.
//!
//! ResNet düğüm başına MLP'den çok daha iyi değerlendiriyor (eşit
//! iterasyonda 34-14), ama ~200 kat yavaş olduğu için gerçek sürede
//! kaybediyor (400 ms'de 16-32). Çözüm: pozisyonları hızlı motorla üret,
//! her birini ResNet'e etiketlet, MLP'yi bu etiketlerle eğit
//! (`tools/egit_np.py`). Öğrenci MLP hızında, öğretmenin yargısıyla oynar.
//!
//! Satır biçimi `kodla`nın 6 sütunlu biçimi: `fen;y;ply;v;z;`
//! `y = lambda * v + (1 - lambda) * z` — yumuşak etiket. `v` öğretmenin
//! değeri, `z` oyunun sonucu (bitmeyen oyunda `-`, o zaman `y = v`). `v` ve
//! `z` ayrıca yazılıyor ki `lambda` veriyi yeniden üretmeden değiştirilebilsin.
//! Bu iki sütun `kodla --spatial`ın kalan-hamle/yol-farkı sütunlarıyla aynı
//! yerde: bu dosyayı yalnızca MLP (QDT3) kodlamasıyla kullan.
//!
//! Kullanım: damit <oyun> <iterasyon> <thread> <cikti.csv> [lambda] [ogretmen.bin] [seed]

use gedik::board::Position;
use gedik::heuristics::{candidate_moves, Rng};
use gedik::mcts::{Config, Mcts};
use gedik::moves::Move;
use gedik::notation::to_fen;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};

/// İlk hamlelerde en iyi hamle yerine ziyaret oranında örnekle: aynı
/// açılıştan hep aynı oyun çıkmasın, veri çeşitlensin.
fn sample(top: &[(Move, u32, f32)], rng: &mut Rng) -> Option<Move> {
    let total: u32 = top.iter().map(|t| t.1).sum();
    if total == 0 {
        return None;
    }
    let mut k = rng.below(total as usize) as u32;
    for t in top {
        if k < t.1 {
            return Some(t.0);
        }
        k -= t.1;
    }
    None
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let games: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(1000);
    let iters: u32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);
    let threads: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    });
    let out = a.get(4).cloned().unwrap_or_else(|| "damit.csv".into());
    let lambda: f32 = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(0.8);
    let teacher_path = a.get(6).cloned().unwrap_or_else(|| "ag_resnet.bin".into());
    let seed: u64 = a.get(7).and_then(|s| s.parse().ok()).unwrap_or(1);

    let teacher = gedik::nn::net_from(&teacher_path).expect("ogretmen ag yuklenemedi");
    eprintln!("damit: {games} oyun, {iters} iterasyon, {threads} thread, lambda {lambda}, ogretmen {teacher_path}");

    let (tx, rx) = mpsc::sync_channel::<Vec<String>>(256);
    let done = Arc::new(AtomicUsize::new(0));
    let rows_total = Arc::new(AtomicUsize::new(0));

    let file = std::fs::File::create(&out).expect("cikti acilamadi");
    let writer = std::thread::spawn(move || {
        let mut w = BufWriter::with_capacity(1 << 22, file);
        writeln!(w, "fen;y;ply;v;z;pi").unwrap();
        for batch in rx {
            for line in batch {
                writeln!(w, "{line}").unwrap();
            }
        }
        w.flush().unwrap();
    });

    let t0 = std::time::Instant::now();
    std::thread::scope(|s| {
        for t in 0..threads {
            let tx = tx.clone();
            let done = Arc::clone(&done);
            let rows_total = Arc::clone(&rows_total);
            s.spawn(move || {
                let tseed = seed
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    .wrapping_add(t as u64 * 7919 + 1);
                let mut m = Mcts::new(tseed);
                let mut cfg = Config::default();
                cfg.root_noise = 0.25;
                cfg.apply(&mut m);
                let mut rng = Rng::new(tseed ^ 0x5bd1_e995);
                let mut scratch = Vec::new();

                for _ in (t..games).step_by(threads) {
                    let open = rng.below(11) as u16;
                    let mut pos = Position::start();
                    for _ in 0..open {
                        let mv = candidate_moves(&pos);
                        if mv.is_empty() || pos.winner().is_some() {
                            break;
                        }
                        pos.make(mv[rng.below(mv.len())]);
                    }
                    if pos.winner().is_some() {
                        done.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }

                    // (fen, sıradaki oyuncu, ply, öğretmen değeri)
                    let mut rows: Vec<(String, usize, u16, f32)> = Vec::with_capacity(128);
                    while pos.winner().is_none() && pos.ply < 300 {
                        let (best, st) = m.search_rollouts(&pos, iters);
                        let Some(best) = best else { break };
                        // Başlangıç pozisyonu binlerce kez tekrar ederdi.
                        if pos.ply >= 2 {
                            let v = teacher.value(&pos, &mut scratch);
                            rows.push((to_fen(&pos), pos.side as usize, pos.ply, v));
                        }
                        let mv = if pos.ply < open + 8 {
                            sample(&st.top, &mut rng).unwrap_or(best)
                        } else {
                            best
                        };
                        pos.make(mv);
                    }

                    let w = pos.winner();
                    let lines: Vec<String> = rows
                        .into_iter()
                        .map(|(fen, side, ply, v)| {
                            match w {
                                Some(w) => {
                                    let z = f32::from(w == side);
                                    let y = lambda * v + (1.0 - lambda) * z;
                                    format!("{fen};{y:.5};{ply};{v:.5};{z};")
                                }
                                None => format!("{fen};{v:.5};{ply};{v:.5};-;"),
                            }
                        })
                        .collect();
                    rows_total.fetch_add(lines.len(), Ordering::Relaxed);
                    tx.send(lines).unwrap();
                    let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if t == 0 && d % 200 < threads {
                        eprintln!(
                            "  {d}/{games} oyun, {} satir, {:.0} sn",
                            rows_total.load(Ordering::Relaxed),
                            t0.elapsed().as_secs_f64()
                        );
                    }
                }
            });
        }
    });
    drop(tx);
    writer.join().unwrap();
    eprintln!(
        "bitti: {} oyun, {} satir -> {out} ({:.0} sn)",
        done.load(Ordering::Relaxed),
        rows_total.load(Ordering::Relaxed),
        t0.elapsed().as_secs_f64()
    );
}
