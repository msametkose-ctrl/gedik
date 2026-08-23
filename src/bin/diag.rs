//! Rollout politikasının ayırt etme gücünü ölçer.
//!
//! Soru: saf rollout, açıkça iyi bir hamleyi (piyonu ilerlet) açıkça kötü bir
//! hamleden (kendi önüne duvar koy) ayırt edebiliyor mu? Ayırt edemiyorsa
//! MCTS'in kök seçimi gürültüden ibaret demektir.

use gedik::board::Position;
use gedik::heuristics::{playout, Rng};
use gedik::notation::parse_move;

fn main() {
    let pos = Position::start();
    let samples: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50_000);

    // A (side 0) e1'de, hedefi 9. sıra. B e9'da.
    let probes = [
        ("e2", "piyonu ilerlet — açıkça iyi"),
        ("e8h", "rakibin önüne duvar — iyi"),
        ("e2h", "KENDİ önüne duvar — açıkça kötü"),
        ("a5h", "tahtanın ucunda alakasız duvar — kötü/nötr"),
        ("d1", "yana kaçış — nötr/hafif kötü"),
    ];

    for wall_prob in [0.30f32, 0.15, 0.05, 0.0] {
        println!("\n=== wall_prob = {wall_prob:.2}   ({samples} rollout/hamle) ===");
        let mut rows: Vec<(f64, f64, &str, &str)> = Vec::new();
        for (mv_str, label) in probes {
            let Some(mv) = parse_move(&pos, mv_str) else {
                println!("{mv_str}: illegal?!");
                continue;
            };
            let mut p = pos;
            p.make(mv);
            // Hamleyi yapan A; A'nın kazanma oranını ölçüyoruz.
            let mut rng = Rng::new(0xC0FFEE ^ (mv.0 as u64) << 20);
            let mut a_wins = 0usize;
            for _ in 0..samples {
                if playout(p, &mut rng, wall_prob, 400) == 0 {
                    a_wins += 1;
                }
            }
            let wr = a_wins as f64 / samples as f64;
            let se = (wr * (1.0 - wr) / samples as f64).sqrt();
            rows.push((wr, se, mv_str, label));
        }
        rows.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (wr, se, mv_str, label) in rows {
            println!("  {mv_str:<5} A kazanma {:.4} ± {:.4}   {label}", wr, 1.96 * se);
        }
    }
}

