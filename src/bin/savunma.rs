//! Bir pozisyonda tehdidi en çok azaltan hamleyi arar.
//! Savunma mümkün mü, yoksa motor haklı mı?
use gedik::board::Position;
use gedik::heuristics::{best_threat, candidate_moves};
use gedik::notation::{from_fen, move_name};

fn main() {
    let fen = std::env::args().nth(1).expect("kullanım: savunma <fen>");
    let p: Position = from_fen(&fen).expect("geçersiz fen");
    let me = p.side as usize;
    let opp = 1 - me;
    let base = best_threat(&p, opp, me);
    let d0 = p.distance(me).unwrap_or(0);
    println!("sira: {}  yol {}  mevcut tehdit {}", if me == 0 { "A" } else { "B" }, d0, base);

    let mut rows: Vec<(u32, i64, String)> = Vec::new();
    for mv in candidate_moves(&p) {
        let mut q = p;
        let name = move_name(&p, mv);
        q.make(mv);
        // Hamleden sonra sıra rakipte; tehdit yine bana yönelik.
        let after = best_threat(&q, opp, me);
        let dist = q.distance(me).map(|v| v as i64).unwrap_or(99);
        rows.push((after, dist, name));
    }
    rows.sort();
    println!("\ntehdidi en aza indiren 8 hamle (tehdit, sonraki yolum, hamle):");
    for (t, d, n) in rows.iter().take(8) {
        println!("   tehdit {t}   yol {d}   {n}");
    }
    let best = rows.first().map(|r| r.0).unwrap_or(base);
    println!(
        "\n=> en iyi savunma tehdidi {} -> {}  ({})",
        base,
        best,
        if best < base { "SAVUNMA MUMKUN" } else { "savunma yok" }
    );
}

