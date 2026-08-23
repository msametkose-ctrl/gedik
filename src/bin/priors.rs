//! Bir pozisyondaki policy prior'larını dökümler — arama neye bakıyor?
//!
//! `priors [fen]`

use gedik::board::Position;
use gedik::heuristics::{candidate_moves, move_priors};
use gedik::notation::{board_string, from_fen, move_name};

fn main() {
    let fen = std::env::args().nth(1);
    let p = match fen {
        Some(f) => from_fen(&f).expect("geçersiz fen"),
        None => Position::start(),
    };
    println!("{}", board_string(&p));
    let moves = candidate_moves(&p);
    let pr = move_priors(&p, &moves);
    let mut v: Vec<(String, f32)> = moves
        .iter()
        .zip(&pr)
        .map(|(&m, &x)| (move_name(&p, m), x))
        .collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("{} aday hamle, en yüksek prior'lar:", moves.len());
    for (n, x) in v.iter().take(12) {
        let bar = "#".repeat((x * 60.0).round().max(0.0) as usize);
        println!("  {n:>5}  {x:.5}  {bar}");
    }
}

