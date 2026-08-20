//! Perft — kural motorunun doğruluk testi için düğüm sayacı.
//!
//! Quoridor'da satrancın aksine referans bir perft tablosu yok; bu sayıları
//! üretip yayınlamak ekosistemdeki gerçek boşluklardan biri. Burada üretilen
//! değerler `tests/perft.rs` içinde regresyon olarak kilitlenir.

use crate::board::Position;
use crate::moves::Move;

pub fn perft(pos: &mut Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    if pos.winner().is_some() {
        return 0;
    }
    let moves = pos.legal_moves();
    if depth == 1 {
        return moves.len() as u64;
    }
    let mut total = 0u64;
    for mv in moves {
        let u = pos.make(mv);
        total += perft(pos, depth - 1);
        pos.unmake(u);
    }
    total
}

/// Kök hamle başına düğüm dağılımı — bug ayıklamanın en hızlı yolu.
pub fn divide(pos: &mut Position, depth: u32) -> Vec<(Move, u64)> {
    let mut out = Vec::new();
    for mv in pos.legal_moves() {
        let u = pos.make(mv);
        let n = if depth <= 1 { 1 } else { perft(pos, depth - 1) };
        pos.unmake(u);
        out.push((mv, n));
    }
    out
}
