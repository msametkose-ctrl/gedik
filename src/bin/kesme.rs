//! "Yolumu kesebilecek KAÇ legal duvar var?"
//!
//! a8h'nin sırrı bu: kendi duvarını, rakibin a8v koymak için ihtiyaç duyduğu
//! **merkeze** koyuyor. İki duvar aynı merkezi paylaşamadığı için rakibin tek
//! kaynağı yok oluyor. Değerlendirme bunu göremiyor — o sadece "yolum kaç"
//! diyor, "yolum kesilebilir mi" demiyor.
//!
//! Ucuz olmasi lazim: yalnizca en kisa yolun kenarlarina degen duvarlara
//! bakiyoruz, tahtanin tamamina degil.
use gedik::bitboard::{Masks, CELLS, WDIM, WSLOTS};
use gedik::board::{goal_row, Position};
use gedik::moves::Move;
use gedik::notation::{move_name, parse_move};

/// Yolumu gercekten uzatan legal duvarlarin sayisi ve en kotu etkisi.
fn kesiciler(p: &Position, side: usize) -> (u32, u32) {
    let opp = 1 - side;
    let m = p.masks();
    let d0 = m.distance_to_row(p.pawn[side] as usize, goal_row(side)).unwrap_or(0);
    if p.walls[opp] == 0 {
        return (0, d0);
    }
    let mut adet = 0u32;
    let mut en_kotu = d0;
    for horizontal in [true, false] {
        for slot in 0..WSLOTS {
            if !p.wall_legal(slot, horizontal) {
                continue;
            }
            let b = 1u64 << slot;
            let nm = if horizontal { Masks::new(p.h | b, p.v) } else { Masks::new(p.h, p.v | b) };
            let d = nm.distance_to_row(p.pawn[side] as usize, goal_row(side)).unwrap_or(d0);
            if d > d0 {
                adet += 1;
                if d > en_kotu { en_kotu = d; }
            }
        }
    }
    (adet, en_kotu)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut pos = Position::start();
    for t in a[1].split([',', ' ']).filter(|t| !t.trim().is_empty()) {
        pos.make(parse_move(&pos, t.trim()).expect(t));
    }
    let me = pos.side as usize;
    println!("{:>6} {:>6} {:>10} {:>10}", "hamle", "yolum", "kesici", "en kotu");
    for s in a[2].split(',').filter(|s| !s.trim().is_empty()) {
        let Some(mv) = parse_move(&pos, s.trim()) else { println!("{s}: illegal"); continue };
        let mut q = pos;
        let _ = move_name(&pos, mv);
        q.make(mv);
        let d = q.distance(me).unwrap_or(0);
        let (adet, kotu) = kesiciler(&q, me);
        println!("{:>6} {d:>6} {adet:>10} {kotu:>10}", s.trim());
    }
    // Hiz olcumu
    let t = std::time::Instant::now();
    let n = 20000;
    let mut acc = 0u32;
    for _ in 0..n { acc = acc.wrapping_add(kesiciler(&pos, me).0); }
    println!("\nhiz: {:.2} us/cagri  (acc {acc})", t.elapsed().as_secs_f64() * 1e6 / n as f64);
    let _ = (CELLS, WDIM, Move::hwall(0));
}

