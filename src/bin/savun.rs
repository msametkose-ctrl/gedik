//! Savunma analizi: her hamlemizden sonra rakip **tek duvarla** yolumuzu en
//! fazla ne kadar uzatabiliyor?
//!
//! Yarışta öndeyken asıl soru "şu an yolum kaç" değil, "rakip bir hamlede
//! yolumu kaç yapabilir". Motorun değerlendirmesi birinciyi görüyor.
use gedik::bitboard::WSLOTS;
use gedik::board::Position;
use gedik::heuristics::candidate_moves_filtered;
use gedik::moves::Move;
use gedik::notation::{move_name, parse_move};

fn dist(p: &Position, s: usize) -> u32 {
    p.distance(s).unwrap_or(999)
}

/// Rakip tek duvarla bizim yolumuzu en fazla kaça çıkarır (ve hangi duvarla).
fn worst_case(p: &Position, me: usize) -> (u32, String) {
    let opp = 1 - me;
    let mut worst = dist(p, me);
    let mut ad = "-".to_string();
    if p.walls[opp] == 0 {
        return (worst, ad);
    }
    for horizontal in [true, false] {
        for slot in 0..WSLOTS {
            if !p.wall_legal(slot, horizontal) {
                continue;
            }
            let mv = if horizontal { Move::hwall(slot) } else { Move::vwall(slot) };
            let mut q = *p;
            if q.side as usize != opp {
                continue;
            }
            if !q.is_legal(mv) {
                continue;
            }
            let nm = move_name(&q, mv);
            q.make(mv);
            let d = dist(&q, me);
            if d > worst {
                worst = d;
                ad = nm;
            }
        }
    }
    (worst, ad)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut pos = Position::start();
    for t in a[1].split([',', ' ']).filter(|t| !t.trim().is_empty()) {
        pos.make(parse_move(&pos, t.trim()).expect(t));
    }
    let me = pos.side as usize;
    println!("sira {me}  yol {} / {}  duvar {:?}", dist(&pos, 0), dist(&pos, 1), pos.walls);
    let (w0, a0) = worst_case(&pos, me);
    println!("hamle yapmadan once: yolum {}, rakip tek duvarla {} yapabilir ({a0})\n", dist(&pos, me), w0);

    let mut rows: Vec<(u32, u32, u32, String, String)> = Vec::new();
    for mv in candidate_moves_filtered(&pos, true) {
        let nm = move_name(&pos, mv);
        let mut q = pos;
        q.make(mv);
        let (w, ad) = worst_case(&q, me);
        rows.push((w, dist(&q, me), dist(&q, 1 - me), nm, ad));
    }
    // En kotu durumu en dusuk olan once; esitlikte kendi yolu kisa olan.
    rows.sort_by(|x, y| x.0.cmp(&y.0).then(x.1.cmp(&y.1)));
    println!("{:>6} {:>8} {:>8} {:>10}  {}", "hamle", "yolum", "rakip", "EN KOTU", "rakibin en iyi duvari");
    for (w, d, od, nm, ad) in rows.iter().take(14) {
        println!("{nm:>6} {d:>8} {od:>8} {w:>10}  {ad}");
    }
}
