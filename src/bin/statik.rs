//! Statik değerlendirmenin ayırt etme gücü: en iyi 24 hamle arasındaki fark.
use gedik::board::Position;
use gedik::heuristics::{candidate_moves_filtered, value_to_move};
use gedik::notation::parse_move;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut pos = Position::start();
    let mut n = 0usize;
    println!(
        "{:>4} {:>10} {:>8} {:>8}",
        "ply", "yayilim", "A yol", "B yol"
    );
    for t in a[1].split([',', ' ']).filter(|t| !t.trim().is_empty()) {
        let mut vals: Vec<f32> = Vec::new();
        for mv in candidate_moves_filtered(&pos, true) {
            let mut q = pos;
            q.make(mv);
            vals.push(1.0 - value_to_move(&q));
        }
        vals.sort_by(|a, b| b.total_cmp(a));
        let k = vals.len().min(24);
        println!(
            "{:>4} {:>10.6} {:>8?} {:>8?}",
            n,
            vals[0] - vals[k - 1],
            pos.distance(0),
            pos.distance(1)
        );
        pos.make(parse_move(&pos, t.trim()).unwrap());
        n += 1;
    }
}
