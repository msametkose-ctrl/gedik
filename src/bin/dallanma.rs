//! Aday hamle sayısı: ağacın genişliği ne kadar?
use gedik::board::Position;
use gedik::heuristics::candidate_moves_filtered;
use gedik::notation::parse_move;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut pos = Position::start();
    let mut n = 0;
    println!(
        "{:>4} {:>6} {:>8} {:>8}",
        "ply", "hamle", "aday", "filtresiz"
    );
    for t in a[1].split([',', ' ']).filter(|t| !t.trim().is_empty()) {
        let c = candidate_moves_filtered(&pos, true).len();
        let f = candidate_moves_filtered(&pos, false).len();
        println!("{:>4} {:>6} {:>8} {:>8}", n, t, c, f);
        pos.make(parse_move(&pos, t.trim()).unwrap());
        n += 1;
    }
}
