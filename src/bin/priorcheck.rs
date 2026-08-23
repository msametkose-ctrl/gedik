use gedik::board::Position;
use gedik::heuristics::{candidate_moves, move_priors};
use gedik::notation::move_name;
fn main() {
    let p = Position::start();
    let moves = candidate_moves(&p);
    let pr = move_priors(&p, &moves);
    let mut v: Vec<(String, f32)> = moves.iter().zip(&pr).map(|(&m, &x)| (move_name(&p, m), x)).collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (n, x) in v.iter().take(10) { println!("{n:>5}  {x:.5}"); }
    println!("...");
    for name in ["e2", "e8h", "e2h", "d8h", "a5h"] {
        if let Some((n, x)) = v.iter().find(|(n, _)| n == name) { println!("{n:>5}  {x:.5}"); }
        else { println!("{name:>5}  (aday degil)"); }
    }
}

