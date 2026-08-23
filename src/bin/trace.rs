//! Tek bir rollout'u hamle hamle izler.

use gedik::board::{goal_row, Position};
use gedik::heuristics::{playout_move, race_winner, Rng};
use gedik::notation::{cell_name, move_name};

fn main() {
    let wall_prob: f32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let mut pos = Position::start();
    let mut rng = Rng::new(12345);

    println!(
        "başlangıç: A {} (hedef satır {}) d={:?} | B {} (hedef satır {}) d={:?}",
        cell_name(pos.pawn[0] as usize),
        goal_row(0),
        pos.distance(0),
        cell_name(pos.pawn[1] as usize),
        goal_row(1),
        pos.distance(1),
    );

    for step in 0..60 {
        if let Some(w) = pos.winner() {
            println!("--> winner() = {} ({})", w, if w == 0 { "A" } else { "B" });
            return;
        }
        if pos.walls[0] == 0 && pos.walls[1] == 0 {
            println!("--> duvarlar bitti, race_winner = {}", race_winner(&pos));
            return;
        }
        let Some(mv) = playout_move(&pos, &mut rng, wall_prob) else {
            println!("--> hamle yok");
            return;
        };
        let who = if pos.side == 0 { "A" } else { "B" };
        let name = move_name(&pos, mv);
        pos.make(mv);
        println!(
            "{step:>3}. {who} {name:<5}  A {} d={:?} | B {} d={:?}",
            cell_name(pos.pawn[0] as usize),
            pos.distance(0),
            cell_name(pos.pawn[1] as usize),
            pos.distance(1),
        );
    }
    println!("60 adımda bitmedi");
}

