//! Bir pozisyonda iki tarafın tehdit maruziyetini ölçer.
//! `tehdit <fen>`
use gedik::board::Position;
use gedik::heuristics::best_threat;
use gedik::notation::{board_string, from_fen};

fn main() {
    let fen = std::env::args().nth(1).expect("kullanım: tehdit <fen>");
    let p: Position = from_fen(&fen).expect("geçersiz fen");
    println!("{}", board_string(&p));
    println!("A yol {:?}  B yol {:?}", p.distance(0), p.distance(1));
    println!(
        "B'nin A'ya tek duvarla verebileceği en büyük zarar: {}",
        best_threat(&p, 1, 0)
    );
    println!(
        "A'nın B'ye tek duvarla verebileceği en büyük zarar: {}",
        best_threat(&p, 0, 1)
    );
}
