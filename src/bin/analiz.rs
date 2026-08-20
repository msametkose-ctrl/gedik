//! Bir hamle listesini oynatıp tahtayı ve motorun en iyi hamlelerini basar.
use gedik::board::Position;
use gedik::engine::Engine;
use gedik::notation::{board_string, move_name, parse_move};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let moves = a.get(1).cloned().unwrap_or_default();
    let spec = a.get(2).cloned().unwrap_or_else(|| "mcts:3000ms".into());
    let mut pos = Position::start();
    for t in moves.split([',', ' ']).filter(|t| !t.trim().is_empty()) {
        let mv = parse_move(&pos, t.trim()).expect(t);
        pos.make(mv);
    }
    println!("{}", board_string(&pos));
    println!(
        "sira {}  yol A={:?} B={:?}  duvar {:?}",
        pos.side,
        pos.distance(0),
        pos.distance(1),
        pos.walls
    );
    if spec == "-" {
        return;
    }
    let mut e = Engine::parse(&spec, 0x1234_5678);
    let c = e.choose(&pos);
    println!(
        "motor: {} ({}) is={}",
        c.mv.map(|m| move_name(&pos, m)).unwrap_or_default(),
        c.info,
        c.work
    );
    for (m, v, wr) in c.top.iter().take(10) {
        println!(
            "   {:>5}  ziyaret {:>7}  wr {:.4}",
            move_name(&pos, *m),
            v,
            wr
        );
    }
}
