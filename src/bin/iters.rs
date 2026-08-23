//! Verilen motorun 3 saniyede kaç iterasyon yaptığını ölçer.
use gedik::board::Position;
use gedik::engine::Engine;

fn main() {
    let spec = std::env::args().nth(1).unwrap_or("mcts:3000ms".into());
    let mut e = Engine::parse(&spec, 42);
    let p = Position::start();
    let c = e.choose(&p);
    println!("{spec:<34} {:>10} iterasyon   {}", c.work, c.info);
}

