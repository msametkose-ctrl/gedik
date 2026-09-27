use gedik::board::Position;
use gedik::mcts::{Config, Mcts};
use gedik::notation::{move_name, parse_move};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut pos = Position::start();
    for t in a[1].split([',', ' ']).filter(|t| !t.trim().is_empty()) {
        pos.make(parse_move(&pos, t.trim()).unwrap());
    }
    for (et, mc) in [
        (2usize, usize::MAX),
        (2, 24),
        (8, usize::MAX),
        (16, 16),
        (16, usize::MAX),
        (32, usize::MAX),
    ] {
        for ms in [500u64, 3000] {
            let mut m = Mcts::new(0x1234);
            let mut c = Config::default();
            c.expand_threshold = et as u32;
            c.max_children = mc;
            c.apply(&mut m);
            let (mv, st) = m.search_time(&pos, ms);
            let spread = st.top.first().map(|t| t.2).unwrap_or(0.0)
                - st.top.last().map(|t| t.2).unwrap_or(0.0);
            println!("et={et:<3} mc={:<6} {ms:>5}ms  it={:>9} dugum={:>9} {}  en iyi {:>5}  ilk-son fark {:.4}",
                if mc==usize::MAX {"-".into()} else {mc.to_string()}, st.rollouts, st.nodes,
                if st.nodes >= 6_000_000 {"TAVAN"} else {"     "},
                mv.map(|m| move_name(&pos, m)).unwrap_or_default(), spread);
        }
    }
}
