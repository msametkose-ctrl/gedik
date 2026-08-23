//! Kökteki hamle değerlerinin yayılımı: arama gerçekten ayırt ediyor mu?
use gedik::board::Position;
use gedik::heuristics::value_to_move;
use gedik::mcts::{Config, Mcts};
use gedik::notation::parse_move;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut pos = Position::start();
    let mut hist: Vec<Position> = Vec::new();
    for t in a[1].split([',', ' ']).filter(|t| !t.trim().is_empty()) {
        hist.push(pos);
        pos.make(parse_move(&pos, t.trim()).unwrap());
    }
    println!("{:>4} {:>10} {:>10} {:>10} {:>10}", "ply", "statik", "20k", "200ms", "3000ms");
    for (i, p) in hist.iter().enumerate().filter(|(i, _)| i % 2 == 1 && *i >= 7) {
        // Statik: çocukların değer yayılımı (1 ply, arama yok)
        let mut vals: Vec<f32> = Vec::new();
        for mv in gedik::heuristics::candidate_moves_filtered(p, true) {
            let mut q = *p;
            q.make(mv);
            vals.push(1.0 - value_to_move(&q));
        }
        vals.sort_by(|a, b| b.total_cmp(a));
        let stat = vals[0] - vals[vals.len().min(24) - 1];

        let mut row = vec![format!("{stat:.4}")];
        for (it, ms) in [(20_000u32, 0u64), (0, 200), (0, 3000)] {
            let mut m = Mcts::new(0x1234);
            Config::default().apply(&mut m);
            let (_, st) = if ms > 0 { m.search_time(p, ms) } else { m.search_rollouts(p, it) };
            let sp = st.top.first().map(|t| t.2).unwrap_or(0.0)
                - st.top.last().map(|t| t.2).unwrap_or(0.0);
            row.push(format!("{sp:.4}"));
        }
        println!("{:>4} {:>10} {:>10} {:>10} {:>10}", i, row[0], row[1], row[2], row[3]);
    }
}

