//! best_threat(rakip -> ben): rakip tek duvarla yolumu ne kadar uzatir?
use gedik::board::Position;
use gedik::heuristics::best_threat;
use gedik::notation::parse_move;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut pos = Position::start();
    for t in a[1].split([',', ' ']).filter(|t| !t.trim().is_empty()) {
        pos.make(parse_move(&pos, t.trim()).expect(t));
    }
    let me = pos.side as usize;
    println!(
        "{:>6} {:>6} {:>14} {:>14}",
        "hamle", "yolum", "bana tehdit", "rakibe tehdit"
    );
    for s in a[2].split(',').filter(|s| !s.trim().is_empty()) {
        let Some(mv) = parse_move(&pos, s.trim()) else {
            println!("{s}: illegal");
            continue;
        };
        let mut q = pos;
        q.make(mv);
        let bana = best_threat(&q, 1 - me, me);
        let ona = best_threat(&q, me, 1 - me);
        println!(
            "{:>6} {:>6} {bana:>14} {ona:>14}",
            s.trim(),
            q.distance(me).unwrap_or(0)
        );
    }
    let t = std::time::Instant::now();
    let n = 200000u32;
    let mut acc = 0u32;
    for _ in 0..n {
        acc = acc.wrapping_add(best_threat(&pos, 1 - me, me));
    }
    println!(
        "\nhiz: {:.3} us/cagri (acc {acc})",
        t.elapsed().as_secs_f64() * 1e6 / n as f64
    );
}
