//! Verilen hamle listesinden itibaren iki motoru oynatır. Kim kazanıyor?
use gedik::board::Position;
use gedik::engine::Engine;
use gedik::notation::{move_name, parse_move};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let moves = a[1].clone();
    let ea = a[2].clone();
    let eb = a[3].clone();
    let n: u32 = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(1);
    let forced = a.get(5).cloned().unwrap_or_default();

    let mut base = Position::start();
    for t in moves.split([',', ' ']).filter(|t| !t.trim().is_empty()) {
        let mv = parse_move(&base, t.trim()).expect(t);
        base.make(mv);
    }
    if !forced.is_empty() {
        for t in forced.split([',', ' ']).filter(|t| !t.trim().is_empty()) {
            let mv = parse_move(&base, t.trim()).expect(t);
            base.make(mv);
        }
    }

    let mut wins = [0u32; 2];
    for g in 0..n {
        let mut pos = base;
        let mut line: Vec<String> = Vec::new();
        let seed = 0x1234_5678u64.wrapping_mul(g as u64 + 1) | 1;
        let mut engines = [Engine::parse(&ea, seed), Engine::parse(&eb, seed ^ 0xabcd)];
        while pos.winner().is_none() && pos.ply < 300 {
            let s = pos.side as usize;
            let c = engines[s].choose(&pos);
            let Some(mv) = c.mv else { break };
            line.push(move_name(&pos, mv));
            pos.make(mv);
        }
        let w = pos.winner().unwrap_or(2);
        if w < 2 { wins[w] += 1; }
        println!("oyun {g}: kazanan {} ({} hamle)  {}", if w==0 {"A"} else if w==1 {"B"} else {"-"}, line.len(), line.join(" "));
    }
    println!("\nA {} - {} B", wins[0], wins[1]);
}

