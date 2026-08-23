use gedik::board::Position;
use gedik::notation::*;

fn normalize_move(s: &str) -> String {
    let s = s.trim();
    if (s.starts_with('h') || s.starts_with('v')) && s.len() == 3 {
        let dir = s.chars().next().unwrap();
        let cell = &s[1..];
        format!("{}{}", cell, dir)
    } else {
        s.to_string()
    }
}

fn analyze_game(title: &str, moves_raw: &str) {
    println!("\n========================================================");
    println!("ANALYSIS: {}", title);
    println!("========================================================");
    let mut pos = Position::start();
    let net = gedik::nn::net();
    let mut scratch = Vec::new();

    let tokens: Vec<&str> = moves_raw
        .split_whitespace()
        .filter(|t| !t.ends_with('.'))
        .collect();

    for (turn, m_str) in tokens.iter().enumerate() {
        let p_turn = pos.side as usize;
        let d0 = pos.distance(0).unwrap_or(99) as i32;
        let d1 = pos.distance(1).unwrap_or(99) as i32;
        let val = if let Some(ref n) = net { n.value(&pos, &mut scratch) } else { 0.5 };
        
        let norm_str = normalize_move(m_str);
        let mv = match parse_move(&pos, &norm_str) {
            Some(m) => m,
            None => {
                println!("Turn {:2} (Ply {:2}): Cannot parse/illegal move '{}' (normalized: '{}')", (turn/2)+1, turn + 1, m_str, norm_str);
                let legals = pos.legal_moves();
                print!("  Legal moves ({}): ", legals.len());
                for lm in legals.iter().take(10) {
                    print!("{} ", move_name(&pos, *lm));
                }
                println!();
                break;
            }
        };

        println!(
            "Ply {:2} [M {:2}.{}] (P{} | W:{}/{} | D P0:{:2} P1:{:2} | d1-d0:{:+2} | NN:{:.3}): Played {}",
            turn + 1,
            (turn / 2) + 1,
            if p_turn == 0 { 1 } else { 2 },
            p_turn,
            pos.walls[0], pos.walls[1],
            d0, d1,
            d1 - d0,
            val,
            move_name(&pos, mv)
        );

        pos.make(mv);
        if let Some(w) = pos.winner() {
            println!(">>> GAME OVER at Ply {}: Winner is P{} ({}) <<<", turn + 1, w, if w == 0 { "White/Bottom (P0)" } else { "Black/Top (P1)" });
            break;
        }
    }
}

fn main() {
    let games = [
        ("Game 1", "e2 e8 e3 e7 e4 hd7 he3 e6 vd3 e5 vd1 hf7 vd5 vc6 hd6 vd8 vf4 e6 ve5 e5 hg5 f4 vh6 f5 f4 f6 f5 g6 vg8 g7 f6 h7 g6 h8 g7 i8 h7 i7 h8 i6 h9"),
        ("Game 2", "e2 e8 e3 e7 e4 hd7 hg7 e6 hd2 f6 e5 f5 hf2 ve6 g5 e5 vc3 ve8 vc5 hf6 hh2 e6 hb6 d6 vb7 d7 f5 ve4 ve2 c7 g5 vh6 h5 c8 h6 hf8 hh5 c9 h7 b9 g7 b8 f7 b7 f8 a7 g8 a6 h8 a5 h9"),
        ("Game 3", "e2 e8 e3 e7 e4 e6 e5 e4 e6 e3 he2 he3 hc2 hc3 e7 hd7 f7 hf7 g7 hh7 vf2 ha3 f7 vb1 e7 d3 d7 vc6 d6 hd5 e6 c3 ha8 b3 f6 hf5 g6 b2 h6 a2 h5 b2 g5"),
        ("Game 4", "e2 e8 e3 e7 e4 e6 e5 e4 e6 e3 e7 e2 he1 he2 e8 he8 d8 hc8 c8 d2 hc1 hc2 ha1 vb1 hg1 hg2 b8 ha8 c8 vd8 ve5 e2 c7 f2 d7 g2 e7 ve7 vf7 h2 vg7 i2 e6 i3 e7 i2 e6 i3 e7 i2 e6 i3 e7 i2 e6 i1"),
        ("Game 5", "e2 e8 e3 e7 e4 e6 e5 e4 e6 he6 he2 vd5 hc2 hc4 f6 vf5 f5 ha4 ve1 hh2 f4 vg3 ve7 g4 g5 g3 hf1 vd3 g6 g2 h6 h2 h7 h1"),
        ("Game 6", "e2 e8 e3 e7 e4 e6 e5 e4 e6 e3 e7 e2 he1 he2 e8 he8 d8 hc8 c8 ha8 hc1 hc2 vf1 ha2 he5 vc7 c7 vc5 c6 d2 c5 c2 c4 b2 d4 b1"),
        ("Game 7", "e2 e8 e3 e7 e4 hd7 hd6 f7 hf6 e7 hb6 d7 va7 e7 vg1 f7 hh6 f8 ha4 f9 hc4 he4 f4 vf3 ve3 hg3 f3 vh2 f2 e9 g2 ve8 g3 d9 h3 c9 vb8 c8 h2 c7 h1 b7 i1 b8 i2 b9 i3 a9 i4 a8 h4 a7 h5 a6 g5 b6 f5 c6 e5 d6 d5 e5 c5 f5 c6 g5 b6 h5 a6 i5 a7 i4 a8 i3 a9")
    ];

    for (title, moves) in games {
        analyze_game(title, moves);
    }
}

