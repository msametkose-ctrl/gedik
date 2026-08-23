//! Politika basi cikti kontrolu: her FEN icin ilk N legal hamlenin logiti.
use gedik::heuristics::candidate_moves_filtered;
use gedik::notation::from_fen;
use std::io::BufRead;
fn main() {
    let Some(net) = gedik::nn::net() else { eprintln!("ag yok"); std::process::exit(2) };
    if !net.has_policy() { eprintln!("politika basi yok"); std::process::exit(3) }
    let mut sc = Vec::new();
    for line in std::io::stdin().lock().lines() {
        let Ok(l) = line else { break };
        let l = l.trim();
        if l.is_empty() { continue }
        let Some(p) = from_fen(l) else { println!("nan"); continue };
        let mv = candidate_moves_filtered(&p, true);
        let lg = net.policy(&p, &mv, &mut sc).unwrap();
        // aksiyon:logit ciftleri
        let s: Vec<String> = mv.iter().zip(&lg).take(8)
            .map(|(m, z)| format!("{}:{:.6}", m.action_id(), z)).collect();
        println!("{}", s.join(" "));
    }
}

