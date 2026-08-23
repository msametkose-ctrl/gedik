//! Ağın verdiği değeri basar — Python ile Rust'ın aynı sayıyı ürettiğini
//! doğrulamak için. Girdi: satır başına bir FEN (stdin).
use gedik::notation::from_fen;
use std::io::BufRead;
fn main() {
    let Some(net) = gedik::nn::net() else {
        eprintln!("ag.bin yok");
        std::process::exit(2);
    };
    let mut scratch = Vec::new();
    for line in std::io::stdin().lock().lines() {
        let Ok(l) = line else { break };
        let l = l.trim();
        if l.is_empty() { continue }
        match from_fen(l) {
            Some(p) => println!("{:.8}", net.value(&p, &mut scratch)),
            None => println!("nan"),
        }
    }
}

