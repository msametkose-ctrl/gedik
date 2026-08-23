//! best_threat'in maliyetini ölçer — değer fonksiyonuna eklenebilir mi?
use gedik::heuristics::{best_threat, value_to_move};
use gedik::notation::from_fen;
use std::time::Instant;

fn main() {
    let fens = [
        "0/0/76/4/10/10/0/0",
        "a11040000150000/8200820000000/66/48/7/1/0/30",
        "1000000000/400000/67/31/8/10/0/6",
    ];
    for f in fens {
        let p = from_fen(f).unwrap();
        let n = 200_000;
        let t = Instant::now();
        let mut acc = 0u32;
        for _ in 0..n { acc += best_threat(&p, 1, 0); }
        let th = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let mut acc2 = 0.0f32;
        for _ in 0..n { acc2 += value_to_move(&p); }
        let vv = t.elapsed().as_secs_f64();
        println!("{:<46} tehdit {:>9.0}/s   deger {:>9.0}/s   oran {:.1}x  (sag {} {:.1})",
            f, n as f64/th, n as f64/vv, th/vv, acc/n, acc2);
    }
}

