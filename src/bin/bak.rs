//! Tek bir pozisyonda motorun ne dusundugunu gosterir.
//!
//! Kaybedilen bir oyunu incelerken "motor burada neyi gormedi" sorusunun
//! cevabi genelde kok cocuklarinin degerlerinde duruyor: hamleler birbirinden
//! ayrismiyorsa (hepsi ayni yuzde) sorun aramada degil degerlendirmede.
//!
//!   bak "e2 e8 e3 e7 ..." "mcts:200000:t=4,ag=tur2\ag64.bin"
use gedik::board::Position;
use gedik::engine::Engine;
use gedik::notation::{move_name, parse_move};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!("kullanim: bak \"<hamleler>\" \"<motor>\" [gosterilecek hamle sayisi]");
        return;
    }
    let n: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);
    let mut pos = Position::start();
    for t in a[1].split_whitespace() {
        match parse_move(&pos, t) {
            Some(mv) => {
                pos.make(mv);
            }
            None => {
                eprintln!("gecersiz hamle: {t}");
                return;
            }
        }
    }
    let mut e = Engine::parse(&a[2], 987_654_321);
    let c = e.choose(&pos);
    println!("sira: {}  ({})", pos.side, c.info);
    let toplam: u32 = c.top.iter().map(|(_, v, _)| *v).sum::<u32>().max(1);
    for (m, v, w) in c.top.iter().take(n) {
        println!(
            "  {:<6} {:>6.1}%  {:>8} ziyaret  (%{:.0})",
            move_name(&pos, *m),
            w * 100.0,
            v,
            100.0 * *v as f32 / toplam as f32
        );
    }
}

