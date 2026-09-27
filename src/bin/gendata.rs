//! Self-play oynayıp değer fonksiyonu için eğitim verisi üretir.
//!
//! Her pozisyon için özellikler + oyunun gerçek sonucu (sıradaki oyuncu
//! kazandı mı) yazılır. Çıktı CSV, `tools/fit.py` ile lojistik regresyona
//! sokulur ve ağırlıklar `heuristics.rs` içine gömülür.
//!
//! Kullanım: `gendata <oyun> <iterasyon> <seed> > veri.csv`

use gedik::board::Position;
use gedik::engine::Engine;
use gedik::heuristics::{candidate_moves, features, Rng, NUM_FEATURES};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let games: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let iters: u32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(3000);
    let seed: u64 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);

    let mut head = String::new();
    for i in 0..NUM_FEATURES {
        head.push_str(&format!("f{i},"));
    }
    head.push_str("side,ply,label");
    println!("{head}");

    let mut rng = Rng::new(seed);
    for g in 0..games {
        // Çeşitlilik için rastgele açılış: derinliği de değiştir.
        let open_plies = 2 + (g % 9) * 2;
        let mut pos = Position::start();
        let mut ok = true;
        for _ in 0..open_plies {
            let mv = candidate_moves(&pos);
            if mv.is_empty() || pos.winner().is_some() {
                ok = false;
                break;
            }
            pos.make(mv[rng.below(mv.len())]);
        }
        if !ok || pos.winner().is_some() {
            continue;
        }

        // Her üçüncü oyunda taraflardan biri kasten zayıf oynasın.
        //
        // İlk eğitim setinin sorunu buydu: iki eşit motor birbirine karşı
        // oynayınca oyunlar hep dengeli gidiyor ve "açıkça kaybedilmiş"
        // pozisyonlar veride neredeyse hiç görünmüyor. Model de o bölgeyi
        // öğrenemiyor — 18'e 8 geride olan pozisyona %92 demesinin bir sebebi
        // buydu. Dengesiz eşleşmeler tam o bölgeyi dolduruyor.
        let weak = iters.max(8) / 8;
        let (it0, it1) = match g % 3 {
            0 => (iters, iters),
            1 => (weak, iters),
            _ => (iters, weak),
        };
        let mut e0 = Engine::parse(&format!("mcts:{it0}"), seed + g as u64 * 31 + 1);
        let mut e1 = Engine::parse(&format!("mcts:{it1}"), seed + g as u64 * 57 + 2);
        let mut rows: Vec<(String, u8)> = Vec::new();

        while pos.winner().is_none() && pos.ply < 400 {
            let f = features(&pos);
            let mut line = String::new();
            for v in f {
                line.push_str(&format!("{v:.5},"));
            }
            line.push_str(&format!("{},{}", pos.side, pos.ply));
            rows.push((line, pos.side));

            let e = if pos.side == 0 { &mut e0 } else { &mut e1 };
            let Some(mv) = e.choose(&pos).mv else { break };
            pos.make(mv);
        }

        let Some(w) = pos.winner() else { continue };
        for (line, side) in rows {
            println!("{line},{}", u8::from(w == side as usize));
        }
        eprintln!("oyun {}/{} bitti ({} yarım hamle)", g + 1, games, pos.ply);
    }
}
