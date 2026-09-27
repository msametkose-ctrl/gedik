//! Öğrenilmiş ağırlıkları sağlık kontrollerinden geçirir.
use gedik::heuristics::{value_with, WEIGHTS_HAND, WEIGHTS_LEARNED};
use gedik::mcts::{Config, Mcts};
use gedik::notation::from_fen;

fn main() {
    let cases = [(
        "18 vs 8, A'nın 7 duvarı (umutsuz yarış)",
        "a11040000150000/8200820000000/66/48/7/1/0/30",
        0.25f32,
        0.35f32,
    )];
    for (ad, fen, vmax, wmax) in cases {
        let p = from_fen(fen).expect("fen");
        println!("\n=== {ad} ===");
        println!(
            "A yol {:?}  B yol {:?}  duvar {:?}  sira {}",
            p.distance(0),
            p.distance(1),
            p.walls,
            p.side
        );
        for (etiket, w) in [("elle", WEIGHTS_HAND), ("ogrenilmis", WEIGHTS_LEARNED)] {
            let v = value_with(&p, &w);
            let mut m = Mcts::new(3);
            let mut c = Config::default();
            c.weights = w;
            c.apply(&mut m);
            let (_, st) = m.search_rollouts(&p, 20_000);
            let mut m2 = Mcts::new(3);
            let mut c2 = Config::default();
            c2.weights = w;
            c2.apply(&mut m2);
            let (_, st2) = m2.search_rollouts(&p, 200_000);
            println!(
                "  {etiket:>11}: statik {v:.4} {}   arama20k {:.4} {}   arama200k {:.4} {}",
                if v < vmax { "OK " } else { "HATA" },
                st.win_rate,
                if st.win_rate < wmax { "OK " } else { "HATA" },
                st2.win_rate,
                if st2.win_rate < wmax { "OK " } else { "HATA" },
            );
        }
    }
}
