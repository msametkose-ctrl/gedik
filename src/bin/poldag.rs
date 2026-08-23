//! Ağın politika dağılımı ne kadar sivri? Elle yazılmışla yan yana.
use gedik::board::Position;
use gedik::heuristics::{candidate_moves_filtered, move_priors};
use gedik::notation::{move_name, parse_move};

fn softmax(z: &[f32], t: f32) -> Vec<f32> {
    let mx = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut o: Vec<f32> = z.iter().map(|v| ((v - mx) / t).exp()).collect();
    let s: f32 = o.iter().sum();
    for v in o.iter_mut() { *v /= s; }
    o
}
fn entropi(p: &[f32]) -> f32 {
    -p.iter().filter(|&&x| x > 1e-9).map(|&x| x * x.ln()).sum::<f32>()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut pos = Position::start();
    for t in a[1].split([',', ' ']).filter(|t| !t.trim().is_empty()) {
        pos.make(parse_move(&pos, t.trim()).expect(t));
    }
    let net = gedik::nn::net().expect("ag yok");
    let mv = candidate_moves_filtered(&pos, true);
    let mut sc = Vec::new();
    let lg = net.policy(&pos, &mv, &mut sc).expect("politika yok");
    let el = move_priors(&pos, &mv);

    println!("{} aday hamle", mv.len());
    for t in [1.0f32, 2.0, 3.0, 5.0] {
        let p = softmax(&lg, t);
        let mut ix: Vec<usize> = (0..p.len()).collect();
        ix.sort_by(|&i, &j| p[j].total_cmp(&p[i]));
        let ilk: Vec<String> = ix.iter().take(4)
            .map(|&i| format!("{}:{:.3}", move_name(&pos, mv[i]), p[i])).collect();
        println!("  ag T={t:<4} en yuksek {:.3}  entropi {:.2}  {}", p[ix[0]], entropi(&p), ilk.join(" "));
    }
    let mut ix: Vec<usize> = (0..el.len()).collect();
    ix.sort_by(|&i, &j| el[j].total_cmp(&el[i]));
    let ilk: Vec<String> = ix.iter().take(4)
        .map(|&i| format!("{}:{:.3}", move_name(&pos, mv[i]), el[i])).collect();
    println!("  elle yazilmis en yuksek {:.3}  entropi {:.2}  {}", el[ix[0]], entropi(&el), ilk.join(" "));
    println!("  (duz dagilim entropisi {:.2})", (mv.len() as f32).ln());
}

