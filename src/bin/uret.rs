//! Sinir ağı eğitim verisi üretir (v2).
//!
//! v1'den farkı: satırda **ham pozisyon** (FEN) ve **aramanın hamle dağılımı**
//! var. 12 özellikli formülün çıktısını kaydetmek modeli o formülün
//! görebildiği şeylerle sınırlıyordu; ham pozisyondan ağ duvarların *nerede*
//! olduğunu da öğrenebilir — mevcut değerlendirmenin tamamen kör olduğu bilgi.
//!
//! Satır biçimi:
//!   fen;label;ply;pi
//! `label` = bu pozisyonda sırası gelen oyuncu oyunu kazandı mı (0/1)
//! `pi`    = kökteki ziyaret dağılımı, "aksiyon:oran" çiftleri boşlukla ayrık
//!
//! Kullanım: gendata2 <oyun> <iterasyon> <seed>
use gedik::board::Position;
use gedik::engine::Engine;
use gedik::heuristics::{candidate_moves, Rng};
use gedik::notation::to_fen;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let games: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    // 60 binden sonra arama gücü artmıyor (ölçüldü), o yüzden varsayılan 40k:
    // veri kalitesi/hız dengesinin tepe noktası.
    // 20 bin: olctugumuz uzerine secildi. 40 binde saatte ~3.4M satir, 20
    // binde ~9M cikiyor; buna karsilik 20 bin ile 200 bin arasindaki guc
    // farki sadece +131 Elo, yani etiket kalitesi cok az dusuyor. Veri
    // miktari bu boyuttaki bir ag icin daha degerli.
    let iters: u32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let seed: u64 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    // 4. arguman: sinir agiyla mi oynasin (0/1). Varsayilan 0.
    //
    // Neden varsayilan kapali: agi, agsiz motorun oyunlarindan ogretip sonra
    // agsiz motora karsi olcuyoruz. Veri uretiminde de ag kullanilsaydi
    // "ag ne kattigi" sorusunun cevabi karisirdi. Sonraki turda 1 verilerek
    // bootstrap dongusu donduruluyor.
    let nn: u32 = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);

    println!("fen;label;ply;pi");
    let mut rng = Rng::new(seed);

    for g in 0..games {
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

        // Dengesiz eşleşmeler: veride "açıkça kaybedilmiş" pozisyon da olsun.
        let weak = iters.max(8) / 8;
        let (it0, it1) = match g % 3 {
            0 => (iters, iters),
            1 => (weak, iters),
            _ => (iters, weak),
        };
        // Kök gürültüsü: aynı açılıştan hep aynı oyun çıkmasın, ve politika
        // hedefi tek bir hamleye çakılmasın.
        let mut e0 = Engine::parse(&format!("mcts:{it0}:noise=0.15,nn={nn}"), seed + g as u64 * 31 + 1);
        let mut e1 = Engine::parse(&format!("mcts:{it1}:noise=0.15,nn={nn}"), seed + g as u64 * 57 + 2);
        let mut rows: Vec<(String, usize)> = Vec::new();

        while pos.winner().is_none() && pos.ply < 400 {
            let strong = if pos.side == 0 { it0 } else { it1 } == iters;
            let e = if pos.side == 0 { &mut e0 } else { &mut e1 };
            let c = e.choose(&pos);
            let Some(mv) = c.mv else { break };

            // Politika hedefini sadece güçlü taraftan al: zayıf aramanın
            // dağılımını öğretmek modeli bozar.
            if strong {
                let total: u32 = c.top.iter().map(|t| t.1).sum();
                if total > 0 {
                    let pi: Vec<String> = c
                        .top
                        .iter()
                        .filter(|t| t.1 > 0)
                        .map(|t| format!("{}:{:.4}", t.0.action_id(), t.1 as f32 / total as f32))
                        .collect();
                    rows.push((format!("{};{};{}", to_fen(&pos), pos.ply, pi.join(" ")), pos.side as usize));
                }
            }
            pos.make(mv);
        }

        let Some(w) = pos.winner() else { continue };
        for (line, side) in rows {
            let mut it = line.splitn(2, ';');
            let fen = it.next().unwrap();
            let rest = it.next().unwrap();
            println!("{fen};{};{}", u8::from(w == side), rest);
        }
        eprintln!("oyun {}/{} bitti ({} yarım hamle)", g + 1, games, pos.ply);
    }
}
