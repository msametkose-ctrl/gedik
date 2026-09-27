//! Yol dayanıklılığı: en kısa yol koridorunun en dar yeri kaç kare?
//!
//! Fikir: değerlendirme "yolum kaç" diyor ama "yolum kırılabilir mi" demiyor.
//! Koridor tek kareye düşüyorsa rakip tek duvarla oradan kesiyor. Bunu
//! mesafe alanından neredeyse bedavaya çıkarabiliyoruz — alan zaten
//! hesaplanıyor.
use gedik::bitboard::CELLS;
use gedik::board::goal_row;
use gedik::board::Position;
use gedik::notation::parse_move;

/// (yol, en dar koridor genisligi, katman genislikleri)
fn corridor(p: &Position, side: usize) -> (u32, u32, Vec<u32>) {
    let m = p.masks();
    let alan = m.distance_field(goal_row(side));
    let bas = p.pawn[side] as usize;
    let d = alan[bas] as u32;
    if d == 0 || d == u8::MAX as u32 {
        return (d, 0, vec![]);
    }
    // Katman k: piyondan en kisa yolla ulasilabilen, hedefe (d-k) uzakta olan kareler.
    let mut kat: u128 = 1u128 << bas;
    let mut genislik = vec![1u32];
    let mut en_dar = u32::MAX;
    for k in 1..=d {
        let hedef_uz = (d - k) as u8;
        let mut maske: u128 = 0;
        for c in 0..CELLS {
            if alan[c] == hedef_uz {
                maske |= 1u128 << c;
            }
        }
        kat = m.expand(kat) & maske;
        let g = kat.count_ones();
        genislik.push(g);
        if k < d && g < en_dar {
            en_dar = g;
        }
        if kat == 0 {
            break;
        }
    }
    (d, if en_dar == u32::MAX { 1 } else { en_dar }, genislik)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut pos = Position::start();
    for t in a[1].split([',', ' ']).filter(|t| !t.trim().is_empty()) {
        pos.make(parse_move(&pos, t.trim()).expect(t));
    }
    let me = pos.side as usize;
    for mv_s in a[2].split(',').filter(|s| !s.trim().is_empty()) {
        let Some(mv) = parse_move(&pos, mv_s.trim()) else {
            println!("{mv_s}: illegal");
            continue;
        };
        let mut q = pos;
        q.make(mv);
        let (d, dar, kat) = corridor(&q, me);
        println!(
            "{:>5}  yol {d}  en dar koridor {dar}  katmanlar {kat:?}",
            mv_s.trim()
        );
    }
}
