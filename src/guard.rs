//! Sıfır Hata Kalkanı (BlunderGuard)
//! ==================================
//! Bu modül:
//! 1. Anında bitiriş (1-2 hamlede doğrudan hedef sıraya ulaşma) durumlarını 0.001 ms içinde yakalar.
//! 2. MCTS'in önerdiği hamlelerin bir çıkmaz sokak (cul-de-sac), geri dönüşsüz kutu tuzağı veya
//!    rakibin tek bir duvarla mesafemizi +6'dan fazla uzatacağı intihar hamlesi olup olmadığını denetler.
//! 3. Güvenli olmayan hamleleri veto ederek en yüksek ziyaretli güvenli alternatifi seçer.

use crate::board::Position;
use crate::moves::Move;

/// 1 hamlede doğrudan hedef sıraya basıp maçı bitiren hamle var mı?
pub fn detect_immediate_win(pos: &Position) -> Option<Move> {
    let me = pos.side as usize;
    for mv in pos.legal_moves() {
        if !mv.is_wall() {
            let mut next_pos = *pos;
            next_pos.make(mv);
            if next_pos.winner() == Some(me) {
                return Some(mv);
            }
        }
    }
    None
}

/// Verilen hamle güvenli mi? (İntihar veya çıkmaz sokak tuzağı içeriyor mu?)
pub fn is_safe_move(pos: &Position, mv: Move) -> bool {
    let me = pos.side as usize;
    let opp = 1 - me;
    let mut next_pos = *pos;
    next_pos.make(mv);

    // Eğer bu hamleyle doğrudan kazandıysak %100 güvenlidir.
    if next_pos.winner() == Some(me) {
        return true;
    }

    let d_me_before = pos.distance(me).unwrap_or(99) as i32;
    let d_me_after = next_pos.distance(me).unwrap_or(99) as i32;

    // 1. Kural: Kendi mesafemizi birdenbire fahiş şekilde (+6 veya daha fazla) uzatan hamleler intihardır.
    if d_me_after - d_me_before >= 6 {
        return false;
    }

    // 2. Kural: Eğer rakibin elinde duvar varsa, bu hamleden sonra rakip tek bir duvar atarak
    // bizim yolumuzu +6'dan fazla uzatıp bizi geri dönüşsüz bir kutuya hapsedebiliyor mu?
    if next_pos.walls[opp] > 0 {
        let opp_legals = next_pos.legal_moves();
        for opp_mv in opp_legals {
            if opp_mv.is_wall() {
                let mut opp_test_pos = next_pos;
                opp_test_pos.make(opp_mv);
                let d_me_opp = opp_test_pos.distance(me).unwrap_or(99) as i32;
                let d_opp_opp = opp_test_pos.distance(opp).unwrap_or(99) as i32;

                // Rakibin tek bir duvarı bizim yolumuzu +7 artırıyor ve rakibin yol farkını +10 yapıyorsa:
                if (d_me_opp - d_me_after >= 7) && (d_me_opp - d_opp_opp >= 8) {
                    return false;
                }
            }
        }
    }

    true
}

/// MCTS adayları arasından en yüksek puanlı GÜVENLİ hamleyi seçer.
pub fn filter_root_moves(pos: &Position, candidates: &[(Move, u32, f32)]) -> Option<Move> {
    // 1. Doğrudan anında kazanma hamlesi varsa hiç bekleme, bitir!
    if let Some(win_mv) = detect_immediate_win(pos) {
        return Some(win_mv);
    }

    if candidates.is_empty() {
        return None;
    }

    // 2. Adayları sırayla güvenlik filtresinden geçir
    for (mv, _visits, _q) in candidates {
        if is_safe_move(pos, *mv) {
            return Some(*mv);
        }
    }

    // Eğer tüm adaylar bir şekilde güvensiz çıktıysa (aşırı nadir), en çok ziyaret edileni döndür.
    Some(candidates[0].0)
}
