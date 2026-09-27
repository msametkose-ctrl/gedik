//! Glendenning cebirsel notasyonu.
//!
//! Sütunlar `a..i` soldan sağa, satırlar `1..9` aşağıdan yukarıya.
//! Piyon hamlesi hedef kare ile yazılır (`e2`), duvar ise çevrelediği dört
//! karenin **kuzeybatısındaki** kare + yön harfiyle (`e6h`, `c3v`).

use crate::bitboard::*;
use crate::board::{Position, START_WALLS};
use crate::moves::*;

/// Kompakt pozisyon dizisi — satrançtaki FEN'in karşılığı.
///
/// `h/v/pawn0/pawn1/walls0/walls1/side/ply`, duvar bitboard'ları hex.
/// Başlangıç pozisyonu: `0/0/76/4/10/10/0/0`
///
/// Web UI'ı durumsuz kılan şey bu: tarayıcı pozisyonu bu dizi olarak tutuyor,
/// sunucu tarafında oturum yok, sayfa yenilenince oyun kaybolmuyor.
pub fn to_fen(p: &Position) -> String {
    format!(
        "{:x}/{:x}/{}/{}/{}/{}/{}/{}",
        p.h, p.v, p.pawn[0], p.pawn[1], p.walls[0], p.walls[1], p.side, p.ply
    )
}

pub fn from_fen(s: &str) -> Option<Position> {
    let f: Vec<&str> = s.trim().split('/').collect();
    if f.len() != 8 {
        return None;
    }
    let h = u64::from_str_radix(f[0], 16).ok()?;
    let v = u64::from_str_radix(f[1], 16).ok()?;
    let pawn = [f[2].parse::<u8>().ok()?, f[3].parse::<u8>().ok()?];
    let walls = [f[4].parse::<u8>().ok()?, f[5].parse::<u8>().ok()?];
    let side = f[6].parse::<u8>().ok()?;
    let ply = f[7].parse::<u16>().ok()?;

    if pawn[0] as usize >= CELLS || pawn[1] as usize >= CELLS || pawn[0] == pawn[1] {
        return None;
    }
    if walls[0] > START_WALLS || walls[1] > START_WALLS || side > 1 {
        return None;
    }
    let mut pos = Position {
        h,
        v,
        pawn,
        walls,
        side,
        hash: 0,
        ply,
    };
    pos.hash = pos.compute_hash();
    // Ulaşılamaz pozisyon motoru sonsuz döngüye sokabilir; baştan reddet.
    if pos.winner().is_none() && !pos.paths_open(h, v) {
        return None;
    }
    Some(pos)
}

pub fn cell_name(i: usize) -> String {
    let (r, c) = (row_of(i), col_of(i));
    format!("{}{}", (b'a' + c as u8) as char, N - r)
}

pub fn parse_cell(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    if b.len() != 2 {
        return None;
    }
    let c = (b[0] as char).to_ascii_lowercase() as u8;
    if !(b'a'..=b'i').contains(&c) {
        return None;
    }
    let n = (b[1] as char).to_digit(10)? as usize;
    if !(1..=N).contains(&n) {
        return None;
    }
    Some(cell(N - n, (c - b'a') as usize))
}

pub fn wall_name(slot: usize, horizontal: bool) -> String {
    let (r, c) = (slot / WDIM, slot % WDIM);
    format!(
        "{}{}{}",
        (b'a' + c as u8) as char,
        N - r,
        if horizontal { 'h' } else { 'v' }
    )
}

pub fn parse_wall(s: &str) -> Option<(usize, bool)> {
    let b = s.as_bytes();
    if b.len() != 3 {
        return None;
    }
    let horizontal = match (b[2] as char).to_ascii_lowercase() {
        'h' => true,
        'v' => false,
        _ => return None,
    };
    let c = (b[0] as char).to_ascii_lowercase() as u8;
    if !(b'a'..=b'i').contains(&c) {
        return None;
    }
    let n = (b[1] as char).to_digit(10)? as usize;
    let (r, cc) = (N - n, (c - b'a') as usize);
    if r >= WDIM || cc >= WDIM {
        return None;
    }
    Some((wslot(r, cc), horizontal))
}

/// Hamleyi notasyona çevirir. Piyon hamleleri için hedef kare gerektiğinden
/// pozisyona ihtiyaç var.
///
/// İllegal hamleler için de çağrılabilir (hata mesajları): hedef tahta dışına
/// düşüyorsa kare adı yerine yön kısaltması döner.
pub fn move_name(pos: &Position, mv: Move) -> String {
    match mv.kind() {
        MoveKind::HWall(s) => wall_name(s, true),
        MoveKind::VWall(s) => wall_name(s, false),
        MoveKind::Pawn(d) => {
            let from = pos.pawn[pos.side as usize] as i32;
            let to = from + PAWN_DELTA[d];
            let (fr, fc) = (from / N as i32, from % N as i32);
            let (tr, tc) = (to / N as i32, to % N as i32);
            // Satır/sütun kayması makul değilse (tahta dışı ya da sarma) yön adı.
            if to < 0 || to >= CELLS as i32 || (tr - fr).abs() > 2 || (tc - fc).abs() > 2 {
                PAWN_NAME[d].to_string()
            } else {
                cell_name(to as usize)
            }
        }
    }
}

/// Kullanıcı girdisini legal bir hamleye çözer.
/// `e2` -> o kareye giden piyon hamlesi, `e6h` -> duvar.
pub fn parse_move(pos: &Position, s: &str) -> Option<Move> {
    let s = s.trim();
    if let Some((slot, horizontal)) = parse_wall(s) {
        let mv = if horizontal {
            Move::hwall(slot)
        } else {
            Move::vwall(slot)
        };
        return if pos.is_legal(mv) { Some(mv) } else { None };
    }
    if let Some(target) = parse_cell(s) {
        let mut v = Vec::with_capacity(8);
        pos.gen_pawn_moves(&pos.masks(), &mut v);
        return v.into_iter().find(|&m| match m.kind() {
            MoveKind::Pawn(d) => pos.pawn_target(d) == target,
            _ => false,
        });
    }
    // "N", "NE" gibi yön kısaltmaları da kabul edilir.
    let up = s.to_ascii_uppercase();
    if let Some(d) = PAWN_NAME.iter().position(|&n| n == up) {
        let mv = Move::pawn(d);
        if pos.is_legal(mv) {
            return Some(mv);
        }
    }
    None
}

const PIECE: [char; 2] = ['A', 'B'];

/// ASCII tahta çizimi.
pub fn board_string(pos: &Position) -> String {
    let m = pos.masks();
    let mut out = String::new();

    out.push_str("     ");
    for c in 0..N {
        out.push((b'a' + c as u8) as char);
        out.push_str("   ");
    }
    out.push('\n');

    for r in 0..N {
        out.push_str(&format!(" {} |", N - r));
        for c in 0..N {
            let i = cell(r, c);
            let ch = if pos.pawn[0] as usize == i {
                PIECE[0]
            } else if pos.pawn[1] as usize == i {
                PIECE[1]
            } else {
                '.'
            };
            out.push(' ');
            out.push(ch);
            out.push(' ');
            if c + 1 < N {
                // (r,c) -> (r,c+1) geçişi kapalıysa dikey duvar var
                out.push(if m.step_open(i, EAST) { ' ' } else { '|' });
            }
        }
        out.push_str("|\n");

        if r + 1 < N {
            out.push_str("   |");
            for c in 0..N {
                let i = cell(r, c);
                out.push_str(if m.step_open(i, SOUTH) { "   " } else { "===" });
                if c + 1 < N {
                    out.push(' ');
                }
            }
            out.push_str("|\n");
        }
    }

    out.push_str(&format!(
        "\n  A (oyuncu 1) duvar: {}   B (oyuncu 2) duvar: {}   sıra: {}\n",
        pos.walls[0], pos.walls[1], PIECE[pos.side as usize]
    ));
    out
}
