//! Hamle temsili.
//!
//! Her hamle **sabit** bir aksiyon ID'sine sahiptir, 0..140:
//!
//! | aralık   | anlam                                |
//! |----------|--------------------------------------|
//! | 0..63    | yatay duvar, slot = id               |
//! | 64..127  | dikey duvar, slot = id - 64          |
//! | 128..139 | piyon hamlesi, yön = id - 128        |
//!
//! ID'nin pozisyondan **bağımsız** olması kritik: "kuzeye git" her zaman 128.
//! OpenSpiel'in Quoridor implementasyonundaki meşhur bug (issue #1158) tam
//! olarak buydu — aksiyon ID'leri duvar dizilimine göre kayıyordu ve RL
//! ajanları tutarlı bir politika öğrenemiyordu.

use crate::bitboard::N;
use std::fmt;

pub const ACT_HWALL: u16 = 0;
pub const ACT_VWALL: u16 = 64;
pub const ACT_PAWN: u16 = 128;
pub const NUM_ACTIONS: usize = 140;

/// Piyon hamlesi yönleri, aksiyon ID sırasıyla.
/// 0-3: düz (N,E,S,W) · 4-7: çift atlama (NN,EE,SS,WW) · 8-11: diyagonal (NE,SE,SW,NW)
pub const PAWN_DELTA: [i32; 12] = [
    -(N as i32),     // 0  N
    1,               // 1  E
    N as i32,        // 2  S
    -1,              // 3  W
    -2 * N as i32,   // 4  NN
    2,               // 5  EE
    2 * N as i32,    // 6  SS
    -2,              // 7  WW
    -(N as i32) + 1, // 8  NE
    N as i32 + 1,    // 9  SE
    N as i32 - 1,    // 10 SW
    -(N as i32) - 1, // 11 NW
];

pub const PAWN_NAME: [&str; 12] = [
    "N", "E", "S", "W", "NN", "EE", "SS", "WW", "NE", "SE", "SW", "NW",
];

/// İki yönün bileşimine karşılık gelen diyagonal hamle indeksi (8..11).
pub fn diagonal_index(delta: i32) -> Option<usize> {
    (8..12).find(|&i| PAWN_DELTA[i] == delta)
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Move(pub u16);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MoveKind {
    /// Yatay duvar, slot 0..63
    HWall(usize),
    /// Dikey duvar, slot 0..63
    VWall(usize),
    /// Piyon hamlesi, yön 0..11
    Pawn(usize),
}

impl Move {
    #[inline(always)]
    pub fn hwall(slot: usize) -> Move {
        Move(ACT_HWALL + slot as u16)
    }

    #[inline(always)]
    pub fn vwall(slot: usize) -> Move {
        Move(ACT_VWALL + slot as u16)
    }

    #[inline(always)]
    pub fn pawn(dir: usize) -> Move {
        Move(ACT_PAWN + dir as u16)
    }

    #[inline(always)]
    pub fn action_id(self) -> usize {
        self.0 as usize
    }

    #[inline(always)]
    pub fn kind(self) -> MoveKind {
        let id = self.0;
        if id < ACT_VWALL {
            MoveKind::HWall(id as usize)
        } else if id < ACT_PAWN {
            MoveKind::VWall((id - ACT_VWALL) as usize)
        } else {
            MoveKind::Pawn((id - ACT_PAWN) as usize)
        }
    }

    #[inline(always)]
    pub fn is_wall(self) -> bool {
        self.0 < ACT_PAWN
    }
}

impl fmt::Debug for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind() {
            MoveKind::HWall(s) => write!(f, "H{}", crate::notation::wall_name(s, true)),
            MoveKind::VWall(s) => write!(f, "V{}", crate::notation::wall_name(s, false)),
            MoveKind::Pawn(d) => write!(f, "{}", PAWN_NAME[d]),
        }
    }
}
