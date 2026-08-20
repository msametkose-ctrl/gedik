//! Zobrist hashing. Tablolar derleme zamanında splitmix64 ile üretilir,
//! dolayısıyla çalışmalar arası deterministik (açılış kitabı / TT dump uyumu için önemli).

use crate::bitboard::{CELLS, WSLOTS};

const fn mix(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

const fn table<const K: usize>(salt: u64) -> [u64; K] {
    let mut a = [0u64; K];
    let mut i = 0;
    while i < K {
        a[i] = mix(salt ^ (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        i += 1;
    }
    a
}

pub const Z_HWALL: [u64; WSLOTS] = table(0x51ed_2701_a1b2_c3d4);
pub const Z_VWALL: [u64; WSLOTS] = table(0x7f4a_7c15_dead_beef);
pub const Z_PAWN0: [u64; CELLS] = table(0x0123_4567_89ab_cdef);
pub const Z_PAWN1: [u64; CELLS] = table(0xfedc_ba98_7654_3210);
/// Kalan duvar sayısı 0..=10 (11 değer).
pub const Z_WALLS0: [u64; 11] = table(0xa5a5_5a5a_1111_2222);
pub const Z_WALLS1: [u64; 11] = table(0x3333_4444_c0ff_ee00);
pub const Z_SIDE: u64 = 0x9d39_247e_33776d41;

#[inline(always)]
pub fn z_pawn(player: usize, c: usize) -> u64 {
    if player == 0 {
        Z_PAWN0[c]
    } else {
        Z_PAWN1[c]
    }
}

#[inline(always)]
pub fn z_walls(player: usize, n: usize) -> u64 {
    if player == 0 {
        Z_WALLS0[n]
    } else {
        Z_WALLS1[n]
    }
}
