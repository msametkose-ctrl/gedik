//! Bitboard temelleri.
//!
//! İki ayrı bit uzayı var:
//!
//! * **Hücre uzayı** — 81 hücre, `u128` içinde `index = row * 9 + col`.
//!   `row 0` tahtanın üstü (Glendenning notasyonunda 9. sıra),
//!   `col 0` en sol (a sütunu).
//!
//! * **Duvar uzayı** — 64 duvar merkezi, `u64` içinde `slot = row * 8 + col`.
//!   Bir duvar merkezi `(r, c)`, `(r,c) (r,c+1) (r+1,c) (r+1,c+1)` hücrelerinin
//!   ortasındaki kesişim noktasıdır. Yatay duvar dikey geçişi, dikey duvar
//!   yatay geçişi engeller. Yatay ve dikey duvarlar ayrı bitboard'larda tutulur.

pub const N: usize = 9;
pub const CELLS: usize = 81;
/// Duvar merkezi ızgarasının boyutu (8x8).
pub const WDIM: usize = 8;
pub const WSLOTS: usize = 64;

/// 81 hücrenin tamamı.
pub const FULL: u128 = (1u128 << CELLS) - 1;

#[inline(always)]
pub const fn cell(r: usize, c: usize) -> usize {
    r * N + c
}

#[inline(always)]
pub const fn row_of(i: usize) -> usize {
    i / N
}

#[inline(always)]
pub const fn col_of(i: usize) -> usize {
    i % N
}

#[inline(always)]
pub const fn bit(i: usize) -> u128 {
    1u128 << i
}

#[inline(always)]
pub const fn wslot(r: usize, c: usize) -> usize {
    r * WDIM + c
}

const fn row_mask_const(r: usize) -> u128 {
    0x1ffu128 << (r * N)
}

const fn col_mask_const(c: usize) -> u128 {
    let mut m = 0u128;
    let mut r = 0;
    while r < N {
        m |= 1u128 << (r * N + c);
        r += 1;
    }
    m
}

pub const ROW_MASK: [u128; N] = {
    let mut a = [0u128; N];
    let mut r = 0;
    while r < N {
        a[r] = row_mask_const(r);
        r += 1;
    }
    a
};

pub const COL_MASK: [u128; N] = {
    let mut a = [0u128; N];
    let mut c = 0;
    while c < N {
        a[c] = col_mask_const(c);
        c += 1;
    }
    a
};

/// En alt sıra hariç her şey — güneye hamle *olasılığı* olan hücreler.
pub const NOT_LAST_ROW: u128 = FULL & !row_mask_const(N - 1);
/// En sağ sütun hariç her şey — doğuya hamle *olasılığı* olan hücreler.
pub const NOT_LAST_COL: u128 = FULL & !col_mask_const(N - 1);

/// Yön indeksleri. Sıralama sabittir; aksiyon ID'leri buna bağlı.
pub const NORTH: usize = 0;
pub const EAST: usize = 1;
pub const SOUTH: usize = 2;
pub const WEST: usize = 3;

/// Bir yönün hücre indeksi üzerindeki etkisi.
pub const DIR_DELTA: [i32; 4] = [-(N as i32), 1, N as i32, -1];

#[inline(always)]
pub const fn perpendicular(d: usize) -> [usize; 2] {
    match d {
        NORTH | SOUTH => [EAST, WEST],
        _ => [NORTH, SOUTH],
    }
}

/// Bir duvar konfigürasyonu için önceden hesaplanmış geçiş maskeleri.
///
/// `open_n` içinde bir hücrenin biti set ise, o hücreden kuzeye hamle
/// hem tahta içinde hem de duvarsızdır. Böylece hamle üretimi ve flood-fill
/// tek bitwise işlemine iner; her hamlede pathfinding yapılmaz.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Masks {
    pub open_n: u128,
    pub open_e: u128,
    pub open_s: u128,
    pub open_w: u128,
}

impl Masks {
    /// Yatay/dikey duvar bitboard'larından maskeleri kurar (8 iterasyon).
    pub fn new(h: u64, v: u64) -> Self {
        let mut blocked_s: u128 = 0;
        let mut blocked_e: u128 = 0;

        for r in 0..WDIM {
            // Yatay duvar merkezi (r, c), c ve c+1 sütunlarında r/r+1 arası geçişi keser.
            let hr = ((h >> (r * WDIM)) & 0xff) as u128;
            blocked_s |= (hr | (hr << 1)) << (r * N);

            // Dikey duvar merkezi (r, c), r ve r+1 satırlarında c/c+1 arası geçişi keser.
            let vr = ((v >> (r * WDIM)) & 0xff) as u128;
            blocked_e |= vr << (r * N);
            blocked_e |= vr << ((r + 1) * N);
        }

        let open_s = NOT_LAST_ROW & !blocked_s;
        let open_e = NOT_LAST_COL & !blocked_e;

        Masks {
            open_n: open_s << N,
            open_e,
            open_s,
            open_w: open_e << 1,
        }
    }

    #[inline(always)]
    pub fn open(&self, d: usize) -> u128 {
        match d {
            NORTH => self.open_n,
            EAST => self.open_e,
            SOUTH => self.open_s,
            _ => self.open_w,
        }
    }

    /// `from` hücresinden `d` yönüne tek adım mümkün mü?
    #[inline(always)]
    pub fn step_open(&self, from: usize, d: usize) -> bool {
        self.open(d) & bit(from) != 0
    }

    /// Bir bit kümesinin bir adımlık komşuluğu (kendisi hariç).
    #[inline(always)]
    pub fn expand(&self, x: u128) -> u128 {
        ((x & self.open_n) >> N)
            | ((x & self.open_s) << N)
            | ((x & self.open_w) >> 1)
            | ((x & self.open_e) << 1)
    }

    /// `from`'dan ulaşılabilen tüm hücreler. Piyonlar geçirgen sayılır
    /// (Quoridor'da piyon kalıcı engel değil — üzerinden atlanabiliyor).
    pub fn reachable(&self, from: usize) -> u128 {
        let mut seen = bit(from);
        loop {
            let next = (seen | self.expand(seen)) & FULL;
            if next == seen {
                return seen;
            }
            seen = next;
        }
    }

    /// `from`'dan `goal_row` sırasına en kısa mesafe (hamle sayısı).
    /// Hedeften geriye doğru dalga yayarak hesaplar. Piyon atlamalarını
    /// hesaba katmaz, dolayısıyla gerçek mesafenin üst sınırı değil,
    /// duvar-only alt sınırıdır — değerlendirme için standart kullanım.
    pub fn distance_to_row(&self, from: usize, goal_row: usize) -> Option<u32> {
        let target = bit(from);
        let mut frontier = ROW_MASK[goal_row];
        if frontier & target != 0 {
            return Some(0);
        }
        let mut seen = frontier;
        let mut d = 0u32;
        loop {
            frontier = self.expand(frontier) & !seen;
            if frontier == 0 {
                return None;
            }
            d += 1;
            if frontier & target != 0 {
                return Some(d);
            }
            seen |= frontier;
        }
    }

    /// `targets` içinden `goal_row`'a **en yakın** olanların bitmask'i.
    ///
    /// Hedeften dalga yayar ve ilk hedef hücreye değdiği anda durur; tüm
    /// mesafe alanını hesaplamaktan çok daha ucuz. Rollout politikasının
    /// sıcak yolu bu — "en kısa yola doğru ilerle" kararı tek çağrıya iniyor.
    /// Beraberlik durumunda birden fazla bit döner (rastgele seçim için).
    pub fn closest_to_row(&self, targets: u128, goal_row: usize) -> u128 {
        if targets == 0 {
            return 0;
        }
        let mut frontier = ROW_MASK[goal_row];
        let mut seen = frontier;
        loop {
            let hit = frontier & targets;
            if hit != 0 {
                return hit;
            }
            frontier = self.expand(frontier) & !seen;
            if frontier == 0 {
                return targets; // hiçbiri ulaşılamıyor
            }
            seen |= frontier;
        }
    }

    /// Tüm hücreler için `goal_row`'a mesafe. Ulaşılamayan hücreler `u8::MAX`.
    pub fn distance_field(&self, goal_row: usize) -> [u8; CELLS] {
        let mut out = [u8::MAX; CELLS];
        let mut frontier = ROW_MASK[goal_row];
        let mut seen = frontier;
        let mut d = 0u8;
        while frontier != 0 {
            let mut f = frontier;
            while f != 0 {
                let i = f.trailing_zeros() as usize;
                f &= f - 1;
                out[i] = d;
            }
            frontier = self.expand(frontier) & !seen;
            seen |= frontier;
            d = d.saturating_add(1);
        }
        out
    }
}
