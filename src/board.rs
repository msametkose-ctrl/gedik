//! Pozisyon, hamle üretimi, apply/undo.
//!
//! Oyuncu 0 en alt sırada başlar (Glendenning notasyonunda e1) ve 9. sıraya
//! (dahili satır 0) ulaşmaya çalışır. Oyuncu 1 tersi.

use crate::bitboard::*;
use crate::moves::*;
use crate::zobrist as z;

pub const START_WALLS: u8 = 10;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Position {
    /// Yatay duvar merkezleri.
    pub h: u64,
    /// Dikey duvar merkezleri.
    pub v: u64,
    /// Piyon hücreleri.
    pub pawn: [u8; 2],
    /// Kalan duvar sayıları.
    pub walls: [u8; 2],
    /// Sıradaki oyuncu.
    pub side: u8,
    pub hash: u64,
    pub ply: u16,
}

#[derive(Clone, Copy)]
pub struct Undo {
    mv: Move,
    prev_pawn: u8,
    prev_hash: u64,
}

#[inline(always)]
fn in_slot(r: i32, c: i32) -> bool {
    r >= 0 && r < WDIM as i32 && c >= 0 && c < WDIM as i32
}

/// Oyuncunun ulaşmaya çalıştığı satır.
#[inline(always)]
pub const fn goal_row(player: usize) -> usize {
    if player == 0 {
        0
    } else {
        N - 1
    }
}

impl Default for Position {
    fn default() -> Self {
        Self::start()
    }
}

impl Position {
    pub fn start() -> Position {
        let mut p = Position {
            h: 0,
            v: 0,
            pawn: [cell(N - 1, N / 2) as u8, cell(0, N / 2) as u8],
            walls: [START_WALLS, START_WALLS],
            side: 0,
            hash: 0,
            ply: 0,
        };
        p.hash = p.compute_hash();
        p
    }

    pub fn compute_hash(&self) -> u64 {
        let mut k = 0u64;
        let mut hh = self.h;
        while hh != 0 {
            let s = hh.trailing_zeros() as usize;
            hh &= hh - 1;
            k ^= z::Z_HWALL[s];
        }
        let mut vv = self.v;
        while vv != 0 {
            let s = vv.trailing_zeros() as usize;
            vv &= vv - 1;
            k ^= z::Z_VWALL[s];
        }
        k ^= z::z_pawn(0, self.pawn[0] as usize);
        k ^= z::z_pawn(1, self.pawn[1] as usize);
        k ^= z::z_walls(0, self.walls[0] as usize);
        k ^= z::z_walls(1, self.walls[1] as usize);
        if self.side == 1 {
            k ^= z::Z_SIDE;
        }
        k
    }

    #[inline(always)]
    pub fn masks(&self) -> Masks {
        Masks::new(self.h, self.v)
    }

    #[inline(always)]
    pub fn h_bit(&self, r: usize, c: usize) -> bool {
        self.h >> wslot(r, c) & 1 != 0
    }

    #[inline(always)]
    pub fn v_bit(&self, r: usize, c: usize) -> bool {
        self.v >> wslot(r, c) & 1 != 0
    }

    /// Oyun bitti mi, bittiyse kazanan kim?
    #[inline]
    pub fn winner(&self) -> Option<usize> {
        if row_of(self.pawn[0] as usize) == goal_row(0) {
            Some(0)
        } else if row_of(self.pawn[1] as usize) == goal_row(1) {
            Some(1)
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn opponent(&self) -> usize {
        1 - self.side as usize
    }

    // ---------------------------------------------------------------- piyon

    /// Sıradaki oyuncunun piyon hamlelerini üretir.
    ///
    /// Resmî kural: rakip komşu karedeyse ve arkası açıksa **düz atlama
    /// zorunludur**; diyagonal ancak arkası duvar/tahta kenarıyla kapalıysa
    /// serbesttir.
    pub fn gen_pawn_moves(&self, m: &Masks, out: &mut Vec<Move>) {
        let me = self.pawn[self.side as usize] as usize;
        let opp = self.pawn[self.opponent()] as usize;

        for d in 0..4usize {
            if !m.step_open(me, d) {
                continue;
            }
            let t = (me as i32 + DIR_DELTA[d]) as usize;
            if t != opp {
                out.push(Move::pawn(d));
                continue;
            }
            // Rakibin üzerinde duruyoruz -> atlama.
            if m.step_open(t, d) {
                out.push(Move::pawn(4 + d)); // düz çift atlama
                continue;
            }
            for p in perpendicular(d) {
                if m.step_open(t, p) {
                    let delta = DIR_DELTA[d] + DIR_DELTA[p];
                    if let Some(idx) = diagonal_index(delta) {
                        out.push(Move::pawn(idx));
                    }
                }
            }
        }
    }

    /// Bir piyon hamlesinin hedef hücresi (legalite kontrolü yapmaz).
    #[inline(always)]
    pub fn pawn_target(&self, dir: usize) -> usize {
        (self.pawn[self.side as usize] as i32 + PAWN_DELTA[dir]) as usize
    }

    // ---------------------------------------------------------------- duvar

    /// Bir kafes noktası (lattice) bir şeye temas ediyor mu?
    /// Kafes koordinatları 0..=9; 0 ve 9 tahta kenarıdır.
    fn lattice_occupied(&self, i: i32, j: i32) -> bool {
        if i <= 0 || i >= N as i32 || j <= 0 || j >= N as i32 {
            return true; // tahta kenarı
        }
        // Bu noktadan geçen yatay duvarlar: merkez kafes (i, j-1), (i, j), (i, j+1)
        for dj in -1..=1i32 {
            let (r, c) = (i - 1, j + dj - 1);
            if in_slot(r, c) && self.h_bit(r as usize, c as usize) {
                return true;
            }
        }
        // Bu noktadan geçen dikey duvarlar: merkez kafes (i-1, j), (i, j), (i+1, j)
        for di in -1..=1i32 {
            let (r, c) = (i + di - 1, j - 1);
            if in_slot(r, c) && self.v_bit(r as usize, c as usize) {
                return true;
            }
        }
        false
    }

    /// Slatton kısayolu: bir duvar ancak gövdesinin **en az iki** noktası
    /// başka bir duvara veya tahta kenarına dayanıyorsa bir bariyeri
    /// tamamlayabilir. Aksi halde kimseyi kesemez, pahalı flood-fill'e gerek yok.
    fn needs_path_check(&self, slot: usize, horizontal: bool) -> bool {
        let (r, c) = ((slot / WDIM) as i32, (slot % WDIM) as i32);
        let nodes: [(i32, i32); 3] = if horizontal {
            [(r + 1, c), (r + 1, c + 1), (r + 1, c + 2)]
        } else {
            [(r, c + 1), (r + 1, c + 1), (r + 2, c + 1)]
        };
        let mut anchors = 0;
        for (i, j) in nodes {
            if self.lattice_occupied(i, j) {
                anchors += 1;
            }
        }
        anchors >= 2
    }

    /// Duvarın *şekilsel* legalitesi: slot boş mu, çakışma/kesişme var mı?
    #[inline]
    pub fn wall_shape_ok(&self, slot: usize, horizontal: bool) -> bool {
        let b = 1u64 << slot;
        if (self.h | self.v) & b != 0 {
            return false; // dolu ya da dik kesişme
        }
        let (r, c) = (slot / WDIM, slot % WDIM);
        if horizontal {
            if c > 0 && self.h_bit(r, c - 1) {
                return false;
            }
            if c + 1 < WDIM && self.h_bit(r, c + 1) {
                return false;
            }
        } else {
            if r > 0 && self.v_bit(r - 1, c) {
                return false;
            }
            if r + 1 < WDIM && self.v_bit(r + 1, c) {
                return false;
            }
        }
        true
    }

    /// Verilen duvar dizilimiyle her iki oyuncunun da hedefine yolu var mı?
    pub fn paths_open(&self, h: u64, v: u64) -> bool {
        let m = Masks::new(h, v);
        m.reachable(self.pawn[0] as usize) & ROW_MASK[goal_row(0)] != 0
            && m.reachable(self.pawn[1] as usize) & ROW_MASK[goal_row(1)] != 0
    }

    /// Duvarın tam legalitesi.
    pub fn wall_legal(&self, slot: usize, horizontal: bool) -> bool {
        if !self.wall_shape_ok(slot, horizontal) {
            return false;
        }
        if !self.needs_path_check(slot, horizontal) {
            return true;
        }
        let b = 1u64 << slot;
        if horizontal {
            self.paths_open(self.h | b, self.v)
        } else {
            self.paths_open(self.h, self.v | b)
        }
    }

    /// Kısayolu kullanmayan referans implementasyon (testlerde karşılaştırma için).
    pub fn wall_legal_slow(&self, slot: usize, horizontal: bool) -> bool {
        if !self.wall_shape_ok(slot, horizontal) {
            return false;
        }
        let b = 1u64 << slot;
        if horizontal {
            self.paths_open(self.h | b, self.v)
        } else {
            self.paths_open(self.h, self.v | b)
        }
    }

    pub fn gen_wall_moves(&self, out: &mut Vec<Move>) {
        if self.walls[self.side as usize] == 0 {
            return;
        }
        for slot in 0..WSLOTS {
            if self.wall_legal(slot, true) {
                out.push(Move::hwall(slot));
            }
        }
        for slot in 0..WSLOTS {
            if self.wall_legal(slot, false) {
                out.push(Move::vwall(slot));
            }
        }
    }

    // ---------------------------------------------------------------- toplu

    pub fn gen_moves(&self, out: &mut Vec<Move>) {
        if self.winner().is_some() {
            return;
        }
        let m = self.masks();
        self.gen_pawn_moves(&m, out);
        self.gen_wall_moves(out);
    }

    pub fn legal_moves(&self) -> Vec<Move> {
        let mut v = Vec::with_capacity(96);
        self.gen_moves(&mut v);
        v
    }

    pub fn is_legal(&self, mv: Move) -> bool {
        match mv.kind() {
            MoveKind::HWall(s) => self.walls[self.side as usize] > 0 && self.wall_legal(s, true),
            MoveKind::VWall(s) => self.walls[self.side as usize] > 0 && self.wall_legal(s, false),
            MoveKind::Pawn(_) => {
                let mut v = Vec::with_capacity(8);
                self.gen_pawn_moves(&self.masks(), &mut v);
                v.contains(&mv)
            }
        }
    }

    // ---------------------------------------------------------------- apply

    pub fn make(&mut self, mv: Move) -> Undo {
        let side = self.side as usize;
        let u = Undo {
            mv,
            prev_pawn: self.pawn[side],
            prev_hash: self.hash,
        };
        match mv.kind() {
            MoveKind::HWall(s) => {
                self.h |= 1u64 << s;
                self.hash ^= z::Z_HWALL[s];
                self.hash ^= z::z_walls(side, self.walls[side] as usize);
                self.walls[side] -= 1;
                self.hash ^= z::z_walls(side, self.walls[side] as usize);
            }
            MoveKind::VWall(s) => {
                self.v |= 1u64 << s;
                self.hash ^= z::Z_VWALL[s];
                self.hash ^= z::z_walls(side, self.walls[side] as usize);
                self.walls[side] -= 1;
                self.hash ^= z::z_walls(side, self.walls[side] as usize);
            }
            MoveKind::Pawn(d) => {
                let from = self.pawn[side] as usize;
                let to = (from as i32 + PAWN_DELTA[d]) as usize;
                self.pawn[side] = to as u8;
                self.hash ^= z::z_pawn(side, from) ^ z::z_pawn(side, to);
            }
        }
        self.side ^= 1;
        self.hash ^= z::Z_SIDE;
        self.ply += 1;
        u
    }

    pub fn unmake(&mut self, u: Undo) {
        self.side ^= 1;
        self.ply -= 1;
        let side = self.side as usize;
        match u.mv.kind() {
            MoveKind::HWall(s) => {
                self.h &= !(1u64 << s);
                self.walls[side] += 1;
            }
            MoveKind::VWall(s) => {
                self.v &= !(1u64 << s);
                self.walls[side] += 1;
            }
            MoveKind::Pawn(_) => {
                self.pawn[side] = u.prev_pawn;
            }
        }
        self.hash = u.prev_hash;
    }

    // ---------------------------------------------------------- yardımcılar

    /// Oyuncunun hedefine en kısa mesafe (duvar-only, atlama yok).
    pub fn distance(&self, player: usize) -> Option<u32> {
        self.masks()
            .distance_to_row(self.pawn[player] as usize, goal_row(player))
    }

    /// Yatay ayna simetrisi — TT ve NN veri artırımı için bedava 2x.
    pub fn mirrored(&self) -> Position {
        let mut p = *self;
        p.h = mirror_walls(self.h);
        p.v = mirror_walls(self.v);
        for i in 0..2 {
            let (r, c) = (row_of(self.pawn[i] as usize), col_of(self.pawn[i] as usize));
            p.pawn[i] = cell(r, N - 1 - c) as u8;
        }
        p.hash = p.compute_hash();
        p
    }
}

/// 8x8 duvar bitboard'unu sütun ekseninde aynalar.
///
/// Merkez `(r, c)` iki hücre sütununu (`c`, `c+1`) kapsar. Ayna altında hücre
/// sütunu `x -> 8-x` gittiği için kapsanan sütunlar `{7-c, 8-c}` olur, yani
/// yeni merkez `c' = 7-c`. Hem yatay hem dikey duvarlar için aynı eşleme.
pub fn mirror_walls(w: u64) -> u64 {
    let mut out = 0u64;
    let mut x = w;
    while x != 0 {
        let s = x.trailing_zeros() as usize;
        x &= x - 1;
        let (r, c) = (s / WDIM, s % WDIM);
        out |= 1u64 << wslot(r, WDIM - 1 - c);
    }
    out
}
