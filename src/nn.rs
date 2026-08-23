//! Küçük değerlendirme ağı (satrançtaki NNUE'nin karşılığı).
//!
//! Neden: mevcut değerlendirme dört sayı görüyor — iki mesafe, iki duvar
//! sayısı. Duvarların **nerede** olduğunu göremiyor, dolayısıyla kurulmakta
//! olan bir tuzağı ancak kurulduktan sonra, mesafe sıçradığında fark ediyor.
//! Ağ ham tahtayı görüyor.
//!
//! Girdi 325 sayı: 313'ü 0/1 (piyon kareleri, duvar yuvaları, duvar
//! sayıları, sıra), 12'si mevcut elle tasarlanmış özellikler. Elle
//! tasarlananları da vermek riski düşürüyor: ağ en kötü ihtimalle mevcut
//! değerlendirmeyi taklit ediyor, üstüne duvar yerleşimini ekliyor.
//!
//! **Kodlama yalnızca burada, Rust'ta yapılıyor.** Eğitim betiği kodlamayı
//! yeniden yazmıyor, `kodla` aracının ürettiği indeks listesini okuyor.
//! Python ile Rust'ın kodlamayı farklı yapması bu tür projelerde en sık
//! görülen ve en zor bulunan hata; o kapı baştan kapalı.
//!
//! Hız: seyrek katman sayesinde ileri geçiş ~4 bin çarpma, ölçülen ~1 µs.
//! Değerlendirme 3-5 kat yavaşlıyor ama bu bedava — ölçtük, arama 300 bin
//! iterasyondan sonra zaten güçlenmiyor (200 bin -> 2 milyon: +19 Elo).

use crate::bitboard::{CELLS, WSLOTS};
use crate::moves::NUM_ACTIONS;
use crate::board::{goal_row, Position};
use crate::heuristics::{features, threat_features, NUM_FEATURES, NUM_THREAT};
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::OnceLock;

pub const N_SPARSE: usize = CELLS + CELLS + WSLOTS + WSLOTS + 11 + 11 + 1; // 313
pub const N_DENSE: usize = NUM_FEATURES + NUM_THREAT; // 12 + 4
pub const N_IN: usize = N_SPARSE + N_DENSE; // 313 + 16 = 329

/// Seyrek girdi indeksleri. Sıradaki oyuncunun bakışıyla: ilk blok **benim**
/// piyonum, ikincisi rakibin. Böylece ağ "kim oynuyor" bilgisini karelerin
/// içinde de görüyor.
pub fn sparse_indices(pos: &Position, out: &mut Vec<u16>) {
    out.clear();
    let me = pos.side as usize;
    let opp = 1 - me;
    let mut base = 0usize;
    out.push((base + pos.pawn[me] as usize) as u16);
    base += CELLS;
    out.push((base + pos.pawn[opp] as usize) as u16);
    base += CELLS;
    for s in 0..WSLOTS {
        if pos.h >> s & 1 != 0 {
            out.push((base + s) as u16);
        }
    }
    base += WSLOTS;
    for s in 0..WSLOTS {
        if pos.v >> s & 1 != 0 {
            out.push((base + s) as u16);
        }
    }
    base += WSLOTS;
    out.push((base + pos.walls[me].min(10) as usize) as u16);
    base += 11;
    out.push((base + pos.walls[opp].min(10) as usize) as u16);
    base += 11;
    if pos.side == 1 {
        out.push(base as u16);
    }
}

pub fn dense_features(pos: &Position) -> [f32; N_DENSE] {
    let f = features(pos);
    let t = threat_features(pos);
    let mut out = [0.0f32; N_DENSE];
    out[..NUM_FEATURES].copy_from_slice(&f);
    out[NUM_FEATURES..].copy_from_slice(&t);
    out
}

pub const SPATIAL_CHANNELS: usize = 13;
pub const SPATIAL_SIZE: usize = SPATIAL_CHANNELS * CELLS; // 1053

/// Belirli bir oyuncunun hedefine giden tüm en kısa yolların üzerinde yer alan hücrelerin bitmask'i.
pub fn shortest_path_mask(pos: &Position, player: usize) -> u128 {
    let m = pos.masks();
    let goal = goal_row(player);
    let field = m.distance_field(goal);
    let start = pos.pawn[player] as usize;
    if start >= CELLS || field[start] == u8::MAX {
        return 0;
    }
    let mut path_mask = crate::bitboard::bit(start);
    let mut frontier = crate::bitboard::bit(start);
    let mut cur_d = field[start];
    while cur_d > 0 {
        let next_d = cur_d - 1;
        let exp = m.expand(frontier);
        let mut next_frontier = 0u128;
        let mut exp_bits = exp;
        while exp_bits != 0 {
            let i = exp_bits.trailing_zeros() as usize;
            exp_bits &= exp_bits - 1;
            if field[i] == next_d {
                next_frontier |= crate::bitboard::bit(i);
            }
        }
        if next_frontier == 0 {
            break;
        }
        path_mask |= next_frontier;
        frontier = next_frontier;
        cur_d = next_d;
    }
    path_mask
}

/// Pozisyonu 13 kanallı 9x9 uzamsal ızgara (spatial tensor) olarak kodlar.
pub fn spatial_planes(pos: &Position) -> [f32; SPATIAL_SIZE] {
    let mut out = [0.0f32; SPATIAL_SIZE];
    let me = pos.side as usize;
    let opp = 1 - me;
    let m = pos.masks();

    // 0: Benim piyonum
    let me_pawn = pos.pawn[me] as usize;
    if me_pawn < CELLS {
        out[0 * CELLS + me_pawn] = 1.0;
    }

    // 1: Rakip piyon
    let opp_pawn = pos.pawn[opp] as usize;
    if opp_pawn < CELLS {
        out[1 * CELLS + opp_pawn] = 1.0;
    }

    // 2: Yatay duvarlar (8x8 yuva -> 9x9 kareler: (r,c) ve (r,c+1))
    for s in 0..WSLOTS {
        if pos.h >> s & 1 != 0 {
            let r = s / 8;
            let c = s % 8;
            out[2 * CELLS + r * 9 + c] = 1.0;
            out[2 * CELLS + r * 9 + c + 1] = 1.0;
        }
    }

    // 3: Dikey duvarlar (8x8 yuva -> 9x9 kareler: (r,c) ve (r+1,c))
    for s in 0..WSLOTS {
        if pos.v >> s & 1 != 0 {
            let r = s / 8;
            let c = s % 8;
            out[3 * CELLS + r * 9 + c] = 1.0;
            out[3 * CELLS + (r + 1) * 9 + c] = 1.0;
        }
    }

    // 4 & 5: Mesafe alanları (benim ve rakibin hedefine)
    let me_field = m.distance_field(goal_row(me));
    let opp_field = m.distance_field(goal_row(opp));
    for i in 0..CELLS {
        out[4 * CELLS + i] = (me_field[i] as f32 / 20.0).min(1.0);
        out[5 * CELLS + i] = (opp_field[i] as f32 / 20.0).min(1.0);
    }

    // 6 & 7: En kısa yol hücreleri
    let me_path = shortest_path_mask(pos, me);
    let opp_path = shortest_path_mask(pos, opp);
    for i in 0..CELLS {
        if me_path & (1u128 << i) != 0 {
            out[6 * CELLS + i] = 1.0;
        }
        if opp_path & (1u128 << i) != 0 {
            out[7 * CELLS + i] = 1.0;
        }
    }

    // 8 & 9: Kritik darboğazlar / chokepoints
    let me_choke = crate::heuristics::chokepoints(pos, me);
    let opp_choke = crate::heuristics::chokepoints(pos, opp);
    for i in 0..CELLS {
        if me_choke & (1u128 << i) != 0 {
            out[8 * CELLS + i] = 1.0;
        }
        if opp_choke & (1u128 << i) != 0 {
            out[9 * CELLS + i] = 1.0;
        }
    }

    // 10 & 11: Kalan duvar sayıları (tüm 9x9 düzlemine yayılmış)
    let me_walls = pos.walls[me] as f32 / 10.0;
    let opp_walls = pos.walls[opp] as f32 / 10.0;
    for i in 0..CELLS {
        out[10 * CELLS + i] = me_walls;
        out[11 * CELLS + i] = opp_walls;
    }

    // 12: Sıra düzlemi
    let side_val = if pos.side == 0 { 1.0 } else { 0.0 };
    for i in 0..CELLS {
        out[12 * CELLS + i] = side_val;
    }

    out
}

#[inline(always)]
fn shift_plane(src: &[f32], dr: isize, dc: isize, dst: &mut [f32; CELLS]) {
    let r_start = (0isize).max(-dr) as usize;
    let r_end = (9isize).min(9 - dr) as usize;
    let c_start = (0isize).max(-dc) as usize;
    let c_end = (9isize).min(9 - dc) as usize;

    dst.fill(0.0);
    for r in r_start..r_end {
        let in_r = (r as isize + dr) as usize;
        let out_row = r * 9;
        let in_row = in_r * 9;
        let out_slice = &mut dst[out_row + c_start..out_row + c_end];
        let in_slice = &src[in_row + (c_start as isize + dc) as usize..in_row + (c_end as isize + dc) as usize];
        out_slice.copy_from_slice(in_slice);
    }
}

#[derive(Clone)]
pub struct Conv2d3x3 {
    pub in_c: usize,
    pub out_c: usize,
    pub weights: Vec<f32>, // [out_c, in_c, 3, 3]
    pub bias: Vec<f32>,    // [out_c]
}

impl Conv2d3x3 {
    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        for oc in 0..self.out_c {
            let b = self.bias[oc];
            let out_plane = &mut output[oc * CELLS..][..CELLS];
            out_plane.fill(b);
        }

        let mut shifted = [0.0f32; CELLS];
        for ic in 0..self.in_c {
            let in_plane = &input[ic * CELLS..][..CELLS];
            for dr in 0..3isize {
                let r_shift = dr - 1;
                for dc in 0..3isize {
                    let c_shift = dc - 1;
                    let k = (dr * 3 + dc) as usize;
                    shift_plane(in_plane, r_shift, c_shift, &mut shifted);

                    for oc in 0..self.out_c {
                        let w = self.weights[oc * self.in_c * 9 + ic * 9 + k];
                        if w == 0.0 {
                            continue;
                        }
                        let out_plane = &mut output[oc * CELLS..][..CELLS];
                        for i in 0..CELLS {
                            out_plane[i] += w * shifted[i];
                        }
                    }
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct Conv2d1x1 {
    pub in_c: usize,
    pub out_c: usize,
    pub weights: Vec<f32>, // [out_c, in_c]
    pub bias: Vec<f32>,    // [out_c]
}

impl Conv2d1x1 {
    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        for oc in 0..self.out_c {
            let b = self.bias[oc];
            let out_plane = &mut output[oc * CELLS..][..CELLS];
            out_plane.fill(b);

            let w_oc = &self.weights[oc * self.in_c..];
            for ic in 0..self.in_c {
                let w = w_oc[ic];
                let in_plane = &input[ic * CELLS..][..CELLS];
                for i in 0..CELLS {
                    out_plane[i] += w * in_plane[i];
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct Linear {
    pub in_f: usize,
    pub out_f: usize,
    pub weights: Vec<f32>, // [out_f, in_f]
    pub bias: Vec<f32>,    // [out_f]
}

impl Linear {
    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        for of in 0..self.out_f {
            let mut sum = self.bias[of];
            let w_row = &self.weights[of * self.in_f..][..self.in_f];
            for (w, &x) in w_row.iter().zip(input) {
                sum += w * x;
            }
            output[of] = sum;
        }
    }
}

#[derive(Clone)]
pub struct ResBlock {
    pub conv1: Conv2d3x3,
    pub conv2: Conv2d3x3,
}

impl ResBlock {
    pub fn forward(&self, input: &[f32], output: &mut [f32], scratch: &mut [f32]) {
        self.conv1.forward(input, scratch);
        for x in scratch.iter_mut() {
            *x = x.max(0.0);
        }
        self.conv2.forward(scratch, output);
        for (out, &inp) in output.iter_mut().zip(input) {
            *out = (*out + inp).max(0.0);
        }
    }
}

#[derive(Clone)]
pub struct ResNet {
    pub channels: usize,
    pub init_conv: Conv2d3x3,
    pub blocks: Vec<ResBlock>,
    pub val_conv: Conv2d1x1,
    pub val_fc1: Linear,
    pub val_fc2: Linear,
    pub pol_conv: Option<Conv2d1x1>,
    pub pol_fc: Option<Linear>,
    pub moves_conv: Option<Conv2d1x1>,
    pub moves_fc1: Option<Linear>,
    pub moves_fc2: Option<Linear>,
    pub delta_conv: Option<Conv2d1x1>,
    pub delta_fc1: Option<Linear>,
    pub delta_fc2: Option<Linear>,
}

impl ResNet {
    pub fn value(&self, pos: &Position) -> f32 {
        let planes = spatial_planes(pos);
        let ch = self.channels;
        const MAX_BUF: usize = 32 * CELLS;
        let mut buf1 = [0.0f32; MAX_BUF];
        let mut buf2 = [0.0f32; MAX_BUF];
        let mut scratch = [0.0f32; MAX_BUF];

        let cur_buf1 = &mut buf1[..ch * CELLS];
        let cur_buf2 = &mut buf2[..ch * CELLS];
        let cur_scratch = &mut scratch[..ch * CELLS];

        // Init Conv + ReLU
        self.init_conv.forward(&planes, cur_buf1);
        for x in cur_buf1.iter_mut() {
            *x = x.max(0.0);
        }

        // ResBlocks
        for block in &self.blocks {
            block.forward(cur_buf1, cur_buf2, cur_scratch);
            cur_buf1.copy_from_slice(cur_buf2);
        }

        // Value Head
        let mut val_conv_out = [0.0f32; 2 * CELLS];
        self.val_conv.forward(cur_buf1, &mut val_conv_out);
        for x in val_conv_out.iter_mut() {
            *x = x.max(0.0);
        }

        let mut fc1_out = [0.0f32; 32];
        self.val_fc1.forward(&val_conv_out, &mut fc1_out);
        for x in fc1_out.iter_mut() {
            *x = x.max(0.0);
        }

        let mut out_v = [0.0f32; 1];
        self.val_fc2.forward(&fc1_out, &mut out_v);

        1.0 / (1.0 + (-out_v[0]).exp())
    }

    pub fn policy(&self, pos: &Position, moves: &[crate::moves::Move]) -> Option<Vec<f32>> {
        let pol_conv = self.pol_conv.as_ref()?;
        let pol_fc = self.pol_fc.as_ref()?;

        let planes = spatial_planes(pos);
        let ch = self.channels;
        const MAX_BUF: usize = 32 * CELLS;
        let mut buf1 = [0.0f32; MAX_BUF];
        let mut buf2 = [0.0f32; MAX_BUF];
        let mut scratch = [0.0f32; MAX_BUF];

        let cur_buf1 = &mut buf1[..ch * CELLS];
        let cur_buf2 = &mut buf2[..ch * CELLS];
        let cur_scratch = &mut scratch[..ch * CELLS];

        self.init_conv.forward(&planes, cur_buf1);
        for x in cur_buf1.iter_mut() {
            *x = x.max(0.0);
        }

        for block in &self.blocks {
            block.forward(cur_buf1, cur_buf2, cur_scratch);
            cur_buf1.copy_from_slice(cur_buf2);
        }

        let mut pol_conv_out = [0.0f32; 4 * CELLS];
        pol_conv.forward(cur_buf1, &mut pol_conv_out);
        for x in pol_conv_out.iter_mut() {
            *x = x.max(0.0);
        }

        let mut all_logits = [0.0f32; NUM_ACTIONS];
        pol_fc.forward(&pol_conv_out, &mut all_logits);

        let mut out = Vec::with_capacity(moves.len());
        for mv in moves {
            let a = mv.action_id() as usize;
            if a < NUM_ACTIONS {
                out.push(all_logits[a]);
            } else {
                out.push(0.0);
            }
        }
        Some(out)
    }
}

#[derive(Clone)]
pub struct MlpNet {
    pub h1: usize,
    pub h2: usize,
    pub pol: Option<(Vec<f32>, Vec<f32>)>,
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
    pub w3: Vec<f32>,
    pub b3: f32,
}

impl MlpNet {
    fn trunk(&self, pos: &Position, scratch: &mut Vec<u16>) -> Vec<f32> {
        sparse_indices(pos, scratch);
        let dense = dense_features(pos);
        let mut a1 = self.b1.clone();
        for &i in scratch.iter() {
            let row = &self.w1[i as usize * self.h1..][..self.h1];
            for (a, w) in a1.iter_mut().zip(row) {
                *a += w;
            }
        }
        for (d, &x) in dense.iter().enumerate() {
            if x == 0.0 {
                continue;
            }
            let row = &self.w1[(N_SPARSE + d) * self.h1..][..self.h1];
            for (a, w) in a1.iter_mut().zip(row) {
                *a += w * x;
            }
        }
        for a in a1.iter_mut() {
            *a = a.max(0.0);
        }
        a1
    }

    pub fn policy(&self, pos: &Position, moves: &[crate::moves::Move], scratch: &mut Vec<u16>) -> Option<Vec<f32>> {
        let (pw, pb) = self.pol.as_ref()?;
        let a1 = self.trunk(pos, scratch);
        let mut out = Vec::with_capacity(moves.len());
        for mv in moves {
            let a = mv.action_id() as usize;
            if a >= NUM_ACTIONS {
                out.push(0.0);
                continue;
            }
            let mut z = pb[a];
            for (i, &x) in a1.iter().enumerate() {
                if x != 0.0 {
                    z += x * pw[i * NUM_ACTIONS + a];
                }
            }
            out.push(z);
        }
        Some(out)
    }

    pub fn value(&self, pos: &Position, scratch: &mut Vec<u16>) -> f32 {
        sparse_indices(pos, scratch);
        let dense = dense_features(pos);

        let mut a1 = self.b1.clone();
        for &i in scratch.iter() {
            let row = &self.w1[i as usize * self.h1..][..self.h1];
            for (a, w) in a1.iter_mut().zip(row) {
                *a += w;
            }
        }
        for (d, &x) in dense.iter().enumerate() {
            if x == 0.0 {
                continue;
            }
            let row = &self.w1[(N_SPARSE + d) * self.h1..][..self.h1];
            for (a, w) in a1.iter_mut().zip(row) {
                *a += w * x;
            }
        }
        for a in a1.iter_mut() {
            *a = a.max(0.0);
        }

        let mut a2 = self.b2.clone();
        for (i, &x) in a1.iter().enumerate() {
            if x == 0.0 {
                continue;
            }
            let row = &self.w2[i * self.h2..][..self.h2];
            for (a, w) in a2.iter_mut().zip(row) {
                *a += w * x;
            }
        }
        let mut z = self.b3;
        for (a, w) in a2.iter().zip(&self.w3) {
            z += a.max(0.0) * w;
        }
        1.0 / (1.0 + (-z).exp())
    }
}

#[derive(Clone)]
pub enum NetKind {
    Mlp(MlpNet),
    Res(ResNet),
}

pub struct Net {
    pub kind: NetKind,
}

impl Net {
    /// `ag.bin` veya `ag_res.bin` biçimini okur:
    /// - "QNN1": MLP değerlendirme ağı
    /// - "QNN2": 2B Uzamsal ResNet modeli
    pub fn from_bytes(buf: &[u8]) -> Option<Net> {
        if buf.len() < 16 {
            return None;
        }
        let magic = &buf[0..4];
        if magic == b"QNN1" {
            let rd_u32 = |o: usize| {
                u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]) as usize
            };
            let n_in = rd_u32(4);
            let h1 = rd_u32(8);
            let h2 = rd_u32(12);
            if n_in > N_IN || n_in < N_SPARSE {
                eprintln!("ag.bin girdi boyutu {n_in}, beklenen {N_IN} — yok sayılıyor");
                return None;
            }
            let eksik = N_IN - n_in;
            let need = n_in * h1 + h1 + h1 * h2 + h2 + h2 + 1;
            if buf.len() < 16 + need * 4 {
                return None;
            }
            let mut o = 16;
            fn take(buf: &[u8], o: &mut usize, n: usize) -> Vec<f32> {
                let base = *o;
                let out = (0..n)
                    .map(|i| f32::from_le_bytes(buf[base + i * 4..base + i * 4 + 4].try_into().unwrap()))
                    .collect();
                *o += n * 4;
                out
            }
            let mut w1 = take(buf, &mut o, n_in * h1);
            if eksik > 0 {
                w1.resize(N_IN * h1, 0.0);
                eprintln!("  (eski model: {eksik} yeni ozellik sifirlandi)");
            }
            let b1 = take(buf, &mut o, h1);
            let w2 = take(buf, &mut o, h1 * h2);
            let b2 = take(buf, &mut o, h2);
            let w3 = take(buf, &mut o, h2);
            let b3 = take(buf, &mut o, 1)[0];

            let pol_len = h1 * NUM_ACTIONS + NUM_ACTIONS;
            let pol = if buf.len() >= o + pol_len * 4 {
                let pw = take(buf, &mut o, h1 * NUM_ACTIONS);
                let pb = take(buf, &mut o, NUM_ACTIONS);
                eprintln!("  (politika başı da var)");
                Some((pw, pb))
            } else {
                None
            };
            Some(Net {
                kind: NetKind::Mlp(MlpNet { h1, h2, pol, w1, b1, w2, b2, w3, b3 }),
            })
        } else if magic == b"QNN2" || magic == b"QNN3" {
            let is_qnn3 = magic == b"QNN3";
            let rd_u32 = |o: usize| {
                u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]) as usize
            };
            let channels = rd_u32(4);
            let n_blocks = rd_u32(8);
            let has_policy = rd_u32(12) != 0;

            let mut o = 16;
            fn take(buf: &[u8], o: &mut usize, n: usize) -> Vec<f32> {
                let base = *o;
                let out = (0..n)
                    .map(|i| f32::from_le_bytes(buf[base + i * 4..base + i * 4 + 4].try_into().unwrap()))
                    .collect();
                *o += n * 4;
                out
            }

            let init_w = take(buf, &mut o, channels * SPATIAL_CHANNELS * 9);
            let init_b = take(buf, &mut o, channels);
            let init_conv = Conv2d3x3 {
                in_c: SPATIAL_CHANNELS,
                out_c: channels,
                weights: init_w,
                bias: init_b,
            };

            let mut blocks = Vec::with_capacity(n_blocks);
            for _ in 0..n_blocks {
                let c1_w = take(buf, &mut o, channels * channels * 9);
                let c1_b = take(buf, &mut o, channels);
                let c2_w = take(buf, &mut o, channels * channels * 9);
                let c2_b = take(buf, &mut o, channels);
                blocks.push(ResBlock {
                    conv1: Conv2d3x3 { in_c: channels, out_c: channels, weights: c1_w, bias: c1_b },
                    conv2: Conv2d3x3 { in_c: channels, out_c: channels, weights: c2_w, bias: c2_b },
                });
            }

            let val_cw = take(buf, &mut o, 2 * channels);
            let val_cb = take(buf, &mut o, 2);
            let val_conv = Conv2d1x1 { in_c: channels, out_c: 2, weights: val_cw, bias: val_cb };

            let val_f1w = take(buf, &mut o, 32 * 162);
            let val_f1b = take(buf, &mut o, 32);
            let val_fc1 = Linear { in_f: 162, out_f: 32, weights: val_f1w, bias: val_f1b };

            let val_f2w = take(buf, &mut o, 1 * 32);
            let val_f2b = take(buf, &mut o, 1);
            let val_fc2 = Linear { in_f: 32, out_f: 1, weights: val_f2w, bias: val_f2b };

            let (pol_conv, pol_fc) = if has_policy {
                let pol_cw = take(buf, &mut o, 4 * channels);
                let pol_cb = take(buf, &mut o, 4);
                let pconv = Conv2d1x1 { in_c: channels, out_c: 4, weights: pol_cw, bias: pol_cb };

                let pol_fw = take(buf, &mut o, NUM_ACTIONS * 324);
                let pol_fb = take(buf, &mut o, NUM_ACTIONS);
                let pfc = Linear { in_f: 324, out_f: NUM_ACTIONS, weights: pol_fw, bias: pol_fb };
                eprintln!("  (ResNet politika başı yüklendi)");
                (Some(pconv), Some(pfc))
            } else {
                (None, None)
            };

            let (moves_conv, moves_fc1, moves_fc2, delta_conv, delta_fc1, delta_fc2) = if is_qnn3 {
                let mc_w = take(buf, &mut o, 1 * channels);
                let mc_b = take(buf, &mut o, 1);
                let mconv = Conv2d1x1 { in_c: channels, out_c: 1, weights: mc_w, bias: mc_b };
                let mf1_w = take(buf, &mut o, 32 * 81);
                let mf1_b = take(buf, &mut o, 32);
                let mfc1 = Linear { in_f: 81, out_f: 32, weights: mf1_w, bias: mf1_b };
                let mf2_w = take(buf, &mut o, 1 * 32);
                let mf2_b = take(buf, &mut o, 1);
                let mfc2 = Linear { in_f: 32, out_f: 1, weights: mf2_w, bias: mf2_b };

                let dc_w = take(buf, &mut o, 1 * channels);
                let dc_b = take(buf, &mut o, 1);
                let dconv = Conv2d1x1 { in_c: channels, out_c: 1, weights: dc_w, bias: dc_b };
                let df1_w = take(buf, &mut o, 32 * 81);
                let df1_b = take(buf, &mut o, 32);
                let dfc1 = Linear { in_f: 81, out_f: 32, weights: df1_w, bias: df1_b };
                let df2_w = take(buf, &mut o, 1 * 32);
                let df2_b = take(buf, &mut o, 1);
                let dfc2 = Linear { in_f: 32, out_f: 1, weights: df2_w, bias: df2_b };

                eprintln!("  (KataGo aciliyet & yol farkı yardımcı başlıkları yüklendi)");
                (Some(mconv), Some(mfc1), Some(mfc2), Some(dconv), Some(dfc1), Some(dfc2))
            } else {
                (None, None, None, None, None, None)
            };

            Some(Net {
                kind: NetKind::Res(ResNet {
                    channels,
                    init_conv,
                    blocks,
                    val_conv,
                    val_fc1,
                    val_fc2,
                    pol_conv,
                    pol_fc,
                    moves_conv,
                    moves_fc1,
                    moves_fc2,
                    delta_conv,
                    delta_fc1,
                    delta_fc2,
                }),
            })
        } else {
            None
        }
    }

    pub fn has_policy(&self) -> bool {
        match &self.kind {
            NetKind::Mlp(m) => m.pol.is_some(),
            NetKind::Res(r) => r.pol_fc.is_some(),
        }
    }

    pub fn policy(&self, pos: &Position, moves: &[crate::moves::Move], scratch: &mut Vec<u16>) -> Option<Vec<f32>> {
        match &self.kind {
            NetKind::Mlp(m) => m.policy(pos, moves, scratch),
            NetKind::Res(r) => r.policy(pos, moves),
        }
    }

    pub fn value(&self, pos: &Position, scratch: &mut Vec<u16>) -> f32 {
        match &self.kind {
            NetKind::Mlp(m) => m.value(pos, scratch),
            NetKind::Res(r) => r.value(pos),
        }
    }
}

static NET: OnceLock<Option<Net>> = OnceLock::new();

/// Ada gore yuklenmis aglar. `net()` tek bir varsayilan ag veriyordu ve
/// bu yuzden iki agi karsilastirmak icin iki ayri surec calistirmak
/// gerekiyordu; her ikisini de ayni zayif tabana karsi olcup aradaki
/// farki cikarmaya calisiyorduk. Tavan etkisi (ikisi de %88 kazaninca
/// fark olculemiyor) bu dolayli yolun bedeliydi. Burasi ayni surecte
/// birden fazla ag tutuyor, boylece `ag=A` ve `ag=B` dogrudan
/// karsilasabiliyor.
///
/// Yuklenen ag `Box::leak` ile sizdiriliyor: sayilari bir avuc (olcum
/// basina 2-3 dosya) ve `&'static` olmasi `Config`'in `Copy` kalmasini
/// sagliyor, ki paralel arama buna dayaniyor.
static NETS: Mutex<Option<HashMap<String, Option<&'static Net>>>> = Mutex::new(None);

/// Belirli bir dosyadaki agi yukler (bir kere; sonrasi onbellekten).
/// Dosya okunamazsa `None` — cagiran taraf dogrusal degerlendirmeye duser.
pub fn net_from(yol: &str) -> Option<&'static Net> {
    let anahtar = yol.to_string();
    let mut kilit = NETS.lock().ok()?;
    let harita = kilit.get_or_insert_with(HashMap::new);
    if let Some(v) = harita.get(&anahtar) {
        return *v;
    }
    let sonuc = match std::fs::read(&anahtar) {
        Ok(buf) => match Net::from_bytes(&buf) {
            Some(n) => {
                match &n.kind {
                    NetKind::Mlp(m) => eprintln!("ag yuklendi: {anahtar} (MLP: {} -> {} -> {} -> 1)", N_IN, m.h1, m.h2),
                    NetKind::Res(r) => eprintln!("ag yuklendi: {anahtar} (2D ResNet: 13 kanallı, {} kanal, {} blok)", r.channels, r.blocks.len()),
                }
                Some(&*Box::leak(Box::new(n)))
            }
            None => {
                eprintln!("ag okunamadi (bicim): {anahtar}");
                None
            }
        },
        Err(e) => {
            eprintln!("ag acilamadi: {anahtar} ({e})");
            None
        }
    };
    harita.insert(anahtar, sonuc);
    sonuc
}

/// Çalışma dizinindeki (ya da exe'nin yanındaki) `ag.bin` dosyasını yükler.
/// Dosya yoksa `None` — motor doğrusal değerlendirmeyle çalışmaya devam eder.
/// Yeniden derlemeden model değiştirilebilsin diye çalışma zamanında okunuyor.
pub fn net() -> Option<&'static Net> {
    NET.get_or_init(|| {
        // QAG ortam degiskeni: ayni derlemeyle farkli aglari karsilastirmak
        // icin. Ag bir kere yukleniyor (OnceLock), yani tek surecte tek ag;
        // iki agi karsilastirmak icin iki surec calistiriyoruz.
        let mut yollar: Vec<std::path::PathBuf> = Vec::new();
        if let Ok(v) = std::env::var("QAG") {
            if !v.trim().is_empty() {
                yollar.push(std::path::PathBuf::from(v));
            }
        }
        yollar.push(std::path::PathBuf::from("ag.bin"));
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                yollar.push(dir.join("ag.bin"));
                yollar.push(dir.join("../../ag.bin"));
            }
        }
        for y in yollar {
            if let Ok(buf) = std::fs::read(&y) {
                match Net::from_bytes(&buf) {
                    Some(n) => {
                        match &n.kind {
                            NetKind::Mlp(m) => eprintln!("ag.bin yüklendi (MLP: {} -> {} -> {} -> 1)", N_IN, m.h1, m.h2),
                            NetKind::Res(r) => eprintln!("ag.bin yüklendi (2D ResNet: 13 kanallı, {} kanal, {} blok)", r.channels, r.blocks.len()),
                        }
                        return Some(n);
                    }
                    None => eprintln!("ag.bin okunamadı: {}", y.display()),
                }
            }
        }
        None
    })
    .as_ref()
}
