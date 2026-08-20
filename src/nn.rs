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
use crate::board::Position;
use crate::heuristics::{features, threat_features, NUM_FEATURES, NUM_THREAT};
use crate::moves::NUM_ACTIONS;
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

pub struct Net {
    pub h1: usize,
    pub h2: usize,
    /// Politika başı: gövdenin ilk katmanını paylaşıyor, çıkışı 140 aksiyon.
    /// `None` ise motorun elle yazılmış hamle sıralaması kullanılıyor.
    pol: Option<(Vec<f32>, Vec<f32>)>, // (w [h1][NUM_ACTIONS], b [NUM_ACTIONS])
    /// [N_IN][h1] satır büyük: seyrek satır toplamı bitişik bellekte.
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>, // [h1][h2]
    b2: Vec<f32>,
    w3: Vec<f32>, // [h2]
    b3: f32,
}

impl Net {
    /// `ag.bin` biçimi: "QNN1", u32 n_in, u32 h1, u32 h2, sonra f32 diziler.
    pub fn from_bytes(buf: &[u8]) -> Option<Net> {
        if buf.len() < 16 || &buf[0..4] != b"QNN1" {
            return None;
        }
        let rd_u32 =
            |o: usize| u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]) as usize;
        let n_in = rd_u32(4);
        let h1 = rd_u32(8);
        let h2 = rd_u32(12);
        // Eski model dosyalarini kabul et: girdi listesinin sonuna yeni yogun
        // ozellik eklendiginde (tehdit ozellikleri gibi) eski agin agirlik
        // matrisi kisa kaliyor. Eksik satirlari sifirla dolduruyoruz — eski
        // ag aynen calismaya devam ediyor, yeni ozellikleri sadece gormuyor.
        // Boyle olmasa kaynak guncellemesi calisan modeli oldururdu.
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
        // Closure yerine yardimci fonksiyon: ofseti disaridan da okumamiz
        // gerekiyor (politika basi dosyada var mi diye bakmak icin).
        fn take(buf: &[u8], o: &mut usize, n: usize) -> Vec<f32> {
            let base = *o;
            let out = (0..n)
                .map(|i| {
                    f32::from_le_bytes(buf[base + i * 4..base + i * 4 + 4].try_into().unwrap())
                })
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

        // Politika başı isteğe bağlı: dosyada yer kaldıysa oradan okunuyor.
        // Böylece yalnızca değer başı olan eski dosyalar da çalışmaya
        // devam ediyor.
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
            h1,
            h2,
            pol,
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
        })
    }

    pub fn has_policy(&self) -> bool {
        self.pol.is_some()
    }

    /// İlk katmanın çıktısı (ReLU sonrası). Değer ve politika bunu paylaşıyor.
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

    /// Verilen hamleler için politika logitleri. Gövdeyi paylaştığı için
    /// ek maliyet yalnızca h1 x hamle sayısı kadar çarpma.
    pub fn policy(
        &self,
        pos: &Position,
        moves: &[crate::moves::Move],
        scratch: &mut Vec<u16>,
    ) -> Option<Vec<f32>> {
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

    /// Sıradaki oyuncunun kazanma olasılığı, [0,1].
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

static NET: OnceLock<Option<Net>> = OnceLock::new();

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
                        eprintln!("ag.bin yüklendi ({} -> {} -> {} -> 1)", N_IN, n.h1, n.h2);
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
