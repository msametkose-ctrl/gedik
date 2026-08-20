//! Aday hamle üretimi ve rollout politikası.
//!
//! Quoridor'da ham branching factor ~60 ve bunun neredeyse tamamı duvar
//! hamlesi. Saf MCTS bu yüzden çöküyor. Buradaki filtreler
//! gorisanson/quoridor-ai'nin yaklaşımını takip ediyor:
//!
//! * **Muhtemel duvarlar** — sadece piyonlara yakın, mevcut duvarlara komşu
//!   veya tahta kenarına dayanan duvarlar değerlendiriliyor.
//! * **Yarış modu** — rakibin duvarı kalmadıysa arama uzayı en kısa yol
//!   adımları + rakibin yolunu uzatan duvarlarla sınırlanıyor.
//! * **En kısa yol önce** — piyon hamleleri hedefe yaklaştırma sırasına göre
//!   diziliyor, MCTS ilk ziyaretleri bu sırada yapıyor.

use crate::bitboard::*;
use crate::board::{goal_row, Position};
use crate::moves::*;

/// Duvar aday filtresi. 128 ham duvar hamlesini tipik olarak 20-45'e indirir.
pub fn wall_is_candidate(pos: &Position, slot: usize, horizontal: bool) -> bool {
    let (r, c) = ((slot / WDIM) as i32, (slot % WDIM) as i32);

    // Tahta kenarına dayanan duvarlar bariyer kurmanın çekirdeği.
    if horizontal {
        if c == 0 || c == WDIM as i32 - 1 {
            return true;
        }
    } else if r == 0 || r == WDIM as i32 - 1 {
        return true;
    }

    // Piyonlara yakınlık.
    for p in 0..2usize {
        let i = pos.pawn[p] as usize;
        let (pr, pc) = (row_of(i) as i32, col_of(i) as i32);
        if (r - pr).abs() <= 2 && (c - pc).abs() <= 2 {
            return true;
        }
    }

    // Mevcut bir duvara komşuluk (yatay eksende 2, dikeyde 1 slot tolerans).
    for dr in -1..=1i32 {
        for dc in -2..=2i32 {
            let (rr, cc) = (r + dr, c + dc);
            if rr < 0 || rr >= WDIM as i32 || cc < 0 || cc >= WDIM as i32 {
                continue;
            }
            if pos.h_bit(rr as usize, cc as usize) || pos.v_bit(rr as usize, cc as usize) {
                return true;
            }
        }
    }
    false
}

/// Piyon hamlelerinin hedef hücrelerinin bitmask'i.
fn pawn_target_mask(pos: &Position, pawn: &[Move]) -> u128 {
    let mut m = 0u128;
    for &mv in pawn {
        if let MoveKind::Pawn(d) = mv.kind() {
            m |= bit(pos.pawn_target(d));
        }
    }
    m
}

/// Piyon hamlelerini hedefe yaklaştırma sırasına göre dizer (en iyi başa).
pub fn sort_pawn_moves(pos: &Position, m: &Masks, pawn: &mut Vec<Move>) {
    let g = goal_row(pos.side as usize);
    let field = m.distance_field(g);
    pawn.sort_by_key(|&mv| match mv.kind() {
        MoveKind::Pawn(d) => field[pos.pawn_target(d)],
        _ => u8::MAX,
    });
}

/// En kısa yol üzerindeki piyon hamleleri.
pub fn best_pawn_steps(pos: &Position, m: &Masks, pawn: &[Move]) -> Vec<Move> {
    let g = goal_row(pos.side as usize);
    let best = m.closest_to_row(pawn_target_mask(pos, pawn), g);
    pawn.iter()
        .copied()
        .filter(|&mv| match mv.kind() {
            MoveKind::Pawn(d) => best & bit(pos.pawn_target(d)) != 0,
            _ => false,
        })
        .collect()
}

/// Arama için aday hamleler. Piyon hamleleri her zaman başta ve en iyi
/// hedefe yaklaştıran önce.
pub fn candidate_moves(pos: &Position) -> Vec<Move> {
    candidate_moves_filtered(pos, true)
}

/// `filter = false` ise duvar aday filtresi devre dışı — 128 duvarın legal
/// olanlarının tamamı üretilir. Filtrenin gerçekten kazandırıp kazandırmadığını
/// ölçmek için.
pub fn candidate_moves_filtered(pos: &Position, filter: bool) -> Vec<Move> {
    let side = pos.side as usize;
    let opp = 1 - side;
    let m = pos.masks();

    let mut pawn = Vec::with_capacity(6);
    pos.gen_pawn_moves(&m, &mut pawn);

    // ---- Yarış modu: rakip artık engelleyemez.
    if pos.walls[opp] == 0 {
        let mut out = best_pawn_steps(pos, &m, &pawn);
        if out.is_empty() {
            out = pawn.clone();
        }
        if pos.walls[side] > 0 {
            // Sadece rakibin yolunu gerçekten uzatan duvarlar anlamlı.
            let base = m
                .distance_to_row(pos.pawn[opp] as usize, goal_row(opp))
                .unwrap_or(0);
            for horizontal in [true, false] {
                for slot in 0..WSLOTS {
                    if filter && !wall_is_candidate(pos, slot, horizontal) {
                        continue;
                    }
                    if !pos.wall_legal(slot, horizontal) {
                        continue;
                    }
                    let b = 1u64 << slot;
                    let nm = if horizontal {
                        Masks::new(pos.h | b, pos.v)
                    } else {
                        Masks::new(pos.h, pos.v | b)
                    };
                    let after = nm
                        .distance_to_row(pos.pawn[opp] as usize, goal_row(opp))
                        .unwrap_or(0);
                    if after > base {
                        out.push(if horizontal {
                            Move::hwall(slot)
                        } else {
                            Move::vwall(slot)
                        });
                    }
                }
            }
        }
        return out;
    }

    // ---- Normal mod.
    sort_pawn_moves(pos, &m, &mut pawn);
    let mut out = pawn;
    out.reserve(48);
    if pos.walls[side] > 0 {
        for horizontal in [true, false] {
            for slot in 0..WSLOTS {
                if (!filter || wall_is_candidate(pos, slot, horizontal))
                    && pos.wall_legal(slot, horizontal)
                {
                    out.push(if horizontal {
                        Move::hwall(slot)
                    } else {
                        Move::vwall(slot)
                    });
                }
            }
        }
    }
    out
}

/// Hızlı xorshift64* — rollout'ların sıcak yolunda.
#[derive(Clone)]
pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed | 1)
    }
    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    #[inline(always)]
    pub fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() >> 32) as usize * n) >> 32
    }
    /// [0,1) aralığında f32.
    #[inline(always)]
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
    /// Bir bitmask içindeki rastgele bir bitin indeksi.
    #[inline(always)]
    pub fn pick_bit(&mut self, mask: u128) -> usize {
        let n = mask.count_ones() as usize;
        let mut k = self.below(n);
        let mut m = mask;
        loop {
            let i = m.trailing_zeros() as usize;
            if k == 0 {
                return i;
            }
            k -= 1;
            m &= m - 1;
        }
    }
}

/// Duvarlar tükendiğinde sonuç tempo ile belirlenir: sıradaki oyuncu
/// `d_me <= d_opp` ise kazanır. Rollout'ları erken kesen en değerli numara —
/// oyunun son ~40 yarım hamlesini tek karşılaştırmaya indiriyor.
///
/// Yaklaşımdır: piyon atlamaları mesafeyi bir hamle kısaltabilir. Rollout
/// sonucu olarak rastgele oyundan çok daha isabetli, ağaç içinde kesin
/// değerlendirme olarak kullanılmamalı.
#[inline]
pub fn race_winner(pos: &Position) -> usize {
    let side = pos.side as usize;
    let opp = 1 - side;
    let m = pos.masks();
    let d_me = m
        .distance_to_row(pos.pawn[side] as usize, goal_row(side))
        .unwrap_or(u32::MAX);
    let d_opp = m
        .distance_to_row(pos.pawn[opp] as usize, goal_row(opp))
        .unwrap_or(u32::MAX);
    if d_me <= d_opp {
        side
    } else {
        opp
    }
}

/// Yaprak değerlendirmesi: sıradaki oyuncunun kazanma olasılığı, [0,1].
///
/// Rollout'un yerine geçiyor. Gerekçe ölçümle: `wall_prob=0.30` ile saf
/// rollout, rakibi engelleyen bir duvarı (0.4526) kendine zarar veren bir
/// duvardan (0.4442) sadece 0.008 ile ayırıyor. 20k rollout 45 kök çocuğa
/// bölününce çocuk başına gürültü ±0.047, yani sinyalin ~6 katı — MCTS duvar
/// hamlelerini sıralayamıyor. Deterministik bir değerlendirme bu gürültüyü
/// tamamen kaldırıyor.
///
/// AlphaZero'da value head'in oturduğu yer de burası; ağı taktığımızda bu
/// fonksiyonun yerini alacak.
pub const NUM_FEATURES: usize = 12;

/// Değerlendirme özellikleri, **sıradaki oyuncunun** açısından.
///
/// Tasarımın merkezinde tek bir fikir var: **duvar koymak bir hamle harcatır.**
/// Bir duvar rakibin yolunu tipik olarak 2 uzatır ama sana 1 tempo'ya mal olur,
/// yani net kazancı ~+1 tempo. Ayrıca rakip hedefe `d_opp` hamle uzaktaysa
/// `d_opp`'dan fazla duvar koymanın anlamı yok — oyun o kadar sürmeyecek.
///
/// Bu iki sınır `usable_*` ve `opt`/`pes` özelliklerinde kodlu:
///
/// * `opt = tempo + kullanılabilir kendi duvarım` — bütün duvarlarımı en iyi
///   şekilde kullanırsam yarış nereye varır. `opt < 0` ise **hiçbir duvar beni
///   kurtarmaz**.
/// * `pes = tempo - kullanılabilir rakip duvarı` — rakip bütün duvarlarını
///   bana harcarsa. `pes > 0` ise **beni durduramaz**.
///
/// Önceki sürümde bunların hiçbiri yoktu; sadece ham duvar farkı vardı ve
/// model ona 1.57 tempo değer biçmişti. Sonuç: 18'e 8 geride olan ama 7 duvarı
/// olan bir pozisyona "%92 kazanıyorum" demek.
#[inline]
pub fn features(pos: &Position) -> [f32; NUM_FEATURES] {
    let side = pos.side as usize;
    let opp = 1 - side;
    let m = pos.masks();
    let d_me = m
        .distance_to_row(pos.pawn[side] as usize, goal_row(side))
        .unwrap_or(40) as i32;
    let d_opp = m
        .distance_to_row(pos.pawn[opp] as usize, goal_row(opp))
        .unwrap_or(40) as i32;
    let w_me = pos.walls[side] as i32;
    let w_opp = pos.walls[opp] as i32;

    // Sıra bizde: yarışı `d_me <= d_opp` ile alırız.
    let tempo = d_opp - d_me;
    // Oyun bitmeden koyabileceğimiz duvar sayısı. Bu kapak sadece `opt`/`pes`
    // için anlamlı: rakip hedefe 2 hamle uzaktayken elindeki 8 duvarın 8'ini
    // de kullanamazsın. Ham duvar farkı (f2) kapaksız kalıyor, yoksa erken
    // oyunda iki taraf da kapağa dayanıp fark sıfırlanıyordu.
    let usable_me = w_me.min(d_opp);
    let usable_opp = w_opp.min(d_me);
    // Duvar başına net +1 tempo.
    let opt = tempo + usable_me;
    let pes = tempo - usable_opp;
    let phase = (w_me + w_opp) as f32 / 20.0;

    [
        1.0,                // 0  sabit
        tempo as f32 / 4.0, // 1  ham tempo
        // 2  duvar farkı — ama sadece yarış hâlâ kazanılabilirken.
        //    `opt < 0` iken duvarın hiçbir değeri yok; bunu modele
        //    söylemezsek doğrusal terim büyümeye devam ediyor ve arama
        //    "duvar harca, değer yükselsin" diye sömürüyor.
        if opt >= 0 {
            (usable_me - usable_opp) as f32 / 4.0
        } else {
            0.0
        },
        phase,                         // 3  oyun evresi
        tempo as f32 / 4.0 * phase,    // 4  tempo x evre
        (d_me + d_opp) as f32 / 16.0,  // 5  oyun ne kadar ilerledi
        f32::from(w_opp == 0),         // 6  rakip engelleyemez
        f32::from(w_me == 0),          // 7  biz engelleyemeyiz
        opt.clamp(-6, 6) as f32 / 6.0, // 8  iyimser yarış
        pes.clamp(-6, 6) as f32 / 6.0, // 9  kötümser yarış
        f32::from(opt < 0),            // 10 duvarlarım yetmez -> kayıp
        f32::from(pes > 0),            // 11 rakibin duvarı yetmez -> kazanç
    ]
}

/// Elle tahmin edilmiş ağırlıklar — karşılaştırma tabanı.
pub const WEIGHTS_HAND: [f32; NUM_FEATURES] =
    [0.0, 1.5, 1.0, 0.0, 0.0, 0.0, 0.3, -0.3, 2.0, 2.0, -3.0, 3.0];

/// 26.319 self-play pozisyonundan lojistik regresyonla öğrenilmiş ağırlıklar.
/// Doğrulama kümesinde doğruluk 0.667 -> 0.772, logloss 0.773 -> 0.461.
///
/// Öğrenilen en çarpıcı şey duvar üstünlüğünün katsayısı: elle 1.4 tahmin
/// etmiştim, gerçek değer 4.43. Quoridor'da elde fazladan duvar bulundurmak
/// tempo farkından çok daha değerliymiş.
/// 156.561 self-play pozisyonu + oyunun mantığından türetilmiş 1.608 kesin
/// örnek üzerinde lojistik regresyonla öğrenildi.
///
/// Üç şey olmadan model bu bölgeyi yanlış öğreniyordu:
///
/// 1. **Kesin örnekler.** Sadece self-play verisiyle "18'e 8 geride, 7 duvar"
///    pozisyonuna 0.42 diyordu; veride o kadar uç pozisyon az olduğu için
///    doğrusal model ılımlı örneklerden dışarı uzatıp duvara fazla değer
///    biçiyordu.
/// 2. **Duvar teriminin kapısı.** Yarış kazanılamaz durumdayken (`opt < 0`)
///    duvar farkı artık sayılmıyor; yoksa terim büyümeye devam ediyor ve
///    arama "duvar harca, değer yükselsin" diye sömürüyor.
/// 3. **Eşik göstergelerinin dondurulması.** "rakip duvarsız", "oyun evresi"
///    gibi terimler veriye iyi uyuyor ama değerlendirmede sıçrama yaratıyor:
///    arama, rakip son duvarını harcadığında değerin zıpladığını görüp o hattı
///    kovalıyordu. Sıfırda tutuluyorlar; taşıdıkları bilgi zaten yarış
///    terimlerinde var.
/// 4. **Egemenlik terimlerine alt sınır.** `kurtulamam` ve `kaybetmem`
///    oyunun mantığından gelen kesin ifadeler. Serbest bırakılınca veri
///    onları zayıflatıyordu (kusurlu rakibe karşı o pozisyonlardan bazen
///    dönülüyor), zayıf ceza da aramaya kaçış deliği açıyordu: motor cezanın
///    kalktığı bir yaprak bulup kaybedilmiş pozisyonu kazanılmış sanıyordu.
pub const WEIGHTS_LEARNED: [f32; NUM_FEATURES] = [
    0.28846, 0.97101, 2.4541, 0.0, 0.0, 0.068557, 0.0, 0.0, -1.3472, -0.96404, -2.5, 2.5,
];

/// Motorun kullandığı ağırlıklar.
///
/// **Öğrenilmiş ağırlıklar kullanılıyor.** Daha önce burada elle yazılmışlar
/// vardı ve bu yorum öğrenilmişleri neden göndermediğimizi anlatıyordu. O
/// gerekçe yanlıştı; kaydını tutuyorum çünkü hata öğreticiydi.
///
/// Reddetme gerekçem şu ölçümdü — kaybedilmiş tek bir pozisyonda arama
/// derinleştikçe değer ne yapıyor:
///
/// | ağırlık | 20 bin iter | 200 bin iter |
/// |---|---|---|
/// | öğrenilmiş | 0.470 | 0.306 |
/// | elle | 0.139 | **0.031** |
///
/// Elle yazılmışta arama kaybı doğruluyor, öğrenilmişte değer yüksek kalıyor.
/// Buradan "öğrenilmişte aramanın sömürdüğü bir kaçış deliği var, demek ki
/// zayıf" sonucunu çıkardım. **Kalibrasyon ölçümünü güç ölçümü sandım.**
///
/// Gerçek güç ölçümü (16 çekirdekli makinede, deney başına ~190 tamamlanmış
/// oyun, rastgele açılış x 2 renk):
///
/// | iterasyon | öğrenilmiş vs elle | Elo |
/// |---|---|---|
/// | 5 bin | 23-8 (%74) | +182 |
/// | 20 bin | 18-11 (%62) | +86 |
/// | **200 bin** | **149-32 (%82)** | **+267** |
///
/// Her bütçede kazanıyor, ve bütçe büyüdükçe farkı açıyor — yani "uzun
/// aramada tutarsızlık" korkusunun tam tersi oluyor. Kötü kalibre olmuş ama
/// daha iyi *sıralayan* bir değerlendirme, iyi kalibre olmuş ama körlüğü olan
/// birinden güçlü çıkıyor: aramanın ihtiyacı olan şey hamleleri doğru
/// sıralamak, olasılığı doğru tahmin etmek değil.
///
/// Kalibrasyon derdi yine de gerçek ve açık iş olarak duruyor: umutsuz
/// pozisyonda 20 bin iterasyonda 0.48 diyor (`tests/engine.rs` içindeki
/// `hopeless_race_is_not_called_winning` eşiği bu yüzden gevşek). Bunun
/// pratikteki bedeli mekik dokuma ve bitmeyen oyunlar.
///
/// Elle yazılmışlara `mcts:20000:w=0` ya da `mctsh:20000` ile dönülebiliyor.
pub const VALUE_WEIGHTS: [f32; NUM_FEATURES] = WEIGHTS_LEARNED;

/// Yaprak değerlendirmesi: sıradaki oyuncunun kazanma olasılığı, [0,1].
///
/// Rollout'un yerine geçiyor. Gerekçe ölçümle: `wall_prob=0.30` ile saf
/// rollout, rakibi engelleyen bir duvarı (0.4526) kendine zarar veren bir
/// duvardan (0.4442) sadece 0.008 ile ayırıyor. 20k iterasyon 45 kök çocuğa
/// bölününce çocuk başına gürültü ±0.047, yani sinyalin ~6 katı — MCTS duvar
/// hamlelerini sıralayamıyor. Deterministik bir değerlendirme bu gürültüyü
/// tamamen kaldırıyor.
///
/// AlphaZero'da value head'in oturduğu yer de burası.
#[inline]
pub fn value_to_move(pos: &Position) -> f32 {
    value_with(pos, &VALUE_WEIGHTS)
}

#[inline]
pub fn value_with(pos: &Position, w: &[f32; NUM_FEATURES]) -> f32 {
    let side = pos.side as usize;
    if let Some(win) = pos.winner() {
        return f32::from(win == side);
    }
    value_with_opt(pos, w, true)
}

/// `soft = false` eski davranış (doyan lojistik) — A/B ölçümü için.
#[inline]
pub fn value_with_opt(pos: &Position, w: &[f32; NUM_FEATURES], soft: bool) -> f32 {
    let side = pos.side as usize;
    if let Some(win) = pos.winner() {
        return f32::from(win == side);
    }
    let f = features(pos);
    let mut x = 0.0;
    for i in 0..NUM_FEATURES {
        x += w[i] * f[i];
    }
    logistic(if soft { squash(x) } else { x })
}

/// Cebirsel yumuşatma: `|x|` büyüdükçe ±SQUASH_LIMIT'e yaklaşır ama **asla**
/// düzleşmez.
///
/// Neden gerekli: özelliklerdeki `clamp(-6, 6)` ve lojistiğin doyması yüzünden
/// açık farkla önde (ya da geride) olan pozisyonlarda bütün hamleler tıpatıp
/// aynı değeri alıyordu. Ölçtüm: 21. plyde motorun en iyi 24 hamlesi arasındaki
/// statik değer farkı **0.0000**. Motor o noktadan sonra tamamen kör oynuyor —
/// hangi hamlenin daha hızlı kazandırdığını göremiyor, kendi önünü kapatan
/// duvarı ilerlemekle eşit puanlıyor. Bu fonksiyon sıralamayı her zaman
/// koruyor: x büyüdükçe çıktı da büyüyor, sadece gitgide daha az.
#[inline]
pub fn squash(x: f32) -> f32 {
    const LIMIT: f32 = 7.0;
    x / (1.0 + x.abs() / LIMIT)
}

#[inline]
fn logistic(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Rollout politikası: `wall_prob` olasılıkla rastgele muhtemel duvar,
/// aksi halde en kısa yola doğru adım.
pub fn playout_move(pos: &Position, rng: &mut Rng, wall_prob: f32) -> Option<Move> {
    let side = pos.side as usize;
    let m = pos.masks();

    if pos.walls[side] > 0 && rng.unit() < wall_prob {
        // Birkaç rastgele deneme; tutmazsa piyon hamlesine düş.
        for _ in 0..8 {
            let r = rng.next_u64();
            let slot = (r % WSLOTS as u64) as usize;
            let horizontal = r & (1 << 20) != 0;
            if wall_is_candidate(pos, slot, horizontal) && pos.wall_legal(slot, horizontal) {
                return Some(if horizontal {
                    Move::hwall(slot)
                } else {
                    Move::vwall(slot)
                });
            }
        }
    }

    let mut pawn = Vec::with_capacity(6);
    pos.gen_pawn_moves(&m, &mut pawn);
    if pawn.is_empty() {
        return None;
    }
    let targets = pawn_target_mask(pos, &pawn);
    let best = m.closest_to_row(targets, goal_row(side));
    let chosen = rng.pick_bit(if best != 0 { best } else { targets });
    pawn.into_iter().find(|&mv| match mv.kind() {
        MoveKind::Pawn(d) => pos.pawn_target(d) == chosen,
        _ => false,
    })
}

/// Tek bir rollout. Kazanan oyuncuyu döndürür.
pub fn playout(mut pos: Position, rng: &mut Rng, wall_prob: f32, max_ply: u16) -> usize {
    loop {
        if let Some(w) = pos.winner() {
            return w;
        }
        if pos.walls[0] == 0 && pos.walls[1] == 0 {
            return race_winner(&pos);
        }
        if pos.ply >= max_ply {
            return race_winner(&pos);
        }
        match playout_move(&pos, rng, wall_prob) {
            Some(mv) => {
                pos.make(mv);
            }
            None => return race_winner(&pos),
        }
    }
}

// ------------------------------------------------------------------ tehdit

/// `victim`in en kısa yolu üzerindeki hücre dizisi.
///
/// Mesafe alanından geriye yürüyerek çıkarılıyor: her adımda hedefe bir daha
/// yakın olan komşuya geç.
fn shortest_path_cells(m: &Masks, from: usize, goal: usize) -> Vec<usize> {
    let field = m.distance_field(goal);
    let mut cur = from;
    let mut out = vec![cur];
    let mut guard = 0;
    while field[cur] > 0 && guard < CELLS {
        guard += 1;
        let want = field[cur] - 1;
        let mut next = None;
        for d in 0..4usize {
            if !m.step_open(cur, d) {
                continue;
            }
            let t = (cur as i32 + DIR_DELTA[d]) as usize;
            if field[t] == want {
                next = Some(t);
                break;
            }
        }
        match next {
            Some(t) => {
                cur = t;
                out.push(cur);
            }
            None => break,
        }
    }
    out
}

/// Bir kenarı kesen duvar slotları: `(slot, yatay_mı)`.
fn walls_blocking(a: usize, b: usize) -> Vec<(usize, bool)> {
    let (ra, ca) = (row_of(a) as i32, col_of(a) as i32);
    let (rb, cb) = (row_of(b) as i32, col_of(b) as i32);
    let mut out = Vec::with_capacity(2);
    if ca == cb {
        // dikey komşuluk -> yatay duvar keser
        let r = ra.min(rb);
        for c in [ca - 1, ca] {
            if c >= 0 && c < WDIM as i32 && r >= 0 && r < WDIM as i32 {
                out.push((wslot(r as usize, c as usize), true));
            }
        }
    } else {
        // yatay komşuluk -> dikey duvar keser
        let c = ca.min(cb);
        for r in [ra - 1, ra] {
            if r >= 0 && r < WDIM as i32 && c >= 0 && c < WDIM as i32 {
                out.push((wslot(r as usize, c as usize), false));
            }
        }
    }
    out
}

/// `attacker`in **tek bir duvarla** `victim`in yolunu en fazla ne kadar
/// uzatabileceği.
///
/// Bu bilgi değerlendirmede hiç yoktu ve motorun en görünür kusuru buydu:
/// rakip 15 kare uzatacak bir hamle hazırlarken motor 3 kare kazandıran
/// duvarı koyuyordu. Kendi savunmasının değerini göremiyordu çünkü savunma
/// duvarının **anlık** kazancı sıfır.
///
/// Ucuz olmasının sebebi: victim'in en kısa yolunu kesmeyen bir duvar o
/// mesafeyi artıramaz. Dolayısıyla 128 duvara değil, yolun üzerindeki
/// kenarları kesen ~20 duvara bakmak yetiyor.
///
/// **Değerlendirmede kullanılmıyor, bilerek.** Değer fonksiyonuna özellik
/// olarak eklendi ve ölçüldü: eşit iterasyonda 12-12 (%50), yani hiçbir şey
/// katmadı — üstelik yaprak başına maliyeti verimliliği 2.8 kat düşürüyordu.
/// Sebebi sonradan anlaşıldı: arama zaten rakibin duvar hamlelerini deneyip
/// sonuçtaki uzun yolu görüyor, yani bilgi aramada mevcut. Ayrıca gerçek
/// oyundan alınan pozisyonlarda ölçtük ki tehdit ortaya çıktığında çoğu kez
/// **savunulacak hamle kalmamış** oluyor; hata daha erken yapılıyor.
///
/// Teşhis aracı olarak duruyor: `tehdit` ve `savunma` ikilileri bunu kullanıyor.
pub fn best_threat(pos: &Position, attacker: usize, victim: usize) -> u32 {
    if pos.walls[attacker] == 0 {
        return 0;
    }
    let m = pos.masks();
    let goal = goal_row(victim);
    let base = match m.distance_to_row(pos.pawn[victim] as usize, goal) {
        Some(d) => d,
        None => return 0,
    };
    let path = shortest_path_cells(&m, pos.pawn[victim] as usize, goal);

    let mut seen_h = 0u64;
    let mut seen_v = 0u64;
    let mut worst = 0u32;
    for w in path.windows(2) {
        for (slot, horizontal) in walls_blocking(w[0], w[1]) {
            let bit = 1u64 << slot;
            let seen = if horizontal { &mut seen_h } else { &mut seen_v };
            if *seen & bit != 0 {
                continue;
            }
            *seen |= bit;
            if !pos.wall_legal(slot, horizontal) {
                continue;
            }
            let nm = if horizontal {
                Masks::new(pos.h | bit, pos.v)
            } else {
                Masks::new(pos.h, pos.v | bit)
            };
            if let Some(after) = nm.distance_to_row(pos.pawn[victim] as usize, goal) {
                worst = worst.max(after.saturating_sub(base));
            }
        }
    }
    worst
}

/// Ağ için tehdit özellikleri: **yolum kesilebilir mi**.
///
/// Değerlendirmenin göremediği şey buydu. Şu pozisyonda motor `d8` oynayıp
/// kaybediyor, `a8h` oynasa kazanıyordu:
///
/// ```text
/// hamle   yolum   bana tehdit
///  a8h        5             0     <- rakip yolumu HIC uzatamiyor, yurur kazanir
///   d8        4             4     <- yol kisaldi ama rakip a8v ile 8 yapiyor
/// ```
///
/// `a8h`'nin sırrı: kendi duvarını, rakibin `a8v` için ihtiyaç duyduğu
/// merkeze koyuyor. İki duvar aynı merkezi paylaşamaz, rakibin tek kaynağı
/// yok oluyor. Mevcut 12 özellik yalnızca "yolum kaç" diyor, o yüzden `d8`
/// (yol 4) `a8h`'den (yol 5) iyi görünüyor.
///
/// Maliyet 0.7 µs/çağrı — tarama tahtanın tamamına değil yalnızca en kısa
/// yolun kenarlarına bakıyor. Bu özellikler **sadece ağa** gidiyor;
/// `features()` 12'de kalıyor ki doğrusal değerlendirme ve onun öğrenilmiş
/// ağırlıkları bozulmasın (nn=0 karşılaştırma tabanı olarak duruyor).
pub const NUM_THREAT: usize = 4;
pub fn threat_features(pos: &Position) -> [f32; NUM_THREAT] {
    let me = pos.side as usize;
    let opp = 1 - me;
    let bana = best_threat(pos, opp, me);
    let ona = best_threat(pos, me, opp);
    [
        (bana.min(12) as f32) / 8.0,
        (ona.min(12) as f32) / 8.0,
        // "Yolum mühürlü" — a8h'yi d8'den ayıran tek şey bu bit.
        f32::from(bana == 0 && pos.walls[opp] > 0),
        f32::from(ona == 0 && pos.walls[me] > 0),
    ]
}

// ---------------------------------------------------------------- prior'lar

/// Bir duvarın iki oyuncunun hedefe mesafesini nasıl değiştirdiği.
#[inline]
fn wall_deltas(pos: &Position, slot: usize, horizontal: bool, base: [u32; 2]) -> [i32; 2] {
    let b = 1u64 << slot;
    let nm = if horizontal {
        Masks::new(pos.h | b, pos.v)
    } else {
        Masks::new(pos.h, pos.v | b)
    };
    let mut out = [0i32; 2];
    for p in 0..2usize {
        let after = nm
            .distance_to_row(pos.pawn[p] as usize, goal_row(p))
            .unwrap_or(base[p]);
        out[p] = after as i32 - base[p] as i32;
    }
    out
}

/// Aday hamleler için policy prior'ı (toplamı 1).
///
/// AlphaZero'da bu, ağın policy head'i. Burada elle yazılmış ama aynı işi
/// görüyor: aramanın hangi hamlelere bakacağını şekillendiriyor.
///
/// Quoridor'da bu kritik, çünkü aday hamlelerin ~%90'ı duvar ve düz UCT
/// hepsine sırayla birer ziyaret dağıtıyor. Prior, rakibin yolunu gerçekten
/// uzatan duvarları öne alıp kendine zarar verenleri dibe atıyor.
pub fn move_priors(pos: &Position, moves: &[Move]) -> Vec<f32> {
    let side = pos.side as usize;
    let opp = 1 - side;
    let m = pos.masks();
    let base = [
        m.distance_to_row(pos.pawn[0] as usize, goal_row(0))
            .unwrap_or(99),
        m.distance_to_row(pos.pawn[1] as usize, goal_row(1))
            .unwrap_or(99),
    ];
    let best_targets = {
        let mut t = 0u128;
        for &mv in moves {
            if let MoveKind::Pawn(d) = mv.kind() {
                t |= bit(pos.pawn_target(d));
            }
        }
        m.closest_to_row(t, goal_row(side))
    };

    let mut logits = Vec::with_capacity(moves.len());
    for &mv in moves {
        let s = match mv.kind() {
            MoveKind::Pawn(d) => {
                if best_targets & bit(pos.pawn_target(d)) != 0 {
                    2.0
                } else {
                    -1.0
                }
            }
            MoveKind::HWall(slot) | MoveKind::VWall(slot) => {
                let horizontal = matches!(mv.kind(), MoveKind::HWall(_));
                let d = wall_deltas(pos, slot, horizontal, base);
                let theirs = d[opp] as f32; // rakibin yolu ne kadar uzadı
                let mine = d[side] as f32; // kendi yolum ne kadar uzadı
                if theirs > 0.0 {
                    0.6 * (theirs - 1.4 * mine) - 0.4
                } else {
                    // Rakibi şu an yavaşlatmayan duvar. Eski sürümde bunlara
                    // -3.0 veriliyordu, yani arama onlara pratikte hiç
                    // bakmıyordu — ve **savunma duvarları tam olarak bu
                    // gruptadır**: rakibin ileride yapacağı güçlü hamleyi
                    // önceden bozan duvarın anlık kazancı sıfırdır.
                    //
                    // Artık düşük ama sıfır olmayan bir ağırlık alıyorlar,
                    // böylece arama gerektiğinde onları da deneyebiliyor.
                    // Kendi yolunu uzatanlar yine dibe iniyor.
                    -1.6 - 0.8 * mine
                }
            }
        };
        logits.push(s);
    }

    softmax(&mut logits, 1.0);
    logits
}

fn softmax(x: &mut [f32], temp: f32) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = ((*v - max) / temp).exp();
        sum += *v;
    }
    let inv = if sum > 0.0 { 1.0 / sum } else { 1.0 };
    for v in x.iter_mut() {
        *v *= inv;
    }
}
