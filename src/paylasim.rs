//! Paylaşımlı ağaç: bütün thread'ler tek bir MCTS ağacında çalışır.
//!
//! Kök paralelliğinde (`mcts::search_parallel_with`) her thread kendi
//! ağacını kuruyordu. Değer yaprağı deterministik olduğu için 32 ağaç
//! neredeyse birebir aynı çıkıyordu: 4 thread'de bir hamlenin ziyareti tek
//! thread'in 4 katının %0,1 yakınındaydı. Yani tüm çekirdekler aynı aramayı
//! tekrar ediyor, 3 sn x 32 thread 3 sn x 1 thread kadar derin görüyordu.
//!
//! Burada düğüm alanları atomik; thread'ler aynı ağaçta iner, yaprağı
//! değerlendirir, sonucu yukarı yazar. Aynı dala yığılmasınlar diye her
//! thread indiği yola **sanal kayıp** ekler: dal, iniş bitene kadar o
//! thread'in gözünde kaybedilmiş görünür ve diğerleri başka dallara yönelir.
//!
//! Genişletme: düğümün `durum`u 0 -> 1 CAS'ı kazanan thread çocukları
//! ayırır (atomik sayaçla, önceden ayrılmış dizide), alanları yazar ve
//! `durum`u 2'ye çeker (Release). Diğerleri 1 görürse o düğümü yaprak sayar.
//!
//! Tek thread'in kurallarıyla aynı kalanlar: çocuk hamleleri ve prior'lar
//! (`Mcts::child_moves`), yaprak değeri (`Mcts::static_value`), PUCT ve FPU,
//! kanıt yayılımı, kökte hamle seçimi (`Mcts::pick_root_move`: tekrar
//! cezası, kalkan, umutsuz pozisyon). Kök paralelliğinin birleştirmesi
//! tekrar cezasını ve kalkanı uygulamıyordu; burada uygulanıyor.

use crate::board::Position;
use crate::mcts::{Mcts, RootMove, SearchStats};
use crate::moves::Move;
use std::sync::atomic::{AtomicBool, AtomicI8, AtomicU16, AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering::*};
use std::time::{Duration, Instant};

/// Değer toplamı sabit noktalı tutuluyor: f32 için CAS döngüsü kökte
/// 32 thread'le darboğaz olurdu, tamsayı `fetch_add` olmuyor.
const OLCEK: f64 = (1u64 << 24) as f64;
/// Bir thread'in indiği her düğüme eklediği sanal kayıp (ziyaret olarak).
const SANAL: u32 = 1;

const ACILMADI: u8 = 0;
const ACILIYOR: u8 = 1;
const ACIK: u8 = 2;
const BITTI: u8 = 3;

#[derive(Default)]
struct Dugum {
    mv: AtomicU16,
    n_children: AtomicU16,
    first_child: AtomicU32,
    /// Bu düğümde sıra kimde.
    side: AtomicU8,
    durum: AtomicU8,
    /// `mcts::Node::proof` ile aynı anlam: +1 sıradaki kazanır, -1 kaybeder.
    proof: AtomicI8,
    prior: AtomicU32,
    visits: AtomicU32,
    sanal: AtomicU32,
    /// Bu düğüme gelen hamleyi yapanın kazanç toplamı x OLCEK.
    value: AtomicU64,
}

impl Dugum {
    fn kur(&self, mv: Move, side: u8, prior: f32) {
        self.mv.store(mv.0, Relaxed);
        self.n_children.store(0, Relaxed);
        self.first_child.store(0, Relaxed);
        self.side.store(side, Relaxed);
        self.proof.store(0, Relaxed);
        self.prior.store(prior.to_bits(), Relaxed);
        self.visits.store(0, Relaxed);
        self.sanal.store(0, Relaxed);
        self.value.store(0, Relaxed);
        self.durum.store(ACILMADI, Release);
    }
    fn deger(&self) -> f64 {
        self.value.load(Relaxed) as f64 / OLCEK
    }
}

/// Hamleler arasında saklanan paylaşımlı ağaç.
pub struct Agac {
    dugumler: Box<[Dugum]>,
    dolu: AtomicUsize,
    tasti: AtomicBool,
    kok: usize,
    kok_pos: Option<Position>,
}

impl Agac {
    pub fn new(kapasite: usize) -> Agac {
        // Sıfırlı ayırma: işletim sistemi sayfaları dokunulana kadar
        // vermiyor, 1 GB'lık dizi anında hazır. Atomikler için sıfır geçerli.
        let dugumler = Box::<[Dugum]>::new_zeroed_slice(kapasite.max(1024));
        // SAFETY: Dugum yalnız atomik tamsayılardan oluşuyor; sıfır bit
        // deseni her biri için geçerli bir değer.
        let dugumler = unsafe { dugumler.assume_init() };
        Agac { dugumler, dolu: AtomicUsize::new(0), tasti: AtomicBool::new(false), kok: 0, kok_pos: None }
    }

    pub fn kapasite(&self) -> usize {
        self.dugumler.len()
    }

    fn sifirla(&mut self, pos: &Position) {
        self.dolu.store(1, Relaxed);
        self.tasti.store(false, Relaxed);
        self.kok = 0;
        self.dugumler[0].kur(Move(0), pos.side, 1.0);
        if pos.winner().is_some() {
            self.dugumler[0].durum.store(BITTI, Relaxed);
        }
        self.kok_pos = Some(*pos);
    }

    /// Önceki aramanın kökü bu pozisyonun kendisi, çocuğu ya da torunuysa
    /// o alt ağaçtan devam et. Eski kardeş dallar dizide yer kaplamaya devam
    /// ediyor; dizi yarıdan fazla dolduysa baştan kur.
    fn yeniden_kullan(&mut self, pos: &Position) -> bool {
        let Some(eski) = self.kok_pos else { return false };
        if self.dolu.load(Relaxed) > self.kapasite() / 2 {
            return false;
        }
        if eski.hash == pos.hash {
            return true;
        }
        let cocuklar = |agac: &Agac, i: usize| {
            let d = &agac.dugumler[i];
            if d.durum.load(Acquire) != ACIK {
                return 0..0;
            }
            let f = d.first_child.load(Relaxed) as usize;
            f..f + d.n_children.load(Relaxed) as usize
        };
        for c in cocuklar(self, self.kok) {
            let mut p1 = eski;
            p1.make(Move(self.dugumler[c].mv.load(Relaxed)));
            if p1.hash == pos.hash {
                self.kok = c;
                self.kok_pos = Some(*pos);
                return true;
            }
            for g in cocuklar(self, c) {
                let mut p2 = p1;
                p2.make(Move(self.dugumler[g].mv.load(Relaxed)));
                if p2.hash == pos.hash {
                    self.kok = g;
                    self.kok_pos = Some(*pos);
                    return true;
                }
            }
        }
        false
    }

    /// PUCT, sanal kayıplarla. `mcts::Mcts::select_child` ile aynı formül.
    fn sec(&self, ebeveyn: usize, c_puct: f32, fpu_reduction: f32) -> usize {
        let p = &self.dugumler[ebeveyn];
        let first = p.first_child.load(Relaxed) as usize;
        let count = p.n_children.load(Relaxed) as usize;
        let pn = p.visits.load(Relaxed) + p.sanal.load(Relaxed);
        let parent_q = if pn > 0 { 1.0 - p.deger() as f32 / pn as f32 } else { 0.5 };
        let fpu = (parent_q - fpu_reduction).clamp(0.0, 1.0);
        let sqrt_n = (pn.max(1) as f32).sqrt();

        let mut best = first;
        let mut best_score = f32::NEG_INFINITY;
        for i in first..first + count {
            let n = &self.dugumler[i];
            let proof = n.proof.load(Relaxed);
            if proof == -1 {
                return i;
            }
            let v = n.visits.load(Relaxed) + n.sanal.load(Relaxed);
            let q = if proof == 1 {
                0.0
            } else if v > 0 {
                // sanal kayıplar ziyarete eklenip değere eklenmediği için
                // inilmekte olan dal kaybediyor gibi görünür
                n.deger() as f32 / v as f32
            } else {
                fpu
            };
            let u = if proof == 1 {
                0.0
            } else {
                c_puct * f32::from_bits(n.prior.load(Relaxed)) * sqrt_n / (1.0 + v as f32)
            };
            if q + u > best_score {
                best_score = q + u;
                best = i;
            }
        }
        best
    }

    /// Tek iniş: seç, gerekirse genişlet, değerlendir, yukarı yaz.
    fn inis(&self, m: &mut Mcts, yol: &mut Vec<usize>) {
        let mut pos = self.kok_pos.expect("kök kurulmadı");
        yol.clear();
        yol.push(self.kok);
        let mut idx = self.kok;
        loop {
            let d = &self.dugumler[idx];
            match d.durum.load(Acquire) {
                BITTI => break,
                ACILIYOR => break,
                ACILMADI => {
                    if idx != self.kok && d.visits.load(Relaxed) < m.expand_threshold {
                        break;
                    }
                    if self.tasti.load(Relaxed) {
                        break; // bellek tavanı: genişletme, sadece değerlendir
                    }
                    if d.durum.compare_exchange(ACILMADI, ACILIYOR, Acquire, Relaxed).is_err() {
                        break;
                    }
                    let Some((moves, priors)) = m.child_moves(&pos, idx == self.kok) else {
                        d.durum.store(BITTI, Release);
                        break;
                    };
                    let k = moves.len();
                    let first = self.dolu.fetch_add(k, Relaxed);
                    if first + k > self.kapasite() {
                        self.tasti.store(true, Relaxed);
                        d.durum.store(ACILMADI, Release);
                        break;
                    }
                    let side = 1 - pos.side;
                    for (i, (&mv, &pr)) in moves.iter().zip(&priors).enumerate() {
                        self.dugumler[first + i].kur(mv, side, pr);
                    }
                    d.first_child.store(first as u32, Relaxed);
                    d.n_children.store(k as u16, Relaxed);
                    d.durum.store(ACIK, Release);
                }
                _ => {}
            }
            let ci = self.sec(idx, m.c_puct, m.fpu_reduction);
            let c = &self.dugumler[ci];
            c.sanal.fetch_add(SANAL, Relaxed);
            pos.make(Move(c.mv.load(Relaxed)));
            yol.push(ci);
            idx = ci;
            if pos.winner().is_some() {
                c.durum.store(BITTI, Release);
                break;
            }
            if c.visits.load(Relaxed) == 0 {
                break; // yeni düğüm, buradan değerlendir
            }
        }

        let leaf_side = pos.side as usize;
        let v = match pos.winner() {
            Some(w) => f32::from(w == leaf_side),
            None => m.static_value(&pos),
        } as f64;
        let kazanc = (v * OLCEK) as u64;
        let kayip = ((1.0 - v) * OLCEK) as u64;
        for (k, &n) in yol.iter().enumerate() {
            let d = &self.dugumler[n];
            d.visits.fetch_add(1, Relaxed);
            let mover = (1 - d.side.load(Relaxed)) as usize;
            d.value.fetch_add(if mover == leaf_side { kazanc } else { kayip }, Relaxed);
            if k > 0 {
                d.sanal.fetch_sub(SANAL, Relaxed);
            }
        }
        if m.use_solver {
            self.kanit_yay(yol);
        }
    }

    /// `Mcts::propagate_proof` ile aynı kurallar. Yarışlar zararsız: kanıt
    /// yalnız 0'dan bir işarete geçiyor, hiçbir thread geri almıyor.
    fn kanit_yay(&self, yol: &[usize]) {
        let Some(&yaprak) = yol.last() else { return };
        let y = &self.dugumler[yaprak];
        if y.durum.load(Acquire) == BITTI && y.proof.load(Relaxed) == 0 && y.n_children.load(Relaxed) == 0 {
            // kazanılmış pozisyon (ya da adaysız düğüm): sıradaki kaybetti
            y.proof.store(-1, Relaxed);
        }
        for k in (1..yol.len()).rev() {
            let cocuk = &self.dugumler[yol[k]];
            let ebeveyn = &self.dugumler[yol[k - 1]];
            if ebeveyn.proof.load(Relaxed) != 0 {
                break;
            }
            if cocuk.proof.load(Relaxed) == -1 {
                ebeveyn.proof.store(1, Relaxed);
                continue;
            }
            if ebeveyn.durum.load(Acquire) != ACIK {
                break;
            }
            let f = ebeveyn.first_child.load(Relaxed) as usize;
            let n = ebeveyn.n_children.load(Relaxed) as usize;
            if (f..f + n).all(|i| self.dugumler[i].proof.load(Relaxed) == 1) {
                ebeveyn.proof.store(-1, Relaxed);
            } else {
                break;
            }
        }
    }
}

/// Paylaşımlı ağaçla arama. `isciler[t]` thread t'nin bağlamı: ayarlar,
/// ağ tamponu, rastgele sayı üreteci; kendi ağaçları kullanılmıyor.
pub fn ara(
    agac: &mut Agac,
    pos: &Position,
    budget_ms: Option<u64>,
    iters: u32,
    isciler: &mut [Mcts],
    reuse: bool,
) -> (Option<Move>, SearchStats) {
    let start = Instant::now();
    if !(reuse && agac.yeniden_kullan(pos)) {
        agac.sifirla(pos);
    }
    let kok = &agac.dugumler[agac.kok];
    let tek_hamle = || {
        kok.durum.load(Acquire) == ACIK && kok.n_children.load(Relaxed) == 1
    };

    let toplam = AtomicU32::new(0);
    let budget = budget_ms.map(Duration::from_millis);
    {
        let agac = &*agac;
        let toplam = &toplam;
        std::thread::scope(|s| {
            for m in isciler.iter_mut() {
                s.spawn(move || {
                    let mut yol = Vec::with_capacity(64);
                    let mut yaptim = 0u32;
                    loop {
                        if let Some(b) = budget {
                            if yaptim % 64 == 0 && start.elapsed() >= b {
                                break;
                            }
                        } else if toplam.load(Relaxed) >= iters {
                            break;
                        }
                        agac.inis(m, &mut yol);
                        yaptim += 1;
                        toplam.fetch_add(1, Relaxed);
                        if yaptim % 64 == 0
                            && (tek_hamle() || agac.dugumler[agac.kok].proof.load(Relaxed) != 0)
                        {
                            break; // tek legal hamle ya da sonuç kanıtlandı
                        }
                    }
                });
            }
        });
    }

    let kok = &agac.dugumler[agac.kok];
    let mut kids = Vec::new();
    if kok.durum.load(Acquire) == ACIK {
        let f = kok.first_child.load(Relaxed) as usize;
        for i in f..f + kok.n_children.load(Relaxed) as usize {
            let d = &agac.dugumler[i];
            kids.push(RootMove {
                mv: Move(d.mv.load(Relaxed)),
                visits: d.visits.load(Relaxed),
                value: d.deger() as f32,
                prior: f32::from_bits(d.prior.load(Relaxed)),
                proof: d.proof.load(Relaxed),
            });
        }
    }
    let (best, mut top) = isciler[0].pick_root_move(pos, &kids);
    let win_rate = top.first().map(|t| t.2).unwrap_or(0.0);
    top.truncate(24);
    (
        best,
        SearchStats {
            rollouts: toplam.load(Relaxed),
            nodes: agac.dolu.load(Relaxed).min(agac.kapasite()),
            win_rate,
            elapsed_s: start.elapsed().as_secs_f64(),
            top,
        },
    )
}
