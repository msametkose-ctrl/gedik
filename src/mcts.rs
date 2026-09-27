//! Heuristic MCTS — UCT + aday filtresi + en kısa yol rollout politikası.
//!
//! Düğümler tek bir düz arena'da (`Vec<Node>`) tutuluyor; bir düğümün
//! çocukları bitişik yerleşiyor, dolayısıyla seçim döngüsü cache dostu.
//!
//! Değer konvansiyonu: `node.value`, o düğüme gelen hamleyi **oynayan**
//! oyuncunun açısından toplam sonuç. Böylece ebeveynden çocuk seçerken
//! kazanma oranı doğrudan sıradaki oyuncunun açısından okunuyor, işaret
//! çevirmeye gerek kalmıyor.

use crate::board::Position;
use crate::heuristics::{
    candidate_moves_filtered, exact_race_winner, move_priors, playout, value_with_opt, Rng,
    NUM_FEATURES, VALUE_WEIGHTS,
};
use crate::moves::Move;
use std::time::{Duration, Instant};

/// Bu kazanma oranının altında pozisyon pratikte kaybedilmiş sayılır ve
/// hamle seçimi istatistik yerine prior'a bırakılır.
const LOST_THRESHOLD: f32 = 0.03;

/// Yaprakta ne yapılacağı.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Leaf {
    /// Klasik rastgele rollout (gorisanson yaklaşımı).
    Rollout,
    /// Deterministik değerlendirme fonksiyonu. NN value head'i buraya girecek.
    Value,
}

#[derive(Clone, Copy)]
struct Node {
    mv: Move,
    first_child: u32,
    n_children: u16,
    /// Bu düğümde sıra kimde.
    side: u8,
    terminal: bool,
    visits: u32,
    value: f32,
    /// Policy prior — bu hamlenin ebeveynindeki olasilik agirligi.
    prior: f32,
    /// Kanıt durumu: `0` bilinmiyor, `+1` bu düğümde sıradaki oyuncu kesin
    /// kazanıyor, `-1` kesin kaybediyor. MCTS-Solver.
    proof: i8,
}

pub struct SearchStats {
    pub rollouts: u32,
    pub nodes: usize,
    pub win_rate: f32,
    pub elapsed_s: f64,
    /// En çok ziyaret edilen ilk birkaç hamle.
    pub top: Vec<(Move, u32, f32)>,
}

pub struct Mcts {
    nodes: Vec<Node>,
    path: Vec<u32>,
    rng: Rng,
    /// UCT keşif sabiti. gorisanson v0.3'te 0.2.
    pub c_uct: f32,
    /// Rollout'ta duvar koyma olasılığı (referans: 0.30).
    pub wall_prob: f32,
    pub max_ply: u16,
    /// Bir düğüm bu kadar ziyaret almadan çocukları açılmaz. Aday sayısı
    /// ~45 olduğu için eşik 1 olsaydı her rollout ~45 düğüm tahsis ederdi;
    /// 2 ile ağaç boyutu yarıya iniyor, güç ölçülebilir şekilde değişmiyor.
    pub expand_threshold: u32,
    pub leaf: Leaf,
    /// Duvar aday filtresi açık mı. Kapatınca 128 duvarın legal olanlarının
    /// tamamı aday oluyor — filtrenin katkısını ölçmek için.
    pub filter_walls: bool,
    /// PUCT + policy prior kullan (kapalıysa klasik UCT).
    pub use_priors: bool,
    /// MCTS-Solver açık mı (kanıtlanmış kazanç/kayıp yayılımı).
    pub use_solver: bool,
    /// PUCT keşif katsayısı.
    pub c_puct: f32,
    /// Ziyaret edilmemiş çocuk için ebeveyn değerinden indirim.
    pub fpu_reduction: f32,
    /// Ağaç bu düğüm sayısını aşınca genişleme durur (arama devam eder).
    /// Düğüm ~24 bayt; 6M ≈ 145 MB. Uzun analizlerde belleği sınırlar.
    pub max_nodes: usize,
    /// Değer fonksiyonu ağırlıkları (A/B ölçümü için değiştirilebilir).
    pub weights: [f32; NUM_FEATURES],
    /// Bir düğümde tutulan en fazla çocuk sayısı (prior'a göre en iyi K).
    /// Quoridor'da aday sayısı 40-60 arasında; hepsini tutmak ağacı
    /// derinleşmeden genişletiyor ve `max_nodes` tavanına erken çarpıyor.
    pub max_children: usize,
    /// Değerin doymasını engelleyen yumuşatma açık mı.
    pub soft_value: bool,
    /// `ag.bin` varsa sinir ağıyla değerlendir. Dosya yoksa sessizce
    /// doğrusal değerlendirmeye düşüyor — motor her hâlükârda çalışıyor.
    pub use_nn: bool,
    /// Bu arama icin ozel ag. `None` ise varsayilan `ag.bin` kullaniliyor.
    /// Iki agi ayni surecte, birbirine karsi oynatabilmek icin var.
    pub net: Option<&'static crate::nn::Net>,
    /// Seyrek indeks tamponu; her yaprakta yeniden ayırmamak için.
    nn_scratch: Vec<u16>,
    /// 0 = elle yazılmış hamle sıralaması, 1 = kökte ağın politikası,
    /// 2 = her düğümde ağın politikası.
    pub policy_mode: u8,
    /// Politika sicakligi. 1 = agin verdigi gibi; buyudukce dagilim
    /// duzlesiyor ve arama daha cok hamleye bakiyor. Olculdu: acilista
    /// agin en yuksek hamlesi 0.434, elle yazilmisin 0.209.
    pub policy_temp: f32,
    /// Kök prior'una karıştırılan gürültü oranı. Tek thread'de 0 (deterministik
    /// ve en güçlü); paralel aramada ağaçları farklılaştırmak için > 0.
    pub root_noise: f32,
    /// Ağacın kökünün Zobrist hash'i (ağaç yeniden kullanımını güvenle doğrulamak için).
    pub root_hash: u64,
    /// MCTS Transposition Table: aynı pozisyona varan dallar için değer ve kanıt önbelleği.
    pub tt: MctsTt,
    /// Oyun geçmişindeki son pozisyonların hash'leri (mekik / sonsuz tekrarı önlemek için).
    pub history: Vec<u64>,
    /// Gumbel AlphaZero araması açık mı (Sequential Halving).
    pub use_gumbel: bool,
    /// Sıfır Hata Kalkanı (BlunderGuard) açık mı.
    pub use_guard: bool,
    /// Kesin Son Oyun Minimax Çözücüsü (EndgameSolver) açık mı.
    pub use_endgame_solver: bool,
}

#[derive(Clone, Copy)]
pub struct MctsTtEntry {
    pub hash: u64,
    pub value: f32,
    pub proof: i8,
}

pub struct MctsTt {
    entries: Vec<Option<MctsTtEntry>>,
    mask: usize,
}

impl MctsTt {
    pub fn new(bits: usize) -> Self {
        let size = 1usize << bits;
        MctsTt {
            entries: vec![None; size],
            mask: size - 1,
        }
    }

    #[inline(always)]
    pub fn get(&self, hash: u64) -> Option<MctsTtEntry> {
        let idx = (hash as usize) & self.mask;
        match self.entries[idx] {
            Some(e) if e.hash == hash => Some(e),
            _ => None,
        }
    }

    #[inline(always)]
    pub fn insert(&mut self, hash: u64, value: f32, proof: i8) {
        let idx = (hash as usize) & self.mask;
        self.entries[idx] = Some(MctsTtEntry { hash, value, proof });
    }

    pub fn clear(&mut self) {
        self.entries.fill(None);
    }

    pub fn len(&self) -> usize {
        self.entries.iter().filter(|e| e.is_some()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Sayisal olarak guvenli softmax: en buyugu cikarmadan exp tasabiliyor.
fn softmax(logits: &[f32], t: f32) -> Vec<f32> {
    let t = if t > 0.01 { t } else { 1.0 };
    let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !mx.is_finite() {
        return vec![1.0 / logits.len().max(1) as f32; logits.len()];
    }
    let mut out: Vec<f32> = logits.iter().map(|z| ((z - mx) / t).exp()).collect();
    let sum: f32 = out.iter().sum();
    if sum > 0.0 {
        for v in out.iter_mut() {
            *v /= sum;
        }
    }
    out
}

impl Mcts {
    /// Bu aramanin kullanacagi ag: once `ag=` ile verilen, yoksa varsayilan.
    fn ag(&self) -> Option<&'static crate::nn::Net> {
        match self.net {
            Some(n) => Some(n),
            None => crate::nn::net(),
        }
    }

    pub fn new(seed: u64) -> Mcts {
        Mcts {
            nodes: Vec::with_capacity(1 << 16),
            path: Vec::with_capacity(128),
            rng: Rng::new(seed),
            c_uct: 0.2,
            wall_prob: 0.30,
            max_ply: 400,
            expand_threshold: 2,
            leaf: Leaf::Value,
            filter_walls: true,
            use_priors: true,
            use_solver: true,
            c_puct: 1.6,
            fpu_reduction: 0.2,
            max_nodes: 6_000_000,
            max_children: usize::MAX,
            soft_value: true,
            use_nn: true,
            net: None,
            nn_scratch: Vec::with_capacity(32),
            policy_mode: 0,
            policy_temp: 1.0,
            weights: VALUE_WEIGHTS,
            root_noise: 0.0,
            root_hash: 0,
            tt: MctsTt::new(18),
            history: Vec::new(),
            use_gumbel: false,
            use_guard: true,
            use_endgame_solver: true,
        }
    }

    pub fn reseed(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
    }

    /// Ağacı temizler ve sıfırlar.
    pub fn reset(&mut self) {
        self.nodes.clear();
        self.root_hash = 0;
        self.tt.clear();
    }

    /// Oynanan hamlenin alt ağacını yeni kök yaparak ağacı korur ve taşır (subtree reuse).
    ///
    /// Bu işlem, önceki aramanın analiz ettiği binlerce düğümü bir sonraki hamleye
    /// aktarır. Hamle ağaçta bulunamazsa veya pozisyon uyuşmazsa ağaç sıfırlanır.
    pub fn advance_tree(&mut self, pos: &Position, mv: Move) -> bool {
        if self.nodes.is_empty() || self.root_hash != pos.hash {
            self.nodes.clear();
            self.root_hash = 0;
            return false;
        }
        let root = self.nodes[0];
        if root.n_children == 0 || root.first_child == 0 {
            self.nodes.clear();
            self.root_hash = 0;
            return false;
        }
        let first = root.first_child as usize;
        let count = root.n_children as usize;
        let mut target_idx = None;
        for i in 0..count {
            if self.nodes[first + i].mv == mv {
                target_idx = Some(first + i);
                break;
            }
        }
        let Some(child_idx) = target_idx else {
            self.nodes.clear();
            self.root_hash = 0;
            return false;
        };

        // Subtree compaction (BFS ile alt ağacı yeni vektöre kopyala)
        let mut new_nodes = Vec::with_capacity(1024);
        let mut queue = std::collections::VecDeque::new();

        let mut new_root = self.nodes[child_idx];
        new_root.mv = Move(0);
        new_root.prior = 1.0;
        new_nodes.push(new_root);
        queue.push_back((child_idx, 0usize));

        while let Some((old_i, new_i)) = queue.pop_front() {
            let old_node = self.nodes[old_i];
            if old_node.n_children > 0 && old_node.first_child > 0 {
                let first_old = old_node.first_child as usize;
                let n_ch = old_node.n_children as usize;
                let first_new = new_nodes.len() as u32;
                new_nodes[new_i].first_child = first_new;

                for k in 0..n_ch {
                    let old_c_idx = first_old + k;
                    if old_c_idx < self.nodes.len() {
                        let child_node = self.nodes[old_c_idx];
                        let new_c_idx = new_nodes.len();
                        new_nodes.push(child_node);
                        if child_node.n_children > 0 {
                            queue.push_back((old_c_idx, new_c_idx));
                        }
                    }
                }
            } else {
                new_nodes[new_i].first_child = 0;
                new_nodes[new_i].n_children = 0;
            }
        }
        self.nodes = new_nodes;
        let mut next_pos = *pos;
        next_pos.make(mv);
        self.root_hash = next_pos.hash;
        true
    }

    /// Sabit rollout bütçesiyle arama.
    pub fn search_rollouts(
        &mut self,
        pos: &Position,
        rollouts: u32,
    ) -> (Option<Move>, SearchStats) {
        self.run(pos, rollouts, None)
    }

    /// Süre bütçesiyle arama.
    pub fn search_time(&mut self, pos: &Position, ms: u64) -> (Option<Move>, SearchStats) {
        self.run(pos, u32::MAX, Some(Duration::from_millis(ms)))
    }

    fn run(
        &mut self,
        pos: &Position,
        max_rollouts: u32,
        budget: Option<Duration>,
    ) -> (Option<Move>, SearchStats) {
        if self.use_gumbel {
            return self.search_gumbel(pos, max_rollouts, budget);
        }

        let start = Instant::now();
        let reuse = !self.nodes.is_empty() && self.root_hash == pos.hash;

        if !reuse {
            self.nodes.clear();
            self.root_hash = pos.hash;
            self.nodes.push(Node {
                mv: Move(0),
                first_child: 0,
                n_children: 0,
                side: pos.side,
                terminal: pos.winner().is_some(),
                visits: 0,
                value: 0.0,
                prior: 1.0,
                proof: 0,
            });
        }

        let mut done = 0u32;
        while done < max_rollouts {
            if let Some(b) = budget {
                if done % 128 == 0 && start.elapsed() >= b {
                    break;
                }
            }
            self.iterate(pos);
            done += 1;
            if self.nodes[0].n_children == 1 {
                break; // tek legal hamle, aramaya gerek yok
            }
            if self.nodes[0].proof != 0 && done % 64 == 0 {
                break; // sonuç kanıtlandı, aramaya devam etmek boşuna
            }
        }

        let root = self.nodes[0];
        let mut top: Vec<(Move, u32, f32)> = (0..root.n_children as u32)
            .map(|i| {
                let n = self.nodes[(root.first_child + i) as usize];
                let wr = if n.visits > 0 {
                    n.value / n.visits as f32
                } else {
                    0.0
                };
                (n.mv, n.visits, wr)
            })
            .collect();
        // En çok ziyaret edilen; eşitlikte kazanma oranı.
        // Mekik / 3-tekrar cezası: geçmişte görülmüş pozisyonlara dönen hamlelerin ziyaret önceliğini düşür.
        if !self.history.is_empty() {
            top.sort_by(|a, b| {
                let mut nxt_a = *pos;
                nxt_a.make(a.0);
                let rep_a = self.history.iter().filter(|&&h| h == nxt_a.hash).count();
                let score_a = a.1 as f32 * (1.0 - (rep_a as f32 * 0.35)).max(0.001);

                let mut nxt_b = *pos;
                nxt_b.make(b.0);
                let rep_b = self.history.iter().filter(|&&h| h == nxt_b.hash).count();
                let score_b = b.1 as f32 * (1.0 - (rep_b as f32 * 0.35)).max(0.001);

                score_b
                    .total_cmp(&score_a)
                    .then(b.1.cmp(&a.1))
                    .then(b.2.total_cmp(&a.2))
            });
        } else {
            top.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.total_cmp(&a.2)));
        }

        // Kanıtlanmış kazanan hamle varsa istatistiğe bakma, onu oyna.
        let proven = if self.use_solver {
            (0..root.n_children as u32)
                .map(|i| self.nodes[(root.first_child + i) as usize])
                .find(|n| n.proof == -1)
                .map(|n| n.mv)
        } else {
            None
        };

        let hopeless = top.first().map(|t| t.2).unwrap_or(0.0) < LOST_THRESHOLD;
        let fallback = if hopeless {
            (0..root.n_children as u32)
                .map(|i| self.nodes[(root.first_child + i) as usize])
                .filter(|n| n.proof != 1)
                .max_by(|a, b| a.prior.total_cmp(&b.prior))
                .or_else(|| {
                    (0..root.n_children as u32)
                        .map(|i| self.nodes[(root.first_child + i) as usize])
                        .max_by(|a, b| a.prior.total_cmp(&b.prior))
                })
                .map(|n| n.mv)
        } else {
            None
        };

        let guarded_best = if self.use_guard {
            crate::guard::filter_root_moves(pos, &top)
        } else {
            None
        };

        let best = proven
            .or(guarded_best)
            .or(fallback)
            .or_else(|| top.first().map(|t| t.0));
        let win_rate = top.first().map(|t| t.2).unwrap_or(0.0);
        top.truncate(24);

        let stats = SearchStats {
            rollouts: done,
            nodes: self.nodes.len(),
            win_rate,
            elapsed_s: start.elapsed().as_secs_f64(),
            top,
        };
        (best, stats)
    }

    /// Gumbel AlphaZero: Sequential Halving tabanlı matematiksel sıfır kör noktalı arama.
    pub fn search_gumbel(
        &mut self,
        pos: &Position,
        max_rollouts: u32,
        budget: Option<Duration>,
    ) -> (Option<Move>, SearchStats) {
        let start = Instant::now();
        self.nodes.clear();
        self.root_hash = pos.hash;
        self.nodes.push(Node {
            mv: Move(0),
            first_child: 0,
            n_children: 0,
            side: pos.side,
            terminal: pos.winner().is_some(),
            visits: 0,
            value: 0.0,
            prior: 1.0,
            proof: 0,
        });

        // Kök düğümü genişlet
        self.iterate(pos);
        let root = self.nodes[0];
        if root.n_children <= 1 {
            let mv = if root.n_children == 1 {
                Some(self.nodes[root.first_child as usize].mv)
            } else {
                None
            };
            return (
                mv,
                SearchStats {
                    rollouts: 1,
                    nodes: self.nodes.len(),
                    win_rate: 0.5,
                    elapsed_s: start.elapsed().as_secs_f64(),
                    top: Vec::new(),
                },
            );
        }

        let n_c = root.n_children as usize;
        let gumbel: Vec<f32> = (0..n_c)
            .map(|_| {
                let u = self.rng.unit().clamp(1e-6, 1.0 - 1e-6);
                -(-u.ln()).ln()
            })
            .collect();

        // En iyi m adayı seç (Gumbel Top-k)
        let m = n_c.min(16);
        let mut candidates: Vec<usize> = (0..n_c).collect();
        candidates.sort_by(|&a, &b| {
            let na = self.nodes[(root.first_child as usize) + a];
            let nb = self.nodes[(root.first_child as usize) + b];
            let za = na.prior.max(1e-6).ln() + gumbel[a];
            let zb = nb.prior.max(1e-6).ln() + gumbel[b];
            zb.total_cmp(&za)
        });
        candidates.truncate(m);

        let phases = if m > 8 {
            4
        } else if m > 4 {
            3
        } else if m > 2 {
            2
        } else {
            1
        };
        let mut done = 1u32;

        for phase in 0..phases {
            if candidates.len() <= 1 {
                break;
            }
            let phase_budget = (max_rollouts / (candidates.len() as u32 * phases as u32)).max(1);

            for &cand_idx in &candidates {
                let child_node_idx = (root.first_child as usize) + cand_idx;
                for _ in 0..phase_budget {
                    if let Some(b) = budget {
                        if done % 128 == 0 && start.elapsed() >= b {
                            break;
                        }
                    }
                    self.iterate_forced(pos, child_node_idx);
                    done += 1;
                }
            }

            // Gumbel Puanı: l_a + g_a + sigma(q_a)
            let max_visits = candidates
                .iter()
                .map(|&c| self.nodes[(root.first_child as usize) + c].visits)
                .max()
                .unwrap_or(1);

            candidates.sort_by(|&a, &b| {
                let na = self.nodes[(root.first_child as usize) + a];
                let nb = self.nodes[(root.first_child as usize) + b];
                let qa = if na.visits > 0 {
                    na.value / na.visits as f32
                } else {
                    0.5
                };
                let qb = if nb.visits > 0 {
                    nb.value / nb.visits as f32
                } else {
                    0.5
                };
                let c_visit = 50.0f32;
                let c_scale = 1.0f32;
                let sigma_a = (c_visit + max_visits as f32) * c_scale * (qa - 0.5);
                let sigma_b = (c_visit + max_visits as f32) * c_scale * (qb - 0.5);
                let score_a = na.prior.max(1e-6).ln() + gumbel[a] + sigma_a;
                let score_b = nb.prior.max(1e-6).ln() + gumbel[b] + sigma_b;
                score_b.total_cmp(&score_a)
            });

            // Sequential Halving: adayların yarısını ele
            if phase < phases - 1 {
                let next_len = ((candidates.len() + 1) / 2).max(1);
                candidates.truncate(next_len);
            }
        }

        let best_cand = candidates[0];
        let best_node = self.nodes[(root.first_child as usize) + best_cand];

        let mut top: Vec<(Move, u32, f32)> = (0..root.n_children as usize)
            .map(|i| {
                let n = self.nodes[(root.first_child as usize) + i];
                let wr = if n.visits > 0 {
                    n.value / n.visits as f32
                } else {
                    0.0
                };
                (n.mv, n.visits, wr)
            })
            .collect();
        top.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.total_cmp(&a.2)));

        let best_mv = if self.use_guard {
            crate::guard::filter_root_moves(pos, &top).or(Some(best_node.mv))
        } else {
            Some(best_node.mv)
        };
        let win_rate = if best_node.visits > 0 {
            best_node.value / best_node.visits as f32
        } else {
            0.5
        };
        top.truncate(24);

        let stats = SearchStats {
            rollouts: done,
            nodes: self.nodes.len(),
            win_rate,
            elapsed_s: start.elapsed().as_secs_f64(),
            top,
        };
        (best_mv, stats)
    }

    fn iterate_forced(&mut self, root_pos: &Position, child_idx: usize) {
        let mut pos = *root_pos;
        self.path.clear();
        self.path.push(0);
        self.path.push(child_idx as u32);
        let mv = self.nodes[child_idx].mv;
        pos.make(mv);

        let mut idx = child_idx;
        loop {
            if self.nodes[idx].terminal {
                break;
            }
            if self.nodes[idx].n_children == 0 {
                if self.nodes[idx].visits < self.expand_threshold {
                    break;
                }
                if self.nodes.len() >= self.max_nodes {
                    break;
                }
                let moves = candidate_moves_filtered(&pos, self.filter_walls);
                if moves.is_empty() {
                    self.nodes[idx].terminal = true;
                    break;
                }
                let priors = if self.use_priors {
                    let ag_politika =
                        self.policy_mode > 0 && self.ag().is_some_and(|n| n.has_policy());
                    let ham = if ag_politika {
                        self.ag()
                            .and_then(|n| n.policy(&pos, &moves, &mut self.nn_scratch))
                    } else {
                        None
                    };
                    match ham {
                        Some(logits) => softmax(&logits, self.policy_temp),
                        None => move_priors(&pos, &moves),
                    }
                } else {
                    vec![1.0 / moves.len() as f32; moves.len()]
                };

                let mut moves = moves;
                if moves.len() > self.max_children {
                    let mut idx_arr: Vec<usize> = (0..moves.len()).collect();
                    idx_arr.sort_by(|&a, &b| {
                        let pa = (!moves[a].is_wall(), priors[a]);
                        let pb = (!moves[b].is_wall(), priors[b]);
                        pb.0.cmp(&pa.0).then_with(|| pb.1.total_cmp(&pa.1))
                    });
                    moves = idx_arr
                        .into_iter()
                        .take(self.max_children)
                        .map(|i| moves[i])
                        .collect();
                }

                let first_child = self.nodes.len() as u32;
                let n_children = moves.len() as u16;
                let side = pos.side;
                for (m, pr) in moves.into_iter().zip(priors) {
                    self.nodes.push(Node {
                        mv: m,
                        first_child: 0,
                        n_children: 0,
                        side,
                        terminal: false,
                        visits: 0,
                        value: 0.0,
                        prior: pr,
                        proof: 0,
                    });
                }
                self.nodes[idx].first_child = first_child;
                self.nodes[idx].n_children = n_children;
                break;
            }

            let best_child = self.select_child(idx);
            self.path.push(best_child as u32);
            let mv = self.nodes[best_child].mv;
            pos.make(mv);
            idx = best_child;
        }

        self.eval_and_backprop(&pos, idx);
    }

    fn iterate(&mut self, root_pos: &Position) {
        let mut pos = *root_pos;
        self.path.clear();
        self.path.push(0);
        let mut idx = 0usize;

        loop {
            if self.nodes[idx].terminal {
                break;
            }
            if self.nodes[idx].n_children == 0 {
                // Kökü her zaman aç; diğerlerinde eşiği bekle.
                if idx != 0 && self.nodes[idx].visits < self.expand_threshold {
                    break;
                }
                if self.nodes.len() >= self.max_nodes {
                    break; // bellek tavanı — genişletme, sadece değerlendir
                }
                let moves = candidate_moves_filtered(&pos, self.filter_walls);
                if moves.is_empty() {
                    self.nodes[idx].terminal = true;
                    break;
                }
                let mut priors = if self.use_priors {
                    let ag_politika = self.policy_mode > 0
                        && (self.policy_mode > 1 || idx == 0)
                        && self.ag().is_some_and(|n| n.has_policy());
                    let ham = if ag_politika {
                        self.ag()
                            .and_then(|n| n.policy(&pos, &moves, &mut self.nn_scratch))
                    } else {
                        None
                    };
                    match ham {
                        Some(logits) => softmax(&logits, self.policy_temp),
                        None => move_priors(&pos, &moves),
                    }
                } else {
                    vec![1.0 / moves.len() as f32; moves.len()]
                };
                if idx == 0 && self.root_noise > 0.0 {
                    let eps = self.root_noise;
                    let mut noise: Vec<f32> = (0..priors.len())
                        .map(|_| -(self.rng.unit().max(1e-6)).ln())
                        .collect();
                    let sum: f32 = noise.iter().sum();
                    if sum > 0.0 {
                        for v in noise.iter_mut() {
                            *v /= sum;
                        }
                        for (p, n) in priors.iter_mut().zip(noise) {
                            *p = (1.0 - eps) * *p + eps * n;
                        }
                    }
                }
                let mut moves = moves;
                if moves.len() > self.max_children {
                    let mut idx_arr: Vec<usize> = (0..moves.len()).collect();
                    idx_arr.sort_by(|&a, &b| {
                        let pa = (!moves[a].is_wall(), priors[a]);
                        let pb = (!moves[b].is_wall(), priors[b]);
                        pb.0.cmp(&pa.0).then(pb.1.total_cmp(&pa.1))
                    });
                    idx_arr.truncate(self.max_children);
                    idx_arr.sort_unstable();
                    moves = idx_arr.iter().map(|&i| moves[i]).collect();
                    priors = idx_arr.iter().map(|&i| priors[i]).collect();
                    let sum: f32 = priors.iter().sum();
                    if sum > 0.0 {
                        for p in priors.iter_mut() {
                            *p /= sum;
                        }
                    }
                }

                let first = self.nodes.len() as u32;
                let child_side = 1 - pos.side;
                for (i, mv) in moves.iter().enumerate() {
                    self.nodes.push(Node {
                        mv: *mv,
                        first_child: 0,
                        n_children: 0,
                        side: child_side,
                        terminal: false,
                        visits: 0,
                        value: 0.0,
                        prior: priors[i],
                        proof: 0,
                    });
                }
                self.nodes[idx].first_child = first;
                self.nodes[idx].n_children = moves.len() as u16;
            }

            let ci = self.select_child(idx);
            pos.make(self.nodes[ci].mv);
            self.path.push(ci as u32);
            idx = ci;

            if pos.winner().is_some() {
                self.nodes[idx].terminal = true;
                break;
            }
            if self.nodes[idx].visits == 0 {
                break; // yeni düğüm -> buradan rollout
            }
        }

        self.eval_and_backprop(&pos, idx);
    }

    fn eval_and_backprop(&mut self, pos: &Position, leaf_idx: usize) {
        let leaf_side = pos.side as usize;
        let v_leaf = match pos.winner() {
            Some(w) => f32::from(w == leaf_side),
            None => {
                if let Some(cached) = self.tt.get(pos.hash) {
                    if cached.proof != 0 && self.nodes[leaf_idx].proof == 0 {
                        self.nodes[leaf_idx].proof = cached.proof;
                    }
                    cached.value
                } else if pos.walls[0] == 0 && pos.walls[1] == 0 {
                    let w = exact_race_winner(pos);
                    let val = f32::from(w == leaf_side);
                    self.tt.insert(pos.hash, val, 0);
                    val
                } else {
                    let val = match self.leaf {
                        Leaf::Rollout => {
                            let w = playout(*pos, &mut self.rng, self.wall_prob, self.max_ply);
                            f32::from(w == leaf_side)
                        }
                        Leaf::Value => {
                            if self.use_nn {
                                match self.ag() {
                                    Some(n) => n.value(pos, &mut self.nn_scratch),
                                    None => value_with_opt(pos, &self.weights, self.soft_value),
                                }
                            } else {
                                value_with_opt(pos, &self.weights, self.soft_value)
                            }
                        }
                    };
                    self.tt.insert(pos.hash, val, self.nodes[leaf_idx].proof);
                    val
                }
            }
        };

        for &n in &self.path {
            let node = &mut self.nodes[n as usize];
            node.visits += 1;
            let mover = (1 - node.side) as usize;
            node.value += if mover == leaf_side {
                v_leaf
            } else {
                1.0 - v_leaf
            };
        }

        if self.use_solver {
            self.propagate_proof();
        }
    }

    /// MCTS-Solver: kanıtlanmış sonucu ağaçta yukarı yayar.
    ///
    /// Kurallar basit ve kesin:
    /// * Terminal düğümde sıradaki oyuncu kaybetmiştir (hamleyi yapan kazandı).
    /// * Bir çocuğu `-1` (rakip kaybediyor) olan düğüm `+1`'dir — kazanan
    ///   hamlemiz var.
    /// * Bütün çocukları `+1` olan düğüm `-1`'dir — ne oynarsak oynayalım
    ///   rakip kazanıyor.
    ///
    /// İstatistiksel ortalamanın aksine bu **kesin**. Quoridor'un son
    /// bölümü saf bir yarışa dönüştüğü için kanıtlar hızla birikiyor ve
    /// motor sonuçlanmış pozisyonlarda hata yapmayı bırakıyor.
    fn propagate_proof(&mut self) {
        if self.path.is_empty() {
            return;
        }
        let leaf = *self.path.last().unwrap() as usize;
        if self.nodes[leaf].terminal && self.nodes[leaf].proof == 0 {
            self.nodes[leaf].proof = -1;
        }
        for k in (1..self.path.len()).rev() {
            let child = self.path[k] as usize;
            let parent = self.path[k - 1] as usize;
            if self.nodes[parent].proof != 0 {
                break; // zaten kanıtlı, yukarısı da güncel
            }
            if self.nodes[child].proof == -1 {
                self.nodes[parent].proof = 1;
                continue;
            }
            let p = self.nodes[parent];
            if p.n_children == 0 {
                break;
            }
            let first = p.first_child as usize;
            let all_lost = (0..p.n_children as usize).all(|i| self.nodes[first + i].proof == 1);
            if all_lost {
                self.nodes[parent].proof = -1;
            } else {
                break; // bu seviyede yeni bilgi yok
            }
        }
    }

    /// PUCT (AlphaZero seçim kuralı):
    ///
    /// ```text
    /// argmax_a  Q(a) + c_puct * P(a) * sqrt(N) / (1 + n(a))
    /// ```
    ///
    /// Düz UCT'den farkı, ziyaret edilmemiş çocukları tek tek dolaşmak zorunda
    /// olmaması. Quoridor'da aday hamlelerin ~%90'ı duvar; UCT hepsine sırayla
    /// birer ziyaret dağıtıp bütçeyi yiyordu. Prior sayesinde arama, rakibin
    /// yolunu gerçekten uzatan birkaç duvara yoğunlaşıyor.
    ///
    /// Ziyaret edilmemiş çocuğun Q'su için FPU (first-play urgency): ebeveynin
    /// değerinden sabit bir indirim. Optimistik sıfır yerine bunu kullanmak,
    /// zaten iyi bilinen bir hamleyi bırakıp rastgele duvar denemesini önlüyor.
    fn select_child(&self, parent: usize) -> usize {
        let p = self.nodes[parent];
        let first = p.first_child as usize;
        let count = p.n_children as usize;

        if !self.use_priors {
            // Karşılaştırma için klasik UCT yolu.
            for i in 0..count {
                if self.nodes[first + i].visits == 0 {
                    return first + i;
                }
            }
            let ln_n = (p.visits.max(1) as f32).ln();
            let mut best = first;
            let mut best_score = f32::NEG_INFINITY;
            for i in 0..count {
                let n = self.nodes[first + i];
                let v = n.visits as f32;
                let score = n.value / v + self.c_uct * (ln_n / v).sqrt();
                if score > best_score {
                    best_score = score;
                    best = first + i;
                }
            }
            return best;
        }

        // Ebeveynin kendi değeri, sıradaki oyuncunun açısına çevrilmiş.
        let parent_q = if p.visits > 0 {
            1.0 - p.value / p.visits as f32
        } else {
            0.5
        };
        let fpu = (parent_q - self.fpu_reduction).clamp(0.0, 1.0);
        let sqrt_n = (p.visits.max(1) as f32).sqrt();

        let mut best = first;
        let mut best_score = f32::NEG_INFINITY;
        for i in 0..count {
            let n = self.nodes[first + i];
            // Kanıtlanmış kazanç varsa doğrudan oraya; kanıtlanmış kayıp
            // çocuklara bakmanın anlamı yok (hepsi kayıpsa yine de birini seçeriz).
            if n.proof == -1 {
                return first + i;
            }
            let q = if n.proof == 1 {
                0.0
            } else if n.visits > 0 {
                n.value / n.visits as f32
            } else {
                fpu
            };
            let u = if n.proof == 1 {
                0.0
            } else {
                self.c_puct * n.prior * sqrt_n / (1.0 + n.visits as f32)
            };
            let score = q + u;
            if score > best_score {
                best_score = score;
                best = first + i;
            }
        }
        best
    }
}

/// Aramanın ayarları. `Copy` olması paralel aramada her thread'in kendi
/// `Mcts`'ini kurabilmesi için gerekli.
#[derive(Clone, Copy)]
pub struct Config {
    pub c_puct: f32,
    pub c_uct: f32,
    pub fpu_reduction: f32,
    pub expand_threshold: u32,
    pub max_nodes: usize,
    pub max_children: usize,
    pub soft_value: bool,
    pub use_nn: bool,
    pub net: Option<&'static crate::nn::Net>,
    pub policy_mode: u8,
    pub policy_temp: f32,
    pub weights: [f32; NUM_FEATURES],
    pub use_priors: bool,
    pub use_solver: bool,
    pub filter_walls: bool,
    pub leaf: Leaf,
    pub wall_prob: f32,
    pub root_noise: f32,
    pub use_gumbel: bool,
    pub use_guard: bool,
    pub use_endgame_solver: bool,
}

impl Default for Config {
    fn default() -> Config {
        let m = Mcts::new(1);
        Config {
            c_puct: m.c_puct,
            c_uct: m.c_uct,
            fpu_reduction: m.fpu_reduction,
            expand_threshold: m.expand_threshold,
            max_nodes: m.max_nodes,
            max_children: m.max_children,
            soft_value: m.soft_value,
            use_nn: m.use_nn,
            net: m.net,
            policy_mode: m.policy_mode,
            policy_temp: m.policy_temp,
            weights: m.weights,
            use_priors: m.use_priors,
            use_solver: m.use_solver,
            filter_walls: m.filter_walls,
            leaf: m.leaf,
            wall_prob: m.wall_prob,
            root_noise: m.root_noise,
            use_gumbel: m.use_gumbel,
            use_guard: m.use_guard,
            use_endgame_solver: m.use_endgame_solver,
        }
    }
}

impl Config {
    pub fn apply(&self, m: &mut Mcts) {
        m.c_puct = self.c_puct;
        m.c_uct = self.c_uct;
        m.fpu_reduction = self.fpu_reduction;
        m.expand_threshold = self.expand_threshold;
        m.max_nodes = self.max_nodes;
        m.max_children = self.max_children;
        m.soft_value = self.soft_value;
        m.use_nn = self.use_nn;
        m.net = self.net;
        m.policy_mode = self.policy_mode;
        m.policy_temp = self.policy_temp;
        m.weights = self.weights;
        m.use_priors = self.use_priors;
        m.use_solver = self.use_solver;
        m.filter_walls = self.filter_walls;
        m.leaf = self.leaf;
        m.wall_prob = self.wall_prob;
        m.root_noise = self.root_noise;
        m.use_gumbel = self.use_gumbel;
        m.use_guard = self.use_guard;
        m.use_endgame_solver = self.use_endgame_solver;
    }
}

/// Kök çocuklarının bir aramadaki özeti.
#[derive(Clone, Copy)]
pub struct RootMove {
    pub mv: Move,
    pub visits: u32,
    pub value: f32,
    pub prior: f32,
    /// `-1` bu hamle kesin kazandırıyor, `1` kesin kaybettiriyor.
    pub proof: i8,
}

impl Mcts {
    /// Kök çocuklarının ham istatistikleri (paralel aramada birleştirmek için).
    pub fn root_moves(&self) -> Vec<RootMove> {
        let root = self.nodes[0];
        (0..root.n_children as u32)
            .map(|i| {
                let n = self.nodes[(root.first_child + i) as usize];
                RootMove {
                    mv: n.mv,
                    visits: n.visits,
                    value: n.value,
                    prior: n.prior,
                    proof: n.proof,
                }
            })
            .collect()
    }
}

/// Kök paralelizasyonu: her thread kendi ağacını kurar, sonunda kök ziyaretleri
/// toplanır.
///
/// Neden paylaşımlı ağaç değil: değer yaprağıyla çalışan arama **deterministik**,
/// dolayısıyla aynı tohumla N ağaç birebir aynı çıkardı. Küçük bir kök gürültüsü
/// ağaçları farklılaştırıyor ve her thread bağımsız bir örneklem üretiyor.
/// Paylaşımlı ağaç + virtual loss daha iyi ölçeklenirdi ama atomik düğüm
/// erişimi ve genişletme kilidi gerektiriyor; buradaki yaklaşımda thread'ler
/// hiçbir şey paylaşmadığı için yarış koşulu riski sıfır.
///
/// Kanıt (MCTS-Solver) sonuçları birleştirmede istatistiği ezer: bir thread
/// kazancı kanıtladıysa o hamle oynanır.
pub fn search_parallel(
    pos: &Position,
    budget_ms: Option<u64>,
    iters: u32,
    threads: usize,
    configure: impl Fn(&mut Mcts, usize) + Send + Sync + Copy,
) -> (Option<Move>, SearchStats) {
    let threads = threads.max(1);
    if threads == 1 {
        let mut m = Mcts::new(0x9e37_79b9);
        configure(&mut m, 0);
        return match budget_ms {
            Some(ms) => m.search_time(pos, ms),
            None => m.search_rollouts(pos, iters),
        };
    }

    let start = Instant::now();
    let results: Vec<(Vec<RootMove>, u32, usize)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let p = *pos;
                s.spawn(move || {
                    let mut m = Mcts::new(0x9e37_79b9_7f4a_7c15u64.wrapping_mul(t as u64 + 1) | 1);
                    configure(&mut m, t);
                    m.root_noise = m.root_noise.max(0.08);
                    let (_, st) = match budget_ms {
                        Some(ms) => m.search_time(&p, ms),
                        None => m.search_rollouts(&p, iters.div_ceil(threads as u32)),
                    };
                    (m.root_moves(), st.rollouts, st.nodes)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Kök istatistiklerini hamle kimliğine göre topla.
    let mut agg: std::collections::HashMap<u16, (u32, f32, i8)> = std::collections::HashMap::new();
    let mut total_iters = 0u32;
    let mut total_nodes = 0usize;
    for (rms, it, nd) in &results {
        total_iters += it;
        total_nodes += nd;
        for rm in rms {
            let e = agg.entry(rm.mv.0).or_insert((0, 0.0, 0));
            e.0 += rm.visits;
            e.1 += rm.value;
            // Kanıt her zaman istatistiği ezer.
            if rm.proof == -1 {
                e.2 = -1;
            } else if rm.proof == 1 && e.2 == 0 {
                e.2 = 1;
            }
        }
    }

    let mut top: Vec<(Move, u32, f32)> = agg
        .iter()
        .map(|(&id, &(v, val, _))| (Move(id), v, if v > 0 { val / v as f32 } else { 0.0 }))
        .collect();
    top.sort_by(|a, b| b.1.cmp(&a.1).then(b.2.total_cmp(&a.2)));

    let proven = agg
        .iter()
        .find(|(_, &(_, _, pr))| pr == -1)
        .map(|(&id, _)| Move(id));

    // Tek thread'deki mantığın aynısı: kaybedilmiş pozisyonda prior'a düş.
    let hopeless = top.first().map(|t| t.2).unwrap_or(0.0) < LOST_THRESHOLD;
    let fallback = if hopeless {
        let mut best_prior: Option<(f32, Move)> = None;
        for (rms, _, _) in &results {
            for rm in rms {
                if rm.proof == 1 {
                    continue;
                }
                if best_prior.map_or(true, |(p, _)| rm.prior > p) {
                    best_prior = Some((rm.prior, rm.mv));
                }
            }
        }
        best_prior.map(|(_, m)| m)
    } else {
        None
    };

    let best = proven.or(fallback).or_else(|| top.first().map(|t| t.0));
    let win_rate = top.first().map(|t| t.2).unwrap_or(0.0);
    top.truncate(24); // 6 gösterim için yeterliydi; politika hedefi için daha fazlası lazım

    (
        best,
        SearchStats {
            rollouts: total_iters,
            nodes: total_nodes,
            win_rate,
            elapsed_s: start.elapsed().as_secs_f64(),
            top,
        },
    )
}
