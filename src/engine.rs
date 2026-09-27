//! Motor tanımı — CLI ve web sunucusunun paylaştığı katman.
//!
//! Bir motor bir metin tanımıyla belirtiliyor:
//!
//! | tanım | anlam |
//! |---|---|
//! | `mcts:1000ms` | süre bütçeli MCTS (değerlendirme yaprağı) |
//! | `mcts:20000` | sabit iterasyon bütçeli MCTS |
//! | `mcts:20000:1.4` | üçüncü alan `c_puct` |
//! | `mcts:20000:cp=1.4,et=8,fpu=0.25` | ayar listesi |
//! | `mctsr:20000` | klasik rastgele rollout yaprağı |
//! | `mctsf:20000` | duvar aday filtresi kapalı |
//! | `mctsu:20000` | policy prior kapalı, klasik UCT |
//! | `mcts:3000ms:pr=0,sv=0,w=0` | bu oturumdan önceki sürüm (prior yok, solver yok, elle ağırlık) |
//! | `mctsh:20000` | elle yazılmış değer ağırlıkları (öğrenilmiş yerine) |
//! | `ab:500ms` | baseline alpha-beta |

use crate::board::Position;
use crate::mcts::{search_parallel, Config, Leaf, Mcts};
use crate::moves::Move;
use crate::search::Searcher;

pub enum Engine {
    /// MCTS: ayarlar, süre bütçesi (ms) ya da iterasyon bütçesi, thread sayısı.
    Mcts {
        cfg: Config,
        ms: Option<u64>,
        iters: u32,
        threads: usize,
        seed: u64,
        label: String,
    },
    AlphaBeta(Searcher, u64),
}

/// Makinedeki çekirdek sayısı.
pub fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Bir aramanın sonucu, arayüze göstermek için.
pub struct Choice {
    pub mv: Option<Move>,
    /// Kısa insan-okunur bilgi ("wr 0.620" ya da "ev 142").
    pub info: String,
    /// Yapılan iş (iterasyon ya da düğüm sayısı).
    pub work: u64,
    /// En çok ziyaret edilen hamleler: (hamle, ziyaret, kazanma oranı).
    /// Alpha-beta bunu doldurmuyor.
    pub top: Vec<(Move, u32, f32)>,
}

impl Engine {
    pub fn parse(spec: &str, seed: u64) -> Engine {
        let mut parts = spec.trim().splitn(3, ':');
        let kind = parts.next().unwrap_or("mcts");
        let arg = parts.next().unwrap_or("1000ms");
        let opts = parts.next();

        match kind {
            "ab" | "alphabeta" => {
                let _ = opts;
                let ms = arg.trim_end_matches("ms").parse().unwrap_or(1000);
                Engine::AlphaBeta(Searcher::new(21), ms)
            }
            _ => {
                let mut cfg = Config::default();
                cfg.leaf = if kind == "mctsr" {
                    Leaf::Rollout
                } else {
                    Leaf::Value
                };
                cfg.filter_walls = kind != "mctsf";
                cfg.use_priors = kind != "mctsu";
                cfg.use_gumbel = kind == "gumbel" || kind == "mctsg";
                if kind == "mctsh" {
                    cfg.weights = crate::heuristics::WEIGHTS_HAND;
                }
                let mut threads = 1usize;

                // Üçüncü alan: ya tek sayı (c_puct) ya da `anahtar=değer` listesi.
                // Örn. `mcts:20000:cp=1.4,et=8,t=4`
                if let Some(spec) = opts {
                    if let Ok(v) = spec.parse::<f32>() {
                        cfg.c_puct = v;
                        cfg.c_uct = v;
                    } else {
                        for kv in spec.split(',') {
                            let Some((k, v)) = kv.split_once('=') else {
                                continue;
                            };
                            // `ag=` degeri bir dosya yolu, sayi degil; sayiya
                            // cevirme denemesinden ONCE ele alinmali yoksa
                            // sessizce atlanir. Iki agi ayni maca sokan anahtar
                            // bu: `mcts:200000:ag=tur2\ag64.bin`.
                            if matches!(k.trim(), "ag" | "net") {
                                cfg.net = crate::nn::net_from(v.trim());
                                cfg.use_nn = cfg.net.is_some();
                                continue;
                            }
                            let Ok(x) = v.parse::<f32>() else { continue };
                            match k.trim() {
                                "cp" | "cpuct" => cfg.c_puct = x,
                                "cu" | "cuct" => cfg.c_uct = x,
                                "gb" | "gumbel" => cfg.use_gumbel = x != 0.0,
                                "fpu" => cfg.fpu_reduction = x,
                                "et" | "expand" => cfg.expand_threshold = x as u32,
                                "mn" | "maxnodes" => cfg.max_nodes = x as usize,
                                "mc" | "maxchildren" => {
                                    cfg.max_children =
                                        if x <= 0.0 { usize::MAX } else { x as usize }
                                }
                                "wp" | "wallprob" => cfg.wall_prob = x,
                                "noise" => cfg.root_noise = x,
                                // Ablasyon anahtarları: eski sürümü yeniden kurmak için.
                                "pr" | "priors" => cfg.use_priors = x != 0.0,
                                "sv" | "solver" => cfg.use_solver = x != 0.0,
                                "sq" | "soft" => cfg.soft_value = x != 0.0,
                                "nn" => cfg.use_nn = x != 0.0,
                                "pol" | "policy" => cfg.policy_mode = x as u8,
                                "pt" | "poltemp" => cfg.policy_temp = x,
                                "guard" | "gd" => cfg.use_guard = x != 0.0,
                                "endgame" | "eg" => cfg.use_endgame_solver = x != 0.0,
                                // w=0 elle tasarlanmış (varsayılan), w=1 öğrenilmiş
                                "w" | "weights" => {
                                    cfg.weights = if x == 0.0 {
                                        crate::heuristics::WEIGHTS_HAND
                                    } else {
                                        crate::heuristics::WEIGHTS_LEARNED
                                    }
                                }
                                "t" | "threads" => {
                                    threads = if x <= 0.0 {
                                        default_threads()
                                    } else {
                                        x as usize
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }

                let (ms, iters) = match arg.strip_suffix("ms") {
                    Some(v) => (Some(v.parse().unwrap_or(1000)), 0),
                    None => (None, arg.parse().unwrap_or(20_000)),
                };
                Engine::Mcts {
                    cfg,
                    ms,
                    iters,
                    threads,
                    seed,
                    label: spec.trim().to_string(),
                }
            }
        }
    }

    pub fn name(&self) -> String {
        match self {
            Engine::Mcts { label, threads, .. } => {
                if *threads > 1 {
                    format!("{label} ({threads} thread)")
                } else {
                    label.clone()
                }
            }
            Engine::AlphaBeta(_, ms) => format!("ab:{ms}ms"),
        }
    }

    pub fn choose(&mut self, pos: &Position) -> Choice {
        self.choose_with_history(pos, &[])
    }

    pub fn choose_with_history(&mut self, pos: &Position, history: &[u64]) -> Choice {
        match self {
            Engine::Mcts {
                cfg,
                ms,
                iters,
                threads,
                seed,
                ..
            } => {
                let c = *cfg;
                let sd = *seed;
                let (mv, st) = if *threads > 1 {
                    search_parallel(pos, *ms, *iters, *threads, move |m, t| {
                        c.apply(m);
                        m.history = history.to_vec();
                        m.reseed(sd.wrapping_mul(0x9e37_79b9).wrapping_add(t as u64 * 7 + 1));
                    })
                } else {
                    let mut m = Mcts::new(sd);
                    c.apply(&mut m);
                    m.history = history.to_vec();
                    match ms {
                        Some(v) => m.search_time(pos, *v),
                        None => m.search_rollouts(pos, *iters),
                    }
                };
                Choice {
                    mv,
                    info: format!("wr {:.3}", st.win_rate),
                    work: st.rollouts as u64,
                    top: st.top,
                }
            }
            Engine::AlphaBeta(s, ms) => {
                let (mv, sc) = s.best_move(pos, 12, *ms);
                Choice {
                    mv,
                    info: format!("ev {sc}"),
                    work: s.nodes,
                    top: Vec::new(),
                }
            }
        }
    }
}
