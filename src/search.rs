//! Faz 2 için basit alpha-beta baseline botu.
//!
//! Bu **nihai motor değil** — Quoridor'da statik değerlendirme zayıf kalıyor
//! (duvar her yere konabildiği için "sakin pozisyon" yok, quiescence anlamsız).
//! Amacı çekirdeği doğrulamak ve oynanabilir bir rakip vermek. Asıl güç
//! heuristic MCTS + PUCT/NN fazında gelecek.

use crate::board::{goal_row, Position};
use crate::moves::*;
use std::time::{Duration, Instant};

pub const WIN: i32 = 1_000_000;

/// Sıradaki oyuncu açısından pozisyon değeri.
pub fn evaluate(pos: &Position) -> i32 {
    let me = pos.side as usize;
    let opp = 1 - me;
    if let Some(w) = pos.winner() {
        return if w == me { WIN } else { -WIN };
    }
    let m = pos.masks();
    let d_me = m
        .distance_to_row(pos.pawn[me] as usize, goal_row(me))
        .unwrap_or(200) as i32;
    let d_opp = m
        .distance_to_row(pos.pawn[opp] as usize, goal_row(opp))
        .unwrap_or(200) as i32;

    // Hedefe kalan mesafe farkı baskın terim; kalan duvar da bir kaynak.
    let mut score = 100 * (d_opp - d_me);
    score += 22 * (pos.walls[me] as i32 - pos.walls[opp] as i32);
    // Sıra bizde olması küçük bir tempo avantajı.
    score += 12;
    score
}

use crate::heuristics::candidate_moves as gen_pruned;

#[derive(Clone, Copy, PartialEq)]
enum Flag {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy)]
struct TtEntry {
    key: u64,
    depth: u8,
    score: i32,
    flag: Flag,
    best: Move,
}

pub struct Searcher {
    tt: Vec<Option<TtEntry>>,
    mask: usize,
    pub nodes: u64,
    deadline: Option<Instant>,
    stopped: bool,
}

impl Searcher {
    pub fn new(tt_bits: usize) -> Searcher {
        let size = 1usize << tt_bits;
        Searcher {
            tt: vec![None; size],
            mask: size - 1,
            nodes: 0,
            deadline: None,
            stopped: false,
        }
    }

    fn out_of_time(&mut self) -> bool {
        if self.stopped {
            return true;
        }
        if self.nodes % 512 == 0 {
            if let Some(d) = self.deadline {
                if Instant::now() >= d {
                    self.stopped = true;
                }
            }
        }
        self.stopped
    }

    /// Iterative deepening. `time_ms` dolduğunda son tamamlanan derinliğin
    /// sonucunu döndürür.
    pub fn best_move(&mut self, pos: &Position, max_depth: u32, time_ms: u64) -> (Option<Move>, i32) {
        self.nodes = 0;
        self.stopped = false;
        self.deadline = Some(Instant::now() + Duration::from_millis(time_ms));

        let mut work = *pos;
        let mut best = None;
        let mut best_score = -WIN;

        for depth in 1..=max_depth {
            let (mv, sc) = self.root(&mut work, depth);
            if self.stopped && best.is_some() {
                break;
            }
            if mv.is_some() {
                best = mv;
                best_score = sc;
            }
            if sc.abs() >= WIN - 1000 {
                break; // kesin sonuç bulundu
            }
        }
        (best, best_score)
    }

    fn root(&mut self, pos: &mut Position, depth: u32) -> (Option<Move>, i32) {
        let mut moves = gen_pruned(pos);
        if moves.is_empty() {
            return (None, evaluate(pos));
        }
        self.order(pos, &mut moves);

        let mut alpha = -WIN - 1;
        let mut best = None;
        for mv in moves {
            let u = pos.make(mv);
            let sc = -self.alphabeta(pos, depth - 1, -WIN - 1, -alpha);
            pos.unmake(u);
            if self.stopped && best.is_some() {
                break;
            }
            if sc > alpha {
                alpha = sc;
                best = Some(mv);
            }
        }
        (best, alpha)
    }

    fn alphabeta(&mut self, pos: &mut Position, depth: u32, mut alpha: i32, mut beta: i32) -> i32 {
        self.nodes += 1;
        if self.out_of_time() {
            return evaluate(pos);
        }
        if let Some(w) = pos.winner() {
            let me = pos.side as usize;
            // Kazanan hamleyi *yapan* taraf sıra değişmiş olduğu için rakip.
            return if w == me {
                WIN - pos.ply as i32
            } else {
                -WIN + pos.ply as i32
            };
        }
        if depth == 0 {
            return evaluate(pos);
        }

        let idx = (pos.hash as usize) & self.mask;
        let mut tt_move = None;
        if let Some(e) = self.tt[idx] {
            if e.key == pos.hash {
                tt_move = Some(e.best);
                if e.depth as u32 >= depth {
                    match e.flag {
                        Flag::Exact => return e.score,
                        Flag::Lower => alpha = alpha.max(e.score),
                        Flag::Upper => beta = beta.min(e.score),
                    }
                    if alpha >= beta {
                        return e.score;
                    }
                }
            }
        }

        let mut moves = gen_pruned(pos);
        if moves.is_empty() {
            return evaluate(pos);
        }
        self.order(pos, &mut moves);
        if let Some(m) = tt_move {
            if let Some(p) = moves.iter().position(|&x| x == m) {
                moves.swap(0, p);
            }
        }

        let orig_alpha = alpha;
        let mut best = moves[0];
        let mut best_score = -WIN - 1;

        for mv in moves {
            let u = pos.make(mv);
            let sc = -self.alphabeta(pos, depth - 1, -beta, -alpha);
            pos.unmake(u);
            if sc > best_score {
                best_score = sc;
                best = mv;
            }
            if sc > alpha {
                alpha = sc;
            }
            if alpha >= beta {
                break;
            }
        }

        let flag = if best_score <= orig_alpha {
            Flag::Upper
        } else if best_score >= beta {
            Flag::Lower
        } else {
            Flag::Exact
        };
        self.tt[idx] = Some(TtEntry {
            key: pos.hash,
            depth: depth as u8,
            score: best_score,
            flag,
            best,
        });

        best_score
    }

    /// Hamle sıralaması: hedefe yaklaştıran piyon hamleleri önce, sonra duvarlar.
    fn order(&self, pos: &Position, moves: &mut [Move]) {
        let me = pos.side as usize;
        let m = pos.masks();
        let field = m.distance_field(goal_row(me));
        let cur = field[pos.pawn[me] as usize];
        moves.sort_by_key(|&mv| match mv.kind() {
            MoveKind::Pawn(d) => {
                let t = pos.pawn_target(d);
                let nd = field[t];
                (nd as i32) - (cur as i32) * 2
            }
            _ => 100,
        });
    }
}
