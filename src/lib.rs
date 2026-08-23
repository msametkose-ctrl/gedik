//! # quoridor
//!
//! Hızlı bitboard tabanlı Quoridor kural çekirdeği.
//!
//! Tasarım kararları:
//!
//! * **Duvarlar 2 x u64.** 8x8 duvar merkezi ızgarası yatay ve dikey için ayrı
//!   birer 64-bit tamsayıya tam oturuyor.
//! * **Hücreler u128.** 81 hücre; geçiş maskeleri (`Masks`) duvarlardan 8
//!   iterasyonda türetiliyor, böylece hamle üretimi ve ulaşılabilirlik
//!   kontrolü saf bitwise işlem oluyor.
//! * **Sabit aksiyon ID'leri (0..140).** Pozisyondan bağımsız; NN policy head'i
//!   doğrudan bu uzaya bağlanabilir.
//! * **Slatton kısayolu.** Bir duvar iki noktadan dayanmıyorsa kimseyi
//!   kesemez, pahalı flood-fill atlanır.

pub mod bitboard;
pub mod board;
pub mod endgame;
pub mod engine;
pub mod guard;
pub mod heuristics;
pub mod mcts;
pub mod moves;
pub mod nn;
pub mod notation;
pub mod perft;
pub mod rating;
pub mod search;
pub mod server;
pub mod zobrist;

pub use board::Position;
pub use moves::{Move, MoveKind, NUM_ACTIONS};
