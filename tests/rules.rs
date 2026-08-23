//! Kural motorunun doğruluk testleri.
//!
//! En kritik olan `shortcut_matches_full_check`: Slatton kısayolunu tam
//! flood-fill'e karşı binlerce rastgele pozisyonda doğrular. Kısayol
//! yanlışsa motor illegal hamle üretir ve her şey çöker.

use gedik::bitboard::*;
use gedik::board::{goal_row, Position};
use gedik::moves::*;
use gedik::notation::*;
use gedik::perft::perft;

/// Deterministik xorshift — test tekrarlanabilirliği için.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn random_position(rng: &mut Rng, plies: usize) -> Position {
    let mut p = Position::start();
    for _ in 0..plies {
        if p.winner().is_some() {
            break;
        }
        let moves = p.legal_moves();
        if moves.is_empty() {
            break;
        }
        let mv = moves[rng.below(moves.len())];
        p.make(mv);
    }
    p
}

// ---------------------------------------------------------------- başlangıç

#[test]
fn start_position_is_sane() {
    let p = Position::start();
    assert_eq!(p.pawn[0] as usize, cell(8, 4));
    assert_eq!(p.pawn[1] as usize, cell(0, 4));
    assert_eq!(p.walls, [10, 10]);
    assert_eq!(p.side, 0);
    assert_eq!(cell_name(p.pawn[0] as usize), "e1");
    assert_eq!(cell_name(p.pawn[1] as usize), "e9");
    assert_eq!(p.distance(0), Some(8));
    assert_eq!(p.distance(1), Some(8));
}

#[test]
fn start_move_count() {
    let p = Position::start();
    let moves = p.legal_moves();
    let pawn = moves.iter().filter(|m| !m.is_wall()).count();
    let walls = moves.iter().filter(|m| m.is_wall()).count();
    // e1'den: kuzey, doğu, batı (güney tahta kenarı).
    assert_eq!(pawn, 3, "başlangıçta 3 piyon hamlesi olmalı");
    // Boş tahtada tek bir duvar kimseyi kesemez -> 128'i de legal.
    assert_eq!(walls, 128, "boş tahtada 128 duvar hamlesi olmalı");
    assert_eq!(moves.len(), 131);
}

// ------------------------------------------------------------ duvar şekli

#[test]
fn wall_shape_conflicts() {
    let mut p = Position::start();
    p.make(Move::hwall(wslot(3, 3))); // d6h
    // Aynı slot dolu
    assert!(!p.wall_shape_ok(wslot(3, 3), true));
    // Dik kesişme aynı merkezde yasak
    assert!(!p.wall_shape_ok(wslot(3, 3), false));
    // Yatay komşu örtüşür
    assert!(!p.wall_shape_ok(wslot(3, 2), true));
    assert!(!p.wall_shape_ok(wslot(3, 4), true));
    // İki slot ötesi serbest
    assert!(p.wall_shape_ok(wslot(3, 1), true));
    assert!(p.wall_shape_ok(wslot(3, 5), true));
    // Farklı merkezde dikey duvar serbest (T bağlantısı)
    assert!(p.wall_shape_ok(wslot(3, 2), false));
    assert!(p.wall_shape_ok(wslot(2, 3), false));
    // Üst/alt satırdaki yatay duvarlar serbest
    assert!(p.wall_shape_ok(wslot(2, 3), true));
    assert!(p.wall_shape_ok(wslot(4, 3), true));
}

#[test]
fn vertical_wall_shape_conflicts() {
    let mut p = Position::start();
    p.make(Move::vwall(wslot(3, 3)));
    assert!(!p.wall_shape_ok(wslot(3, 3), false));
    assert!(!p.wall_shape_ok(wslot(2, 3), false));
    assert!(!p.wall_shape_ok(wslot(4, 3), false));
    assert!(p.wall_shape_ok(wslot(1, 3), false));
    assert!(p.wall_shape_ok(wslot(5, 3), false));
    assert!(!p.wall_shape_ok(wslot(3, 3), true));
    assert!(p.wall_shape_ok(wslot(3, 2), true));
}

// ------------------------------------------------------------ duvar etkisi

#[test]
fn horizontal_wall_blocks_two_columns() {
    let mut p = Position::start();
    p.make(Move::hwall(wslot(3, 3))); // merkez (3,3): (3,3)-(4,3) ve (3,4)-(4,4) kesilir
    let m = p.masks();
    assert!(!m.step_open(cell(3, 3), SOUTH));
    assert!(!m.step_open(cell(3, 4), SOUTH));
    assert!(!m.step_open(cell(4, 3), NORTH));
    assert!(!m.step_open(cell(4, 4), NORTH));
    assert!(m.step_open(cell(3, 2), SOUTH));
    assert!(m.step_open(cell(3, 5), SOUTH));
    // Yatay hareketi etkilemez
    assert!(m.step_open(cell(3, 3), EAST));
    assert!(m.step_open(cell(4, 3), EAST));
}

#[test]
fn vertical_wall_blocks_two_rows() {
    let mut p = Position::start();
    p.make(Move::vwall(wslot(3, 3)));
    let m = p.masks();
    assert!(!m.step_open(cell(3, 3), EAST));
    assert!(!m.step_open(cell(4, 3), EAST));
    assert!(!m.step_open(cell(3, 4), WEST));
    assert!(!m.step_open(cell(4, 4), WEST));
    assert!(m.step_open(cell(2, 3), EAST));
    assert!(m.step_open(cell(5, 3), EAST));
    assert!(m.step_open(cell(3, 3), SOUTH));
}

// ----------------------------------------------------------------- atlama

fn pos_with(pawn0: usize, pawn1: usize, side: u8, h: u64, v: u64) -> Position {
    let mut p = Position {
        h,
        v,
        pawn: [pawn0 as u8, pawn1 as u8],
        walls: [10, 10],
        side,
        hash: 0,
        ply: 0,
    };
    p.hash = p.compute_hash();
    p
}

fn pawn_targets(p: &Position) -> Vec<String> {
    let mut v = Vec::new();
    p.gen_pawn_moves(&p.masks(), &mut v);
    let mut names: Vec<String> = v
        .iter()
        .map(|&m| match m.kind() {
            MoveKind::Pawn(d) => cell_name(p.pawn_target(d)),
            _ => unreachable!(),
        })
        .collect();
    names.sort();
    names
}

#[test]
fn straight_jump_is_mandatory_when_open() {
    // A e5 (4,4), B e6 (3,4). A'nın kuzeyi B, B'nin kuzeyi açık -> düz atlama.
    let p = pos_with(cell(4, 4), cell(3, 4), 0, 0, 0);
    let t = pawn_targets(&p);
    // e7 = (2,4) düz atlama; diyagonaller (d6/f6) olmamalı.
    assert!(t.contains(&"e7".to_string()), "düz atlama üretilmeli: {t:?}");
    assert!(!t.contains(&"d6".to_string()), "atlama açıkken diyagonal olmamalı: {t:?}");
    assert!(!t.contains(&"f6".to_string()), "atlama açıkken diyagonal olmamalı: {t:?}");
    assert!(!t.contains(&"e6".to_string()), "rakibin karesine girilemez: {t:?}");
    // Yanlar ve geri normal
    assert!(t.contains(&"d5".to_string()));
    assert!(t.contains(&"f5".to_string()));
    assert!(t.contains(&"e4".to_string()));
    assert_eq!(t.len(), 4);
}

#[test]
fn diagonal_allowed_when_wall_behind_opponent() {
    // A e5 (4,4), B e6 (3,4), B'nin arkasına yatay duvar: (2,4)-(3,4) kesilmeli.
    // Yatay merkez (2,4) sütun 4 ve 5'i keser -> (2,4)-(3,4) kapanır.
    let p = pos_with(cell(4, 4), cell(3, 4), 0, 1u64 << wslot(2, 4), 0);
    let m = p.masks();
    assert!(!m.step_open(cell(3, 4), NORTH), "duvar B'nin arkasını kapatmalı");
    let t = pawn_targets(&p);
    assert!(!t.contains(&"e7".to_string()), "düz atlama kapalı olmalı: {t:?}");
    assert!(t.contains(&"d6".to_string()), "sol diyagonal olmalı: {t:?}");
    assert!(t.contains(&"f6".to_string()), "sağ diyagonal olmalı: {t:?}");
}

#[test]
fn diagonal_allowed_at_board_edge() {
    // B en üst sırada (0,4), A hemen altında (1,4). Düz atlama tahta dışı.
    let p = pos_with(cell(1, 4), cell(0, 4), 0, 0, 0);
    let t = pawn_targets(&p);
    assert!(t.contains(&"d9".to_string()), "{t:?}");
    assert!(t.contains(&"f9".to_string()), "{t:?}");
    assert_eq!(t.iter().filter(|s| s.ends_with('9')).count(), 2);
}

#[test]
fn diagonal_blocked_by_side_wall() {
    // A e5, B e6, B'nin arkası kapalı, ayrıca B'nin batısı da kapalı.
    // Dikey merkez (3,3): (3,3)-(3,4) ve (4,3)-(4,4) kesilir -> B'nin batısı kapalı.
    let h = 1u64 << wslot(2, 4);
    let v = 1u64 << wslot(3, 3);
    let p = pos_with(cell(4, 4), cell(3, 4), 0, h, v);
    let t = pawn_targets(&p);
    assert!(!t.contains(&"d6".to_string()), "batı diyagonali kapalı olmalı: {t:?}");
    assert!(t.contains(&"f6".to_string()), "doğu diyagonali açık olmalı: {t:?}");
    // A'nın kendi batı hamlesi de aynı duvarla kapanır: (4,4)-(4,3)
    assert!(!t.contains(&"d5".to_string()), "{t:?}");
}

// -------------------------------------------------- yol kapatma yasağı

#[test]
fn wall_that_seals_a_player_is_illegal() {
    // B (0,0)'da. v(0,0) doğuyu 0. ve 1. satırda keser.
    // h(1,0) eklenirse {(0,0),(1,0)} bölgesi tamamen kapanır -> illegal olmalı.
    let p = pos_with(cell(8, 4), cell(0, 0), 0, 0, 1u64 << wslot(0, 0));
    assert!(p.wall_shape_ok(wslot(1, 0), true), "şekilsel olarak yerleşebilir");
    assert!(!p.wall_legal(wslot(1, 0), true), "B'yi hapsettiği için illegal olmalı");
    assert!(!p.wall_legal_slow(wslot(1, 0), true));
    // Kısayol bu duvar için tam kontrol istemek zorunda
    assert!(!p.paths_open(1u64 << wslot(1, 0), 1u64 << wslot(0, 0)));
    // Aynı duvar B başka yerdeyse legal
    let q = pos_with(cell(8, 4), cell(4, 4), 0, 0, 1u64 << wslot(0, 0));
    assert!(q.wall_legal(wslot(1, 0), true));
}

#[test]
fn shortcut_matches_full_check() {
    let mut rng = Rng(0xdead_beef_cafe_1234);
    let mut checked = 0u64;
    let mut shortcut_hits = 0u64;
    for trial in 0..400 {
        let p = random_position(&mut rng, 4 + trial % 40);
        for slot in 0..WSLOTS {
            for horizontal in [true, false] {
                let fast = p.wall_legal(slot, horizontal);
                let slow = p.wall_legal_slow(slot, horizontal);
                assert_eq!(
                    fast, slow,
                    "kısayol uyuşmazlığı: slot {slot} horizontal {horizontal}\nh={:#x} v={:#x} pawns={:?}",
                    p.h, p.v, p.pawn
                );
                if p.wall_shape_ok(slot, horizontal) {
                    checked += 1;
                    if fast {
                        shortcut_hits += 1;
                    }
                }
            }
        }
    }
    assert!(checked > 10_000, "yeterince örnek üretilmedi");
    assert!(shortcut_hits > 0);
}

#[test]
fn generated_moves_are_all_legal() {
    let mut rng = Rng(0x1234_5678_9abc_def0);
    for trial in 0..300 {
        let mut p = random_position(&mut rng, 3 + trial % 60);
        if p.winner().is_some() {
            continue;
        }
        for mv in p.legal_moves() {
            assert!(p.is_legal(mv), "üretilen hamle legal değil: {mv:?}");
            // Hamle sonrası her iki oyuncunun da yolu olmalı
            let u = p.make(mv);
            assert!(
                p.masks().reachable(p.pawn[0] as usize) & ROW_MASK[goal_row(0)] != 0,
                "A'nın yolu kapandı: {mv:?}"
            );
            assert!(
                p.masks().reachable(p.pawn[1] as usize) & ROW_MASK[goal_row(1)] != 0,
                "B'nin yolu kapandı: {mv:?}"
            );
            p.unmake(u);
        }
    }
}

#[test]
fn pawns_never_share_a_cell() {
    let mut rng = Rng(0x0f0f_0f0f_1111_2222);
    for _ in 0..2000 {
        let p = random_position(&mut rng, 40);
        assert_ne!(p.pawn[0], p.pawn[1]);
    }
}

// ------------------------------------------------------------ make/unmake

#[test]
fn make_unmake_is_exact_inverse() {
    let mut rng = Rng(0xabcd_ef01_2345_6789);
    for trial in 0..200 {
        let mut p = random_position(&mut rng, 2 + trial % 50);
        if p.winner().is_some() {
            continue;
        }
        let before = p;
        for mv in p.legal_moves() {
            let u = p.make(mv);
            assert_eq!(p.hash, p.compute_hash(), "artımlı hash bozuldu: {mv:?}");
            p.unmake(u);
            assert!(
                p.h == before.h
                    && p.v == before.v
                    && p.pawn == before.pawn
                    && p.walls == before.walls
                    && p.side == before.side
                    && p.ply == before.ply
                    && p.hash == before.hash,
                "unmake geri getirmedi: {mv:?}"
            );
        }
    }
}

#[test]
fn wall_count_decrements_and_runs_out() {
    let mut p = Position::start();
    let mut slot = 0;
    for _ in 0..20 {
        // Birbirine değmeyen duvarlar koy
        while !p.wall_legal(slot, true) {
            slot += 1;
        }
        p.make(Move::hwall(slot));
        slot += 2;
    }
    assert_eq!(p.walls, [0, 0]);
    let moves = p.legal_moves();
    assert!(
        moves.iter().all(|m| !m.is_wall()),
        "duvar bitince duvar hamlesi üretilmemeli"
    );
}

// ------------------------------------------------------------------ ayna

#[test]
fn mirror_is_an_involution_and_preserves_move_count() {
    let mut rng = Rng(0x5555_aaaa_5555_aaaa);
    for _ in 0..300 {
        let p = random_position(&mut rng, 30);
        let m = p.mirrored();
        let back = m.mirrored();
        assert_eq!(p.h, back.h);
        assert_eq!(p.v, back.v);
        assert_eq!(p.pawn, back.pawn);
        assert_eq!(p.hash, back.hash);
        assert_eq!(
            p.legal_moves().len(),
            m.legal_moves().len(),
            "ayna legal hamle sayısını korumalı"
        );
        assert_eq!(p.distance(0), m.distance(0));
        assert_eq!(p.distance(1), m.distance(1));
    }
}

// -------------------------------------------------------------- notasyon

#[test]
fn notation_round_trips() {
    for i in 0..CELLS {
        assert_eq!(parse_cell(&cell_name(i)), Some(i));
    }
    for slot in 0..WSLOTS {
        for horizontal in [true, false] {
            let name = wall_name(slot, horizontal);
            assert_eq!(parse_wall(&name), Some((slot, horizontal)), "{name}");
        }
    }
    // Glendenning tezindeki örnek
    assert_eq!(wall_name(wslot(3, 4), true), "e6h");
    assert_eq!(parse_wall("e6h"), Some((wslot(3, 4), true)));
}

#[test]
fn parse_move_accepts_squares_and_walls() {
    let p = Position::start();
    assert_eq!(parse_move(&p, "e2"), Some(Move::pawn(NORTH)));
    assert_eq!(parse_move(&p, "d1"), Some(Move::pawn(WEST)));
    assert_eq!(parse_move(&p, "e6h"), Some(Move::hwall(wslot(3, 4))));
    assert_eq!(parse_move(&p, "c3v"), Some(Move::vwall(wslot(6, 2))));
    assert_eq!(parse_move(&p, "e3"), None, "iki kare ilerlemek illegal");
    assert_eq!(parse_move(&p, "zz9"), None);
}

// ----------------------------------------------------------------- perft

/// Referans perft değerleri. Quoridor'da yayınlanmış bir perft tablosu yok;
/// bunlar bu motorun ürettiği ilk set ve regresyon kilidi olarak duruyor.
#[test]
fn perft_regression() {
    let mut p = Position::start();
    assert_eq!(perft(&mut p, 0), 1);
    assert_eq!(perft(&mut p, 1), 131);
    assert_eq!(perft(&mut p, 2), 16_677);
    assert_eq!(perft(&mut p, 3), 2_062_264);
}

/// Ayna simetrisi perft'i korumalı. Hamle üretimindeki yön asimetrilerini
/// (özellikle diyagonal atlama eşlemelerini) yakalayan güçlü bir test.
#[test]
fn perft_is_mirror_invariant() {
    let mut rng = Rng(0xc0de_1234_5678_9abc);
    for trial in 0..40 {
        let p = random_position(&mut rng, 6 + trial % 30);
        if p.winner().is_some() {
            continue;
        }
        let mut a = p;
        let mut b = p.mirrored();
        for depth in 1..=2 {
            assert_eq!(
                perft(&mut a, depth),
                perft(&mut b, depth),
                "ayna perft uyuşmazlığı derinlik {depth}: h={:#x} v={:#x} pawns={:?}",
                p.h,
                p.v,
                p.pawn
            );
        }
    }
}

#[test]
fn pawn_targets_stay_on_board_and_are_reachable() {
    let mut rng = Rng(0x2468_ace0_1357_bdf0);
    for _ in 0..500 {
        let p = random_position(&mut rng, 35);
        if p.winner().is_some() {
            continue;
        }
        let mut mv = Vec::new();
        p.gen_pawn_moves(&p.masks(), &mut mv);
        let from = p.pawn[p.side as usize] as usize;
        for m in mv {
            let MoveKind::Pawn(d) = m.kind() else {
                unreachable!()
            };
            let t = p.pawn_target(d);
            assert!(t < CELLS, "hedef tahta dışı");
            // Satır/sütun kayması en fazla 2 ve wraparound yok
            let (dr, dc) = (
                row_of(t) as i32 - row_of(from) as i32,
                col_of(t) as i32 - col_of(from) as i32,
            );
            assert!(dr.abs() <= 2 && dc.abs() <= 2, "anlamsız kayma {dr},{dc}");
            assert!(dr.abs() + dc.abs() >= 1);
            // Rakibin karesine girilmiyor
            assert_ne!(t, p.pawn[p.opponent()] as usize);
        }
    }
}

// --------------------------------------------------------------- mesafe

#[test]
fn distance_field_agrees_with_single_queries() {
    let mut rng = Rng(0x7777_8888_9999_aaaa);
    for _ in 0..200 {
        let p = random_position(&mut rng, 25);
        let m = p.masks();
        for player in 0..2 {
            let g = goal_row(player);
            let field = m.distance_field(g);
            for i in 0..CELLS {
                let q = m.distance_to_row(i, g);
                match q {
                    Some(d) => assert_eq!(field[i] as u32, d, "hücre {i} oyuncu {player}"),
                    None => assert_eq!(field[i], u8::MAX, "hücre {i} ulaşılamaz olmalı"),
                }
            }
        }
    }
}

#[test]
fn walls_can_lengthen_but_never_sever_the_path() {
    let mut rng = Rng(0xfeed_face_1234_5678);
    for _ in 0..500 {
        let p = random_position(&mut rng, 50);
        assert!(p.distance(0).is_some(), "A'nın yolu her zaman olmalı");
        assert!(p.distance(1).is_some(), "B'nin yolu her zaman olmalı");
    }
}

// ------------------------------------------------------------------- fen

#[test]
fn fen_round_trips() {
    let mut rng = Rng(0x1111_2222_3333_4444);
    for _ in 0..500 {
        let p = random_position(&mut rng, 40);
        let f = to_fen(&p);
        let q = from_fen(&f).expect(&format!("kendi ürettiği fen okunamadı: {f}"));
        assert_eq!(p.h, q.h);
        assert_eq!(p.v, q.v);
        assert_eq!(p.pawn, q.pawn);
        assert_eq!(p.walls, q.walls);
        assert_eq!(p.side, q.side);
        assert_eq!(p.ply, q.ply);
        assert_eq!(p.hash, q.hash, "hash yeniden kurulmalı");
        assert_eq!(to_fen(&q), f);
    }
}

#[test]
fn fen_start_is_canonical() {
    assert_eq!(to_fen(&Position::start()), "0/0/76/4/10/10/0/0");
    let p = from_fen("0/0/76/4/10/10/0/0").unwrap();
    assert_eq!(p.legal_moves().len(), 131);
}

#[test]
fn fen_rejects_garbage() {
    for bad in [
        "",
        "bozuk",
        "0/0/76/4/10/10/0",           // eksik alan
        "0/0/76/76/10/10/0/0",        // iki piyon aynı karede
        "0/0/81/4/10/10/0/0",         // tahta dışı piyon
        "0/0/76/4/11/10/0/0",         // duvar sayısı fazla
        "0/0/76/4/10/10/2/0",         // geçersiz sıra
        "zz/0/76/4/10/10/0/0",        // hex değil
        // B'yi (0,0) köşesinde hapseden duvarlar: v(0,0) slot 0 + h(1,0) slot 8
        "100/1/76/0/8/8/0/0",
    ] {
        assert!(from_fen(bad).is_none(), "kabul edilmemeliydi: {bad:?}");
    }
}

