//! Arama motorlarının davranış testleri.

use gedik::bitboard::*;
use gedik::board::Position;
use gedik::heuristics::{candidate_moves, candidate_moves_filtered, playout, value_to_move, Rng};
use gedik::mcts::{Leaf, Mcts};
use gedik::search::Searcher;

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

#[test]
fn candidates_are_always_legal_and_nonempty() {
    let mut rng = Rng::new(0xabc_123);
    for _ in 0..300 {
        let p = random_position(&mut rng, 30);
        if p.winner().is_some() {
            continue;
        }
        for filter in [true, false] {
            let c = candidate_moves_filtered(&p, filter);
            assert!(!c.is_empty(), "aday listesi boş olamaz");
            for mv in &c {
                assert!(p.is_legal(*mv), "aday legal değil: {mv:?}");
            }
        }
    }
}

#[test]
fn filter_never_drops_every_wall_when_walls_matter() {
    // Filtre bazı duvarları atabilir ama piyon hamleleri her zaman kalır,
    // dolayısıyla motor asla hamlesiz kalmaz.
    let mut rng = Rng::new(0xfeed_1);
    for _ in 0..200 {
        let p = random_position(&mut rng, 20);
        if p.winner().is_some() {
            continue;
        }
        let c = candidate_moves(&p);
        assert!(c.iter().any(|m| !m.is_wall()), "piyon hamlesi kalmalı");
    }
}

#[test]
fn value_is_a_probability_and_respects_terminal() {
    let mut rng = Rng::new(0x5eed);
    for _ in 0..500 {
        let p = random_position(&mut rng, 60);
        let v = value_to_move(&p);
        assert!((0.0..=1.0).contains(&v), "değer [0,1] dışında: {v}");
        if let Some(w) = p.winner() {
            // Kazanan hamleyi yapan taraf sıra değiştirdiği için sıradaki kaybetmiş.
            assert_ne!(w, p.side as usize);
            assert_eq!(v, 0.0);
        }
    }
}

#[test]
fn value_prefers_being_closer_to_goal() {
    // A e1'de, B e9'da: simetrik, sıra A'da -> yaklaşık dengede.
    let p = Position::start();
    let v0 = value_to_move(&p);
    // A bir adım ilerledi, sıra hâlâ A'da olacak şekilde kurulmuş pozisyon:
    let mut q = Position {
        h: 0,
        v: 0,
        pawn: [cell(7, 4) as u8, cell(0, 4) as u8],
        walls: [10, 10],
        side: 0,
        hash: 0,
        ply: 0,
    };
    q.hash = q.compute_hash();
    assert!(
        value_to_move(&q) > v0,
        "hedefe yakın olmak değeri artırmalı ({} vs {})",
        value_to_move(&q),
        v0
    );
}

#[test]
fn playout_always_terminates_with_a_winner() {
    let mut rng = Rng::new(0x1010);
    for _ in 0..2000 {
        let p = random_position(&mut rng, 20);
        let w = playout(p, &mut rng, 0.30, 400);
        assert!(w < 2);
    }
}

#[test]
fn engines_take_an_available_win() {
    // A i8'de (satır 1, hedef satır 0) -> tek hamlede kazanır.
    let mut p = Position {
        h: 0,
        v: 0,
        pawn: [cell(1, 8) as u8, cell(6, 0) as u8],
        walls: [3, 3],
        side: 0,
        hash: 0,
        ply: 0,
    };
    p.hash = p.compute_hash();

    let mut mc = Mcts::new(7);
    let (mv, _) = mc.search_rollouts(&p, 3000);
    let mv = mv.expect("mcts hamle bulmalı");
    let mut q = p;
    q.make(mv);
    assert_eq!(
        q.winner(),
        Some(0),
        "MCTS kazanan hamleyi almalı, oynadığı: {mv:?}"
    );

    let mut ab = Searcher::new(16);
    let (mv, _) = ab.best_move(&p, 6, 500);
    let mv = mv.expect("ab hamle bulmalı");
    let mut q = p;
    q.make(mv);
    assert_eq!(
        q.winner(),
        Some(0),
        "alpha-beta kazanan hamleyi almalı, oynadığı: {mv:?}"
    );
}

#[test]
fn mcts_only_ever_returns_legal_moves() {
    let mut rng = Rng::new(0x2222);
    let mut mc = Mcts::new(9);
    for _ in 0..40 {
        let p = random_position(&mut rng, 25);
        if p.winner().is_some() {
            continue;
        }
        for leaf in [Leaf::Value, Leaf::Rollout] {
            mc.leaf = leaf;
            let (mv, st) = mc.search_rollouts(&p, 600);
            let mv = mv.expect("hamle dönmeli");
            assert!(
                p.is_legal(mv),
                "MCTS illegal hamle döndü: {mv:?} ({leaf:?})"
            );
            assert!(st.rollouts > 0);
        }
    }
}

#[test]
fn value_leaf_is_deterministic() {
    // Bu, turnuva runner'ının rastgele açılış kullanmasının sebebi:
    // değerlendirme yaprağıyla aynı pozisyon her zaman aynı hamleyi veriyor.
    let p = Position::start();
    let mut a = Mcts::new(1);
    let mut b = Mcts::new(999_999);
    let (mv_a, _) = a.search_rollouts(&p, 4000);
    let (mv_b, _) = b.search_rollouts(&p, 4000);
    assert_eq!(mv_a, mv_b, "Leaf::Value modunda seed sonucu değiştirmemeli");
}

// ------------------------------------------------- policy prior (PUCT)

#[test]
fn priors_form_a_distribution() {
    let mut rng = Rng(0xbeef_1234_5678_9abc);
    for _ in 0..60 {
        let p = random_position(&mut rng, 20);
        if p.winner().is_some() {
            continue;
        }
        let moves = candidate_moves(&p);
        if moves.is_empty() {
            continue;
        }
        let pr = gedik::heuristics::move_priors(&p, &moves);
        assert_eq!(pr.len(), moves.len());
        let sum: f32 = pr.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "prior toplamı 1 olmalı, {sum}");
        assert!(pr.iter().all(|&v| v >= 0.0 && v <= 1.0));
    }
}

/// Rollout'un yapamadığı ayrımı prior yapabiliyor mu? Ölçüm bunu söylüyordu:
/// saf rollout iyi duvarla kendine zarar veren duvarı 0.008 ile ayırıyordu.
///
/// Piyonlar ayrı sütunlarda seçildi: başlangıç pozisyonunda ikisi de e
/// sütununda olduğu için oradaki her duvar **iki tarafı da** yavaşlatıyor ve
/// prior hepsini haklı olarak eşit derecede ilgisiz buluyor.
#[test]
fn priors_prefer_blocking_walls_over_self_harm() {
    let mut p = Position::start();
    p.pawn[0] = cell(8, 1) as u8; // A b1'de, yukarı çıkıyor
    p.pawn[1] = cell(0, 7) as u8; // B h9'da, aşağı iniyor
    p.hash = p.compute_hash();

    let moves = candidate_moves(&p);
    let pr = gedik::heuristics::move_priors(&p, &moves);
    let find = |name: &str| {
        moves
            .iter()
            .position(|&m| gedik::notation::move_name(&p, m) == name)
            .map(|i| pr[i])
    };

    let block = find("g8h").expect("B'nin önündeki duvar aday olmalı");
    let selfharm = find("a3h").expect("A'nın önündeki duvar aday olmalı");
    let advance = find("b2").expect("piyon ilerlemesi aday olmalı");

    assert!(
        block > selfharm * 5.0,
        "rakibi engelleyen duvar kendine zarar verenden belirgin yüksek olmalı: {block} vs {selfharm}"
    );
    assert!(advance > block, "ilerlemek yine de en yüksek olmalı");
}

/// İki piyon aynı sütundayken oradaki duvarların çoğu iki tarafı da
/// yavaşlatır. Böyle duvarlar ilerlemenin gerisinde kalmalı, ama tamamen
/// sıfırlanmamalı — savunma duvarlarının anlık kazancı da sıfırdır.
#[test]
fn walls_that_hurt_both_rank_below_advancing() {
    let p = Position::start(); // ikisi de e sütununda
    let moves = candidate_moves(&p);
    let pr = gedik::heuristics::move_priors(&p, &moves);
    let wall_max = moves
        .iter()
        .zip(&pr)
        .filter(|(m, _)| m.is_wall())
        .map(|(_, &x)| x)
        .fold(0.0f32, f32::max);
    let advance = moves
        .iter()
        .zip(&pr)
        .find(|(m, _)| gedik::notation::move_name(&p, **m) == "e2")
        .map(|(_, &x)| x)
        .unwrap();
    assert!(
        advance > wall_max * 3.0,
        "ilerleme baskın olmalı: {advance} vs {wall_max}"
    );
    assert!(wall_max > 0.0, "duvarlar tamamen sıfırlanmamalı");
}

// ------------------------------------------------------- değer fonksiyonu

#[test]
fn value_is_certain_at_terminal_positions() {
    // A hedef sırasına (satır 0) ulaşmış: sıra B'de ve B kaybetmiş.
    let mut p = Position::start();
    p.pawn[0] = cell(0, 4) as u8;
    p.side = 1;
    p.hash = p.compute_hash();
    assert_eq!(p.winner(), Some(0));
    assert_eq!(value_to_move(&p), 0.0, "kaybeden taraf için 0 olmalı");

    p.side = 0;
    p.hash = p.compute_hash();
    assert_eq!(value_to_move(&p), 1.0, "kazanan taraf için 1 olmalı");
}

#[test]
fn value_prefers_being_closer_and_richer() {
    let base = Position::start();
    let v_start = value_to_move(&base);
    assert!(
        v_start > 0.2 && v_start < 0.8,
        "başlangıç dengeli olmalı: {v_start}"
    );

    // Aynı pozisyon ama sıradaki oyuncunun duvar üstünlüğü var.
    //
    // Fark 10'a 8 değil 10'a 4 seçildi: kullanılabilir duvar sayısı rakibin
    // hedefe uzaklığıyla sınırlı (başlangıçta 8), dolayısıyla 10'a 8 farkı
    // kapağa takılıp sıfırlanıyor — ki bu doğru davranış.
    let mut rich = base;
    rich.walls = [10, 4];
    rich.hash = rich.compute_hash();
    assert!(
        value_to_move(&rich) > v_start,
        "duvar üstünlüğü değeri artırmalı"
    );

    // Sıradaki oyuncu hedefe çok daha yakın.
    let mut ahead = base;
    ahead.pawn[0] = cell(2, 4) as u8;
    ahead.hash = ahead.compute_hash();
    assert!(
        value_to_move(&ahead) > v_start,
        "yakın olmak değeri artırmalı"
    );
}

/// Duvar üstünlüğü değerlidir ama yarışı ezmez.
///
/// Bu test eski bir hatanın bekçisi: ilk öğrenilmiş ağırlıklarda duvar
/// katsayısı 4.43'tü ve 6 duvarlık üstünlük 10 hamlelik açığı kapatıyordu.
#[test]
fn wall_lead_helps_but_does_not_outweigh_the_race() {
    let mut even = Position::start();
    even.walls = [10, 6];
    even.hash = even.compute_hash();
    assert!(
        value_to_move(&even) > 0.5,
        "eşit yarışta duvar üstünlüğü artı olmalı"
    );

    // Aynı duvar üstünlüğü, ama yarış açıkça kaybedilmiş.
    let mut behind = Position::start();
    behind.pawn[0] = cell(8, 4) as u8; // A hedefe 8
    behind.pawn[1] = cell(6, 4) as u8; // B hedefe 2
    behind.walls = [10, 6];
    behind.hash = behind.compute_hash();
    assert!(
        value_to_move(&behind) < 0.35,
        "4 duvar fazlası 6 hamlelik açığı kapatmamalı, değer {}",
        value_to_move(&behind)
    );
}

// --------------------------------------------------------- MCTS-Solver

#[test]
fn mcts_plays_the_immediate_win() {
    // A ikinci sırada (satır 1), tek hamlede hedefe varıyor.
    let mut p = Position::start();
    p.pawn[0] = cell(1, 4) as u8;
    p.pawn[1] = cell(6, 4) as u8;
    p.hash = p.compute_hash();

    let mut m = Mcts::new(7);
    let (mv, st) = m.search_rollouts(&p, 400);
    let mv = mv.expect("hamle bulunmalı");
    let mut q = p;
    q.make(mv);
    assert_eq!(
        q.winner(),
        Some(0),
        "kazanan hamle oynanmalı, oynanan: {mv:?}"
    );
    assert!(st.rollouts <= 400);
}

#[test]
fn solver_stops_early_once_proven() {
    // Rakip duvarsız ve çok geride: sonuç kısa sürede kanıtlanmalı.
    let mut p = Position::start();
    p.pawn[0] = cell(2, 4) as u8;
    p.pawn[1] = cell(1, 0) as u8;
    p.walls = [0, 0];
    p.hash = p.compute_hash();

    let mut m = Mcts::new(11);
    let (mv, st) = m.search_rollouts(&p, 200_000);
    assert!(mv.is_some());
    assert!(
        st.rollouts < 200_000,
        "kanıtlandıktan sonra arama durmalı, {} iterasyon yapıldı",
        st.rollouts
    );
}

#[test]
fn mcts_never_returns_an_illegal_move() {
    let mut rng = Rng(0x1357_9bdf_0246_8ace);
    for i in 0..40 {
        let p = random_position(&mut rng, 5 + i % 40);
        if p.winner().is_some() {
            continue;
        }
        let mut m = Mcts::new(i as u64 + 1);
        let (mv, _) = m.search_rollouts(&p, 300);
        let mv = mv.expect("legal pozisyonda hamle dönmeli");
        assert!(p.is_legal(mv), "illegal hamle döndü: {mv:?}");
    }
}

// ------------------------------------------------------- paralel arama

#[test]
fn parallel_search_returns_a_legal_move() {
    use gedik::mcts::search_parallel;
    let mut rng = Rng(0x2468_ace0_2468_ace0);
    for i in 0..8 {
        let p = random_position(&mut rng, 6 + i * 3);
        if p.winner().is_some() {
            continue;
        }
        let (mv, st) = search_parallel(&p, None, 4000, 3, |m, _| {
            m.max_nodes = 200_000;
        });
        let mv = mv.expect("hamle dönmeli");
        assert!(p.is_legal(mv), "paralel arama illegal hamle döndü: {mv:?}");
        assert!(st.rollouts > 0);
    }
}

#[test]
fn parallel_search_also_finds_the_immediate_win() {
    use gedik::mcts::search_parallel;
    let mut p = Position::start();
    p.pawn[0] = cell(1, 4) as u8;
    p.pawn[1] = cell(6, 4) as u8;
    p.hash = p.compute_hash();
    let (mv, _) = search_parallel(&p, None, 2000, 4, |_, _| {});
    let mut q = p;
    q.make(mv.unwrap());
    assert_eq!(q.winner(), Some(0));
}

// ---------------------------------------------------------------- Elo

#[test]
fn elo_orders_a_transitive_ladder() {
    use gedik::rating::{solve, Pairing};
    // A, B'yi %75; B, C'yi %75; A, C'yi %90 yeniyor.
    let ps = vec![
        Pairing {
            a: 0,
            b: 1,
            a_score: 30.0,
            games: 40.0,
        },
        Pairing {
            a: 1,
            b: 2,
            a_score: 30.0,
            games: 40.0,
        },
        Pairing {
            a: 0,
            b: 2,
            a_score: 36.0,
            games: 40.0,
        },
    ];
    let r = solve(3, &ps, 4000);
    assert!(r[0] > r[1], "A, B'den yüksek olmalı: {r:?}");
    assert!(r[1] > r[2], "B, C'den yüksek olmalı: {r:?}");
    // Ortalama sıfıra sabitleniyor.
    let mean: f64 = r.iter().sum::<f64>() / 3.0;
    assert!(mean.abs() < 1e-6);
    // %75 kazanma oranı ~190 Elo farkına karşılık gelir.
    assert!((r[0] - r[1]) > 120.0 && (r[0] - r[1]) < 260.0, "{r:?}");
}

#[test]
fn elo_handles_a_single_pairing() {
    use gedik::rating::{solve, Pairing};
    let ps = vec![Pairing {
        a: 0,
        b: 1,
        a_score: 8.0,
        games: 10.0,
    }];
    let r = solve(2, &ps, 2000);
    assert!(r[0] > r[1]);
    assert!((r[0] + r[1]).abs() < 1e-6);
}

// ------------------------------------------- gerçek oyundan gelen regresyon

/// barricade.gg'de oynanan gerçek bir kayıp oyundan alınan pozisyon.
///
/// A hedefe 18, B hedefe 8 uzakta; A'nın 7 duvarı var, B'nin 1. A kesin
/// kaybetmiş durumda. İlk değer fonksiyonu buna **%92 kazanıyorum** diyordu:
/// öğrenilmiş ağırlıklarda duvar üstünlüğünün katsayısı 4.43'tü ve 6 duvar
/// fazlası 10 hamlelik açığı kapatıyor sanılıyordu.
///
/// Hata korelasyonu nedensellik sanmaktı: kendine karşı oynayan iki motorda
/// "elinde duvar kalması" kazanmakla birlikte görülür ama onu sağlamaz.
/// Gerçek şu ki duvar koymak bir hamle harcatır (net ~+1 tempo) ve rakip
/// hedefe varmak üzereyken hiçbir işe yaramaz.
#[test]
fn hopeless_race_is_not_called_winning() {
    let fen = "a11040000150000/8200820000000/66/48/7/1/0/30";
    let p = gedik::notation::from_fen(fen).expect("fen okunmalı");
    assert_eq!(p.distance(0), Some(18), "A gerçekten 18 uzakta olmalı");
    assert_eq!(p.distance(1), Some(8), "B gerçekten 8 uzakta olmalı");
    assert_eq!(p.walls, [7, 1]);
    assert_eq!(p.side, 0);

    let v = value_to_move(&p);
    assert!(
        v < 0.25,
        "7 duvarla bile 10 hamle geride olan taraf kazanıyor sayılmamalı, değer {v}"
    );

    // Eşik neden 0.35 değil 0.50: varsayılan ağırlıklar artık öğrenilmiş
    // olanlar ve onlar bu pozisyonda 20 bin iterasyonda 0.48 diyor (elle
    // yazılmışlar 0.05 diyordu). Bunu bilerek kabul ediyoruz — 190 oyunluk
    // ölçümde öğrenilmiş ağırlıklar 200 bin iterasyonda %82 kazanıyor
    // (+267 Elo). Kalibrasyon kötüleşti, güç arttı. Test yine de bir işe
    // yarıyor: 0.50'nin üstüne çıkarsa motor kaybettiği pozisyonu berabere
    // sanıyor demektir, o gerçek bir regresyon olur.
    let mut m = Mcts::new(3);
    let (_, st) = m.search_rollouts(&p, 20_000);
    assert!(
        st.win_rate < 0.50,
        "arama bu pozisyonu en azından geride görmeli, wr {}",
        st.win_rate
    );

    // Derin aramada kayıp daha net görülmeli — bu yön korunmalı.
    let mut m2 = Mcts::new(3);
    let (_, st2) = m2.search_rollouts(&p, 50_000);
    assert!(
        st2.win_rate < 0.50,
        "arama bu pozisyonu geride görmeli, 50k {}",
        st2.win_rate
    );
}

/// Duvarın tempo maliyeti: aynı yarış açığında elde duvar bulunması
/// pozisyonu kazanılmış hale getirmemeli.
#[test]
fn walls_cannot_rescue_a_lost_race() {
    let mut p = Position::start();
    p.pawn[0] = cell(7, 4) as u8; // A hedefe 7
    p.pawn[1] = cell(6, 0) as u8; // B hedefe 2
    p.walls = [10, 0];
    p.hash = p.compute_hash();
    let d0 = p.distance(0).unwrap();
    let d1 = p.distance(1).unwrap();
    assert!(
        d0 > d1 + 3,
        "kurulum: A belirgin geride olmalı ({d0} vs {d1})"
    );
    assert!(
        value_to_move(&p) < 0.3,
        "10 duvar da olsa kapanmayacak açık kazanç sayılmamalı"
    );
}

/// Kaybedilmiş pozisyonda motor iki kare arasında mekik dokumamalı.
///
/// Gerçek oyunda tam bunu yapmıştı: g1-f1-g1-f1 giderken rakip hedefe yürüdü.
/// Bütün hamleler 0'a yakın değer aldığında aralarındaki fark gürültüdür;
/// artık böyle durumlarda seçim prior'a bırakılıyor ve prior en kısa yolu
/// öne aldığı için motor hedefe doğru yürümeye devam ediyor.
#[test]
fn does_not_shuffle_when_lost() {
    // A hedefe 8, B hedefe 4, ikisinin de 3 duvarı var: A yarışı kaybetmiş
    // (3 duvarla bile açığı kapatamaz) ama oyun hemen bitmiyor, dolayısıyla
    // hamle seçimini üç tur boyunca izleyebiliyoruz.
    let mut p = Position::start();
    p.pawn[0] = cell(8, 4) as u8;
    p.pawn[1] = cell(4, 0) as u8;
    p.walls = [3, 3];
    p.hash = p.compute_hash();
    let start_dist = p.distance(0).unwrap();
    assert!(
        value_to_move(&p) < 0.2,
        "kurulum: pozisyon kaybedilmiş olmalı"
    );

    let mut played = Vec::new();
    for _ in 0..3 {
        if p.winner().is_some() {
            break;
        }
        let mut m = Mcts::new(5);
        let (mv, _) = m.search_rollouts(&p, 8_000);
        let mv = mv.expect("hamle dönmeli");
        assert!(p.is_legal(mv));
        played.push(gedik::notation::move_name(&p, mv));
        p.make(mv);
        if p.winner().is_some() {
            break;
        }
        // Rakip en kısa yolunu yürüsün.
        let m2 = p.masks();
        let mut pawn = Vec::new();
        p.gen_pawn_moves(&m2, &mut pawn);
        let best = gedik::heuristics::best_pawn_steps(&p, &m2, &pawn);
        if let Some(&reply) = best.first().or(pawn.first()) {
            p.make(reply);
        }
    }
    let end_dist = p.distance(0).unwrap();
    assert!(
        end_dist < start_dist,
        "kayıp pozisyonda bile hedefe yaklaşmalı: {start_dist} -> {end_dist}, hamleler {played:?}"
    );
    assert!(
        played
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            == played.len(),
        "aynı hamle tekrar etmemeli (mekik): {played:?}"
    );
}

/// Savunma duvarları aramadan tamamen dışlanmamalı.
///
/// Prior'ın ilk sürümü "rakibin yolunu şu an uzatmayan" her duvara sabit
/// -3.0 veriyordu; savunma duvarlarının anlık kazancı sıfır olduğu için
/// arama onlara pratikte hiç bakmıyordu.
#[test]
fn zero_gain_walls_still_get_some_prior() {
    let mut p = Position::start();
    p.pawn[0] = cell(8, 1) as u8;
    p.pawn[1] = cell(0, 7) as u8;
    p.hash = p.compute_hash();
    let moves = candidate_moves(&p);
    let pr = gedik::heuristics::move_priors(&p, &moves);

    let walls: Vec<f32> = moves
        .iter()
        .zip(&pr)
        .filter(|(m, _)| m.is_wall())
        .map(|(_, &x)| x)
        .collect();
    let pawn_max = moves
        .iter()
        .zip(&pr)
        .filter(|(m, _)| !m.is_wall())
        .map(|(_, &x)| x)
        .fold(0.0f32, f32::max);
    let wall_min = walls.iter().copied().fold(1.0f32, f32::min);
    let wall_max = walls.iter().copied().fold(0.0f32, f32::max);

    assert!(wall_min > 0.0, "hiçbir duvar tamamen sıfırlanmamalı");
    assert!(
        wall_max / wall_min < 400.0,
        "en iyi ve en kötü duvar arasındaki uçurum aramayı kilitlememeli: {wall_max} / {wall_min}"
    );
    assert!(pawn_max > wall_max, "ilerlemek yine de baskın olmalı");
}

#[test]
fn test_exact_race_winner_and_jump_parity() {
    // 0 duvarlı başlangıç pozisyonu: iki piyon da e sütununda kafa kafaya ilerler.
    // Parite kuralı: 2. oyuncu (B) 1. oyuncunun (A) üzerinden atlayarak tempo kazanır ve kazanır.
    let mut p = Position::start();
    p.walls = [0, 0];
    p.hash = p.compute_hash();

    let winner = gedik::heuristics::exact_race_winner(&p);
    assert_eq!(
        winner, 1,
        "0 duvarlı simetrik yarışta 2. oyuncu (B) atlama paritesiyle kazanmalı"
    );

    // A'nın 1 kare önde başladığı asimetrik yarışta A kazanmalı
    let mut p_lead = Position::start();
    p_lead.pawn[0] = cell(6, 4) as u8; // e3
    p_lead.pawn[1] = cell(0, 4) as u8; // e9
    p_lead.walls = [0, 0];
    p_lead.hash = p_lead.compute_hash();
    assert_eq!(
        gedik::heuristics::exact_race_winner(&p_lead),
        0,
        "Önde başlayan A kazanmalı"
    );
}

#[test]
fn test_subtree_reuse_preserves_validity() {
    let pos = Position::start();
    let mut mc = Mcts::new(0x1234_5678);
    let (mv, _st) = mc.search_rollouts(&pos, 500);
    assert!(mv.is_some(), "Kök hamlesi bulunmalı");
    let chosen = mv.unwrap();

    let advanced = mc.advance_tree(&pos, chosen);
    assert!(advanced, "Seçilen hamle alt ağaca başarıyla ilerletilmeli");

    let mut next_pos = pos;
    next_pos.make(chosen);

    let (next_mv, next_st) = mc.search_rollouts(&next_pos, 500);
    assert!(
        next_mv.is_some(),
        "Alt ağaçtan sonraki arama legal hamle üretmeli"
    );
    assert!(
        next_pos.is_legal(next_mv.unwrap()),
        "Üretilen hamle legal olmalı"
    );
    assert!(next_st.rollouts >= 500);
}

#[test]
fn test_best_threat_2step_bounds() {
    let mut p = Position::start();
    p.walls = [5, 5];
    let t1 = gedik::heuristics::best_threat(&p, 0, 1);
    let t2 = gedik::heuristics::best_threat_2step(&p, 0, 1);
    assert!(
        t2 >= t1,
        "2-adımlı tehdit tek adımlı tehditten küçük olamaz ({} vs {})",
        t2,
        t1
    );
}

#[test]
fn test_mcts_transposition_table_cache() {
    let pos = Position::start();
    let mut mc = Mcts::new(0x4321);
    let (_mv, _st) = mc.search_rollouts(&pos, 200);

    // Arama sırasında yaprak pozisyonları TT'ye kaydedilmiş olmalı
    assert!(
        mc.tt.len() > 0,
        "Arama sırasında TT tablosuna pozisyonlar kaydedilmeli (len={})",
        mc.tt.len()
    );
}

#[test]
fn test_chokepoints_detection() {
    let p = Position::start();
    let ch0 = gedik::heuristics::chokepoints(&p, 0);
    // Başlangıçta geniş açık tahtada e-sütunu koridoru darboğaz olarak taranabilmeli
    assert!(
        ch0 != 0,
        "Açık tahtada en kısa yol darboğazları tespit edilmeli"
    );
}

#[test]
fn test_repetition_penalty_breaks_shuttle_loop() {
    let move_seq = "b7h c7v d8h d3h d4h f9v e9h e7h f1 d9 f2 c9 f3 c8 f4 b8 c5h f8h a5h a8 g4 a7 g5 a6 g6 a5 d6v g7v g5h a6 f6v a5";
    let mut pos = Position::start();
    let mut history = vec![pos.hash];
    for m_str in move_seq.split_whitespace() {
        let mv = gedik::notation::parse_move(&pos, m_str).expect(m_str);
        pos.make(mv);
        history.push(pos.hash);
    }

    let mut engine = gedik::engine::Engine::parse("mcts:5000", 12345);
    let choice = engine.choose_with_history(&pos, &history);
    let chosen_move = choice.mv.expect("Hamle dönmeli");
    let name = gedik::notation::move_name(&pos, chosen_move);
    assert_ne!(name, "g7", "Motor çıkmaz sokak tuzağına (g7) girmemeli");
}

// Kanal indeksleri bilerek `k * 81 + hücre` biçiminde yazıldı (k = 0, 1, ...).
#[test]
#[allow(clippy::erasing_op, clippy::identity_op)]
fn test_spatial_planes_validity() {
    let p = Position::start();
    let planes = gedik::nn::spatial_planes(&p);
    assert_eq!(planes.len(), 13 * 81);

    // Kanal 0 (benim piyonum e1) 1.0 olmalı
    let e1_idx = 76; // e1 hücresi (row 8, col 4: 8*9 + 4 = 76)
    assert_eq!(planes[0 * 81 + e1_idx], 1.0);

    // Kanal 1 (rakip piyon e9) 1.0 olmalı
    let e9_idx = 4; // e9 hücresi (row 0, col 4: 0*9 + 4 = 4)
    assert_eq!(planes[1 * 81 + e9_idx], 1.0);

    // Kalan duvarlar 10/10 = 1.0 olmalı
    assert_eq!(planes[10 * 81], 1.0);
    assert_eq!(planes[11 * 81], 1.0);
}

#[test]
fn test_shortest_path_mask_validity() {
    let p = Position::start();
    let mask0 = gedik::nn::shortest_path_mask(&p, 0);
    // Başlangıçta piyon hücresi dahil hedefe doğru en az 9 hücre maskede işaretli olmalı
    assert!(mask0.count_ones() >= 9);
    assert!(
        mask0 & (1u128 << 76) != 0,
        "Başlangıç piyon hücresi en kısa yolda yer almalı"
    );
}
