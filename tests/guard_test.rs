use gedik::board::Position;
use gedik::guard::{detect_immediate_win, filter_root_moves};
use gedik::notation::parse_move;

#[test]
fn test_detect_immediate_win() {
    let mut pos = Position::start();
    // Beyazı e8 hücresine koy (row 1, col 4 = 13), hedef row 0 (e9)
    pos.pawn[0] = 13;
    pos.side = 0;
    let win_mv = detect_immediate_win(&pos);
    assert!(
        win_mv.is_some(),
        "e8'deki piyon e9'a basıp doğrudan kazanmalı"
    );
}

#[test]
fn test_filter_root_moves_chooses_safe_candidate() {
    let pos = Position::start();
    let e2 = parse_move(&pos, "e2").unwrap();
    let candidates = vec![(e2, 1000, 0.60)];
    let best = filter_root_moves(&pos, &candidates);
    assert_eq!(best, Some(e2));
}
