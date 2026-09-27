use gedik::board::Position;
use gedik::endgame::EndgameSolver;
use gedik::notation::parse_move;

#[test]
fn test_endgame_solver_immediate_win() {
    let mut pos = Position::start();
    // Beyazı e8'e getir
    let e2 = parse_move(&pos, "e2").unwrap();
    pos.make(e2);

    let mut solver = EndgameSolver::new(16);
    let result = solver.solve(&pos, 4, 100);
    assert!(result.is_some(), "Endgame solver bir hamle bulmalı");
}
