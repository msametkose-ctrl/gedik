# Gedik

[![CI](https://github.com/msametkose-ctrl/gedik/actions/workflows/ci.yml/badge.svg)](https://github.com/msametkose-ctrl/gedik/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A Quoridor engine written in Rust. Bitboard rules core, PUCT tree search,
and a small learned evaluation network. No dependencies — `cargo build`
works offline.

*Gedik* is Turkish for **the breach in a wall**. That is the whole game:
open one for yourself, deny your opponent theirs.

```
     a   b   c   d   e   f   g   h   i
 9 | .   .   .   .   .   .   .   .   . |
   |                    === ===        |
 8 | .   .   .   .   .   .   A   .   . |
   |        === ===                    |
 7 | .   .   .   . | .   .   .   .   . |
```

---

## Quick start

You need [Rust](https://rustup.rs). Then:

```sh
./basla.sh          # Linux/macOS  — builds, serves, opens the browser
basla.bat           # Windows
```

That is it. A board opens at `http://127.0.0.1:8080` and you can play.
First build takes about 15 seconds; after that it is instant.

The web UI lets you play against the engine, watch two engines play each
other, replay positions from FEN, and see a live strength table.

## Playing from the command line

```sh
cargo build --release

./target/release/gedik play 0                    # you move first
./target/release/gedik play 1 mcts:60000         # engine moves first, weaker
./target/release/gedik match mcts:200000 mcts:20000 12 42 6
./target/release/gedik tournament 4 mcts:20000 mcts:60000 ab:300ms
./target/release/gedik perft 4
```

## Engine strings

Every command takes an engine spec:

| spec | meaning |
|---|---|
| `mcts:300000` | 300 000 PUCT iterations |
| `mcts:3000ms` | 3 second budget instead |
| `mcts:200000:t=0` | use all cores (root parallel) |
| `mcts:200000:nn=0` | disable the network, use the linear evaluation |
| `mcts:200000:w=0` | use the hand-written weights instead of the fitted ones |
| `mctsu:20000` | classic UCT, no policy prior |
| `mctsr:20000` | random rollouts instead of an evaluation leaf |
| `ab:1000ms` | alpha-beta baseline |

Keys: `cp` c_puct · `fpu` first-play urgency · `et` expand threshold ·
`mn` node cap · `mc` max children · `noise` root noise · `t` threads ·
`nn` network on/off · `pol` learned move ordering · `pt` its temperature ·
`sq` non-saturating value · `pr` priors · `sv` solver.

---

## How strong, and how do we know

Every claim below is a measured match: random openings, both colours,
hundreds of games. The Elo figure comes from the score, the bracket is a
Wilson 95% interval.

| change | result | Elo |
|---|---|---|
| evaluation leaf instead of random rollouts | 93.8% | +450 |
| PUCT with policy priors instead of plain UCT | 80.0% | +241 |
| fitted linear weights instead of hand-written | 149–32 | **+267** |
| neural evaluation instead of linear | 473–104 | **+263** |

Search scaling, measured on the same machine:

| budget | result | Elo |
|---|---|---|
| 20 000 → 200 000 iterations | 132–62 (68.0%) | +131 |
| 200 000 → 2 000 000 iterations | 19–17 (52.8%) | +19, inconclusive |

**Search stops paying somewhere past 200 000 iterations.** That is the most
useful thing in this table. It means a 10-second think is not meaningfully
stronger than a 0.3-second one, and it means the evaluation — not the
search — is the ceiling. It is also why the network is affordable: it costs
1.5× per node, and that compute was going to waste anyway.

Things that were tried and **rejected on the evidence**:

| idea | result |
|---|---|
| deeper, narrower tree (`et=16`) | 65–135 (32.5%), −127 Elo |
| cap children per node (`mc=24`) | 85–105 (44.7%), inconclusive |
| threat terms in the *linear* evaluation | 50.0%, no effect |

The last one is worth a note: the same information, fed to the *network*
instead of hand-weighted into the linear formula, was worth roughly +134
Elo. The feature was fine; the linear model could not use it.

### What the network sees that the formula did not

The linear evaluation sees four numbers: both path lengths and both wall
counts. It cannot see **where** the walls are. Here is a real position it
lost from:

```
move   my path   opponent can lengthen it by
 a8h         5                             0     <- unbreakable, walks home and wins
  d8         4                             4     <- shorter path, but a8v makes it 8
```

The old engine played `d8` with 182 000 of 200 000 visits and lost. `a8h`
wins 4–0 in playouts; `d8` loses 0–4. The trick is that `a8h` occupies the
wall centre the opponent needs for `a8v` — two walls cannot share a centre,
so the threat evaporates. Path length alone cannot express that.

Reproduce it:

```sh
./target/release/savun "e2,e8,e3,e7,e4,e6,e5,e4,e6,e3,e7,e2,e2h,e3h,c2h,c3h,\
a2h,g3h,f2v,d2,e8,d9h,f9v,e8v,c8h,c2,d7v,b2,e7h,b9h"
```

---

## Design

**Bitboards.** 81 cells in a `u128`; the 8×8 grid of wall centres fits two
`u64`s, one per orientation. Direction masks are derived from the walls in
8 iterations, so move generation and reachability are pure bitwise work.

**Slatton's shortcut.** A wall can only cut the board if at least two of its
three lattice nodes are already anchored. Checking that first skips the
expensive flood fill on almost every wall.

**Fixed action ids (0..140).** Independent of position, so a policy head
maps straight onto them.

**PUCT search** with policy priors, first-play urgency, an expansion
threshold, a node ceiling, and an MCTS-Solver that propagates proven
wins and losses. Root parallelisation for multi-core.

**Evaluation.** Either a 12-feature logistic function or a small network:
313 sparse inputs (pawn squares, every wall slot, wall counts, side to move)
plus 16 dense features → 64 → 32 → 1. About 23 000 parameters, ~4 000
multiplies per evaluation thanks to the sparse first layer.

The dense features include the two that matter most and are not visible in
the raw board: **how far can one opponent wall lengthen my path**, and
**is my path sealed**. They cost 0.7 µs to compute and pay for themselves
many times over.

---

## Training

The pipeline is three steps, and the network file is loaded at runtime —
you do not have to rebuild to swap models.

```sh
# 1. self-play data: positions, game results, and the search's move
#    distribution. Run several in parallel, one per core.
./target/release/uret 1000000 20000 <seed> 0 > data/part1.csv

# 2. encode. Feature extraction lives ONLY here, in Rust. The trainer
#    never parses a board, so the two can never disagree.
cat data/*.csv > data/all.csv
./target/release/kodla data/all.csv data/train.bin

# 3. train (PyTorch, CUDA if available)
python tools/egit.py data/train.bin ag.bin --epoch 30 --h1 64 --h2 32
```

Then just restart the server; `ag.bin` next to the binary is picked up
automatically. Delete it to fall back to the linear evaluation.

A note on step 2: keeping the encoder in one language is not fussiness.
A Python/Rust feature mismatch trains a model that scores well offline and
plays like garbage, and the cause is nearly invisible. The Rust and PyTorch
forward passes were checked against each other and agree to 1.2 × 10⁻⁷.

`tools/fit_np.py` fits the 12-feature linear model instead, if you want the
dependency-free evaluation.

---

## Development

```sh
cargo test --release      # 52 tests: rules, perft, search invariants
cargo run --release --bin gedik -- perft 5
```

The tests cover the rule corners that are easy to get wrong — mandatory
straight jumps, diagonal jumps when blocked, the no-full-block rule — plus
perft node counts and search invariants like "never return an illegal move"
and "take an available win".

Running several matches in parallel and want to watch? Redirect each to
`deney/<name>__<n>.txt` and the web UI aggregates them live, with Wilson
intervals and an Elo estimate, so you can stop early once a result is
decided.

Code comments are in Turkish. They explain *why*, including the mistakes —
several comments exist specifically to record a wrong turn so it is not
repeated.

## Contributing

Pull requests welcome. There is one rule, and it is the reason this engine
works: **a change to playing strength is accepted on match results, not on
argument.**

```sh
./target/release/gedik match "mcts:200000:<your change>" "mcts:200000" 12 42 6
```

Every time this project skipped that step it was wrong — including three
times in a row on the same change, which cost it 267 Elo until someone
finally played the games. [CONTRIBUTING.md](CONTRIBUTING.md) has the details
and a list of what actually needs doing.

The largest open problem: **move ordering is still hand-written and has
never been trained.** The network format, the search hooks (`pol=`, `pt=`)
and the training data are all in place; the first attempt measured worse
than the hand-written version and nobody knows why yet.

## Credits

The move-ordering approach follows the ideas in
[gorisanson/quoridor-ai](https://github.com/gorisanson/quoridor-ai).
Notation is Glendenning's.

## License

MIT — see [LICENSE](LICENSE).
