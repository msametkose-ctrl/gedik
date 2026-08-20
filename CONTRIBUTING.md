# Contributing to Gedik

Contributions are welcome. This file is mostly about **one rule**, because
that rule is what makes this project work.

## The rule: measure it, or it does not go in

A change to the engine's playing strength is accepted or rejected on match
results. Not on reasoning, not on how obviously better it looks, not on how
much cleaner the code is. On games.

```sh
cargo build --release
./target/release/gedik match "mcts:200000:<your change>" "mcts:200000" 12 42 6
```

That is 12 random openings played from both colours, 24 games. Paste the
output into your pull request.

This is not bureaucracy. Every single time this project skipped that step,
it was wrong:

* A "deeper, narrower tree" looked obviously right. It measured **32.5%**
  — about 127 Elo *worse*.
* A fitted evaluation was rejected three times on a calibration probe that
  looked damning. When someone finally played the games, it was **+267 Elo**.
  A calibration measurement had been mistaken for a strength measurement,
  and the engine was held back for no reason.
* A threat feature was measured at "exactly 50%, no effect" and reverted.
  That was 12 games — an interval of roughly ±25%. Re-measured properly
  later, on the right model, the same idea was worth about **+134 Elo**.

Small samples lie confidently. 24 games gives you roughly ±20%. Before
claiming anything, ask whether your interval excludes 50%.

### If you have many cores

Run several shards with different seeds in parallel, redirect each to
`deney/<name>__<n>.txt`, and open the web UI — it aggregates them live with
a Wilson interval and an Elo estimate, so you can stop as soon as the result
is decided.

```sh
mkdir -p deney
for i in 1 2 3 4; do
  ./target/release/gedik match "mcts:200000:nn=1" "mcts:200000:nn=0" \
      20 $((100 + i)) 6 > deney/mine__$i.txt &
done
./target/release/gedik serve   # watch the table at 127.0.0.1:8080
```

## Before you open a pull request

```sh
cargo test --release      # 52 tests
cargo fmt                 # CI enforces this
cargo clippy --release --all-targets   # please don't add new warnings
```

CI runs build and tests on Linux, Windows and macOS, plus a `perft 4` node
count. If perft changes, move generation changed — say so explicitly in the
PR, because every measurement in the README was taken under the old rules.

## What actually needs doing

Roughly in order of expected value:

**A better move ordering.** The policy that decides which moves are worth
searching is still hand-written and has never been trained. A policy head
exists in the network format and in the search (`pol=1` root-only, `pol=2`
everywhere, `pt=` temperature) and the training data already contains the
search's visit distribution — but the first attempt measured *worse* than
the hand-written ordering, and nobody has worked out why. The network is not
obviously bad: it picks the search's top move 56% of the time and is in its
top three 84% of the time. Something about how it is used in the tree is
wrong. This is the biggest open lever in the project.

**Shuffling in lost positions.** Roughly 10% of self-play games hit the
400-ply cap. The engine knows it is losing, so every move looks the same to
it, and it stops making progress. There is a fallback that walks toward the
goal when the position looks lost, but it triggers on a fixed threshold and
clearly isn't enough. Games that never end also waste training data.

**Another bootstrap round.** The current network was trained on games played
by the *previous* engine. Regenerate with `uret ... 1` (network on) and
retrain — the standard self-improvement loop. Each round gives less than the
last, but there are rounds left.

**Endgame knowledge.** Once both players are past each other and no wall can
change anything, the result is decided by a subtraction. The engine still
searches those positions.

**Opening book.** Cheap Elo. The engine burns full searches on positions
that are theory.

**Search that scales past 300k.** Right now going from 200k to 2M iterations
is worth about +19 Elo, which is inside the noise. Whoever fixes that changes
the shape of the project.

## Code style

Comments are in Turkish. You do not have to write Turkish — English comments
in new code are fine and will not be rewritten.

What matters more: comments here explain **why**, not what. Several exist
specifically to record a wrong turn so nobody repeats it. If you fix
something subtle, leave a note about what fooled you. That is the most
valuable kind of comment in this codebase.

## Reporting a bad move

The most useful bug report is a position where the engine plays something
losing. Include the move list, what it played, and what you think it should
have played. `savun` and `analiz` are there to help you show it:

```sh
./target/release/analiz "e2,e8,e3,e7" "mcts:200000"
./target/release/savun  "e2,e8,e3,e7"
```

A single such report started the work that produced the network's threat
features, worth roughly +134 Elo.

## License

By contributing you agree your work is licensed under the MIT License.
