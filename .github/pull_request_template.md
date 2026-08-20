## What this changes

<!-- One or two sentences. -->

## Measurement

<!--
If this touches playing strength, paste a match result. Anything else is
a guess, and this project has been wrong every time it guessed.

  ./target/release/gedik match "mcts:200000:<change>" "mcts:200000" 12 42 6

24 games is roughly ±20%. If your interval includes 50%, say so — an
honest "inconclusive" is useful and will not be held against you.

Not needed for docs, refactors with no behaviour change, or bug fixes with
a failing test that now passes.
-->

```
paste match output here
```

## Checklist

- [ ] `cargo test --release` passes
- [ ] `cargo fmt` clean
- [ ] No new clippy warnings
- [ ] If move generation changed, `perft` counts are updated and called out
