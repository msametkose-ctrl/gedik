---
name: The engine played a bad move
about: The most useful report you can file
title: 'Bad move: '
labels: bad-move
---

**Move list** (from the start, comma or space separated)

```
e2,e8,e3,e7,...
```

**What it played:**
**What you think it should play:**
**Why:**

**Engine settings** (e.g. `mcts:300000`, or which option in the web UI):

---

Helpful if you have it — these two tools print exactly what the engine sees:

```sh
./target/release/analiz "<moves>" "mcts:200000"
./target/release/savun  "<moves>"
```

If you can, play the position out both ways and say who wins:

```sh
./target/release/gedik match ... # or just play it in the web UI
```
