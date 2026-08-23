"""Self-play verisinden değer fonksiyonu ağırlıklarını öğrenir.

Lojistik regresyon, L2 düzenlileştirme, tam-toplu gradyan inişi.
Çıktı doğrudan heuristics.rs içine yapıştırılacak Rust dizisi.

Kullanım:  python3 tools/fit.py veri1.csv [veri2.csv ...]

Birleştirilmiş dosyalarda tekrar eden başlık satırları sorun değil, atlanıyor.

AKIL SAĞLIĞI KONTROLLERİ
------------------------
Öğrenilen ağırlıklar aşağıdaki apaçık pozisyonları doğru bilmek zorunda.
Bu kontroller bir kez pahalıya mal olmuş bir hatanın bekçisi: ilk model
"18'e 8 geride ama 7 duvarım var" pozisyonuna %92 kazanıyorum diyordu,
çünkü kendine karşı oynayan iki motorda duvar bulundurmak kazanmayla
*birlikte görünüyor* — sebebi olmuyor. Sayısal doğruluk iyi çıksa bile
bu kontroller geçmiyorsa ağırlıklar kullanılmaz.
"""
import sys, csv, math

# (ad, özellik vektörü, beklenen aralık)
SANITY = [
    ("kayip yaris: 18'e 8 geride, 7 duvar",
     [1, -2.5, 1.5, 0.4, -1, 1.625, 0, 0, -0.5, -1, 1, 0], (0.0, 0.25)),
    ("baslangic pozisyonu",
     [1, 0, 0, 1, 0, 1, 0, 0, 1, -1, 0, 0], (0.35, 0.65)),
]

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

def dot(w, x):
    return sum(wi * xi for wi, xi in zip(w, x))

def load(paths):
    X, y = [], []
    ncol = None
    for p in paths:
        with open(p) as f:
            for row in csv.reader(f):
                if not row or row[0] == "f0":     # baslik (birlestirmede tekrar eder)
                    continue
                try:
                    feats = [float(v) for v in row[:-3]]
                    label = int(row[-1])
                except ValueError:
                    continue
                if ncol is None:
                    ncol = len(feats)
                if len(feats) != ncol:
                    continue
                X.append(feats)
                y.append(label)
    return X, y, ncol

def fit(X, y, epochs=6000, lr=0.6, l2=2e-4):
    n, d = len(X), len(X[0])
    w = [0.0] * d
    for ep in range(epochs):
        g = [0.0] * d
        loss = 0.0
        for xi, yi in zip(X, y):
            p = sigmoid(dot(w, xi))
            e = p - yi
            for j in range(d):
                g[j] += e * xi[j]
            pc = min(max(p, 1e-9), 1 - 1e-9)
            loss -= yi * math.log(pc) + (1 - yi) * math.log(1 - pc)
        for j in range(d):
            w[j] -= lr * (g[j] / n + l2 * w[j])
        if ep % 1500 == 0:
            print(f"  epoch {ep:>5}  loss {loss/n:.4f}", file=sys.stderr)
    return w

def report(X, y, w, name):
    ok = sum(1 for xi, yi in zip(X, y) if (sigmoid(dot(w, xi)) >= 0.5) == (yi == 1))
    ll = -sum(
        yi * math.log(max(sigmoid(dot(w, xi)), 1e-9))
        + (1 - yi) * math.log(max(1 - sigmoid(dot(w, xi)), 1e-9))
        for xi, yi in zip(X, y)
    )
    print(f"{name}: n={len(X)}  dogruluk={ok/len(X):.4f}  logloss={ll/len(X):.4f}", file=sys.stderr)

def sanity(w):
    print("\nakil sagligi kontrolleri:", file=sys.stderr)
    passed = True
    for name, feats, (lo, hi) in SANITY:
        if len(feats) != len(w):
            print(f"  ATLANDI ({name}): ozellik sayisi uyusmuyor", file=sys.stderr)
            continue
        v = sigmoid(dot(w, feats))
        good = lo <= v <= hi
        passed &= good
        print(f"  [{'GECTI' if good else 'KALDI'}] {name}: {v:.4f} (beklenen {lo}-{hi})", file=sys.stderr)
    return passed

if __name__ == "__main__":
    paths = sys.argv[1:]
    if not paths:
        print("kullanim: python3 tools/fit.py veri.csv [...]", file=sys.stderr)
        raise SystemExit(1)
    X, y, ncol = load(paths)
    if not X:
        print("veri okunamadi", file=sys.stderr)
        raise SystemExit(1)
    tr = [i for i in range(len(X)) if i % 5 != 0]
    va = [i for i in range(len(X)) if i % 5 == 0]
    Xtr, ytr = [X[i] for i in tr], [y[i] for i in tr]
    Xva, yva = [X[i] for i in va], [y[i] for i in va]
    print(f"{len(X)} pozisyon, {ncol} ozellik, egitim {len(Xtr)}, dogrulama {len(Xva)}", file=sys.stderr)

    w = fit(Xtr, ytr)
    report(Xtr, ytr, w, "egitim")
    report(Xva, yva, w, "dogrulama")
    good = sanity(w)

    print("pub const VALUE_WEIGHTS: [f32; NUM_FEATURES] = [")
    print("    " + ", ".join(f"{v:.5}" for v in w) + ",")
    print("];")
    if not good:
        print("\nUYARI: akil sagligi kontrolleri gecmedi, bu agirliklari kullanma.", file=sys.stderr)
        raise SystemExit(2)
