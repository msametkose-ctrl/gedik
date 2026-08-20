"""Değer fonksiyonu ağırlıklarını öğrenir (numpy, vektörize, ağırlıklı).

İki kaynak birleştiriliyor:
  1. Self-play verisi — kalibrasyonu ve ince dengeleri o taşıyor.
  2. **Kesin bilinen pozisyonlar** — oyunun mantığından türetilmiş, tartışmaya
     kapalı örnekler. Küçük bir ağırlıkla ekleniyorlar.

İkincisi neden gerekli: veride "10 hamle geride ama 7 duvarım var" kadar uç
pozisyon azdır, model de daha ılımlı örneklerden doğrusal olarak dışarı uzatıp
duvara olduğundan fazla değer biçer. Doğrusal bir model tek başına "yarış
kaybedilince duvarın değeri sıfırlanır" diyemez. Bu örnekler tam o sınırı
gösteriyor.

Sonda akıl sağlığı kontrolleri var; geçmezse betik hata koduyla çıkar.
"""
import sys, numpy as np

NAMES = ["sabit", "tempo", "duvar farki", "evre", "tempo x evre", "ilerleme",
         "rakip duvarsiz", "ben duvarsiz", "iyimser yaris", "kotumser yaris",
         "kurtulamam", "kaybetmem"]

def feats(d_me, d_opp, w_me, w_opp):
    """heuristics.rs::features ile birebir aynı hesap."""
    tempo = d_opp - d_me
    usable_me, usable_opp = min(w_me, d_opp), min(w_opp, d_me)
    opt, pes = tempo + usable_me, tempo - usable_opp
    phase = (w_me + w_opp) / 20.0
    wall = (usable_me - usable_opp) / 4 if opt >= 0 else 0.0
    return [1.0, tempo / 4, wall, phase, tempo / 4 * phase,
            (d_me + d_opp) / 16, float(w_opp == 0), float(w_me == 0),
            max(-6, min(6, opt)) / 6, max(-6, min(6, pes)) / 6,
            float(opt < 0), float(pes > 0)]

def known_positions():
    """Sonucu oyunun mantığından belli olan pozisyonlar."""
    rows, labels = [], []
    for d_me in range(4, 20):
        for d_opp in range(1, 12):
            for w_me in range(0, 11, 2):
                for w_opp in range(0, 11, 5):
                    tempo = d_opp - d_me
                    opt = tempo + min(w_me, d_opp)
                    pes = tempo - min(w_opp, d_me)
                    # Bütün duvarlarımı kullansam bile yarışı alamıyorum:
                    # duvar başına net +1 tempo kazandırır, o kadar.
                    if opt <= -3:
                        rows.append(feats(d_me, d_opp, w_me, w_opp)); labels.append(0.0)
                    # Rakip bütün duvarlarını harcasa bile beni durduramaz.
                    elif pes >= 3:
                        rows.append(feats(d_me, d_opp, w_me, w_opp)); labels.append(1.0)
    return np.array(rows), np.array(labels)

SANITY = [
    ("kayip yaris: 18'e 8 geride, 7 duvar", feats(18, 8, 7, 1), (0.0, 0.25)),
    ("baslangic pozisyonu",                 feats(8, 8, 10, 10), (0.35, 0.65)),
    ("kazanilmis yaris: 3'e 9 onde, duvarsiz", feats(3, 9, 0, 0), (0.75, 1.0)),
]

def sig(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def load(paths):
    """Eski CSV'den özellikleri **yeniden** hesaplar.

    Veriyi tekrar üretmeye gerek yok: kaydedilmiş özelliklerden pozisyonun
    kendisi geri çözülebiliyor. f1 tempo'yu, f3 duvar toplamını, f2 duvar
    farkını, f5 mesafe toplamını veriyor; dördü birlikte d_me, d_opp, w_me,
    w_opp'u tek anlamlı biçimde belirliyor. Böylece özellik tasarımını
    değiştirdikçe aynı oyunlardan yeni özellik matrisi çıkarabiliyoruz.
    """
    Xs, ys = [], []
    for p in paths:
        a = np.genfromtxt(p, delimiter=",", invalid_raise=False)
        a = a[~np.isnan(a).any(axis=1)]
        old = a[:, :-3]
        tempo = np.rint(old[:, 1] * 4).astype(int)
        wsum = np.rint(old[:, 3] * 20).astype(int)
        wdiff = np.rint(old[:, 2] * 4).astype(int)   # eski f2 = ham duvar farki
        dsum = np.rint(old[:, 5] * 16).astype(int)
        w_me = (wsum + wdiff) // 2
        w_opp = wsum - w_me
        d_me = (dsum - tempo) // 2
        d_opp = dsum - d_me
        good = (w_me >= 0) & (w_opp >= 0) & (d_me >= 0) & (d_opp >= 0)
        rows = np.array([feats(int(a_), int(b_), int(c_), int(d_))
                         for a_, b_, c_, d_ in
                         zip(d_me[good], d_opp[good], w_me[good], w_opp[good])])
        Xs.append(rows); ys.append(a[good, -1])
    return np.vstack(Xs), np.concatenate(ys)

# Aramanın sömürebileceği özellikler.
#
# "rakip duvarsiz" / "ben duvarsiz" gibi eşik göstergeleri veriye iyi uyuyor
# ama değerlendirmede sıçrama yaratıyor: arama, rakip son duvarını harcadığında
# değerin birden +1.44 zıpladığını görüp o hattı kovalıyor ve kaybedilmiş bir
# pozisyonu kazanılmış sanıyor. Aynı şey oyun evresi için de geçerli — duvar
# harcandıkça evre düşüyor, değer yükseliyor, motor "duvar harca" diye
# ödüllendiriliyor.
#
# Taşıdıkları bilgi zaten `iyimser/kotumser yaris` ve `kurtulamam/kaybetmem`
# içinde var, o yüzden ağırlıklarını sıfırda tutuyoruz.
FROZEN = [3, 4, 6, 7]

# Egemenlik terimlerinin alt sınırı.
#
# `kurtulamam` (opt<0) ve `kaybetmem` (pes>0) oyunun mantığından gelen kesin
# ifadeler, istatistiksel bir eğilim değil. Serbest bıraktığımızda veri onları
# zayıflatıyordu (-0.65 gibi) çünkü kusurlu bir rakibe karşı o pozisyonlardan
# bazen dönülebiliyor. Zayıf bir ceza ise **aramanın kaçış deliği** oluyor:
# motor, cezanın kalktığı bir yaprak buluyor ve kaybedilmiş pozisyonu
# kazanılmış sanıyor. Ceza, hiçbir yaprağın kaçamayacağı kadar büyük olmalı.
BOUNDS = {10: (None, -2.5), 11: (2.5, None)}

def fit(X, y, sw, epochs=9000, lr=1.0, l2=1e-5):
    w = np.zeros(X.shape[1]); tot = sw.sum()
    mask = np.ones(X.shape[1]); mask[FROZEN] = 0.0
    for i, (lo, hi) in BOUNDS.items():
        w[i] = hi if hi is not None else lo
    for ep in range(epochs):
        p = sig(X @ w)
        w -= lr * (X.T @ ((p - y) * sw) / tot + l2 * w) * mask
        for i, (lo, hi) in BOUNDS.items():
            if lo is not None: w[i] = max(w[i], lo)
            if hi is not None: w[i] = min(w[i], hi)
        if ep % 3000 == 0:
            pc = np.clip(p, 1e-9, 1 - 1e-9)
            ll = -np.sum(sw * (y * np.log(pc) + (1 - y) * np.log(1 - pc))) / tot
            print(f"  epoch {ep:>5}  loss {ll:.4f}", file=sys.stderr)
    return w

def report(X, y, w, name):
    p = sig(X @ w); pc = np.clip(p, 1e-9, 1 - 1e-9)
    print(f"{name}: n={len(X)}  dogruluk={np.mean((p>=0.5)==(y==1)):.4f}  "
          f"logloss={-np.mean(y*np.log(pc)+(1-y)*np.log(1-pc)):.4f}", file=sys.stderr)

if __name__ == "__main__":
    X, y = load(sys.argv[1:])
    m = np.arange(len(X)) % 5 != 0
    Xtr, ytr, Xva, yva = X[m], y[m], X[~m], y[~m]

    Xk, yk = known_positions()
    # Kesin örnekler self-play verisinin ~%15'i kadar ağırlık taşısın:
    # kalibrasyonu bozacak kadar baskın değil, sınırı öğretecek kadar güçlü.
    kw = 0.15 * len(Xtr) / len(Xk)
    print(f"{len(X)} self-play + {len(Xk)} kesin ornek (agirlik {kw:.2f})", file=sys.stderr)

    Xall = np.vstack([Xtr, Xk])
    yall = np.concatenate([ytr, yk])
    sw = np.concatenate([np.ones(len(Xtr)), np.full(len(Xk), kw)])

    w = fit(Xall, yall, sw)
    report(Xtr, ytr, w, "self-play egitim")
    report(Xva, yva, w, "self-play dogrulama")
    report(Xk, yk, w, "kesin ornekler")

    print("\nogrenilen agirliklar:", file=sys.stderr)
    for n_, v in zip(NAMES, w):
        print(f"  {n_:<16} {v:>8.3f}", file=sys.stderr)

    print("\nakil sagligi kontrolleri:", file=sys.stderr)
    ok = True
    for name, f, (lo, hi) in SANITY:
        v = float(sig(np.array(f) @ w)); good = lo <= v <= hi; ok &= good
        print(f"  [{'GECTI' if good else 'KALDI'}] {name}: {v:.4f} (beklenen {lo}-{hi})", file=sys.stderr)

    print("pub const VALUE_WEIGHTS: [f32; NUM_FEATURES] = [")
    print("    " + ", ".join(f"{v:.5}" for v in w) + ",")
    print("];")
    if not ok:
        print("\nUYARI: kontroller gecmedi.", file=sys.stderr); sys.exit(2)
