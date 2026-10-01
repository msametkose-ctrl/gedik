"""MLP değerlendirme ağını yalnızca numpy ile eğitir — torch gerekmez.

`egit.py`nin MLP yolunun bağımlılıksız karşılığı. Girdi `kodla`nın QDT3
dosyası (kodlama yine yalnızca Rust'ta), çıktı motorun okuduğu QNN1 `ag.bin`.

Asıl kullanım damıtma: `damit` ResNet'in yumuşak etiketlerini yazar, bu
betik hızlı MLP'yi onlarla eğitir. Etiketler 0/1 değil olasılık; kayıp
yumuşak hedefli ikili çapraz entropi.

Kullanım:
    python egit_np.py egitim.bin ag.bin [--epoch 8] [--h1 64] [--h2 32]
                      [--yukle eski_ag.bin] [--lr 1e-3]
"""
import argparse
import struct
import sys
import time

import numpy as np


def veri_oku(yol):
    with open(yol, "rb") as f:
        bas = f.read(24)
    if bas[:4] != b"QDT3":
        sys.exit(f"{yol}: QDT3 degil (kodla'yi --spatial olmadan calistir)")
    n, n_sparse, n_dense, K, P = struct.unpack("<IIIII", bas[4:24])
    dt = np.dtype([("y", "<f4"), ("k", "<u2"), ("idx", "<u2", K), ("d", "<f4", n_dense),
                   ("pa", "<u2", P), ("pp", "<f4", P)])
    ham = np.memmap(yol, dtype=dt, mode="r", offset=24, shape=(n,))
    idx = np.array(ham["idx"], dtype=np.int32)
    # Bos yuvalar (0xFFFF) en sondaki kukla sutuna gidiyor, sonra atiliyor.
    idx[idx == 0xFFFF] = n_sparse + n_dense
    return idx, np.array(ham["d"], dtype=np.float32), np.array(ham["y"], dtype=np.float32), n_sparse, n_dense


def girdi(idx, d, satir, n_in, n_sparse):
    """Seyrek indeksleri + yogun ozellikleri tek bir (B, n_in) matrise acar."""
    b = len(satir)
    x = np.zeros((b, n_in + 1), dtype=np.float32)
    x[np.arange(b)[:, None], idx[satir]] = 1.0
    x = x[:, :n_in]
    x[:, n_sparse:] = d[satir]
    return x


def qnn1_oku(yol, n_in):
    buf = open(yol, "rb").read()
    if buf[:4] != b"QNN1":
        sys.exit(f"{yol}: QNN1 degil")
    n, h1, h2 = struct.unpack("<III", buf[4:16])
    if n != n_in:
        sys.exit(f"{yol}: girdi {n}, beklenen {n_in}")
    a = np.frombuffer(buf, dtype="<f4", offset=16)
    o = 0

    def al(k):
        nonlocal o
        t = a[o:o + k].copy()
        o += k
        return t

    w1 = al(n_in * h1).reshape(n_in, h1)
    b1 = al(h1)
    w2 = al(h1 * h2).reshape(h1, h2)
    b2 = al(h2)
    w3 = al(h2)
    b3 = al(1)
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "w3": w3, "b3": b3}


def qnn1_yaz(p, yol):
    """Motorun okudugu yerlesim: w1 [n_in][h1], w2 [h1][h2] (bkz. nn.rs MlpNet)."""
    n_in, h1 = p["w1"].shape
    h2 = p["w2"].shape[1]
    with open(yol, "wb") as f:
        f.write(b"QNN1")
        f.write(struct.pack("<III", n_in, h1, h2))
        for k in ("w1", "b1", "w2", "b2", "w3", "b3"):
            f.write(np.ascontiguousarray(p[k], dtype="<f4").tobytes())


def ileri(p, x):
    z1 = x @ p["w1"] + p["b1"]
    a1 = np.maximum(z1, 0)
    z2 = a1 @ p["w2"] + p["b2"]
    a2 = np.maximum(z2, 0)
    z3 = a2 @ p["w3"] + p["b3"][0]
    return z1, a1, z2, a2, z3


def kayip(z, y):
    # Sayisal olarak guvenli BCE-with-logits: max(z,0) - z*y + log(1+exp(-|z|))
    return float(np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("girdi")
    ap.add_argument("cikti")
    ap.add_argument("--epoch", type=int, default=8)
    ap.add_argument("--h1", type=int, default=64)
    ap.add_argument("--h2", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--yukle", default="", help="baslangic agirliklari (ayni boyutta QNN1)")
    ap.add_argument("--tohum", type=int, default=7)
    a = ap.parse_args()

    idx, dense, y, n_sparse, n_dense = veri_oku(a.girdi)
    n = len(y)
    n_in = n_sparse + n_dense
    kes = int(n * 0.95)  # son %5 dogrulama: dosya oyun sirasinda, yani ayri oyunlar
    print(f"{n:,} satir ({kes:,} egitim), girdi {n_in}, ag {n_in}->{a.h1}->{a.h2}->1", flush=True)

    rng = np.random.default_rng(a.tohum)
    if a.yukle:
        p = qnn1_oku(a.yukle, n_in)
        if p["w1"].shape[1] != a.h1 or p["w2"].shape[1] != a.h2:
            sys.exit("--yukle boyutu --h1/--h2 ile uyusmuyor")
        print(f"baslangic: {a.yukle}", flush=True)
    else:
        p = {
            "w1": (rng.standard_normal((n_in, a.h1)) * np.sqrt(2.0 / 40)).astype(np.float32),
            "b1": np.zeros(a.h1, np.float32),
            "w2": (rng.standard_normal((a.h1, a.h2)) * np.sqrt(2.0 / a.h1)).astype(np.float32),
            "b2": np.zeros(a.h2, np.float32),
            "w3": (rng.standard_normal(a.h2) * np.sqrt(1.0 / a.h2)).astype(np.float32),
            "b3": np.zeros(1, np.float32),
        }
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(v) for k, v in p.items()}
    b1_, b2_, eps = 0.9, 0.999, 1e-8
    adim = 0
    toplam_adim = a.epoch * ((kes + a.batch - 1) // a.batch)

    dg = np.arange(kes, n)

    def dogrulama():
        top = 0.0
        for s in range(0, len(dg), 16384):
            satir = dg[s:s + 16384]
            z = ileri(p, girdi(idx, dense, satir, n_in, n_sparse))[-1]
            top += kayip(z, y[satir]) * len(satir)
        return top / max(len(dg), 1)

    # Hedeflerin kendi entropisi: yumusak etikette kaybin inebilecegi taban.
    yy = np.clip(y[dg], 1e-6, 1 - 1e-6)
    taban = float(np.mean(-(yy * np.log(yy) + (1 - yy) * np.log(1 - yy))))
    en_iyi = dogrulama()
    en_iyi_p = {k: t.copy() for k, t in p.items()}
    print(f"  baslangic dogrulama {en_iyi:.4f}  (taban {taban:.4f})", flush=True)

    t0 = time.time()
    for ep in range(a.epoch):
        sira = rng.permutation(kes)
        top, adet = 0.0, 0
        for s in range(0, kes, a.batch):
            satir = np.sort(sira[s:s + a.batch])
            x = girdi(idx, dense, satir, n_in, n_sparse)
            t = y[satir]
            z1, a1, z2, a2, z3 = ileri(p, x)
            top += kayip(z3, t) * len(satir)
            adet += len(satir)

            b = len(satir)
            dz3 = (1.0 / (1.0 + np.exp(-z3)) - t) / b
            g = {}
            g["w3"] = a2.T @ dz3
            g["b3"] = np.array([dz3.sum()], np.float32)
            dz2 = np.outer(dz3, p["w3"]) * (z2 > 0)
            g["w2"] = a1.T @ dz2
            g["b2"] = dz2.sum(0)
            dz1 = (dz2 @ p["w2"].T) * (z1 > 0)
            g["w1"] = x.T @ dz1
            g["b1"] = dz1.sum(0)

            # Kosinus ogrenme orani, AdamW
            adim += 1
            lr = a.lr * 0.5 * (1 + np.cos(np.pi * min(adim / toplam_adim, 1.0)))
            for k in p:
                m[k] = b1_ * m[k] + (1 - b1_) * g[k]
                v[k] = b2_ * v[k] + (1 - b2_) * g[k] * g[k]
                mh = m[k] / (1 - b1_ ** adim)
                vh = v[k] / (1 - b2_ ** adim)
                p[k] -= (lr * (mh / (np.sqrt(vh) + eps) + a.wd * p[k])).astype(np.float32)

        d = dogrulama()
        yildiz = ""
        if d < en_iyi:
            en_iyi = d
            en_iyi_p = {k: t.copy() for k, t in p.items()}
            yildiz = "  <- en iyi"
        print(f"  epoch {ep + 1}/{a.epoch}  egitim {top / adet:.4f}  dogrulama {d:.4f}  "
              f"({time.time() - t0:.0f} sn){yildiz}", flush=True)

    qnn1_yaz(en_iyi_p, a.cikti)
    print(f"yazildi: {a.cikti} (dogrulama {en_iyi:.4f}, taban {taban:.4f})", flush=True)


if __name__ == "__main__":
    main()
