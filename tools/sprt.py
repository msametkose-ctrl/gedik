"""SPRT ile terfi testi: A, B'den Elo1 kadar güçlü mü, yoksa en fazla Elo0 mı?

Birkaç `gedik match` parçasını farklı seed'lerle paralel çalıştırır, çıktıları
`deney/<isim>__<n>.txt` dosyalarına yazar (web arayüzü bunları da toplar) ve
her oyun bittiğinde log-olabilirlik oranını (LLR) günceller. LLR sınırlardan
birini geçince bütün parçaları durdurur.

Örnek (tek thread'li motorlar, 30 parça):
    python tools/sprt.py "mcts:400ms:ag=yeni.bin" "mcts:400ms" --isim yeni --parca 30

Tüm çekirdek kullanan motorlarda (t=0) parçalar birbirinin CPU'sunu çalar;
o durumda --parca 1 kullan.

Neden oyun yerine açılış çifti: her açılış iki renkle oynanır, ikisi aynı
açılışın bilgisini taşır. Çifti tek örnek sayıp (0, 0.5, 1) puanla varyansı
çiftlerden ölçmek, oyunları bağımsız saymaktan daha dürüst (Fishtest'in
pentanomial yaklaşımının beraberliksiz hali). Genelleştirilmiş SPRT:
LLR = n (s1 - s0)(2 m - s0 - s1) / (2 v)
"""

import argparse
import math
import os
import re
import subprocess
import sys
import threading
import time

SATIR = re.compile(r"oyun\s+(\d+)/\d+\s+açılış\s+(\d+).*kazanan (\S)")


def elo_skor(elo):
    return 1.0 / (1.0 + 10 ** (-elo / 400.0))


def skor_elo(s):
    s = min(max(s, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10(1.0 / s - 1.0)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    m = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return m - h, m + h


def llr(ciftler, s0, s1):
    n = len(ciftler)
    if n < 2:
        return 0.0
    m = sum(ciftler) / n
    # Varyansa 4 sözde çift (0, 0.5, 0.5, 1; varyansı 0.125) ekle. Eskiden
    # yalnız 1e-3 taban vardı: renk avantajı yüzünden çiftler 1-1 bitince
    # varyans ~0 çıkıyor ve LLR birkaç çiftte sahte bir ret sınırına
    # iniyordu (2,5 sn maçında 2 çiftte -0.83). Sözde çiftler az oyunda
    # temkinli, çok oyunda etkisiz.
    v = (sum((x - m) ** 2 for x in ciftler) + 4 * 0.125) / (n + 4)
    return n * (s1 - s0) * (2 * m - s0 - s1) / (2 * v)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--isim", default="sprt")
    ap.add_argument("--parca", type=int, default=30)
    ap.add_argument("--acilis", type=int, default=4, help="açılış yarım hamle")
    ap.add_argument("--elo0", type=float, default=0.0)
    ap.add_argument("--elo1", type=float, default=20.0)
    ap.add_argument("--alfa", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--azami", type=int, default=4000, help="en fazla oyun")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--exe", default=os.path.join("target", "release", "gedik"))
    args = ap.parse_args()

    args.exe = os.path.normpath(args.exe)
    if os.name == "nt" and not args.exe.lower().endswith(".exe"):
        args.exe += ".exe"
    s0, s1 = elo_skor(args.elo0), elo_skor(args.elo1)
    alt = math.log(args.beta / (1 - args.alfa))
    ust = math.log((1 - args.beta) / args.alfa)
    os.makedirs("deney", exist_ok=True)

    kilit = threading.Lock()
    # (parça, açılış) -> o çiftte A'nın kazandığı oyunlar listesi
    acik = {}
    ciftler = []
    durum = {"a": 0, "b": 0, "bitmedi": 0}
    surecler = []

    cift_basina = max(1, args.azami // (2 * args.parca))

    def oku(i, p, dosya):
        for satir in p.stdout:
            dosya.write(satir)
            dosya.flush()
            m = SATIR.search(satir)
            if not m:
                continue
            acilis, kazanan = int(m.group(2)), m.group(3)
            with kilit:
                if kazanan == "A":
                    durum["a"] += 1
                elif kazanan == "B":
                    durum["b"] += 1
                else:
                    durum["bitmedi"] += 1
                anahtar = (i, acilis)
                acik.setdefault(anahtar, []).append(kazanan)
                if len(acik[anahtar]) == 2:
                    sonuc = acik.pop(anahtar)
                    # Bitmeyen oyun yarım puan sayılır.
                    ciftler.append(
                        sum(1.0 if k == "A" else 0.5 if k == "-" else 0.0 for k in sonuc) / 2
                    )
        dosya.close()

    # Aynı isimle önceki parçalar varsa (kesilmiş bir maç) onların bitmiş
    # çiftlerini sayıma kat; yeni parçalar sonraki numaralardan ve farklı
    # seed'lerle başlar, açılışlar tekrar etmez.
    ilk = 0
    while os.path.exists(os.path.join("deney", f"{args.isim}__{ilk + 1}.txt")):
        with open(os.path.join("deney", f"{args.isim}__{ilk + 1}.txt"), encoding="utf-8") as f:
            onceki = {}
            for satir in f:
                m = SATIR.search(satir)
                if not m:
                    continue
                k = m.group(3)
                durum["a" if k == "A" else "b" if k == "B" else "bitmedi"] += 1
                onceki.setdefault(int(m.group(2)), []).append(k)
            for sonuc in onceki.values():
                if len(sonuc) == 2:
                    ciftler.append(
                        sum(1.0 if k == "A" else 0.5 if k == "-" else 0.0 for k in sonuc) / 2
                    )
        ilk += 1
    if ilk:
        print(f"önceki {ilk} parçadan {len(ciftler)} çift alındı", flush=True)

    for i in range(ilk, ilk + args.parca):
        yol = os.path.join("deney", f"{args.isim}__{i + 1}.txt")
        dosya = open(yol, "w", encoding="utf-8")
        p = subprocess.Popen(
            [args.exe, "match", args.a, args.b, str(cift_basina),
             str(args.seed + i * 100003), str(args.acilis)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace",
        )
        surecler.append(p)
        threading.Thread(target=oku, args=(i, p, dosya), daemon=True).start()

    print(f"A = {args.a}\nB = {args.b}")
    print(f"SPRT elo0={args.elo0} elo1={args.elo1}  sınırlar [{alt:.2f}, {ust:.2f}]  "
          f"{args.parca} parça", flush=True)

    t0 = time.time()
    karar = None
    son_n = -1
    while True:
        time.sleep(2)
        with kilit:
            l = llr(ciftler, s0, s1)
            n = len(ciftler)
            a, b, u = durum["a"], durum["b"], durum["bitmedi"]
            ort = sum(ciftler) / n if n else 0.5
        if n != son_n:
            son_n = n
            lo, hi = wilson(a, a + b)
            print(f"[{time.time() - t0:6.0f} sn] {a}-{b}"
                  f"{f' ({u} bitmedi)' if u else ''}  çift {n}  "
                  f"%{100 * ort:.1f} ({skor_elo(ort):+.0f} Elo)  "
                  f"Wilson %{100 * lo:.0f}-{100 * hi:.0f}  LLR {l:+.2f}", flush=True)
        if l >= ust:
            karar = "H1 KABUL: A daha güçlü (terfi)"
        elif l <= alt:
            karar = "H0 KABUL: A yeterince güçlü değil (ret)"
        elif all(p.poll() is not None for p in surecler):
            karar = "karar çıkmadı: oyunlar bitti"
        if karar:
            break

    for p in surecler:
        if p.poll() is None:
            p.kill()
    a, b = durum["a"], durum["b"]
    lo, hi = wilson(a, a + b)
    print(f"\nSONUÇ: {karar}\nA {a} - {b} B  oran %{100 * a / max(1, a + b):.1f}  "
          f"Wilson %{100 * lo:.1f}-{100 * hi:.1f}  LLR {l:+.2f}  "
          f"({time.time() - t0:.0f} sn)")
    sys.exit(0 if karar.startswith("H1") else 1)


if __name__ == "__main__":
    main()
