"""Değerlendirme ağını eğitir ve motorun okuduğu ag.bin dosyasını yazar.

Girdi `kodla.exe`nin ürettiği ikili dosya. Bu betik FEN görmüyor, tahtayı
bilmiyor, kodlama yapmıyor — sadece indeksleri okuyor. Kodlama tek yerde,
Rust tarafında. Python ile Rust'ın kodlamayı farklı yapması bu tür
projelerdeki en sinsi hata; o kapı kapalı.

Kullanım:
    python egit.py egitim.bin ag.bin [--epoch 12] [--h1 64] [--h2 32]
"""
import argparse, struct, sys, time
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    sys.exit("torch yok. Kur: pip install torch --index-url https://download.pytorch.org/whl/cu121")


def veri_oku(yol):
    """QDT2/QDT3 dosyasini okur.

    QDT3'te ek olarak politika hedefi var: aramanin o pozisyonda hangi
    hamleye kac kez baktigi. Motorun "hangi hamlelere bakmaya deger"
    sezgisi su an elle yazilmis tahminlerden ibaret; bunu ogretebilmek
    icin.

    Sabit genislikte oldugu icin tek `fromfile` yetiyor: 4 milyon satir
    saniyeler icinde geliyor.
    """
    with open(yol, "rb") as f:
        bas = f.read(24)
    sihir = bas[:4]
    if sihir not in (b"QDT2", b"QDT3"):
        sys.exit(f"{yol}: QDT2/QDT3 baslikli degil (kodla.exe surumu eski olabilir)")
    if sihir == b"QDT2":
        n, n_sparse, n_dense, K = struct.unpack("<IIII", bas[4:20])
        P, ofs = 0, 20
        dt = np.dtype([("y", "<f4"), ("k", "<u2"), ("idx", "<u2", K), ("d", "<f4", n_dense)])
    else:
        n, n_sparse, n_dense, K, P = struct.unpack("<IIIII", bas[4:24])
        ofs = 24
        dt = np.dtype([("y", "<f4"), ("k", "<u2"), ("idx", "<u2", K), ("d", "<f4", n_dense),
                       ("pa", "<u2", P), ("pp", "<f4", P)])
    ham = np.fromfile(yol, dtype=dt, offset=ofs)
    if len(ham) < n:
        print(f"  uyari: baslik {n:,} diyor, dosyada {len(ham):,} var (kesilmis) - olan kullanilacak")
        n = len(ham)
    ham = ham[:n]
    print(f"  {n:,} satir, seyrek {n_sparse}, yogun {n_dense}, K={K}" + (f", politika P={P}" if P else " (politika yok)"))
    pol = (ham["pa"].astype(np.int64), ham["pp"].astype(np.float32)) if P else None
    return ham["idx"].astype(np.int64), ham["d"].astype(np.float32), ham["y"].astype(np.float32), n_sparse, n_dense, pol


NUM_ACTIONS = 140


class Ag(nn.Module):
    """Iki basli ag: deger ("bu tahta iyi mi") ve politika ("hangi hamleye
    bakmaya deger"). Ilk katmani paylasiyorlar - hem daha ucuz hem de iki
    gorev birbirinin ogrenmesine yardim ediyor."""

    def __init__(self, n_in, h1, h2, politika=True):
        super().__init__()
        self.l1 = nn.Linear(n_in, h1)
        self.l2 = nn.Linear(h1, h2)
        self.l3 = nn.Linear(h2, 1)
        self.pol = nn.Linear(h1, NUM_ACTIONS) if politika else None

    def forward(self, x):
        g = torch.relu(self.l1(x))
        v = torch.relu(self.l2(g))
        v = self.l3(v).squeeze(-1)
        p = self.pol(g) if self.pol is not None else None
        return v, p


def yaz(model, yol, n_in, h1, h2):
    """Rust tarafının beklediği düzen: W1 [n_in][h1] satır büyük."""
    with open(yol, "wb") as f:
        f.write(b"QNN1")
        f.write(struct.pack("<III", n_in, h1, h2))
        parcalar = [
            model.l1.weight.detach().cpu().numpy().T,  # [n_in][h1]
            model.l1.bias.detach().cpu().numpy(),
            model.l2.weight.detach().cpu().numpy().T,  # [h1][h2]
            model.l2.bias.detach().cpu().numpy(),
            model.l3.weight.detach().cpu().numpy().reshape(-1),
            model.l3.bias.detach().cpu().numpy(),
        ]
        # Politika basi dosyanin sonunda, istege bagli: yalnizca deger basi
        # olan eski dosyalar da okunabiliyor.
        if model.pol is not None:
            parcalar.append(model.pol.weight.detach().cpu().numpy().T)  # [h1][140]
            parcalar.append(model.pol.bias.detach().cpu().numpy())
        for t in parcalar:
            f.write(np.ascontiguousarray(t, dtype="<f4").tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("girdi"); ap.add_argument("cikti")
    ap.add_argument("--epoch", type=int, default=12)
    ap.add_argument("--h1", type=int, default=64)
    ap.add_argument("--h2", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=3e-3)
    # Bellek tavani: 6M satir ~1.2 GB indeks + yogun. Gece kosusu bundan
    # fazlasini uretebiliyor; sessizce kesmek yerine soyleyip kesiyoruz.
    ap.add_argument("--maxsatir", type=int, default=6_000_000)
    ap.add_argument("--politikasiz", action="store_true", help="yalnizca deger basi")
    ap.add_argument("--polagirlik", type=float, default=1.0)
    a = ap.parse_args()

    print("Veri okunuyor...")
    idx, dense, y, n_sparse, n_dense, pol = veri_oku(a.girdi)
    n_in = n_sparse + n_dense
    n = len(y)
    if n > a.maxsatir:
        print(f"  {n:,} satirin son {a.maxsatir:,} tanesi kullanilacak (bellek tavani)")
        idx, dense, y = idx[-a.maxsatir:], dense[-a.maxsatir:], y[-a.maxsatir:]
        if pol is not None:
            pol = (pol[0][-a.maxsatir:], pol[1][-a.maxsatir:])
        n = len(y)
    if n < 5000:
        sys.exit(f"cok az veri ({n}) - egitim anlamsiz")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        yet = torch.cuda.get_device_capability(0)
        print(f"cihaz: cuda — {torch.cuda.get_device_name(0)} (sm_{yet[0]}{yet[1]}), torch {torch.__version__}")
    else:
        # 50 serisi (Blackwell, sm_120) eski CUDA derlemelerinde gorunmez.
        # Bu ag CPU'da da makul surede egitiliyor, o yuzden durmuyoruz.
        print(f"cihaz: cpu — torch {torch.__version__} kartı görmedi. "
              "CPU'da da eğitilir, sadece daha yavaş.")

    # Dolgu (0xFFFF) fazladan bir sutuna gidiyor, sonra o sutun atiliyor.
    # Boylece scatter tek satirda oluyor, dallanma yok.
    idx = np.where(idx == 0xFFFF, n_in, idx)

    # Son %5 dogrulama. Karistirmadan **ayiriyoruz**: veri oyun sirasiyla
    # geliyor; karistirip ayirmak ayni oyunun pozisyonlarini iki tarafa
    # dagitir ve dogrulama skoru olduğundan iyi cikar.
    kes = int(n * 0.95)
    perm = np.random.default_rng(7).permutation(kes)

    politika_var = pol is not None and not a.politikasiz
    model = Ag(n_in, a.h1, a.h2, politika=politika_var).to(dev)
    if politika_var:
        pa_t = torch.from_numpy(np.where(pol[0] == 0xFFFF, NUM_ACTIONS, pol[0])).to(dev)
        pp_t = torch.from_numpy(pol[1]).to(dev)
        print("  politika basi da egitiliyor")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=max(1, a.epoch * (kes // a.batch + 1)))
    kayip_fn = nn.BCEWithLogitsLoss()

    idx_t = torch.from_numpy(idx).to(dev)
    dense_t = torch.from_numpy(dense).to(dev)
    y_t = torch.from_numpy(y).to(dev)

    def batch_yap(satirlar):
        s_t = torch.from_numpy(satirlar).to(dev)
        b = len(satirlar)
        x = torch.zeros((b, n_in + 1), device=dev)
        x.scatter_(1, idx_t[s_t], 1.0)
        x = x[:, :n_in]
        x[:, n_sparse:] = dense_t[s_t]
        return x, y_t[s_t]

    # EN IYI epoch'u sakla, sonuncuyu degil.
    #
    # Ilk gece kosusunda 256x32 agin dogrulama kaybi 5. epoch'ta 0.3982'ye
    # inip sonra 9 epoch boyunca 0.4413'e TIRMANDI - klasik ezberleme. Biz
    # 14. epoch'u yazdik, yani en kotusunu. Olcumde o ag -10 Elo verdi,
    # ezberlemeyen kucuk ag +129. Ag boyutu degil, bu hata belirleyiciydi.
    en_iyi = float("inf")
    en_iyi_ep = 0
    en_iyi_durum = None

    t0 = time.time()
    for ep in range(a.epoch):
        model.train()
        np.random.default_rng(100 + ep).shuffle(perm)
        toplam, adet = 0.0, 0
        for s in range(0, kes, a.batch):
            satirlar = perm[s:s + a.batch]
            x, t = batch_yap(satirlar)
            opt.zero_grad(set_to_none=True)
            v, p = model(x)
            kayip = kayip_fn(v, t)
            if politika_var:
                s_t = torch.from_numpy(satirlar).to(dev)
                # Hedef dagilim: aramanin ziyaret oranlari. Dolgu sutununu
                # (NUM_ACTIONS) ekleyip sonra atiyoruz.
                hedef = torch.zeros((len(satirlar), NUM_ACTIONS + 1), device=dev)
                hedef.scatter_(1, pa_t[s_t], pp_t[s_t])
                hedef = hedef[:, :NUM_ACTIONS]
                topl = hedef.sum(1, keepdim=True)
                gecerli = (topl.squeeze(1) > 1e-6)
                if gecerli.any():
                    hedef = hedef / topl.clamp(min=1e-6)
                    logp = torch.log_softmax(p, dim=1)
                    pk = -(hedef * logp).sum(1)
                    kayip = kayip + a.polagirlik * pk[gecerli].mean()
            kayip.backward(); opt.step()
            try: sched.step()
            except Exception: pass
            toplam += kayip.item() * len(satirlar); adet += len(satirlar)

        model.eval()
        with torch.no_grad():
            dg_satir = np.arange(kes, n)
            dogru, top, dg_kayip = 0, 0, 0.0
            for s in range(0, len(dg_satir), a.batch):
                satirlar = dg_satir[s:s + a.batch]
                x, t = batch_yap(satirlar)
                z, _ = model(x)
                dg_kayip += kayip_fn(z, t).item() * len(satirlar)
                dogru += ((z > 0).float() == t).sum().item(); top += len(satirlar)
        dg = dg_kayip / max(top, 1)
        yildiz = ""
        if dg < en_iyi - 1e-5:
            en_iyi, en_iyi_ep = dg, ep + 1
            en_iyi_durum = {k: v.detach().clone() for k, v in model.state_dict().items()}
            yildiz = "  <- en iyi"
        print(f"  epoch {ep+1:>2}/{a.epoch}  egitim {toplam/max(adet,1):.4f}  "
              f"dogrulama {dg:.4f}  isabet {100*dogru/max(top,1):.1f}%  "
              f"({time.time()-t0:.0f} sn){yildiz}")
        # Erken durdurma: 4 epoch boyunca iyilesme yoksa ezberlemeye
        # basladi demektir, devam etmek modeli bozuyor.
        if ep + 1 - en_iyi_ep >= 4:
            print(f"  4 epoch iyilesme yok - duruldu ({ep+1}. epoch)")
            break

    if en_iyi_durum is not None:
        model.load_state_dict(en_iyi_durum)
    print(f"secilen: {en_iyi_ep}. epoch (dogrulama {en_iyi:.4f})")
    yaz(model, a.cikti, n_in, a.h1, a.h2)
    print(f"yazildi: {a.cikti}")


if __name__ == "__main__":
    main()
