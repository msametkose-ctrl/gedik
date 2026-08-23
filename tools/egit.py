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


def veri_oku(yol, maxsatir=1_000_000):
    """QDT2/QDT3/QDT4/QDT5 dosyasini memmap ile anında okur."""
    with open(yol, "rb") as f:
        bas = f.read(24)
    sihir = bas[:4]
    if sihir not in (b"QDT2", b"QDT3", b"QDT4", b"QDT5"):
        sys.exit(f"{yol}: QDT2/QDT3/QDT4/QDT5 baslikli degil (kodla.exe surumu eski olabilir)")
    if sihir == b"QDT2":
        n, n_sparse, n_dense, K = struct.unpack("<IIII", bas[4:20])
        P, ofs = 0, 20
        dt = np.dtype([("y", "<f4"), ("k", "<u2"), ("idx", "<u2", K), ("d", "<f4", n_dense)])
        ham = np.memmap(yol, dtype=dt, mode='r', offset=ofs, shape=(n,))
        if n > maxsatir:
            print(f"  {n:,} satirdan son {maxsatir:,} tanesi secildi", flush=True)
            ham = ham[-maxsatir:]
        return "mlp", ham["idx"].astype(np.int64), ham["d"].astype(np.float32), ham["y"].astype(np.float32), n_sparse, n_dense, None
    elif sihir == b"QDT3":
        n, n_sparse, n_dense, K, P = struct.unpack("<IIIII", bas[4:24])
        ofs = 24
        dt = np.dtype([("y", "<f4"), ("k", "<u2"), ("idx", "<u2", K), ("d", "<f4", n_dense),
                       ("pa", "<u2", P), ("pp", "<f4", P)])
        ham = np.memmap(yol, dtype=dt, mode='r', offset=ofs, shape=(n,))
        if n > maxsatir:
            print(f"  {n:,} satirdan son {maxsatir:,} tanesi secildi", flush=True)
            ham = ham[-maxsatir:]
        pol = (ham["pa"].astype(np.int64), ham["pp"].astype(np.float32)) if P else None
        return "mlp", ham["idx"].astype(np.int64), ham["d"].astype(np.float32), ham["y"].astype(np.float32), n_sparse, n_dense, pol
    elif sihir == b"QDT5": # 2D Spatial Planes with KataGo Multi-Target
        n, channels, spatial_size, P = struct.unpack("<IIII", bas[4:20])
        ofs = 20
        dt = np.dtype([("y", "<f4"), ("moves", "<f4"), ("delta", "<f4"), ("planes", "<f4", spatial_size), ("pa", "<u2", P), ("pp", "<f4", P)])
        ham = np.memmap(yol, dtype=dt, mode='r', offset=ofs, shape=(n,))
        offset = 0
        if n > maxsatir:
            offset = n - maxsatir
            n = maxsatir
            print(f"  {len(ham):,} satirdan son {n:,} tanesi secildi", flush=True)
        else:
            print(f"  {n:,} satirin tamami secildi", flush=True)
        print(f"  2B Uzamsal QDT5 (KataGo Çoklu Hedef): {channels} kanal x 9x9, politika P={P}", flush=True)
        return "resnet", ham, offset, n, channels, spatial_size, P > 0, True
    else: # QDT4: 2D Spatial Planes
        n, channels, spatial_size, P = struct.unpack("<IIII", bas[4:20])
        ofs = 20
        dt = np.dtype([("y", "<f4"), ("planes", "<f4", spatial_size), ("pa", "<u2", P), ("pp", "<f4", P)])
        ham = np.memmap(yol, dtype=dt, mode='r', offset=ofs, shape=(n,))
        offset = 0
        if n > maxsatir:
            offset = n - maxsatir
            n = maxsatir
            print(f"  {len(ham):,} satirdan son {n:,} tanesi secildi", flush=True)
        else:
            print(f"  {n:,} satirin tamami secildi", flush=True)
        print(f"  2B Uzamsal QDT4: {channels} kanal x 9x9" + (f", politika P={P}" if P else ""), flush=True)
        return "resnet", ham, offset, n, channels, spatial_size, P > 0, False


NUM_ACTIONS = 140


class Ag(nn.Module):
    """MLP Değerlendirme Ağı."""

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


class ResBlockPyTorch(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.relu(out + res)
        return out


class AgResNet(nn.Module):
    """2B Uzamsal ResNet Modeli (KataGo Çoklu Hedef Destekli)."""

    def __init__(self, in_channels=13, channels=32, n_blocks=2, politika=True, katago=True):
        super().__init__()
        self.channels = channels
        self.n_blocks = n_blocks
        self.katago = katago
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.blocks = nn.ModuleList([ResBlockPyTorch(channels) for _ in range(n_blocks)])
        
        # Value head: 1x1 conv (channels -> 2) -> Linear(162 -> 32) -> Linear(32 -> 1)
        self.val_conv = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.val_fc1 = nn.Sequential(
            nn.Linear(2 * 9 * 9, 32),
            nn.ReLU(inplace=True)
        )
        self.val_fc2 = nn.Linear(32, 1)

        # Policy head: 1x1 conv (channels -> 4) -> Linear(324 -> 140)
        if politika:
            self.pol_conv = nn.Sequential(
                nn.Conv2d(channels, 4, kernel_size=1),
                nn.ReLU(inplace=True)
            )
            self.pol_fc = nn.Linear(4 * 9 * 9, NUM_ACTIONS)
        else:
            self.pol_conv = None
            self.pol_fc = None

        # KataGo Multi-Target Auxiliary Heads: Moves-to-end & Final Distance Delta
        if katago:
            self.moves_conv = nn.Sequential(
                nn.Conv2d(channels, 1, kernel_size=1),
                nn.ReLU(inplace=True)
            )
            self.moves_fc1 = nn.Sequential(
                nn.Linear(1 * 9 * 9, 32),
                nn.ReLU(inplace=True)
            )
            self.moves_fc2 = nn.Linear(32, 1)

            self.delta_conv = nn.Sequential(
                nn.Conv2d(channels, 1, kernel_size=1),
                nn.ReLU(inplace=True)
            )
            self.delta_fc1 = nn.Sequential(
                nn.Linear(1 * 9 * 9, 32),
                nn.ReLU(inplace=True)
            )
            self.delta_fc2 = nn.Linear(32, 1)
        else:
            self.moves_conv, self.moves_fc1, self.moves_fc2 = None, None, None
            self.delta_conv, self.delta_fc1, self.delta_fc2 = None, None, None

    def forward(self, x):
        feat = self.init_conv(x)
        for block in self.blocks:
            feat = block(feat)
        
        v = self.val_conv(feat).flatten(1)
        v = self.val_fc1(v)
        v = self.val_fc2(v).squeeze(-1)

        p = self.pol_fc(self.pol_conv(feat).flatten(1)) if self.pol_conv is not None else None
        m = self.moves_fc2(self.moves_fc1(self.moves_conv(feat).flatten(1))).squeeze(-1) if self.moves_conv is not None else None
        d = self.delta_fc2(self.delta_fc1(self.delta_conv(feat).flatten(1))).squeeze(-1) if self.delta_conv is not None else None

        return v, p, m, d


def yaz(model, yol, n_in, h1, h2):
    """MLP modelini QNN1 formatında kaydeder."""
    with open(yol, "wb") as f:
        f.write(b"QNN1")
        f.write(struct.pack("<III", n_in, h1, h2))
        parcalar = [
            model.l1.weight.detach().cpu().numpy().T,
            model.l1.bias.detach().cpu().numpy(),
            model.l2.weight.detach().cpu().numpy().T,
            model.l2.bias.detach().cpu().numpy(),
            model.l3.weight.detach().cpu().numpy().reshape(-1),
            model.l3.bias.detach().cpu().numpy(),
        ]
        if model.pol is not None:
            parcalar.append(model.pol.weight.detach().cpu().numpy().T)
            parcalar.append(model.pol.bias.detach().cpu().numpy())
        for t in parcalar:
            f.write(np.ascontiguousarray(t, dtype="<f4").tobytes())


def yaz_resnet(model, yol):
    """ResNet modelini QNN3 (KataGo) veya QNN2 formatında kaydeder."""
    with open(yol, "wb") as f:
        magic = b"QNN3" if model.katago else b"QNN2"
        f.write(magic)
        has_pol = 1 if model.pol_conv is not None else 0
        f.write(struct.pack("<III", model.channels, model.n_blocks, has_pol))
        
        parcalar = [
            model.init_conv[0].weight.detach().cpu().numpy(),
            model.init_conv[0].bias.detach().cpu().numpy(),
        ]
        for b in model.blocks:
            parcalar.append(b.conv1.weight.detach().cpu().numpy())
            parcalar.append(b.conv1.bias.detach().cpu().numpy())
            parcalar.append(b.conv2.weight.detach().cpu().numpy())
            parcalar.append(b.conv2.bias.detach().cpu().numpy())
        
        parcalar.extend([
            model.val_conv[0].weight.detach().cpu().numpy().reshape(2, model.channels),
            model.val_conv[0].bias.detach().cpu().numpy(),
            model.val_fc1[0].weight.detach().cpu().numpy(),
            model.val_fc1[0].bias.detach().cpu().numpy(),
            model.val_fc2.weight.detach().cpu().numpy(),
            model.val_fc2.bias.detach().cpu().numpy(),
        ])
        
        if has_pol:
            parcalar.extend([
                model.pol_conv[0].weight.detach().cpu().numpy().reshape(4, model.channels),
                model.pol_conv[0].bias.detach().cpu().numpy(),
                model.pol_fc.weight.detach().cpu().numpy(),
                model.pol_fc.bias.detach().cpu().numpy(),
            ])
        
        if model.katago:
            parcalar.extend([
                model.moves_conv[0].weight.detach().cpu().numpy().reshape(1, model.channels),
                model.moves_conv[0].bias.detach().cpu().numpy(),
                model.moves_fc1[0].weight.detach().cpu().numpy(),
                model.moves_fc1[0].bias.detach().cpu().numpy(),
                model.moves_fc2.weight.detach().cpu().numpy(),
                model.moves_fc2.bias.detach().cpu().numpy(),
                model.delta_conv[0].weight.detach().cpu().numpy().reshape(1, model.channels),
                model.delta_conv[0].bias.detach().cpu().numpy(),
                model.delta_fc1[0].weight.detach().cpu().numpy(),
                model.delta_fc1[0].bias.detach().cpu().numpy(),
                model.delta_fc2.weight.detach().cpu().numpy(),
                model.delta_fc2.bias.detach().cpu().numpy(),
            ])

        for t in parcalar:
            f.write(np.ascontiguousarray(t, dtype="<f4").tobytes())


def model_yukle_resnet(model, yol):
    """QNN2 veya QNN3 dosyasından ağırlıkları okuyup modele yükler."""
    with open(yol, "rb") as f:
        sihir = f.read(4)
        if sihir not in (b"QNN2", b"QNN3"):
            print(f"Uyarı: {yol} QNN2/QNN3 formatında değil, sıfırdan başlanıyor.")
            return
        c, n_blocks, has_pol = struct.unpack("<III", f.read(12))
        if c != model.channels or n_blocks != model.n_blocks:
            print(f"Uyarı: Boyut uyumsuz ({c}k, {n_blocks}b != {model.channels}k, {model.n_blocks}b), sıfırdan başlanıyor.")
            return

        def oku_tensor(shape):
            eleman = 1
            for s in shape: eleman *= s
            buf = f.read(4 * eleman)
            arr = np.frombuffer(buf, dtype="<f4").reshape(shape)
            return torch.from_numpy(arr.copy())

        model.init_conv[0].weight.data.copy_(oku_tensor(model.init_conv[0].weight.shape))
        model.init_conv[0].bias.data.copy_(oku_tensor(model.init_conv[0].bias.shape))

        for b in model.blocks:
            b.conv1.weight.data.copy_(oku_tensor(b.conv1.weight.shape))
            b.conv1.bias.data.copy_(oku_tensor(b.conv1.bias.shape))
            b.conv2.weight.data.copy_(oku_tensor(b.conv2.weight.shape))
            b.conv2.bias.data.copy_(oku_tensor(b.conv2.bias.shape))

        model.val_conv[0].weight.data.copy_(oku_tensor((2, model.channels)).reshape(2, model.channels, 1, 1))
        model.val_conv[0].bias.data.copy_(oku_tensor((2,)))
        model.val_fc1[0].weight.data.copy_(oku_tensor(model.val_fc1[0].weight.shape))
        model.val_fc1[0].bias.data.copy_(oku_tensor(model.val_fc1[0].bias.shape))
        model.val_fc2.weight.data.copy_(oku_tensor(model.val_fc2.weight.shape))
        model.val_fc2.bias.data.copy_(oku_tensor(model.val_fc2.bias.shape))

        if has_pol and model.pol_conv is not None:
            model.pol_conv[0].weight.data.copy_(oku_tensor((4, model.channels)).reshape(4, model.channels, 1, 1))
            model.pol_conv[0].bias.data.copy_(oku_tensor((4,)))
            model.pol_fc.weight.data.copy_(oku_tensor(model.pol_fc.weight.shape))
            model.pol_fc.bias.data.copy_(oku_tensor(model.pol_fc.bias.shape))

        print(f"  {yol} model ağırlıkları başarıyla yüklendi (Warm-Start Fine-Tuning).", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("girdi"); ap.add_argument("cikti")
    ap.add_argument("--epoch", type=int, default=10)
    ap.add_argument("--h1", type=int, default=64)
    ap.add_argument("--h2", type=int, default=32)
    ap.add_argument("--channels", type=int, default=32, help="ResNet kanal sayisi")
    ap.add_argument("--blocks", type=int, default=2, help="ResBlock sayisi")
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--maxsatir", type=int, default=1_000_000)
    ap.add_argument("--politikasiz", action="store_true", help="yalnizca deger basi")
    ap.add_argument("--polagirlik", type=float, default=1.0)
    ap.add_argument("--yukle", type=str, default="", help="Mevcut model agirliklarini baslangic olarak yukle")
    a = ap.parse_args()

    girdiler = [g.strip() for g in a.girdi.split(",") if g.strip()]
    if len(girdiler) == 1:
        res = veri_oku(girdiler[0], a.maxsatir)
        mtype = res[0]
    else:
        mtype = "resnet"

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        yet = torch.cuda.get_device_capability(0)
        print(f"cihaz: cuda — {torch.cuda.get_device_name(0)} (sm_{yet[0]}{yet[1]}), torch {torch.__version__}", flush=True)
    else:
        print(f"cihaz: cpu — torch {torch.__version__}", flush=True)

    if mtype == "resnet":
        datasets = []
        total_samples = 0
        any_katago = False
        channels = 13
        for g in girdiler:
            r = veri_oku(g, a.maxsatir)
            _, ham, offset, n, ch, spatial_size, has_p, is_katago = r
            channels = ch
            datasets.append({
                "ham": ham, "offset": offset, "n": n, "has_p": has_p, "is_katago": is_katago
            })
            total_samples += n
            if is_katago: any_katago = True
            print(f"  [Replay Buffer] {g}: {n:,} pozisyon (katago={is_katago})", flush=True)

        politika_var = any(ds["has_p"] for ds in datasets) and not a.politikasiz
        print(f"Model: 2B ResNet ({channels} kanal, {a.channels} filtre, {a.blocks} ResBlock, katago={any_katago})", flush=True)
        model = AgResNet(in_channels=channels, channels=a.channels, n_blocks=a.blocks, politika=politika_var, katago=any_katago).to(dev)

        if a.yukle:
            model_yukle_resnet(model, a.yukle)

        # Doğrulama kümesi: her veri setinin sonundan dengeli örneklem
        val_list_x, val_list_y = [], []
        for ds in datasets:
            n_ds = ds["n"]
            val_count = min(15000, max(500, int(n_ds * 0.05)))
            raw_val = ds["ham"][ds["offset"] + n_ds - val_count : ds["offset"] + n_ds]
            val_list_x.append(np.ascontiguousarray(raw_val["planes"]).reshape(-1, channels, 9, 9))
            val_list_y.append(np.ascontiguousarray(raw_val["y"]))
        val_x = torch.from_numpy(np.concatenate(val_list_x, axis=0)).to(dev)
        val_y = torch.from_numpy(np.concatenate(val_list_y, axis=0)).to(dev)
        val_n = len(val_y)
        print(f"  Doğrulama Kümesi: {val_n:,} pozisyon GPU'da hazır.", flush=True)

        loaded_ds = []
        for ds in datasets:
            n_ds = ds["n"]
            train_n = int(n_ds * 0.95)
            raw = ds["ham"][ds["offset"] : ds["offset"] + train_n]
            print(f"  RAM'e alınıyor: {train_n:,} satır...", flush=True)
            d_planes = torch.from_numpy(np.ascontiguousarray(raw["planes"])).reshape(-1, channels, 9, 9)
            d_y = torch.from_numpy(np.ascontiguousarray(raw["y"]))
            d_pa = torch.from_numpy(np.ascontiguousarray(raw["pa"])).long() if ds["has_p"] else None
            d_pp = torch.from_numpy(np.ascontiguousarray(raw["pp"])) if ds["has_p"] else None
            d_moves = torch.from_numpy(np.ascontiguousarray(raw["moves"])) if ds["is_katago"] else None
            d_delta = torch.from_numpy(np.ascontiguousarray(raw["delta"])) if ds["is_katago"] else None
            loaded_ds.append({
                "n": train_n, "planes": d_planes, "y": d_y,
                "pa": d_pa, "pp": d_pp, "moves": d_moves, "delta": d_delta,
                "is_katago": ds["is_katago"], "has_p": ds["has_p"]
            })

        batch_size = a.batch
        steps_per_epoch = min(1000, max(100, total_samples // batch_size))
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=a.lr, total_steps=max(1, a.epoch * steps_per_epoch))
        kayip_fn = nn.BCEWithLogitsLoss()

        en_iyi = float("inf")
        en_iyi_ep = 0
        en_iyi_durum = None

        t0 = time.time()
        for ep in range(a.epoch):
            model.train()
            toplam, adet = 0.0, 0
            ep_t0 = time.time()

            for step in range(steps_per_epoch):
                bx_list, by_list = [], []
                bpa_list, bpp_list = [], []
                bm_list, bd_list, bkat_mask = [], [], []

                for ds_idx, lds in enumerate(loaded_ds):
                    b_sub = batch_size // len(loaded_ds)
                    idx_sub = torch.randint(0, lds["n"], (b_sub,))

                    bx_list.append(lds["planes"][idx_sub])
                    by_list.append(lds["y"][idx_sub])

                    if lds["has_p"]:
                        bpa_list.append(lds["pa"][idx_sub])
                        bpp_list.append(lds["pp"][idx_sub])
                    else:
                        bpa_list.append(torch.full((b_sub, 24), 0xFFFF, dtype=torch.long))
                        bpp_list.append(torch.zeros((b_sub, 24), dtype=torch.float32))

                    if lds["is_katago"]:
                        bm_list.append(lds["moves"][idx_sub])
                        bd_list.append(lds["delta"][idx_sub])
                        bkat_mask.append(torch.ones(b_sub, dtype=torch.bool))
                    else:
                        bm_list.append(torch.zeros(b_sub, dtype=torch.float32))
                        bd_list.append(torch.zeros(b_sub, dtype=torch.float32))
                        bkat_mask.append(torch.zeros(b_sub, dtype=torch.bool))

                bx = torch.cat(bx_list, dim=0).to(dev, non_blocking=True)
                by = torch.cat(by_list, dim=0).to(dev, non_blocking=True)
                bpa = torch.cat(bpa_list, dim=0).to(dev, non_blocking=True)
                bpp = torch.cat(bpp_list, dim=0).to(dev, non_blocking=True)
                bm = torch.cat(bm_list, dim=0).to(dev, non_blocking=True)
                bd = torch.cat(bd_list, dim=0).to(dev, non_blocking=True)
                bmask = torch.cat(bkat_mask, dim=0).to(dev, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                v, p, m, d = model(bx)
                kayip = kayip_fn(v, by)

                if any_katago and m is not None and d is not None and bmask.any():
                    loss_m = torch.nn.functional.smooth_l1_loss(m[bmask], bm[bmask])
                    loss_d = torch.nn.functional.smooth_l1_loss(d[bmask], bd[bmask])
                    kayip = kayip + 0.15 * loss_m + 0.15 * loss_d

                if politika_var and p is not None:
                    bpa = torch.where(bpa == 0xFFFF, NUM_ACTIONS, bpa)
                    hedef = torch.zeros((len(bx), NUM_ACTIONS + 1), device=dev)
                    hedef.scatter_(1, bpa, bpp)
                    hedef = hedef[:, :NUM_ACTIONS]
                    topl = hedef.sum(1, keepdim=True)
                    gecerli = (topl.squeeze(1) > 1e-6)
                    if gecerli.any():
                        hedef = hedef / topl.clamp(min=1e-6)
                        logp = torch.log_softmax(p, dim=1)
                        pk = -(hedef * logp).sum(1)
                        kayip = kayip + a.polagirlik * pk[gecerli].mean()

                kayip.backward()
                opt.step()
                try: sched.step()
                except Exception: pass
                toplam += kayip.item() * len(bx)
                adet += len(bx)

            model.eval()
            with torch.no_grad():
                dogru, top, dg_kayip = 0, 0, 0.0
                for s in range(0, val_n, batch_size * 2):
                    bx = val_x[s : s + batch_size * 2]
                    by = val_y[s : s + batch_size * 2]
                    z, _, _, _ = model(bx)
                    dg_kayip += kayip_fn(z, by).item() * len(bx)
                    dogru += ((z > 0).float() == by).sum().item()
                    top += len(bx)
            dg = dg_kayip / max(top, 1)
            yildiz = ""
            if dg < en_iyi - 1e-5:
                en_iyi, en_iyi_ep = dg, ep + 1
                en_iyi_durum = {k: v.detach().clone() for k, v in model.state_dict().items()}
                yildiz = "  <- en iyi"
            print(f"  epoch {ep+1:>2}/{a.epoch}  egitim {toplam/max(adet,1):.4f}  "
                  f"dogrulama {dg:.4f}  isabet {100*dogru/max(top,1):.1f}%  "
                  f"({time.time()-ep_t0:.1f} sn / ep){yildiz}", flush=True)
            if ep + 1 - en_iyi_ep >= 5:
                print(f"  5 epoch iyilesme yok - duruldu ({ep+1}. epoch)", flush=True)
                break

        if en_iyi_durum is not None:
            model.load_state_dict(en_iyi_durum)
        print(f"Toplam süre: {time.time()-t0:.1f} sn | Seçilen: {en_iyi_ep}. epoch (dogrulama {en_iyi:.4f})", flush=True)
        yaz_resnet(model, a.cikti)
        print(f"Model dosyası yazıldı: {a.cikti}", flush=True)
        return
    else:
        _, data_x, dense, y, n_sparse, n_dense, pol = res
        n = len(y)
        politika_var = pol is not None and not a.politikasiz
        n_in = n_sparse + n_dense
        print(f"Model: MLP ({n_in} -> {a.h1} -> {a.h2} -> 1)", flush=True)
        data_x = np.where(data_x == 0xFFFF, n_in, data_x)
        model = Ag(n_in, a.h1, a.h2, politika=politika_var).to(dev)
        idx_t = torch.from_numpy(data_x).to(dev)
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

        if politika_var:
            pa_t = torch.from_numpy(np.where(pol[0] == 0xFFFF, NUM_ACTIONS, pol[0])).to(dev)
            pp_t = torch.from_numpy(pol[1]).to(dev)
            print("  politika basi da egitiliyor", flush=True)
            def batch_pol(satirlar):
                s_t = torch.from_numpy(satirlar).to(dev)
                hedef = torch.zeros((len(satirlar), NUM_ACTIONS + 1), device=dev)
                hedef.scatter_(1, pa_t[s_t], pp_t[s_t])
                return hedef[:, :NUM_ACTIONS]
        else:
            batch_pol = None

        kes = int(n * 0.95)
        perm = np.random.default_rng(7).permutation(kes)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=a.lr, total_steps=max(1, a.epoch * (kes // a.batch + 1)))
        kayip_fn = nn.BCEWithLogitsLoss()

        en_iyi = float("inf")
        en_iyi_ep = 0
        en_iyi_durum = None

        t0 = time.time()
        for ep in range(a.epoch):
            model.train()
            np.random.default_rng(100 + ep).shuffle(perm)
            toplam, adet = 0.0, 0
            ep_t0 = time.time()
            for s in range(0, kes, a.batch):
                satirlar = perm[s:s + a.batch]
                x, t = batch_yap(satirlar)
                opt.zero_grad(set_to_none=True)
                v, p = model(x)
                kayip = kayip_fn(v, t)
                if politika_var:
                    hedef = batch_pol(satirlar)
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
                for s in range(0, len(dg_satir), a.batch * 2):
                    satirlar = dg_satir[s:s + a.batch * 2]
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
                  f"({time.time()-ep_t0:.1f} sn / ep){yildiz}", flush=True)
            if ep + 1 - en_iyi_ep >= 4:
                print(f"  4 epoch iyilesme yok - duruldu ({ep+1}. epoch)", flush=True)
                break

        if en_iyi_durum is not None:
            model.load_state_dict(en_iyi_durum)
        print(f"Toplam süre: {time.time()-t0:.1f} sn | Seçilen: {en_iyi_ep}. epoch (dogrulama {en_iyi:.4f})", flush=True)
        yaz(model, a.cikti, n_in, a.h1, a.h2)
        print(f"Model dosyası yazıldı: {a.cikti}", flush=True)


if __name__ == "__main__":
    main()
