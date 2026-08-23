"""
Quoridor AI - Sürekli Otomatik Pilot Eğitim Döngüsü (Continuous RL Loop)
========================================================================
Bu script:
1. Sessiz modda (9 CPU iş parçacığı ve düşük işlemci önceliğiyle, sıfır ısınma/fan gürültüsü)
   belirlenen miktarda (varsayılan 5.000) maç üretir.
2. Üretilen veriyi KataGo QDT5 ikili formatına dönüştürür.
3. 8 Milyonluk Replay Buffer ve RTX 5070 Ti GPU ile yeni nesil modeli eğitir (~30 saniye).
4. Yeni model ile mevcut şampiyon model arasında 10 maçlık turnuva oynatır.
5. Yeni model üstün gelirse şampiyon ilan edilip ag.bin olarak atanır.
6. Döngü bir sonraki nesil için otomatik olarak yeniden başlar!
"""

import os
import sys
import time
import shutil
import glob
import subprocess
import argparse

# Windows düşük işlemci önceliği bayrağı (Bilgisayarı kasmaması için)
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000 if sys.platform == "win32" else 0

def log(mesaj):
    zaman = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{zaman}] {mesaj}", flush=True)

def calistir(komut, aciklama=None):
    if aciklama:
        log(f"▶ {aciklama}")
    log(f"  Komut: {' '.join(komut)}")
    p = subprocess.Popen(komut, creationflags=BELOW_NORMAL_PRIORITY_CLASS)
    rc = p.wait()
    if rc != 0:
        log(f"❌ HATA: Komut başarısız oldu (Çıkış kodu: {rc})")
        return False
    return True

def mevcut_nesli_bul():
    modeller = glob.glob("champion_gen_*.bin")
    if not modeller:
        return 4 # Gen-4 ile başla
    nesiller = []
    for m in modeller:
        try:
            n = int(m.split("_")[-1].replace(".bin", ""))
            nesiller.append(n)
        except Exception:
            pass
    return max(nesiller) + 1 if nesiller else 4

def replay_buffer_listesi():
    temel = ["spatial_egitim.bin", "selfplay_spatial.bin", "selfplay_v3.bin"]
    mevcut = [f for f in temel if os.path.exists(f)]
    # Önceki döngülerin bin dosyalarını da ekle
    donguler = sorted(glob.glob("selfplay_cycle_*.bin"))
    mevcut.extend(donguler)
    return ",".join(mevcut)

def main():
    ap = argparse.ArgumentParser(description="Quoridor AI Auto-Pilot Training Loop")
    ap.add_argument("--oyun", type=int, default=5000, help="Döngü başına üretilecek maç sayısı (varsayılan: 5000)")
    ap.add_argument("--iter", type=int, default=800, help="MCTS iterasyon sayısı (varsayılan: 800)")
    ap.add_argument("--threads", type=int, default=9, help="Sessiz mod CPU thread sayısı (varsayılan: 9)")
    ap.add_argument("--epoch", type=int, default=10, help="GPU eğitim epoch sayısı (varsayılan: 10)")
    ap.add_argument("--batch", type=int, default=4096, help="GPU batch boyutu (varsayılan: 4096)")
    ap.add_argument("--dongu-sayisi", type=int, default=100, help="Kaç döngü çalıştırılacak (varsayılan: 100)")
    args = ap.parse_args()

    print("=" * 65)
    print("🏆 QUORIDOR AI OTO-PİLOT SÜREKLİ EĞİTİM DÖNGÜSÜ")
    print(f"  Döngü Başına Maç:  {args.oyun:,} oyun")
    print(f"  MCTS İterasyonu:   {args.iter} iter / hamle")
    print(f"  Sessiz Mod İşlemci: {args.threads} CPU Thread (Düşük Öncelik)")
    print(f"  GPU Eğitimi:       RTX 5070 Ti ({args.epoch} Epoch, Batch {args.batch})")
    print("=" * 65)

    if not os.path.exists("ag.bin"):
        sys.exit("HATA: Başlangıç modeli 'ag.bin' bulunamadı!")

    nesil = mevcut_nesli_bul()

    for d in range(args.dongu_sayisi):
        print("\n" + "=" * 65)
        log(f"🌟 NESİL {nesil} BAŞLATILIYOR (Döngü {d+1}/{args.dongu_sayisi})")
        print("=" * 65)

        csv_dosya = f"selfplay_cycle_{nesil}.csv"
        bin_dosya = f"selfplay_cycle_{nesil}.bin"
        yeni_model = f"ag_gen{nesil}.bin"

        # 1. AŞAMA: Sessiz Modda Öz-Oyun Veri Üretimi
        t0 = time.time()
        ok = calistir([
            "target/release/selfplay.exe",
            str(args.oyun),
            str(args.iter),
            str(args.threads),
            csv_dosya
        ], f"1. AŞAMA: {args.oyun:,} Maçlık Öz-Oyun Üretimi ({args.threads} Sessiz Thread)")
        if not ok:
            log("❌ Öz-oyun üretimi başarısız oldu, döngü durduruluyor.")
            break
        sure_oyun = (time.time() - t0) / 60.0
        log(f"✅ Öz-Oyun tamamlandı: {csv_dosya} ({sure_oyun:.1f} dakika)")

        # 2. AŞAMA: QDT5 Formatına Kodlama
        ok = calistir([
            "target/release/kodla.exe",
            "--katago",
            csv_dosya,
            bin_dosya
        ], "2. AŞAMA: Veri QDT5 İkili Formatına Kodlanıyor")
        if not ok:
            log("❌ Kodlama başarısız oldu.")
            break

        # 3. AŞAMA: Replay Buffer ile GPU Eğitimi
        replay_str = replay_buffer_listesi()
        log(f"📦 Aktif Replay Buffer Dosyaları: {replay_str}")
        t0_egitim = time.time()
        ok = calistir([
            "python", "tools/egit.py",
            replay_str,
            yeni_model,
            "--channels", "32",
            "--blocks", "4",
            "--epoch", str(args.epoch),
            "--batch", str(args.batch),
            "--lr", "0.001",
            "--yukle", "ag.bin"
        ], f"3. AŞAMA: RTX 5070 Ti ile Gen-{nesil} Modeli Eğitiliyor")
        if not ok:
            log("❌ GPU Eğitimi başarısız oldu.")
            break
        sure_egitim = time.time() - t0_egitim
        log(f"✅ Model eğitimi tamamlandı: {yeni_model} ({sure_egitim:.1f} saniye)")

        # 4. AŞAMA: Şampiyonluk Turnuvası (Yeni Model vs Mevcut Şampiyon)
        log("4. AŞAMA: Şampiyonluk Turnuvası (10 Maç - 5 Açılış x 2 Renk)...")
        turnuva_cmd = [
            "target/release/quoridor.exe",
            "match",
            f"mcts:5000:ag={yeni_model}",
            "mcts:5000:ag=ag.bin",
            "5"
        ]
        p = subprocess.Popen(turnuva_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", creationflags=BELOW_NORMAL_PRIORITY_CLASS)
        cikis, _ = p.communicate()
        print(cikis)

        # Skoru parse et (örn. "A 6 - 4 B")
        kazanan_yeni = False
        a_skor, b_skor = 0, 0
        for satir in cikis.splitlines():
            if satir.startswith("A ") and " - " in satir and " B" in satir:
                try:
                    parca = satir.split()
                    a_skor = int(parca[1])
                    b_skor = int(parca[3])
                    if a_skor >= b_skor:
                        kazanan_yeni = True
                except Exception:
                    pass

        # Turnuva Raporu
        rapor_satir = f"Nesil {nesil} | Skor: Gen-{nesil} ({a_skor}) vs Eski ({b_skor}) | Tarih: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        with open("championship_history.log", "a", encoding="utf-8") as f:
            f.write(rapor_satir)

        if kazanan_yeni:
            log(f"🏆 YENİ ŞAMPİYON! Gen-{nesil} ({a_skor} - {b_skor}) ile üstün geldi.")
            # Mevcut ag.bin'i arşivle
            shutil.copy("ag.bin", f"champion_gen_{nesil-1}.bin")
            # Yeni modeli ag.bin yap
            shutil.copy(yeni_model, "ag.bin")
            log(f"👑 'ag.bin' güncellendi -> Artık Gen-{nesil} Şampiyon!")
        else:
            log(f"⚠️ Gen-{nesil} ({a_skor} - {b_skor}) şampiyonu geçemedi. Eski 'ag.bin' korunuyor, veri havuzda tutuluyor.")

        log(f"🎉 Döngü tamamlandı! Bir sonraki nesle geçiliyor...")
        nesil += 1

if __name__ == "__main__":
    main()
