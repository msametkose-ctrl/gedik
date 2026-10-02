# Devir notu: Gedik'i en güçlü hale getirme planı

Bu dosya, çalışmayı bulut oturumundan kullanıcının kendi laptopuna taşımak
için yazıldı. Yeni oturum buradan başlasın.

## Proje kuralı (pazarlıksız)

**Ölç, yoksa girmez.** Oyun gücünü etkileyen her değişiklik maç sonucuyla
kabul ya da ret edilir (bkz. `CONTRIBUTING.md`). Az oyun yalan söyler:
24 oyun ±20% demek. Wilson aralığı %50'yi dışlamadan "daha iyi" denmez.

## Kullanıcının hedef senaryosu

Kullanıcı motoru **hamle başına 2-3 saniye, tüm çekirdeklerle** oynatıyor
(web arayüzü, `gedik serve`). İterasyon başına güç önemsiz; **gerçek süredeki
güç** önemli. Her ölçüm bu senaryoya göre yapılmalı (kısa denemeler için
400 ms, son karar için `mcts:2500ms:t=0`).

## Donanım (laptop)

- Ryzen 9 9955HX3D, 16 çekirdek / 32 thread, 32 GB RAM
- RTX 5070 Ti Laptop. PyTorch için **cu128** gerekir (RTX 50 serisi):
  `pip install torch --index-url https://download.pytorch.org/whl/cu128`
- Windows. Uzun işlerde şarja takılı ve performans modunda tutulmalı.
  Isınma ölçümleri bozar.

## Şu anki durum (`claude/dreamy-fermat-lz00su` dalı)

| Değişiklik | Ölçüm |
|---|---|
| Varsayılan ağ yeniden MLP (`ag.bin`). ResNet artık `ag_resnet.bin` | ResNet gerçek sürede kaybediyor: 400 ms'de 16-32, 2,5 sn'de 4-9 |
| Hamleler arası ağaç yeniden kullanımı (`ru=1`, varsayılan açık; web sunucusu motoru önbellekte tutuyor) | 400 ms'de 29-18 |
| Damıtma: ResNet öğretmen → MLP öğrenci (329→128→32→1). Yeni `ag.bin` bu öğrenci, eski MLP `ag_mlp.bin` | Eski MLP'ye karşı 400 ms'de 55-36 (%60, ≈+74 Elo) |
| `gedik.bat`: dalı `%USERPROFILE%\gedik-guncel` klasörüne çekip derler ve web arayüzünü açar | — |
| Final kontrol: yeni varsayılan, eski varsayılana (ResNet, ağaç yeniden kullanımı kapalı) karşı, 2,5 sn, tüm çekirdekler | Bulut 10-6, laptop 28-21 (1 bitmedi). **Toplam 38-27 (%58,5, ≈+59 Elo)**, Wilson %46-70. Yön tutarlı, ama aralık hâlâ %50'yi kesiyor; kullanıcı isteğiyle 65 oyunda durduruldu |

Önemli bulgular:
- ResNet (QNN3, 32 kanal × 4 blok) düğüm başına çok daha akıllı. Eşit
  iterasyonda (3000) MLP'yi 34-14 yeniyor. Ama değerlendirmesi ~430 µs,
  MLP'ninki ~2 µs. Darboğaz hız.
- Gumbel kök araması şu haliyle PUCT'a 5-75 kaybediyor. `src/mcts.rs`
  içindeki `iterate_forced` fonksiyonunda `side = pos.side` hatası var;
  bu düzeltilince 15-65 oluyor. Düzeltme repoda değil.
- Guard açık/kapalı farksız (40-38). ResNet politikası pol=2, pol=0'ı
  29-19 yeniyor.
- 6M düğüm sınırına ~190 bin iterasyonda ulaşılıyor.

## Laptop oturumu (2 Ekim 2026)

| Değişiklik | Ölçüm | Commit |
|---|---|---|
| `tools/sprt.py`: paralel parçalı SPRT (elo0=0, elo1=+20), açılış çifti varyansı + 4 sözde çift, kesilen maçı dosyalardan sürdürme | — | `6b31f69`, `cb79a71`, `6832ad7` |
| Damıtılmış `ag.bin`, eski MLP'ye karşı, 400 ms | **70-32 (%68,6)**, Wilson %59-77, H1 kabul | — |
| Gumbel: `iterate_forced` kopyası kaldırıldı (side, prior hizası, terminal), süre aşamalara bölündü | PUCT'a karşı 0-32 → 12-41 (%22,6). Hâlâ zayıf, varsayılan değil | `5be328d` |
| `does_not_shuffle_when_lost` testi duvar harcamayı da ilerleme sayıyor | Yeni ağ pozisyonu %23-31 görüyor, duvarla direniyor; mekik değil | `99119f9` |
| `match` motorlara son 16 pozisyonu veriyor (web arayüzü gibi) | Eskiden iki taraf duvarlıyken 400 hamle sağ-sol yapıyordu | `65b7146` |
| Canlı maç izleme (`GEDIK_CANLI`, `/api/canli`) ve sıfırdan yeni web arayüzü | — | `67886bb`, `c251fad` |
| **Paylaşımlı ağaç + sanal kayıp** (`src/paylasim.rs`, artık çok thread'de varsayılan; `tp=0` eski kök paralelliği). Kök paralelliğinde 32 ağaç neredeyse aynıydı, tüm çekirdekler tek aramayı tekrar ediyordu | 400 ms, 4 thread: **72-30 (%70,6)**. 400 ms, tüm çekirdekler: **18-0**, Wilson %82-100 | `72218db` |
| Web arayüzü seviyeleri: hepsi tüm çekirdek, 3 sn eklendi | 2 sn ≈ 6,1M iterasyon; eski "Maksimum" (1M) ≈ 0,3 sn idi | `a8fbed2` |

Açık sorular:
- Paylaşımlı ağaç 2 sn'de 40M düğüm kapasitesini (`tn`) dolduruyor; sonra yalnız değerlendirme yapıyor. `et` (genişletme eşiği) ve `tn` ayarı, ayrıca kökte düğüm başına TT önbelleği denenmeli.
- Motor seçenekleri virgülle ayrılır: `mcts:3000ms:t=0,sv=0`. İki nokta ile yazılan anahtarlar sessizce yok sayılıyor.
- 2,5 sn'de her ağaç 6M düğüm tavanına dayanıyor (~190k iterasyon/ağaç). 3 sn ve tavanı yükseltmek ölçülmedi.
- 2,5 sn maçlarında hamle başına ortalama ~3,9 sn ölçüldü: bir motor süreyi aşıyor (muhtemelen `ru=0` tarafı her hamlede dev ağacı yeniden ayırıyor). Bakılmalı.
- 2,5 sn final ölçümü kesinleşmedi; gece bir kez daha (`--parca 1`, SPRT sürdürme ile) çalıştırılabilir.

## Araçlar

```
cargo build --release
target\release\gedik match "<motor A>" "<motor B>" <açılış> <seed> <paralel>
target\release\damit <oyun> <iterasyon> <thread> <cikti.csv> [lambda=0.8] [ogretmen.bin] [seed]
target\release\kodla <girdi.csv> <cikti.bin>                 # MLP için QDT3
target\release\kodla --spatial <girdi.csv> <cikti.bin>       # ResNet için
python tools\egit_np.py veri.bin ag.bin --h1 128 --epoch 16 --lr 2e-3   # numpy MLP
python tools\egit.py veri.bin ag.bin --channels 32 --blocks 4           # torch, ResNet/MLP
```

Bu araçlar hakkında bilinmesi gerekenler:
- `damit` satır biçimi `fen;y;ply;v;z;` şeklinde. Başlıkta `pi` sütunu
  var ama henüz doldurulmuyor (politika hedefi yok).
- `damit` dosyası, `kodla --spatial` ile kullanılmamalı: `v` ve `z`
  sütunları spatial biçimin kalan-hamle ve yol-farkı sütunlarıyla aynı
  yere düşüyor.
- Motor dizesinin anahtarları: `t=0` tüm çekirdekleri kullanır, `ag=dosya`
  ağı seçer, `ru=0/1` ağaç yeniden kullanımını kapatır ya da açar.
  Ayrıntı README'nin "Engine strings" bölümünde.

**Dikkat:** `gedik.bat` her çalıştırmada `git checkout -f -B` ile klasörü
uzak dala sıfırlar. Geliştirmeyi **ayrı bir klonda** yap
(örn. `%USERPROFILE%\gedik-dev`), sık commit ve push et.

## Bir haftalık plan

Kaldıraç, kendi kendine oyunla sürekli öğrenen bir döngü ve hızlı ama
akıllı bir ağ. Her adım maçla doğrulanır.

### 1. gün: Altyapı ve ölçüm
1. Ortamı kur: Rust, Python, numpy ve torch cu128.
   `torch.cuda.is_available()` doğrulanmalı.
2. SPRT (sıralı olasılık oranı testi) ile terfi testi yaz. Elo0=0 ve
   Elo1=+20 hipotezlerini karşılaştırır ve sonuç belli olunca durur.
   Paralel parça çalıştırmayı (`deney/*__n.txt`) kullanmalı. 30 thread ile
   maçlar artık dakikalar içinde biter.
3. Gumbel'deki `side` hatasını düzelt ve PUCT'a karşı yeniden ölç.

### 2. gün: Değerlendirme hızı
1. MLP'nin ilk katmanını artımlı yap (NNUE tarzı akümülatör). Hamle
   yapılıp geri alınırken yalnızca değişen girdiler güncellenir.
   Hedef: değerlendirme süresinin 3-5 kat düşmesi.
2. int16/int8 nicemleme ve SIMD (AVX2/AVX-512; Zen 5 destekliyor).
3. `gedik` binary'sindeki bench ve iterasyon araçlarıyla hamle başına
   iterasyonu ölç, sonra 2,5 sn maçıyla gücü doğrula.

### 3-5. gün: Kendi kendine oyun döngüsü (AlphaZero tarzı)
1. **Üret:** En iyi ağla kendi kendine oyun oynat. `damit`'e kökteki
   ziyaret dağılımını (`pi`) yazdır ki politika hedefi olsun. 30 thread
   ile hedef gecede 30-50 bin oyun.
2. **Eğit (GPU):** Son N neslin verisiyle eğit.
   - Öğretmen tarafı: ResNet, değer ve politika başlığıyla
     (`tools/egit.py`). Daha büyük modeller denenebilir (64 kanal, 6-8 blok).
   - Oyun tarafı: Hızlı MLP, öğretmenden damıtılır. MLP'ye küçük bir
     politika başlığı eklemek de denenmeli.
3. **Terfi:** Yeni ağ eskiye karşı SPRT ile oynar, yalnız kazanırsa terfi
   eder.
4. Döngü `tools/auto_loop.py` ile otomatikleştirilebilir (önce mevcut
   halini incele). Hedef gecede 2-3 nesil.

### 6. gün: Arama ayarları
1. `cpuct`, FPU, sanal kayıp ve kök gürültüsü gibi parametreleri SPSA ile
   ayarla.
2. Kök paralelliği (thread başına ayrı ağaç) yerine paylaşılan tek ağaç
   ve sanal kayıp dene. 16 çekirdekte daha derin bir ağaç vermesi
   beklenir.
3. ResNet'i oyun sırasında GPU'da toplu değerlendirme: aramadaki 32-64
   yaprak birlikte gönderilir. Başarılı olursa akıllı ağ gerçek sürede de
   güçlü olur. Büyük iş; ancak önceki adımlar bittiyse girişilmeli.
4. Zaman yönetimi: kritik pozisyonda daha uzun, bariz hamlede daha kısa
   düşün; ortalama süre aynı kalır.

### 7. gün: Uçlar ve cila
1. `src/endgame.rs` (kesin son oyun çözücü) şu an devrede mi kontrol et.
   Değilse bağla ve ölç.
2. Açılış kitabı: ilk 6-8 hamle için uzun analizle bir tablo çıkar.
3. Final: en iyi sürümü eski sürümlere karşı 2,5 sn'de 100'den fazla
   oyunla ölç. README'yi güncelle, commit ve push et.

Tahmini toplam kazanç birkaç yüz Elo. En büyük pay kendi kendine oyun
döngüsünün ve hız iyileştirmelerinin.
