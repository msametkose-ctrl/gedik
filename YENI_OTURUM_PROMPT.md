Merhaba. Gedik adlı Quoridor motorunu (Rust, MCTS ve sinir ağı
değerlendirmesi) bu laptopta, bir hafta içinde olabildiğince güçlü hale
getirmek istiyorum.

Bağlam:
- Repo: github.com/msametkose-ctrl/gedik, dal `claude/dreamy-fermat-lz00su`.
  Çalışmalarımızı bu dala commit edip push et.
- Önceki (bulut) oturumda yapılanlar, ölçüm sonuçları ve bir haftalık
  plan repodaki `DEVAM.md` dosyasında. **İlk iş onu ve `CONTRIBUTING.md`'yi
  baştan sona oku.**
- Proje kuralı: "ölç, yoksa girmez". Oyun gücüne dair her değişiklik maç
  sonucuyla, Wilson aralığıyla kanıtlanmalı.
- Benim senaryom: motoru hamle başına 2-3 saniye, tüm çekirdeklerle
  düşündürüyorum (web arayüzü). Önemli olan gerçek süredeki güç.
- Donanım: Windows laptop, Ryzen 9 9955HX3D (16 çekirdek / 32 thread),
  32 GB RAM, RTX 5070 Ti Laptop. PyTorch için cu128 sürümü gerekiyor.
- `gedik.bat`, `%USERPROFILE%\gedik-guncel` klasörünü her çalıştırmada uzak
  dala sıfırlıyor. Geliştirmeyi ayrı bir klonda yap
  (`%USERPROFILE%\gedik-dev`).

Senden istediğim:
1. Ortamı kur ve doğrula: Rust release derlemesi, `cargo test --release`,
   Python, numpy, torch cu128 ve `torch.cuda.is_available()`.
2. `DEVAM.md`'deki planı 1. günden başlayarak uygula. Önce SPRT ile hızlı
   terfi testini kur, çünkü sonraki her adımı o hızlandırır.
3. Her adımın sonunda bana kısa bir durum ver: ne yapıldı, maç skoru ne,
   commitlendi mi.
4. Uzun işleri (veri üretimi, eğitim, maçlar) arka planda çalıştır.
   Bekleme sırasında beni bilgilendir; uzun süre sessiz bekleme.

Türkçe konuşalım.
