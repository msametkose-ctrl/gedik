//! `gendata2` CSV'sini eğitime hazır ikili dosyaya çevirir.
//!
//! Neden ayrı bir araç: kodlamayı **yalnızca Rust yapıyor**. Eğitim betiği
//! FEN'i görmüyor, sadece burada üretilen indeks listesini okuyor. Python ile
//! Rust'ın kodlamayı farklı yapması bu tür projelerdeki en sinsi hata —
//! model güzelce eğitiliyor, motorda çöp çıkıyor ve sebebi haftalarca
//! bulunamıyor. Tek kodlayıcı, o kapı kapalı.
//!
//! Biçim (küçük endian), **sabit genişlikli**:
//!   "QDT3", u32 satır, u32 n_sparse, u32 n_dense, u32 K, u32 P
//!   satır başına: f32 etiket, u16 k, u16 indeks x K (boşlar 0xFFFF),
//!                 f32 yoğun x n_dense,
//!                 u16 aksiyon x P (boşlar 0xFFFF), f32 oran x P
//!
//! Son iki alan **politika hedefi**: aramanın o pozisyonda hangi hamleye
//! kaç kez baktığı. Motorun hamle sıralaması ("hangi hamlelere bakmaya
//! değer") şu an elle yazılmış tahminlerden ibaret; bu alan onu da
//! öğretebilmek için.
//!
//! Sabit genişlik şart: milyonlarca satırı Python'da tek tek çözmek dakikalar
//! sürüyor ve gigabaytlarca bellek yiyor. Böyle olunca numpy dosyanın
//! tamamını tek `fromfile` çağrısıyla okuyor.
//!
//! Kullanım: kodla <girdi.csv> <cikti.bin>
use gedik::nn::{dense_features, sparse_indices, N_DENSE, N_SPARSE};
use gedik::notation::from_fen;
use std::io::{BufRead, BufWriter, Write};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (gi, co) = (a[1].clone(), a[2].clone());

    let f = std::fs::File::open(&gi).expect("girdi acilamadi");
    let rd = std::io::BufReader::with_capacity(1 << 20, f);
    let out = std::fs::File::create(&co).expect("cikti acilamadi");
    let mut w = BufWriter::with_capacity(1 << 20, out);

    // Başlığı önce yer tutucu olarak yazıp sonunda düzeltiyoruz: satır
    // sayısını baştan bilmiyoruz (dosya kesilmiş olabilir).
    // En fazla kaç aktif indeks olabilir: 2 piyon + 20 duvar + 2 sayaç + sıra.
    const K: usize = 26;
    // Aramanin ilk 24 hamlesi: `SearchStats.top` zaten 24'e kirpiyor.
    const P: usize = 24;
    w.write_all(b"QDT3").unwrap();
    w.write_all(&0u32.to_le_bytes()).unwrap();
    w.write_all(&(N_SPARSE as u32).to_le_bytes()).unwrap();
    w.write_all(&(N_DENSE as u32).to_le_bytes()).unwrap();
    w.write_all(&(K as u32).to_le_bytes()).unwrap();
    w.write_all(&(P as u32).to_le_bytes()).unwrap();

    let mut idx: Vec<u16> = Vec::with_capacity(32);
    let mut n = 0u32;
    let mut atlanan = 0u64;

    for line in rd.lines() {
        let Ok(line) = line else { break };
        let line = line.trim();
        // Başlık satırı ve birleştirmeden gelen tekrarları atla.
        if line.is_empty() || line.starts_with("fen;") {
            continue;
        }
        // fen;label;ply;pi  — süreç öldürüldüyse son satır yarım olabilir.
        let mut it = line.split(';');
        let (Some(fen), Some(lab)) = (it.next(), it.next()) else {
            atlanan += 1;
            continue;
        };
        let _ply = it.next();
        let pi_ham = it.next().unwrap_or("");
        let (Some(pos), Ok(y)) = (from_fen(fen), lab.parse::<f32>()) else {
            atlanan += 1;
            continue;
        };
        if !(0.0..=1.0).contains(&y) {
            atlanan += 1;
            continue;
        }

        sparse_indices(&pos, &mut idx);
        let d = dense_features(&pos);

        assert!(idx.len() <= K, "aktif indeks {} > K {}", idx.len(), K);
        w.write_all(&y.to_le_bytes()).unwrap();
        w.write_all(&(idx.len() as u16).to_le_bytes()).unwrap();
        for j in 0..K {
            let v = idx.get(j).copied().unwrap_or(0xFFFF);
            w.write_all(&v.to_le_bytes()).unwrap();
        }
        for x in d {
            w.write_all(&x.to_le_bytes()).unwrap();
        }
        // Politika: "aksiyon:oran" ciftleri, P'ye kadar, kalani 0xFFFF.
        let mut akt = [0xFFFFu16; P];
        let mut oran = [0.0f32; P];
        for (j, parca) in pi_ham.split_whitespace().take(P).enumerate() {
            if let Some((a, o)) = parca.split_once(':') {
                if let (Ok(a), Ok(o)) = (a.parse::<u16>(), o.parse::<f32>()) {
                    akt[j] = a;
                    oran[j] = o;
                }
            }
        }
        for a in akt {
            w.write_all(&a.to_le_bytes()).unwrap();
        }
        for o in oran {
            w.write_all(&o.to_le_bytes()).unwrap();
        }
        n += 1;
        if n % 200_000 == 0 {
            eprintln!("  {n} satir");
        }
    }
    w.flush().unwrap();
    drop(w);

    // Başlıktaki satır sayısını düzelt.
    use std::io::{Seek, SeekFrom};
    let mut f = std::fs::OpenOptions::new().write(true).open(&co).unwrap();
    f.seek(SeekFrom::Start(4)).unwrap();
    f.write_all(&n.to_le_bytes()).unwrap();

    println!("{n} satir yazildi ({atlanan} atlandi) -> {co}");
}
