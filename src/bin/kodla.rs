//! `gendata2` CSV'sini eğitime hazır ikili dosyaya çevirir.
//!
//! Neden ayrı bir araç: kodlamayı **yalnızca Rust yapıyor**. Eğitim betiği
//! FEN'i görmüyor, sadece burada üretilen indeks listesini okuyor. Python ile
//! Rust'ın kodlamayı farklı yapması bu tür projelerdeki en sinsi hata —
//! model güzelce eğitiliyor, motorda çöp çıkıyor ve sebebi haftalarca
//! bulunamıyor. Tek kodlayıcı, o kapı kapalı.
//!
//! Biçimler:
//! - QDT3 (MLP için): "QDT3", u32 satır, u32 n_sparse, u32 n_dense, u32 K, u32 P
//! - QDT4 (2B ResNet için): "QDT4", u32 satır, u32 channels (13), u32 spatial_size (1053), u32 P
//!
//! Kullanım:
//!     kodla <girdi.csv> <cikti.bin>
//!     kodla --spatial <girdi.csv> <cikti.bin>
use gedik::nn::{
    dense_features, sparse_indices, spatial_planes, N_DENSE, N_SPARSE, SPATIAL_CHANNELS,
    SPATIAL_SIZE,
};
use gedik::notation::from_fen;
use std::io::{BufRead, BufWriter, Write};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let is_spatial = args.iter().any(|a| a == "--spatial" || a == "--katago");
    let clean_args: Vec<String> = args
        .into_iter()
        .filter(|a| a != "--spatial" && a != "--katago")
        .collect();

    if clean_args.len() < 3 {
        eprintln!("Kullanım: kodla [--spatial|--katago] <girdi.csv> <cikti.bin>");
        return;
    }

    let (gi, co) = (clean_args[1].clone(), clean_args[2].clone());

    let f = std::fs::File::open(&gi).expect("girdi acilamadi");
    let rd = std::io::BufReader::with_capacity(1 << 20, f);
    let out = std::fs::File::create(&co).expect("cikti acilamadi");
    let mut w = BufWriter::with_capacity(1 << 20, out);

    const K: usize = 26;
    const P: usize = 24;

    if is_spatial {
        w.write_all(b"QDT5").unwrap();
        w.write_all(&0u32.to_le_bytes()).unwrap();
        w.write_all(&(SPATIAL_CHANNELS as u32).to_le_bytes())
            .unwrap();
        w.write_all(&(SPATIAL_SIZE as u32).to_le_bytes()).unwrap();
        w.write_all(&(P as u32).to_le_bytes()).unwrap();
    } else {
        w.write_all(b"QDT3").unwrap();
        w.write_all(&0u32.to_le_bytes()).unwrap();
        w.write_all(&(N_SPARSE as u32).to_le_bytes()).unwrap();
        w.write_all(&(N_DENSE as u32).to_le_bytes()).unwrap();
        w.write_all(&(K as u32).to_le_bytes()).unwrap();
        w.write_all(&(P as u32).to_le_bytes()).unwrap();
    }

    let mut idx: Vec<u16> = Vec::with_capacity(32);
    let mut n = 0u32;
    let mut atlanan = 0u64;

    for line in rd.lines() {
        let Ok(line) = line else { break };
        let line = line.trim();
        if line.is_empty() || line.starts_with("fen;") {
            continue;
        }
        let parts: Vec<&str> = line.split(';').collect();
        if parts.len() < 3 {
            atlanan += 1;
            continue;
        }
        let fen = parts[0];
        let Some(pos) = from_fen(fen) else {
            atlanan += 1;
            continue;
        };

        // Otomatik etiket, kalan hamle, yol farkı ve pi tespiti:
        // Formatlar:
        // 6 sütun: fen;winner;ply;rem_plies;final_delta;pi
        // 4 sütun: fen;winner;ply;pi veya fen;ply;pi;winner
        let (y, rem_plies, final_delta, pi_ham) = if parts.len() >= 6 {
            let y_val = parts[1].parse::<f32>().unwrap_or(0.5);
            let rem = parts[3].parse::<f32>().unwrap_or(0.0) / 50.0;
            let d_delta = parts[4].parse::<f32>().unwrap_or(0.0) / 10.0;
            (y_val, rem, d_delta, parts[5])
        } else if parts.len() >= 4 {
            if parts[1] == "0" || parts[1] == "1" {
                (parts[1].parse::<f32>().unwrap(), 0.0, 0.0, parts[3])
            } else if parts[3] == "0" || parts[3] == "1" {
                (parts[3].parse::<f32>().unwrap(), 0.0, 0.0, parts[2])
            } else {
                atlanan += 1;
                continue;
            }
        } else if parts.len() >= 2 && (parts[1] == "0" || parts[1] == "1") {
            (
                parts[1].parse::<f32>().unwrap(),
                0.0,
                0.0,
                parts.get(2).copied().unwrap_or(""),
            )
        } else {
            atlanan += 1;
            continue;
        };

        if !(0.0..=1.0).contains(&y) {
            atlanan += 1;
            continue;
        }

        w.write_all(&y.to_le_bytes()).unwrap();

        if is_spatial {
            w.write_all(&rem_plies.to_le_bytes()).unwrap();
            w.write_all(&final_delta.to_le_bytes()).unwrap();
            let planes = spatial_planes(&pos);
            for val in planes {
                w.write_all(&val.to_le_bytes()).unwrap();
            }
        } else {
            sparse_indices(&pos, &mut idx);
            let d = dense_features(&pos);
            assert!(idx.len() <= K, "aktif indeks {} > K {}", idx.len(), K);
            w.write_all(&(idx.len() as u16).to_le_bytes()).unwrap();
            for j in 0..K {
                let v = idx.get(j).copied().unwrap_or(0xFFFF);
                w.write_all(&v.to_le_bytes()).unwrap();
            }
            for x in d {
                w.write_all(&x.to_le_bytes()).unwrap();
            }
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

    println!("{n} satir yazildi ({atlanan} atlandi, spatial={is_spatial}) -> {co}");
}
