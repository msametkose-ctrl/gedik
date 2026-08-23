//! Bir pozisyonun değerlendirme özelliklerini basar.
//! Eğitim betiğinin doğrulama örneklerini üretmek için.
use gedik::heuristics::{features, value_to_move};
use gedik::notation::from_fen;

fn main() {
    let fen = std::env::args().nth(1).expect("kullanım: features <fen>");
    let p = from_fen(&fen).expect("geçersiz fen");
    let f = features(&p);
    println!(
        "{}",
        f.iter().map(|v| format!("{v:.5}")).collect::<Vec<_>>().join(",")
    );
    eprintln!("değer {:.4}  A yol {:?}  B yol {:?}", value_to_move(&p), p.distance(0), p.distance(1));
}

