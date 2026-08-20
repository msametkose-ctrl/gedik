//! Turnuva sonuçlarından Elo hesabı.
//!
//! Basit kazanma yüzdesi yanıltıcı: bir motor sadece zayıf rakiplerle
//! oynadıysa yüzdesi şişer. Elo, tüm eşleşmeleri birlikte çözerek
//! "A'yı yenen B'yi yenen C" zincirlerini hesaba katıyor.
//!
//! Model standart lojistik:
//!
//! ```text
//! P(i, j'yi yener) = 1 / (1 + 10^((R_j - R_i) / 400))
//! ```
//!
//! Ağırlıklar maksimum olabilirlikle, gradyan çıkışıyla çözülüyor.
//! Ortalama 0'a sabitleniyor (Elo sadece farklar açısından anlamlı).

#[derive(Clone, Copy, Debug)]
pub struct Pairing {
    pub a: usize,
    pub b: usize,
    /// A'nın topladığı puan (galibiyet 1, berabere 0.5).
    pub a_score: f64,
    pub games: f64,
}

const SCALE: f64 = 400.0 / std::f64::consts::LN_10;

fn expected(ri: f64, rj: f64) -> f64 {
    1.0 / (1.0 + ((rj - ri) / SCALE).exp())
}

/// Elo derecelendirmeleri. `n` motor sayısı.
pub fn solve(n: usize, pairings: &[Pairing], iters: usize) -> Vec<f64> {
    let mut r = vec![0.0f64; n];
    if n == 0 || pairings.is_empty() {
        return r;
    }
    let lr = 8.0;
    for _ in 0..iters {
        let mut grad = vec![0.0f64; n];
        for p in pairings {
            if p.games <= 0.0 {
                continue;
            }
            let e = expected(r[p.a], r[p.b]);
            // d/dR_a  loglik = (gerçek - beklenen) / SCALE
            let d = (p.a_score - e * p.games) / SCALE;
            grad[p.a] += d;
            grad[p.b] -= d;
        }
        for i in 0..n {
            r[i] += lr * grad[i];
        }
        // Ortalamayı sıfırla — model sadece farklara duyarlı.
        let mean: f64 = r.iter().sum::<f64>() / n as f64;
        for v in r.iter_mut() {
            *v -= mean;
        }
    }
    r
}

/// Kaba standart hata: skorun binom belirsizliğini Elo'ya çevirir.
pub fn elo_stderr(score: f64, games: f64) -> f64 {
    if games < 2.0 {
        return 0.0;
    }
    let p = (score / games).clamp(0.02, 0.98);
    let se_p = (p * (1.0 - p) / games).sqrt();
    // dElo/dp = SCALE / (p(1-p))
    SCALE * se_p / (p * (1.0 - p)).max(1e-6) * (p * (1.0 - p))
}
