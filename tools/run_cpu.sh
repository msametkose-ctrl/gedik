#!/usr/bin/env bash
set -e

NUM_GAMES=${1:-100000}
MCTS_ITERS=${2:-800}
NUM_THREADS=$(nproc)

echo "=========================================================="
echo "⚡ Quoridor AI - Yüksek Performanslı CPU Öz-Oyun Motoru"
echo "  Maç Sayısı:     $NUM_GAMES"
echo "  MCTS İterasyon:  $MCTS_ITERS"
echo "  İş Parçacığı:   $NUM_THREADS CPU Thread"
echo "=========================================================="

# 1. Gerekli paketleri kur
if ! command -v cargo &> /dev/null; then
    echo "📦 Rust derleyicisi kuruluyor..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi
source "$HOME/.cargo/env"

# 2. İşlemciye Özel Yerel SIMD Optimizasyonuyla Derle (En Yüksek Hız)
echo "⚡ CPU donanımına özel (target-cpu=native) derleniyor..."
RUSTFLAGS="-C target-cpu=native" cargo build --release --bin selfplay
chmod +x target/release/selfplay

# 3. Öz-Oyun Üretimi
echo "🚀 $NUM_THREADS CPU çekirdeği ile üretim başladı..."
target/release/selfplay "$NUM_GAMES" "$MCTS_ITERS" "$NUM_THREADS" selfplay_cpu.csv

# 4. Çıktıyı Sıkıştır
echo "📦 Veri dosyası sıkıştırılıyor..."
gzip -k -f selfplay_cpu.csv

echo "=========================================================="
echo "✅ BİTTİ! Üretilen Veri: selfplay_cpu.csv.gz"
echo "=========================================================="
