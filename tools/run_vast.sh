#!/usr/bin/env bash
set -e

NUM_GAMES=${1:-50000}
MCTS_ITERS=${2:-800}
NUM_THREADS=$(nproc)

echo "=========================================================="
echo "🚀 Quoridor AI - Vast.ai Otomatik Üretim & Eğitim Scripti"
echo "  Maç Sayısı:    $NUM_GAMES"
echo "  MCTS İterasyon: $MCTS_ITERS"
echo "  CPU Çekirdeği: $NUM_THREADS"
echo "=========================================================="

# 1. Rust Kurulumu
if ! command -v cargo &> /dev/null; then
    echo "📦 Rust derleyicisi kuruluyor..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi
source "$HOME/.cargo/env"

# 2. Release Derleme
echo "⚡ Rust ikili dosyaları derleniyor..."
cargo build --release --bins
chmod +x target/release/selfplay target/release/kodla target/release/quoridor

# 3. Öz-Oyun Üretimi
echo "🎮 $NUM_THREADS çekirdek ile $NUM_GAMES maçlık üretim başlatılıyor..."
target/release/selfplay "$NUM_GAMES" "$MCTS_ITERS" "$NUM_THREADS" selfplay_vast.csv

# 4. QDT5 Kodlama
echo "🔢 Veri kümesi QDT5 ikili uzamsal formata kodlanıyor..."
target/release/kodla --katago selfplay_vast.csv selfplay_vast.bin

# 5. GPU Eğitimi
echo "🧠 GPU üzerinde KataGo çoklu hedefli model eğitiliyor..."
python3 tools/egit.py selfplay_vast.bin ag_vast_champion.bin --channels 32 --blocks 4 --epoch 15 --batch 4096 --lr 0.001 --yukle ag.bin

# 6. Doğrulama Turnuvası
echo "⚔️ Eski model ile 10 maçlık şampiyonluk turnuvası..."
target/release/quoridor match mcts:5000:ag=ag_vast_champion.bin mcts:5000:ag=ag.bin 5

echo "=========================================================="
echo "🏆 İŞLEM TAMAMLANDI! Yeni model: ag_vast_champion.bin"
echo "=========================================================="
