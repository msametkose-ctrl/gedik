#!/bin/sh
# Gedik — tek komutla derle + sunucuyu başlat + tarayıcıyı aç.
#   ./basla.sh          -> http://127.0.0.1:8080
#   ./basla.sh 9000     -> başka port
set -e
cd "$(dirname "$0")"
PORT=${1:-8080}

if ! command -v cargo >/dev/null 2>&1; then
  echo "Rust kurulu değil. Kurmak için:"
  echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
  echo "Kurulumdan sonra yeni bir terminal aç ve bu betiği tekrar çalıştır."
  exit 1
fi

echo "Derleniyor (ilk sefer ~15 saniye, sonrakiler anında)..."
cargo build --release

URL="http://127.0.0.1:$PORT"
(
  sleep 1
  if   command -v open     >/dev/null 2>&1; then open "$URL" >/dev/null 2>&1
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL" >/dev/null 2>&1
  fi
) &

echo ""
echo "Tarayıcıda aç:  $URL"
echo "Durdurmak için: Ctrl-C"
echo ""
exec ./target/release/gedik serve "$PORT"
