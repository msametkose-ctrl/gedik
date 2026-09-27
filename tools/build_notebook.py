import json

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🚀 Quoridor AI: Google Colab Bulut Öz-Oyun ve GPU Eğitimi\n",
            "\n",
            "Bu Colab Notebook'u ile kendi bilgisayarınızı hiç yormadan, tamamen **Google'ın ücretsiz bulut sunucuları (CPU + GPU)** üzerinde:\n",
            "1. 20.000 - 100.000 maçlık **KataGo + Gumbel AlphaZero Öz-Oyun Verisi** üretebilirsiniz.\n",
            "2. Üretilen verileri doğrudan **Bulut GPU (T4 / V100)** üzerinde eğitebilirsiniz.\n",
            "3. Yeni şampiyon modeli indirip bilgisayarınıza aktarabilirsiniz."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🛠️ Adım 1: Donanım Kontrolü ve Rust Kurulumu"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. GPU Kontrolü\n",
            "!nvidia-smi\n",
            "\n",
            "# 2. Rust Derleyici Kurulumu\n",
            "!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y\n",
            "\n",
            "import os\n",
            "os.environ['PATH'] = f\"{os.path.expanduser('~')}/.cargo/bin:\" + os.environ['PATH']\n",
            "%env PATH=/root/.cargo/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n",
            "\n",
            "!cargo --version"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📦 Adım 2: Proje Dosyalarını Yükleme ve Çıkartma\n",
            "\n",
            "**Öneri:** Sol taraftaki klasör paneline (📁) `quoridor_colab.zip` dosyasını sürükleyip bırakın, ardından bu hücreyi çalıştırın."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, zipfile, shutil\n",
            "from google.colab import files\n",
            "\n",
            "zip_path = None\n",
            "for p in ['/content/quoridor_colab.zip', 'quoridor_colab.zip']:\n",
            "    if os.path.exists(p):\n",
            "        zip_path = p\n",
            "        break\n",
            "\n",
            "if not zip_path:\n",
            "    print('⚠️ quoridor_colab.zip bulunamadı. Lütfen bilgisayarınızdaki quoridor_colab.zip dosyasını seçin:')\n",
            "    uploaded = files.upload()\n",
            "    for k in uploaded.keys():\n",
            "        if k.endswith('.zip'):\n",
            "            zip_path = k\n",
            "            break\n",
            "\n",
            "if zip_path and os.path.exists(zip_path):\n",
            "    print(f\"📦 '{zip_path}' çıkartılıyor...\")\n",
            "    with zipfile.ZipFile(zip_path, 'r') as zip_ref:\n",
            "        zip_ref.extractall('/content')\n",
            "    \n",
            "    # Windows ters eğik çizgi yollarını Linux için otomatik onar\n",
            "    for root, dirs, files_list in os.walk('/content'):\n",
            "        for f in files_list:\n",
            "            if '\\\\' in f:\n",
            "                old_full = os.path.join(root, f)\n",
            "                new_rel = f.replace('\\\\', '/')\n",
            "                new_full = os.path.join('/content', new_rel)\n",
            "                os.makedirs(os.path.dirname(new_full), exist_ok=True)\n",
            "                shutil.move(old_full, new_full)\n",
            "                \n",
            "    print('✅ Dosyalar başarıyla çıkartıldı ve dizinler doğrulandı!')\n",
            "else:\n",
            "    raise FileNotFoundError(\"quoridor_colab.zip bulunamadı! Lütfen Colab sol panelindeki Dosyalar alanına quoridor_colab.zip dosyasını yükleyin.\")\n",
            "\n",
            "assert os.path.exists('/content/Cargo.toml'), 'Cargo.toml bulunamadı!'\n",
            "assert os.path.exists('/content/tools/egit.py'), 'tools/egit.py bulunamadı!'\n",
            "print('✅ Gerekli tüm kaynak dosyalar hazır!')\n",
            "!ls -la /content"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## ⚡ Adım 3: Yüksek Performanslı Linux İkili Dosyalarını Derleme"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%cd /content\n",
            "!source $HOME/.cargo/env && cargo build --release --bins\n",
            "!chmod +x /content/target/release/selfplay /content/target/release/kodla /content/target/release/quoridor\n",
            "assert os.path.exists('/content/target/release/selfplay'), 'Derleme başarısız oldu!'\n",
            "print('✅ Rust ikili dosyaları (selfplay, kodla, quoridor) başarıyla derlendi!')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🎮 Adım 4: Bulutta Öz-Oyun (Self-Play) Veri Üretimi\n",
            "\n",
            "Google Colab'in tüm CPU çekirdekleri kullanılarak KataGo formatında maçlar oynanır.\n",
            "- `OYUN_SAYISI`: Üretilecek maç sayısı (örn. 20000, 50000, 100000)\n",
            "- `MCTS_ITER`: Hamle başına düşünme iterasyonu (800 önerilir)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%cd /content\n",
            "import multiprocessing\n",
            "\n",
            "OYUN_SAYISI = 20000   # İsteğe göre 50000 veya 100000 yapabilirsiniz\n",
            "MCTS_ITER = 800\n",
            "CEKIRDEK = multiprocessing.cpu_count()\n",
            "\n",
            "print(f'🚀 {CEKIRDEK} CPU çekirdeği ile {OYUN_SAYISI} maçlık üretim başlatılıyor...')\n",
            "!/content/target/release/selfplay {OYUN_SAYISI} {MCTS_ITER} {CEKIRDEK} selfplay_colab.csv\n",
            "assert os.path.exists('/content/selfplay_colab.csv'), 'Öz-oyun üretimi başarısız oldu!'\n",
            "print('✅ Veri üretimi başarıyla tamamlandı!')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🔢 Adım 5: Veriyi KataGo Çoklu Hedef (QDT5) Formatına Kodlama"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%cd /content\n",
            "!/content/target/release/kodla --katago selfplay_colab.csv selfplay_colab.bin\n",
            "assert os.path.exists('/content/selfplay_colab.bin'), 'Kodlama başarısız oldu!'\n",
            "print('✅ QDT5 ikili veri kümesi hazır!')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🧠 Adım 6: Bulut GPU ile KataGo Sinir Ağını Eğitme"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%cd /content\n",
            "# GPU üzerinde 4-ResBlock modelini eğit (warm-start from ag.bin)\n",
            "!python3 /content/tools/egit.py /content/selfplay_colab.bin /content/ag_colab_champion.bin --channels 32 --blocks 4 --epoch 12 --batch 2048 --lr 0.001 --yukle /content/ag.bin\n",
            "assert os.path.exists('/content/ag_colab_champion.bin'), 'Model eğitimi başarısız oldu!'\n",
            "print('✅ Yeni model başarıyla eğitildi: ag_colab_champion.bin')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## ⚔️ Adım 7: Şampiyonluk Testi ve Modeli İndirme"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%cd /content\n",
            "from google.colab import files\n",
            "\n",
            "# Yeni model ile eski model arasında 10 maçlık turnuva\n",
            "!/content/target/release/quoridor match mcts:5000:ag=ag_colab_champion.bin mcts:5000:ag=ag.bin 5\n",
            "\n",
            "# Modeli bilgisayarınıza indirin\n",
            "files.download('/content/ag_colab_champion.bin')\n",
            "print('📥 ag_colab_champion.bin indirme başlatıldı!')"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "T4",
            "provenance": []
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

with open("quoridor_colab_egitim.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("quoridor_colab_egitim.ipynb başarıyla güncellendi!")
