# 1 CELL Colab — Airi V2
# Dipisah 2 cell biar gak reinstall deps terus.
# Cell lama (1 cell penuh) pensiun — pakai 2 cell di bawah.

# ================= CELL 1: INSTALL (sekali per runtime) =================
from google.colab import drive
drive.mount('/content/drive')

# llama-cpp prebuilt CUDA 13 (sesuaikan dgn torch Colab yg cu130)
!pip install -q llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu130/
# dependensi pendamping (+gradio buat wajah)
!pip install -q -U transformers huggingface_hub "Pillow>=10,<13" psutil gradio
# paket airi (tanpa deps, sudah di-install di atas)
!pip install -q -U --no-deps git+https://github.com/VB-7/Ai-text2text-colab.git

# >>> WAJIB: Restart runtime (Runtime > Restart session), baru run CELL 2 <<<

# ================= CELL 2: RUN (boleh di-run ulang kapan aja) =================
from airi import init_models, launch_gradio

# pilih model dasar: "gemma12b" / "gemma_e4b" / "gemma_e2b" / "qwen9b" / "qwen4b" / "qwen2b"
init_models("gemma12b")
launch_gradio()

# ============ UPDATE AIRI (hanya kalau airi.py di GitHub berubah) ============
# !pip install -q --no-cache-dir --no-deps --force-reinstall git+https://github.com/VB-7/Ai-text2text-colab.git
# habis itu run ulang CELL 2. Deps (cell 1) gak perlu diulang.
