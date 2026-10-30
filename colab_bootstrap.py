# 1 CELL Colab — Airi V2
from google.colab import drive
drive.mount('/content/drive')

# llama-cpp prebuilt CUDA (biar cepet, jangan dari source)
!pip install -q llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121/
# dependensi pendamping
!pip install -q -U transformers huggingface_hub Pillow psutil markdown bleach
# paket airi (tanpa deps, sudah di-install di atas)
!pip install -q -U --no-deps git+https://github.com/VB-7/Ai-text2text-colab.git

from airi import init_models, launch_ui

# pilih model dasar: "gemma12b" / "gemma_e4b" / "gemma_e2b" / "qwen9b" / "qwen4b" / "qwen2b"
init_models("gemma12b")
launch_ui()
