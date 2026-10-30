# Airi — chatbot Colab

Buat chat AI di Colab pake model uncensored-nya HauhauCS. Model ngambil dari HuggingFace, Drive cuma buat nyimpen history chat.

## Cara pakai (1 cell)

```python
from google.colab import drive
drive.mount('/content/drive')
!pip install -q llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu130/
!pip install -q -U transformers huggingface_hub Pillow psutil markdown bleach gradio
!pip install -q -U --no-deps git+https://github.com/VB-7/Ai-text2text-colab.git
from airi import init_models, launch_gradio
init_models("gemma12b")
launch_gradio()
```

Urutan install jangan dibalik — `llama-cpp` harus dari wheel CUDA, kalau dari source nunggunya lama + jadinya CPU-only.

## Ganti model

Tinggal ganti nama di `init_models`:

`gemma12b` / `gemma_e4b` / `gemma_e2b` / `qwen9b` / `qwen4b` / `qwen2b`

File GGUF + mmproj + setting context per model udah diatur di `MODEL_PRESETS` dalam `airi.py`.

## Catatan

- Token HF cuma perlu kalau modelnya gated. Repo yang dipake sekarang publik, jadi kosong pun jalan.
- Param LLM dikunci di kode, gak ada slider.
- History kesimpen di Drive (`Save/History_AI_V2/`), jadi sesi berikutnya masih nyambung + ada ingatan RAG.
