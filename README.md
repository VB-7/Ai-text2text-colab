[![Buka di Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VB-7/Ai-text2text-colab/blob/main/Airi.ipynb)

# Airi — chatbot + image Colab (pilih mode di Cell 2)

Satu Colab untuk dua kebutuhan. Model ngambil dari HuggingFace, Drive cuma buat history/state + backup gambar.

- **Generate Text** = chat AI pake Gemma4 12B uncensored (modul `airi`).
- **Generate Image** = SDXL Illustrious + LoRA + LLM E4B prompt-enhancer (modul `airi_image`).

Gak ada pilihan model. Sengaja dikunci biar simpel.

## Cara pakai (2 cell)

**Cell 1 — install (cukup sekali per runtime):**

```python
from google.colab import drive
drive.mount('/content/drive')
# Deteksi CUDA torch Colab -> pilih wheel llama-cpp yang cocok (tahan update Colab)
import subprocess, torch
_cu = (torch.version.cuda or "12.1").replace(".", "")
print("torch", torch.__version__, "| cuda", torch.version.cuda, "-> wheel cu" + _cu)
_url = f"https://abetlen.github.io/llama-cpp-python/whl/cu{_cu}/"
if subprocess.run(["pip", "install", "-q", "llama-cpp-python",
                    "--extra-index-url", _url]).returncode != 0:
    print("wheel cu" + _cu + " tidak ada, fallback ke cu121")
    subprocess.run(["pip", "install", "-q", "llama-cpp-python",
                    "--extra-index-url",
                    "https://abetlen.github.io/llama-cpp-python/whl/cu121/"],
                   check=True)
!pip install -U --upgrade -q transformers huggingface_hub gradio Pillow diffusers psutil
!pip install -q -U --no-deps git+https://github.com/VB-7/Ai-text2text-colab.git
```

> Habis Cell 1, **wajib restart runtime** (Runtime > Restart session), baru run Cell 2.

**Cell 2 — pilih mode di dropdown, run (boleh di-run ulang kapan aja):**

```python
MODE = "Generate Text"  #@param ["Generate Text", "Generate Image"]

if MODE == "Generate Text":
    from airi import init_models, launch_gradio
    init_models()  # langsung Gemma4 12B
    launch_gradio()
else:
    !pip install -q -U xformers diffusers compel peft sentencepiece accelerate torchao
    from airi_image import launch_gradio as launch_image_gradio
    launch_image_gradio()
```

Mode Text langsung jalan (deps sudah di Cell 1). Mode Image nambah install sekali (`xformers/diffusers/compel/peft/sentencepiece/accelerate/torchao`). Wheel `llama-cpp` dipilih otomatis mengikuti CUDA torch yang kepasang di Colab (fallback cu121) — `llama-cpp` harus dari wheel CUDA, kalau dari source nunggunya lama + jadinya CPU-only.

**Update (hanya kalau `.py` di GitHub berubah):**

```python
!pip install -q --no-cache-dir --no-deps --force-reinstall git+https://github.com/VB-7/Ai-text2text-colab.git
```

Habis itu run ulang Cell 2. Cell 1 gak perlu diulang.

## Model

Text (modul `airi`, file `airi.py` — jangan ditimpa, beda fungsi dengan image):
- Repo: `HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced`
- File: `Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf`
- mmproj: `mmproj-Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-BF16.gguf`
- n_ctx: `157286`, `swa_full=False`, `flash_attn=True`

Image (modul `airi_image`, file `airi_image.py`):
- SDXL: mirror `BoRnno0b/mirror` (sub `OLD`), file `Dvine-v11.1-Illustrious.safetensors`
  (nama file di Drive versi CivitAI beda kapital — sudah di-mapping di kode)
- LLM enhancer: `HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive` (E4B cukup, hemat VRAM bareng SDXL)
- LoRA: 3 dari Drive + 1 dari HF (lihat `LORAS` di `airi_image.py`)

## Catatan

- Token HF cuma perlu kalau modelnya gated. Repo yang dipake sekarang publik, jadi kosong pun jalan.
- Param LLM dikunci di kode, gak ada slider.
- History chat kesimpen di Drive (`Save/History_AI_V2/`), hasil gambar bisa auto-backup ke Drive dari UI.
- Kalau nanti nama repo diganti, ganti juga semua URL `git+https://github.com/VB-7/Ai-text2text-colab.git` di file ini dan `colab_bootstrap.py`.
