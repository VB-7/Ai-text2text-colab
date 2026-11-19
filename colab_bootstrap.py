# 2 CELL Colab — Airi (Text + Image, pilih di Cell 2)
# Cell 1 install sekali per runtime, Cell 2 pilih mode.

# ================= CELL 1: INSTALL (sekali per runtime) =================
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
# paket airi (T2T + T2I, tanpa deps, sudah di-install di atas)
!pip install -q -U --no-deps git+https://github.com/VB-7/Ai-text2text-colab.git

# >>> WAJIB: Restart runtime (Runtime > Restart session), baru run CELL 2 <<<

# ================= CELL 2: RUN (pilih mode di dropdown, boleh di-run ulang) =================
MODE = "Generate Text"  #@param ["Generate Text", "Generate Image"]

if MODE == "Generate Text":
    # --- T2T: Gemma4 12B (deps sudah di Cell 1, langsung jalan) ---
    from airi import init_models, launch_gradio
    init_models()  # langsung Gemma4 12B, tanpa pilih-pilih
    launch_gradio()

else:
    # --- T2I: SDXL Illustrious + LLM E4B ---
    !pip install -q -U xformers diffusers compel peft sentencepiece accelerate torchao
    from airi_image import launch_gradio as launch_image_gradio
    launch_image_gradio()  # UI + tombol Stop, default share=True

# ============ UPDATE (hanya kalau .py di GitHub berubah) ============
# !pip install -q --no-cache-dir --no-deps --force-reinstall git+https://github.com/VB-7/Ai-text2text-colab.git
# habis itu run ulang CELL 2. Cell 1 gak perlu diulang.
