# 2 CELL Colab — Airi (Text + Image, pilih di Cell 2)
# Cell 1 install base sekali per runtime, Cell 2 pilih mode.

# ================= CELL 1: INSTALL BASE (sekali per runtime) =================
from google.colab import drive
drive.mount('/content/drive')

# base bareng (dipakai dua mode)
!pip install -q -U transformers huggingface_hub "Pillow>=10,<13" psutil gradio
# paket airi (T2T + T2I, tanpa deps, sudah di-install di atas)
!pip install -q -U --no-deps git+https://github.com/VB-7/Ai-text2text-colab.git

# >>> WAJIB: Restart runtime (Runtime > Restart session), baru run CELL 2 <<<

# ================= CELL 2: RUN (pilih mode di dropdown, boleh di-run ulang) =================
MODE = "Generate Text"  #@param ["Generate Text", "Generate Image"]

if MODE == "Generate Text":
    # --- T2T: Gemma4 12B ---
    !pip install -q llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu130/
    from airi import init_models, launch_gradio
    init_models()  # langsung Gemma4 12B, tanpa pilih-pilih
    launch_gradio()

else:
    # --- T2I: SDXL Illustrious + LLM E4B ---
    !pip install -q diffusers compel safetensors tqdm
    from airi_image import launch_gradio as launch_image_gradio
    launch_image_gradio()  # UI + tombol Stop, default share=True

# ============ UPDATE (hanya kalau .py di GitHub berubah) ============
# !pip install -q --no-cache-dir --no-deps --force-reinstall git+https://github.com/VB-7/Ai-text2text-colab.git
# habis itu run ulang CELL 2. Cell 1 gak perlu diulang.
