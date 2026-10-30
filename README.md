# AIRI v2 — pip-installable, pro

Upload isi folder `v2/` ini sebagai repo GitHub (root repo = folder ini).

## Pakai di Colab (1 cell)
Lihat `colab_bootstrap.py`. Intinya:
```
!pip install -q llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121/
!pip install -q -U transformers huggingface_hub Pillow psutil markdown bleach
!pip install -q -U --no-deps git+https://github.com/VB-7/Ai-text2text-colab.git
from airi import init_models, launch_ui
init_models("gemma12b")
launch_ui()
```
`llama-cpp` wajib via wheel cu121 (biar cepet). Baris `xformers/diffusers`-mu gak perlu buat chat ini.

## Pilih model dasar
`init_models("gemma12b" | "gemma_e4b" | "gemma_e2b" | "qwen9b" | "qwen4b" | "qwen2b")` atau env `AIRI_MODEL`. File + mmproj + n_ctx + handler sudah diisi per model di `MODEL_PRESETS`.

## Aturan
- Model 100% HuggingFace. Drive HANYA `Save/History_AI_V2/` (history + state).
- Param LLM permanen di kode. Token: Secrets → `Secrets/hf_token.txt` → hardcode (Colab only).
- `airi(text)` full, `airi_stream(text)` yield buat efek mengetik.
