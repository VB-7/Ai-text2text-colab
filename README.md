# AIRI v2 — pip-installable, pro

Upload isi folder `v2/` ini sebagai repo GitHub (root repo = folder ini).

## Pakai di Colab (1 cell)
Lihat `colab_bootstrap.py`. Intinya:
```
%pip install -U git+https://github.com/<GITHUB_KAMU>/airi.git
from airi import init_models, airi
init_models("gemma12b")
print(airi("halo"))
```

## Pilih model dasar
Di `airi.py` → `MODEL_PRESETS`:
- `gemma12b` — siap jalan (file + mmproj sudah diisi)
- `qwen9b` — repo sudah diisi, nama file GGUF auto-detect (Q4_K_M)
- `gemma_e4b / gemma_e2b / qwen2b / qwen4b / qwenvl8b` — isi `repo` dulu (kirim list Drive biar aku isikan)

Ganti via `init_models("qwen9b")` atau env `AIRI_MODEL`.

## Aturan
- Model 100% HuggingFace. Drive HANYA `Save/History_AI_V2/` (history + state).
- Param LLM permanen di kode. Token: Secrets → `Secrets/hf_token.txt` → hardcode (Colab only).
- `airi(text)` full, `airi_stream(text)` yield buat efek mengetik.
