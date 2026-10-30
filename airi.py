# AIRI core - modul pip-installable (import airi)
# Model dari HF saja. Drive cuma history/state.
# Ganti model cukup: init_models("qwen9b") / ("gemma12b") / dst.

import os
import json
import base64
import gc
import time
import warnings
from io import BytesIO

import psutil
import torch
import numpy as np
import PIL.Image as PILImage

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
warnings.filterwarnings("ignore")

MODEL_PRESETS = {
    "gemma12b": {
        "repo": "HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced",
        "file": "Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf",
        "mmproj": "mmproj-Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-BF16.gguf",
        "handler": "gemma4",
        "n_ctx": 157286,  # 60% dari 256K
        "swa_full": False,
    },
    "gemma_e4b": {
        "repo": "HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive",
        "file": "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf",
        "mmproj": "mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf",
        "handler": "gemma4",
        "n_ctx": 131072,  # full n_ctx
        "swa_full": False,
    },
    "gemma_e2b": {
        "repo": "HauhauCS/Gemma-4-E2B-Uncensored-HauhauCS-Aggressive",
        "file": "Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf",
        "mmproj": "mmproj-Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-f16.gguf",
        "handler": "gemma4",
        "n_ctx": 131072,  # full n_ctx
        "swa_full": False,
    },
    "qwen9b": {
        "repo": "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive",
        "file": "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf",
        "mmproj": "mmproj-Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-BF16.gguf",
        "handler": "qwen25vl",  # Qwen-VL family; belum tested di 3.5, fallback: MTMDChatHandler
        "n_ctx": 170393,  # 65% dari 262K
    },
    "qwen2b": {
        "repo": "HauhauCS/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive",
        "file": "Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-BF16.gguf",
        "mmproj": "mmproj-Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-f16.gguf",
        "handler": "qwen25vl",
        "n_ctx": 262144,  # full n_ctx
    },
    "qwen4b": {
        "repo": "HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive",
        "file": "Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf",
        "mmproj": "mmproj-Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-BF16.gguf",
        "handler": "qwen25vl",
        "n_ctx": 183500,  # 70% dari 262K
    },
}

EMBED_ID = "BAAI/bge-base-en-v1.5"
RERANK_ID = "BAAI/bge-reranker-base"

ROOT_DRIVE = "/content/drive/MyDrive/Ai_Norvex/"
SAVE_DIR = os.path.join(ROOT_DRIVE, "Save/History_AI_V2")
SAVE_PATH = os.path.join(SAVE_DIR, "airi_v2.json")

ACTIVE_MODEL = "gemma12b"
HF_TOKEN_HARDCODE = ""  # isi di Colab aja, jangan commit berisi

# PARAM PERMANEN: temp 0.75, top_p 0.85, top_k 20, min_p 0.05, repeat 1.0, presence 0.5, max 5024

_llm = None
_embed_tok = None
_embed_mod = None
_rerank_tok = None
_rerank_mod = None
CHAT_HISTORY = []


def _resolve_token():
    try:
        from google.colab import userdata
        t = userdata.get('HF_TOKEN')
        if t and t.strip():
            return t.strip()
    except Exception:
        pass
    try:
        p = os.path.join(ROOT_DRIVE, "Secrets/hf_token.txt")
        if os.path.exists(p):
            t = open(p).read().strip()
            if t:
                return t
    except Exception:
        pass
    if HF_TOKEN_HARDCODE.strip():
        return HF_TOKEN_HARDCODE.strip()
    return os.environ.get("HF_TOKEN") or None


def list_model_files(repo):
    from huggingface_hub import HfApi
    api = HfApi(token=_resolve_token() or None)
    return [f for f in api.list_repo_files(repo_id=repo)]


def _auto_pick(files, must_contain=("Q4_K_M", ".gguf"), skip=("mmproj",)):
    c = [f for f in files if all(m in f for m in must_contain) and not any(s in f.lower() for s in skip)]
    if c:
        return sorted(c)[0]
    g = [f for f in files if f.endswith(".gguf") and not any(s in f.lower() for s in skip)]
    return sorted(g)[0] if g else None


def _resolve_preset(name):
    p = MODEL_PRESETS.get(name)
    if not p or not p.get("repo"):
        raise ValueError(f"Preset '{name}' belum diisi repo-nya. Isi dulu di MODEL_PRESETS.")
    repo, f, m = p["repo"], p.get("file"), p.get("mmproj")
    need_mm = p.get("handler") in ("gemma4", "qwen25vl")
    if f is None or (m is None and need_mm):
        files = list_model_files(repo)
        if f is None:
            f = _auto_pick(files)
        if m is None and need_mm:
            mm = [x for x in files if "mmproj" in x.lower() and x.endswith(".gguf")]
            m = sorted(mm)[0] if mm else None
    return repo, f, m, p.get("handler")


def log_mem(tag):
    ram = psutil.virtual_memory()
    msg = f"[{tag}] RAM {ram.used / 1e9:.1f}GB"
    if torch.cuda.is_available():
        try:
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            msg += f" | VRAM {torch.cuda.memory_allocated() / 1e9:.1f}/{total:.1f}GB"
        except Exception:
            pass
    print(msg, flush=True)


def ensure_drive():
    if not os.path.exists("/content/drive"):
        from google.colab import drive
        drive.mount('/content/drive')
    os.makedirs(SAVE_DIR, exist_ok=True)


def load_history():
    global CHAT_HISTORY
    if os.path.exists(SAVE_PATH):
        try:
            d = json.load(open(SAVE_PATH, encoding='utf-8'))
            CHAT_HISTORY = d if isinstance(d, list) else []
        except Exception:
            CHAT_HISTORY = []
    return CHAT_HISTORY


def save_history():
    try:
        json.dump(CHAT_HISTORY, open(SAVE_PATH, 'w', encoding='utf-8'), ensure_ascii=False)
    except Exception:
        pass


def init_models(name=None):
    global _llm, _embed_tok, _embed_mod, _rerank_tok, _rerank_mod, ACTIVE_MODEL
    from huggingface_hub import hf_hub_download
    name = name or os.environ.get("AIRI_MODEL", ACTIVE_MODEL)
    ensure_drive()
    load_history()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    t0 = time.time()
    if _llm is None or ACTIVE_MODEL != name:
        repo, f, m, handler = _resolve_preset(name)
        tok = _resolve_token()
        mp = hf_hub_download(repo_id=repo, filename=f, token=tok)
        pp = MODEL_PRESETS.get(name, {})
        kw = dict(model_path=mp, n_ctx=pp.get("n_ctx", 131072), n_gpu_layers=-1, n_batch=1024,
                  n_ubatch=512, swa_full=pp.get("swa_full", False), flash_attn=True, verbose=False)
        if handler == "gemma4" and m:
            from llama_cpp.llama_chat_format import Gemma4ChatHandler
            mm = hf_hub_download(repo_id=repo, filename=m, token=tok)
            kw["chat_handler"] = Gemma4ChatHandler(clip_model_path=mm, verbose=False)
        elif handler == "qwen25vl" and m:
            from llama_cpp.llama_chat_format import Qwen25VLChatHandler
            mm = hf_hub_download(repo_id=repo, filename=m, token=tok)
            kw["chat_handler"] = Qwen25VLChatHandler(clip_model_path=mm, verbose=False)
        elif handler == "qwen25vl":
            print("Peringatan: mmproj tidak ketemu, jalan teks-only.", flush=True)
            kw["chat_format"] = "qwen"
        elif handler is None:
            kw["chat_format"] = "qwen"
        from llama_cpp import Llama
        _llm = Llama(**kw)
        ACTIVE_MODEL = name
    if _embed_mod is None:
        from transformers import AutoModel, AutoTokenizer
        tok = _resolve_token()
        _embed_tok = AutoTokenizer.from_pretrained(EMBED_ID, token=tok, trust_remote_code=False)
        _embed_mod = AutoModel.from_pretrained(EMBED_ID, token=tok, trust_remote_code=False).to("cpu").eval()
    if _rerank_mod is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = _resolve_token()
        _rerank_tok = AutoTokenizer.from_pretrained(RERANK_ID, token=tok, trust_remote_code=False)
        _rerank_mod = AutoModelForSequenceClassification.from_pretrained(RERANK_ID, token=tok, trust_remote_code=False).to("cpu").eval()
    log_mem(f"ready [{ACTIVE_MODEL}] {time.time() - t0:.0f}s")
    return ACTIVE_MODEL


def content_to_text(c):
    if isinstance(c, str):
        return c
    if isinstance(c, dict):
        return c.get("text", "") if c.get("type") == "text" else ""
    if isinstance(c, list):
        return "".join(x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text")
    return str(c)


def img_to_part(img_in):
    if isinstance(img_in, dict):
        img_in = img_in.get("data") or img_in.get("path") or ""
    s = str(img_in).strip() if isinstance(img_in, str) else ""
    if s.startswith("data:image"):
        return {"type": "image_url", "image_url": {"url": s}}
    if s and os.path.exists(s):
        try:
            im = PILImage.open(s).convert("RGB")
            if im.size[0] > 1024 or im.size[1] > 1024:
                im.thumbnail((1024, 1024), PILImage.Resampling.LANCZOS)
            b = BytesIO()
            im.save(b, format="JPEG", quality=90)
            return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()}}
        except Exception as e:
            return {"type": "text", "text": f"[gambar gagal: {e}]"}
    if len(s) > 1000:
        return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + s}}
    return {"type": "text", "text": "[gambar kosong]"}


def get_vec(text):
    try:
        if _embed_tok is None:
            return None
        i = _embed_tok(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cpu")
        with torch.no_grad():
            e = _embed_mod(**i)[0][:, 0]
            e = torch.nn.functional.normalize(e, p=2, dim=1)
        return e.cpu().numpy()[0].tolist()
    except Exception:
        return None


def sim(a, b):
    try:
        a = np.array(a, dtype=np.float64)
        b = np.array(b, dtype=np.float64)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        s = float(np.dot(a, b) / (na * nb))
        return 0.0 if np.isnan(s) or np.isinf(s) else s
    except Exception:
        return 0.0


def recall(user_input, hist, threshold=0.62):
    try:
        if not isinstance(hist, list) or len(hist) < 5:
            return ""
        uv = get_vec(user_input)
        if uv is None:
            return ""
        cands = []
        for i in range(max(0, len(hist) - 20)):
            m = hist[i]
            if not isinstance(m, dict) or m.get("role") != "user":
                continue
            pv = m.get("vector")
            if pv is None:
                continue
            if sim(uv, pv) >= threshold:
                atxt = ""
                if i + 1 < len(hist) and hist[i + 1].get("role") == "assistant":
                    atxt = content_to_text(hist[i + 1].get("content", ""))
                ctxt = content_to_text(m.get("content", ""))
                if ctxt.strip():
                    cands.append(f"User: {ctxt}\nAssistant: {atxt}")
        if not cands:
            return ""
        pairs = [[user_input, c] for c in cands]
        with torch.no_grad():
            inp = _rerank_tok(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cpu")
            sc = _rerank_mod(**inp).logits.flatten().float()
            rs = torch.sigmoid(sc).cpu().numpy().tolist()
        ranked = sorted(zip(rs, cands), key=lambda x: x[0], reverse=True)
        out = "\n\n[INGATAN RELEVAN]"
        for _, t in ranked[:10]:
            out += "\n" + t
        return out + "\n[AKHIR INGATAN]"
    except Exception:
        return ""


def get_prompt():
    c = globals().get('CHATBOT', "")
    if isinstance(c, str) and c.strip() and c.strip() not in ('...', '"..."'):
        return c
    return "Kamu adalah Airi, asisten riset pintar."


def _clean_path(f):
    if isinstance(f, dict):
        return f.get("path") or f.get("name") or ""
    if hasattr(f, "path"):
        try:
            return f.path
        except Exception:
            return ""
    return f if isinstance(f, str) else ""


def build_msgs(hist, txt, imgs, sys):
    base = [{"role": "system", "content": sys}]
    win = hist[-20:] if len(hist) > 20 else hist
    for m in win:
        if m.get("role") == "user":
            parts = []
            cc = m.get("content", "")
            if isinstance(cc, list):
                for x in cc:
                    if not isinstance(x, dict):
                        continue
                    t = x.get("type")
                    if t == "text":
                        parts.append({"type": "text", "text": x.get("text", "")})
                    elif t == "file":
                        parts.append(img_to_part(_clean_path(x.get("file", {}))))
                    elif t == "image_url":
                        parts.append(x)
            else:
                parts.append({"type": "text", "text": content_to_text(cc)})
            base.append({"role": "user", "content": parts})
        elif m.get("role") == "assistant":
            base.append({"role": "assistant", "content": content_to_text(m.get("content", ""))})
    cur = []
    if txt:
        cur.append({"type": "text", "text": txt})
    for im in (imgs or []):
        cur.append(img_to_part(im))
    base.append({"role": "user", "content": cur})
    return base


def _user_content(text, images):
    out = []
    if text:
        out.append({"type": "text", "text": text})
    for im in (images or []):
        p = _clean_path(im)
        if isinstance(p, str) and p.startswith("data:image"):
            out.append({"type": "image_url", "image_url": {"url": p}})
        elif p and os.path.exists(p):
            out.append({"type": "file", "file": {"path": p}})
    return out


def airi(text, images=None):
    global CHAT_HISTORY
    if _llm is None:
        return "[❌ jalankan init_models() dulu]"
    mem = recall(text or "", CHAT_HISTORY) if (text or "").strip() else ""
    msgs = build_msgs(CHAT_HISTORY, text or "", images or [], get_prompt() + mem)
    try:
        r = _llm.create_chat_completion(messages=msgs, temperature=0.75, top_p=0.85, top_k=20,
                                        min_p=0.05, repeat_penalty=1.0, presence_penalty=0.5,
                                        max_tokens=5024, stream=False)
        out = r["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[❌ {e}]"
    CHAT_HISTORY.append({"role": "user", "content": _user_content(text, images)})
    v = get_vec(text) if text else None
    if v is not None:
        CHAT_HISTORY[-1]["vector"] = v
    CHAT_HISTORY.append({"role": "assistant", "content": [{"type": "text", "text": out}]})
    save_history()
    return out


def airi_stream(text, images=None):
    if _llm is None:
        yield "[❌ jalankan init_models() dulu]"
        return
    mem = recall(text or "", CHAT_HISTORY) if (text or "").strip() else ""
    msgs = build_msgs(CHAT_HISTORY, text or "", images or [], get_prompt() + mem)
    try:
        full = ""
        for chunk in _llm.create_chat_completion(messages=msgs, temperature=0.75, top_p=0.85, top_k=20,
                                                min_p=0.05, repeat_penalty=1.0, presence_penalty=0.5,
                                                max_tokens=5024, stream=True):
            d = chunk["choices"][0].get("delta", {}).get("content", "")
            if d:
                full += d
                yield full
        CHAT_HISTORY.append({"role": "user", "content": _user_content(text, images)})
        v = get_vec(text) if text else None
        if v is not None:
            CHAT_HISTORY[-1]["vector"] = v
        CHAT_HISTORY.append({"role": "assistant", "content": [{"type": "text", "text": full}]})
        save_history()
    except Exception as e:
        yield f"[❌ {e}]"


def get_history():
    return CHAT_HISTORY


def clear_history():
    global CHAT_HISTORY
    CHAT_HISTORY = []
    save_history()
    return []


_MODEL_LABELS = [
    ("Gemma 12B", "gemma12b"),
    ("Gemma E4B", "gemma_e4b"),
    ("Gemma E2B", "gemma_e2b"),
    ("Qwen 9B", "qwen9b"),
    ("Qwen 4B", "qwen4b"),
    ("Qwen 2B", "qwen2b"),
]


def _gradio_history():
    out = []
    for m in CHAT_HISTORY:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "")
        if role not in ("user", "assistant"):
            continue
        out.append({"role": role, "content": m.get("content", "")})
    return out


def launch_gradio(share=False):
    """Wajah Gradio — responsif di HP. Ganti model via dropdown, kirim teks + gambar."""
    import gradio as gr
    with gr.Blocks(title="Airi") as demo:
        gr.Markdown("# Airi")
        with gr.Row():
            model_dd = gr.Dropdown(choices=_MODEL_LABELS, value=ACTIVE_MODEL, label="Model")
            clear_btn = gr.Button("Hapus riwayat", variant="stop")
        status_md = gr.Markdown(f"Model aktif: **{ACTIVE_MODEL}**")
        chat = gr.Chatbot(value=_gradio_history(), label="Chat", height=500, render_markdown=True)
        with gr.Row():
            msg = gr.MultimodalTextbox(placeholder="Ketik pesan...",
                                       file_count="multiple", file_types=["image"], scale=7)
            send_btn = gr.Button("Kirim", variant="primary", scale=1)

        def _switch(name):
            init_models(name)
            return f"Model aktif: **{name}**"

        def _respond(data, hist):
            txt = ""
            files = []
            if isinstance(data, dict):
                txt = data.get("text", "") or ""
                files = data.get("files", []) or []
            else:
                txt = str(data or "")
            if not txt and not files:
                yield data, hist or []
                return
            hist = (hist or []) + [{"role": "user", "content": txt or "(gambar)"}]
            hist.append({"role": "assistant", "content": "⏳ Airi sedang berpikir..."})
            yield {"text": "", "files": []}, hist
            try:
                for partial in airi_stream(txt, list(files)):
                    hist[-1]["content"] = partial
                    yield {"text": "", "files": []}, hist
            except Exception as e:
                hist[-1]["content"] = f"[gagal: {e}]"
                yield {"text": "", "files": []}, hist

        def _clear():
            clear_history()
            return []

        model_dd.change(fn=_switch, inputs=model_dd, outputs=status_md)
        send_btn.click(fn=_respond, inputs=[msg, chat], outputs=[msg, chat])
        msg.submit(fn=_respond, inputs=[msg, chat], outputs=[msg, chat])
        clear_btn.click(fn=_clear, outputs=chat)
        demo.queue()
    demo.launch(share=share)
    return demo
