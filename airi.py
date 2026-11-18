# AIRI core - modul pip-installable (import airi)
# Model TUNGGAL: Gemma4 12B dari HF. Drive cuma history/state.
# Sengaja TANPA pilihan model biar simpel. Tinggal: init_models() -> launch_gradio()

import os
import json
import base64
import gc
import time
import logging
import warnings
from io import BytesIO

import psutil
import torch
import numpy as np
import PIL.Image as PILImage

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
warnings.filterwarnings("ignore")
for _ln in ("huggingface_hub", "transformers", "torchvision", "gradio"):
    try:
        logging.getLogger(_ln).setLevel(logging.ERROR)
    except Exception:
        pass

# ===== MODEL TUNGGAL (jangan diubah-ubah) =====
MODEL_LABEL = "gemma4-12b"
MODEL_REPO = "HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced"
MODEL_FILE = "Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf"
MODEL_MMPROJ = "mmproj-Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-BF16.gguf"
MODEL_N_CTX = 157286  # 60% dari 256K, aman buat VRAM Colab
MODEL_SWA_FULL = False

EMBED_ID = "BAAI/bge-base-en-v1.5"
RERANK_ID = "BAAI/bge-reranker-base"

ROOT_DRIVE = "/content/drive/MyDrive/Ai_Norvex/"
SAVE_DIR = os.path.join(ROOT_DRIVE, "Save/History_AI_V2")
SAVE_PATH = os.path.join(SAVE_DIR, "airi_v2.json")

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


def init_models(_ignored=None):
    """Nyalakan Gemma4 12B + embedding + reranker. Tanpa argumen, tanpa pilihan."""
    global _llm, _embed_tok, _embed_mod, _rerank_tok, _rerank_mod
    from huggingface_hub import hf_hub_download
    ensure_drive()
    load_history()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    t0 = time.time()
    if _llm is None:
        print(f"Load {MODEL_LABEL}: {MODEL_REPO}", flush=True)
        tok = _resolve_token()
        mp = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, token=tok)
        try:
            mm = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_MMPROJ, token=tok)
        except Exception as e:
            print(f"Peringatan: mmproj gagal ({e}), lanjut teks-only.", flush=True)
            mm = None
        kw = dict(model_path=mp, n_ctx=MODEL_N_CTX, n_gpu_layers=-1, n_batch=1024,
                  n_ubatch=512, swa_full=MODEL_SWA_FULL, flash_attn=True, verbose=False)
        if mm:
            from llama_cpp.llama_chat_format import Gemma4ChatHandler
            kw["chat_handler"] = Gemma4ChatHandler(clip_model_path=mm, verbose=False)
        else:
            kw["chat_format"] = "qwen"
        from llama_cpp import Llama
        _llm = Llama(**kw)
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
    log_mem(f"ready [{MODEL_LABEL}] {time.time() - t0:.0f}s")
    return MODEL_LABEL


def pick_model():
    """Kompatibilitas lama: dulu ada pilihan, sekarang selalu gemma4-12b."""
    print(f"Model dikunci: {MODEL_LABEL} (tanpa pilihan).", flush=True)
    return MODEL_LABEL


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
    t = load_system_prompt()
    if t:
        return t
    c = globals().get('CHATBOT', "")
    if isinstance(c, str) and c.strip() and c.strip() not in ('...', '"..."'):
        return c
    return DEFAULT_PROMPT


PROMPT_PATH = os.path.join(SAVE_DIR, "system_prompt.txt")
DEFAULT_PROMPT = "Kamu adalah Airi, asisten riset pintar."


def load_system_prompt():
    try:
        if os.path.exists(PROMPT_PATH):
            t = open(PROMPT_PATH, encoding='utf-8').read().strip()
            if t:
                return t
    except Exception:
        pass
    return ""


def save_system_prompt(text):
    try:
        ensure_drive()
        with open(PROMPT_PATH, 'w', encoding='utf-8') as f:
            f.write(text or "")
        return True
    except Exception:
        return False


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


def launch_gradio(share=True):
    """Wajah Gradio. Kirim HANYA via tombol (Enter = baris baru). Setting = laci system prompt."""
    import gradio as gr
    with gr.Blocks(title="Airi") as demo:
        gr.Markdown(f"# Airi ({MODEL_LABEL})")
        with gr.Row():
            setting_btn = gr.Button("⚙ Setting", scale=1)
            clear_btn = gr.Button("Hapus riwayat", variant="stop", scale=1)
        settings = gr.Group(visible=False)
        with settings:
            prompt_box = gr.Textbox(value=load_system_prompt() or DEFAULT_PROMPT,
                                    label="System prompt", lines=6, max_lines=12)
            save_btn = gr.Button("Simpan prompt", variant="primary")
            save_status = gr.Markdown("")
        chat = gr.Chatbot(value=_gradio_history(), label="Chat", height=500, render_markdown=True)
        with gr.Row():
            msg = gr.MultimodalTextbox(placeholder="Ketik pesan... (Enter = baris baru)",
                                       file_count="multiple", file_types=["image"],
                                       scale=7, elem_id="airi-msg")
            send_btn = gr.Button("Kirim", variant="primary", scale=1)
        gr.HTML("""<script>
(function(){
  function bind(){
    var ta = document.querySelector('#airi-msg textarea');
    if(!ta || ta._airiBound){ return; }
    ta._airiBound = true;
    ta.addEventListener('keydown', function(e){
      if(e.key === 'Enter' && !e.shiftKey){
        e.preventDefault();
        e.stopPropagation();
        var s = ta.selectionStart, en = ta.selectionEnd;
        ta.value = ta.value.slice(0, s) + '\\n' + ta.value.slice(en);
        ta.selectionStart = ta.selectionEnd = s + 1;
        ta.dispatchEvent(new Event('input', {bubbles: true}));
      }
    }, true);
  }
  new MutationObserver(bind).observe(document.body, {childList: true, subtree: true});
  bind();
})();
</script>""")

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

        def _toggle(opened):
            opened = not opened
            return opened, gr.update(visible=opened)

        def _save_prompt(t):
            return "tersimpan ✓" if save_system_prompt(t or "") else "[gagal simpan]"

        send_btn.click(fn=_respond, inputs=[msg, chat], outputs=[msg, chat])
        clear_btn.click(fn=_clear, outputs=chat)
        set_state = gr.State(False)
        setting_btn.click(fn=_toggle, inputs=set_state, outputs=[set_state, settings])
        save_btn.click(fn=_save_prompt, inputs=prompt_box, outputs=save_status)
        demo.queue()
    demo.launch(share=share)
    return demo
