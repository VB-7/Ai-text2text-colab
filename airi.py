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


_MD_TAGS = ["p", "br", "b", "strong", "i", "em", "code", "pre",
             "ul", "ol", "li", "h1", "h2", "h3", "h4",
             "blockquote", "table", "thead", "tbody", "tr", "th", "td", "hr", "a"]
_MD_ATTRS = {"a": ["href", "title"]}


def render_md(text):
    s = text if isinstance(text, str) else content_to_text(text)
    try:
        import markdown as _mdlib
        import bleach as _bleach
    except Exception:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
    try:
        html = _mdlib.markdown(s or "", extensions=["fenced_code", "tables", "sane_lists"])
        return _bleach.clean(html, tags=_MD_TAGS, attributes=_MD_ATTRS, strip=True)
    except Exception:
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")


def render_many(texts):
    return [render_md(t) for t in (texts or [])]


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


def build_msgs(hist, txt, imgs, sys):
    base = [{"role": "system", "content": sys}]
    win = hist[-20:] if len(hist) > 20 else hist
    for m in win:
        if m.get("role") == "user":
            parts = []
            cc = m.get("content", "")
            if isinstance(cc, list):
                for x in cc:
                    if x.get("type") == "text":
                        parts.append({"type": "text", "text": x.get("text", "")})
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
    CHAT_HISTORY.append({"role": "user", "content": [{"type": "text", "text": text or ""}]})
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
        CHAT_HISTORY.append({"role": "user", "content": [{"type": "text", "text": text or ""}]})
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


def register_colab_callbacks():
    try:
        from google.colab import output as _o
        _o.register_callback('airi.chat', lambda d: airi((d or {}).get('text', ''), (d or {}).get('images', [])))
        _o.register_callback('airi.history', lambda: get_history())
        _o.register_callback('airi.history_simple', lambda: [
            {"role": m.get("role", ""), "text": content_to_text(m.get("content", ""))}
            for m in CHAT_HISTORY if isinstance(m, dict)
        ])
        _o.register_callback('airi.clear', lambda: clear_history())
        _o.register_callback('airi.init', lambda d: init_models((d or {}).get('name', 'gemma12b')))
        _o.register_callback('airi.render', lambda t: render_md(t if isinstance(t, str) else ""))
        _o.register_callback('airi.render_many', lambda lst: render_many(lst if isinstance(lst, list) else []))
    except Exception:
        pass


_UI_HTML = """
<style>
#airi-log pre{background:#0a0d13;border:1px solid #2c3a55;border-radius:8px;padding:8px;overflow:auto}
#airi-log code{background:#0a0d13;padding:1px 5px;border-radius:5px}
#airi-log pre code{background:none;padding:0}
#airi-log table{border-collapse:collapse;width:100%;margin:6px 0}
#airi-log th,#airi-log td{border:1px solid #2c3a55;padding:4px 8px;text-align:left}
#airi-log ul,#airi-log ol{margin:4px 0;padding-left:22px}
#airi-log h1,#airi-log h2,#airi-log h3,#airi-log h4{margin:6px 0 4px}
#airi-log p{margin:4px 0}
#airi-log a{color:#7cc4ff}
</style>
<div id="airi-ui" style="font-family:system-ui;background:#0b0e14;color:#e8ecf3;border:1px solid #222a3a;border-radius:14px;padding:12px;max-width:720px">
<div style="display:flex;gap:8px;margin-bottom:8px">
<select id="airi-model" style="flex:1;min-height:44px;background:#162032;color:#fff;border:1px solid #2c3a55;border-radius:10px;padding:8px">
<option value="gemma12b">Gemma 12B</option>
<option value="gemma_e4b">Gemma E4B</option>
<option value="gemma_e2b">Gemma E2B</option>
<option value="qwen9b">Qwen 9B</option>
<option value="qwen4b">Qwen 4B</option>
<option value="qwen2b">Qwen 2B</option>
</select>
<button id="airi-clear" style="min-height:44px;background:#3a1620;color:#fff;border:1px solid #5a2c3a;border-radius:10px;padding:8px 12px">Hapus</button>
</div>
<div id="airi-status" style="font-size:12px;color:#9aa6bd;margin-bottom:8px">siap</div>
<div id="airi-log" style="display:flex;flex-direction:column;gap:8px;max-height:420px;overflow:auto;margin-bottom:8px"></div>
<div id="airi-thumbs" style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px"></div>
<div style="display:flex;gap:8px">
<input id="airi-file" type="file" accept="image/*" multiple style="display:none">
<button id="airi-imgbtn" style="min-height:44px;background:#162032;color:#fff;border:1px solid #2c3a55;border-radius:10px;padding:8px 12px">Gambar</button>
<input id="airi-text" placeholder="Ketik pesan..." style="flex:1;min-height:44px;background:#0f1522;color:#fff;border:1px solid #2c3a55;border-radius:10px;padding:8px">
<button id="airi-send" style="min-height:44px;background:#2563eb;color:#fff;border:0;border-radius:10px;padding:8px 16px;font-weight:600">Kirim</button>
</div>
</div>
<script>
(function(){
if(window._airiBound) return; window._airiBound = true;
var pendingImages = [];
function esc(s){
  return (s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>');
}
function bubble(role, html){
  var log = document.getElementById('airi-log');
  var d = document.createElement('div');
  d.style.cssText = role === 'user'
    ? 'align-self:flex-end;background:#2563eb;color:#fff;border-radius:12px;padding:8px 12px;max-width:85%;word-break:break-word'
    : 'align-self:flex-start;background:#162032;color:#e8ecf3;border:1px solid #2c3a55;border-radius:12px;padding:8px 12px;max-width:85%;word-break:break-word';
  d.innerHTML = html;
  log.appendChild(d); log.scrollTop = log.scrollHeight;
  return d;
}
function valOf(res){
  if(!res || !res.data) return '';
  if(typeof res.data['text/plain'] === 'string') return res.data['text/plain'];
  if(res.data['application/json'] !== undefined) return res.data['application/json'];
  return '';
}
async function loadHist(){
  try{
    var r = await google.colab.kernel.invokeFunction('airi.history_simple', [], {});
    var h = valOf(r) || [];
    if(typeof h === 'string'){ try{ h = JSON.parse(h); }catch(e){ h = []; } }
    var texts = h.map(function(m){ return m.text || ''; });
    var r2 = await google.colab.kernel.invokeFunction('airi.render_many', [texts], {});
    var htmls = valOf(r2) || [];
    if(typeof htmls === 'string'){ try{ htmls = JSON.parse(htmls); }catch(e){ htmls = []; } }
    h.forEach(function(m, i){ bubble(m.role, htmls[i] || esc(m.text)); });
  }catch(e){}
}
document.getElementById('airi-imgbtn').onclick = function(){ document.getElementById('airi-file').click(); };
document.getElementById('airi-file').onchange = function(e){
  Array.prototype.forEach.call(e.target.files, function(f){
    var rd = new FileReader();
    rd.onload = function(){
      pendingImages.push(rd.result);
      var im = document.createElement('img');
      im.src = rd.result;
      im.style.cssText = 'width:64px;height:64px;object-fit:cover;border-radius:8px;border:1px solid #2c3a55';
      document.getElementById('airi-thumbs').appendChild(im);
    };
    rd.readAsDataURL(f);
  });
  e.target.value = '';
};
document.getElementById('airi-clear').onclick = async function(){
  await google.colab.kernel.invokeFunction('airi.clear', [], {});
  document.getElementById('airi-log').innerHTML = '';
};
document.getElementById('airi-model').onchange = async function(e){
  var st = document.getElementById('airi-status');
  st.textContent = 'ganti model: ' + e.target.value + ' ...';
  await google.colab.kernel.invokeFunction('airi.init', [{name: e.target.value}], {});
  st.textContent = 'model: ' + e.target.value;
};
async function send(){
  var inp = document.getElementById('airi-text');
  var txt = inp.value; if(!txt && !pendingImages.length) return;
  inp.value = '';
  bubble('user', esc(txt || '(gambar)'));
  document.getElementById('airi-thumbs').innerHTML = '';
  var imgs = pendingImages; pendingImages = [];
  var wait = bubble('assistant', esc('mengetik...'));
  try{
    var r = await google.colab.kernel.invokeFunction('airi.chat', [{text: txt, images: imgs}], {});
    var out = valOf(r);
    if(typeof out !== 'string') out = JSON.stringify(out);
    var i = 0;
    var timer = setInterval(function(){
      i += 6; wait.innerHTML = esc(out.slice(0, i));
      document.getElementById('airi-log').scrollTop = 99999;
      if(i < out.length) return;
      clearInterval(timer);
      google.colab.kernel.invokeFunction('airi.render', [out], {}).then(function(r2){
        var html = valOf(r2);
        wait.innerHTML = (typeof html === 'string' && html) ? html : esc(out);
      }).catch(function(){ wait.innerHTML = esc(out); });
    }, 20);
  }catch(e){ wait.textContent = '[gagal: ' + e + ']'; }
}
document.getElementById('airi-send').onclick = send;
document.getElementById('airi-text').onkeydown = function(e){ if(e.key === 'Enter') send(); };
loadHist();
})();
</script>
"""


def launch_ui():
    register_colab_callbacks()
    try:
        from IPython.display import display, HTML
    except Exception:
        print("Butuh IPython/Colab untuk wajah.")
        return
    display(HTML(_UI_HTML))
