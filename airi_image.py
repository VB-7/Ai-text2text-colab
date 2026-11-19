import os
import gc
import re
import sys
import uuid
import torch
import shutil
import random
import threading
import base64
import warnings
import diffusers
from io import BytesIO
from llama_cpp import Llama
from tqdm.notebook import tqdm
from google.colab import drive
from compel import CompelForSDXL
from PIL import Image as PILImage
from safetensors.torch import load_file
from IPython.display import display, Image
from huggingface_hub import hf_hub_download, login
from llama_cpp.llama_chat_format import Gemma4ChatHandler
from diffusers.utils import convert_all_state_dict_to_peft
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

# ----- SETTING ----- #
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
warnings.filterwarnings("ignore")

# ----- LOGIN HF (ala GitHub: Secrets -> file Drive -> env) ----- #
# JANGAN taruh token asli di file ini — file ini ikut ke-publish.
def _resolve_hf_token():
    try:
        from google.colab import userdata
        t = userdata.get('HF_TOKEN')
        if t and t.strip():
            return t.strip()
    except Exception:
        pass
    try:
        p = "/content/drive/MyDrive/Ai_Norvex/Secrets/hf_token.txt"
        if os.path.exists(p):
            t = open(p).read().strip()
            if t:
                return t
    except Exception:
        pass
    return os.environ.get("HF_TOKEN", "")

HF_TOKEN = _resolve_hf_token()
if HF_TOKEN.strip():
    os.environ['HF_TOKEN'] = HF_TOKEN
    try:
        login(token=HF_TOKEN)
    except Exception as e:
        print(f"[WARN] HF login gagal (lanjut tanpa login): {e}")

# ----- GOOGLE DRIVE (malas: cuma mount kalau kepakai) ----- #
# Drive BUKAN sumber model lagi (model dari HF -> local, ala GitHub).
# Drive cuma dipakai untuk: (1) LoRA yang cuma ada di Drive, (2) backup hasil gambar.
def ensure_drive():
    if not os.path.exists("/content/drive"):
        print(" Requesting Google Drive access...")
        drive.mount("/content/drive")
        print(" Google Drive access obtained successfully.")
    return "/content/drive/MyDrive/Ai_Norvex"

# ------- PATHS: local dulu, yang di-load selalu dari local (ala GitHub) ------- #
BASE_LOCAL = "/content/Ai_Norvex"
LOCAL_SDXL_DIR = os.path.join(BASE_LOCAL, "SDXL")
LOCAL_LLM_DIR = os.path.join(BASE_LOCAL, "LLM")
LOCAL_LORA_DIR = os.path.join(BASE_LOCAL, "LoRA")
local_lora_dir = LOCAL_LORA_DIR  # nama lama, biar fungsi bawah tetap jalan

# SDXL: nama file SENGAJA beda tiap sumber (pembuat sama, boornob, tapi nulisnya beda).
# - HF    = mirror boornob: "Dvine-v11.1-Illustrious.safetensors" (repo BoRnno0b/mirror, sub OLD)
# - Local = nama kanonis : "dvine-v11.1-illustrious.safetensors"
SDXL_HF_REPO = "BoRnno0b/mirror"
SDXL_HF_SUB = "OLD"
SDXL_HF_FILE = "Dvine-v11.1-Illustrious.safetensors"
SDXL_LOCAL_FILE = "dvine-v11.1-illustrious.safetensors"
local_file_sdxl = os.path.join(LOCAL_SDXL_DIR, SDXL_LOCAL_FILE)

# LLM prompt-enhancer: E4B cukup (hemat VRAM bareng SDXL).
LLM_HF_REPO = "HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive"
LLM_GGUF_FILE = "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf"
LLM_MMPROJ_FILE = "mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf"
local_file_llm = os.path.join(LOCAL_LLM_DIR, LLM_GGUF_FILE)
local_file_vision = os.path.join(LOCAL_LLM_DIR, LLM_MMPROJ_FILE)

# LoRA: sumber HF dulu; yang cuma ada di Drive pakai "drive_file".
# Kalau nanti ketemu di HF, pindahkan ke "hf_*" biar seragam.
LORAS = [
    {"file": "Louise Zero no Tsukaima Illustrious E11 v1.safetensors",
     "weight": 0.40, "trigger": "louise francoise le blanc de la valliere",
     "drive_file": "Louise Zero no Tsukaima Illustrious E11 v1.safetensors"},
    {"file": "Niji_Semi_realism_F_N_R_epoch_10.safetensors",
     "weight": 0.65, "trigger": "SemiRrealism",
     "drive_file": "Niji_Semi_realism_F_N_R_epoch_10.safetensors"},
    {"file": "StS_Age_Slider_Illustrious_v1.safetensors",
     "weight": -5.0, "trigger": "",
     "drive_file": "StS_Age_Slider_Illustrious_v1.safetensors"},
    {"file": "Fat_pussy_under_panties-v1.0.safetensors",
     "weight": 0.30, "trigger": "",
     "hf_repo": "BoRnno0b/mirror", "hf_sub": "Lora",
     "hf_file": "Fat_pussy_under_panties-v1.0.safetensors"},
]
LORA_DRIVE_DIR = "Model_AI/AI_Image/Lora_XL"  # relatif ke Ai_Norvex di Drive

# -------- OUTPUT (local utama, Drive cuma backup) -------- #
local_results_image = os.path.join(BASE_LOCAL, "Outputs")

# ----- LOGIC FUNCTIONS ----- #
def clearing_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def copy_file_with_progress(src, dst, description):
    if not os.path.exists(src):
        raise FileNotFoundError(f"{src} tidak ditemukan.")
    file_size = os.path.getsize(src)
    chunk_size = 4096 * 1024
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        with tqdm(total=file_size, unit="B", unit_scale=True, desc=description) as pbar:
            while True:
                buf = fsrc.read(chunk_size)
                if not buf:
                    break
                fdst.write(buf)
                pbar.update(len(buf))

def download_hf_to_local(repo, sub, hf_file, target_local):
    """Download dari HF kalau belum ada di local. Kembalikan path local."""
    if os.path.exists(target_local) and os.path.getsize(target_local) > 1000:
        return target_local
    print(f"[INFO] Download HF: {hf_file}", flush=True)
    got = hf_hub_download(
        repo_id=repo,
        filename=hf_file,
        subfolder=sub if sub else None,
        local_dir=os.path.dirname(target_local),
        local_dir_use_symlinks=False,
        token=HF_TOKEN or None,
    )
    if os.path.abspath(got) != os.path.abspath(target_local):
        os.makedirs(os.path.dirname(target_local), exist_ok=True)
        os.replace(got, target_local)
    return target_local

def ensure_model_on_local(model_type):
    """Alur ala GitHub: local dulu, kalau gak ada download HF. Load selalu dari local."""
    os.makedirs(LOCAL_LORA_DIR, exist_ok=True)

    if model_type == "sdxl":
        os.makedirs(LOCAL_SDXL_DIR, exist_ok=True)
        p = download_hf_to_local(SDXL_HF_REPO, SDXL_HF_SUB, SDXL_HF_FILE, local_file_sdxl)
        print(f"[INFO] SDXL model is ready at: {p}")
        return True

    if model_type in ("llm", "vision"):
        os.makedirs(LOCAL_LLM_DIR, exist_ok=True)
        download_hf_to_local(LLM_HF_REPO, "", LLM_GGUF_FILE, local_file_llm)
        if model_type == "vision" or use_vision:
            download_hf_to_local(LLM_HF_REPO, "", LLM_MMPROJ_FILE, local_file_vision)
        print(f"[INFO] LLM model is ready at: {local_file_llm}")
        return True

    if model_type == "lora":
        print("[INFO] Checking & Preparing all LoRAs on local storage...")
        for item in LORAS:
            loc_path = os.path.join(LOCAL_LORA_DIR, item["file"])
            if os.path.exists(loc_path) and os.path.getsize(loc_path) > 1000:
                continue
            if item.get("hf_repo"):
                download_hf_to_local(item["hf_repo"], item.get("hf_sub", ""),
                                     item.get("hf_file", item["file"]), loc_path)
            elif item.get("drive_file"):
                # Cuma ada di Drive (belum ada di HF) — copy sekali ke local.
                root = ensure_drive()
                d_path = os.path.join(root, LORA_DRIVE_DIR, item["drive_file"])
                if os.path.exists(d_path) and os.path.getsize(d_path) > 1000:
                    copy_file_with_progress(d_path, loc_path, f"Copy LoRA: {item['file']}")
                else:
                    print(f"[WARN] LoRA tidak ada di local maupun Drive, dilewati: {item['file']}")
        return True

    raise ValueError("Invalid model_type provided.")

# ----- DEFAULT SYSTEM PROMPT (Level 1 bila user kosong) ----- #
DEFAULT_LLM_SYSTEM = (
    "Kamu adalah prompt-enhancer untuk SDXL Illustrious. "
    "Kembangkan ide user menjadi prompt gambar yang detail: subjek, pakaian, "
    "pose, ekspresi, latar, pencahayaan, gaya. "
    "Output HANYA comma-separated tags, tanpa kalimat pembuka."
)

# ----- LEVEL 2 TERKUNCI (selalu ditempel paling akhir system prompt) ----- #
LEVEL2_LOCKED = (
    "Kamu adalah AI arsitek pembuat prompt untuk SDXL. "
    "Respon HANYA berupa prompt — sangat dilarang merespon apapun selain prompt, "
    "tanpa pembuka, penutup, atau penjelasan. "
    "Tugasmu MENYEMPURNAKAN prompt user, bukan merombak: perbaiki tatabahasa dan detail, "
    "tetap vivid dan kreatif, tapi JANGAN ubah visi/misi ide aslinya. "
    "Penyempurnaan bukan merombak total. "
    "Minimal 1 paragraf berisi elemen penting prompt. "
    "Jangan sisipkan style (anime, realistis, dll) kecuali diminta; "
    "tapi WAJIB sisipkan prompt booster/quality (masterpiece, best quality, realistic_depth, dll)."
)

TRIGGER_RULE = (
    "Aturan trigger LoRA (wajib dipatuhi mentah): gunakan dan salin PERSIS tatabahasa, kata, "
    "kalimat, dan cara penulisan trigger di bawah ini ke dalam prompt. "
    "Jangan ubah ejaan atau kapitalisasinya."
)

def _active_lora_triggers(raw_prompt):
    """Trigger LoRA AKTIF yang disebut di prompt user (case-insensitive)."""
    found = []
    low = str(raw_prompt or "").lower()
    for item in LORAS:
        trig = (item.get("trigger") or "").strip()
        if not trig:
            continue
        if not (use_lora and os.path.exists(os.path.join(LOCAL_LORA_DIR, item["file"]))):
            continue
        if trig.lower() in low:
            found.append(item)
    return found

def build_system_prompt(level1, raw_prompt):
    """Level1 (user/default) + Level2 terkunci + blok trigger bila cocok."""
    lvl1 = (level1 or "").strip() or DEFAULT_LLM_SYSTEM
    parts = [lvl1, LEVEL2_LOCKED]
    hits = _active_lora_triggers(raw_prompt)
    if hits:
        lines = [TRIGGER_RULE, "Daftar trigger LoRA yang terdeteksi di prompt user:"]
        for i, it in enumerate(hits, 1):
            lines.append(f'{i}. {it["file"]}:\n   - "{it["trigger"]}"')
        parts.append("\n".join(lines))
    return "\n\n".join(parts)

def normalize_drive_dir(user_in, default="Ai_Norvex/Outputs"):
    """Normalisasi input lokasi Drive user -> path absolut di bawah MyDrive."""
    s = str(user_in or "").strip().replace("\\", "/")
    low = s.lower()
    for p in ("/content/drive/mydrive/", "content/drive/mydrive/",
              "/content/drive/", "content/drive/", "mydrive/"):
        if low.startswith(p):
            s = s[len(p):]
            break
    s = re.sub(r"/+", "/", s).strip().strip("/")
    if not s:
        s = default.strip().strip("/")
    ensure_drive()
    return os.path.join("/content/drive/MyDrive", s)

# ----- CONFIGURATION ----- #
pipe_sdxl = None
pipe_llm = None
pipe_vision = None
compel_engine = None

# ----- REMOVER HELPERS ----- #
def safe_remove_adapter(pipeline, adapter_name):
    try:
        if hasattr(pipeline, "delete_adapter"):
            pipeline.delete_adapter(adapter_name)
    except Exception:
        pass

def purge_all_adapters(pipeline):
    try:
        if hasattr(pipeline, "unload_lora_weights"):
            pipeline.unload_lora_weights()
    except Exception:
        pass

    for module in [pipeline.unet, getattr(pipeline, "text_encoder", None), getattr(pipeline, "text_encoder_2", None)]:
        if module is None: continue
        if hasattr(module, "peft_config") and isinstance(module.peft_config, dict):
            for adapter_name in list(module.peft_config.keys()):
                safe_remove_adapter(module, adapter_name)
            module.peft_config = {}
        if hasattr(module, "active_adapters"):
            if isinstance(module.active_adapters, list): module.active_adapters.clear()
            else: module.active_adapters = None

# ----- ENGINES LOADERS ----- #
def load_sdxl():
    global pipe_sdxl, compel_engine

    if pipe_sdxl is not None:
        del pipe_sdxl
        if compel_engine is not None:
            del compel_engine
            compel_engine = None
        clearing_cuda()

    filename_sdxl = os.path.basename(local_file_sdxl)
    pipe_sdxl = StableDiffusionXLPipeline.from_single_file(
        local_file_sdxl,
        torch_dtype=torch.float16,
        use_safetensors=True,
        safety_checker=None
    ).to("cuda")

    pipe_sdxl.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe_sdxl.scheduler.config)
    pipe_sdxl.enable_vae_slicing()
    pipe_sdxl.enable_vae_tiling()
    compel_engine = CompelForSDXL(pipe_sdxl)
    clearing_cuda()
    print("[INFO] Successfully loaded Base SDXL Engine!")

def load_llm():
    global pipe_llm, pipe_vision

    if pipe_llm is not None:
        del pipe_llm
        pipe_llm = None
        clearing_cuda()

    if use_vision:
        if pipe_vision is None:
            try:
                pipe_vision = Gemma4ChatHandler(
                    clip_model_path=local_file_vision,
                    verbose=False
                )
            except Exception:
                pipe_vision = None
    else:
        if pipe_vision is not None:
            del pipe_vision
            pipe_vision = None
            clearing_cuda()

    pipe_llm = Llama(
        model_path=local_file_llm,
        chat_handler=pipe_vision,
        n_ctx=5049,
        n_gpu_layers=10,
        n_batch=512,
        flash_attn=True,
        swa_full=False,
        verbose=False
    )
    filename_llm = os.path.basename(local_file_llm)
    print(f"[INFO] Successfully loaded LLM: {filename_llm}")

def load_lora():
    global pipe_sdxl

    if pipe_sdxl is None:
        load_sdxl()

    print("\n[LoRA Control] Injecting LoRAs into SDXL Pipeline...")
    purge_all_adapters(pipe_sdxl)

    active_adapter_names = []
    active_adapter_weights = []

    for idx, item in enumerate(LORAS):
        lora_path = os.path.join(local_lora_dir, item["file"])

        if not os.path.exists(lora_path):
            continue

            lora_filename = os.path.basename(lora_path)
            lora_dir = os.path.dirname(lora_path)

            success = False

            # ======================================================
            # METHOD 1 : Native Loader
            # ======================================================
            adapter_name = f"{uuid.uuid4().hex[:8]}_lora_{idx}"
            safe_remove_adapter(pipe_sdxl, adapter_name)

            try:
                pipe_sdxl.load_lora_weights(
                    lora_dir,
                    weight_name=lora_filename,
                    adapter_name=adapter_name
                )
                print(f"  ├─ ✅ [Native Loader] Success '{lora_filename}' -> ({adapter_name})")
                success = True

            except Exception:
                safe_remove_adapter(pipe_sdxl, adapter_name)

            # ======================================================
            # METHOD 2 : PEFT Conversion
            # ======================================================
            if not success:
                adapter_name = f"{uuid.uuid4().hex[:8]}_lora_{idx}"
                safe_remove_adapter(pipe_sdxl, adapter_name)

                try:
                    raw_dict = load_file(lora_path)
                    peft_dict = convert_all_state_dict_to_peft(raw_dict)

                    clean_peft_dict = {
                        k: v
                        for k, v in peft_dict.items()
                        if not any(
                            te in k.lower()
                            for te in [
                                "text_encoder",
                                "text_model",
                                "te1",
                                "te2",
                                "conditioner",
                            ]
                        )
                    }

                    pipe_sdxl.load_lora_weights(
                        clean_peft_dict,
                        adapter_name=adapter_name
                    )

                    print(f"  ├─ 🛡️ [PEFT Convert] Success '{lora_filename}' -> ({adapter_name})")
                    success = True

                except Exception:
                    safe_remove_adapter(pipe_sdxl, adapter_name)

            # ======================================================
            # METHOD 3 : Strict U-Net
            # ======================================================
            if not success:
                adapter_name = f"{uuid.uuid4().hex[:8]}_lora_{idx}"
                safe_remove_adapter(pipe_sdxl, adapter_name)

                try:
                    raw_dict = load_file(lora_path)

                    unet_dict = {
                        k: v
                        for k, v in raw_dict.items()
                        if "lora" in k.lower()
                        and not any(
                            te in k.lower()
                            for te in [
                                "text_encoder",
                                "te_",
                                "encoder",
                            ]
                        )
                    }

                    if unet_dict:
                        pipe_sdxl.load_lora_weights(
                            unet_dict,
                            adapter_name=adapter_name
                        )

                        print(f"  ├─ 🛡️ [Strict U-Net] Success '{lora_filename}' -> ({adapter_name})")
                        success = True

                except Exception:
                    safe_remove_adapter(pipe_sdxl, adapter_name)

            # ======================================================
            # RESULT
            # ======================================================
            if success:
                active_adapter_names.append(adapter_name)
                active_adapter_weights.append(item.get("weight", 1.0))
            else:
                print(f"  ├─ ❌ [LoRA Error Final] File '{lora_filename}' gagal dimuat!")

    if active_adapter_names:
        pipe_sdxl.set_adapters(
            active_adapter_names,
            adapter_weights=active_adapter_weights
        )
        print(f"✨ [LoRA Engine] Successfully Activated {len(active_adapter_names)} LoRAs!")

# ----- CONFIGURATION MACHINE AND INITIAL STATUS ----- #
use_llm = False
use_sdxl = True
use_lora = False
use_vision = False

del_llm = False
del_lora = False

# ----- HELPER LOGIC FUNCTION ----- #
def sanitize_prompt(prompt_string):
    cleaned = re.sub(r',(\s*,)+', ',', prompt_string)
    tags = [p.strip() for p in cleaned.split(',') if p.strip()]
    unique_tags = []
    seen = set()
    for tag in tags:
        clean_tag_for_check = re.sub(r'[()\[\]\s\d.:+--]', '', tag).lower()
        if clean_tag_for_check:
            if clean_tag_for_check not in seen:
                unique_tags.append(tag)
                seen.add(clean_tag_for_check)
        else:
            unique_tags.append(tag)
    return ', '.join(unique_tags)

def _convert_file_to_base64_llm_format(file_path):
    try:
        img = PILImage.open(file_path).convert("RGB")
        MAX_VISION_RESOLUTION = 1024
        orig_width, orig_height = img.size
        if orig_width > MAX_VISION_RESOLUTION or orig_height > MAX_VISION_RESOLUTION:
            img.thumbnail((MAX_VISION_RESOLUTION, MAX_VISION_RESOLUTION), PILImage.Resampling.LANCZOS)
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=90)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
    except Exception as e:
        return {"type": "text", "text": f"[Gagal memuat visual: {str(e)}]"}

# ----- MAIN ENGINE FUNCTION SDXL ---- #
def execute_SDXL_generation(final_prompt, negative_prompt, steps, cfg, width, height, seed, batch_count, save_dir=None):
    global pipe_sdxl, compel_engine
    if pipe_sdxl is None or compel_engine is None:
        raise RuntimeError("Pipeline SDXL atau Compel Engine belum siap/loaded!")

    print(f" ℹ️ [Debug Info] Python Version: {sys.version.split(' ')[0]}")
    print(f" ℹ️ [Debug Info] Diffusers Version: {diffusers.__version__}")
    clearing_cuda()

    os.makedirs(local_results_image, exist_ok=True)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    complete_prompt_txt = sanitize_prompt(final_prompt)
    pipe_sdxl.to(torch_dtype=torch.float16)

    # 1. Compel Handling
    conditioning = compel_engine(complete_prompt_txt, negative_prompt=negative_prompt if negative_prompt else "")
    prompt_embeds = conditioning.embeds.to("cuda")
    negative_prompt_embeds = conditioning.negative_embeds.to("cuda")
    pooled_prompt_embeds = conditioning.pooled_embeds.to("cuda")
    negative_pooled_prompt_embeds = conditioning.negative_pooled_embeds.to("cuda")

    output_images = []

    # 2. Inference Mode
    with torch.inference_mode():
        for i in range(int(batch_count)):
            current_seed = seed + i
            generator = torch.Generator("cuda").manual_seed(current_seed)

            print(f"[Render] Gambar {i+1}/{batch_count} [Seed: {current_seed}]...")
            if _stop_event.is_set():
                print("  └─ [STOP] Dihentikan user.")
                break
            try:
                _pipe_kw = dict(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    num_inference_steps=int(steps),
                    guidance_scale=float(cfg),
                    width=int(width),
                    height=int(height),
                    generator=generator
                )
                try:
                    image = pipe_sdxl(
                        callback_on_step_end=_sd_callback,
                        callback_on_step_end_tensor_inputs=["latents"],
                        **_pipe_kw
                    ).images[0]
                except TypeError:
                    try:
                        image = pipe_sdxl(callback_on_step_end=_sd_callback, **_pipe_kw).images[0]
                    except TypeError:
                        image = pipe_sdxl(**_pipe_kw).images[0]

                fname = f"aegis_{current_seed}.png"
                path_local = os.path.join(local_results_image, fname)

                image.save(path_local)
                if save_dir:
                    try:
                        shutil.copy(path_local, os.path.join(save_dir, fname))
                    except Exception as e:
                        print(f"  └─ [WARN] Backup Drive gagal: {e}")
                output_images.append(path_local)

                del image

            except _StopGeneration:
                print("  └─ [STOP] Dihentikan user.")
                break
            except Exception as e:
                print(f"  └─ [Error] Gagal render gambar {i+1}: {e}")

    del prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds, conditioning
    clearing_cuda()

    print("="*65)
    print("Rendering selesai! Gambar siap ditampilkan!")
    print("="*65)
    return output_images

# ----- HELPER FUNCTION LLM ----- #
def execute_LLM_generation(system_prompt, raw_combined_prompt):
    global pipe_llm
    if pipe_llm is None:
        if ensure_model_on_local("llm"):
            load_llm()
            clearing_cuda()

    messages = [
        {"role": "system", "content": system_prompt or DEFAULT_LLM_SYSTEM},
        {"role": "user", "content": raw_combined_prompt}
    ]

    completion = pipe_llm.create_chat_completion(
        messages=messages,
        temperature=0.75,
        top_p=0.95,
        top_k=64,
        min_p=0.05,
        repeat_penalty=1.08,
        presence_penalty=0.4,
        frequency_penalty=0.0,
        max_tokens=512
    )

    final_prompt_txt = completion["choices"][0]["message"]["content"].strip()
    clearing_cuda()
    return final_prompt_txt

def execute_LLM_VISION_generation(system_prompt, raw_combined_prompt, user_file_image):
    global pipe_llm, pipe_vision

    if pipe_llm is None:
        if ensure_model_on_local("llm"):
            if use_vision:
                ensure_model_on_local("vision")
            load_llm()
            clearing_cuda()

    image_payload = _convert_file_to_base64_llm_format(user_file_image)

    messages = [
        {"role": "system", "content": system_prompt or DEFAULT_LLM_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": raw_combined_prompt},
                image_payload
            ]
        }
    ]

    completion = pipe_llm.create_chat_completion(
        messages=messages,
        temperature=0.7,
        max_tokens=512
    )

    final_prompt_txt = completion["choices"][0]["message"]["content"].strip()
    clearing_cuda()
    return final_prompt_txt

# ----- DECISION MAKING FUNCTION ----- #
def decision_making_function(raw_combined_prompt, negative_prompt, system_prompt, steps, cfg, width, height, seed, batch_count, auto_save_gd=True, location_save_gd="Ai_Norvex/Outputs"):
    global use_llm, use_lora, pipe_llm, pipe_vision, pipe_sdxl, del_llm, del_lora

    # 0. Seed: -1 = acak 1..9999999, <=0 selain itu = peringatan + acak
    try:
        seed = int(seed)
    except Exception:
        seed = -1
    if seed == -1:
        seed = random.randint(1, 9999999)
        print(f"[INFO] Seed auto: {seed}")
    elif seed <= 0:
        print("[WARN] Set seed -1 atau >0. Pakai seed acak.")
        seed = random.randint(1, 9999999)
        print(f"[INFO] Seed auto: {seed}")
    try:
        batch_count = max(1, int(batch_count))
    except Exception:
        batch_count = 1

    # 1. LoRA siap dulu (biar deteksi trigger akurat), kecuali dimatikan
    if use_lora:
        ensure_model_on_local("lora")

    # 2. LLM enhance — sekali, bukan per batch. False = pakai mentah tanpa unload.
    if use_llm:
        if pipe_llm is None:
            if ensure_model_on_local("llm"):
                load_llm()
                clearing_cuda()
        if pipe_llm is None:
            print("[WARN] LLM gagal load, pakai prompt mentah.")
            final_prompt = raw_combined_prompt
        else:
            full_system = build_system_prompt(system_prompt, raw_combined_prompt)
            final_prompt = execute_LLM_generation(full_system, raw_combined_prompt)
            print("[INFO] Prompt final dari LLM:")
            print(final_prompt)
    else:
        final_prompt = raw_combined_prompt

    # 3. Cleanup LLM bila diminta (Del_LLM)
    if del_llm and (pipe_llm is not None or pipe_vision is not None):
        print("🧹 [LLM Cleanup] Hapus LLM dari RAM/VRAM...")
        pipe_llm = None
        pipe_vision = None
        clearing_cuda()

    # 4. Handling SDXL & LoRA (mati = bersihkan sebersih-bersihnya)
    if pipe_sdxl is None:
        if ensure_model_on_local("sdxl"):
            load_sdxl()

    if use_lora:
        load_lora()
    elif pipe_sdxl is not None:
        print("🧹 [LoRA Cleanup] Melepas semua LoRA adapter dari SDXL...")
        purge_all_adapters(pipe_sdxl)
        clearing_cuda()

    # 5. Lokasi simpan Drive (dinormalisasi, anti nyasar/duplikat)
    save_dir = None
    if auto_save_gd:
        try:
            save_dir = normalize_drive_dir(location_save_gd)
            os.makedirs(save_dir, exist_ok=True)
            print(f"[INFO] Auto-save Drive: {save_dir}")
        except Exception as e:
            print(f"[WARN] Backup Drive nonaktif: {e}")
            save_dir = None

    print("🎨 [Pipeline] Memulai eksekusi render SDXL Engine...")
    generated_images = execute_SDXL_generation(
        final_prompt=final_prompt,
        negative_prompt=negative_prompt,
        steps=int(steps),
        cfg=float(cfg),
        width=int(int(width) // 8 * 8),
        height=int(int(height) // 8 * 8),
        seed=seed,
        batch_count=batch_count,
        save_dir=save_dir
    )

    return generated_images


# ----- STOP FLAG + GRADIO UI ----- #
_stop_event = threading.Event()
_running = threading.Event()

class _StopGeneration(Exception):
    pass

def request_stop():
    """Tombol Stop — menghentikan generate berikutnya & mid-diffusion."""
    _stop_event.set()
    return "⏹ Sinyal stop dikirim — berhenti di checkpoint terdekat..."

def _sd_callback(pipe, step_idx, timestep, callback_kwargs):
    if _stop_event.is_set():
        raise _StopGeneration()
    return callback_kwargs

def launch_gradio(share=True):
    """UI Gradio T2I: 11 pengaturan + seed/ukuran + tombol Generate & Stop."""
    try:
        import gradio as gr
    except ImportError:
        print("Gradio belum ada. Jalankan dulu di Colab:  !pip install -q gradio")
        return None

    def _ui_generate(prompt, sys_prompt, n_prompt, steps, cfg, width, height,
                     seed, batch, use_llm_in, del_llm_in, use_lora_in,
                     auto_gd, loc_gd, progress=gr.Progress()):
        global use_llm, use_lora, del_llm
        if _running.is_set():
            yield [], "⏳ Masih jalan — tekan Stop dulu kalau mau batalkan.", ""
            return
        _running.set()
        _stop_event.clear()
        outs, final_txt, status = [], "", "Mulai..."
        try:
            use_llm, use_lora, del_llm = bool(use_llm_in), bool(use_lora_in), bool(del_llm_in)
            try:
                seed = int(seed)
            except Exception:
                seed = -1
            if seed == -1:
                seed = random.randint(1, 9999999)
            elif seed <= 0:
                seed = random.randint(1, 9999999)
            try:
                batch = max(1, int(batch))
            except Exception:
                batch = 1

            if _stop_event.is_set():
                yield [], "⏹ Dihentikan sebelum mulai.", ""
                return

            if use_lora:
                yield [], "Siapkan LoRA...", ""
                ensure_model_on_local("lora")

            if use_llm:
                if pipe_llm is None and not _stop_event.is_set():
                    yield [], "Load LLM E4B...", ""
                    if ensure_model_on_local("llm"):
                        load_llm()
                        clearing_cuda()
                if pipe_llm is None or _stop_event.is_set():
                    final_txt = prompt
                else:
                    yield [], "Enhance prompt via LLM...", ""
                    final_txt = execute_LLM_generation(
                        build_system_prompt(sys_prompt, prompt), prompt)
            else:
                final_txt = prompt
            yield [], "Render SDXL...", final_txt

            if del_llm and (pipe_llm is not None or pipe_vision is not None):
                pipe_globals = globals()
                pipe_globals["pipe_llm"] = None
                pipe_globals["pipe_vision"] = None
                clearing_cuda()

            if pipe_sdxl is None and not _stop_event.is_set():
                yield [], "Load SDXL...", final_txt
                if ensure_model_on_local("sdxl"):
                    load_sdxl()
            if use_lora and not _stop_event.is_set():
                load_lora()
            elif pipe_sdxl is not None:
                purge_all_adapters(pipe_sdxl)
                clearing_cuda()

            save_dir = None
            if auto_gd and not _stop_event.is_set():
                try:
                    save_dir = normalize_drive_dir(loc_gd)
                    os.makedirs(save_dir, exist_ok=True)
                except Exception as e:
                    status = f"[WARN] Backup Drive nonaktif: {e}"

            for i in range(batch):
                if _stop_event.is_set():
                    status = f"⏹ Dihentikan user ({len(outs)}/{batch} jadi)."
                    break
                progress((i) / max(1, batch), desc=f"Gambar {i+1}/{batch}")
                imgs = execute_SDXL_generation(
                    final_prompt=final_txt, negative_prompt=n_prompt,
                    steps=int(steps), cfg=float(cfg),
                    width=int(int(width) // 8 * 8), height=int(int(height) // 8 * 8),
                    seed=int(seed) + i, batch_count=1, save_dir=save_dir)
                outs += imgs
                progress((i + 1) / max(1, batch), desc=f"Gambar {i+1}/{batch} jadi")
                status = f"Jadi {len(outs)}/{batch} (seed {seed+i})" + (f" — {status}" if status.startswith("[WARN]") else "")
                yield list(outs), status, final_txt
            else:
                status = f"✅ Selesai {len(outs)}/{batch} gambar (seed awal {seed})."
            yield list(outs), status, final_txt
        except Exception as e:
            yield list(outs), f"[gagal: {e}]", final_txt
        finally:
            _running.clear()

    with gr.Blocks(title="Airi T2I") as demo:
        gr.Markdown("# Airi T2I — SDXL + LLM E4B")
        with gr.Row():
            prompt_box = gr.Textbox(label="Prompt", lines=3, scale=7,
                                    placeholder="1girl, ... (tulis trigger LoRA bila pakai)")
            gen_btn = gr.Button("🎨 Generate", variant="primary", scale=1)
            stop_btn = gr.Button("⏹ Stop", variant="stop", scale=1)
        with gr.Row():
            sys_box = gr.Textbox(label="System prompt LLM (kosong = default)", lines=2, scale=1)
            neg_box = gr.Textbox(label="Negative prompt", lines=2, scale=1,
                                 value="blurry, low quality, worst quality, watermark, deformed")
        with gr.Row():
            steps_s = gr.Slider(5, 100, value=28, step=1, label="Steps")
            cfg_s = gr.Slider(1, 20, value=6.5, step=0.5, label="Guidance (CFG)")
            batch_s = gr.Slider(1, 4, value=1, step=1, label="Batch")
            seed_n = gr.Number(value=-1, precision=0, label="Seed (-1 = acak)")
        with gr.Row():
            w_s = gr.Slider(512, 1536, value=832, step=8, label="Width")
            h_s = gr.Slider(512, 1536, value=1216, step=8, label="Height")
        with gr.Row():
            llm_c = gr.Checkbox(value=True, label="Use LLM")
            delllm_c = gr.Checkbox(value=False, label="Del LLM sesudahnya")
            lora_c = gr.Checkbox(value=True, label="Use LoRA")
            gd_c = gr.Checkbox(value=True, label="Auto-save Drive")
            loc_t = gr.Textbox(value="Ai_Norvex/Outputs", label="Folder Drive")
        status_box = gr.Textbox(label="Status", interactive=False)
        final_box = gr.Textbox(label="Prompt final (dari LLM)", lines=3, interactive=False)
        gal = gr.Gallery(label="Hasil", columns=2)

        gen_btn.click(fn=_ui_generate,
                      inputs=[prompt_box, sys_box, neg_box, steps_s, cfg_s, w_s, h_s,
                              seed_n, batch_s, llm_c, delllm_c, lora_c, gd_c, loc_t],
                      outputs=[gal, status_box, final_box])
        stop_btn.click(fn=request_stop, outputs=[status_box])
        demo.queue(max_threads=2)
    demo.launch(share=share)
    return demo
