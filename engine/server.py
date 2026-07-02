"""Saglitz Photo Studio — local image-generation engine (mflux / Apple MLX).

A small FastAPI server that loads one or more image models (FLUX.1 and
Z-Image-Turbo) into memory and exposes them over HTTP so both the native
SwiftUI app and an MCP server (for Claude Code) can request images. Fully
local, no external API cost.

Multiple models can be held in RAM at once (e.g. FLUX schnell ~12 GB 8-bit +
Z-Image-Turbo ~6 GB 4-bit). The default model is preloaded at startup; any
other registered model is loaded lazily on first use and then cached.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from PIL import Image, ImageDraw, ImageFilter, ImageFont
# Reject decompression-bomb images (~7k×7k) so a crafted huge file can't exhaust
# RAM and kill the single engine worker (PIL raises DecompressionBombError > 2×).
Image.MAX_IMAGE_PIXELS = 50_000_000

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.models.z_image import ZImage

# Default wall-clock ceiling for post-processing subprocesses (ffmpeg/ffprobe/
# say/PIL helpers). Generous for big media, bounded so a hung helper can't pin a
# request or worker thread forever. Long jobs (training/import) pass their own.
_FFMPEG_TIMEOUT = 900
# Ceiling for a single model/voice download — larger than any real checkpoint
# (~24 GB) but bounded so a runaway/hostile stream can't fill the disk.
_MAX_DOWNLOAD_BYTES = 60 * 1024 ** 3


def _run(cmd, *, timeout=_FFMPEG_TIMEOUT, **kw):
    """subprocess.run with a mandatory timeout — no helper subprocess may hang a
    request/worker indefinitely. Raises subprocess.TimeoutExpired on overrun."""
    return subprocess.run(cmd, timeout=timeout, **kw)


# --- paths & config ------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent          # .../SaglitzPhotoStudio
OUTPUTS = ROOT / "outputs"                              # legacy flat output dir
HISTORY_FILE = OUTPUTS / "history.json"                 # global aggregate history
PROJECTS = ROOT / "projects"                            # new: per-project folders
DT_MODELS_DIR = ROOT / "dt-models"                      # Draw Things .ckpt weights
DT_LORAS_DIR = ROOT / "dt-loras"                        # user-imported LoRA files
OUTPUTS.mkdir(exist_ok=True)
PROJECTS.mkdir(exist_ok=True)
DT_MODELS_DIR.mkdir(exist_ok=True)
DT_LORAS_DIR.mkdir(exist_ok=True)

# User-chosen output root (where generated images are saved). Persisted so it
# survives restarts; the app keeps it in sync. All per-project writes/reads use
# the PROJECTS global, so reassigning it relocates outputs live.
_OUTPUT_CFG = ROOT / ".output_root"


def _load_output_root() -> None:
    global PROJECTS
    try:
        if _OUTPUT_CFG.exists():
            p = Path(_OUTPUT_CFG.read_text().strip())
            if str(p):
                p.mkdir(parents=True, exist_ok=True)
                PROJECTS = p
    except Exception:
        pass


_load_output_root()

# Draw Things LoRA `version` strings by base-model family (needed in config-json
# for custom LoRAs not registered in custom_lora.json). "flux1" is the only one
# the CLI docs confirm; the rest are best-effort and may need adjusting.
_LORA_VERSION_BY_FAMILY = {
    "FLUX.1": "flux1",
    "FLUX.2": "flux2",
    "SDXL": "sdxl_base_v0.9",
    "Z-Image": "z_image",
    "Qwen": "qwen_image",
}
_ALLOWED_LORA_EXT = {".ckpt", ".safetensors"}

# A generation belongs to a project; images land in projects/<project>/. When a
# caller (e.g. Claude via MCP) doesn't name one, it goes to DEFAULT_PROJECT.
# Legacy images (saved before projects existed, url=/outputs/...) are surfaced
# under ARCHIVE_PROJECT without moving any files on disk.
DEFAULT_PROJECT = "Genel"     # persisted folder name — matches the app + existing installs; don't rename without migration
ARCHIVE_PROJECT = "Arşiv"     # persisted folder name — see above

# FLUX weights live as locally-downloaded folders so we never re-download.
_FLUX_DIRS = {
    "schnell": Path.home() / "Documents/huggingface/models/black-forest-labs/FLUX.1-schnell",
    "dev": Path.home() / "Documents/huggingface/models/black-forest-labs/FLUX.1-dev",
}
# Z-Image-Turbo: pre-quantized mflux weights (resolved from the HF cache once
# downloaded). 4-bit ~5.9 GB (default) or 8-bit ~11 GB — override with env.
ZIMAGE_TURBO_PATH = os.environ.get(
    "SAGLITZ_ZIMAGE_PATH", "filipstrand/Z-Image-Turbo-mflux-4bit"
)

# QUANTIZE applies to FLUX load (None = float16; 8/4 = smaller + faster).
# Z-Image-Turbo weights above are already quantized, so no quantize is applied.
_q = os.environ.get("SAGLITZ_QUANTIZE", "").strip()
QUANTIZE: Optional[int] = int(_q) if _q.isdigit() else None


def _build_flux(name: str) -> Flux1:
    path = os.environ.get("SAGLITZ_MODEL_PATH") or str(_FLUX_DIRS[name])
    return Flux1(
        model_config=ModelConfig.from_name(model_name=path, base_model=name),
        quantize=QUANTIZE,
    )


def _build_zimage_turbo(name: str) -> ZImage:
    return ZImage(
        model_config=ModelConfig.z_image_turbo(),
        model_path=ZIMAGE_TURBO_PATH,
    )


def _mflux_lora_args(loras: list[dict]) -> tuple[list[str], list[float]]:
    """Resolve imported LoRA files in dt-loras/ to absolute paths + scales."""
    paths, scales = [], []
    for l in loras or []:
        fn = os.path.basename(str(l.get("file", "")))
        p = DT_LORAS_DIR / fn
        if not p.is_file():
            raise HTTPException(status_code=400, detail=f"LoRA not found: {fn}")
        paths.append(str(p))
        try:
            scales.append(float(l.get("weight", 1.0)))
        except (TypeError, ValueError):
            scales.append(1.0)
    return paths, scales


def _build_flux_with_loras(name: str, paths: list[str], scales: list[float]) -> Flux1:
    """Fresh FLUX model with LoRAs baked in (mflux applies LoRAs at build time)."""
    path = os.environ.get("SAGLITZ_MODEL_PATH") or str(_FLUX_DIRS[name])
    return Flux1(
        model_config=ModelConfig.from_name(model_name=path, base_model=name),
        quantize=QUANTIZE, lora_paths=paths, lora_scales=scales,
    )


# Model registry. Each spec knows how to build the mflux model and its defaults.
#   default_steps  — diffusion steps when the request omits them
#   use_guidance   — pass `guidance` (CFG)? distilled models (schnell/turbo) skip it
#   negative       — does the model accept a negative_prompt?
MODEL_SPECS: dict[str, dict[str, Any]] = {
    "schnell": {
        "label": "FLUX.1 schnell", "build": _build_flux,
        "default_steps": 4, "use_guidance": False, "negative": False,
    },
    "dev": {
        "label": "FLUX.1 dev", "build": _build_flux,
        "default_steps": 20, "use_guidance": True, "negative": False,
    },
    "z-image-turbo": {
        "label": "Z-Image Turbo", "build": _build_zimage_turbo,
        "default_steps": 9, "use_guidance": False, "negative": True,
    },
}

# Draw Things models, generated via the `draw-things-cli` subprocess (Phase 1).
# Maps our model id -> DT .ckpt id + defaults. The full catalog (download/delete)
# arrives with the model manager; here we expose the curated, already-usable set.
# Curated catalog of Draw Things image models for the model manager (Phase 2).
# Models are referenced by their .ckpt id everywhere; generation uses it once
# downloaded.
DT_CATALOG: list[dict[str, Any]] = [
    {"ckpt": "flux_2_klein_4b_q6p.ckpt", "label": "FLUX.2 Klein 4B", "family": "FLUX.2",
     "default_steps": 4, "negative": True, "use_guidance": True},
    {"ckpt": "z_image_turbo_1.0_q6p.ckpt", "label": "Z-Image Turbo (DT)", "family": "Z-Image",
     "default_steps": 9, "negative": True, "use_guidance": False},
    {"ckpt": "qwen_image_2512_q6p.ckpt", "label": "Qwen Image 2512", "family": "Qwen",
     "default_steps": 30, "negative": True, "use_guidance": True},
    {"ckpt": "flux_1_schnell_q8p.ckpt", "label": "FLUX.1 schnell (DT)", "family": "FLUX.1",
     "default_steps": 4, "negative": False, "use_guidance": False},
    {"ckpt": "sd_xl_base_1.0_q6p_q8p.ckpt", "label": "SDXL Base 1.0", "family": "SDXL",
     "default_steps": 30, "negative": True, "use_guidance": True},
    {"ckpt": "flux_1_canny_dev_q5p.ckpt", "label": "FLUX.1 Canny · ControlNet", "family": "FLUX.1",
     "default_steps": 20, "negative": False, "use_guidance": True, "control": "canny"},
]
_DT_BY_CKPT = {m["ckpt"]: m for m in DT_CATALOG}


def _dt_downloaded(ckpt: str) -> bool:
    return (DT_MODELS_DIR / ckpt).exists()


def _dt_resolve(name: str) -> Optional[dict[str, Any]]:
    """Resolve a DT catalog entry by its ckpt id. Curated entries carry tuned
    defaults; any other downloaded .ckpt gets a generic entry so the full Draw
    Things catalog is generatable (the CLI applies the model's recommended
    settings)."""
    if name in _DT_BY_CKPT:
        return _DT_BY_CKPT[name]
    if name.endswith(".ckpt") and _dt_downloaded(name):
        return {"ckpt": name, "label": name, "family": "Community",
                "default_steps": 20, "negative": True, "use_guidance": True}
    return None


# --- full Draw Things catalog (parsed from `draw-things-cli models list`) ------
_dt_catalog_cache: Optional[list[dict]] = None
_dt_catalog_lock = threading.Lock()


def _parse_models_list(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith(("MODEL", "---", "Models directory")):
            continue
        parts = re.split(r"\s{2,}", s)        # columns are padded with 2+ spaces
        if len(parts) < 4 or not parts[0].endswith(".ckpt"):
            continue
        hf = parts[4] if len(parts) > 4 else ""
        rows.append({
            "ckpt": parts[0], "name": parts[1],
            "source": parts[2].lower(),
            "downloaded": parts[3].strip().lower() == "yes",
            "hf": "" if hf in ("-", "") else hf,
        })
    return rows


def _dt_full_catalog(refresh: bool = False) -> list[dict]:
    global _dt_catalog_cache
    with _dt_catalog_lock:
        if _dt_catalog_cache is not None and not refresh:
            return _dt_catalog_cache
    rows: list[dict] = []
    try:
        proc = _run(
            [_DT_CLI, "models", "list", "--models-dir", str(DT_MODELS_DIR)],
            capture_output=True, text=True, timeout=90)
        rows = _parse_models_list(proc.stdout)
    except Exception:
        rows = []
    # Make sure curated models are present even if the listing fails.
    have = {r["ckpt"] for r in rows}
    for m in DT_CATALOG:
        if m["ckpt"] not in have:
            rows.append({"ckpt": m["ckpt"], "name": m["label"],
                         "source": "official", "downloaded": _dt_downloaded(m["ckpt"]), "hf": ""})
            have.add(m["ckpt"])
    # Surface the user's own imported models (real .ckpt files in dt-models/ that
    # the catalog doesn't list), skipping companion encoders and tiny stubs.
    for im in _imported_models(have):
        rows.append(im)
    with _dt_catalog_lock:
        _dt_catalog_cache = rows
    return rows


# Companion artifacts that aren't standalone models (text encoders / VAEs).
_COMPANION_HINTS = ("t5_", "clip_", "umt5", "_vae", "vae_", "text_encoder",
                    "qwen_3", "_te_", "autoencoder", ".partial")


def _imported_models(have: set) -> list[dict]:
    out = []
    for p in DT_MODELS_DIR.glob("*.ckpt"):
        n = p.name
        if n in have:
            continue
        low = n.lower()
        if any(h in low for h in _COMPANION_HINTS):
            continue
        has_td = (DT_MODELS_DIR / (n + "-tensordata")).exists()
        try:
            big = p.stat().st_size > 50_000_000
        except OSError:
            big = False
        if not (big or has_td):          # skip broken/partial stubs
            continue
        out.append({"ckpt": n, "name": n.replace(".ckpt", "").replace("_", " "),
                    "source": "imported", "downloaded": True, "hf": ""})
    return out


# Map a Civitai `baseModel` string to our DT (base label, version) pair.
def _base_from_hint(hint: str) -> dict:
    h = (hint or "").lower()
    table = [
        (("flux.2", "flux 2", "flux2"), "FLUX.2", "flux2"),
        (("flux",), "FLUX.1", "flux1"),
        (("pony", "sdxl", "sd xl", "illustrious", "noobai"), "SDXL", "sdxl_base_v0.9"),
        (("z-image", "z image", "z_image"), "Z-Image", "z_image"),
        (("qwen",), "Qwen", "qwen_image"),
        (("sd 2", "sd2", "2.1", "768"), "SD 2.x", "sd_v2"),
        (("sd 1", "sd1", "1.5"), "SD 1.5", "sd_v1"),
    ]
    for keys, label, version in table:
        if any(k in h for k in keys):
            return {"base": label, "version": version}
    return {"base": None, "version": None}


def _drop_custom_json_entry(ckpt: str) -> None:
    """Remove a model's entry from dt-models/custom.json (the DT model spec store)."""
    cj = DT_MODELS_DIR / "custom.json"
    if not cj.exists():
        return
    try:
        entries = json.loads(cj.read_text())
        kept = [e for e in entries if e.get("file") != ckpt]
        if len(kept) != len(entries):
            cj.write_text(json.dumps(kept, indent=2))
    except Exception:
        pass


def _remove_dead_stub(ckpt: str) -> None:
    """Delete a freshly-produced but unusable model stub (failed import)."""
    try:
        (DT_MODELS_DIR / ckpt).unlink(missing_ok=True)
        (DT_MODELS_DIR / (ckpt + "-tensordata")).unlink(missing_ok=True)
    except Exception:
        pass
    _drop_custom_json_entry(ckpt)
    _dt_full_catalog(refresh=True)


def _cleanup_models_dir() -> None:
    """One-shot hygiene at startup: drop empty nested junk dirs and tiny orphan
    .ckpt stubs (failed imports / a LoRA accidentally imported as a checkpoint)."""
    # 1) Empty nested path junk like dt-models/Users/.../dt-loras
    junk = DT_MODELS_DIR / "Users"
    if junk.exists():
        try:
            shutil.rmtree(junk)
        except Exception:
            pass
    # 2) Orphan tiny stubs: <1MB, no -tensordata, not a companion, not curated/custom.
    keep = set(_DT_BY_CKPT)
    cj = DT_MODELS_DIR / "custom.json"
    if cj.exists():
        try:
            keep |= {e.get("file") for e in json.loads(cj.read_text())}
        except Exception:
            pass
    for p in DT_MODELS_DIR.glob("*.ckpt"):
        n, low = p.name, p.name.lower()
        if n in keep or any(h in low for h in _COMPANION_HINTS):
            continue
        if (DT_MODELS_DIR / (n + "-tensordata")).exists():
            continue
        try:
            if p.stat().st_size < 1_000_000:
                p.unlink(missing_ok=True)
        except OSError:
            pass

DEFAULT_MODEL = os.environ.get("SAGLITZ_BASE_MODEL", "schnell")
if DEFAULT_MODEL not in MODEL_SPECS:
    DEFAULT_MODEL = "schnell"
DEFAULT_STEPS = MODEL_SPECS[DEFAULT_MODEL]["default_steps"]

# --- model state ---------------------------------------------------------
from collections import OrderedDict

_models: "OrderedDict[str, Any]" = OrderedDict()   # name -> loaded mflux model (LRU)
_load_errors: dict[str, str] = {}       # name -> last load error
# Keep at most this many big diffusion models resident — they are 6–12 GB each,
# so without a cap a session that touches several models swaps the machine to a
# crawl. LRU-evict the rest. Overridable for big-RAM setups.
_MAX_RESIDENT_MODELS = max(1, int(os.environ.get("SAGLITZ_MAX_MODELS", "1")))


def _free_mlx() -> None:
    """Release evicted-model memory back to the OS (Python gc + MLX's buffer cache)."""
    import gc
    gc.collect()
    try:
        import mlx.core as mx
        (getattr(mx, "clear_cache", None) or getattr(getattr(mx, "metal", None), "clear_cache", lambda: None))()
    except Exception:
        pass


# Idle memory reclaim: when no generation has happened for a while, drop ALL
# cached models (diffusion + the auxiliary torch/MLX ones) so an idle app isn't
# holding GBs on the user's Mac. They reload on the next request.
_last_activity = time.time()
_IDLE_FREE_SECS = int(os.environ.get("SAGLITZ_IDLE_FREE_SECS", "600"))   # 10 min; 0 disables


def _touch_activity() -> None:
    global _last_activity
    _last_activity = time.time()


def _free_all_models() -> None:
    """Drop every cached model. MUST run on the engine thread (MLX thread-locality)."""
    global _blip, _audioldm, _stableaudio, _rembg_session
    _models.clear()
    _blip = None
    _audioldm = None
    _stableaudio = None
    _rembg_session = None
    try:
        _kokoro_pipes.clear()
    except Exception:
        pass
    _free_mlx()
    print("↓ idle: released all models, memory reclaimed")


def _idle_reaper() -> None:
    while True:
        time.sleep(60)
        if _IDLE_FREE_SECS <= 0 or _gen_lock.locked():
            continue
        if (time.time() - _last_activity) > _IDLE_FREE_SECS and (
                _models or _blip or _audioldm or _stableaudio or _rembg_session):
            try:
                _engine.submit(_free_all_models).result(timeout=180)
            except Exception:
                pass


threading.Thread(target=_idle_reaper, daemon=True).start()
_state: dict[str, Any] = {"status": "loading", "detail": "Loading model into memory…",
                          "model": DEFAULT_MODEL, "quantize": QUANTIZE}
_gen_lock = threading.Lock()
# MLX GPU streams are THREAD-LOCAL, so every MLX op (model construction AND
# generation) must run on one consistent thread. A single-worker executor
# guarantees that, even though uvicorn dispatches requests across its own pool.
_engine = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-engine")


def _mlx(fn: Callable, *args, **kwargs):
    """Run an MLX/MPS model call on the single engine thread. MLX GPU streams are
    thread-local, so EVERY model op (audio TTS/clone/transcribe, SFX, rembg, BLIP)
    must share this one thread — running them on uvicorn worker threads can crash
    or corrupt the GPU stream when they overlap a diffusion job. Serializes with
    image/video generation, which is the correct, safe behaviour."""
    _touch_activity()
    return _engine.submit(fn, *args, **kwargs).result()


def _ensure_model(name: str) -> Any:
    """Load `name` if not already cached. MUST run on the MLX engine thread.
    Keeps only _MAX_RESIDENT_MODELS resident, LRU-evicting the rest so memory
    stays bounded no matter how many models a session touches."""
    _touch_activity()
    if name in _models:
        _models.move_to_end(name)                 # mark most-recently-used
        return _models[name]
    while len(_models) >= _MAX_RESIDENT_MODELS and _models:
        old, m = _models.popitem(last=False)      # drop least-recently-used
        del m
        _free_mlx()
        print(f"↓ unloaded '{old}' to free memory")
    spec = MODEL_SPECS[name]
    t0 = time.time()
    model = spec["build"](name)
    _models[name] = model
    _load_errors.pop(name, None)
    print(f"✓ '{name}' loaded ({time.time() - t0:.0f}s)")
    return model


def _load_default_model() -> None:
    try:
        t0 = time.time()
        _ensure_model(DEFAULT_MODEL)
        _state.update(status="ready", detail=f"Ready (loaded in {time.time() - t0:.0f}s)")
    except Exception as exc:
        _load_errors[DEFAULT_MODEL] = f"{type(exc).__name__}: {exc}"
        _state.update(status="error", detail=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()


# --- projects & history --------------------------------------------------
def _safe_project(name: Optional[str]) -> str:
    """Turn an arbitrary project name into a safe folder name (no traversal).

    Keeps unicode letters/numbers (so Turkish names work), spaces, dash and
    underscore; strips path separators and `..`. Falls back to DEFAULT_PROJECT.
    """
    name = (name or "").strip()
    name = name.replace("/", "-").replace("\\", "-")
    name = re.sub(r"\.{2,}", ".", name).strip(". ")
    name = re.sub(r"[^\w\s\-]", "", name, flags=re.UNICODE).strip()
    return name or DEFAULT_PROJECT


def _project_dir(project: str) -> Path:
    d = PROJECTS / project
    d.mkdir(parents=True, exist_ok=True)
    return d


# All history.json / manifest.json read-modify-write goes through this lock and
# an atomic temp-file replace. Without it, a generation's append could interleave
# with an unlocked delete/upscale write, and a torn write would leave invalid JSON
# that _read_* silently treats as empty — wiping the user's history on next write.
_history_lock = threading.Lock()


def _atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    os.replace(tmp, path)


_HISTORY_MAX = 5000        # cap global history so it can't grow without bound


def _read_history() -> list[dict]:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())
        except json.JSONDecodeError:
            # Don't silently return [] — the next append would then overwrite and
            # WIPE history. Set the corrupt file aside so it can be recovered.
            try:
                HISTORY_FILE.rename(HISTORY_FILE.with_suffix(".corrupt.json"))
            except OSError:
                pass
            return []
    return []


def _append_history(entry: dict) -> None:
    with _history_lock:
        hist = _read_history()
        hist.insert(0, entry)
        del hist[_HISTORY_MAX:]        # keep only the most recent, bounded
        _atomic_write_json(HISTORY_FILE, hist)


def _append_app_manifest(project: str, name: str, fname: str, gen_id: str,
                         prompt: str, negative_prompt: Optional[str],
                         width: int, height: int, steps: int,
                         guidance: Optional[float], seed: int, elapsed: float) -> None:
    """Write a per-project manifest.json in the SwiftUI app's StoredGeneration
    JSON shape, so the native app's gallery shows engine/MCP generations too
    (one shared `projects/<project>/` folder for app + engine + Claude)."""
    mf = _project_dir(project) / "manifest.json"
    with _history_lock:
        try:
            rows = json.loads(mf.read_text()) if mf.exists() else []
        except json.JSONDecodeError:
            rows = []
        rows.insert(0, {
            "id": gen_id, "file": fname, "project": project,
            "prompt": prompt, "negativePrompt": negative_prompt,
            "width": width, "height": height, "steps": steps,
            "guidance": guidance, "seed": seed,
            "modelID": name, "providerID": "local", "costUSD": None,
            "elapsed": round(elapsed, 2),
            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        _atomic_write_json(mf, rows)


def _entry_project(entry: dict) -> str:
    """Project an entry belongs to; legacy (project-less) entries -> ARCHIVE."""
    return entry.get("project") or ARCHIVE_PROJECT


def _list_projects() -> list[str]:
    names = {p.name for p in PROJECTS.iterdir() if p.is_dir()} if PROJECTS.exists() else set()
    names |= {_entry_project(e) for e in _read_history()}
    names.add(DEFAULT_PROJECT)
    return sorted(names)


# --- API models ----------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str
    model: Optional[str] = None          # registry key; None -> DEFAULT_MODEL
    project: Optional[str] = None        # folder to save into; None -> DEFAULT_PROJECT
    width: int = 1024
    height: int = 1024
    steps: Optional[int] = None
    guidance: float = 3.5
    seed: Optional[int] = None
    negative_prompt: Optional[str] = None  # only used by models that support it
    image_path: Optional[str] = None       # img2img: local path of the init image
    image_strength: Optional[float] = None  # img2img strength 0..1
    loras: Optional[list[dict]] = None      # DT only: [{"file": name, "weight": w}]
    scheduler: Optional[str] = None         # mflux only: linear | flow_match_euler_discrete


def _round16(v: int) -> int:
    # Round to a multiple of 64 (FLUX/FLUX.2 reject non-/64 dims; 64 is also valid
    # for SD/SDXL). Ceiling at 4096 so the 4K hi-res-fix path can actually reach a
    # true 4K long edge instead of the old 2048 ceiling.
    v = max(256, min(4096, int(v)))
    return v - (v % 64)


_ALLOWED_IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}


_ALLOWED_PATH_ROOTS = [os.path.realpath(os.path.expanduser("~")), "/Volumes"]


def _is_allowed_path(p: str) -> bool:
    """Confine a user-supplied filesystem path to the home folder or an external
    volume. The local API has no auth, so this blocks a malicious/injected caller
    from reading system files or another user's data via these path endpoints."""
    try:
        rp = os.path.realpath(p)
    except Exception:
        return False
    return any(rp == r or rp.startswith(r + os.sep) for r in _ALLOWED_PATH_ROOTS)


def _safe_image_path(p: str) -> str:
    """Validate an img2img init path: must be an existing image file inside an
    allowed root (guards path-traversal / arbitrary file reads via the local API)."""
    real = os.path.realpath(p)
    if not os.path.isfile(real):
        raise HTTPException(status_code=400, detail="image_path: file not found")
    if os.path.splitext(real)[1].lower() not in _ALLOWED_IMG_EXT:
        raise HTTPException(status_code=400, detail="image_path: unsupported image type")
    if not _is_allowed_path(real):
        raise HTTPException(status_code=400, detail="image_path: outside your home / external drives")
    return real


def _safe_loras(loras: Optional[list[dict]], reg: dict) -> Optional[list[dict]]:
    """Validate LoRA refs against the dt-loras/ folder and attach the base
    model's DT `version` string. Returns config-json-ready dicts, or None."""
    if not loras:
        return None
    fallback_version = _LORA_VERSION_BY_FAMILY.get(reg.get("family"))
    out: list[dict] = []
    for ref in loras:
        name = os.path.basename(str(ref.get("file", "")))  # strip any path parts
        if not name:
            continue
        path = (DT_LORAS_DIR / name)
        if not path.is_file():
            raise HTTPException(status_code=400, detail=f"LoRA not found: {name}")
        if path.suffix.lower() not in _ALLOWED_LORA_EXT:
            raise HTTPException(status_code=400, detail=f"Unsupported LoRA type: {name}")
        
        # Ensure symlink in DT_MODELS_DIR so draw-things-cli can find it relative to models-dir
        link_path = DT_MODELS_DIR / name
        if not link_path.exists():
            try:
                os.symlink(str(path), str(link_path))
            except Exception:
                try:
                    shutil.copy2(str(path), str(link_path))
                except Exception:
                    pass

        try:
            weight = float(ref.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        # Prefer the LoRA's own detected version (from its safetensors metadata),
        # falling back to the selected base model's family mapping.
        version = _lora_info(name).get("version") or fallback_version
        entry = {"file": name, "weight": round(weight, 3)}
        if version:
            entry["version"] = version
        out.append(entry)
    return out or None


# --- app -----------------------------------------------------------------
app = FastAPI(title="Saglitz Photo Studio Engine")


@app.on_event("startup")
def _startup() -> None:
    _cleanup_models_dir()                # drop nested junk + dead stubs
    _engine.submit(_load_default_model)  # build default model on the MLX thread


@app.get("/api/status")
def status() -> dict:
    # Keep legacy keys (status/detail/model/quantize) for the SwiftUI app, plus
    # multi-model detail: which models are loaded and any load errors.
    return {
        **_state,
        "default_model": DEFAULT_MODEL,
        "loaded": sorted(_models.keys()),
        "errors": _load_errors,
    }


@app.get("/api/config")
def config() -> dict:
    return {
        "model": DEFAULT_MODEL,
        "quantize": QUANTIZE,
        "default_steps": DEFAULT_STEPS,
        "default_project": DEFAULT_PROJECT,
        "projects": _list_projects(),
        "models": [
            {"id": k, "label": v["label"], "default_steps": v["default_steps"],
             "negative": v["negative"]}
            for k, v in MODEL_SPECS.items()
        ] + [
            {"id": m["ckpt"],
             "label": _CURATED_META[m["ckpt"]]["label"] if m["ckpt"] in _CURATED_META else m["name"],
             "default_steps": _CURATED_META[m["ckpt"]]["default_steps"] if m["ckpt"] in _CURATED_META else 20,
             "negative": True}
            for m in _dt_full_catalog() if _dt_downloaded(m["ckpt"])
        ],
        "resolutions": [
            {"label": "Square 1:1", "width": 1024, "height": 1024},
            {"label": "Landscape 3:2", "width": 1216, "height": 832},
            {"label": "Portrait 2:3", "width": 832, "height": 1216},
            {"label": "Wide 16:9", "width": 1344, "height": 768},
            {"label": "Phone 9:16", "width": 768, "height": 1344},
        ],
        "output_root": str(PROJECTS),
    }


class OutputRef(BaseModel):
    path: str


@app.get("/api/config/output")
def get_output() -> dict:
    return {"path": str(PROJECTS)}


@app.post("/api/config/output")
def set_output(ref: OutputRef) -> dict:
    global PROJECTS
    p = Path(os.path.realpath(ref.path))
    if not _is_allowed_path(str(p)):
        raise HTTPException(status_code=400,
                            detail="Output folder must be in your home folder or an external drive.")
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Folder unavailable: {exc}")
    PROJECTS = p
    try:
        _OUTPUT_CFG.write_text(str(p))
    except Exception:
        pass
    return {"path": str(PROJECTS)}


# --- BYOK cloud keys (fal.ai video + ElevenLabs TTS) -----------------------------
_BYOK_CFG = ROOT / ".byok.json"


def _load_keys() -> dict:
    try:
        return json.loads(_BYOK_CFG.read_text())
    except Exception:
        return {}


def _save_keys(d: dict) -> None:
    # Create at 0600 from the first byte — no world-readable race window for secrets.
    try:
        data = json.dumps(d).encode()
        fd = os.open(str(_BYOK_CFG), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)
        os.chmod(_BYOK_CFG, 0o600)   # tighten if the file pre-existed with looser perms
    except Exception:
        pass


class KeysRequest(BaseModel):
    fal: Optional[str] = None
    elevenlabs: Optional[str] = None
    hf: Optional[str] = None          # Hugging Face token (Stable Audio Open, gated)


@app.get("/api/config/keys")
def get_keys() -> dict:
    """Report only PRESENCE of each key, never the secret itself."""
    k = _load_keys()
    return {"fal": bool(k.get("fal")), "elevenlabs": bool(k.get("elevenlabs")),
            "hf": bool(k.get("hf"))}


@app.post("/api/config/keys")
def set_keys(req: KeysRequest) -> dict:
    k = _load_keys()
    # Empty string clears a key; None leaves it unchanged.
    if req.fal is not None:
        k["fal"] = req.fal.strip()
    if req.elevenlabs is not None:
        k["elevenlabs"] = req.elevenlabs.strip()
    if req.hf is not None:
        k["hf"] = req.hf.strip()
    _save_keys({kk: vv for kk, vv in k.items() if vv})
    return get_keys()


@app.get("/api/projects")
def projects() -> list[str]:
    """All known project names (folders + any seen in history)."""
    return _list_projects()


@app.get("/api/history")
def history(project: Optional[str] = None) -> list[dict]:
    """Generation history, newest first. Pass ?project=Name to filter; omit for all."""
    hist = _read_history()
    if project:
        hist = [e for e in hist if _entry_project(e) == project]
    return hist


# --- Draw Things model manager (Phase 2) --------------------------------------
_dt_downloads: dict[str, dict[str, Any]] = {}   # ckpt -> {status, error}
_dt_dl_lock = threading.Lock()
_dt_dl_procs: dict[str, "subprocess.Popen"] = {}   # ckpt -> running proc (for cancel)
_dt_dl_cancel: set[str] = set()                    # ckpts the user asked to cancel


def _dt_want_cancel(ckpt: str) -> bool:
    with _dt_dl_lock:
        return ckpt in _dt_dl_cancel


def _dt_finish_cancel(ckpt: str, *partials) -> None:
    """Wipe the partials and clear all state for a cancelled download."""
    for p in partials:
        try:
            p.unlink()
        except OSError:
            pass
    with _dt_dl_lock:
        _dt_downloads.pop(ckpt, None)
        _dt_dl_cancel.discard(ckpt)
        _dt_dl_procs.pop(ckpt, None)


def _dt_partial_mb(ckpt: str) -> Optional[int]:
    p = DT_MODELS_DIR / (ckpt + ".partial")
    if p.exists():
        return round(p.stat().st_size / 1e6)
    if _dt_downloaded(ckpt):
        return round((DT_MODELS_DIR / ckpt).stat().st_size / 1e6)
    return None


def _dt_download_worker(ckpt: str) -> None:
    """Download with stall recovery. `models ensure` resumes from the .partial on
    disk; when it hangs (a known Draw Things issue) we kill it and re-run. CRITICAL:
    if a resume makes NO progress, the .partial is corrupt and resuming will hang
    forever on it — so we wipe the .partial and start fresh. Up to 10 attempts."""
    last_err = ""
    partial = DT_MODELS_DIR / (ckpt + ".partial")
    pmap = DT_MODELS_DIR / (ckpt + ".partial.map")
    for attempt in range(10):
        if _dt_want_cancel(ckpt):
            _dt_finish_cancel(ckpt, partial, pmap)
            return
        start_size = _dt_partial_mb(ckpt) or 0
        try:
            proc = subprocess.Popen(
                [_DT_CLI, "models", "ensure", "--models-dir", str(DT_MODELS_DIR), "--model", ckpt],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with _dt_dl_lock:
                _dt_dl_procs[ckpt] = proc
        except Exception as exc:
            with _dt_dl_lock:
                _dt_downloads[ckpt] = {"status": "error", "error": str(exc)}
            return
        last_size, stalls = start_size, 0
        while proc.poll() is None:
            time.sleep(10)
            if _dt_want_cancel(ckpt):
                proc.kill()
                _dt_finish_cancel(ckpt, partial, pmap)
                return
            sz = _dt_partial_mb(ckpt) or 0
            if sz > last_size:
                last_size, stalls = sz, 0
            else:
                stalls += 1
            if stalls >= 9:                        # ~90s with no growth -> stalled
                last_err = "Download stalled; retrying."
                proc.kill()
                break
        try:
            err_out = proc.stderr.read().decode(errors="ignore") if proc.stderr else ""
        except Exception:
            err_out = ""
        try:
            proc.kill()
        except Exception:
            pass
        if _dt_downloaded(ckpt):                    # finished (even if rc was odd)
            with _dt_dl_lock:
                _dt_downloads[ckpt] = {"status": "done"}
            return
        # Resume made no real progress → the .partial is corrupt. Wipe it so the
        # next attempt downloads from byte 0 (this is what fixes the dead-stall).
        end_size = _dt_partial_mb(ckpt) or 0
        if end_size <= start_size + 1 and attempt < 9:
            for p in (partial, pmap):
                try:
                    p.unlink()
                except OSError:
                    pass
            last_err = "Corrupt partial cleared; downloading from scratch."
        elif err_out.strip():
            last_err = err_out[-300:]
        with _dt_dl_lock:                            # keep UI in "downloading" while we retry
            _dt_downloads[ckpt] = {"status": "downloading",
                                   "error": f"resuming ({attempt + 1}/10)…"}
        time.sleep(2)
    with _dt_dl_lock:
        _dt_downloads[ckpt] = {"status": "error", "error": last_err or "Download failed."}


_CURATED_META = {m["ckpt"]: m for m in DT_CATALOG}


@app.get("/api/dt/models")
def dt_models(refresh: bool = False) -> list[dict]:
    """Full Draw Things model catalog (official + community) with download status."""
    out = []
    for m in _dt_full_catalog(refresh=refresh):
        ckpt = m["ckpt"]
        with _dt_dl_lock:
            dl = dict(_dt_downloads.get(ckpt, {}))
        meta = _CURATED_META.get(ckpt)
        out.append({
            "ckpt": ckpt,
            "label": meta["label"] if meta else m["name"],
            "family": meta["family"] if meta else (
                "Imported" if m["source"] == "imported"
                else "Official" if m["source"] == "official" else "Community"),
            "downloaded": _dt_downloaded(ckpt),
            "status": dl.get("status"),          # downloading | done | error | None
            "error": dl.get("error"),
            "size_mb": _dt_partial_mb(ckpt),
        })
    return out


class DTModelRef(BaseModel):
    ckpt: str


def _dt_catalog_ckpts() -> set:
    return {m["ckpt"] for m in _dt_full_catalog()}


@app.post("/api/dt/models/download")
def dt_download(ref: DTModelRef) -> dict:
    if ref.ckpt not in _dt_catalog_ckpts():
        raise HTTPException(status_code=400, detail="Unknown model.")
    if _dt_downloaded(ref.ckpt):
        return {"status": "done"}
    with _dt_dl_lock:
        if _dt_downloads.get(ref.ckpt, {}).get("status") == "downloading":
            return {"status": "downloading"}
        _dt_downloads[ref.ckpt] = {"status": "downloading"}
    threading.Thread(target=_dt_download_worker, args=(ref.ckpt,), daemon=True).start()
    return {"status": "downloading"}


@app.post("/api/dt/models/download/cancel")
def dt_download_cancel(ref: DTModelRef) -> dict:
    """Stop an in-flight model download and wipe its partial."""
    with _dt_dl_lock:
        _dt_dl_cancel.add(ref.ckpt)
        p = _dt_dl_procs.get(ref.ckpt)
    if p is not None:
        try:
            p.kill()
        except Exception:
            pass
    return {"status": "cancelling"}


@app.post("/api/dt/models/delete")
def dt_delete(ref: DTModelRef) -> dict:
    p = DT_MODELS_DIR / os.path.basename(ref.ckpt)
    try:
        p.unlink(missing_ok=True)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Delete failed: {exc}")
    with _dt_dl_lock:
        _dt_downloads.pop(ref.ckpt, None)
    _dt_full_catalog(refresh=True)
    return {"status": "deleted"}


# --- Import a user's own model file (.safetensors/.ckpt/...) ------------------
_dt_import: dict[str, Any] = {"status": "idle"}
_dt_import_lock = threading.Lock()
_IMPORT_EXT = {".safetensors", ".ckpt", ".pth", ".pt", ".bin", ".zip"}


class ImportRef(BaseModel):
    path: str
    name: Optional[str] = None
    trigger_word: Optional[str] = None


def _dt_import_worker(path: str, name: Optional[str], trigger: Optional[str]) -> None:
    try:
        cmd = [_DT_CLI, "models", "import", path, "--models-dir", str(DT_MODELS_DIR), "--replace"]
        if name:
            cmd += ["--name", name]
        if trigger:
            cmd += ["--trigger-word", trigger]
        proc = _run(cmd, capture_output=True, text=True, timeout=1800)   # large model convert can be slow
        ok = proc.returncode == 0
        log = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", (proc.stdout or "") + "\n" + (proc.stderr or "")).replace("\r", "")
        with _dt_import_lock:
            _dt_import.update(status="done" if ok else "error", log=log[-2000:])
        _dt_full_catalog(refresh=True)   # surface the new model in the catalog
    except Exception as exc:
        with _dt_import_lock:
            _dt_import.update(status="error", log=str(exc))


@app.post("/api/dt/models/import")
def dt_import(ref: ImportRef) -> dict:
    with _dt_import_lock:
        if _dt_import.get("status") == "running":
            raise HTTPException(status_code=409, detail="An import is already in progress.")
    src = os.path.realpath(ref.path)
    if not os.path.isfile(src):
        raise HTTPException(status_code=400, detail="File not found.")
    if not _is_allowed_path(src):
        raise HTTPException(status_code=400, detail="File must be in your home folder or an external drive.")
    if os.path.splitext(src)[1].lower() not in _IMPORT_EXT:
        raise HTTPException(status_code=400, detail="Unsupported format (.safetensors/.ckpt/…).")
    # Slugify the model name before it reaches `draw-things-cli --name --replace`
    # (mirrors training) so a crafted name can't traverse / clobber a curated ckpt.
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", ref.name).strip("_") if ref.name else None
    with _dt_import_lock:
        _dt_import.clear()
        _dt_import.update(status="running", file=os.path.basename(src))
    threading.Thread(target=_dt_import_worker,
                     args=(src, safe_name, ref.trigger_word), daemon=True).start()
    return {"status": "running"}


@app.get("/api/dt/models/import/status")
def dt_import_status() -> dict:
    with _dt_import_lock:
        return dict(_dt_import)


# --- LoRA library (import-based; the CLI has no LoRA download catalog) ---------
class LoRAImportRef(BaseModel):
    path: str            # source file the user picked (copied into dt-loras/)


class LoRARef(BaseModel):
    file: str            # file name within dt-loras/


def _read_safetensors_meta(path: str) -> dict:
    """Read a .safetensors file's __metadata__ JSON header (cheap; header only)."""
    try:
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            if n <= 0 or n > 50_000_000:
                return {}
            header = json.loads(f.read(n).decode("utf-8", "ignore"))
        return header.get("__metadata__", {}) or {}
    except Exception:
        return {}


def _detect_lora_base(meta: dict) -> dict:
    """Infer the base model family + DT `version` a LoRA targets, from metadata."""
    blob = (" ".join(str(v) for v in meta.values()) + " " + " ".join(meta.keys())).lower()
    checks = [
        (("flux.2", "flux2", "flux_2"), "FLUX.2", "flux2"),
        (("flux", "flux1", "flux.1"), "FLUX.1", "flux1"),
        (("sdxl", "xl-v1", "xl_base", "stable-diffusion-xl", "sd_xl"), "SDXL", "sdxl_base_v0.9"),
        (("z_image", "z-image"), "Z-Image", "z_image"),
        (("qwen",), "Qwen", "qwen_image"),
        (("sd_v2", "v2-1", "768-v"), "SD 2.x", "sd_v2"),
        (("sd_v1", "v1-5", "sd1.5", "sd_1.5"), "SD 1.5", "sd_v1"),
    ]
    for keys, label, version in checks:
        if any(k in blob for k in keys):
            return {"base": label, "version": version}
    return {"base": None, "version": None}


def _lora_meta_path(name: str) -> Path:
    return DT_LORAS_DIR / (name + ".meta.json")


def _lora_info(name: str) -> dict:
    mp = _lora_meta_path(name)
    if mp.exists():
        try:
            return json.loads(mp.read_text())
        except Exception:
            return {}
    return {}


@app.get("/api/dt/loras")
def dt_loras() -> list[dict]:
    """Imported LoRA files available to apply at generation, with detected base."""
    out = []
    for p in sorted(DT_LORAS_DIR.glob("*")):
        if p.suffix.lower() in _ALLOWED_LORA_EXT and p.is_file():
            info = _lora_info(p.name)
            out.append({"file": p.name, "size_mb": round(p.stat().st_size / 1e6),
                        "base": info.get("base"), "version": info.get("version")})
    return out


@app.post("/api/dt/loras/import")
def dt_lora_import(ref: LoRAImportRef) -> dict:
    src = os.path.realpath(ref.path)
    if not os.path.isfile(src):
        raise HTTPException(status_code=400, detail="File not found.")
    if not _is_allowed_path(src):
        raise HTTPException(status_code=400, detail="File must be in your home folder or an external drive.")
    if os.path.splitext(src)[1].lower() not in _ALLOWED_LORA_EXT:
        raise HTTPException(status_code=400, detail="Only .ckpt / .safetensors.")
    dest = DT_LORAS_DIR / os.path.basename(src)
    shutil.copy2(src, dest)
    # Auto-detect the base model from the safetensors metadata and remember it.
    info = {"base": None, "version": None}
    if dest.suffix.lower() == ".safetensors":
        info.update(_detect_lora_base(_read_safetensors_meta(str(dest))))
    try:
        _lora_meta_path(dest.name).write_text(json.dumps(info))
    except Exception:
        pass
    return {"status": "imported", "file": dest.name, "base": info.get("base")}


@app.post("/api/dt/loras/delete")
def dt_lora_delete(ref: LoRARef) -> dict:
    name = os.path.basename(ref.file)
    for target in (DT_LORAS_DIR / name, DT_MODELS_DIR / name, _lora_meta_path(name)):
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass
    return {"status": "deleted"}


# --- LoRA training (draw-things-cli train lora) -------------------------------
_train: dict[str, Any] = {"status": "idle"}     # idle | running | done | error
_train_lock = threading.Lock()
_train_proc: Optional["subprocess.Popen"] = None   # running train subprocess (for cancel)
_train_cancel = False


class TrainRequest(BaseModel):
    base: str                       # base model ckpt (a downloaded DT model)
    dataset: str                    # directory of images (+ optional .txt captions)
    steps: int = 200
    rank: int = 16
    name: str = "my_lora"
    dry_run: bool = False


def _train_worker(cmd: list[str], out_prefix: str, dry: bool) -> None:
    global _train_proc, _train_cancel
    try:
        # Popen (not _run) so a 2-hour run can be cancelled by killing the process.
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        with _train_lock:
            _train_proc = proc
        try:
            out, err = proc.communicate(timeout=7200)
        except subprocess.TimeoutExpired:
            proc.kill(); out, err = proc.communicate()
        with _train_lock:
            _train_proc = None
        if _train_cancel:                       # user cancelled → not an error
            _train_cancel = False
            with _train_lock:
                _train.clear(); _train.update(status="idle")
            return
        ok = proc.returncode == 0
        log = ((out or "") + "\n" + (err or "")).strip()
        # Strip ANSI escapes / carriage returns so the JSON status stays clean.
        log = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", log).replace("\r", "")
        produced = None
        if ok and not dry:
            # Move the produced LoRA(s) into the LoRA library so they're usable.
            for p in sorted(DT_MODELS_DIR.glob(out_prefix + "*")):
                if p.suffix.lower() in _ALLOWED_LORA_EXT and p.is_file():
                    dest = DT_LORAS_DIR / p.name
                    shutil.move(str(p), str(dest))
                    produced = dest.name
        with _train_lock:
            _train.update(status="done" if ok else "error",
                          returncode=proc.returncode, produced=produced,
                          log=log[-3000:])
    except Exception as exc:
        with _train_lock:
            _train.update(status="error", log=str(exc), produced=None)


@app.post("/api/dt/train/start")
def dt_train_start(req: TrainRequest) -> dict:
    with _train_lock:
        if _train.get("status") == "running":
            raise HTTPException(status_code=409, detail="A training run is already in progress.")
    if req.base not in _DT_BY_CKPT:
        raise HTTPException(status_code=400, detail="Unknown base model.")
    if not _dt_downloaded(req.base):
        raise HTTPException(status_code=409, detail="Base model not downloaded.")
    ds = os.path.realpath(req.dataset)
    if not os.path.isdir(ds):
        raise HTTPException(status_code=400, detail="Dataset folder not found.")
    if not _is_allowed_path(ds):
        raise HTTPException(status_code=400, detail="Dataset must be in your home folder or an external drive.")
    imgs = [f for f in os.listdir(ds)
            if os.path.splitext(f)[1].lower() in _ALLOWED_IMG_EXT]
    if not imgs:
        raise HTTPException(status_code=400, detail="No images in dataset (.png/.jpg).")
    # Sanitize the output name to a bare slug (no path parts).
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", req.name).strip("_") or "my_lora"
    steps = max(1, min(5000, int(req.steps)))
    rank = max(1, min(128, int(req.rank)))

    cmd = [
        _DT_CLI, "train", "lora",
        "--models-dir", str(DT_MODELS_DIR),
        "--model", req.base,
        "--dataset", ds,
        "--steps", str(steps),
        "--rank", str(rank),
        "--output", slug,
        "--name", slug,
    ]
    if req.dry_run:
        cmd.append("--dry-run")

    with _train_lock:
        _train.clear()
        _train.update(status="running", base=req.base, dataset=ds, steps=steps,
                      rank=rank, name=slug, dry_run=req.dry_run, log="")
    global _train_cancel
    _train_cancel = False          # clear any stale cancel from a prior run
    threading.Thread(target=_train_worker, args=(cmd, slug, req.dry_run), daemon=True).start()
    return {"status": "running", "name": slug, "images": len(imgs)}


@app.post("/api/dt/train/cancel")
def dt_train_cancel() -> dict:
    """Stop an in-flight training run and reset to idle."""
    global _train_cancel
    _train_cancel = True
    with _train_lock:
        p = _train_proc
    if p is not None:
        try:
            p.kill()
        except Exception:
            pass
    return {"status": "cancelling"}


@app.get("/api/dt/train/status")
def dt_train_status() -> dict:
    with _train_lock:
        return dict(_train)


@app.post("/api/generate")
def generate(req: GenerateRequest) -> dict:
    name = req.model or DEFAULT_MODEL
    dt_reg = _dt_resolve(name)
    is_dt = dt_reg is not None
    if name not in MODEL_SPECS and not is_dt:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{name}'.",
        )
    if is_dt and not _dt_downloaded(dt_reg["ckpt"]):
        raise HTTPException(status_code=409,
                            detail=f"Model not downloaded: {dt_reg['label']}. Download it from Models first.")
    # The mflux default must have finished its startup load; on-demand mflux
    # models load inside the call. DT models are a subprocess (no preload).
    if name == DEFAULT_MODEL and _state["status"] != "ready":
        raise HTTPException(status_code=409, detail=f"Model not ready: {_state['detail']}")

    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Empty prompt.")

    reg = dt_reg if is_dt else MODEL_SPECS[name]
    project = _safe_project(req.project)
    image_path = _safe_image_path(req.image_path) if req.image_path else None
    width, height = _round16(req.width), _round16(req.height)
    steps = min(150, req.steps) if req.steps and req.steps > 0 else reg["default_steps"]
    seed = req.seed if req.seed is not None else int.from_bytes(uuid.uuid4().bytes[:4], "big")

    if not _gen_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Another generation is already running.")
    try:
        # Run the work on the dedicated engine thread and block for the result.
        fn = _dt_generate if is_dt else _do_generate
        kwargs: dict[str, Any] = {}
        if is_dt:
            kwargs["loras"] = _safe_loras(req.loras, dt_reg)
        else:
            kwargs["loras"] = req.loras        # mflux: applied for FLUX schnell/dev
            kwargs["scheduler"] = _safe_scheduler(req.scheduler)  # mflux only
        return _engine.submit(
            fn, name, project, prompt, width, height, steps,
            req.guidance, seed, req.negative_prompt,
            image_path, req.image_strength, **kwargs,
        ).result()
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
    finally:
        _gen_lock.release()


_ALLOWED_SCHEDULERS = {"linear", "flow_match_euler_discrete"}


def _safe_scheduler(s: Optional[str]) -> Optional[str]:
    """Whitelist the mflux scheduler; None -> model default."""
    if not s:
        return None
    s = s.strip()
    return s if s in _ALLOWED_SCHEDULERS else None


def _do_generate(name: str, project: str, prompt: str, width: int, height: int,
                 steps: int, guidance: float, seed: int,
                 negative_prompt: Optional[str],
                 image_path: Optional[str] = None,
                 image_strength: Optional[float] = None,
                 loras: Optional[list[dict]] = None,
                 scheduler: Optional[str] = None) -> dict:
    """Runs on the single MLX engine thread (consistent GPU stream)."""
    spec = MODEL_SPECS[name]
    try:
        if loras:
            # mflux applies LoRAs only to FLUX models, and only at build time.
            if name not in ("schnell", "dev"):
                raise HTTPException(
                    status_code=400,
                    detail="This model doesn't support LoRA. Choose FLUX schnell/dev or a Draw Things model.")
            paths, scales = _mflux_lora_args(loras)
            model = _build_flux_with_loras(name, paths, scales)   # fresh build w/ LoRA
        else:
            model = _ensure_model(name)
    except HTTPException:
        raise
    except Exception as exc:
        _load_errors[name] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        hint = ""
        if name == "z-image-turbo":
            hint = (" Weights may not be downloaded — download them over WiFi: "
                    "HF_HUB_OFFLINE=0 ./engine-venv/bin/huggingface-cli download "
                    f"{ZIMAGE_TURBO_PATH}")
        raise HTTPException(status_code=503,
                            detail=f"'{name}' failed to load: {type(exc).__name__}: {exc}.{hint}")

    kwargs: dict[str, Any] = dict(
        seed=seed, prompt=prompt, width=width, height=height, num_inference_steps=steps,
    )
    if scheduler:
        kwargs["scheduler"] = scheduler
    if spec["use_guidance"]:
        kwargs["guidance"] = guidance
    if spec["negative"] and negative_prompt:
        kwargs["negative_prompt"] = negative_prompt
    if image_path:                                   # img2img
        kwargs["image_path"] = image_path
        kwargs["image_strength"] = image_strength if image_strength is not None else 0.6

    t0 = time.time()
    image = model.generate_image(**kwargs)
    elapsed = time.time() - t0
    gen_id = uuid.uuid4().hex[:12]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    fname = f"{stamp}_{name}_seed{seed}_{gen_id[:6]}.png"
    image.save(path=str(_project_dir(project) / fname))
    entry = {
        "id": gen_id, "file": fname, "project": project,
        "url": f"/media/{project}/{fname}",
        "prompt": prompt, "width": width, "height": height, "steps": steps,
        "guidance": guidance if spec["use_guidance"] else None,
        "negative_prompt": negative_prompt if spec["negative"] else None,
        "seed": seed, "model": name, "scheduler": scheduler,
        "quantize": QUANTIZE, "elapsed_sec": round(elapsed, 1), "ts": time.time(),
    }
    _append_history(entry)
    _append_app_manifest(project, name, fname, gen_id, prompt,
                         negative_prompt if spec["negative"] else None,
                         width, height, steps,
                         guidance if spec["use_guidance"] else None, seed, elapsed)
    return entry


_DT_CLI = shutil.which("draw-things-cli") or "draw-things-cli"
# Safety net so a hung CLI can't pin the global generation lock forever. Generous
# enough never to kill a real render (4K hi-res-fix can take >10 min).
_DT_GEN_TIMEOUT = 1800


def _is_edit_ckpt(name: str) -> bool:
    """Instruction-edit models: image + a text instruction edits the image."""
    s = name.lower()
    return "kontext" in s or "_edit_" in s or "pix2pix" in s


def _dt_run(out_path: str, reg: dict, prompt: str, width: int, height: int,
            steps: int, guidance: float, seed: int,
            negative_prompt: Optional[str] = None,
            image_path: Optional[str] = None,
            image_strength: Optional[float] = None,
            loras: Optional[list[dict]] = None) -> None:
    """Build + run one draw-things-cli generate to `out_path`. Raises on failure."""
    cmd = [
        _DT_CLI, "generate",
        "--models-dir", str(DT_MODELS_DIR),
        "--model", reg["ckpt"],
        "--prompt", prompt,
        "--seed", str(seed),
        "--width", str(width), "--height", str(height),
        "--steps", str(steps),
        "--output", str(out_path),
    ]
    if reg.get("use_guidance"):
        cmd += ["--cfg", str(guidance)]
    if reg.get("negative") and negative_prompt:
        cmd += ["--negative-prompt", negative_prompt]
    if image_path:
        cmd += ["--image", image_path]
        # Control hint + instruction-edit models (Kontext/Qwen-Edit/Pix2Pix) take
        # the image directly; only plain img2img uses a denoising strength.
        if not (reg.get("control") or _is_edit_ckpt(reg["ckpt"])):
            cmd += ["--strength", str(image_strength if image_strength is not None else 0.6)]
    if loras:                                        # LoRA via JSON override
        cmd += ["--config-json", json.dumps({"loras": loras})]

    # Popen (not run) so a generation can be cancelled by killing the process.
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    with _proc_lock:
        global _current_dt_proc
        _current_dt_proc = proc
    try:
        out, err = proc.communicate(timeout=_DT_GEN_TIMEOUT)
    except subprocess.TimeoutExpired:
        # A hung CLI would otherwise pin _gen_lock forever, wedging ALL generation.
        proc.kill()
        try:
            proc.communicate(timeout=10)
        except Exception:
            pass
        with _proc_lock:
            _current_dt_proc = None
        raise HTTPException(status_code=504, detail="Generation timed out (CLI did not respond).")
    with _proc_lock:
        _current_dt_proc = None
    if proc.returncode != 0 or not os.path.exists(out_path):
        if proc.returncode is not None and proc.returncode < 0:   # killed by a signal
            raise HTTPException(status_code=499, detail="Generation cancelled.")
        tail = (err or out or "")[-500:]
        raise HTTPException(status_code=500, detail=f"draw-things-cli error: {tail}")


_current_dt_proc: Optional[subprocess.Popen] = None
_proc_lock = threading.Lock()


@app.post("/api/cancel")
def cancel_generation() -> dict:
    """Cancel the running Draw Things generation by killing its subprocess."""
    with _proc_lock:
        p = _current_dt_proc
    if p is not None and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=3)
        except Exception:
            p.kill()
        return {"status": "cancelled"}
    return {"status": "idle"}


def _dt_generate(name: str, project: str, prompt: str, width: int, height: int,
                 steps: int, guidance: float, seed: int,
                 negative_prompt: Optional[str],
                 image_path: Optional[str] = None,
                 image_strength: Optional[float] = None,
                 loras: Optional[list[dict]] = None) -> dict:
    """Generate via the Draw Things CLI subprocess. Writes the PNG straight into
    projects/<project>/ and records both manifests, like the mflux path."""
    reg = _dt_resolve(name)
    gen_id = uuid.uuid4().hex[:12]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    fname = f"{stamp}_{name}_seed{seed}_{gen_id[:6]}.png"
    out_path = _project_dir(project) / fname

    t0 = time.time()
    _dt_run(str(out_path), reg, prompt, width, height, steps, guidance, seed,
            negative_prompt, image_path, image_strength, loras)
    elapsed = time.time() - t0

    entry = {
        "id": gen_id, "file": fname, "project": project,
        "url": f"/media/{project}/{fname}",
        "prompt": prompt, "width": width, "height": height, "steps": steps,
        "guidance": guidance if reg.get("use_guidance") else None,
        "negative_prompt": negative_prompt if reg.get("negative") else None,
        "seed": seed, "model": name,
        "quantize": None, "elapsed_sec": round(elapsed, 1), "ts": time.time(),
    }
    _append_history(entry)
    _append_app_manifest(project, name, fname, gen_id, prompt,
                         negative_prompt if reg.get("negative") else None,
                         width, height, steps,
                         guidance if reg.get("use_guidance") else None, seed, elapsed)
    return entry


# --- Region inpaint (Generation Area over part of an image) --------------------
# The CLI has no mask flag, so this crops the selected region, regenerates it
# (img2img guided by the prompt), and composites it back with a feathered edge.
class InpaintRequest(BaseModel):
    model: str
    project: Optional[str] = None
    prompt: str
    negative_prompt: Optional[str] = None
    base_file: str                  # file within projects/<project>/ to edit
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0
    # Brush inpaint: a local PNG mask (white = repaint, black = keep), base-sized.
    # When given, the region (x/y/w/h) is derived from the mask's bounding box and
    # the feathered brush shape drives the composite (only painted pixels change).
    mask_path: Optional[str] = None
    steps: Optional[int] = None
    guidance: float = 3.5
    seed: Optional[int] = None
    strength: float = 0.75          # how much the region may change (img2img)
    loras: Optional[list[dict]] = None


def _do_inpaint(name: str, project: str, prompt: str,
                negative_prompt: Optional[str], base_path: str,
                x: int, y: int, w: int, h: int, steps: int, guidance: float,
                seed: int, strength: float, loras: Optional[list[dict]],
                mask_path: Optional[str] = None) -> dict:
    dt_reg = _dt_resolve(name)
    is_dt = dt_reg is not None
    spec = dt_reg if is_dt else MODEL_SPECS[name]

    base = Image.open(base_path).convert("RGB")
    bw, bh = base.size

    # Brush mask: derive the region from the painted area's bounding box, padded so
    # the regenerated patch has surrounding context. The feathered brush shape (not
    # a rectangle) drives the final composite, so only painted pixels change.
    brush_mask = None
    if mask_path:
        mfull = Image.open(mask_path).convert("L").resize((bw, bh))
        bbox = mfull.getbbox()
        if not bbox:
            raise HTTPException(status_code=400, detail="Mask is empty — no painted area.")
        bx0, by0, bx1, by1 = bbox
        pad = max(48, (bx1 - bx0) // 4, (by1 - by0) // 4)
        x = max(0, bx0 - pad); y = max(0, by0 - pad)
        w = min(bw, bx1 + pad) - x; h = min(bh, by1 + pad) - y
        brush_mask = mfull.crop((x, y, x + w, y + h))

    # Clamp the region to the image.
    x = max(0, min(x, bw - 16)); y = max(0, min(y, bh - 16))
    w = max(16, min(w, bw - x)); h = max(16, min(h, bh - y))

    crop = base.crop((x, y, x + w, y + h))
    # Generation dims must be multiples of 64 (DT) / 16 (mflux); 64 satisfies both.
    r64 = lambda v: max(64, min(2048, (int(v) // 64) * 64))
    gw, gh = r64(w), r64(h)
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        cin = os.path.join(td, "in.png")
        cout = os.path.join(td, "out.png")
        crop.resize((gw, gh)).save(cin)
        if is_dt:
            _dt_run(cout, dt_reg, prompt, gw, gh, steps, guidance, seed,
                    negative_prompt, cin, strength, loras)
        else:
            mdl = _ensure_model(name)              # mflux img2img on the crop
            kwargs: dict[str, Any] = dict(
                seed=seed, prompt=prompt, width=gw, height=gh, num_inference_steps=steps,
                image_path=cin,
                image_strength=strength if strength is not None else 0.6)
            if spec["use_guidance"]:
                kwargs["guidance"] = guidance
            if spec["negative"] and negative_prompt:
                kwargs["negative_prompt"] = negative_prompt
            mdl.generate_image(**kwargs).save(path=cout)
        region = Image.open(cout).convert("RGB").resize((w, h))
    elapsed = time.time() - t0

    # Feathered mask so the regenerated patch blends into the original. With a brush
    # mask we feather the painted shape itself; otherwise an inset rectangle.
    if brush_mask is not None:
        feather = max(2, min(w, h) // 24)
        mask = brush_mask.filter(ImageFilter.GaussianBlur(feather))
    else:
        feather = max(2, min(w, h) // 12)
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).rectangle([feather, feather, w - feather, h - feather], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(feather))

    out = base.copy()
    out.paste(region, (x, y), mask)

    gen_id = uuid.uuid4().hex[:12]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    fname = f"{stamp}_{name}_inpaint_{gen_id[:6]}.png"
    out.save(str(_project_dir(project) / fname))
    entry = {
        "id": gen_id, "file": fname, "project": project,
        "url": f"/media/{project}/{fname}",
        "prompt": prompt, "width": bw, "height": bh, "steps": steps,
        "guidance": guidance if spec.get("use_guidance") else None,
        "negative_prompt": negative_prompt if spec.get("negative") else None,
        "seed": seed, "model": name,
        "quantize": None, "elapsed_sec": round(elapsed, 1), "ts": time.time(),
    }
    _append_history(entry)
    _append_app_manifest(project, name, fname, gen_id, prompt,
                         negative_prompt if spec.get("negative") else None,
                         bw, bh, steps,
                         guidance if spec.get("use_guidance") else None, seed, elapsed)
    return entry


@app.post("/api/inpaint")
def inpaint(req: InpaintRequest) -> dict:
    dt_reg = _dt_resolve(req.model)
    is_dt = dt_reg is not None
    if not is_dt and req.model not in MODEL_SPECS:
        raise HTTPException(status_code=400, detail=f"Unknown model '{req.model}'.")
    if is_dt and not _dt_downloaded(dt_reg["ckpt"]):
        raise HTTPException(status_code=409, detail="Model not downloaded.")
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Empty prompt.")
    project = _safe_project(req.project)
    base_path = PROJECTS / project / os.path.basename(req.base_file)
    if not base_path.is_file():
        raise HTTPException(status_code=400, detail="Base image not found.")
    spec = dt_reg if is_dt else MODEL_SPECS[req.model]
    steps = min(150, req.steps) if req.steps and req.steps > 0 else spec["default_steps"]
    seed = req.seed if req.seed is not None else int.from_bytes(uuid.uuid4().bytes[:4], "big")

    mask_path = _safe_image_path(req.mask_path) if req.mask_path else None
    if not _gen_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Another generation is already running.")
    try:
        return _engine.submit(
            _do_inpaint, req.model, project, prompt, req.negative_prompt,
            str(base_path), req.x, req.y, req.w, req.h, steps, req.guidance,
            seed, req.strength, (_safe_loras(req.loras, dt_reg) if is_dt else None),
            mask_path,
        ).result()
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
    finally:
        _gen_lock.release()


# --- Local video generation (draw-things-cli video models -> mp4) -------------
# Kept separate from the image history so the photo gallery never mixes in clips.
class VideoRequest(BaseModel):
    model: str                       # a video-capable DT .ckpt id
    project: Optional[str] = None
    prompt: str
    width: int = 832
    height: int = 448
    steps: Optional[int] = None
    frames: int = 25                 # 4n+1 (e.g. 13/25/49/81)
    guidance: float = 5.0
    seed: Optional[int] = None
    image_path: Optional[str] = None  # image-to-video source frame
    image_strength: float = 0.3       # i2v: low = faithful to the image, high = more motion
    smooth: bool = True               # interpolate to 2x fps (full-res, fast)


def _smooth_video(path: str, factor: int = 2) -> None:
    """Motion-compensated frame interpolation (ffmpeg minterpolate) — generate
    fewer frames, then double the fps at FULL resolution for smooth, fast clips.
    Silently no-ops if ffmpeg is missing or it fails."""
    ff = shutil.which("ffmpeg")
    if not ff:
        return
    fps = 16.0
    fp = shutil.which("ffprobe")
    if fp:
        try:
            pr = _run([fp, "-v", "error", "-select_streams", "v:0",
                                 "-show_entries", "stream=avg_frame_rate",
                                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                                capture_output=True, text=True)
            n, d = pr.stdout.strip().split("/")
            fps = (float(n) / float(d)) if float(d) else 16.0
        except Exception:
            fps = 16.0
    target = max(8, int(round(fps * factor)))
    tmp = path + ".smooth.mp4"
    r = _run([ff, "-y", "-v", "error", "-i", path, "-vf",
                        f"minterpolate=fps={target}:mi_mode=mci:mc_mode=aobmc:vsbmc=1", tmp],
                       capture_output=True, text=True)
    if r.returncode == 0 and os.path.isfile(tmp):
        os.replace(tmp, path)


def _do_video(name: str, project: str, prompt: str, width: int, height: int,
              steps: int, frames: int, guidance: float, seed: int,
              image_path: Optional[str], smooth: bool = True,
              image_strength: float = 0.3) -> dict:
    reg = _dt_resolve(name)
    if reg is None:
        raise HTTPException(status_code=400, detail=f"'{name}' is not a Draw Things video model.")
    r64 = lambda v: max(64, min(1280, (int(v) // 64) * 64))
    w, h = r64(width), r64(height)
    f = max(5, min(121, int(frames)))
    if (f - 1) % 4 != 0:                       # snap to 4n+1
        f = ((f - 1) // 4) * 4 + 1
    gen_id = uuid.uuid4().hex[:12]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    fname = f"{stamp}_{name}_vid_{gen_id[:6]}.mp4"
    out = str(_project_dir(project) / fname)
    cmd = [_DT_CLI, "generate", "--models-dir", str(DT_MODELS_DIR), "--model", reg["ckpt"],
           "--prompt", prompt, "--seed", str(seed),
           "--width", str(w), "--height", str(h), "--steps", str(steps),
           "--frames", str(f), "--output", out]
    if reg.get("use_guidance"):
        cmd += ["--cfg", str(guidance)]
    if image_path:                                   # image-to-video
        # A LOW denoising strength keeps the source image (high = it dissolves into
        # noise). 0.25–0.4 preserves the subject while still adding real motion.
        cmd += ["--image", image_path, "--strength", str(max(0.1, min(0.85, image_strength)))]
    t0 = time.time()
    # Popen (not _run) + register the proc so /api/cancel can stop a video render,
    # and a hung CLI can't pin _gen_lock for the whole timeout, wedging generation.
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    with _proc_lock:
        global _current_dt_proc
        _current_dt_proc = proc
    try:
        v_out, v_err = proc.communicate(timeout=_DT_GEN_TIMEOUT)   # render can be minutes
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.communicate(timeout=10)
        except Exception:
            pass
        with _proc_lock:
            _current_dt_proc = None
        raise HTTPException(status_code=504, detail="Video generation timed out (CLI did not respond).")
    with _proc_lock:
        _current_dt_proc = None
    if proc.returncode != 0 or not os.path.isfile(out):
        if proc.returncode is not None and proc.returncode < 0:    # killed by /api/cancel
            raise HTTPException(status_code=499, detail="Video generation cancelled.")
        raise HTTPException(status_code=500,
                            detail=f"Video generation failed: {(v_err or v_out or '')[-300:]}")
    if smooth:
        _smooth_video(out, factor=2)       # full-res frame interpolation
    elapsed = time.time() - t0
    return {"id": gen_id, "file": fname, "project": project,
            "url": f"/media/{project}/{fname}", "kind": "video",
            "prompt": prompt, "width": w, "height": h, "frames": f,
            "seed": seed, "model": name, "elapsed_sec": round(elapsed, 1), "ts": time.time()}


class UpscaleVideoRequest(BaseModel):
    project: Optional[str] = None
    file: str
    height: int = 1080          # target height (e.g. 1080 = HD, 2160 = 4K)


@app.post("/api/upscale/video")
def upscale_video(req: UpscaleVideoRequest) -> dict:
    proj = _safe_project(req.project)
    src = _project_dir(proj) / os.path.basename(req.file)
    if not src.is_file():
        raise HTTPException(status_code=400, detail="Video not found.")
    ff = shutil.which("ffmpeg")
    if not ff:
        raise HTTPException(status_code=500, detail="ffmpeg is not installed.")
    h = max(480, min(2160, int(req.height)))
    out_name = f"{src.stem}_hd{h}.mp4"
    out = _project_dir(proj) / out_name
    cmd = [ff, "-y", "-v", "error", "-i", str(src),
           "-vf", f"scale=-2:{h}:flags=lanczos,unsharp=5:5:0.8:5:5:0.0",
           "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(out)]
    r = _run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out.is_file():
        raise HTTPException(status_code=500, detail=f"Upscale failed: {(r.stderr or '')[-200:]}")
    return {"file": out_name, "project": proj, "url": f"/media/{proj}/{out_name}"}


# Social delivery formats (marketers/designers): fixed pro dimensions per aspect.
_SOCIAL_DIMS = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080), "4:5": (1080, 1350)}


class CombineRequest(BaseModel):
    project: Optional[str] = None
    files: list[str]
    aspect: str = "16:9"


@app.post("/api/video/combine")
def combine_videos(req: CombineRequest) -> dict:
    proj = _safe_project(req.project)
    d = _project_dir(proj)
    paths = [d / os.path.basename(f) for f in req.files]
    paths = [p for p in paths if p.is_file()]
    if not paths:
        raise HTTPException(status_code=400, detail="No clips to combine.")
    ff = shutil.which("ffmpeg")
    if not ff:
        raise HTTPException(status_code=500, detail="ffmpeg is not installed.")
    ff_probe = shutil.which("ffprobe")
    W, H = _SOCIAL_DIMS.get(req.aspect, (1920, 1080))

    def _probe(p: "Path") -> "tuple[bool, float]":
        """(has_audio, duration_seconds) — best-effort; silent fallback if probe fails."""
        if not ff_probe:
            return False, 0.0
        try:
            has_a = bool(_run(
                [ff_probe, "-v", "error", "-select_streams", "a", "-show_entries",
                 "stream=index", "-of", "csv=p=0", str(p)],
                capture_output=True, text=True).stdout.strip())
            dur = _run(
                [ff_probe, "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(p)], capture_output=True, text=True).stdout.strip()
            return has_a, float(dur) if dur else 0.0
        except Exception:
            return False, 0.0

    meta = [_probe(p) for p in paths]
    inputs: list[str] = []
    for p in paths:                                  # video/audio inputs first (indices 0..N-1)
        inputs += ["-i", str(p)]
    silent_idx: dict[int, int] = {}                  # clip i -> lavfi input index, for clips with no audio
    for i, (has_a, dur) in enumerate(meta):
        if not has_a:
            silent_idx[i] = len(paths) + len(silent_idx)
            inputs += ["-f", "lavfi", "-t", f"{max(dur, 0.1):.3f}",
                       "-i", "anullsrc=r=44100:cl=stereo"]
    filters: list[str] = []
    for i, p in enumerate(paths):
        filters.append(
            f"[{i}:v]scale={W}:{H}:force_original_aspect_ratio=decrease,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30[v{i}]")
        asrc = f"[{silent_idx[i]}:a]" if i in silent_idx else f"[{i}:a]"
        filters.append(f"{asrc}aresample=44100,asetpts=PTS-STARTPTS[a{i}]")
    pairs = "".join(f"[v{i}][a{i}]" for i in range(len(paths)))
    concat = pairs + f"concat=n={len(paths)}:v=1:a=1[out][outa]"
    fc = ";".join(filters) + ";" + concat
    out_name = f"export_{uuid.uuid4().hex[:6]}_{req.aspect.replace(':', 'x')}.mp4"
    out = d / out_name
    cmd = [ff, "-y", "-v", "error"] + inputs + [
        "-filter_complex", fc, "-map", "[out]", "-map", "[outa]",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", str(out)]
    r = _run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out.is_file():
        raise HTTPException(status_code=500, detail=f"Merge failed: {(r.stderr or '')[-200:]}")
    return {"file": out_name, "project": proj, "url": f"/media/{proj}/{out_name}"}


class SeqClip(BaseModel):
    file: str
    start: float = 0.0      # trim in (seconds)
    end: float = 0.0        # trim out (seconds); 0 = to end


class SeqTitle(BaseModel):
    text: str
    start: float = 0.0
    end: float = 0.0                   # 0 = until the end
    position: str = "bottom"           # top | center | bottom
    color: Optional[str] = None        # hex (#RRGGBB); falls back to brand colour


class SeqBrand(BaseModel):
    logo: Optional[str] = None         # full path to a (transparent) logo PNG
    color: Optional[str] = None        # accent hex for titles
    logo_corner: str = "tr"            # tl | tr | bl | br


class SequenceRequest(BaseModel):
    project: Optional[str] = None
    clips: list[SeqClip]
    aspect: str = "16:9"
    transition: str = "none"          # none | crossfade
    transition_dur: float = 0.5
    music: Optional[str] = None        # full path to an audio file (optional)
    titles: list[SeqTitle] = []        # burned text/title layers (Phase D)
    brand: Optional[SeqBrand] = None   # logo watermark + accent colour (Phase D)


def _vdur(path) -> float:
    fp = shutil.which("ffprobe")
    if not fp:
        return 0.0
    try:
        r = _run([fp, "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                           capture_output=True, text=True)
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def _has_audio(pth) -> bool:
    """True if the file has at least one audio stream (AI clips often have none)."""
    fp = shutil.which("ffprobe")
    if not fp:
        return False
    try:
        r = _run([fp, "-v", "error", "-select_streams", "a", "-show_entries",
                  "stream=index", "-of", "csv=p=0", str(pth)], capture_output=True, text=True)
        return bool(r.stdout.strip())
    except Exception:
        return False


@app.post("/api/video/sequence")
def render_sequence(req: SequenceRequest) -> dict:
    """Render an ordered, trimmed sequence to one social-format mp4, optionally
    with crossfade transitions and a background music track. Each clip keeps its
    own audio (silence for clips that have none); the music track is mixed under
    it as a ducked bed."""
    proj = _safe_project(req.project)
    d = _project_dir(proj)
    clips = [(d / os.path.basename(c.file), c.start, c.end) for c in req.clips]
    clips = [(p, s, e) for (p, s, e) in clips if p.is_file()]
    if not clips:
        raise HTTPException(status_code=400, detail="No clips in sequence.")
    ff = shutil.which("ffmpeg")
    if not ff:
        raise HTTPException(status_code=500, detail="ffmpeg is not installed.")
    W, H = _SOCIAL_DIMS.get(req.aspect, (1920, 1080))
    n = len(clips)
    T = max(0.1, min(2.0, req.transition_dur)) if (req.transition == "crossfade" and n > 1) else 0.0

    inputs: list[str] = []
    segs: list[str] = []
    durs: list[float] = []
    for i, (p, s, e) in enumerate(clips):
        inputs += ["-i", str(p)]
        full = _vdur(p)
        segdur = (e - s) if (e and e > s) else (full - s if full > s else full)
        durs.append(max(0.2, segdur))
        trim = f"trim=start={max(0.0, s)}" + (f":end={e}" if e and e > s else "")
        segs.append(
            f"[{i}:v]{trim},setpts=PTS-STARTPTS,"
            f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30[v{i}]")
    fc = ";".join(segs)
    if T > 0:
        cur, cum = "v0", durs[0]
        for i in range(1, n):
            off = max(0.1, cum - T)
            lbl = f"x{i}"
            fc += f";[{cur}][v{i}]xfade=transition=fade:duration={T}:offset={off}[{lbl}]"
            cur, cum = lbl, cum + durs[i] - T
        out_label = cur
    else:
        fc += ";" + "".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1[vout]"
        out_label = "vout"

    audio_in: list[str] = []
    # Confine the music track to the project tree (the app copies a picked track in,
    # like /api/audio/musicbed) so a request can't mux an arbitrary local file into
    # the output. Invalid/out-of-tree → silently no music (don't read it).
    mpath = None
    if req.music:
        try:
            mpath = _safe_project_file(d, req.music)
        except HTTPException:
            mpath = None
    if mpath:
        audio_in = ["-i", mpath]          # music becomes input index n

    # --- Phase D: burn title layers + a brand logo watermark on the final video ---
    overlay_in: list[str] = []
    tmp_layers: list[str] = []
    base = n + (1 if mpath else 0)        # next free input index after clips (+music)
    cur = out_label
    brand_color = req.brand.color if req.brand else None
    for j, t in enumerate(req.titles or []):
        if not (t.text or "").strip():
            continue
        tp = d / f".title_{uuid.uuid4().hex[:6]}.png"
        _render_title(t.text, W, H, t.position, t.color or brand_color, str(tp))
        tmp_layers.append(str(tp))
        idx = base + len(overlay_in) // 2
        overlay_in += ["-i", str(tp)]
        s = max(0.0, t.start)
        e = t.end if (t.end and t.end > s) else 1e9
        nl = f"tt{j}"
        fc += f";[{cur}][{idx}:v]overlay=0:0:enable='between(t,{s:.2f},{e:.2f})'[{nl}]"
        cur = nl
    # The brand logo is a user-chosen global asset (Preferences), so it may live
    # outside the project — but it must be a real image, not an arbitrary file to
    # bake into the output. Validate extension/existence; skip silently if invalid.
    brand_logo = None
    if req.brand and req.brand.logo:
        try:
            brand_logo = _safe_image_path(req.brand.logo)
        except HTTPException:
            brand_logo = None
    if brand_logo:
        idx = base + len(overlay_in) // 2
        overlay_in += ["-i", brand_logo]
        lh = int(H * 0.10); mg = int(W * 0.03)
        corner = {"tl": f"{mg}:{mg}", "tr": f"W-w-{mg}:{mg}",
                  "bl": f"{mg}:H-h-{mg}", "br": f"W-w-{mg}:H-h-{mg}"}.get(
                      (req.brand.logo_corner or "tr"), f"W-w-{mg}:{mg}")
        fc += f";[{idx}:v]scale=-1:{lh}[lg];[{cur}][lg]overlay={corner}[wm]"
        cur = "wm"
    out_label = cur

    # --- Audio: each clip keeps its own audio (silence where a clip has none),
    # joined to match the video (concat, or acrossfade under a crossfade transition),
    # with the optional music track mixed in as a ducked, looped bed. ---
    a_segs: list[str] = []
    for i, (p, s, e) in enumerate(clips):
        if _has_audio(p):
            atrim = f"atrim=start={max(0.0, s)}" + (f":end={e}" if e and e > s else "")
            a_segs.append(f"[{i}:a]{atrim},asetpts=PTS-STARTPTS,aresample=48000[a{i}]")
        else:
            a_segs.append(
                f"anullsrc=r=48000:cl=stereo,atrim=duration={durs[i]:.3f},asetpts=PTS-STARTPTS[a{i}]")
    fc += ";" + ";".join(a_segs)
    if T > 0:
        acur = "a0"
        for i in range(1, n):
            albl = f"ax{i}"
            fc += f";[{acur}][a{i}]acrossfade=d={T}[{albl}]"
            acur = albl
        a_out = acur
    else:
        fc += ";" + "".join(f"[a{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1[aout]"
        a_out = "aout"
    if mpath:
        fc += (f";[{n}:a]volume=0.35,aloop=loop=-1:size=2000000000[mbed]"
               f";[{a_out}][mbed]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[amix]")
        a_out = "amix"

    map_args = ["-map", f"[{out_label}]", "-map", f"[{a_out}]",
                "-c:a", "aac", "-b:a", "192k"]

    out_name = f"timeline_{uuid.uuid4().hex[:6]}_{req.aspect.replace(':', 'x')}.mp4"
    out = d / out_name
    cmd = [ff, "-y", "-v", "error"] + inputs + audio_in + overlay_in + [
        "-filter_complex", fc] + map_args + [
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(out)]
    r = _run(cmd, capture_output=True, text=True)
    for p in tmp_layers:
        try: os.remove(p)
        except OSError: pass
    if r.returncode != 0 or not out.is_file():
        raise HTTPException(status_code=500, detail=f"Sequence render failed: {(r.stderr or '')[-200:]}")
    return {"file": out_name, "project": proj, "url": f"/media/{proj}/{out_name}"}


# --- Voiceover (macOS TTS) + burned captions (text is exact -> 100% accurate) ----
class VoiceoverRequest(BaseModel):
    project: Optional[str] = None
    file: str
    script: str
    voice: str = "Samantha"            # local macOS `say` voice
    captions: bool = True
    engine: str = "local"              # local | elevenlabs
    voice_id: Optional[str] = None     # ElevenLabs voice id (when engine=elevenlabs)


def _adur(path) -> float:
    fp = shutil.which("ffprobe")
    try:
        r = _run([fp, "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                           capture_output=True, text=True)
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def _vdims(path) -> tuple[int, int]:
    fp = shutil.which("ffprobe")
    try:
        r = _run([fp, "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
                           capture_output=True, text=True)
        w, h = r.stdout.strip().split("x")
        return int(w), int(h)
    except Exception:
        return 1920, 1080


def _caption_segments(script: str, total: float, max_chars: int = 42):
    words = script.split()
    chunks: list[str] = []
    cur = ""
    for w in words:
        if cur and len(cur) + len(w) + 1 > max_chars:
            chunks.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        chunks.append(cur)
    tot = sum(len(c) for c in chunks) or 1
    segs = []
    t = 0.0
    for c in chunks:
        dur = total * len(c) / tot
        segs.append((c, t, t + dur)); t += dur
    return segs


def _render_caption(text: str, W: int, H: int, out_path: str) -> None:
    import textwrap
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    size = max(28, int(W * 0.045))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", size)
    except Exception:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
    wrap = max(16, int(W / (size * 0.55)))
    lines = textwrap.wrap(text, wrap)
    lh = int(size * 1.25)
    y = int(H * 0.84) - lh * len(lines)
    o = max(2, size // 18)
    for ln in lines:
        bb = dr.textbbox((0, 0), ln, font=font)
        x = (W - (bb[2] - bb[0])) // 2
        for dx in (-o, 0, o):
            for dy in (-o, 0, o):
                dr.text((x + dx, y + dy), ln, font=font, fill=(0, 0, 0, 235))
        dr.text((x, y), ln, font=font, fill=(255, 255, 255, 255))
        y += lh
    img.save(out_path)


def _hex_rgb(h: Optional[str], default=(255, 255, 255)) -> tuple:
    try:
        h = (h or "").lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    except Exception:
        return default


def _render_title(text: str, W: int, H: int, position: str, color: Optional[str], out_path: str) -> None:
    """Full-frame transparent PNG with a centered title at top/center/bottom.
    Same heavy outline as captions so it reads over any footage."""
    import textwrap
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    size = max(34, int(W * 0.06))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", size)
    except Exception:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
    wrap = max(12, int(W / (size * 0.55)))
    lines = textwrap.wrap(text, wrap)
    lh = int(size * 1.22)
    block = lh * len(lines)
    if position == "top":
        y = int(H * 0.08)
    elif position == "center":
        y = (H - block) // 2
    else:
        y = int(H * 0.86) - block
    fill = _hex_rgb(color) + (255,)
    o = max(2, size // 16)
    for ln in lines:
        bb = dr.textbbox((0, 0), ln, font=font)
        x = (W - (bb[2] - bb[0])) // 2
        for dx in (-o, 0, o):
            for dy in (-o, 0, o):
                dr.text((x + dx, y + dy), ln, font=font, fill=(0, 0, 0, 235))
        dr.text((x, y), ln, font=font, fill=fill)
        y += lh
    img.save(out_path)


@app.post("/api/video/voiceover")
def voiceover(req: VoiceoverRequest) -> dict:
    proj = _safe_project(req.project)
    d = _project_dir(proj)
    src = d / os.path.basename(req.file)
    if not src.is_file():
        raise HTTPException(status_code=400, detail="Video not found.")
    script = (req.script or "").strip()
    if not script:
        raise HTTPException(status_code=400, detail="Empty text.")
    ff = shutil.which("ffmpeg")
    if not ff:
        raise HTTPException(status_code=500, detail="ffmpeg not found.")
    td = tempfile.mkdtemp()
    try:
        aiff = _synth_tts(script, td, req.engine, req.voice, req.voice_id)
        D = _adur(aiff) or 5.0
        W, H = _vdims(src)
        overlays = []
        if req.captions:
            for i, (txt, s, e) in enumerate(_caption_segments(script, D)):
                p = os.path.join(td, f"cap{i}.png")
                _render_caption(txt, W, H, p)
                overlays.append((p, s, e))
        inputs = ["-i", str(src), "-i", aiff]
        for p, _, _ in overlays:
            inputs += ["-i", p]
        # Hold the LAST frame to fill the narration (no ugly looping of the clip).
        fc = f"[0:v]tpad=stop_mode=clone:stop_duration=600,trim=0:{D:.2f},setpts=PTS-STARTPTS[b0]"
        cur = "b0"
        for i, (p, s, e) in enumerate(overlays):
            lbl = f"o{i}"
            fc += f";[{cur}][{i + 2}:v]overlay=0:0:enable='between(t,{s:.2f},{e:.2f})'[{lbl}]"
            cur = lbl
        out_name = f"vo_{uuid.uuid4().hex[:6]}.mp4"
        out = d / out_name
        cmd = [ff, "-y", "-v", "error"] + inputs + [
            "-filter_complex", fc, "-map", f"[{cur}]", "-map", "1:a",
            "-c:v", "libx264", "-crf", "18", "-c:a", "aac", "-t", f"{D:.2f}",
            "-pix_fmt", "yuv420p", str(out)]
        r = _run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not out.is_file():
            raise HTTPException(status_code=500, detail=f"Voiceover failed: {(r.stderr or '')[-200:]}")
        return {"file": out_name, "project": proj, "url": f"/media/{proj}/{out_name}"}
    finally:
        import shutil as _sh
        _sh.rmtree(td, ignore_errors=True)


@app.get("/api/videos")
def list_videos(project: Optional[str] = None) -> list[dict]:
    proj = _safe_project(project)
    d = _project_dir(proj)
    out = []
    for p in sorted(d.glob("*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True):
        out.append({"file": p.name, "project": proj, "url": f"/media/{proj}/{p.name}"})
    return out


@app.post("/api/generate/video")
def generate_video(req: VideoRequest) -> dict:
    reg = _dt_resolve(req.model)
    if reg is None or not _dt_downloaded(reg["ckpt"]):
        raise HTTPException(status_code=409, detail="Video model not downloaded.")
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Empty prompt.")
    project = _safe_project(req.project)
    steps = min(150, req.steps) if req.steps and req.steps > 0 else reg.get("default_steps", 20)
    seed = req.seed if req.seed is not None else int.from_bytes(uuid.uuid4().bytes[:4], "big")
    image_path = _safe_image_path(req.image_path) if req.image_path else None
    if not _gen_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Another generation is already running.")
    try:
        return _engine.submit(_do_video, req.model, project, prompt, req.width, req.height,
                              steps, req.frames, req.guidance, seed, image_path, req.smooth,
                              req.image_strength).result()
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
    finally:
        _gen_lock.release()


# --- Outpaint (extend the image beyond its borders) ---------------------------
# Reuses the inpaint engine: expand the canvas, edge-extend the new border as the
# img2img init context, then regenerate the border (high strength) and composite.
class OutpaintRequest(BaseModel):
    model: str
    project: Optional[str] = None
    prompt: str
    base_file: str
    side: str            # left | right | top | bottom
    amount: int = 256
    steps: Optional[int] = None
    guidance: float = 4.0
    seed: Optional[int] = None
    strength: float = 0.95
    loras: Optional[list[dict]] = None


@app.post("/api/outpaint")
def outpaint(req: OutpaintRequest) -> dict:
    dt_reg = _dt_resolve(req.model)
    is_dt = dt_reg is not None
    if not is_dt and req.model not in MODEL_SPECS:
        raise HTTPException(status_code=400, detail=f"Unknown model '{req.model}'.")
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Empty prompt.")
    project = _safe_project(req.project)
    base_path = PROJECTS / project / os.path.basename(req.base_file)
    if not base_path.is_file():
        raise HTTPException(status_code=400, detail="Base image not found.")
    side = req.side if req.side in ("left", "right", "top", "bottom") else "right"
    amt = max(64, min(1024, int(req.amount)))
    spec = dt_reg if is_dt else MODEL_SPECS[req.model]
    steps = min(150, req.steps) if req.steps and req.steps > 0 else spec["default_steps"]
    seed = req.seed if req.seed is not None else int.from_bytes(uuid.uuid4().bytes[:4], "big")

    base = Image.open(str(base_path)).convert("RGB")
    bw, bh = base.size
    overlap = 24
    if side in ("left", "right"):
        cw, ch = bw + amt, bh
    else:
        cw, ch = bw, bh + amt
    canvas = Image.new("RGB", (cw, ch))
    if side == "right":
        canvas.paste(base, (0, 0))
        canvas.paste(base.crop((bw - 2, 0, bw, bh)).resize((amt, bh)), (bw, 0))
        mrect = [bw - overlap, 0, cw, ch]
    elif side == "left":
        canvas.paste(base, (amt, 0))
        canvas.paste(base.crop((0, 0, 2, bh)).resize((amt, bh)), (0, 0))
        mrect = [0, 0, amt + overlap, ch]
    elif side == "bottom":
        canvas.paste(base, (0, 0))
        canvas.paste(base.crop((0, bh - 2, bw, bh)).resize((bw, amt)), (0, bh))
        mrect = [0, bh - overlap, cw, ch]
    else:  # top
        canvas.paste(base, (0, amt))
        canvas.paste(base.crop((0, 0, bw, 2)).resize((bw, amt)), (0, 0))
        mrect = [0, 0, cw, amt + overlap]

    if not _gen_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Another generation is already running.")
    td = None
    try:
        td = tempfile.mkdtemp()
        eb = os.path.join(td, "base.png"); canvas.save(eb)
        mask = Image.new("L", (cw, ch), 0)
        ImageDraw.Draw(mask).rectangle(mrect, fill=255)
        mm = os.path.join(td, "mask.png"); mask.save(mm)
        return _engine.submit(
            _do_inpaint, req.model, project, prompt, None, eb, 0, 0, 0, 0,
            steps, req.guidance, seed, req.strength,
            (_safe_loras(req.loras, dt_reg) if is_dt else None), mm,
        ).result()
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
    finally:
        _gen_lock.release()
        if td:
            shutil.rmtree(td, ignore_errors=True)


class DeleteRef(BaseModel):
    project: str
    file: str


@app.post("/api/delete")
def delete_generation(ref: DeleteRef) -> dict:
    """Delete one generated image + its manifest/history entries."""
    project = _safe_project(ref.project)
    fname = os.path.basename(ref.file)
    fp = PROJECTS / project / fname
    try:
        fp.unlink(missing_ok=True)
    except OSError:
        pass
    mf = PROJECTS / project / "manifest.json"
    with _history_lock:
        if mf.exists():
            try:
                rows = [r for r in json.loads(mf.read_text()) if r.get("file") != fname]
                _atomic_write_json(mf, rows)
            except json.JSONDecodeError:
                pass
        # Scope the global-history removal to THIS project+file so a same-named
        # image in another project isn't dropped (filenames are timestamp+uuid, but
        # be precise). Legacy rows without a project are left untouched.
        hist = [e for e in _read_history()
                if not (e.get("file") == fname and e.get("project") == project)]
        _atomic_write_json(HISTORY_FILE, hist)
    return {"status": "deleted"}


# --- Image Interpreter (local BLIP captioning -> prompt) ----------------------
_blip = None
_blip_lock = threading.Lock()


def _load_blip():
    global _blip
    with _blip_lock:
        if _blip is None:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            name = "Salesforce/blip-image-captioning-base"
            _blip = (BlipProcessor.from_pretrained(name),
                     BlipForConditionalGeneration.from_pretrained(name))
    return _blip


class InterrogateRef(BaseModel):
    project: str
    file: str


@app.post("/api/interrogate")
def interrogate_image(ref: InterrogateRef) -> dict:
    """Analyze an image with a local vision model and return a descriptive prompt."""
    project = _safe_project(ref.project)
    path = PROJECTS / project / os.path.basename(ref.file)
    if not path.is_file():
        raise HTTPException(status_code=400, detail="Image not found.")
    try:
        def _run_blip():
            proc, mdl = _load_blip()
            img = Image.open(str(path)).convert("RGB")
            inputs = proc(img, return_tensors="pt")
            out = mdl.generate(**inputs, max_new_tokens=60)
            return proc.decode(out[0], skip_special_tokens=True).strip()
        return {"prompt": _mlx(_run_blip)}     # MLX/torch must run on the engine thread
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Captioning error: {type(exc).__name__}: {exc}")


# --- Upscale (Lanczos) + instruction edit (resolve project/file internally) ---
class UpscaleRef(BaseModel):
    project: str
    file: str
    scale: int = 2


@app.post("/api/upscale")
def upscale_image(ref: UpscaleRef) -> dict:
    """Enlarge an existing image (Lanczos) and save it as a new generation."""
    project = _safe_project(ref.project)
    src = PROJECTS / project / os.path.basename(ref.file)
    if not src.is_file():
        raise HTTPException(status_code=400, detail="Image not found.")
    s = max(2, min(4, int(ref.scale)))
    img = Image.open(str(src)).convert("RGB")
    up = img.resize((img.width * s, img.height * s), Image.LANCZOS)
    gen_id = uuid.uuid4().hex[:12]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    fname = f"{stamp}_upscale{s}x_{gen_id[:6]}.png"
    up.save(str(_project_dir(project) / fname))
    entry = {
        "id": gen_id, "file": fname, "project": project,
        "url": f"/media/{project}/{fname}", "prompt": f"upscale {s}x",
        "width": up.width, "height": up.height, "steps": None, "guidance": None,
        "negative_prompt": None, "seed": 0, "model": f"upscale-{s}x",
        "quantize": None, "elapsed_sec": 0, "ts": time.time(),
    }
    _append_history(entry)
    _append_app_manifest(project, f"upscale-{s}x", fname, gen_id, f"upscale {s}x",
                         None, up.width, up.height, None, None, 0, 0)
    return entry


class EditRef(BaseModel):
    project: str
    file: str
    instruction: str
    model: str
    steps: Optional[int] = None
    seed: Optional[int] = None


@app.post("/api/edit")
def edit_image(ref: EditRef) -> dict:
    """Instruction-edit an existing image with an edit model (Kontext/Qwen-Edit)."""
    reg = _dt_resolve(ref.model)
    if reg is None or not _dt_downloaded(reg["ckpt"]):
        raise HTTPException(status_code=400, detail="A downloaded edit model is required.")
    project = _safe_project(ref.project)
    src = PROJECTS / project / os.path.basename(ref.file)
    if not src.is_file():
        raise HTTPException(status_code=400, detail="Image not found.")
    prompt = (ref.instruction or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Empty instruction.")
    img = Image.open(str(src))
    w, h = _round16(img.width), _round16(img.height)
    steps = ref.steps if ref.steps and ref.steps > 0 else reg["default_steps"]
    seed = ref.seed if ref.seed is not None else int.from_bytes(uuid.uuid4().bytes[:4], "big")
    if not _gen_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Another generation is already running.")
    try:
        return _engine.submit(
            _dt_generate, ref.model, project, prompt, w, h, steps,
            3.5, seed, None, str(src), None, None,
        ).result()
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
    finally:
        _gen_lock.release()


# --- Civitai browse + download (SFW-filtered general model browser) -----------
_CIVITAI_TOKEN_FILE = ROOT / ".civitai_token"


def _civitai_token() -> str:
    try:
        return _CIVITAI_TOKEN_FILE.read_text().strip() if _CIVITAI_TOKEN_FILE.exists() else ""
    except Exception:
        return ""


class CivitaiTokenRef(BaseModel):
    token: str


def _validate_civitai_url(url: str, allow_nsfw: bool = False) -> None:
    """SSRF guard: allow civitai.com. If allow_nsfw is True, also allow civitai.red."""
    p = urllib.parse.urlparse(url)
    host = (p.hostname or "").lower().rstrip(".")
    is_civitai_com = (host == "civitai.com" or host.endswith(".civitai.com"))
    is_civitai_red = (host == "civitai.red" or host.endswith(".civitai.red"))
    
    if p.scheme != "https":
        raise HTTPException(status_code=400, detail="Invalid scheme.")
        
    if is_civitai_red and not allow_nsfw:
        raise HTTPException(status_code=403, detail="civitai.red is unavailable while adult content (NSFW) is off.")
        
    if not (is_civitai_com or is_civitai_red):
        raise HTTPException(status_code=400, detail="Invalid download URL (only civitai.com and civitai.red).")


@app.post("/api/civitai/token")
def civitai_set_token(ref: CivitaiTokenRef) -> dict:
    try:
        # Write the credential with owner-only (0600) permissions.
        fd = os.open(str(_CIVITAI_TOKEN_FILE), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, ref.token.strip().encode())
        finally:
            os.close(fd)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"set": bool(ref.token.strip())}


@app.get("/api/civitai/token")
def civitai_token_status() -> dict:
    return {"set": bool(_civitai_token())}


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    """Block non-https redirects and strip the bearer token on cross-host hops
    (so a civitai.com 302 to a CDN can't exfiltrate the API key)."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        newreq = super().redirect_request(req, fp, code, msg, headers, newurl)
        if newreq is None:
            return None
        if urllib.parse.urlparse(newurl).scheme != "https":
            return None
        old = (urllib.parse.urlparse(req.full_url).hostname or "").lower()
        new = (urllib.parse.urlparse(newurl).hostname or "").lower()
        if new != old:
            for store in ("headers", "unredirected_hdrs"):
                d = getattr(newreq, store, None)
                if isinstance(d, dict):
                    for k in [k for k in d if k.lower() == "authorization"]:
                        d.pop(k, None)
        return newreq


_civitai_opener = urllib.request.build_opener(_SafeRedirect)


def _civitai_request(url: str, token: str = "", timeout: int = 25):
    if urllib.parse.urlparse(url).scheme != "https":
        raise HTTPException(status_code=400, detail="Invalid URL.")
    req = urllib.request.Request(url, headers={"User-Agent": "SaglitzPhotoStudio"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    return _civitai_opener.open(req, timeout=timeout)


@app.get("/api/civitai/search")
def civitai_search(query: str = "", type: str = "Checkpoint",
                   page: int = 1, token: str = "", nsfw: bool = False) -> list[dict]:
    """Browse Civitai models. type: Checkpoint | LORA."""
    params = {"limit": 24, "nsfw": "true" if nsfw else "false"}
    if query:
        params["query"] = query        # Civitai uses cursor paging with query, not page
    else:
        params["page"] = max(1, page)
    if type and type != "All":
        params["types"] = type
    url = "https://civitai.com/api/v1/models?" + urllib.parse.urlencode(params)
    try:
        with _civitai_request(url, token or _civitai_token()) as r:
            data = json.loads(r.read())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Civitai unreachable: {exc}")
    def _file_dict(x: dict) -> dict:
        meta = x.get("metadata") or {}
        return {
            "name": x.get("name") or "model.safetensors",
            "size_mb": round((x.get("sizeKB") or 0) / 1024),
            "type": x.get("type") or "Model",
            "fp": meta.get("fp"),            # fp16 / fp32 / bf16
            "format": meta.get("format"),    # SafeTensor / PickleTensor
            "precision": meta.get("size"),   # full / pruned
            "download_url": x.get("downloadUrl"),
            "primary": bool(x.get("primary")),
        }

    out = []
    for m in data.get("items", []):
        if m.get("nsfw") and not nsfw:
            continue
        model_versions = m.get("modelVersions") or []
        if not model_versions:
            continue
        # Build the per-version file lists so the app can offer version + file choice.
        versions = []
        for v in model_versions:
            vfiles = [_file_dict(x) for x in (v.get("files") or []) if x.get("downloadUrl")]
            # Weight files first (Model/Pruned Model), drop pure VAE/Config rows.
            weight_files = [f for f in vfiles
                            if "model" in (f.get("type") or "").lower()] or vfiles
            if not weight_files:
                continue
            versions.append({
                "id": v.get("id"), "name": v.get("name") or "v",
                "base": v.get("baseModel"), "files": weight_files,
            })
        if not versions:
            continue

        v0 = model_versions[0]
        default_files = versions[0]["files"]
        f = next((x for x in default_files if x.get("primary")), default_files[0])
        if nsfw:
            imgs = v0.get("images") or []
        else:
            imgs = [i for i in (v0.get("images") or []) if not i.get("nsfw") or i.get("nsfwLevel", 0) <= 1]
        out.append({
            "id": m.get("id"), "name": m.get("name"), "type": m.get("type"),
            "base": versions[0]["base"],
            "size_mb": f.get("size_mb") or 0,
            "download_url": f.get("download_url"),
            "filename": f.get("name") or f"{m.get('name','model')}.safetensors",
            "thumb": (imgs[0].get("url") if imgs else None),
            "nsfw": m.get("nsfw", False) or False,
            "versions": versions,
        })
    return out


class CivitaiDownloadRef(BaseModel):
    download_url: str
    filename: str
    kind: str            # "checkpoint" | "lora"
    token: str = ""
    nsfw: bool = False
    base: str = ""       # Civitai baseModel hint (e.g. "SDXL 1.0", "Pony", "Flux.1 D")


_civitai_dl: dict[str, Any] = {"status": "idle"}
_civitai_lock = threading.Lock()
_civitai_cancel = False                 # set by the cancel endpoint; the worker checks it


class _DLCancelled(Exception):
    """Raised inside a download worker when the user cancels."""


def _civitai_download_worker(url: str, filename: str, kind: str, token: str, base_hint: str = "") -> None:
    try:
        dest_dir = DT_LORAS_DIR if kind == "lora" else DT_MODELS_DIR
        fn = os.path.basename(filename) or "civitai_model.safetensors"
        if os.path.splitext(fn)[1].lower() not in (".safetensors", ".ckpt", ".pt", ".bin"):
            fn += ".safetensors"
        tmp = dest_dir / (fn + ".part")
        with _civitai_request(url, token, timeout=60) as r, open(tmp, "wb") as out:
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            while True:
                if _civitai_cancel:
                    raise _DLCancelled()
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if done > _MAX_DOWNLOAD_BYTES:    # guard against a runaway/hostile stream filling the disk
                    raise RuntimeError("Download exceeded the size limit.")
                with _civitai_lock:
                    _civitai_dl.update(status="downloading", done_mb=round(done / 1e6),
                                       total_mb=round(total / 1e6), file=fn)
        final = dest_dir / fn
        if final.exists():                        # never silently clobber a curated/existing weight
            final = dest_dir / f"{final.stem}_{uuid.uuid4().hex[:6]}{final.suffix}"
            fn = final.name
        os.replace(tmp, final)
        if kind == "lora":
            info = {"base": None, "version": None}
            if final.suffix.lower() == ".safetensors":
                info.update(_detect_lora_base(_read_safetensors_meta(str(final))))
            if not info.get("base") and base_hint:
                info.update(_base_from_hint(base_hint))
            _lora_meta_path(final.name).write_text(json.dumps(info))
            with _civitai_lock:
                _civitai_dl.update(status="done", file=final.name, kind="lora",
                                   base=info.get("base"), version=info.get("version"))
        else:
            # Snapshot the catalog so we can report exactly which ckpt the import produced.
            before = {m["ckpt"] for m in _dt_full_catalog(refresh=True)}
            proc = _run(
                [_DT_CLI, "models", "import", str(final), "--models-dir", str(DT_MODELS_DIR), "--replace"],
                capture_output=True, text=True, timeout=1800)
            after_rows = _dt_full_catalog(refresh=True)
            after = {m["ckpt"] for m in after_rows}
            new_ckpts = [c for c in (after - before) if _dt_downloaded(c)]
            produced = new_ckpts[0] if new_ckpts else None
            ok = proc.returncode == 0 and produced is not None
            if not ok:
                # Failed/unsupported import: clean up any tiny orphan stub it left behind
                # so it doesn't clutter the catalog as a dead model.
                for c in new_ckpts:
                    _remove_dead_stub(c)
                detail = (proc.stderr or proc.stdout or "").strip()
                if not detail:
                    detail = "This model architecture could not be imported into the Draw Things engine (may be unsupported)."
                with _civitai_lock:
                    _civitai_dl.update(status="error", file=final.name, kind="checkpoint",
                                       log=detail[-500:])
            else:
                with _civitai_lock:
                    _civitai_dl.update(status="done", file=final.name, kind="checkpoint",
                                       produced_ckpt=produced, base=base_hint or None,
                                       log=(proc.stdout or "")[-200:])
    except _DLCancelled:
        try:
            tmp.unlink(missing_ok=True)
        except (NameError, OSError):
            pass
        with _civitai_lock:
            _civitai_dl.update(status="idle")
    except Exception as exc:
        try:                          # drop the orphaned .part so failures don't pile up GBs
            tmp.unlink(missing_ok=True)
        except (NameError, OSError):
            pass
        with _civitai_lock:
            _civitai_dl.update(status="error", log=str(exc))


@app.post("/api/civitai/download")
def civitai_download(ref: CivitaiDownloadRef) -> dict:
    global _civitai_cancel
    _validate_civitai_url(ref.download_url, allow_nsfw=ref.nsfw)
    with _civitai_lock:
        if _civitai_dl.get("status") == "downloading":
            raise HTTPException(status_code=409, detail="A Civitai download is already in progress.")
        _civitai_dl.clear()
        _civitai_dl.update(status="downloading", done_mb=0, total_mb=0)
    _civitai_cancel = False        # clear any stale cancel from a prior run
    threading.Thread(target=_civitai_download_worker,
                     args=(ref.download_url, ref.filename, ref.kind,
                           ref.token or _civitai_token(), ref.base), daemon=True).start()
    return {"status": "downloading"}


@app.post("/api/civitai/download/cancel")
def civitai_download_cancel() -> dict:
    """Stop the in-flight Civitai download (the worker wipes its .part)."""
    global _civitai_cancel
    _civitai_cancel = True
    return {"status": "cancelling"}


@app.get("/api/civitai/download/status")
def civitai_download_status() -> dict:
    with _civitai_lock:
        return dict(_civitai_dl)


@app.get("/media/{project}/{fname}")
def get_media(project: str, fname: str) -> FileResponse:
    if ".." in project or ".." in fname or "/" in fname:
        raise HTTPException(status_code=404, detail="Not found")
    path = PROJECTS / project / fname
    if not path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path)


@app.get("/outputs/{fname}")
def get_output(fname: str) -> FileResponse:
    # Legacy images (pre-projects). Kept so old history entries still resolve.
    path = OUTPUTS / fname
    if not path.exists() or ".." in fname:
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path)


# === Cloud / BYOK: ElevenLabs TTS + fal.ai video ================================

def _synth_tts(script: str, td: str, engine: str, voice: Optional[str], voice_id: Optional[str]) -> str:
    """Render narration to an audio file inside `td`; return its path.
    engine='elevenlabs' uses the BYOK key (natural voices); else macOS `say`."""
    if (engine or "local") == "elevenlabs":
        key = _load_keys().get("elevenlabs")
        if not key:
            raise HTTPException(status_code=400, detail="No ElevenLabs key. Settings → Cloud keys.")
        vid = voice_id or "21m00Tcm4TlvDq8ikWAM"   # 'Rachel' default
        out = os.path.join(td, "vo.mp3")
        try:
            with httpx.Client(timeout=httpx.Timeout(connect=10, read=120, write=30, pool=10)) as c:
                r = c.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{vid}",
                    headers={"xi-api-key": key, "Content-Type": "application/json"},
                    json={"text": script, "model_id": "eleven_multilingual_v2",
                          "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}})
            if r.status_code >= 400:
                raise HTTPException(status_code=502, detail=f"ElevenLabs rejected ({r.status_code}): {r.text[:200]}")
            with open(out, "wb") as f:
                f.write(r.content)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"ElevenLabs call failed: {exc}")
        if not os.path.isfile(out) or os.path.getsize(out) < 500:
            raise HTTPException(status_code=502, detail="ElevenLabs could not generate audio.")
        return out
    say = shutil.which("say")
    if not say:
        raise HTTPException(status_code=500, detail="say not found.")
    out = os.path.join(td, "vo.aiff")
    _run([say, "-v", voice or "Samantha", "-o", out, script], capture_output=True)
    if not os.path.isfile(out):
        raise HTTPException(status_code=500, detail="Could not generate voiceover.")
    return out


_TTS_NOVELTY = {"Bad News", "Bells", "Boing", "Bubbles", "Cellos", "Organ", "Trinoids",
                "Zarvox", "Whisper", "Jester", "Bahh", "Wobble", "Albert", "Fred",
                "Junior", "Superstar", "Good News", "Ralph", "Kathy", "Hysterical",
                "Deranged", "Pipe Organ"}


@app.get("/api/tts/voices")
def tts_voices() -> dict:
    """Usable local macOS voices, plus ElevenLabs voices when a key is set."""
    local: list[str] = []
    say = shutil.which("say")
    if say:
        try:
            out = _run([say, "-v", "?"], capture_output=True, text=True).stdout
            for line in out.splitlines():
                m = re.match(r"^(.+?)\s+([a-z]{2}_[A-Z]{2})\s", line)
                if not m:
                    continue
                name, loc = m.group(1).strip(), m.group(2)
                if loc.startswith("en") and name not in _TTS_NOVELTY and "(" not in name:
                    local.append(name)
        except Exception:
            pass
    cloud: list[dict] = []
    key = _load_keys().get("elevenlabs")
    if key:
        try:
            with httpx.Client(timeout=10) as c:
                r = c.get("https://api.elevenlabs.io/v1/voices", headers={"xi-api-key": key})
                for v in r.json().get("voices", []):
                    cloud.append({"id": v.get("voice_id"), "name": v.get("name")})
        except Exception:
            pass
    return {"local": local, "elevenlabs": cloud}


def _safe_media_path(path: str) -> str:
    """Confine an image that is about to be UPLOADED to a third party (fal.ai) to
    the projects/output tree, so a crafted request can't exfiltrate an arbitrary
    local file off the user's machine. The app stores all references in projects."""
    real = os.path.realpath(path)
    roots = [os.path.realpath(str(PROJECTS))] + [os.path.realpath(r) for r in _storage_roots()]
    if not any(real == r or real.startswith(r + os.sep) for r in roots):
        raise HTTPException(status_code=400, detail="Image can only be loaded from a project folder.")
    if not os.path.isfile(real) or os.path.splitext(real)[1].lower() not in _ALLOWED_IMG_EXT:
        raise HTTPException(status_code=400, detail="Invalid image.")
    return real


def _data_uri(path: str) -> str:
    ext = os.path.splitext(path)[1].lstrip(".").lower() or "png"
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    with open(path, "rb") as f:
        return f"data:image/{mime};base64," + base64.b64encode(f.read()).decode()


def _find_video_url(res):
    """fal results vary: {video:{url}} | {videos:[{url}]} | {url}."""
    if not isinstance(res, dict):
        return None
    v = res.get("video")
    if isinstance(v, dict) and v.get("url"):
        return v["url"]
    vs = res.get("videos")
    if isinstance(vs, list) and vs and isinstance(vs[0], dict):
        return vs[0].get("url")
    return res.get("url")


class CloudVideoRequest(BaseModel):
    project: Optional[str] = None
    model: str                              # fal.ai model id
    prompt: str
    image_path: Optional[str] = None        # local first frame (i2v)
    tail_image_path: Optional[str] = None   # local last frame (FLF2V, if model supports)
    aspect: str = "16:9"
    duration: int = 5
    resolution: Optional[str] = None        # "480p" | "720p" | "1080p"


_FAL_HOST_RE = re.compile(r"^https://([a-z0-9-]+\.)*fal\.(run|ai)(/|$)", re.IGNORECASE)
_FAL_MODEL_RE = re.compile(r"^[\w][\w./-]{1,120}$")


def _fal_url_ok(url: Optional[str]) -> bool:
    return bool(url and _FAL_HOST_RE.match(url))


# Result videos are served from fal's media CDN (fal.media), not the API host.
_FAL_MEDIA_RE = re.compile(r"^https://([a-z0-9-]+\.)*fal\.(run|ai|media)(/|$)", re.IGNORECASE)


def _fal_media_ok(url: Optional[str]) -> bool:
    return bool(url and _FAL_MEDIA_RE.match(url))


@app.post("/api/generate/video/cloud")
def generate_video_cloud(req: CloudVideoRequest) -> dict:
    """Generate a clip on fal.ai (BYOK) — Seedance / Kling / Veo / Wan etc. —
    poll the queue, download the result into the project."""
    key = _load_keys().get("fal")
    if not key:
        raise HTTPException(status_code=400, detail="No fal.ai key. Settings → Cloud keys.")
    if not (req.prompt or "").strip():
        raise HTTPException(status_code=400, detail="Empty prompt.")
    if not _FAL_MODEL_RE.match(req.model.strip().strip("/")):
        raise HTTPException(status_code=400, detail="Invalid model id.")
    proj = _safe_project(req.project)
    d = _project_dir(proj)
    headers = {"Authorization": f"Key {key}", "Content-Type": "application/json"}
    body: dict = {"prompt": req.prompt, "aspect_ratio": req.aspect, "duration": str(req.duration)}
    if req.resolution:
        body["resolution"] = req.resolution
    if req.image_path:
        body["image_url"] = _data_uri(_safe_media_path(req.image_path))
    if req.tail_image_path:
        body["tail_image_url"] = _data_uri(_safe_media_path(req.tail_image_path))
    base = f"https://queue.fal.run/{req.model.strip().strip('/')}"
    out_name = f"cloud_{uuid.uuid4().hex[:6]}.mp4"
    out = d / out_name
    try:
        with httpx.Client(timeout=httpx.Timeout(connect=10, read=90, write=90, pool=10)) as c:
            sub = c.post(base, headers=headers, json=body)
            if sub.status_code >= 400:
                raise HTTPException(status_code=502, detail=f"fal.ai rejected ({sub.status_code}): {sub.text[:300]}")
            j = sub.json()
            status_url = j.get("status_url")
            response_url = j.get("response_url")
            if not status_url or not response_url:
                raise HTTPException(status_code=502, detail=f"fal.ai unexpected response: {str(j)[:200]}")
            # Never send the API key to a non-fal host (guards against a spoofed
            # response redirecting our Authorization header off-domain).
            if not (_fal_url_ok(status_url) and _fal_url_ok(response_url)):
                raise HTTPException(status_code=502, detail="fal.ai unexpected response URL.")
            for _ in range(360):                       # ~12 min
                time.sleep(2.0)
                stj = c.get(status_url, headers=headers).json()
                s = stj.get("status")
                if s == "COMPLETED":
                    break
                if s in ("ERROR", "FAILED"):
                    raise HTTPException(status_code=502, detail=f"fal.ai generation error: {str(stj)[:200]}")
            else:
                raise HTTPException(status_code=504, detail="fal.ai timeout (12 min).")
            res = c.get(response_url, headers=headers).json()
            video_url = _find_video_url(res)
            if not video_url:
                raise HTTPException(status_code=502, detail=f"fal.ai result has no video: {str(res)[:200]}")
            if not _fal_media_ok(video_url):   # only fetch the result from a fal-owned host (incl. fal.media CDN)
                raise HTTPException(status_code=502, detail="fal.ai unexpected result URL.")
            with c.stream("GET", video_url) as r:
                r.raise_for_status()
                done = 0
                with open(out, "wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
                        done += len(chunk)
                        if done > _MAX_DOWNLOAD_BYTES:   # guard against a runaway result download
                            raise HTTPException(status_code=502, detail="Result video exceeded the size limit.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"fal.ai call failed: {exc}")
    if not out.is_file() or out.stat().st_size < 1000:
        raise HTTPException(status_code=502, detail="Could not download cloud video.")
    return {"file": out_name, "project": proj, "url": f"/media/{proj}/{out_name}"}


# === Audio Studio: local Kokoro TTS (sentence-by-sentence) ======================
_kokoro_pipes: dict[str, Any] = {}
_kokoro_lock = threading.Lock()

# The voice-id PREFIX letter IS Kokoro's lang_code: a=US, b=UK English, e=Spanish,
# f=French, i=Italian, p=Portuguese(BR), h=Hindi · 2nd letter f=female, m=male.
# (Japanese 'j' / Chinese 'z' need extra misaki deps — added later.)
KOKORO_VOICES = [
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "af_aoede", "af_kore", "af_nova",
    "am_adam", "am_michael", "am_onyx", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_puck",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    "ef_dora", "em_alex", "em_santa",
    "ff_siwis",
    "if_sara", "im_nicola",
    "pf_dora", "pm_alex", "pm_santa",
]


def _kokoro(lang: str):
    p = _kokoro_pipes.get(lang)
    if p is None:
        from kokoro import KPipeline
        p = KPipeline(lang_code=lang)
        _kokoro_pipes[lang] = p
    return p


class AudioTTSRequest(BaseModel):
    project: Optional[str] = None
    text: str
    engine: str = "kokoro"           # kokoro | piper | elevenlabs
    voice: str = "af_heart"          # kokoro voice id OR piper voice id
    voice_id: Optional[str] = None   # elevenlabs voice id
    speed: float = 1.0


@app.get("/api/audio/voices")
def audio_voices() -> dict:
    """Local Kokoro voices, plus ElevenLabs voices when a key is set."""
    cloud: list[dict] = []
    key = _load_keys().get("elevenlabs")
    if key:
        try:
            with httpx.Client(timeout=10) as c:
                r = c.get("https://api.elevenlabs.io/v1/voices", headers={"xi-api-key": key})
                for v in r.json().get("voices", []):
                    cloud.append({"id": v.get("voice_id"), "name": v.get("name")})
        except Exception:
            pass
    piper = [{"id": m["id"], "lang": m["lang"], "gender": m.get("gender")}
             for m in PIPER_CATALOG if _piper_downloaded(m["id"])]
    return {"kokoro": KOKORO_VOICES, "piper": piper, "elevenlabs": cloud}


def _audio_tag(engine: str, voice: str) -> str:
    """Short language tag for the output filename (tts_<tag>_<hash>.wav)."""
    if engine == "piper":
        return (voice.split("_", 1)[0] or "xx")[:2].lower()         # tr, de, ru…
    if engine == "elevenlabs":
        return "el"
    return {"a": "en", "b": "en", "e": "es", "f": "fr", "i": "it", "p": "pt"}.get(voice[:1], "en")


@app.post("/api/audio/tts")
def audio_tts(req: AudioTTSRequest) -> dict:
    """Synthesize narration to a .wav in the project (local Kokoro or cloud EL)."""
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text.")
    proj = _safe_project(req.project)
    d = _project_dir(proj)
    out_name = f"tts_{_audio_tag(req.engine, req.voice)}_{uuid.uuid4().hex[:6]}.wav"
    out = d / out_name
    if req.engine == "elevenlabs":
        td = tempfile.mkdtemp()
        try:
            mp3 = _synth_tts(text, td, "elevenlabs", None, req.voice_id)
            ff = shutil.which("ffmpeg")
            if not ff:
                raise HTTPException(status_code=500, detail="ffmpeg not found.")
            _run([ff, "-y", "-v", "error", "-i", mp3, str(out)], capture_output=True)
        finally:
            import shutil as _sh
            _sh.rmtree(td, ignore_errors=True)
    elif req.engine == "piper":
        import wave
        if not _piper_downloaded(req.voice):
            raise HTTPException(status_code=400, detail="This voice isn't downloaded — get it from Audio → Models.")
        try:
            v = _piper_voice(req.voice)
            with wave.open(str(out), "wb") as wf:
                v.synthesize_wav(text, wf)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Piper generation error: {exc}")
    else:
        import numpy as np
        import soundfile as sf
        lang = req.voice[0] if req.voice else "a"   # voice prefix IS the Kokoro lang code
        try:
            def _run_kokoro():
                with _kokoro_lock:
                    pipe = _kokoro(lang)
                    return [np.asarray(audio) for _, _, audio in
                            pipe(text, voice=req.voice, speed=max(0.5, min(2.0, req.speed)))]
            chunks = _mlx(_run_kokoro)          # MLX must run on the engine thread
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Kokoro generation error: {exc}")
        if not chunks:
            raise HTTPException(status_code=500, detail="Could not generate audio.")
        sf.write(str(out), np.concatenate(chunks), 24000)
    if not out.is_file() or out.stat().st_size < 500:
        raise HTTPException(status_code=500, detail="Could not generate audio.")
    return {"file": out_name, "project": proj, "url": f"/media/{proj}/{out_name}",
            "duration": round(_adur(out), 2)}


@app.get("/api/audios")
def list_audios(project: Optional[str] = None) -> list[dict]:
    proj = _safe_project(project)
    d = _project_dir(proj)
    out = []
    for p in sorted(d.glob("*.wav"), key=lambda x: x.stat().st_mtime, reverse=True):
        out.append({"file": p.name, "project": proj, "url": f"/media/{proj}/{p.name}"})
    for p in sorted(d.glob("*.mp3"), key=lambda x: x.stat().st_mtime, reverse=True):
        out.append({"file": p.name, "project": proj, "url": f"/media/{proj}/{p.name}"})
    return out


# === Piper — downloadable multilingual local voices (TR, DE, + many) ============
PIPER_DIR = ROOT / "piper-voices"
PIPER_DIR.mkdir(exist_ok=True)
_piper_cache: dict[str, Any] = {}
_piper_lock = threading.Lock()

# Curated catalog. `hf` = path under huggingface rhasspy/piper-voices/<hf>.onnx(.json).
PIPER_CATALOG = [
    {"id": "tr_TR-dfki-medium", "lang": "Turkish", "gender": "M", "hf": "tr/tr_TR/dfki/medium/tr_TR-dfki-medium"},
    {"id": "de_DE-thorsten-medium", "lang": "German", "gender": "M", "hf": "de/de_DE/thorsten/medium/de_DE-thorsten-medium"},
    {"id": "de_DE-kerstin-low", "lang": "German", "gender": "F", "hf": "de/de_DE/kerstin/low/de_DE-kerstin-low"},
    {"id": "ru_RU-dmitri-medium", "lang": "Russian", "gender": "M", "hf": "ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium"},
    {"id": "ru_RU-irina-medium", "lang": "Russian", "gender": "F", "hf": "ru/ru_RU/irina/medium/ru_RU-irina-medium"},
    {"id": "ar_JO-kareem-medium", "lang": "Arabic", "gender": "M", "hf": "ar/ar_JO/kareem/medium/ar_JO-kareem-medium"},
    {"id": "pl_PL-gosia-medium", "lang": "Polish", "gender": "F", "hf": "pl/pl_PL/gosia/medium/pl_PL-gosia-medium"},
    {"id": "uk_UA-ukrainian_tts-medium", "lang": "Ukrainian", "gender": "F", "hf": "uk/uk_UA/ukrainian_tts/medium/uk_UA-ukrainian_tts-medium"},
    {"id": "nl_NL-mls-medium", "lang": "Dutch", "gender": "M", "hf": "nl/nl_NL/mls/medium/nl_NL-mls-medium"},
    {"id": "ro_RO-mihai-medium", "lang": "Romanian", "gender": "M", "hf": "ro/ro_RO/mihai/medium/ro_RO-mihai-medium"},
    {"id": "sv_SE-nst-medium", "lang": "Swedish", "gender": "M", "hf": "sv/sv_SE/nst/medium/sv_SE-nst-medium"},
]


def _piper_downloaded(vid: str) -> bool:
    return (PIPER_DIR / f"{vid}.onnx").exists() and (PIPER_DIR / f"{vid}.onnx.json").exists()


def _piper_voice(vid: str):
    with _piper_lock:
        v = _piper_cache.get(vid)
        if v is None:
            from piper import PiperVoice
            v = PiperVoice.load(str(PIPER_DIR / f"{vid}.onnx"))
            _piper_cache[vid] = v
        return v


@app.get("/api/audio/piper/catalog")
def piper_catalog() -> list[dict]:
    return [{**m, "downloaded": _piper_downloaded(m["id"])} for m in PIPER_CATALOG]


class PiperRef(BaseModel):
    id: str


@app.post("/api/audio/piper/download")
def piper_download(ref: PiperRef) -> dict:
    m = next((x for x in PIPER_CATALOG if x["id"] == ref.id), None)
    if not m:
        raise HTTPException(status_code=400, detail="Unknown voice.")
    if _piper_downloaded(ref.id):
        return {"status": "done"}
    base = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
    try:
        with httpx.Client(timeout=httpx.Timeout(connect=10, read=300, write=30, pool=10),
                          follow_redirects=True) as c:
            for ext in (".onnx", ".onnx.json"):
                dst = PIPER_DIR / f"{ref.id}{ext}"
                with c.stream("GET", f"{base}/{m['hf']}{ext}") as r:
                    r.raise_for_status()
                    with open(dst, "wb") as f:
                        for chunk in r.iter_bytes():
                            f.write(chunk)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Download failed: {exc}")
    return {"status": "done" if _piper_downloaded(ref.id) else "error"}


# === Audio Phase B: voice cloning (F5-TTS MLX) + Whisper transcription ==========
_f5_lock = threading.Lock()
WHISPER_REPO = "mlx-community/whisper-base-mlx"


def _transcribe(path: str) -> str:
    import mlx_whisper
    return _mlx(lambda: mlx_whisper.transcribe(path, path_or_hf_repo=WHISPER_REPO))["text"].strip()


class AudioCloneRequest(BaseModel):
    project: Optional[str] = None
    text: str                        # what to say in the cloned voice
    ref_file: str                    # reference audio: project filename (kept in project)
    ref_text: Optional[str] = None   # transcript of the reference (auto if absent)
    engine: str = "f5"               # f5 (local, EN/ZH) | elevenlabs (cloud, all langs)


def _safe_project_file(d: Path, name: str) -> str:
    """Resolve `name` to a file INSIDE project dir `d` only (basename, realpath-
    confirmed containment). Rejects absolute paths / traversal — the app copies
    any picked reference into the project first, so this is never limiting."""
    p = (d / os.path.basename(name)).resolve()
    if str(p).startswith(str(d.resolve()) + os.sep) and p.is_file():
        return str(p)
    raise HTTPException(status_code=400, detail="Reference audio not found.")


@app.post("/api/audio/transcribe")
def audio_transcribe(req: AudioCloneRequest) -> dict:
    """Transcribe a reference clip so the user can review/edit it before cloning."""
    proj = _safe_project(req.project)
    d = _project_dir(proj)
    refp = _safe_project_file(d, req.ref_file)
    try:
        return {"text": _transcribe(refp)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")


def _elevenlabs_clone(refp: str, text: str, key: str, out_path: str) -> None:
    """Instant voice clone via ElevenLabs (multilingual, commercial): add a temp
    voice from the reference, synthesize, then delete the voice (no slot kept)."""
    headers = {"xi-api-key": key}
    with httpx.Client(timeout=httpx.Timeout(connect=10, read=180, write=60, pool=10)) as c:
        with open(refp, "rb") as f:
            add = c.post("https://api.elevenlabs.io/v1/voices/add", headers=headers,
                         data={"name": "saglitz_tmp_clone"},
                         files={"files": (os.path.basename(refp), f, "audio/wav")})
        if add.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"ElevenLabs clone rejected ({add.status_code}): {add.text[:200]}")
        vid = add.json().get("voice_id")
        if not vid:
            raise HTTPException(status_code=502, detail="ElevenLabs did not produce audio.")
        try:
            tts = c.post(f"https://api.elevenlabs.io/v1/text-to-speech/{vid}",
                         headers={**headers, "Content-Type": "application/json"},
                         json={"text": text, "model_id": "eleven_multilingual_v2"})
            if tts.status_code >= 400:
                raise HTTPException(status_code=502, detail=f"ElevenLabs TTS error ({tts.status_code}): {tts.text[:200]}")
            mp3 = out_path + ".mp3"
            with open(mp3, "wb") as g:
                g.write(tts.content)
            ff = shutil.which("ffmpeg")
            _run([ff, "-y", "-v", "error", "-i", mp3, out_path], capture_output=True)
            try:
                os.remove(mp3)
            except OSError:
                pass
        finally:
            try:
                c.delete(f"https://api.elevenlabs.io/v1/voices/{vid}", headers=headers)
            except Exception:
                pass


@app.post("/api/audio/clone")
def audio_clone(req: AudioCloneRequest) -> dict:
    """Clone the voice in `ref_file` and speak `text` in it — F5 (local, EN/ZH) or
    ElevenLabs (cloud, all languages incl. Turkish, commercial)."""
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text.")
    proj = _safe_project(req.project)
    d = _project_dir(proj)
    refp = _safe_project_file(d, req.ref_file)
    out_name = f"clone_{'el' if req.engine == 'elevenlabs' else 'f5'}_{uuid.uuid4().hex[:6]}.wav"
    out = d / out_name
    if req.engine == "elevenlabs":
        key = _load_keys().get("elevenlabs")
        if not key:
            raise HTTPException(status_code=400, detail="No ElevenLabs key. Settings → Cloud keys.")
        _elevenlabs_clone(refp, text, key, str(out))
        ref_text = ""
    else:
        ref_text = (req.ref_text or "").strip()
        if not ref_text:
            try:
                ref_text = _transcribe(refp)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Reference transcription failed: {exc}")
        try:
            from f5_tts_mlx.generate import generate
            def _run_f5():
                with _f5_lock:
                    generate(generation_text=text, ref_audio_path=refp,
                             ref_audio_text=ref_text, output_path=str(out))
            _mlx(_run_f5)                       # MLX must run on the engine thread
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Cloning error: {exc}")
    if not out.is_file() or out.stat().st_size < 500:
        raise HTTPException(status_code=500, detail="Could not generate clone.")
    return {"file": out_name, "project": proj, "url": f"/media/{proj}/{out_name}",
            "ref_text": ref_text, "duration": round(_adur(out), 2)}


# === Audio: Sound FX / ambience generation (AudioLDM, text->audio) ==============
_audioldm = None
_audioldm_lock = threading.Lock()


_stableaudio = None


def _sfx_dev():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _sfx_backend_name() -> str:
    """Which SFX engine will be used: Stable Audio Open if an HF token is set
    (better quality + commercially usable), else local AudioLDM."""
    return "stableaudio" if _load_keys().get("hf") else "audioldm"


def _sfx_generate(prompt: str, duration: float, steps: int):
    """Returns (audio_ndarray, sample_rate, backend)."""
    import numpy as np
    import torch
    global _audioldm, _stableaudio
    dev = _sfx_dev()
    hf = _load_keys().get("hf")
    if hf:
        if _stableaudio is None:
            from diffusers import StableAudioPipeline
            _stableaudio = StableAudioPipeline.from_pretrained(
                "stabilityai/stable-audio-open-1.0", torch_dtype=torch.float32, token=hf).to(dev)
        result = _stableaudio(prompt, negative_prompt="low quality",
                              num_inference_steps=max(20, min(250, steps)),
                              audio_end_in_s=max(1.0, min(45.0, duration)))
        a = result.audios[0]
        a = a.cpu().numpy() if hasattr(a, "cpu") else np.asarray(a)
        if a.ndim == 2:                      # (channels, samples) -> (samples, channels)
            a = a.T
        return a, 44100, "stableaudio"
    if _audioldm is None:
        from diffusers import AudioLDMPipeline
        _audioldm = AudioLDMPipeline.from_pretrained(
            "cvssp/audioldm-l-full", torch_dtype=torch.float32).to(dev)
    a = _audioldm(prompt, negative_prompt="low quality, distorted, glitchy, noise",
                  guidance_scale=3.5, num_inference_steps=max(20, min(200, steps)),
                  audio_length_in_s=max(1.0, min(30.0, duration))).audios[0]
    return a, 16000, "audioldm"


class SFXRequest(BaseModel):
    project: Optional[str] = None
    prompt: str
    duration: float = 8.0
    steps: int = 120


@app.get("/api/audio/sfx/backend")
def audio_sfx_backend() -> dict:
    return {"backend": _sfx_backend_name()}


@app.post("/api/audio/sfx")
def audio_sfx(req: SFXRequest) -> dict:
    """Generate a sound effect / ambience from a text prompt (not speech)."""
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Empty description.")
    proj = _safe_project(req.project)
    d = _project_dir(proj)
    out_name = f"sfx_{uuid.uuid4().hex[:6]}.wav"
    out = d / out_name
    import soundfile as sf
    try:
        with _audioldm_lock:
            audio, sr, _ = _mlx(_sfx_generate, prompt, req.duration, req.steps)   # torch-MPS on engine thread
        sf.write(str(out), audio, sr)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Sound effect error: {exc}")
    if not out.is_file() or out.stat().st_size < 500:
        raise HTTPException(status_code=500, detail="Generation failed.")
    return {"file": out_name, "project": proj, "url": f"/media/{proj}/{out_name}",
            "duration": round(_adur(out), 2)}


# === Audio Phase C: word-level edit (Whisper timestamps + F5 resynth + splice) ==
class WordsRequest(BaseModel):
    project: Optional[str] = None
    file: str


@app.post("/api/audio/words")
def audio_words(req: WordsRequest) -> dict:
    """Transcribe with word-level timestamps so the UI can show editable words."""
    proj = _safe_project(req.project)
    d = _project_dir(proj)
    fp = _safe_project_file(d, req.file)
    import mlx_whisper
    try:
        r = _mlx(lambda: mlx_whisper.transcribe(fp, path_or_hf_repo=WHISPER_REPO, word_timestamps=True))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")
    words = [{"word": w["word"].strip(), "start": round(float(w["start"]), 3),
              "end": round(float(w["end"]), 3)}
             for s in r["segments"] for w in s.get("words", [])]
    return {"text": r["text"].strip(), "words": words}


class WordEditRequest(BaseModel):
    project: Optional[str] = None
    file: str
    index: int                 # which word to replace
    replacement: str           # new text for that word


@app.post("/api/audio/word_edit")
def audio_word_edit(req: WordEditRequest) -> dict:
    """Replace one word in a speech clip: resynthesize it in the clip's own voice
    (F5) and splice it back at the word's timestamps with short crossfades."""
    proj = _safe_project(req.project)
    d = _project_dir(proj)
    fp = _safe_project_file(d, req.file)
    repl = (req.replacement or "").strip()
    if not repl:
        raise HTTPException(status_code=400, detail="Empty word.")
    ff = shutil.which("ffmpeg")
    if not ff:
        raise HTTPException(status_code=500, detail="ffmpeg not found.")
    import mlx_whisper
    try:
        r = _mlx(lambda: mlx_whisper.transcribe(fp, path_or_hf_repo=WHISPER_REPO, word_timestamps=True))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")
    words = [w for s in r["segments"] for w in s.get("words", [])]
    if req.index < 0 or req.index >= len(words):
        raise HTTPException(status_code=400, detail="Invalid word.")
    s, e = float(words[req.index]["start"]), float(words[req.index]["end"])
    ref_text = r["text"].strip()
    dur = _adur(fp)
    td = tempfile.mkdtemp()
    try:
        newp = os.path.join(td, "new.wav")
        from f5_tts_mlx.generate import generate
        def _run_f5_edit():
            with _f5_lock:
                generate(generation_text=repl, ref_audio_path=fp, ref_audio_text=ref_text, output_path=newp)
        _mlx(_run_f5_edit)                      # MLX must run on the engine thread
        out_name = f"wedit_{uuid.uuid4().hex[:6]}.wav"
        out = d / out_name
        cf = 0.03
        has_pre, has_post = s > 0.05, e < dur - 0.05
        if has_pre and has_post:
            fc = (f"[0]atrim=0:{s:.3f},aresample=24000[pre];"
                  f"[0]atrim={e:.3f},aresample=24000,asetpts=PTS-STARTPTS[post];"
                  f"[1]aresample=24000[mid];"
                  f"[pre][mid]acrossfade=d={cf}[a];[a][post]acrossfade=d={cf}[out]")
        elif has_post:
            fc = (f"[0]atrim={e:.3f},aresample=24000,asetpts=PTS-STARTPTS[post];"
                  f"[1]aresample=24000[mid];[mid][post]acrossfade=d={cf}[out]")
        elif has_pre:
            fc = (f"[0]atrim=0:{s:.3f},aresample=24000[pre];"
                  f"[1]aresample=24000[mid];[pre][mid]acrossfade=d={cf}[out]")
        else:
            fc = "[1]aresample=24000[out]"
        rc = _run([ff, "-y", "-v", "error", "-i", fp, "-i", newp,
                             "-filter_complex", fc, "-map", "[out]", str(out)],
                            capture_output=True, text=True)
        if rc.returncode != 0 or not out.is_file():
            raise HTTPException(status_code=500, detail=f"Splice failed: {(rc.stderr or '')[-200:]}")
        return {"file": out_name, "project": proj, "url": f"/media/{proj}/{out_name}",
                "duration": round(_adur(out), 2)}
    finally:
        import shutil as _sh
        _sh.rmtree(td, ignore_errors=True)


# === Image: AI background removal (rembg + BiRefNet, MIT, best edges) ============
_rembg_session = None
_rembg_lock = threading.Lock()


def _rembg():
    global _rembg_session
    if _rembg_session is None:
        from rembg import new_session
        _rembg_session = new_session("birefnet-general")   # SOTA, MIT
    return _rembg_session


class RemoveBGRequest(BaseModel):
    project: Optional[str] = None
    file: str


@app.post("/api/image/removebg")
def image_removebg(req: RemoveBGRequest) -> dict:
    """AI background removal (BiRefNet) — cleaner edges than the instant Vision cut."""
    proj = _safe_project(req.project)
    d = _project_dir(proj)
    src = _safe_project_file(d, req.file)
    from rembg import remove
    from PIL import Image
    try:
        with _rembg_lock:
            out_img = _mlx(lambda: remove(Image.open(src).convert("RGBA"), session=_rembg()))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not remove background: {exc}")
    out_name = f"nobg_{uuid.uuid4().hex[:6]}.png"
    out = d / out_name
    out_img.save(out)
    return {"file": out_name, "project": proj, "url": f"/media/{proj}/{out_name}",
            "width": out_img.width, "height": out_img.height}


# === Built-in model weights download (mflux / HF, progress-tracked) =============
# The bundle ships code but no weights; this lets a fresh user pull a starter
# model from the app with progress, instead of an opaque inline first-load fetch.
_MODEL_REPOS: dict[str, tuple[str, str, int]] = {
    # model id -> (HF repo, human label, approx GB). Z-Image Turbo is the starter:
    # small, Apache-2.0 (commercial-safe), fast, self-contained mflux 4-bit weights.
    "z-image-turbo": ("filipstrand/Z-Image-Turbo-mflux-4bit", "Z-Image Turbo", 6),
}
_model_downloads: dict[str, dict] = {}      # model -> {status, mb, error}
_model_dl_lock = threading.Lock()


def _hf_repo_cache_mb(repo: str) -> int:
    try:
        from huggingface_hub import constants as _hc
        d = Path(_hc.HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))
    except Exception:
        return 0
    total = 0
    if d.exists():
        for p in d.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    return round(total / 1e6)


def _model_download_worker(model: str, repo: str) -> None:
    with _model_dl_lock:
        _model_downloads[model] = {"status": "downloading", "mb": _hf_repo_cache_mb(repo)}
    stop = {"v": False}

    def _watch():
        while not stop["v"]:
            with _model_dl_lock:
                if _model_downloads.get(model, {}).get("status") == "downloading":
                    _model_downloads[model]["mb"] = _hf_repo_cache_mb(repo)
            time.sleep(3)

    try:
        # Force online even if the launcher pinned HF_HUB_OFFLINE=1 (older bundles).
        import huggingface_hub.constants as _hc
        _hc.HF_HUB_OFFLINE = False
        os.environ["HF_HUB_OFFLINE"] = "0"
        from huggingface_hub import snapshot_download
        threading.Thread(target=_watch, daemon=True).start()
        snapshot_download(repo)
        stop["v"] = True
        with _model_dl_lock:
            _model_downloads[model] = {"status": "done", "mb": _hf_repo_cache_mb(repo)}
    except Exception as exc:
        stop["v"] = True
        with _model_dl_lock:
            _model_downloads[model] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


class ModelDownloadReq(BaseModel):
    model: str


@app.get("/api/model/catalog")
def model_catalog() -> dict:
    """Downloadable built-in models + whether their weights are already present."""
    out = []
    for mid, (repo, label, gb) in _MODEL_REPOS.items():
        out.append({"id": mid, "label": label, "gb": gb,
                    "downloaded": _hf_repo_cache_mb(repo) > gb * 500})   # > ~half expected
    return {"models": out}


@app.post("/api/model/download")
def model_download(req: ModelDownloadReq) -> dict:
    entry = _MODEL_REPOS.get(req.model)
    if not entry:
        raise HTTPException(status_code=400, detail=f"No downloadable weights registered for '{req.model}'.")
    with _model_dl_lock:
        if _model_downloads.get(req.model, {}).get("status") == "downloading":
            return {"status": "downloading"}
    threading.Thread(target=_model_download_worker, args=(req.model, entry[0]), daemon=True).start()
    return {"status": "started"}


@app.get("/api/model/download/status")
def model_download_status(model: str) -> dict:
    with _model_dl_lock:
        return dict(_model_downloads.get(model, {"status": "idle"}))


# === Logo Creator: styled lettermarks (reuses the image + rembg pipeline) ========
# Prompt templates per style. {t} = the letter/word, {c} = accent-colour word.
LOGO_STYLES: dict[str, dict] = {
    "crystal":  {"label": "Crystal", "p":
        "a single logo lettermark '{t}', sculpted from clear faceted crystal, sharp clean "
        "geometric facets, {c} and cyan light refraction with prism dispersion, glossy polished "
        "gemstone, bright specular highlights, centered, deep near-black background, 3D product "
        "render, studio lighting, ultra sharp, highly detailed, minimalist"},
    "amethyst": {"label": "Amethyst gem", "p":
        "a single logo lettermark '{t}', carved from a faceted {c} amethyst gemstone, sharp "
        "crystalline facets, deep saturated colour, glossy, dramatic studio lighting, centered, "
        "dark background, 3D render, ultra sharp, minimalist"},
    "gold":     {"label": "Gold", "p":
        "a single logo lettermark '{t}', polished 3D gold metal, luxury, reflective, subtle "
        "engraving, warm rim light, centered, dark background, 3D render, ultra sharp, premium, minimalist"},
    "chrome":   {"label": "Chrome", "p":
        "a single logo lettermark '{t}', liquid chrome metal, mirror-reflective, futuristic, {c} "
        "reflections, centered, dark studio background, 3D render, ultra sharp, minimalist"},
    "glass":    {"label": "Glass", "p":
        "a single logo lettermark '{t}', frosted translucent glass, soft {c} tint, gentle "
        "refraction, soft studio lighting, centered, dark background, 3D render, clean, minimalist"},
    "neon":     {"label": "Neon", "p":
        "a single logo lettermark '{t}', glowing {c} neon tube sign, bright emissive glow, subtle "
        "reflection, dark wall background, centered, night, ultra sharp, minimalist"},
    "gradient": {"label": "Flat gradient", "p":
        "a clean minimalist flat logo lettermark '{t}', smooth {c} gradient fill, modern geometric "
        "vector style, centered, plain white background, crisp, simple"},
    "matte3d":  {"label": "Soft 3D", "p":
        "a single logo lettermark '{t}', soft matte 3D inflated rounded letter, smooth clay render, "
        "pastel {c}, soft global illumination, centered, light neutral background, clean, minimalist"},
    "emblem":   {"label": "Emblem", "p":
        "an elegant emblem logo monogram '{t}', {c} and gold, refined line work, symmetrical badge, "
        "luxury brand mark, centered, dark background, crisp, detailed"},
    "holographic": {"label": "Holographic", "p":
        "a single logo lettermark '{t}', iridescent holographic foil, rainbow {c} sheen, shifting "
        "spectral reflections, glossy, centered, dark background, 3D render, ultra sharp, minimalist"},
    "marble":   {"label": "Marble", "p":
        "a single logo lettermark '{t}', polished white marble with fine {c} and gold veins, "
        "smooth stone, soft studio lighting, centered, neutral background, 3D render, luxury, minimalist"},
    "obsidian": {"label": "Obsidian", "p":
        "a single logo lettermark '{t}', carved black obsidian volcanic glass, glossy reflective, "
        "subtle {c} edge glow, dramatic lighting, centered, dark background, 3D render, ultra sharp, minimalist"},
    "rosegold": {"label": "Rose gold", "p":
        "a single logo lettermark '{t}', polished rose gold metal, luxury, soft reflections, warm "
        "pink highlights, centered, dark background, 3D render, premium, ultra sharp, minimalist"},
    "ice":      {"label": "Frost / ice", "p":
        "a single logo lettermark '{t}', frozen clear ice with frost, cold {c} and cyan tint, tiny "
        "bubbles and cracks, glossy, centered, dark background, 3D render, ultra sharp, minimalist"},
    "liquidmetal": {"label": "Liquid metal", "p":
        "a single logo lettermark '{t}', flowing liquid mercury metal, mirror-reflective chrome with "
        "{c} reflections, smooth organic surface, centered, dark studio background, 3D render, ultra sharp"},
    "paper":    {"label": "Paper cut", "p":
        "a single logo lettermark '{t}', layered paper cut craft, soft shadows between {c} paper "
        "layers, handmade, centered, clean light background, top-down, minimalist, crisp"},
    "outline":  {"label": "Monoline", "p":
        "a clean minimalist monoline logo lettermark '{t}', single even {c} outline stroke, modern "
        "geometric vector, flat, centered, plain white background, crisp, simple, elegant"},
    "wood":     {"label": "Carved wood", "p":
        "a single logo lettermark '{t}', carved polished walnut wood, natural grain, warm, soft "
        "studio lighting, centered, neutral background, 3D render, premium, minimalist"},
}
_LOGO_NEG = ("extra letters, words, paragraph, caption, watermark, signature, blurry, low quality, "
             "jpeg artifacts, deformed, distorted, busy background, frame, border, mockup")


class LogoRequest(BaseModel):
    text: str
    kind: str = "lettermark"         # lettermark | symbol | wordmark
    style: str = "crystal"
    color: str = "violet"
    font: Optional[str] = None       # OFL font id; wordmark render / img2img structure
    reference: Optional[str] = None  # example logo (path/file) to base the result on (img2img)
    reference_strength: float = 0.6  # 0..1 — higher keeps more of the reference
    model: Optional[str] = None
    seed: Optional[int] = None
    count: int = 1
    transparent: bool = False
    project: Optional[str] = None
    width: int = 1024
    height: int = 1024


# --- OFL fonts (bundled, redistributable) for exact wordmarks / font-guided logos --
_FONTS_DIR = Path(__file__).resolve().parent / "fonts"


def _list_fonts() -> list[dict]:
    out = []
    if _FONTS_DIR.exists():
        for f in sorted(_FONTS_DIR.glob("*.ttf")):
            out.append({"id": f.stem, "label": f.stem.replace("-", " ")})
    return out


def _font_file(font_id: str) -> str:
    p = (_FONTS_DIR / f"{os.path.basename(font_id or '')}.ttf").resolve()
    if not p.exists() or _FONTS_DIR.resolve() not in p.parents:
        raise HTTPException(status_code=400, detail=f"Unknown font '{font_id}'.")
    return str(p)


_LOGO_COLORS = {
    "violet": (150, 110, 240), "purple": (150, 90, 220), "indigo": (99, 102, 241),
    "gold": (212, 175, 55), "white": (245, 245, 250), "black": (18, 18, 22),
    "cyan": (80, 200, 230), "pink": (240, 120, 180), "red": (230, 70, 70),
    "blue": (80, 120, 240), "green": (70, 200, 130), "orange": (240, 150, 60),
}


def _logo_rgb(c: str) -> tuple:
    c = (c or "").strip().lower()
    if c in _LOGO_COLORS:
        return _LOGO_COLORS[c]
    if c.startswith("#"):
        h = c[1:]
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        if len(h) == 6:
            try:
                return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
            except ValueError:
                pass
    return _LOGO_COLORS["violet"]


def _fit_font(draw, text: str, font_path: str, W: int, H: int):
    from PIL import ImageFont
    size = H
    for _ in range(24):
        f = ImageFont.truetype(font_path, size)
        bb = draw.textbbox((0, 0), text, font=f)
        if (bb[2] - bb[0]) <= W * 0.84 and (bb[3] - bb[1]) <= H * 0.66:
            return f, bb
        size = max(8, int(size * 0.9))
    f = ImageFont.truetype(font_path, size)
    return f, draw.textbbox((0, 0), text, font=f)


def _render_wordmark(text, font_id, color, width, height, transparent, proj) -> dict:
    """Exact text in an OFL font — no model, fully commercial-clean."""
    from PIL import Image, ImageDraw
    W, H = _round16(width), _round16(height)
    fp = _font_file(font_id)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0) if transparent else (14, 13, 20, 255))
    d = ImageDraw.Draw(img)
    f, bb = _fit_font(d, text, fp, W, H)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    d.text(((W - tw) / 2 - bb[0], (H - th) / 2 - bb[1]), text, font=f, fill=(*_logo_rgb(color), 255))
    dd = _project_dir(proj)
    name = f"wordmark_{uuid.uuid4().hex[:6]}.png"
    img.save(dd / name)
    return {"file": name, "project": proj, "url": f"/media/{proj}/{name}", "width": W, "height": H}


def _render_letterform_init(text, font_id, width, height, proj) -> str:
    """White text on near-black — an img2img structure so styling keeps this font."""
    from PIL import Image, ImageDraw
    W, H = _round16(width), _round16(height)
    fp = _font_file(font_id)
    img = Image.new("RGB", (W, H), (8, 8, 10))
    d = ImageDraw.Draw(img)
    f, bb = _fit_font(d, text, fp, W, H)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    d.text(((W - tw) / 2 - bb[0], (H - th) / 2 - bb[1]), text, font=f, fill=(238, 238, 245))
    p = _project_dir(proj) / f"_init_{uuid.uuid4().hex[:6]}.png"
    img.save(p)
    return str(p)


@app.get("/api/logo/fonts")
def logo_fonts() -> dict:
    return {"fonts": _list_fonts()}


@app.get("/api/logo/styles")
def logo_styles() -> dict:
    return {"styles": [{"id": k, "label": v["label"]} for k, v in LOGO_STYLES.items()]}


def _logo_cutout(proj: str, fname: str) -> Optional[dict]:
    """Remove a generated logo's background -> transparent PNG (reuses rembg/BiRefNet)."""
    from rembg import remove
    from PIL import Image
    d = _project_dir(proj)
    src = _safe_project_file(d, fname)
    try:
        with _rembg_lock:
            out_img = _mlx(lambda: remove(Image.open(src).convert("RGBA"), session=_rembg()))
    except Exception:
        return None
    out_name = f"logo_{uuid.uuid4().hex[:6]}.png"
    out_img.save(d / out_name)
    return {"file": out_name, "project": proj, "url": f"/media/{proj}/{out_name}",
            "width": out_img.width, "height": out_img.height}


@app.post("/api/logo/generate")
def logo_generate(req: LogoRequest) -> dict:
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Enter a letter or short word.")
    limit = 120 if req.kind == "symbol" else 24   # symbol takes a description, not a short label
    if len(text) > limit:
        raise HTTPException(status_code=400, detail=f"Keep it under {limit} characters.")
    color = (req.color or "violet").strip() or "violet"
    proj = _safe_project(req.project or "Logos")
    n = max(1, min(6, req.count))

    # Pure wordmark — exact text in the chosen OFL font, no model (commercial-clean).
    if req.kind == "wordmark":
        if not req.font:
            raise HTTPException(status_code=400, detail="Pick a font for the wordmark.")
        res = _render_wordmark(text, req.font, color, req.width, req.height, req.transparent, proj)
        return {"kind": "wordmark", "font": req.font, "results": [res]}

    style = LOGO_STYLES.get(req.style)
    if not style:
        raise HTTPException(status_code=400, detail=f"Unknown style '{req.style}'.")
    if req.kind == "symbol":
        # reuse the chosen material, applied to a described symbol/icon instead of a
        # letter. Strip any leftover '{t}' first so a style whose template lacks the
        # "'{t}'," prefix doesn't crash .format() with a KeyError.
        material = re.sub(r"^.*?'\{t\}',\s*", "", style["p"]).replace("{t}", "").format(c=color)
        prompt = "a minimalist logo symbol / icon of " + text + ", " + material
    else:
        prompt = style["p"].format(t=text, c=color)

    # img2img structure: a reference logo (base the result on it) takes priority,
    # else the exact font letterform. Uses the commercial base model — NO ControlNet
    # (the only catalog ControlNet is FLUX Canny, which is non-commercial).
    init_path, init_strength = None, None
    if req.reference:
        init_path = _safe_image_path(req.reference)
        init_strength = max(0.2, min(0.9, req.reference_strength))
    elif req.font:
        init_path = _render_letterform_init(text, req.font, req.width, req.height, proj)
        init_strength = 0.72

    results = []
    for i in range(n):
        seed = (req.seed + i) if req.seed is not None else int.from_bytes(uuid.uuid4().bytes[:4], "big")
        gr = GenerateRequest(prompt=prompt, negative_prompt=_LOGO_NEG, model=req.model,
                             width=req.width, height=req.height, seed=seed, project=proj,
                             image_path=init_path, image_strength=init_strength)
        res = generate(gr)                       # reuses the gen lock + model dispatch
        if req.transparent:
            res = _logo_cutout(proj, res["file"]) or res
        res["seed"] = seed
        results.append(res)
    return {"prompt": prompt, "style": req.style, "font": req.font, "results": results}


# === Storage: model disk usage map + cleanup ====================================
_VIDEO_KW = ("wan", "ltx", "hunyuan", "mochi", "cog", "svd", "skyreels", "anisora", "animate")
_AUDIO_KW = ("audioldm", "f5-tts", "whisper", "stable-audio", "musicgen")


def _path_size(p: str) -> int:
    if os.path.isfile(p):
        try:
            return os.path.getsize(p)
        except OSError:
            return 0
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _storage_roots() -> list[str]:
    h = os.path.expanduser("~")
    return [str(DT_MODELS_DIR.resolve()), str(PIPER_DIR.resolve()),
            os.path.realpath(os.path.join(h, ".u2net")),
            os.path.realpath(os.path.join(h, ".cache/huggingface/hub")),
            os.path.realpath(os.path.join(h, "Documents/huggingface/models"))]


# label keyword -> (what it's for, whether deleting it affects THIS app)
_USAGE_MAP = [
    ("schnell", "Image generation · fast", True),
    ("flux.1-dev", "Image generation · quality", True),
    ("flux_1_canny", "Sketch → image (ControlNet)", True),
    ("flux_1_depth", "Depth ControlNet", True),
    ("flux_2_klein", "Image generation (FLUX.2)", True),
    ("z-image", "Image generation · photoreal", True),
    ("z_image", "Image generation · photoreal", True),
    ("qwen_image", "Image generation", True),
    ("qwen-image", "Image generation", True),
    ("sd_xl", "Image generation / inpaint (SDXL)", True),
    ("famegrid", "Inpaint / realism", True),
    ("fooocus", "SDXL inpaint", True),
    ("ssd_1b", "Image generation (compact SDXL)", True),
    ("wan", "Video generation", True),
    ("ltx", "Video generation", True),
    ("hunyuan", "Video generation", True),
    ("umt5", "Video text-encoder (dependency)", True),
    ("open_clip", "Image encoder (dependency)", True),
    ("clip_vit", "Image encoder (dependency)", True),
    ("_vae", "VAE (dependency)", True),
    ("vae_", "VAE (dependency)", True),
    ("kokoro", "Local TTS voices", True),
    ("f5-tts", "Voice cloning", True),
    ("vocos", "Voice cloning vocoder (F5 dependency)", True),
    ("whisper", "Transcription (word-edit · clone)", True),
    ("audioldm2", "Old Sound FX — UNUSED, safe to delete", False),
    ("audioldm-s", "Old small Sound FX — UNUSED", False),
    ("audioldm-l", "Sound FX (current)", True),
    ("stable-audio", "Sound FX · HQ", True),
    ("birefnet", "AI background removal", True),
    ("isnet", "Background removal (alternate — optional)", False),
    ("u2net", "Background removal (alternate — optional)", False),
    ("blip", "Image describe / interrogate", True),
    ("qwen_3", "Prompt helper (dependency)", True),
]


def _model_usage(label: str, category: str):
    s = label.lower()
    for kw, usage, in_use in _USAGE_MAP:
        if kw in s:
            return usage, in_use
    if category == "voice":
        return "TTS voice (downloaded)", True
    return "Not used by this app — safe to delete", False


@app.get("/api/storage")
def storage() -> dict:
    """Disk usage of every downloaded model, categorized, with deletable paths."""
    items = []
    home = os.path.expanduser("~")
    for p in DT_MODELS_DIR.glob("*.ckpt"):
        sz = _path_size(str(p))
        td = DT_MODELS_DIR / (p.name + "-tensordata")
        if td.exists():
            sz += _path_size(str(td))
        low = p.name.lower()
        cat = "video" if any(k in low for k in _VIDEO_KW) else "image"
        items.append({"label": p.stem, "bytes": sz, "path": str(p), "category": cat})
    for p in PIPER_DIR.glob("*.onnx"):
        items.append({"label": p.stem, "bytes": _path_size(str(p)) + _path_size(str(p) + ".json"),
                      "path": str(p), "category": "voice"})
    u2 = os.path.join(home, ".u2net")
    if os.path.isdir(u2):
        for f in os.listdir(u2):
            if f.endswith(".onnx"):
                fp = os.path.join(u2, f)
                items.append({"label": f[:-5], "bytes": _path_size(fp), "path": fp, "category": "tools"})
    hub = os.path.join(home, ".cache/huggingface/hub")
    if os.path.isdir(hub):
        for d in os.listdir(hub):
            if not d.startswith("models--"):
                continue
            fp = os.path.join(hub, d)
            name = d[len("models--"):].replace("--", "/")
            low = name.lower()
            cat = ("audio_ai" if any(k in low for k in _AUDIO_KW)
                   else "voice" if "kokoro" in low
                   else "tools" if "blip" in low or "birefnet" in low
                   else "image" if ("z-image" in low or "flux" in low or "qwen" in low or "sdxl" in low)
                   else "other")
            items.append({"label": name, "bytes": _path_size(fp), "path": fp, "category": cat})
    mflux_root = os.path.join(home, "Documents/huggingface/models")
    if os.path.isdir(mflux_root):
        for prov in os.listdir(mflux_root):
            pd = os.path.join(mflux_root, prov)
            if not os.path.isdir(pd):
                continue
            for mdl in os.listdir(pd):
                md = os.path.join(pd, mdl)
                if os.path.isdir(md):
                    items.append({"label": f"{prov}/{mdl}", "bytes": _path_size(md),
                                  "path": md, "category": "image"})
    for it in items:
        it["usage"], it["in_use"] = _model_usage(it["label"], it["category"])
    items.sort(key=lambda x: -x["bytes"])
    return {"total_bytes": sum(i["bytes"] for i in items), "items": items}


class StoragePathRequest(BaseModel):
    path: str


@app.post("/api/storage/delete")
def storage_delete(req: StoragePathRequest) -> dict:
    rp = os.path.realpath(req.path)
    roots = _storage_roots()
    if not any(rp == r or rp.startswith(r + os.sep) for r in roots):
        raise HTTPException(status_code=400, detail="Path not allowed.")
    if not os.path.exists(rp):
        raise HTTPException(status_code=404, detail="Not found.")
    import shutil as _sh
    if os.path.isdir(rp):
        _sh.rmtree(rp, ignore_errors=True)
    else:
        try:
            os.remove(rp)
        except OSError:
            pass
        for c in (rp + ".json", rp + "-tensordata"):     # ckpt/onnx companions
            if os.path.isdir(c):
                _sh.rmtree(c, ignore_errors=True)
            elif os.path.isfile(c):
                os.remove(c)
    return {"ok": True}


# === Video: auto-subtitles (Whisper transcript → burned, time-synced captions) ==
class SubtitleRequest(BaseModel):
    project: Optional[str] = None
    file: str


@app.post("/api/video/subtitle")
def video_subtitle(req: SubtitleRequest) -> dict:
    """Transcribe a clip's speech (Whisper) and burn time-synced captions in."""
    proj = _safe_project(req.project)
    d = _project_dir(proj)
    src = d / os.path.basename(req.file)
    if not src.is_file():
        raise HTTPException(status_code=400, detail="Video not found.")
    ff = shutil.which("ffmpeg")
    if not ff:
        raise HTTPException(status_code=500, detail="ffmpeg not found.")
    import mlx_whisper
    try:
        r = _mlx(lambda: mlx_whisper.transcribe(str(src), path_or_hf_repo=WHISPER_REPO))  # MLX on engine thread
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")
    segs = [(s["text"].strip(), float(s["start"]), float(s["end"]))
            for s in r.get("segments", []) if s.get("text", "").strip()]
    if not segs:
        raise HTTPException(status_code=400, detail="No speech found (silent clip).")
    W, H = _vdims(src)
    td = tempfile.mkdtemp()
    try:
        overlays = []
        for i, (txt, s, e) in enumerate(segs):
            p = os.path.join(td, f"sub{i}.png")
            _render_caption(txt, W, H, p)
            overlays.append((p, s, e))
        inputs = ["-i", str(src)]
        for p, _, _ in overlays:
            inputs += ["-i", p]
        fc = "[0:v]null[b]"
        cur = "b"
        for i, (p, s, e) in enumerate(overlays):
            lbl = f"o{i}"
            fc += f";[{cur}][{i + 1}:v]overlay=0:0:enable='between(t,{s:.2f},{e:.2f})'[{lbl}]"
            cur = lbl
        out_name = f"sub_{uuid.uuid4().hex[:6]}.mp4"
        out = d / out_name
        cmd = [ff, "-y", "-v", "error"] + inputs + [
            "-filter_complex", fc, "-map", f"[{cur}]", "-map", "0:a?",
            "-c:a", "copy", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(out)]
        rc = _run(cmd, capture_output=True, text=True)
        if rc.returncode != 0 or not out.is_file():
            raise HTTPException(status_code=500, detail=f"Subtitle render failed: {(rc.stderr or '')[-200:]}")
        return {"file": out_name, "project": proj, "url": f"/media/{proj}/{out_name}", "lines": len(segs)}
    finally:
        import shutil as _sh
        _sh.rmtree(td, ignore_errors=True)


# === Video: smart social reframe (crop to a target aspect around a focus point) ==
_REFRAME_DIMS = {"9:16": (1080, 1920), "1:1": (1080, 1080), "4:5": (1080, 1350), "16:9": (1920, 1080)}


class ReframeRequest(BaseModel):
    project: Optional[str] = None
    file: str
    aspect: str = "9:16"
    cx: float = 0.5            # focus point (normalized, top-left origin)
    cy: float = 0.5


@app.post("/api/video/reframe")
def video_reframe(req: ReframeRequest) -> dict:
    """Crop a clip to a social aspect, centered on the focus point, then scale to
    the exact platform size. (Saliency focus is computed app-side via Vision.)"""
    proj = _safe_project(req.project)
    d = _project_dir(proj)
    src = d / os.path.basename(req.file)
    if not src.is_file():
        raise HTTPException(status_code=400, detail="Video not found.")
    ff = shutil.which("ffmpeg")
    if not ff:
        raise HTTPException(status_code=500, detail="ffmpeg not found.")
    W, H = _vdims(src)
    tw, th = _REFRAME_DIMS.get(req.aspect, (1080, 1920))
    a = tw / th
    if W / H > a:
        cw, ch = int(round(H * a)), H
    else:
        cw, ch = W, int(round(W / a))
    cw, ch = max(2, min(W, cw)), max(2, min(H, ch))
    x = int(round(max(0, min(W - cw, req.cx * W - cw / 2))))
    y = int(round(max(0, min(H - ch, req.cy * H - ch / 2))))
    out_name = f"reframe_{req.aspect.replace(':', 'x')}_{uuid.uuid4().hex[:6]}.mp4"
    out = d / out_name
    fc = f"crop={cw}:{ch}:{x}:{y},scale={tw}:{th},setsar=1"
    cmd = [ff, "-y", "-v", "error", "-i", str(src), "-vf", fc,
           "-map", "0:a?", "-c:a", "copy", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(out)]
    rc = _run(cmd, capture_output=True, text=True)
    if rc.returncode != 0 or not out.is_file():
        raise HTTPException(status_code=500, detail=f"Reframe failed: {(rc.stderr or '')[-200:]}")
    return {"file": out_name, "project": proj, "url": f"/media/{proj}/{out_name}"}


# === Image → motion video (Ken Burns: slow zoom/pan, no model) ==================
_KB_DIMS = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080), "4:5": (1080, 1350)}


class KenBurnsRequest(BaseModel):
    project: Optional[str] = None
    file: str
    duration: float = 5.0
    motion: str = "zoom_in"        # zoom_in | zoom_out | pan_right | pan_left
    aspect: str = "16:9"


@app.post("/api/image/kenburns")
def image_kenburns(req: KenBurnsRequest) -> dict:
    """Animate a still image with a slow zoom/pan (ffmpeg zoompan) → an mp4 clip."""
    proj = _safe_project(req.project)
    d = _project_dir(proj)
    src = _safe_project_file(d, req.file)
    ff = shutil.which("ffmpeg")
    if not ff:
        raise HTTPException(status_code=500, detail="ffmpeg not found.")
    W, H = _KB_DIMS.get(req.aspect, (1920, 1080))
    dur = max(2.0, min(15.0, req.duration))
    fr = int(dur * 30)
    cy = "ih/2-(ih/zoom/2)"
    cx = "iw/2-(iw/zoom/2)"
    if req.motion == "zoom_out":
        z, x, y = f"max(1.4-on/{fr}*0.4,1.0)", cx, cy
    elif req.motion == "pan_right":
        z, x, y = "1.25", f"(iw-iw/zoom)*on/{fr}", cy
    elif req.motion == "pan_left":
        z, x, y = "1.25", f"(iw-iw/zoom)*(1-on/{fr})", cy
    else:  # zoom_in
        z, x, y = f"min(1.0+on/{fr}*0.4,1.4)", cx, cy
    vf = (f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
          f"scale=iw*4:ih*4,zoompan=z='{z}':x='{x}':y='{y}':d={fr}:fps=30:s={W}x{H}")
    out_name = f"motion_{uuid.uuid4().hex[:6]}.mp4"
    out = d / out_name
    cmd = [ff, "-y", "-v", "error", "-loop", "1", "-i", src, "-t", f"{dur}",
           "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30", str(out)]
    rc = _run(cmd, capture_output=True, text=True)
    if rc.returncode != 0 or not out.is_file():
        raise HTTPException(status_code=500, detail=f"Ken Burns failed: {(rc.stderr or '')[-200:]}")
    return {"file": out_name, "project": proj, "url": f"/media/{proj}/{out_name}"}


# === Audio: multi-speaker dialogue (a voice per speaker, concatenated) ==========
def _tts_to_wav(text: str, engine: str, voice: str, voice_id: Optional[str], out_path: str) -> None:
    """Synthesize one line to a 24kHz mono wav at `out_path` (kokoro/piper/EL)."""
    text = (text or "").strip()
    if engine == "elevenlabs":
        key = _load_keys().get("elevenlabs")
        if not key:
            raise HTTPException(status_code=400, detail="No ElevenLabs key.")
        td = tempfile.mkdtemp()
        try:
            mp3 = _synth_tts(text, td, "elevenlabs", None, voice_id or voice)
            ff = shutil.which("ffmpeg")
            _run([ff, "-y", "-v", "error", "-i", mp3, "-ar", "24000", "-ac", "1", out_path],
                           capture_output=True)
        finally:
            import shutil as _sh
            _sh.rmtree(td, ignore_errors=True)
    elif engine == "piper":
        import wave
        if not _piper_downloaded(voice):
            raise HTTPException(status_code=400, detail=f"Piper voice not downloaded: {voice}")
        tmp = out_path + ".raw.wav"
        with wave.open(tmp, "wb") as wf:
            _piper_voice(voice).synthesize_wav(text, wf)
        ff = shutil.which("ffmpeg")
        _run([ff, "-y", "-v", "error", "-i", tmp, "-ar", "24000", "-ac", "1", out_path],
                       capture_output=True)
        try: os.remove(tmp)
        except OSError: pass
    else:  # kokoro
        import numpy as np
        import soundfile as sf
        lang = voice[0] if voice else "a"
        def _run_kok():
            with _kokoro_lock:
                return [np.asarray(a) for _, _, a in _kokoro(lang)(text, voice=voice)]
        chunks = _mlx(_run_kok)                 # MLX must run on the engine thread
        sf.write(out_path, np.concatenate(chunks) if chunks else np.zeros(2400), 24000)


class DialogueLine(BaseModel):
    text: str
    voice: str = "af_heart"
    engine: str = "kokoro"
    voice_id: Optional[str] = None


class DialogueRequest(BaseModel):
    project: Optional[str] = None
    lines: list[DialogueLine]
    gap: float = 0.45


@app.post("/api/audio/dialogue")
def audio_dialogue(req: DialogueRequest) -> dict:
    """Render a multi-speaker conversation: each line in its own voice, joined."""
    lines = [ln for ln in (req.lines or []) if (ln.text or "").strip()]
    if not lines:
        raise HTTPException(status_code=400, detail="Empty dialogue.")
    proj = _safe_project(req.project)
    d = _project_dir(proj)
    ff = shutil.which("ffmpeg")
    if not ff:
        raise HTTPException(status_code=500, detail="ffmpeg not found.")
    td = tempfile.mkdtemp()
    try:
        gap = os.path.join(td, "gap.wav")
        _run([ff, "-y", "-v", "error", "-f", "lavfi", "-i",
                        "anullsrc=r=24000:cl=mono", "-t", f"{max(0.05, min(2.0, req.gap))}", gap],
                       capture_output=True)
        parts = []
        for i, ln in enumerate(lines):
            p = os.path.join(td, f"line{i}.wav")
            try:
                _tts_to_wav(ln.text, ln.engine, ln.voice, ln.voice_id, p)
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Could not generate line {i+1}: {exc}")
            parts.append(p)
            if i < len(lines) - 1:
                parts.append(gap)
        listf = os.path.join(td, "list.txt")
        with open(listf, "w") as f:
            for p in parts:
                f.write(f"file '{p}'\n")
        out_name = f"dialogue_{uuid.uuid4().hex[:6]}.wav"
        out = d / out_name
        rc = _run([ff, "-y", "-v", "error", "-f", "concat", "-safe", "0",
                             "-i", listf, "-c", "copy", str(out)], capture_output=True, text=True)
        if rc.returncode != 0 or not out.is_file():
            raise HTTPException(status_code=500, detail=f"Merge failed: {(rc.stderr or '')[-200:]}")
        return {"file": out_name, "project": proj, "url": f"/media/{proj}/{out_name}",
                "duration": round(_adur(out), 2)}
    finally:
        import shutil as _sh
        _sh.rmtree(td, ignore_errors=True)


# === Audio: music bed under a voiceover, with sidechain ducking =================
class MusicBedRequest(BaseModel):
    project: Optional[str] = None
    file: str                  # the voice/audio clip (project file)
    music: str                 # the background music (copied into the project)
    volume: float = 0.5        # base music level before ducking


@app.post("/api/audio/musicbed")
def audio_musicbed(req: MusicBedRequest) -> dict:
    """Lay a music bed under a voice clip; the music ducks under speech
    (sidechain compression keyed off the voice)."""
    proj = _safe_project(req.project)
    d = _project_dir(proj)
    voice = _safe_project_file(d, req.file)
    music = _safe_project_file(d, req.music)
    ff = shutil.which("ffmpeg")
    if not ff:
        raise HTTPException(status_code=500, detail="ffmpeg not found.")
    vol = max(0.05, min(1.0, req.volume))
    out_name = f"mix_{uuid.uuid4().hex[:6]}.wav"
    out = d / out_name
    fc = (f"[1:a]volume={vol},aresample=24000[m];"
          f"[m][0:a]sidechaincompress=threshold=0.02:ratio=10:attack=15:release=350[duck];"
          f"[duck][0:a]amix=inputs=2:duration=first:dropout_transition=0,aresample=24000[out]")
    cmd = [ff, "-y", "-v", "error", "-i", voice, "-i", music,
           "-filter_complex", fc, "-map", "[out]", str(out)]
    rc = _run(cmd, capture_output=True, text=True)
    if rc.returncode != 0 or not out.is_file():
        raise HTTPException(status_code=500, detail=f"Mixing failed: {(rc.stderr or '')[-200:]}")
    return {"file": out_name, "project": proj, "url": f"/media/{proj}/{out_name}",
            "duration": round(_adur(out), 2)}


# === Audio: SRT subtitle / transcript export ===================================
def _srt_time(t: float) -> str:
    h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60); ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


@app.post("/api/audio/srt")
def audio_srt(req: SubtitleRequest) -> dict:
    """Export an audio/video clip's transcript as a .srt (Whisper segments)."""
    proj = _safe_project(req.project)
    d = _project_dir(proj)
    src = _safe_project_file(d, req.file)
    import mlx_whisper
    try:
        r = _mlx(lambda: mlx_whisper.transcribe(src, path_or_hf_repo=WHISPER_REPO))  # MLX on engine thread
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")
    segs = [s for s in r.get("segments", []) if s.get("text", "").strip()]
    if not segs:
        raise HTTPException(status_code=400, detail="No speech found.")
    lines = []
    for i, s in enumerate(segs, 1):
        lines.append(f"{i}\n{_srt_time(float(s['start']))} --> {_srt_time(float(s['end']))}\n{s['text'].strip()}\n")
    base = os.path.splitext(os.path.basename(src))[0]
    out_name = f"{base}.srt"
    (d / out_name).write_text("\n".join(lines), encoding="utf-8")
    return {"file": out_name, "project": proj, "url": f"/media/{proj}/{out_name}", "lines": len(segs)}
