"""
Shared runtime for UI studios: config, generators, history helpers.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from generators import (
    GenerationResult,
    detect_hardware,
    get_text_generator,
    init_history_db,
    initialize_generators,
    load_config,
    load_prompt_library,
    resolve_text_provider,
    save_config,
    save_generation,
    save_prompt_library,
)

PROVIDER_LABELS = {
    "Local Ollama": "ollama",
    "Ollama Cloud": "ollama_cloud",
    "xAI (Grok)": "xai",
}
LABELS_BY_PROVIDER = {v: k for k, v in PROVIDER_LABELS.items()}

CONFIG = load_config()
HW = detect_hardware()

TEXT_GEN = None
IMAGE_GEN = None
VIDEO_GEN = None
VOICE_GEN = None

DB = init_history_db(CONFIG.get("history", {}).get("db_path", "./data/history.db"))
PROMPT_LIB_PATH = CONFIG.get("prompt_library", {}).get("file", "./data/prompt_library.json")
PROMPT_LIBRARY = load_prompt_library(PROMPT_LIB_PATH)

OUTPUT_DIR = Path(CONFIG.get("app", {}).get("output_dir", "./outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def log_line(message: str) -> None:
    print(f"[LuminaForge] {message}", flush=True)


@contextmanager
def log_op(action: str, **fields: Any) -> Iterator[None]:
    """Time an operation and always print a one-line result."""
    t0 = time.perf_counter()
    err: Exception | None = None
    try:
        yield
    except Exception as exc:
        err = exc
        raise
    finally:
        dt = time.perf_counter() - t0
        bits = " ".join(f"{k}={v}" for k, v in fields.items() if v not in (None, ""))
        if err is None:
            log_line(f"{action} ok {dt:.2f}s {bits}".rstrip())
        else:
            log_line(f"{action} failed {dt:.2f}s {bits} error={type(err).__name__}: {err}".rstrip())


def lazy_init():
    global TEXT_GEN, IMAGE_GEN, VIDEO_GEN, VOICE_GEN
    if TEXT_GEN is None:
        t0 = time.perf_counter()
        gens = initialize_generators(CONFIG)
        TEXT_GEN = gens["text"]
        IMAGE_GEN = gens["image"]
        VIDEO_GEN = gens["video"]
        VOICE_GEN = gens["voice"]
        log_line(f"generators ready {time.perf_counter() - t0:.2f}s")
    return TEXT_GEN, IMAGE_GEN, VIDEO_GEN, VOICE_GEN


def save_result_to_history(result: GenerationResult, project: Optional[str] = None) -> int:
    try:
        return save_generation(DB, result, project=project)
    except Exception:
        return -1


def format_metadata(result: GenerationResult) -> Dict[str, Any]:
    return result.to_dict() if hasattr(result, "to_dict") else result.metadata


def enhance_and_update(prompt: str, style: str, modality: str) -> str:
    tgen, _, _, _ = lazy_init()
    if not prompt or len(prompt.strip()) < 2:
        return prompt
    t0 = time.perf_counter()
    try:
        result = tgen.enhance_prompt(prompt, style=style, target_modality=modality)
        log_line(f"enhance {modality} ok {time.perf_counter() - t0:.2f}s style={style}")
        return result
    except Exception as exc:
        log_line(
            f"enhance {modality} failed {time.perf_counter() - t0:.2f}s "
            f"style={style} error={type(exc).__name__}: {exc}"
        )
        return prompt


def get_ollama_model_choices():
    return get_text_model_choices()


def get_text_model_choices():
    tgen, _, _, _ = lazy_init()
    try:
        models = tgen.list_models()
        if models:
            return models
    except Exception:
        pass
    provider = getattr(tgen, "provider", None) or resolve_text_provider(CONFIG.get("text", {}))
    if provider == "xai":
        return CONFIG.get("text", {}).get("xai_models") or ["grok-4.6", "grok-4.5", "grok-4"]
    return CONFIG.get("text", {}).get("available_models", ["llama3.2:3b"])


def text_model_label(provider: Optional[str] = None) -> str:
    p = provider or resolve_text_provider(CONFIG.get("text", {}))
    if p == "xai":
        return "xAI Model"
    if p == "ollama_cloud":
        return "Ollama Cloud Model"
    return "Ollama Model"


def get_image_model_choices():
    _, igen, _, _ = lazy_init()
    try:
        return igen.list_models()
    except Exception:
        return [CONFIG.get("image", {}).get("default_model", "black-forest-labs/FLUX.1-schnell")]


def get_voice_profiles():
    _, _, _, vgen = lazy_init()
    try:
        profs = vgen.list_voice_profiles()
        return [p["name"] for p in profs] + ["builtin_male", "builtin_female"]
    except Exception:
        return ["builtin_male", "builtin_female"]


def add_to_prompt_library(title: str, prompt: str, category: str) -> str:
    global PROMPT_LIBRARY
    if not title or not prompt:
        return "Title and prompt required."
    new_item = {
        "id": f"user_{int(time.time())}",
        "category": category or "User",
        "title": title,
        "prompt": prompt,
        "tags": ["user-added"],
    }
    PROMPT_LIBRARY.append(new_item)
    save_prompt_library(PROMPT_LIBRARY, PROMPT_LIB_PATH)
    return f"Added: {title}"


def get_status_markdown() -> str:
    tgen, igen, vgen, _ = lazy_init()
    txt_ok = tgen.validate() if hasattr(tgen, "validate") else False
    img_ok = igen.validate() if hasattr(igen, "validate") else False
    vid_ok = vgen.validate() if hasattr(vgen, "validate") else False

    provider = getattr(tgen, "provider", None) or resolve_text_provider(CONFIG.get("text", {}))
    if provider == "xai":
        text_label, text_icon = "xAI (Grok)", "⚡"
    elif provider == "ollama_cloud":
        text_label, text_icon = "Ollama Cloud", "☁️"
    else:
        text_label, text_icon = "Ollama (local)", "🖥️"

    return f"""
**{text_icon} {text_label} (Text)**: {'✅ Connected' if txt_ok else '❌ Not reachable'}  
**Image Backend**: {'✅ Ready' if img_ok else '⚠️ Check diffusers / ComfyUI'}  
**Video (ComfyUI)**: {'✅ Detected' if vid_ok else '⚠️ Start ComfyUI on port 8188 for best results'}  
**Hardware**: {HW.get('device')} ({HW.get('vram_gb')} GB)
"""


def load_recent_gallery_items(limit: int = 8) -> List[str]:
    from generators.utils import load_recent_history

    rows = load_recent_history(DB, limit)
    items = []
    for r in rows:
        path = r.get("output_path")
        if path and isinstance(path, str):
            if any(path.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]):
                items.append(path)
            elif path.endswith((".wav", ".mp3")):
                items.append(path)
    return items


def history_rows(search: Optional[str], limit: int, mod: str) -> List[List[Any]]:
    from generators.utils import load_recent_history

    rows = load_recent_history(DB, int(limit), None if mod == "All" else mod)
    if search:
        s = search.lower()
        rows = [r for r in rows if s in (r.get("prompt", "") + r.get("model_id", "")).lower()]
    return [
        [r.get("id"), r.get("modality"), r.get("prompt", "")[:80], r.get("model_id"), r.get("timestamp")]
        for r in rows
    ]


def apply_ollama_cloud(use_cloud_val, url_val, key_val, persist=False):
    label = "Ollama Cloud" if use_cloud_val else "Local Ollama"
    return apply_text_backend(label, None, url_val, key_val, "", persist)


def apply_text_backend(
    provider_label,
    local_url,
    cloud_url,
    ollama_key,
    xai_key,
    persist=False,
):
    global CONFIG, TEXT_GEN
    from ui.layout import mode_badge_html
    import gradio as gr

    text_section = CONFIG.setdefault("text", {})
    provider = PROVIDER_LABELS.get(provider_label, resolve_text_provider(text_section))
    text_section["provider"] = provider
    text_section["use_cloud"] = provider == "ollama_cloud"

    if local_url:
        text_section["base_url"] = str(local_url).strip()
    if cloud_url:
        text_section["cloud_base_url"] = str(cloud_url).strip()

    if ollama_key and str(ollama_key).strip() and not str(ollama_key).strip().startswith("••••"):
        text_section["api_key"] = str(ollama_key).strip()
    if xai_key and str(xai_key).strip() and not str(xai_key).strip().startswith("••••"):
        text_section["xai_api_key"] = str(xai_key).strip()

    if provider == "xai":
        current_default = str(text_section.get("default_model") or "")
        if ":" in current_default or current_default.startswith("llama"):
            text_section["default_model"] = text_section.get("xai_default_model") or "grok-4.6"
            text_section["enhancer_model"] = text_section["default_model"]
        text_section.setdefault("xai_base_url", "https://api.x.ai/v1")
        text_section.setdefault("xai_default_model", "grok-4.6")
    elif provider in ("ollama", "ollama_cloud"):
        current_default = str(text_section.get("default_model") or "")
        if current_default.lower().startswith("grok"):
            fallback = (text_section.get("available_models") or ["llama3.2:3b"])[0]
            text_section["default_model"] = fallback
            text_section["enhancer_model"] = fallback

    try:
        TEXT_GEN = get_text_generator(CONFIG)
        if persist:
            save_config(CONFIG)
        status = TEXT_GEN.get_status() if hasattr(TEXT_GEN, "get_status") else {}
        names = {
            "xai": "⚡ xAI (Grok)",
            "ollama_cloud": "☁️ Ollama Cloud",
            "ollama": "🖥️ Local Ollama",
        }
        base = status.get("base_url") or TEXT_GEN.base_url
        msg = f"✅ Switched to **{names.get(provider, provider)}**. Base: {base}"
        if persist:
            msg += " (saved to config.yaml)"
        if provider == "xai" and not getattr(TEXT_GEN, "api_key", None):
            msg += " — add an xAI key (`XAI_API_KEY` or the field below)."

        choices = get_text_model_choices()
        default = getattr(TEXT_GEN, "default_model", None)
        if default and default not in choices:
            choices = [default] + list(choices)
        return (
            msg,
            gr.update(value=mode_badge_html(provider)),
            gr.update(choices=choices, value=default, label=text_model_label(provider)),
        )
    except Exception as e:
        return f"❌ Failed to apply: {e}", gr.update(), gr.update()
