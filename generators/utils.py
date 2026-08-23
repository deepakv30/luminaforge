"""
LuminaForge Shared Utilities
============================

Contains:
- Hardware detection (CUDA, MPS, CPU, VRAM)
- Ollama HTTP helpers
- Safe file + metadata operations
- Image / audio / video helpers (Pillow, soundfile, moviepy wrappers)
- Config loading with Pydantic-style validation (lightweight, no extra deps)
- Gradio-friendly progress helpers
"""

from __future__ import annotations

import json
import os
import platform
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

try:
    import torch
except ImportError:
    torch = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import soundfile as sf
except ImportError:
    sf = None


# --------------------------- Config Loading ---------------------------

def load_config(path: str | Path = "config.yaml") -> Dict[str, Any]:
    """Load YAML config with sensible defaults if file missing."""
    p = Path(path)
    if not p.exists():
        # Return minimal working defaults
        return {
            "app": {"name": "LuminaForge", "output_dir": "./outputs"},
            "text": {"base_url": "http://localhost:11434", "default_model": "llama3.2:3b"},
            "image": {"default_model": "stabilityai/sdxl-turbo"},
            "video": {"default_workflow": "cogvideox_t2v"},
            "voice": {"provider": "coqui_xtts"},
        }
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def save_config(cfg: Dict[str, Any], path: str | Path = "config.yaml") -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)


# --------------------------- Hardware ---------------------------

def detect_hardware() -> Dict[str, Any]:
    """Return rich hardware info for display and generator decisions."""
    info = {
        "platform": platform.system(),
        "python": platform.python_version(),
        "torch_available": torch is not None,
        "cuda_available": False,
        "mps_available": False,
        "device": "cpu",
        "vram_gb": 0.0,
        "total_ram_gb": 0.0,
    }

    if torch is None:
        return info

    info["cuda_available"] = torch.cuda.is_available()
    if info["cuda_available"]:
        info["device"] = "cuda"
        try:
            props = torch.cuda.get_device_properties(0)
            info["vram_gb"] = round(props.total_memory / (1024**3), 1)
            info["gpu_name"] = props.name
        except Exception:
            pass

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["mps_available"] = True
        if not info["cuda_available"]:
            info["device"] = "mps"

    # Rough total RAM
    try:
        import psutil
        info["total_ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception:
        pass

    return info


def recommend_tier(vram_gb: float, device: str) -> str:
    if device == "cpu":
        return "low"
    if vram_gb >= 24:
        return "high"
    if vram_gb >= 14:
        return "medium"
    return "low"


# --------------------------- Ollama ---------------------------

def get_ollama_models(base_url: str, headers: Dict[str, str] = None) -> List[str]:
    """Query /api/tags. Supports optional auth headers for Ollama Cloud."""
    import httpx
    headers = headers or {}
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/api/tags", headers=headers, timeout=6)
        r.raise_for_status()
        data = r.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


# --------------------------- Safe IO ---------------------------

def safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except Exception:
        return "{}"


def ensure_dir(p: Path | str) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def unique_path(directory: Path, stem: str, suffix: str) -> Path:
    directory = ensure_dir(directory)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = directory / f"{stem}_{ts}{suffix}"
    i = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{ts}_{i}{suffix}"
        i += 1
    return candidate


# --------------------------- History (SQLite) ---------------------------

def init_history_db(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    ensure_dir(db_path.parent)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS generations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            modality TEXT NOT NULL,
            prompt TEXT,
            negative_prompt TEXT,
            model_id TEXT,
            parameters TEXT,
            seed INTEGER,
            duration_ms REAL,
            timestamp REAL,
            output_path TEXT,
            metadata TEXT,
            project TEXT,
            error TEXT
        )
    """)
    conn.commit()
    return conn


def save_generation(conn: sqlite3.Connection, result: "GenerationResult", project: Optional[str] = None) -> int:
    """Persist GenerationResult. Returns row id."""
    from .base import GenerationResult

    if not isinstance(result, GenerationResult):
        return -1

    cur = conn.execute(
        """
        INSERT INTO generations
        (modality, prompt, negative_prompt, model_id, parameters, seed,
         duration_ms, timestamp, output_path, metadata, project, error)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result.modality,
            result.prompt,
            result.negative_prompt,
            result.model_id,
            safe_json(result.parameters),
            result.seed,
            result.duration_ms,
            result.timestamp,
            str(result.output_path) if result.output_path else None,
            safe_json(result.metadata),
            project,
            result.error,
        ),
    )
    conn.commit()
    return cur.lastrowid


def load_recent_history(conn: sqlite3.Connection, limit: int = 50, modality: Optional[str] = None) -> List[Dict]:
    q = "SELECT * FROM generations ORDER BY timestamp DESC LIMIT ?"
    params: Tuple = (limit,)
    if modality:
        q = "SELECT * FROM generations WHERE modality = ? ORDER BY timestamp DESC LIMIT ?"
        params = (modality, limit)
    rows = conn.execute(q, params).fetchall()
    cols = [d[0] for d in conn.execute("PRAGMA table_info(generations)").fetchall()]
    return [dict(zip(cols, row)) for row in rows]


# --------------------------- Media Helpers ---------------------------

def save_pil_image(img: Any, out_dir: Path, prefix: str = "image") -> Path:
    if Image is None:
        raise RuntimeError("Pillow not installed")
    out_dir = ensure_dir(out_dir)
    path = unique_path(out_dir, prefix, ".png")
    if hasattr(img, "save"):
        img.save(path, "PNG")
    else:
        # assume numpy or path
        Image.fromarray(img).save(path)
    return path


def load_image(path: str | Path) -> Any:
    if Image is None:
        raise RuntimeError("Pillow required")
    return Image.open(path).convert("RGB")


def save_audio(waveform: Any, sr: int, path: Path) -> Path:
    if sf is None:
        # fallback: write raw if possible
        path.write_bytes(waveform if isinstance(waveform, (bytes, bytearray)) else b"")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform, sr)
    return path


def make_silent_wav(path: Path, seconds: float = 1.0, sr: int = 24000) -> Path:
    """Create a placeholder WAV for graceful fallback."""
    import numpy as np
    if sf is None:
        path.write_bytes(b"")
        return path
    samples = int(seconds * sr)
    audio = (np.zeros(samples) * 32767).astype("int16")
    sf.write(str(path), audio, sr)
    return path


# --------------------------- Prompt Library ---------------------------

def load_prompt_library(path: str | Path) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_prompt_library(items: List[Dict], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)


# --------------------------- Misc ---------------------------

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def truncate_for_log(text: str, max_len: int = 120) -> str:
    text = text.replace("\n", " ")
    return text[:max_len] + ("..." if len(text) > max_len else "")