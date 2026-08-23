"""
LuminaForge Voice Generator (XTTS + Piper + Profiles)
=====================================================

Productivity focus:
- Voice profiles are first-class citizens. Clone once, use everywhere.
- One-click selection from History or Voice Lab.
- "Send to Voice" from any other tab loads the text and lets you pick profile instantly.

Implementation notes:
- Coqui XTTS-v2 gives best zero-shot quality but is heavy to install.
- Piper is fast + lightweight.
- We provide graceful fallbacks and clear instructions when backends missing.
- Profiles live in data/voice_profiles/<name>/{ref.wav, metadata.json}
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import GenerationResult, VoiceGenerator, make_metadata
from .utils import (
    detect_hardware,
    ensure_dir,
    load_config,
    unique_path,
    make_silent_wav,
    save_audio,
)

BUILTIN_PROFILES = {"builtin_male", "builtin_female"}


def _patch_transformers_for_xtts() -> None:
    """coqui-tts 0.27 still imports a helpers name removed in transformers 5."""
    try:
        import torch
        import transformers.pytorch_utils as pu

        if not hasattr(pu, "isin_mps_friendly"):
            def isin_mps_friendly(elements, test_elements, assume_unique=False, invert=False):
                return torch.isin(
                    elements, test_elements, assume_unique=assume_unique, invert=invert
                )

            pu.isin_mps_friendly = isin_mps_friendly
    except Exception:
        pass


def _patch_torchaudio_load_without_ffmpeg() -> None:
    """XTTS uses torchaudio.load -> torchcodec, which needs system FFmpeg libs we may not have."""
    try:
        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio
    except Exception:
        return

    original = getattr(torchaudio, "load", None)
    if original is None:
        return

    def load(uri, *args, **kwargs):
        try:
            return original(uri, *args, **kwargs)
        except Exception:
            data, sr = sf.read(str(uri), always_2d=False)
            if np.issubdtype(data.dtype, np.integer):
                maxv = float(np.iinfo(data.dtype).max) or 32768.0
                data = data.astype(np.float32) / maxv
            else:
                data = data.astype(np.float32)
            if data.ndim == 1:
                wav = torch.from_numpy(data).unsqueeze(0)
            else:
                wav = torch.from_numpy(np.transpose(data))
            return wav, int(sr)

    torchaudio.load = load  # type: ignore[method-assign]


_patch_transformers_for_xtts()
os.environ.setdefault("COQUI_TOS_AGREED", "1")

try:
    from TTS.api import TTS as CoquiTTS
except ImportError:
    CoquiTTS = None

_patch_torchaudio_load_without_ffmpeg()

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    import soundfile as sf
    import numpy as np
except ImportError:
    sf = None
    np = None

try:
    import wave
except ImportError:
    wave = None

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*Siglip2ImageProcessorFast.*")


PROFILE_DIR = Path("data/voice_profiles")


class VoiceProfileManager:
    """Manage saved voice profiles (reference audio + metadata)."""

    def __init__(self, base_dir: Path = PROFILE_DIR):
        self.base_dir = ensure_dir(base_dir)

    def list_profiles(self) -> List[Dict[str, Any]]:
        profiles = []
        for p in self.base_dir.iterdir():
            if p.is_dir():
                meta_path = p / "metadata.json"
                ref = p / "reference.wav"
                meta = {}
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                profiles.append({
                    "name": p.name,
                    "reference_path": str(ref) if ref.exists() else None,
                    "metadata": meta,
                })
        return profiles

    def save_profile(self, name: str, reference_audio: Path, metadata: Optional[Dict] = None) -> Path:
        prof_dir = ensure_dir(self.base_dir / name)
        target_ref = prof_dir / "reference.wav"
        shutil.copy2(reference_audio, target_ref)

        meta = metadata or {}
        meta.update({"created": time.time(), "name": name})
        (prof_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        return prof_dir

    def delete_profile(self, name: str) -> bool:
        prof = self.base_dir / name
        if prof.exists():
            shutil.rmtree(prof)
            return True
        return False

    def get_profile(self, name: str) -> Optional[Dict]:
        for p in self.list_profiles():
            if p["name"] == name:
                return p
        return None


class CoquiVoiceGenerator:
    """High quality zero-shot voice cloning using Coqui XTTS-v2."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("voice", {})
        self.profile_manager = VoiceProfileManager()
        self.output_dir = Path(config.get("app", {}).get("output_dir", "./outputs")) / "audio"
        ensure_dir(self.output_dir)
        self.hardware = detect_hardware()
        self.tts = None
        self._loaded_model = None

    def _load_tts(self):
        if CoquiTTS is None:
            raise RuntimeError(
                "Voice cloning needs coqui-tts. In the project venv run: "
                "python -m pip install 'coqui-tts[codec]>=0.27'"
            )
        model_name = self.config.get("coqui", {}).get(
            "model", "tts_models/multilingual/multi-dataset/xtts_v2"
        )
        if self.tts is None or self._loaded_model != model_name:
            print(f"[Voice] Loading Coqui XTTS on CPU (first run downloads ~2GB): {model_name}", flush=True)
            self.tts = CoquiTTS(model_name=model_name, gpu=False, progress_bar=True)
            self._loaded_model = model_name
            print("[Voice] XTTS ready", flush=True)
        return self.tts

    def generate(
        self,
        text: str,
        voice_profile: Optional[str] = None,
        speed: float = 1.0,
        emotion: str = "neutral",
        language: str = "en",
        seed: Optional[int] = None,
        speaker_wav: Optional[str] = None,
        **kwargs,
    ) -> GenerationResult:
        start = time.time()
        profile = self.profile_manager.get_profile(voice_profile) if voice_profile else None
        ref = speaker_wav or (profile.get("reference_path") if profile else None)
        language = language or self.config.get("coqui", {}).get("language", "en")

        if not text or not str(text).strip():
            return self._fallback_generate(text or "", voice_profile, "Empty text")
        if not ref or not Path(ref).exists():
            return self._fallback_generate(text, voice_profile, "No reference audio on this profile")

        try:
            tts = self._load_tts()
            out_path = unique_path(self.output_dir, "clone", ".wav")
            kwargs_tts: Dict[str, Any] = {
                "text": str(text).strip(),
                "file_path": str(out_path),
                "speaker_wav": str(ref),
                "language": language,
            }
            # speed is supported by some XTTS builds
            try:
                tts.tts_to_file(**kwargs_tts, speed=float(speed))
            except TypeError:
                tts.tts_to_file(**kwargs_tts)

            duration_ms = (time.time() - start) * 1000
            return GenerationResult(
                modality="voice",
                output=str(out_path),
                output_path=out_path,
                prompt=text,
                model_id="coqui_xtts_v2",
                parameters={
                    "voice_profile": voice_profile,
                    "speed": speed,
                    "emotion": emotion,
                    "language": language,
                    "cloned": True,
                },
                duration_ms=duration_ms,
                metadata=make_metadata(
                    {"profile": profile, "reference": str(ref), "cloned": True},
                    self.hardware,
                ),
            )
        except Exception as e:
            return self._fallback_generate(text, voice_profile, str(e))

    def _fallback_generate(self, text: str, profile: Optional[str], reason: str) -> GenerationResult:
        out = unique_path(self.output_dir, "fallback_voice", ".wav")
        make_silent_wav(out, seconds=max(1.5, len(text) / 18))
        return GenerationResult(
            modality="voice",
            output=str(out),
            output_path=out,
            prompt=text,
            model_id="fallback",
            error=f"Using silent placeholder. Reason: {reason}",
            metadata=make_metadata({"reason": reason, "profile": profile}, self.hardware),
        )

    def save_voice_profile(self, name: str, reference_audio_path: Path, metadata: Optional[Dict] = None) -> Path:
        return self.profile_manager.save_profile(name, reference_audio_path, metadata)

    def list_voice_profiles(self) -> List[Dict[str, Any]]:
        return self.profile_manager.list_profiles()

    def delete_voice_profile(self, name: str) -> bool:
        return self.profile_manager.delete_profile(name)

    def validate(self) -> bool:
        return CoquiTTS is not None


class HybridVoiceGenerator:
    """Piper for stock voices; XTTS zero-shot cloning when a saved profile has reference audio."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.piper = PiperVoiceGenerator(config)
        self.cloner = CoquiVoiceGenerator(config)
        self.profile_manager = self.piper.profile_manager
        self.hardware = detect_hardware()

    def _reference_wav(self, voice_profile: Optional[str]) -> Optional[str]:
        if not voice_profile or voice_profile in BUILTIN_PROFILES:
            return None
        profile = self.profile_manager.get_profile(voice_profile)
        if not profile:
            return None
        ref = profile.get("reference_path")
        if ref and Path(ref).exists() and Path(ref).stat().st_size > 1000:
            return str(ref)
        return None

    def generate(self, text: str, voice_profile: Optional[str] = None, **kwargs) -> GenerationResult:
        ref = self._reference_wav(voice_profile)
        if ref:
            if CoquiTTS is None:
                result = self.piper.generate(text=text, voice_profile=voice_profile, **kwargs)
                result.error = (
                    "Profile saved, but cloning needs coqui-tts. "
                    "Run: .venv/bin/python -m pip install 'coqui-tts[codec]>=0.27' "
                    "then restart. Used Piper stock voice instead."
                )
                return result
            print(f"[Voice] Cloning from profile '{voice_profile}' ({ref})", flush=True)
            result = self.cloner.generate(
                text=text, voice_profile=voice_profile, speaker_wav=ref, **kwargs
            )
            if result.error:
                piper = self.piper.generate(text=text, voice_profile=voice_profile, **kwargs)
                piper.error = (
                    f"Clone failed ({result.error}). Played Piper stock voice instead."
                )
                return piper
            return result
        return self.piper.generate(text=text, voice_profile=voice_profile, **kwargs)

    def save_voice_profile(self, name: str, reference_audio_path: Path, metadata: Optional[Dict] = None) -> Path:
        return self.profile_manager.save_profile(name, reference_audio_path, metadata)

    def list_voice_profiles(self) -> List[Dict[str, Any]]:
        return self.profile_manager.list_profiles()

    def delete_voice_profile(self, name: str) -> bool:
        return self.profile_manager.delete_profile(name)

    def validate(self) -> bool:
        return True


class PiperVoiceGenerator:
    """Fast, high-quality local TTS using Piper (recommended for Python 3.12+).

    Models are automatically downloaded on first use from Hugging Face.
    Default voice: en_US-lessac-medium (good quality, small).
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("voice", {})
        self.profile_manager = VoiceProfileManager()
        self.output_dir = Path(config.get("app", {}).get("output_dir", "./outputs")) / "audio"
        ensure_dir(self.output_dir)
        self.hardware = detect_hardware()
        self.models_dir = Path(self.config.get("piper", {}).get("models_dir", "./data/piper_models"))
        ensure_dir(self.models_dir)
        self._voice_cache = {}

    def _get_piper_voice(self, model_name: str):
        """Load or download a Piper voice model."""
        try:
            from piper import PiperVoice
        except ImportError:
            raise RuntimeError(
                "piper-tts not installed. Run: pip install piper-tts\n"
                "Then restart the app."
            )

        model_path = self.models_dir / f"{model_name}.onnx"
        config_path = self.models_dir / f"{model_name}.onnx.json"

        if model_name in self._voice_cache:
            return self._voice_cache[model_name]

        if not model_path.exists():
            print(f"[Voice] Downloading Piper voice model: {model_name} ...")
            self._download_piper_model(model_name, model_path, config_path)

        voice = PiperVoice.load(str(model_path), config_path=str(config_path) if config_path.exists() else None)
        self._voice_cache[model_name] = voice
        return voice

    def _download_piper_model(self, model_name: str, model_path: Path, config_path: Path):
        """Download a Piper voice from Hugging Face (reliable direct links)."""
        try:
            import urllib.request
        except Exception:
            raise RuntimeError("urllib not available for model download")

        # Reliable direct links for popular Piper voices
        voice_urls = {
            "en_US-lessac-medium": {
                "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx?download=true",
                "json": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json?download=true",
            },
            "en_US-amy-medium": {
                "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx?download=true",
                "json": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx.json?download=true",
            },
        }

        urls = voice_urls.get(model_name, voice_urls["en_US-lessac-medium"])

        try:
            print(f"[Voice] Downloading {model_name} (first time use)...")
            urllib.request.urlretrieve(urls["onnx"], model_path)
            try:
                urllib.request.urlretrieve(urls["json"], config_path)
            except Exception:
                pass
            print(f"[Voice] Model ready at {model_path}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to auto-download Piper model '{model_name}'.\n"
                f"Please manually download from https://huggingface.co/rhasspy/piper-voices and place in {self.models_dir}\n"
                f"Error: {e}"
            )

    def generate(self, text: str, voice_profile: Optional[str] = None, **kwargs) -> GenerationResult:
        start = time.time()
        model_name = self.config.get("piper", {}).get("default_model", "en_US-lessac-medium")

        try:
            voice = self._get_piper_voice(model_name)

            out_path = unique_path(self.output_dir, "piper", ".wav")
            if wave is not None:
                with wave.open(str(out_path), "wb") as wav_file:
                    # Use first chunk to set wave parameters
                    chunks = list(voice.synthesize(text))
                    if chunks:
                        first = chunks[0]
                        wav_file.setnchannels(getattr(first, "sample_channels", 1))
                        wav_file.setsampwidth(getattr(first, "sample_width", 2))
                        wav_file.setframerate(getattr(first, "sample_rate", voice.config.sample_rate))

                        for chunk in chunks:
                            if hasattr(chunk, "audio_int16_bytes"):
                                wav_file.writeframes(chunk.audio_int16_bytes)
                            elif hasattr(chunk, "audio_int16_array"):
                                wav_file.writeframes(chunk.audio_int16_array.tobytes())
                            else:
                                import numpy as _np
                                audio = (_np.array(chunk.audio_float_array) * 32767).astype("int16").tobytes()
                                wav_file.writeframes(audio)
            else:
                make_silent_wav(out_path, seconds=max(1.5, len(text) / 15))

            duration_ms = (time.time() - start) * 1000

            return GenerationResult(
                modality="voice",
                output=str(out_path),
                output_path=out_path,
                prompt=text,
                model_id=f"piper-{model_name}",
                parameters={"voice_profile": voice_profile, "model": model_name},
                duration_ms=duration_ms,
                metadata=make_metadata({"piper_model": model_name}, self.hardware),
            )
        except Exception as e:
            # Final fallback
            out = unique_path(self.output_dir, "fallback", ".wav")
            make_silent_wav(out, seconds=max(1.5, len(text) / 15))
            return GenerationResult(
                modality="voice",
                output=str(out),
                output_path=out,
                prompt=text,
                model_id="piper-fallback",
                error=f"Piper generation failed: {e}. Using placeholder audio.",
                metadata=make_metadata({}, self.hardware),
            )

    def save_voice_profile(self, name: str, reference_audio_path: Path, metadata=None):
        return self.profile_manager.save_profile(name, reference_audio_path, metadata)

    def list_voice_profiles(self):
        return self.profile_manager.list_profiles()

    def delete_voice_profile(self, name: str):
        return self.profile_manager.delete_profile(name)

    def validate(self):
        return False


class FallbackVoiceGenerator:
    """Absolute last resort: pyttsx3 or silent files."""

    def __init__(self, config: Dict[str, Any]):
        self.profile_manager = VoiceProfileManager()
        self.output_dir = Path(config.get("app", {}).get("output_dir", "./outputs")) / "audio"
        ensure_dir(self.output_dir)
        self.engine = None

    def _get_engine(self):
        if pyttsx3 and self.engine is None:
            self.engine = pyttsx3.init()
        return self.engine

    def generate(self, text: str, **kwargs) -> GenerationResult:
        out = unique_path(self.output_dir, "pyttsx", ".wav")
        eng = self._get_engine()
        if eng:
            # pyttsx3 doesn't write wav easily cross-platform; use placeholder
            pass
        make_silent_wav(out, seconds=2)
        return GenerationResult(
            modality="voice",
            output=str(out),
            output_path=out,
            prompt=text,
            model_id="system_fallback",
            error="Installed voice backend not detected. Placeholder audio saved.",
            metadata=make_metadata({}, {}),
        )

    def save_voice_profile(self, name, ref, meta=None):
        return self.profile_manager.save_profile(name, ref, meta)

    def list_voice_profiles(self):
        return self.profile_manager.list_profiles()

    def delete_voice_profile(self, name):
        return self.profile_manager.delete_profile(name)

    def validate(self):
        return True


def get_voice_generator(config: Optional[Dict] = None) -> VoiceGenerator:
    cfg = config or load_config()
    provider = cfg.get("voice", {}).get("provider", "hybrid")
    if provider == "coqui_xtts":
        return CoquiVoiceGenerator(cfg)
    if provider == "piper":
        # Still wrap so saved profiles clone when coqui-tts is present.
        return HybridVoiceGenerator(cfg)
    return HybridVoiceGenerator(cfg)