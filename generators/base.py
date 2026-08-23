"""
LuminaForge Generators - Base Abstractions
==========================================

Design decisions:
- Use Protocol (structural typing) for maximum flexibility.
- Abstract base classes also provided for inheritance when needed.
- Every generator returns a rich `GenerationResult` dataclass containing:
    - primary output (text / PIL.Image / path to video/audio)
    - metadata dict (everything needed for history, reproducibility, "Send to...")
    - duration, seed, model_id, etc.
- This enables perfect "end-to-end chaining" without fragile string passing.
- All generators are expected to be stateless or hold minimal state.
  The UI + workflows layer orchestrates state.

Adding a new backend (e.g. a new local LLM server) only requires implementing
the protocol and registering it. No changes to app.py or workflows.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, Any, Optional, Dict, List, Union
from pathlib import Path
import time


@dataclass
class GenerationResult:
    """Unified return type across all modalities. This is the contract."""
    modality: str                    # "text" | "image" | "video" | "voice"
    output: Any                      # str (text), PIL.Image, Path (media), or bytes
    output_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    prompt: str = ""
    negative_prompt: str = ""
    model_id: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Safe serialization for history DB and export."""
        d = {
            "modality": self.modality,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "model_id": self.model_id,
            "parameters": self.parameters,
            "seed": self.seed,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "output_path": str(self.output_path) if self.output_path else None,
            "error": self.error,
        }
        # Do not serialize large binary content here
        if self.modality == "text":
            d["text"] = self.output if isinstance(self.output, str) else str(self.output)
        return d


class TextGenerator(Protocol):
    """Protocol for any text/LLM backend."""

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        model: Optional[str] = None,
        seed: Optional[int] = None,
        stream: bool = False,
    ) -> Union[GenerationResult, Any]:
        """Return GenerationResult or async/stream iterator."""
        ...

    def enhance_prompt(
        self,
        prompt: str,
        style: str = "detailed cinematic",
        target_modality: str = "image",
    ) -> str:
        """Return an improved prompt suitable for target_modality."""
        ...

    def list_models(self) -> List[str]:
        ...

    def validate(self) -> bool:
        """Quick health check (Ollama reachable, model present, etc.)."""
        ...


class ImageGenerator(Protocol):
    """Protocol for text-to-image / img2img backends."""

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int = 20,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        model: Optional[str] = None,
        init_image: Optional[Any] = None,   # PIL or path for img2img
        strength: float = 0.75,
        batch_size: int = 1,
    ) -> Union[GenerationResult, List[GenerationResult]]:
        ...

    def list_models(self) -> List[str]:
        ...

    def get_available_presets(self) -> List[tuple]:
        ...

    def validate(self) -> bool:
        ...


class VideoGenerator(Protocol):
    """Protocol for text-to-video and image-to-video."""

    def generate(
        self,
        prompt: str,
        init_image: Optional[Any] = None,
        duration: float = 5.0,
        fps: int = 16,
        width: int = 832,
        height: int = 480,
        motion_strength: float = 1.0,
        seed: Optional[int] = None,
        workflow: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> GenerationResult:
        ...

    def list_workflows(self) -> List[str]:
        ...

    def postprocess(self, video_path: Path, operations: List[str]) -> Path:
        """Apply MoviePy / FFmpeg operations (caption, upscale, speed, etc.)."""
        ...

    def validate(self) -> bool:
        ...


class VoiceGenerator(Protocol):
    """Protocol for TTS + voice cloning."""

    def generate(
        self,
        text: str,
        voice_profile: Optional[str] = None,   # name of saved profile or builtin
        speed: float = 1.0,
        emotion: str = "neutral",
        language: str = "en",
        seed: Optional[int] = None,
        **kwargs,
    ) -> GenerationResult:
        ...

    def save_voice_profile(
        self,
        name: str,
        reference_audio_path: Path,
        metadata: Optional[Dict] = None,
    ) -> Path:
        """Save reference audio + metadata. Returns profile directory."""
        ...

    def list_voice_profiles(self) -> List[Dict[str, Any]]:
        ...

    def delete_voice_profile(self, name: str) -> bool:
        ...

    def validate(self) -> bool:
        ...


# Optional: concrete ABCs for people who prefer inheritance
class BaseTextGenerator(ABC):
    @abstractmethod
    def generate(self, *args, **kwargs) -> GenerationResult:
        pass

    @abstractmethod
    def enhance_prompt(self, prompt: str, **kwargs) -> str:
        pass


class BaseImageGenerator(ABC):
    @abstractmethod
    def generate(self, *args, **kwargs) -> GenerationResult:
        pass


# Utility type for the registry pattern
GeneratorRegistry = Dict[str, Any]  # modality -> instance

# Helper for rich metadata that every generator should populate
def make_metadata(
    extra: Optional[Dict] = None,
    hardware_info: Optional[Dict] = None,
) -> Dict[str, Any]:
    meta = {
        "hardware": hardware_info or {},
        "luminaforge_version": "0.1.0",
    }
    if extra:
        meta.update(extra)
    return meta