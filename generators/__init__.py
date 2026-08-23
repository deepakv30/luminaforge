"""
LuminaForge Generators Package
==============================

Public API:
    from generators import get_text_generator, get_image_generator, ...

This allows the rest of the app (especially workflows and UI) to stay
decoupled from concrete implementations.
"""

from .base import (
    GenerationResult,
    TextGenerator,
    ImageGenerator,
    VideoGenerator,
    VoiceGenerator,
)
from .text import OllamaTextGenerator, get_text_generator, resolve_text_provider
from .xai_text import XAITextGenerator
from .image import get_image_generator, DiffusersImageGenerator, ComfyUIImageGenerator
from .video import get_video_generator, ComfyUIVideoGenerator
from .voice import get_voice_generator, VoiceProfileManager
from .utils import (
    load_config,
    save_config,
    detect_hardware,
    init_history_db,
    save_generation,
    load_prompt_library,
    save_prompt_library,
)

__all__ = [
    "GenerationResult",
    "TextGenerator",
    "ImageGenerator",
    "VideoGenerator",
    "VoiceGenerator",
    "get_text_generator",
    "resolve_text_provider",
    "get_image_generator",
    "get_video_generator",
    "get_voice_generator",
    "OllamaTextGenerator",
    "XAITextGenerator",
    "VoiceProfileManager",
    "load_config",
    "save_config",
    "detect_hardware",
    "init_history_db",
    "save_generation",
    "load_prompt_library",
    "save_prompt_library",
]


# Convenience re-exports
text_generator = None
image_generator = None
video_generator = None
voice_generator = None


def initialize_generators(cfg=None):
    """Call once at app startup. Returns dict of live generators."""
    global text_generator, image_generator, video_generator, voice_generator
    config = cfg or load_config()
    text_generator = get_text_generator(config)
    image_generator = get_image_generator(config)
    video_generator = get_video_generator(config)
    voice_generator = get_voice_generator(config)
    return {
        "text": text_generator,
        "image": image_generator,
        "video": video_generator,
        "voice": voice_generator,
    }
