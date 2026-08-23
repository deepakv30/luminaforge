from .dashboard import build_dashboard
from .text import build_text_studio
from .image import build_image_studio
from .video import build_video_studio
from .voice import build_voice_lab
from .workflows import build_workflows
from .library import build_library
from .history import build_history
from .models import build_model_manager
from .settings import build_settings

__all__ = [
    "build_dashboard",
    "build_text_studio",
    "build_image_studio",
    "build_video_studio",
    "build_voice_lab",
    "build_workflows",
    "build_library",
    "build_history",
    "build_model_manager",
    "build_settings",
]
