"""
Reusable Gradio components for LuminaForge.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import gradio as gr


def model_selector(
    label: str = "Model",
    choices: Optional[List[str]] = None,
    value: Optional[str] = None,
    interactive: bool = True,
) -> gr.Dropdown:
    return gr.Dropdown(
        choices=choices or ["llama3.2:3b"],
        value=value,
        label=label,
        interactive=interactive,
        scale=1,
        min_width=160,
    )


def parameter_sliders() -> Dict[str, gr.Slider]:
    return {
        "temperature": gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature"),
        "top_p": gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top P"),
        "max_tokens": gr.Slider(64, 4096, value=1024, step=32, label="Max Tokens"),
    }


def send_to_buttons() -> Tuple[gr.Button, gr.Button, gr.Button]:
    with gr.Row(elem_classes=["send-to-row", "wrap-row"]):
        to_image = gr.Button(
            "Send to Image Studio →",
            variant="secondary",
            elem_classes=["send-to-btn"],
            min_width=140,
        )
        to_video = gr.Button(
            "Send to Video Studio →",
            variant="secondary",
            elem_classes=["send-to-btn"],
            min_width=140,
        )
        to_voice = gr.Button(
            "Send to Voice Lab →",
            variant="secondary",
            elem_classes=["send-to-btn"],
            min_width=140,
        )
    return to_image, to_video, to_voice


def metadata_display() -> gr.JSON:
    return gr.JSON(label="Generation Metadata", elem_classes=["metadata-box"])


def prompt_box(label: str = "Prompt", lines: int = 3) -> gr.Textbox:
    return gr.Textbox(
        label=label,
        lines=lines,
        placeholder="Describe what you want to create...",
    )


def negative_prompt_box() -> gr.Textbox:
    return gr.Textbox(
        label="Negative Prompt",
        value="blurry, low quality, deformed, ugly, watermark",
        lines=2,
    )


def seed_and_batch() -> Dict[str, gr.Component]:
    return {
        "seed": gr.Number(label="Seed (optional)", value=None, precision=0),
        "batch": gr.Slider(1, 8, value=1, step=1, label="Batch Size"),
    }


def preset_chips(presets: Sequence[Tuple[str, str]], target: gr.Textbox) -> None:
    """Wrapping chip row. Each chip writes its prompt into target."""
    with gr.Row(elem_classes=["preset-row", "wrap-row"]):
        gr.Markdown("**Presets:**", elem_classes=["preset-label"])
        for name, text in presets:
            gr.Button(name, size="sm", elem_classes=["preset-btn"], min_width=120).click(
                lambda p=text: p, inputs=None, outputs=target, queue=False
            )


def fluid_gallery(label: str, height: str = "min(50dvh, 560px)", columns: int = 2, **kwargs: Any) -> gr.Gallery:
    classes = list(kwargs.pop("elem_classes", []) or [])
    if "gallery-fluid" not in classes:
        classes.append("gallery-fluid")
    return gr.Gallery(
        label=label,
        columns=columns,
        height=height,
        object_fit="contain",
        elem_classes=classes,
        **kwargs,
    )


def generate_button(label: str, **kwargs: Any) -> gr.Button:
    classes = list(kwargs.pop("elem_classes", []) or [])
    classes.extend(["generate-sticky"])
    return gr.Button(
        label,
        variant="primary",
        size="lg",
        elem_classes=classes,
        min_width=160,
        **kwargs,
    )
