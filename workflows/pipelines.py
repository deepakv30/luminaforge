"""
LuminaForge End-to-End Pipelines & Wizards
==========================================

This is where the "killer feature" lives: seamless chaining.

Every pipeline function receives a context dict and returns an updated context.
The UI can pass state between tabs using these functions + "Send to..." buttons.

Example flow:
    text -> enhance -> image -> video -> voiceover script -> final voice

These are intentionally simple and extremely transparent so a developer
can read them in <5 minutes and extend them.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from pathlib import Path

from generators import (
    get_text_generator,
    get_image_generator,
    get_video_generator,
    get_voice_generator,
    GenerationResult,
)


def run_social_reel_pipeline(
    idea: str,
    config: Optional[Dict] = None,
    progress_cb=None,
) -> Dict[str, Any]:
    """
    "Social Reel Creator" wizard.
    1. Enhance idea into video script (text)
    2. Generate hero image
    3. Generate short video from image + script
    4. Generate voiceover from script
    Returns rich context with all artifacts + metadata.
    """
    ctx: Dict[str, Any] = {"idea": idea, "steps": []}
    text_gen = get_text_generator(config)
    image_gen = get_image_generator(config)
    video_gen = get_video_generator(config)
    voice_gen = get_voice_generator(config)

    # Step 1: Script
    if progress_cb:
        progress_cb("Enhancing prompt into reel script...")
    script_res = text_gen.generate(
        prompt=f"Turn this idea into a 25-35 second high-energy social media reel script: {idea}",
        temperature=0.75,
        max_tokens=220,
    )
    ctx["script"] = script_res.output if isinstance(script_res, GenerationResult) else str(script_res)
    ctx["steps"].append("script")

    # Step 2: Hero visual
    if progress_cb:
        progress_cb("Generating hero image...")
    img_prompt = text_gen.enhance_prompt(idea, style="cinematic social media", target_modality="image")
    img_res = image_gen.generate(img_prompt, steps=4, guidance_scale=3.5)
    ctx["hero_image"] = img_res.output_path if isinstance(img_res, GenerationResult) else None
    ctx["steps"].append("image")

    # Step 3: Short video (image-to-video preferred)
    if progress_cb:
        progress_cb("Creating short video clip...")
    vid_res = video_gen.generate(
        prompt=ctx["script"][:280],
        init_image=ctx["hero_image"],
        duration=5.0,
        fps=16,
    )
    ctx["video"] = vid_res.output_path if isinstance(vid_res, GenerationResult) else None
    ctx["steps"].append("video")

    # Step 4: Voiceover
    if progress_cb:
        progress_cb("Generating voiceover...")
    voice_res = voice_gen.generate(
        text=ctx["script"],
        voice_profile=None,  # user can swap later in Voice Lab
        speed=1.05,
    )
    ctx["voiceover"] = voice_res.output_path if isinstance(voice_res, GenerationResult) else None
    ctx["steps"].append("voice")

    ctx["final_package"] = {
        "script": ctx["script"],
        "hero_image": str(ctx.get("hero_image", "")),
        "video": str(ctx.get("video", "")),
        "voiceover": str(ctx.get("voiceover", "")),
    }
    return ctx


def run_educational_explainer(
    topic: str,
    config: Optional[Dict] = None,
    progress_cb=None,
) -> Dict[str, Any]:
    """Educational Explainer: clean narration + supporting visuals."""
    ctx: Dict[str, Any] = {"topic": topic}
    text_gen = get_text_generator(config)
    image_gen = get_image_generator(config)

    if progress_cb:
        progress_cb("Writing educational script...")
    script = text_gen.generate(
        prompt=f"Write a clear 45-second educational explainer script about: {topic}. Use simple language.",
        max_tokens=300,
    )
    ctx["script"] = script.output if isinstance(script, GenerationResult) else str(script)

    if progress_cb:
        progress_cb("Creating supporting visuals...")
    visual_prompt = text_gen.enhance_prompt(ctx["script"][:200], target_modality="image")
    img = image_gen.generate(visual_prompt, steps=6)
    ctx["visual"] = img.output_path if isinstance(img, GenerationResult) else None

    return ctx


# Simple hand-off helpers used by the UI "Send to..." buttons
def prepare_for_image(text: str, text_gen) -> str:
    return text_gen.enhance_prompt(text, style="vivid", target_modality="image")


def prepare_for_video(text: str, text_gen) -> str:
    return text_gen.enhance_prompt(text, style="cinematic", target_modality="video")


def prepare_script_for_voice(text: str) -> str:
    # Clean up for TTS
    return text.replace("\n\n", ". ").strip()[:1800]
