from __future__ import annotations

import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import gradio as gr

from ui.components import generate_button, prompt_box
from ui.services import format_metadata, get_voice_profiles, lazy_init, log_line, log_op, save_result_to_history


def _audio_to_path(audio: Any) -> Optional[str]:
    """Normalize Gradio Audio values (path, FileData, dict) to a filesystem path."""
    if audio is None or audio == "":
        return None
    if isinstance(audio, (list, tuple)):
        if not audio:
            return None
        first = audio[0]
        if isinstance(first, (str, Path)):
            return str(first)
        return _audio_to_path(first)
    if isinstance(audio, dict):
        path = audio.get("path") or audio.get("name") or audio.get("orig_name")
        return str(path) if path else None
    path_attr = getattr(audio, "path", None)
    if path_attr:
        return str(path_attr)
    if isinstance(audio, (str, Path)):
        return str(audio)
    return None


def _safe_profile_name(name: Any, audio_path: Optional[str]) -> str:
    raw = (name or "").strip() if isinstance(name, str) else ""
    if not raw and audio_path:
        raw = Path(audio_path).stem
    raw = re.sub(r"[^\w\-]+", "_", raw).strip("_")
    return raw[:60] or f"voice_{int(time.time())}"


def build_voice_lab(context_state: gr.State) -> SimpleNamespace:
    gr.Markdown("## Voice Lab")
    with gr.Tabs(elem_classes=["sub-tabs"]):
        with gr.Tab("Voice Cloning"):
            gr.Markdown(
                "### Create a reusable voice profile\n\n"
                "Upload **5–15 seconds** of clean speech and save a name. "
                "TTS / Voiceover then **clones that speaker** with XTTS (CPU, first run downloads a ~2GB model). "
                "`builtin_male` / `builtin_female` stay on fast Piper and are not clones."
            )
            with gr.Row(elem_classes=["studio-shell"]):
                with gr.Column(scale=5, min_width=320, elem_classes=["studio-controls"]):
                    ref_audio = gr.Audio(
                        label="Reference Audio (clear speech, 5-15s)",
                        type="filepath",
                        sources=["upload", "microphone"],
                        format="wav",
                    )
                    profile_name = gr.Textbox(
                        label="Profile Name (required)",
                        placeholder="Type a name, e.g. my_deep_narrator",
                    )
                    profile_meta = gr.Textbox(label="Notes (optional)", lines=2)
                    save_profile_btn = generate_button("Save Voice Profile")
                with gr.Column(scale=7, min_width=320, elem_classes=["studio-canvas"]):
                    profile_status = gr.Markdown(
                        "Upload or record audio, type a **profile name**, then save."
                    )

            def save_profile(audio, name, meta):
                audio_path = _audio_to_path(audio)
                typed_name = (name or "").strip() if isinstance(name, str) else ""
                if not audio_path:
                    log_line("save voice profile skipped: no audio payload")
                    return "Please upload or record reference audio first."
                if not Path(audio_path).exists():
                    log_line(f"save voice profile skipped: missing file {audio_path}")
                    return "The audio file could not be found. Re-upload or record it, then save again."
                profile_id = _safe_profile_name(typed_name, audio_path)
                if not typed_name:
                    log_line(f"save voice profile using filename as name={profile_id}")
                try:
                    with log_op("save voice profile", name=profile_id):
                        _, _, _, vgen = lazy_init()
                        vgen.save_voice_profile(profile_id, Path(audio_path), {"notes": meta or ""})
                    hint = "" if typed_name else f" (named from the audio file: `{profile_id}`)"
                    return f"✅ Saved profile: **{profile_id}**{hint}. Use Refresh Profiles in TTS / Voiceover to select it."
                except Exception as e:
                    log_line(f"save voice profile failed error={e}")
                    return f"Error saving profile: {e}"

            save_profile_btn.click(
                save_profile, inputs=[ref_audio, profile_name, profile_meta], outputs=profile_status
            )

        with gr.Tab("TTS / Voiceover"):
            gr.Markdown(
                "### Generate voice from text\n\n"
                "Pick a **saved profile** to clone that voice. First clone can take several minutes on CPU. "
                "Use builtin voices for instant Piper TTS."
            )
            with gr.Row(elem_classes=["studio-shell"]):
                with gr.Column(scale=5, min_width=320, elem_classes=["studio-controls"]):
                    voice_text = prompt_box("Text to speak", lines=4)
                    with gr.Row(elem_classes=["wrap-row"]):
                        voice_profile = gr.Dropdown(
                            choices=get_voice_profiles(),
                            value="builtin_male",
                            label="Voice Profile",
                            interactive=True,
                            min_width=160,
                        )
                        refresh_profiles = gr.Button("Refresh Profiles", min_width=140)
                    with gr.Accordion("Advanced Parameters", open=False):
                        with gr.Row(elem_classes=["wrap-row"]):
                            speed = gr.Slider(0.6, 1.6, 1.0, 0.05, label="Speed")
                            emotion = gr.Dropdown(
                                ["neutral", "happy", "sad", "excited", "serious"],
                                value="neutral",
                                label="Emotion / Style",
                                min_width=140,
                            )
                    gr.Markdown("**Tip:** Use the waveform preview after generation. Speed control affects playback feel.")
                    gen_voice_btn = generate_button("Generate Voiceover")

                with gr.Column(scale=7, min_width=320, elem_classes=["studio-canvas"]):
                    voice_status = gr.Markdown()
                    voice_audio = gr.Audio(label="Generated Audio", type="filepath")
                    with gr.Accordion("Metadata", open=False):
                        voice_meta = gr.JSON()

            def do_voice(text, profile, spd, emo):
                with log_op("generate voice", profile=profile):
                    _, _, _, vgen = lazy_init()
                    res = vgen.generate(text=text, voice_profile=profile, speed=spd, emotion=emo)
                    save_result_to_history(res)
                    path = str(res.output_path) if res.output_path else None
                    cloned = bool((res.metadata or {}).get("cloned") or (res.parameters or {}).get("cloned"))
                    if res.error:
                        status = f"⚠️ {res.error}"
                    elif cloned:
                        status = f"✅ Cloned voiceover ({res.model_id})."
                    else:
                        status = "✅ Voiceover ready (Piper stock voice)."
                    return path, format_metadata(res), status

            gen_voice_btn.click(
                do_voice,
                inputs=[voice_text, voice_profile, speed, emotion],
                outputs=[voice_audio, voice_meta, voice_status],
            )
            refresh_profiles.click(
                lambda: gr.update(choices=get_voice_profiles()),
                outputs=voice_profile,
                queue=False,
            )

    return SimpleNamespace()
