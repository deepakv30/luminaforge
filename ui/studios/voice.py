from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from ui.components import generate_button, prompt_box
from ui.services import format_metadata, get_voice_profiles, lazy_init, log_op, save_result_to_history


def build_voice_lab(context_state: gr.State) -> SimpleNamespace:
    gr.Markdown("## Voice Lab")
    with gr.Tabs(elem_classes=["sub-tabs"]):
        with gr.Tab("Voice Cloning"):
            gr.Markdown(
                "### Create a reusable voice profile\n\n"
                "**Recommended backend**: `piper-tts` (install with `pip install piper-tts`). "
                "Coqui XTTS often fails on Python 3.12+."
            )
            with gr.Row(elem_classes=["studio-shell"]):
                with gr.Column(scale=5, min_width=320, elem_classes=["studio-controls"]):
                    ref_audio = gr.Audio(label="Reference Audio (clear speech, 5-15s)", type="filepath")
                    profile_name = gr.Textbox(label="Profile Name", placeholder="e.g. my_deep_narrator")
                    profile_meta = gr.Textbox(label="Notes (optional)", lines=2)
                    save_profile_btn = generate_button("Save Voice Profile")
                with gr.Column(scale=7, min_width=320, elem_classes=["studio-canvas"]):
                    profile_status = gr.Markdown()

            def save_profile(audio_path, name, meta):
                if not audio_path or not name:
                    return "Please provide audio and a name."
                try:
                    with log_op("save voice profile", name=name):
                        _, _, _, vgen = lazy_init()
                        vgen.save_voice_profile(name, Path(audio_path), {"notes": meta})
                    return f"✅ Saved profile: {name}"
                except Exception as e:
                    return f"Error saving profile: {e}"

            save_profile_btn.click(
                save_profile, inputs=[ref_audio, profile_name, profile_meta], outputs=profile_status
            )

        with gr.Tab("TTS / Voiceover"):
            gr.Markdown("### Generate voice from text using saved profiles")
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
                    voice_audio = gr.Audio(label="Generated Audio", type="filepath")
                    with gr.Accordion("Metadata", open=False):
                        voice_meta = gr.JSON()

            def do_voice(text, profile, spd, emo):
                with log_op("generate voice", profile=profile):
                    _, _, _, vgen = lazy_init()
                    res = vgen.generate(text=text, voice_profile=profile, speed=spd, emotion=emo)
                    save_result_to_history(res)
                    return str(res.output_path) if res.output_path else None, format_metadata(res)

            gen_voice_btn.click(
                do_voice,
                inputs=[voice_text, voice_profile, speed, emotion],
                outputs=[voice_audio, voice_meta],
            )
            refresh_profiles.click(
                lambda: gr.update(choices=get_voice_profiles()),
                outputs=voice_profile,
                queue=False,
            )

    return SimpleNamespace()
