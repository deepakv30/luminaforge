from __future__ import annotations

from types import SimpleNamespace

import gradio as gr

from ui.components import generate_button, preset_chips, prompt_box, send_to_buttons
from ui.services import (
    CONFIG,
    enhance_and_update,
    format_metadata,
    get_text_model_choices,
    text_model_label,
    lazy_init,
    log_op,
    save_result_to_history,
)
from generators import GenerationResult


TEXT_PRESETS = [
    (
        "Cinematic Portrait",
        "Cinematic portrait of a confident person, dramatic side lighting, moody teal and amber color grading, shot on 85mm lens, film grain",
    ),
    (
        "Product Hero",
        "Professional product photography of a sleek modern device on reflective dark surface, dramatic cinematic lighting, luxury aesthetic",
    ),
    (
        "Social Reel Hook",
        "Eye-catching vertical 9:16 thumbnail, surprised happy person pointing at floating glowing icons, bright energetic background",
    ),
]


def build_text_studio(context_state: gr.State) -> SimpleNamespace:
    gr.Markdown("### Text Generation + Intelligent Prompt Enhancement")

    with gr.Row(elem_classes=["studio-shell"]):
        with gr.Column(scale=5, min_width=320, elem_classes=["studio-controls"]):
            with gr.Row(elem_classes=["wrap-row"]):
                text_model = gr.Dropdown(
                    choices=get_text_model_choices(),
                    value=CONFIG.get("text", {}).get("default_model"),
                    label=text_model_label(),
                    interactive=True,
                    min_width=180,
                    allow_custom_value=True,
                )
                refresh_models = gr.Button("🔄 Refresh Models", size="sm", min_width=140)

            gr.Markdown(
                "ℹ️ Switch Local Ollama / Ollama Cloud / xAI (Grok) in **Settings** — "
                "the model list and header badge update live."
            )

            system_prompt = gr.Textbox(
                label="System Prompt (optional)",
                value="You are a helpful, concise, creative assistant.",
                lines=2,
            )
            prompt = prompt_box("Your Prompt / Idea", lines=4)
            preset_chips(TEXT_PRESETS, prompt)

            with gr.Row(elem_classes=["wrap-row"]):
                enhance_btn = gr.Button("🚀 Enhance Prompt (AI)", variant="secondary", min_width=160)
                enhance_style = gr.Dropdown(
                    ["detailed cinematic", "professional marketing", "vivid storytelling", "concise technical"],
                    value="detailed cinematic",
                    label="Enhancement Style",
                    min_width=180,
                )

            with gr.Accordion("Advanced Parameters", open=False):
                temp = gr.Slider(0, 1.5, 0.7, 0.05, label="Temperature", info="Creativity vs determinism")
                top_p = gr.Slider(0.1, 1, 0.9, 0.05, label="Top P", info="Nucleus sampling")
                max_tok = gr.Slider(64, 4096, 1024, 32, label="Max Tokens", info="Response length")

            generate_btn = generate_button("Generate")

        with gr.Column(scale=7, min_width=320, elem_classes=["studio-canvas"]):
            output_text = gr.Textbox(label="Generated Text", lines=12)
            with gr.Accordion("Metadata", open=False):
                meta_text = gr.JSON(label="Metadata")
            to_img, to_vid, to_voice = send_to_buttons()

    def do_generate(p, sys_p, model, t, tp, mt):
        with log_op("generate text", model=model):
            tgen, _, _, _ = lazy_init()
            res = tgen.generate(
                prompt=p,
                system_prompt=sys_p or None,
                temperature=t,
                top_p=tp,
                max_tokens=int(mt),
                model=model,
                stream=False,
            )
            if isinstance(res, GenerationResult):
                save_result_to_history(res)
                return res.output, format_metadata(res)
            return str(res), {}

    generate_btn.click(
        do_generate,
        inputs=[prompt, system_prompt, text_model, temp, top_p, max_tok],
        outputs=[output_text, meta_text],
    )

    enhance_btn.click(
        lambda p, style: enhance_and_update(p, style, "image"),
        inputs=[prompt, enhance_style],
        outputs=prompt,
    )

    def send_text_to_image(txt, current_ctx):
        enhanced = enhance_and_update(txt, "detailed cinematic", "image")
        current_ctx = current_ctx or {}
        current_ctx.update({"image_prompt": enhanced, "last_text": txt})
        return enhanced, current_ctx

    to_img.click(
        send_text_to_image,
        inputs=[output_text, context_state],
        outputs=[prompt, context_state],
    )

    def send_to_video(txt, ctx):
        ctx = ctx or {}
        ctx["video_prompt"] = enhance_and_update(txt, "cinematic", "video")
        return ctx

    to_vid.click(send_to_video, inputs=[output_text, context_state], outputs=context_state)

    def send_to_voice(txt, ctx):
        ctx = ctx or {}
        ctx["voice_text"] = txt
        return ctx

    to_voice.click(
        send_to_voice,
        inputs=[output_text, context_state],
        outputs=context_state,
        queue=False,
    )

    refresh_models.click(
        lambda: gr.update(choices=get_text_model_choices()),
        outputs=text_model,
        queue=False,
    )

    return SimpleNamespace(text_model=text_model, prompt=prompt, output_text=output_text)
