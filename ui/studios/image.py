from __future__ import annotations

from types import SimpleNamespace

import gradio as gr

from generators import GenerationResult
from ui.components import (
    fluid_gallery,
    generate_button,
    negative_prompt_box,
    preset_chips,
    prompt_box,
)
from ui.services import (
    CONFIG,
    enhance_and_update,
    format_metadata,
    get_image_model_choices,
    lazy_init,
    log_op,
    save_result_to_history,
)

IMAGE_PRESETS = [
    ("Cinematic Portrait", "Cinematic portrait, dramatic lighting, film grain, 85mm lens"),
    ("Product Shot", "Professional product photography on dark reflective marble, luxury lighting"),
    ("Fantasy Environment", "Mystical ancient forest at twilight, glowing mushrooms, volumetric god rays"),
]


def build_image_studio(context_state: gr.State) -> SimpleNamespace:
    gr.Markdown("### Text-to-Image & Img2Img")

    with gr.Row(elem_classes=["studio-shell"]):
        with gr.Column(scale=5, min_width=320, elem_classes=["studio-controls"]):
            img_prompt = prompt_box("Image Prompt", lines=3)
            neg_prompt = negative_prompt_box()
            preset_chips(IMAGE_PRESETS, img_prompt)

            with gr.Row(elem_classes=["wrap-row"]):
                img_model = gr.Dropdown(
                    choices=get_image_model_choices(),
                    value=CONFIG.get("image", {}).get("default_model"),
                    label="Image Model",
                    min_width=180,
                )
                img_width = gr.Dropdown(
                    [512, 768, 896, 1024, 1152, 1344],
                    value=1024,
                    label="Width",
                    info="Higher = more VRAM",
                    min_width=110,
                )
                img_height = gr.Dropdown(
                    [512, 768, 896, 1024, 1152, 1344],
                    value=1024,
                    label="Height",
                    info="Higher = more VRAM",
                    min_width=110,
                )

            with gr.Accordion("Image to Image", open=False):
                init_image = gr.Image(label="Init Image (optional for img2img)", type="filepath")
                strength = gr.Slider(0.1, 1.0, 0.65, label="Img2Img Strength")

            with gr.Accordion("Advanced Parameters", open=False):
                with gr.Row(elem_classes=["wrap-row"]):
                    steps = gr.Slider(1, 60, value=4, step=1, label="Steps")
                    guidance = gr.Slider(1, 15, value=3.5, step=0.5, label="Guidance Scale")
                    seed_img = gr.Number(label="Seed", value=None)
                batch_size = gr.Slider(1, 4, 1, 1, label="Batch")

            with gr.Row(elem_classes=["wrap-row"]):
                img_gen_btn = generate_button("Generate Image(s)")
                enhance_img_btn = gr.Button("Enhance Prompt for Image", min_width=160)

        with gr.Column(scale=7, min_width=320, elem_classes=["studio-canvas"]):
            img_gallery = fluid_gallery("Generated Images", height="min(55dvh, 560px)", columns=2)
            with gr.Accordion("Last Image Metadata", open=False):
                img_meta = gr.JSON(label="Last Image Metadata")
            to_video_from_img = gr.Button(
                "Send to Video Studio →",
                variant="secondary",
                elem_classes=["send-to-btn"],
                min_width=160,
            )

    def generate_images_with_progress(p, neg, w, h, st, gs, sd, init, stren, bs, model, progress=gr.Progress()):
        with log_op("generate image", model=model, size=f"{w}x{h}", steps=st):
            _, igen, _, _ = lazy_init()
            progress(0, desc="Starting generation...")
            res = igen.generate(
                prompt=p,
                negative_prompt=neg,
                width=int(w),
                height=int(h),
                steps=int(st),
                guidance_scale=float(gs),
                seed=int(sd) if sd else None,
                model=model,
                init_image=init,
                strength=float(stren),
                batch_size=int(bs),
            )
            progress(1.0, desc="Processing results...")
            images = []
            metas = []
            if isinstance(res, list):
                for i, r in enumerate(res):
                    progress((i + 1) / len(res), desc=f"Processing image {i + 1}")
                    if r.output_path:
                        images.append(str(r.output_path))
                    save_result_to_history(r)
                    metas.append(format_metadata(r))
            elif isinstance(res, GenerationResult):
                if res.output_path:
                    images.append(str(res.output_path))
                save_result_to_history(res)
                metas = format_metadata(res)
            return images, metas

    img_gen_btn.click(
        generate_images_with_progress,
        inputs=[
            img_prompt, neg_prompt, img_width, img_height, steps, guidance,
            seed_img, init_image, strength, batch_size, img_model,
        ],
        outputs=[img_gallery, img_meta],
    )

    enhance_img_btn.click(
        lambda p: enhance_and_update(p, "detailed cinematic", "image"),
        inputs=img_prompt,
        outputs=img_prompt,
    )

    def handoff_image_to_video(gallery, prompt):
        if gallery and len(gallery) > 0:
            last_img = gallery[-1] if isinstance(gallery[-1], str) else gallery[-1][0]
            return {"init_image_for_video": last_img, "video_prompt": prompt or ""}
        return {}

    to_video_from_img.click(
        handoff_image_to_video,
        inputs=[img_gallery, img_prompt],
        outputs=context_state,
        queue=False,
    )

    return SimpleNamespace(img_prompt=img_prompt, img_gallery=img_gallery)
