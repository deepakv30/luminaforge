from __future__ import annotations

from types import SimpleNamespace

import gradio as gr

from ui.components import generate_button, prompt_box
from ui.services import CONFIG, enhance_and_update, format_metadata, lazy_init, log_op, save_result_to_history


def build_video_studio(context_state: gr.State) -> SimpleNamespace:
    gr.Markdown("### Text-to-Video & Image-to-Video (via ComfyUI recommended)")

    with gr.Row(elem_classes=["studio-shell"]):
        with gr.Column(scale=5, min_width=320, elem_classes=["studio-controls"]):
            vid_prompt = prompt_box("Video Prompt / Description", lines=3)

            with gr.Row(elem_classes=["wrap-row"]):
                vid_workflow = gr.Dropdown(
                    choices=["cogvideox_t2v", "mochi_i2v", "ltx_t2v"],
                    value=CONFIG.get("video", {}).get("default_workflow"),
                    label="Workflow",
                    min_width=160,
                )
                vid_duration = gr.Slider(2, 12, 5, 1, label="Duration (seconds)")
                vid_fps = gr.Slider(8, 30, 16, 1, label="FPS")

            with gr.Accordion("Advanced Parameters", open=False):
                with gr.Row(elem_classes=["wrap-row"]):
                    vid_width = gr.Slider(256, 1280, 832, 32, label="Width")
                    vid_height = gr.Slider(256, 1280, 480, 32, label="Height")
                init_for_vid = gr.Image(label="Starting Image (Image-to-Video)", type="filepath")
                vid_model = gr.Textbox(label="Model / Checkpoint (ComfyUI)", value="")

            with gr.Row(elem_classes=["wrap-row"]):
                vid_gen_btn = generate_button("Generate Video Clip")
                enhance_vid_btn = gr.Button("Enhance Prompt for Video", min_width=160)

        with gr.Column(scale=7, min_width=320, elem_classes=["studio-canvas"]):
            vid_output = gr.Video(label="Generated / Processed Video", height="min(50dvh, 480px)")
            with gr.Accordion("Metadata", open=False):
                vid_meta = gr.JSON()
            to_voice_from_vid = gr.Button(
                "Send Script to Voice Lab →",
                variant="secondary",
                elem_classes=["send-to-btn"],
                min_width=160,
            )

    def do_video(p, wf, dur, fps, w, h, init_img, model):
        with log_op("generate video", workflow=wf, duration=dur):
            _, _, vgen, _ = lazy_init()
            res = vgen.generate(
                prompt=p,
                init_image=init_img,
                duration=float(dur),
                fps=int(fps),
                width=int(w),
                height=int(h),
                workflow=wf,
                model=model or None,
            )
            save_result_to_history(res)
            video_path = str(res.output_path) if res.output_path else None
            return video_path, format_metadata(res)

    vid_gen_btn.click(
        do_video,
        inputs=[vid_prompt, vid_workflow, vid_duration, vid_fps, vid_width, vid_height, init_for_vid, vid_model],
        outputs=[vid_output, vid_meta],
    )

    enhance_vid_btn.click(
        lambda p: enhance_and_update(p, "cinematic", "video"),
        inputs=vid_prompt,
        outputs=vid_prompt,
    )

    def handoff_video_to_voice(video_path, prompt):
        return {"voice_text": prompt or "Add a compelling voiceover here."}

    to_voice_from_vid.click(
        handoff_video_to_voice,
        inputs=[vid_output, vid_prompt],
        outputs=context_state,
        queue=False,
    )

    return SimpleNamespace(vid_prompt=vid_prompt, vid_output=vid_output)
