from __future__ import annotations

from types import SimpleNamespace

import gradio as gr

from ui.components import fluid_gallery
from ui.services import CONFIG, HW, get_status_markdown, load_recent_gallery_items, log_op
from workflows.pipelines import run_educational_explainer, run_social_reel_pipeline


def build_dashboard(demo: gr.Blocks) -> SimpleNamespace:
    gr.Markdown("### System Status & Quick Actions")

    with gr.Row(elem_classes=["wrap-row"]):
        with gr.Column(scale=1, min_width=280, elem_classes=["status-card"]):
            status_text = gr.Markdown(value=get_status_markdown())
            with gr.Accordion("Hardware details", open=False):
                gr.JSON(value=HW, label="Hardware Detection")

        with gr.Column(scale=2, min_width=280):
            gr.Markdown("#### Quick Start Workflows")
            with gr.Row(elem_classes=["wrap-row"]):
                reel_btn = gr.Button("🎬 Social Reel Creator", variant="primary", min_width=180)
                edu_btn = gr.Button("📚 Educational Explainer", variant="secondary", min_width=180)
            gr.Markdown(
                "These wizards automatically chain Text → Image → Video → Voice. "
                "Results appear in History and can be further refined."
            )

    recent_gallery = fluid_gallery("Recent Outputs", height="min(30dvh, 240px)", columns=2)

    def run_reel_quick(idea: str = "Launch a new productivity app called Lumina"):
        with log_op("pipeline social_reel"):
            ctx = run_social_reel_pipeline(idea, CONFIG)
        return (
            f"Social Reel pipeline completed. Check History and Video/Voice tabs.\n\n"
            f"Script preview:\n{str(ctx.get('script', ''))[:280]}"
        )

    def run_edu_quick(idea: str = "Explain how local AI studios work"):
        with log_op("pipeline educational_explainer"):
            ctx = run_educational_explainer(idea, CONFIG)
        return (
            f"Educational Explainer completed.\n\n"
            f"{str(ctx.get('final_package', ctx))[:280]}"
        )

    reel_btn.click(fn=run_reel_quick, inputs=[], outputs=status_text)
    edu_btn.click(fn=run_edu_quick, inputs=[], outputs=status_text)

    demo.load(fn=load_recent_gallery_items, outputs=recent_gallery, queue=False)

    return SimpleNamespace(status_text=status_text, recent_gallery=recent_gallery)
