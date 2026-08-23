from __future__ import annotations

import json
from types import SimpleNamespace

import gradio as gr

from ui.components import generate_button
from ui.services import CONFIG, log_op
from workflows.pipelines import run_educational_explainer, run_social_reel_pipeline


def build_workflows() -> SimpleNamespace:
    gr.Markdown("### Pre-built Productivity Wizards")

    with gr.Row(elem_classes=["studio-shell"]):
        with gr.Column(scale=5, min_width=320, elem_classes=["studio-controls"]):
            wizard_idea = gr.Textbox(
                label="Core Idea / Topic",
                value="Launch a beautiful local AI productivity tool",
            )
            wizard_type = gr.Dropdown(
                ["Social Reel Creator", "Educational Explainer"],
                value="Social Reel Creator",
            )
            run_wizard = generate_button("Run Full Wizard Pipeline")
            gr.Markdown(
                "**How chaining works**: Every generation returns rich metadata. "
                '"Send to..." buttons in other tabs populate prompts/images/text automatically.'
            )

        with gr.Column(scale=7, min_width=320, elem_classes=["studio-canvas"]):
            wizard_output = gr.Textbox(label="Pipeline Result Summary", lines=12)
            with gr.Accordion("Generated Artifacts", open=False):
                wizard_artifacts = gr.JSON(label="Generated Artifacts")

    def run_wizard_fn(idea, wtype):
        with log_op("pipeline wizard", kind=wtype):
            if wtype == "Social Reel Creator":
                ctx = run_social_reel_pipeline(idea, CONFIG)
            else:
                ctx = run_educational_explainer(idea, CONFIG)
        summary = f"Pipeline completed.\n\n{json.dumps(ctx.get('final_package', ctx), indent=2)[:1500]}"
        return summary, ctx.get("final_package", ctx)

    run_wizard.click(run_wizard_fn, inputs=[wizard_idea, wizard_type], outputs=[wizard_output, wizard_artifacts])

    return SimpleNamespace()
