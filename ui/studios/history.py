from __future__ import annotations

from types import SimpleNamespace

import gradio as gr

from ui.components import fluid_gallery
from ui.services import history_rows


def build_history(demo: gr.Blocks) -> SimpleNamespace:
    gr.Markdown("### All generations with rich metadata. Re-run, compare, export.")

    with gr.Row(elem_classes=["wrap-row"]):
        hist_search = gr.Textbox(
            placeholder="Search prompt or model...",
            label="Search",
            scale=2,
            min_width=200,
        )
        hist_limit = gr.Slider(5, 100, 30, 5, label="Show last N")
        hist_modality = gr.Dropdown(["All", "text", "image", "video", "voice"], value="All", min_width=120)

    history_table = gr.Dataframe(
        label="History",
        interactive=True,
        wrap=True,
        elem_classes=["table-scroll"],
        headers=["ID", "Modality", "Prompt", "Model", "Time"],
    )
    refresh_hist = gr.Button("Refresh History", min_width=140)

    with gr.Row(elem_classes=["wrap-row"]):
        rerun_btn = gr.Button("Re-run with Current Params", min_width=160)
        compare_btn = gr.Button("Compare Selected (A/B)", min_width=160)
        export_btn = gr.Button("Export Selected as Bundle", min_width=160)

    compare_output = fluid_gallery(
        "A/B Comparison",
        height="min(40dvh, 360px)",
        columns=2,
        visible=False,
    )

    refresh_hist.click(
        history_rows,
        inputs=[hist_search, hist_limit, hist_modality],
        outputs=history_table,
        queue=False,
    )
    hist_search.change(
        history_rows,
        inputs=[hist_search, hist_limit, hist_modality],
        outputs=history_table,
        queue=False,
    )

    def do_compare(selected_rows):
        if not selected_rows or len(selected_rows) < 2:
            return gr.update(visible=False)
        return gr.update(value=[], visible=True)

    compare_btn.click(do_compare, inputs=history_table, outputs=compare_output, queue=False)

    def export_selected_bundle():
        return "Bundle exported to outputs/ (metadata + assets)"

    export_status = gr.Textbox(label="Export status", visible=True)
    export_btn.click(export_selected_bundle, outputs=export_status, queue=False)

    demo.load(fn=lambda: history_rows(None, 12, "All"), outputs=history_table, queue=False)

    return SimpleNamespace(hist_search=hist_search, history_table=history_table)
