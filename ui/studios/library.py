from __future__ import annotations

from types import SimpleNamespace

import gradio as gr

from ui.services import CONFIG, PROMPT_LIBRARY, add_to_prompt_library


def build_library() -> SimpleNamespace:
    gr.Markdown("### Searchable Prompt Templates + AI Enhancement")

    with gr.Row(elem_classes=["wrap-row"]):
        search = gr.Textbox(
            label="Search templates",
            placeholder="marketing, portrait, education...",
            scale=2,
            min_width=200,
        )
        category_filter = gr.Dropdown(
            ["All"] + CONFIG.get("prompt_library", {}).get("categories", []),
            value="All",
            label="Category",
            min_width=140,
        )

    lib_output = gr.Dataframe(
        value=[[p["category"], p["title"], p["prompt"][:90] + "..."] for p in PROMPT_LIBRARY[:12]],
        headers=["Category", "Title", "Prompt Preview"],
        interactive=False,
        wrap=True,
        elem_classes=["table-scroll"],
    )

    selected_prompt = gr.Textbox(label="Selected / Loaded Prompt", lines=3)
    with gr.Row(elem_classes=["wrap-row"]):
        load_btn = gr.Button("Load Selected into Current Prompt", min_width=180)
        enhance_from_lib = gr.Button("Enhance This Prompt", min_width=160)

    with gr.Accordion("Add Your Own Template", open=False):
        new_title = gr.Textbox(label="Title")
        new_cat = gr.Dropdown(CONFIG.get("prompt_library", {}).get("categories", ["User"]), label="Category")
        new_prompt = gr.Textbox(label="Prompt", lines=3)
        add_btn = gr.Button("Add to Library")
        add_status = gr.Markdown()

    def filter_library(q, cat):
        items = PROMPT_LIBRARY
        if cat != "All":
            items = [x for x in items if x["category"] == cat]
        if q:
            q = q.lower()
            items = [x for x in items if q in x["title"].lower() or q in x["prompt"].lower()]
        return [[x["category"], x["title"], x["prompt"][:100]] for x in items[:20]]

    search.change(filter_library, inputs=[search, category_filter], outputs=lib_output, queue=False)
    category_filter.change(filter_library, inputs=[search, category_filter], outputs=lib_output, queue=False)

    def load_first_from_df(df):
        if df is not None and len(df) > 0:
            title = df[0][1]
            for item in PROMPT_LIBRARY:
                if item["title"] == title:
                    return item["prompt"]
        return ""

    load_btn.click(load_first_from_df, inputs=lib_output, outputs=selected_prompt, queue=False)
    add_btn.click(add_to_prompt_library, inputs=[new_title, new_prompt, new_cat], outputs=add_status, queue=False)

    # Enhance button was previously unwired in a useful way; keep it local to selected_prompt.
    from ui.services import enhance_and_update

    enhance_from_lib.click(
        lambda p: enhance_and_update(p, "detailed cinematic", "image"),
        inputs=selected_prompt,
        outputs=selected_prompt,
    )

    return SimpleNamespace(search=search, selected_prompt=selected_prompt)
