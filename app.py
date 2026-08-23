#!/usr/bin/env python3
"""
LuminaForge - Local Multimodal AI Productivity Studio
=====================================================

Main Gradio application entrypoint.

Run:
    python app.py
"""

from __future__ import annotations

import os
import shutil
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*Siglip.*Fast.*")
warnings.filterwarnings("ignore", message=".*device_discovery.*")
warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_ENTITY.*")

import gradio as gr

from ui.layout import build_header, build_sidebar, wire_nav
from generators import resolve_text_provider
from ui.services import CONFIG, HW
from ui.studios import (
    build_dashboard,
    build_history,
    build_image_studio,
    build_library,
    build_model_manager,
    build_settings,
    build_text_studio,
    build_video_studio,
    build_voice_lab,
    build_workflows,
)
from ui.theme import SHELL_JS, get_css_paths, get_head_html, get_runtime_css, get_theme


def create_app() -> gr.Blocks:
    initial_provider = resolve_text_provider(CONFIG.get("text", {}))

    with gr.Blocks(
        title="LuminaForge",
        analytics_enabled=False,
        fill_width=True,
        fill_height=True,
    ) as demo:
        mode_indicator, global_search, search_btn = build_header(initial_provider)
        context_state = gr.State({})
        nav_buttons = build_sidebar()

        with gr.Tabs(
            elem_id="lf-main-tabs",
            elem_classes=["main-tabs"],
            selected="dashboard",
        ) as main_tabs:
            with gr.Tab("Dashboard", id="dashboard", scale=1):
                build_dashboard(demo)
            with gr.Tab("Text Studio", id="text", scale=1):
                text = build_text_studio(context_state)
            with gr.Tab("Image Studio", id="image", scale=1):
                build_image_studio(context_state)
            with gr.Tab("Video Studio", id="video", scale=1):
                build_video_studio(context_state)
            with gr.Tab("Voice Lab", id="voice", scale=1):
                build_voice_lab(context_state)
            with gr.Tab("Workflows", id="workflows", scale=1):
                build_workflows()
            with gr.Tab("Prompt Library", id="library", scale=1):
                build_library()
            with gr.Tab("History", id="history", scale=1):
                history = build_history(demo)
            with gr.Tab("Model Manager", id="models", scale=1):
                build_model_manager()
            with gr.Tab("Settings", id="settings", scale=1):
                build_settings(mode_indicator, text.text_model)

        wire_nav(nav_buttons, main_tabs)

        def go_search(query: str):
            from ui.services import history_rows

            q = query or ""
            return gr.update(selected="history"), q, history_rows(q, 30, "All")

        for trigger in (global_search.submit, search_btn.click):
            trigger(
                go_search,
                inputs=global_search,
                outputs=[main_tabs, history.hist_search, history.history_table],
                queue=False,
            )

        gr.Markdown(
            "LuminaForge • Fully local • MIT License • Make something beautiful today.",
            elem_classes=["lf-footer"],
        )

    return demo


_BROWSER_BINS = (
    "firefox",
    "chromium",
    "chromium-browser",
    "google-chrome",
    "google-chrome-stable",
    "brave-browser",
    "microsoft-edge",
)


def _should_open_browser() -> bool:
    flag = os.environ.get("LUMINAFORGE_INBROWSER", "").strip().lower()
    if flag in ("1", "true", "yes"):
        return True
    if flag in ("0", "false", "no"):
        return False
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    # xdg-open exists on headless boxes too; require a real browser binary.
    return any(shutil.which(name) for name in _BROWSER_BINS)


if __name__ == "__main__":
    print("Starting LuminaForge...", flush=True)
    print(f"Hardware: {HW}", flush=True)
    theme = get_theme()
    app = create_app()
    app.queue(default_concurrency_limit=CONFIG.get("ui", {}).get("concurrency_limit", 1))
    open_browser = _should_open_browser()
    print(f"Launching on http://0.0.0.0:7860 (inbrowser={open_browser})", flush=True)
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=open_browser,
        show_error=True,
        theme=theme,
        css=get_runtime_css(CONFIG),
        css_paths=get_css_paths(),
        js=SHELL_JS,
        head=get_head_html(),
        pwa=True,
    )
