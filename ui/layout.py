"""
App shell: header, collapsible sidebar nav, main tab switcher.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import gradio as gr

# (tab_id, sidebar label)
NAV_GROUPS: List[Tuple[str, List[Tuple[str, str]]]] = [
    (
        "Create",
        [
            ("dashboard", "Dashboard"),
            ("text", "Text Studio"),
            ("image", "Image Studio"),
            ("video", "Video Studio"),
            ("voice", "Voice Lab"),
            ("workflows", "Workflows"),
        ],
    ),
    (
        "Library",
        [
            ("library", "Prompt Library"),
            ("history", "History"),
        ],
    ),
    (
        "System",
        [
            ("models", "Model Manager"),
            ("settings", "Settings"),
        ],
    ),
]

TAB_IDS = [tid for _, items in NAV_GROUPS for tid, _ in items]


def mode_badge_html(provider: str | bool) -> str:
    if provider is True:
        provider = "ollama_cloud"
    if provider is False or provider is None:
        provider = "ollama"
    provider = str(provider).lower()
    if provider in ("xai", "spacexai", "grok"):
        return '<span class="status-badge status-xai">⚡ xAI</span>'
    if provider in ("ollama_cloud", "cloud"):
        return '<span class="status-badge status-cloud">☁️ Cloud</span>'
    return '<span class="status-badge status-local">🖥️ Local</span>'


def build_header(initial_provider: str | bool):
    with gr.Row(elem_classes=["top-bar"]):
        gr.Markdown("**✨ LuminaForge**", elem_classes=["lumina-header"])
        mode_indicator = gr.HTML(
            value=mode_badge_html(initial_provider),
            elem_classes=["mode-indicator"],
        )
        global_search = gr.Textbox(
            placeholder="Search history…",
            scale=3,
            container=False,
            elem_classes=["global-search"],
            label="Search",
            show_label=False,
        )
        search_btn = gr.Button("🔍", size="sm", min_width=44, elem_classes=["search-btn"])
    return mode_indicator, global_search, search_btn


def build_sidebar() -> Dict[str, gr.Button]:
    buttons: Dict[str, gr.Button] = {}
    with gr.Sidebar(label="LuminaForge", open=True, width=240, elem_id="lf-sidebar"):
        gr.Markdown("**LuminaForge**")
        for group, items in NAV_GROUPS:
            gr.Markdown(group, elem_classes=["nav-group-label"])
            for tab_id, label in items:
                buttons[tab_id] = gr.Button(
                    label,
                    variant="secondary",
                    size="sm",
                    elem_id=f"nav-{tab_id}",
                    elem_classes=["nav-btn"],
                    min_width=0,
                )
    return buttons


def wire_nav(buttons: Dict[str, gr.Button], tabs: gr.Tabs) -> None:
    def selector(tab_id: str) -> Callable:
        def _select():
            return gr.update(selected=tab_id)

        return _select

    for tab_id, btn in buttons.items():
        btn.click(selector(tab_id), outputs=tabs, queue=False)
