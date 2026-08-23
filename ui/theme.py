"""
LuminaForge Gradio Theme + CSS/JS/head helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import gradio as gr

CSS_DIR = Path(__file__).resolve().parent / "css"

THEME_LABELS = {
    "Dark (Default)": "dark",
    "High Contrast": "high-contrast",
    "Light (Experimental)": "light",
}


def get_theme():
    return gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
        spacing_size="md",
        radius_size="md",
        text_size="md",
    ).set(
        button_primary_background_fill="*primary_600",
        button_primary_background_fill_hover="*primary_700",
        block_title_text_color="*neutral_100",
        block_label_text_color="*neutral_300",
    )


def get_css_paths() -> List[Path]:
    return [
        CSS_DIR / "tokens.css",
        CSS_DIR / "shell.css",
        CSS_DIR / "responsive.css",
    ]


def get_runtime_css(config: dict | None = None) -> str:
    """Inline CSS variables from config.yaml (loaded first, then css_paths)."""
    ui = (config or {}).get("ui", {})
    maxw = ui.get("content_max_width", 1680)
    tile = ui.get("gallery_min_tile", 160)
    cols = ui.get("gallery_columns", 3)
    return (
        ":root {"
        f"--lf-content-max: {maxw}px;"
        f"--lf-gallery-tile: {tile}px;"
        f"--lf-gallery-max-cols: {cols};"
        "}"
    )


def load_custom_css(config: dict | None = None) -> str:
    parts = [get_runtime_css(config)]
    for path in get_css_paths():
        parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


# Back-compat alias used by older launch snippets
CUSTOM_CSS = None


def get_head_html() -> str:
    return """
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0b1120">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
"""


SHELL_JS = """
() => {
  const SIDEBAR_BP = window.matchMedia("(max-width: 1023px)");

  const sidebarEl = () => document.querySelector(".sidebar");
  const isOpen = (el) => el && el.classList.contains("open");

  const syncSidebar = () => {
    const el = sidebarEl();
    if (!el) return;
    const open = isOpen(el);
    const wantClosed = SIDEBAR_BP.matches;
    if (wantClosed && open) el.querySelector(".toggle-button")?.click();
    if (!wantClosed && !open) el.querySelector(".toggle-button")?.click();
  };

  const markNav = () => {
    const selected = document.querySelector("#lf-main-tabs [data-tab-id][aria-selected='true']")
      || document.querySelector("#lf-main-tabs button.selected");
    const id = selected?.getAttribute("data-tab-id");
    document.querySelectorAll("[id^='nav-']").forEach((btn) => {
      const match = id && btn.id === `nav-${id}`;
      btn.classList.toggle("active", !!match);
    });
  };

  SIDEBAR_BP.addEventListener("change", syncSidebar);

  const start = Date.now();
  const iv = setInterval(() => {
    if (sidebarEl() || Date.now() - start > 6000) {
      clearInterval(iv);
      syncSidebar();
      markNav();
    }
  }, 120);

  const obs = new MutationObserver(markNav);
  const boot = () => {
    const tabs = document.getElementById("lf-main-tabs");
    if (tabs) obs.observe(tabs, { attributes: true, subtree: true, attributeFilter: ["aria-selected", "class"] });
  };
  setTimeout(boot, 400);
}
"""


THEME_JS = """
(label) => {
  const map = {
    "Dark (Default)": "dark",
    "High Contrast": "high-contrast",
    "Light (Experimental)": "light"
  };
  const key = map[label] || "dark";
  document.documentElement.setAttribute("data-lf-theme", key);
  const darkEls = () => document.querySelectorAll(".gradio-container, body, html");
  if (key === "light") {
    darkEls().forEach((el) => el.classList.remove("dark"));
  } else {
    darkEls().forEach((el) => el.classList.add("dark"));
  }
}
"""
