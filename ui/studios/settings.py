from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from generators import resolve_text_provider
from ui.services import (
    CONFIG,
    HW,
    LABELS_BY_PROVIDER,
    OUTPUT_DIR,
    apply_text_backend,
)
from ui.theme import THEME_JS


def build_settings(mode_indicator: gr.HTML, text_model: gr.Dropdown) -> SimpleNamespace:
    gr.Markdown("### Configuration & Performance")

    with gr.Row(elem_classes=["wrap-row"]):
        out_path = gr.Textbox(value=str(OUTPUT_DIR), label="Output Directory", min_width=200)
        low_vram = gr.Checkbox(
            value=CONFIG.get("hardware", {}).get("low_vram_mode", False),
            label="Low VRAM Mode",
        )

    gr.Markdown("**Hardware Tips**")
    gr.Markdown(
        f"""
- Detected: {HW.get('device')} | VRAM ~{HW.get('vram_gb')} GB | MPS: {HW.get('mps_available')}
- For 8-12GB cards: use FLUX.1-schnell with 4 steps + low_vram_mode or CPU offload.
- Apple Silicon: MPS works well for smaller models.
"""
    )

    gr.Markdown("### Text backend")
    text_cfg = CONFIG.get("text", {})
    initial_provider = resolve_text_provider(text_cfg)
    initial_label = LABELS_BY_PROVIDER.get(initial_provider, "Local Ollama")
    ollama_key_init = text_cfg.get("api_key", "")
    xai_key_init = text_cfg.get("xai_api_key", "") or (
        ollama_key_init if str(ollama_key_init).startswith("xai-") else ""
    )
    ollama_key_display = "" if not ollama_key_init or str(ollama_key_init).startswith("xai-") else "•••••••• (key loaded)"
    xai_key_display = "" if not xai_key_init else "•••••••• (key loaded)"

    provider_choice = gr.Dropdown(
        ["Local Ollama", "Ollama Cloud", "xAI (Grok)"],
        value=initial_label,
        label="Provider",
        info="Local Ollama, hosted Ollama Cloud, or xAI Grok (XAI_API_KEY)",
    )
    local_base_url = gr.Textbox(
        value=text_cfg.get("base_url", "http://localhost:11434"),
        label="Local Ollama URL",
        visible=initial_provider == "ollama",
    )
    cloud_base_url = gr.Textbox(
        value=text_cfg.get("cloud_base_url", "https://api.ollama.com"),
        label="Ollama Cloud Base URL",
        placeholder="https://api.ollama.com",
        visible=initial_provider == "ollama_cloud",
    )
    cloud_api_key = gr.Textbox(
        value=ollama_key_display,
        label="Ollama Cloud API Key",
        placeholder="Or leave blank to use $OLLAMA_API_KEY",
        type="password",
        visible=initial_provider == "ollama_cloud",
    )
    xai_api_key = gr.Textbox(
        value=xai_key_display,
        label="xAI API Key",
        placeholder="xai-… or leave blank to use $XAI_API_KEY",
        type="password",
        visible=initial_provider == "xai",
        info="Create a key at https://console.x.ai",
    )

    def _toggle_provider_fields(label):
        p = {"Local Ollama": "ollama", "Ollama Cloud": "ollama_cloud", "xAI (Grok)": "xai"}.get(label, "ollama")
        return (
            gr.update(visible=p == "ollama"),
            gr.update(visible=p == "ollama_cloud"),
            gr.update(visible=p == "ollama_cloud"),
            gr.update(visible=p == "xai"),
        )

    provider_choice.change(
        _toggle_provider_fields,
        inputs=provider_choice,
        outputs=[local_base_url, cloud_base_url, cloud_api_key, xai_api_key],
        queue=False,
    )

    with gr.Row(elem_classes=["wrap-row"]):
        apply_cloud_btn = gr.Button("Apply (Live Reload)", variant="primary", min_width=160)
        save_cloud_btn = gr.Button("Save to config.yaml & Apply", min_width=180)

    cloud_status = gr.Markdown()

    apply_cloud_btn.click(
        fn=lambda p, loc, url, ok, xk: apply_text_backend(p, loc, url, ok, xk, persist=False),
        inputs=[provider_choice, local_base_url, cloud_base_url, cloud_api_key, xai_api_key],
        outputs=[cloud_status, mode_indicator, text_model],
        queue=False,
    )
    save_cloud_btn.click(
        fn=lambda p, loc, url, ok, xk: apply_text_backend(p, loc, url, ok, xk, persist=True),
        inputs=[provider_choice, local_base_url, cloud_base_url, cloud_api_key, xai_api_key],
        outputs=[cloud_status, mode_indicator, text_model],
        queue=False,
    )

    gr.Markdown(
        "**Note:** Backend changes apply immediately for new generations. "
        "xAI uses `https://api.x.ai/v1` (Grok). Prefer `$XAI_API_KEY` over committing keys."
    )

    gr.Markdown("### Theme & Accessibility")
    theme_choice = gr.Dropdown(
        ["Dark (Default)", "High Contrast", "Light (Experimental)"],
        value="Dark (Default)",
        label="UI Theme",
    )
    theme_choice.change(fn=None, inputs=theme_choice, js=THEME_JS, queue=False)

    gr.Markdown("### Danger zone")
    clear_btn = gr.Button("Clear History (danger)", variant="stop")

    def clear_history():
        from generators import init_history_db
        import ui.services as services

        dbp = Path(CONFIG.get("history", {}).get("db_path", "./data/history.db"))
        if dbp.exists():
            dbp.unlink()
        services.DB = init_history_db(dbp)
        return "History cleared."

    clear_status = gr.Textbox(label="Clear status")
    clear_btn.click(clear_history, outputs=clear_status, queue=False)

    # Keep unused widgets referenced so Gradio renders them.
    _ = (out_path, low_vram)

    return SimpleNamespace(theme_choice=theme_choice, cloud_status=cloud_status)
