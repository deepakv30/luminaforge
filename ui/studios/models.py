from __future__ import annotations

from types import SimpleNamespace

import gradio as gr

from ui.services import lazy_init, log_op


def build_model_manager() -> SimpleNamespace:
    gr.Markdown("### Ollama + Configured Models + Voice Profiles")
    gr.Markdown(
        "Supports **local Ollama**, **Ollama Cloud**, and **xAI (Grok)**.\n"
        "Pick the backend in **Settings**. xAI uses `XAI_API_KEY` (or `text.xai_api_key` in config)."
    )

    with gr.Row(elem_classes=["wrap-row"]):
        ollama_status = gr.Markdown()
        pull_model_name = gr.Textbox(label="Ollama model to pull (e.g. llama3.2:3b)", min_width=200)
        pull_btn = gr.Button("Pull Model", min_width=120)

    ollama_list = gr.JSON(label="Available Ollama Models")

    def get_ollama_status():
        tgen, _, _, _ = lazy_init()
        status = tgen.get_status() if hasattr(tgen, "get_status") else {}
        provider = status.get("provider") or ("ollama_cloud" if status.get("use_cloud") else "ollama")
        names = {"xai": "⚡ xAI", "ollama_cloud": "☁️ Ollama Cloud", "ollama": "🖥️ Local Ollama"}
        return (
            f"**{names.get(provider, provider)}**: {status.get('connected', False)} | "
            f"Default: {status.get('default_model')} | Base: {status.get('base_url')}",
            status.get("available_models", []),
        )

    def pull_model(m):
        with log_op("pull model", model=m):
            tgen, _, _, _ = lazy_init()
            return tgen.pull_model(m) if tgen else {"error": "not init"}

    pull_btn.click(pull_model, inputs=pull_model_name, outputs=ollama_status)

    refresh_ollama = gr.Button("Refresh Ollama Status", min_width=160)
    refresh_ollama.click(get_ollama_status, outputs=[ollama_status, ollama_list], queue=False)

    gr.Markdown("#### Voice Profiles")
    voice_profiles_list = gr.JSON()
    refresh_vp = gr.Button("Refresh Voice Profiles", min_width=160)
    refresh_vp.click(
        lambda: (lazy_init()[3].list_voice_profiles()),
        outputs=voice_profiles_list,
        queue=False,
    )

    gr.Markdown(
        "Edit `config.yaml` directly for image/video models, or use the Settings tab. "
        "Then restart the app for full effect."
    )

    return SimpleNamespace(ollama_status=ollama_status)
