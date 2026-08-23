# Model Recommendations for LuminaForge

## Text / LLM (Ollama)

LuminaForge supports both **local Ollama** and **Ollama Cloud**.

| Hardware Tier     | Recommended Model          | Use Case                     | Notes                          |
|-------------------|----------------------------|------------------------------|--------------------------------|
| Any (fast)        | `llama3.2:3b` or `1b`      | Prompt enhancement + chat    | Excellent speed/quality        |
| 16GB+             | `qwen2.5:7b` or `gemma2:9b`| High quality writing         | Great reasoning                |
| 24GB+             | `llama3.1:8b` / `mistral`  | Long context, complex tasks  | Best overall                   |

**Local (default)**: Run your own `ollama serve`.

**Ollama Cloud**:
- Set `use_cloud: true` in `config.yaml` under `text:`.
- Provide your API key (or `OLLAMA_API_KEY` environment variable).
- Useful when you want powerful models without running them locally.

**Tip**: Keep enhancer model small (`llama3.2:3b`) for instant prompt improvements.

## Image Generation (diffusers)

**Best daily driver (2025-2026)**:
- `black-forest-labs/FLUX.1-schnell` — 4 steps, beautiful results, fast on 12-16GB+
- `stabilityai/sdxl-turbo` — extremely fast for iteration

**Quality alternatives**:
- SDXL base, SD 3.5 medium, AuraFlow, etc.

**Low VRAM (≤12GB)**:
- Use `sdxl-turbo` or `flux-schnell` + `enable_cpu_offload: true`
- Or run via ComfyUI with `--lowvram` flags

## Video

**Primary recommendation**: ComfyUI + latest community video nodes
- CogVideoX-2B / 5B
- Mochi 1
- LTX Video
- Hunyuan Video

**Workflows shipped**:
- `cogvideox_t2v.json`
- `mochi_i2v.json`

Load them in ComfyUI and customize.

## Voice

**Recommended**: Piper TTS (`pip install piper-tts`)

- Very fast, high quality, actively maintained
- Works great on Python 3.12+
- Small on-disk models (~50-100MB)

**Coqui XTTS-v2**:
- Best zero-shot cloning quality
- **Does not support Python 3.12+** on PyPI (as of mid-2026)
- If you need it, create venv with Python 3.11

Always record 8-12s of clean reference speech for best clones when using voice profiles.

## ComfyUI Bridge

LuminaForge will auto-detect a running ComfyUI at `http://localhost:8188`.
You can use any advanced workflows you already have (IP-Adapter, ControlNet, Reactor, etc.).

## Adding New Models (2-5 minutes)

1. Text: `ollama pull <model>`
2. Image: Edit `config.yaml` → `image.default_model`
3. Video: Drop new workflow JSON in `workflows/comfyui_workflows/`
4. Voice: Add profile via Voice Lab UI

Then restart or reload in Model Manager.