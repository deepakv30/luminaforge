# ✨ LuminaForge

**A beautiful, local-only multimodal AI productivity studio.**

Generate high-quality text, images, videos, and voiceovers using only your own computer and open-source models. No API keys, no subscriptions, no data leaving your machine.

> **"Dramatically boost local productivity for content creators, marketers, educators, designers, and developers."**

---

## Why LuminaForge?

Most local AI tools are fragmented toys. LuminaForge is different:

- **End-to-end workflows are the killer feature** — One-click "Send to Image / Video / Voice" that passes rich context between modalities without copy-paste.
- **Model flexibility is first-class** — Swap models by editing one line in `config.yaml` or using the Model Manager UI.
- **Hardware-aware by design** — Excellent defaults for 8–12 GB, 16–24 GB, and 24 GB+ GPUs + Apple Silicon.
- **Built for real work** — Prompt enhancer, template library, rich history, batch support, voice profiles, and metadata on everything.
- **Truly private & cost-free** — 100% local by default (Ollama + diffusers + optional ComfyUI).

If you have ever spent hours copying prompts between tools, this is the antidote.

---

## 5-Minute Quickstart

```bash
# 1. Clone and enter
git clone <your-repo>/luminaforge.git
cd luminforge

# 2. (Recommended) Create venv + install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Start Ollama (if not already running)
ollama serve
ollama pull llama3.2:3b

# 4. Launch LuminaForge
python app.py
```

Open http://localhost:7860 — the beautiful interface will launch in your browser.

**First run tips**:
- Text Studio works with local Ollama or Ollama Cloud (see Model Manager / config).
- Image Studio needs `torch` + `diffusers` (already in requirements).
- Video works best when you also run [ComfyUI](https://github.com/comfyanonymous/ComfyUI) locally.
- **Voice**: Install `pip install piper-tts` (recommended). Coqui XTTS does not yet support Python 3.12+. Piper is faster and works great.

---

## Hardware & Model Recommendations

See the full table in [docs/MODEL_RECOMMENDATIONS.md](docs/MODEL_RECOMMENDATIONS.md).

**Quick guidance**:

| Your Hardware       | Best Image Model                  | Text Model        | Video Strategy             |
|---------------------|-----------------------------------|-------------------|----------------------------|
| 8–12 GB             | `stabilityai/sdxl-turbo` or FLUX-schnell + low VRAM | `llama3.2:3b`    | ComfyUI (short clips)     |
| 16–24 GB            | `black-forest-labs/FLUX.1-schnell` | `qwen2.5:7b`     | ComfyUI (excellent)       |
| 24 GB+              | FLUX.1-schnell or dev             | `llama3.1:8b`    | ComfyUI + longer clips    |
| Apple Silicon       | SDXL-Turbo or small Flux          | llama3.2 family  | ComfyUI (M-series nodes)  |

---

## UI Tour (All 10 Tabs)

### 1. Dashboard
Hardware status, quick-start wizard buttons, recent activity at a glance.

### 2. Text Studio
- Full chat-style + advanced generation
- Prominent **Enhance Prompt** button (the secret weapon)
- Streaming-ready architecture
- Send-to-Image / Video / Voice buttons

### 3. Image Studio
Text-to-Image + Img2Img, negative prompts, resolution presets, batch, seed, live gallery with metadata. "Enhance" + "Send to Video".

### 4. Video Studio
Text-to-Video and Image-to-Video powered primarily by ComfyUI workflows. Duration/FPS controls + post-processing. Send to Voice Lab.

### 5. Voice Lab
- **Voice Cloning**: Upload/record reference audio → save named profile
- **TTS / Voiceover**: Pick profile, paste text or load from history, control speed/emotion, preview + download WAV

### 6. Workflows & Pipelines
Pre-built wizards:
- Social Reel Creator (idea → script → image → video → voiceover)
- Educational Explainer

Plus room to grow custom chains.

### 7. Prompt Library & Enhancer
15 high-quality curated templates across Marketing, Video Scripts, Education, Storytelling, etc. One-click load + AI enhance. Add your own.

### 8. History & Projects
Filterable, searchable history across all modalities. Re-run, compare, export with full metadata. Group into projects.

### 9. Model Manager
- Ollama: list / pull / delete
- Image/Video model config
- Voice profile management
- "Validate & Reload" pattern (restart for full effect in v0.1)

### 10. Settings
Paths, performance toggles (quantization, slicing, tiling), hardware guidance, danger zone (clear data).

---

## Architecture & Design Highlights

```text
app.py                  # Gradio UI + orchestration
├── generators/
│   ├── base.py         # Protocols + GenerationResult (the contract)
│   ├── text.py         # Ollama + powerful prompt enhancer
│   ├── image.py        # diffusers + ComfyUI bridge
│   ├── video.py        # ComfyUI workflows + MoviePy postproc
│   ├── voice.py        # XTTS + Piper + profile manager
│   └── utils.py        # Hardware, history (SQLite), helpers
├── workflows/          # High-level pipelines + ComfyUI JSONs
├── ui/                 # Theme + reusable components
├── data/               # prompt_library.json + history.db + voice_profiles/
└── config.yaml         # Single source of truth for everything
```

**Key abstractions**:
- `GenerationResult` carries everything needed for reproducibility and chaining.
- Every generator implements a simple Protocol.
- "Send to..." actions update a shared `gr.State` context.

This design makes the project extremely maintainable.

---

## Customization Guide (Developer Friendly)

**Swap the default image model** (2 minutes):
Edit `config.yaml`:
```yaml
image:
  default_model: "stabilityai/sdxl-turbo"
```

**Add a new prompt template**:
Use the UI (Prompt Library tab) or edit `data/prompt_library.json`.

**Add a new pipeline wizard**:
1. Add function in `workflows/pipelines.py`
2. Wire a button in `app.py`

**Use your own ComfyUI workflows**:
Export from ComfyUI → drop JSON in `workflows/comfyui_workflows/your_wf.json` → select in Video Studio.

---

## Troubleshooting

**Ollama not connected**:
- For **local**: Run `ollama serve` and make sure `text.base_url` in config matches.
- For **Ollama Cloud**: Set `text.use_cloud: true` and provide `api_key` (or set the `OLLAMA_API_KEY` environment variable). See config.yaml for details.

To use Ollama Cloud:
1. Sign up and get an API key at https://ollama.com
2. In `config.yaml`:
   ```yaml
   text:
     use_cloud: true
     cloud_base_url: "https://api.ollama.com"
     api_key: "your-key-here"   # or omit and use env var
   ```
3. Restart the app. Cloud models will be listed in Text Studio and Model Manager.

**Image generation OOM / slow**:
- Enable `low_vram_mode: true`
- Reduce steps (Flux works great at 4)
- Use `sdxl-turbo`

**Video not working**:
- ComfyUI must be running on port 8188 with video nodes installed.
- The app gracefully falls back to placeholders.

**Voice not working / "No matching distribution" for TTS**:
- Coqui XTTS (`pip install TTS`) does **not** support Python 3.12+ yet.
- **Recommended fix**: `pip install piper-tts` (much easier, very good quality).
- The app defaults to Piper and will auto-download a voice model.

**Apple Silicon**:
MPS is supported for many operations. Some diffusers features work better on CUDA.

---

## Project Structure

```
luminforge/
├── app.py
├── config.yaml
├── requirements.txt
├── README.md
├── generators/...
├── workflows/...
├── ui/...
├── data/...
├── scripts/setup.sh
├── docs/
└── tests/
```

---

## Roadmap & Future Ideas

- Local RAG over your own documents
- Agentic multi-step planning
- Music / sound effect generation
- 3D asset generation (SVD + TripoSR style)
- Multi-user / team mode with projects
- Better batch + A/B comparison tooling
- Native support for more video models as they mature

Contributions extremely welcome.

---

## License

MIT License. Build whatever you want.

---

## Credits & Philosophy

Built with love for the local AI community in 2026.

Special thanks to the amazing work from:
- Ollama team
- Hugging Face (diffusers, transformers)
- ComfyUI community
- Coqui TTS / Piper teams
- Gradio team

LuminaForge exists to prove that **local multimodal tools can be more delightful and productive** than their cloud counterparts when designed with care.

Now go create something beautiful — entirely on your own machine.

**Run `python app.py` and start forging.**
