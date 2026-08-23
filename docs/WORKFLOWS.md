# LuminaForge Workflows & Chaining Guide

## Built-in Wizards

1. **Social Reel Creator**
   - Idea → Enhanced script → Hero image → 5s video clip → Voiceover
   - One-click from Dashboard or Workflows tab

2. **Educational Explainer**
   - Topic → Clean script + supporting diagram

## End-to-End Chaining (The Magic)

Every generation returns a `GenerationResult` containing:
- Output (text, image path, video path, audio path)
- Full prompt + parameters + seed + model
- Rich metadata

The UI "Send to →" buttons pass this context to the next tab.

Example flow you can do in <90 seconds:
1. Text Studio: Type rough idea → click Enhance → Generate
2. Click "Send to Image Studio"
3. Image Studio: Adjust → Generate beautiful visual
4. Click "Send to Video Studio"
5. Video Studio: Generate 5s motion clip
6. Click "Send to Voice Lab"
7. Generate professional voiceover

No copy-paste ever required.

## Custom Pipelines (for developers)

See `workflows/pipelines.py`. The functions are deliberately tiny and explicit.

You can register new wizards by:
1. Adding a new function in pipelines.py
2. Adding a button in the Workflows tab
3. Wiring the outputs to the History + Gallery

## ComfyUI Workflows

Place real exported workflows (with your custom nodes) in:
`workflows/comfyui_workflows/`

LuminaForge can trigger them by name.

## Post-processing

Video tab supports basic MoviePy operations (caption, speed, simple upscale).
Extend in `generators/video.py: postprocess()`.

## Tips

- Keep clips short (3-8s) on local hardware, then upscale in a second pass.
- Use prompt enhancer aggressively — it is tuned for each modality.
- Save voice profiles early. They become extremely powerful when reused across projects.