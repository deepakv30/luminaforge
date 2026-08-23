#!/usr/bin/env bash
# LuminaForge Setup Script
# ========================
# One-command bootstrap for most Linux / macOS users.

set -e

echo "🚀 LuminaForge Setup"
echo "===================="

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "Installing core requirements..."
pip install -r requirements.txt

# Voice support (recommended)
pip install piper-tts

# Optional heavy extras
# pip install TTS bitsandbytes   # Note: TTS/Coqui often fails on Python 3.12+

echo ""
echo "✅ Core packages installed."

# Create default directories
mkdir -p data/voice_profiles data/projects outputs/{images,videos,audio}

# Copy example config if missing
if [ ! -f config.yaml ]; then
    echo "No config.yaml found — using the one in the repo."
fi

echo ""
echo "Next steps:"
echo "1. Install and run Ollama: https://ollama.com"
echo "   ollama serve"
echo "   ollama pull llama3.2:3b"
echo ""
echo "2. Install voice support (recommended):"
echo "   pip install piper-tts"
echo ""
echo "3. (Recommended for Video) Install ComfyUI: https://github.com/comfyanonymous/ComfyUI"
echo "   Start it on http://localhost:8188"
echo ""
echo "4. Run the app:"
echo "   python app.py"
echo ""
echo "Enjoy creating locally! ✨"