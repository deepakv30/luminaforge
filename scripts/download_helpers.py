"""
LuminaForge Model & Asset Helper (optional CLI)
===============================================

Usage examples:
    python scripts/download_helpers.py ollama llama3.2:3b
    python scripts/download_helpers.py check
"""

import sys
from generators.text import OllamaTextGenerator
from generators.utils import load_config


def main():
    cfg = load_config()
    text_cfg = cfg.get("text", {})
    tgen = OllamaTextGenerator(text_cfg)

    if len(sys.argv) < 2:
        print("Commands: check | ollama <model>")
        return

    cmd = sys.argv[1]
    if cmd == "check":
        print("Ollama status:", tgen.get_status())
    elif cmd == "ollama" and len(sys.argv) > 2:
        model = sys.argv[2]
        print(f"Pulling {model}...")
        res = tgen.pull_model(model)
        print(res)
    else:
        print("Unknown command")


if __name__ == "__main__":
    main()
