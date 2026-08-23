"""
LuminaForge Text Generator (Ollama-first)
=========================================

Core design:
- Primary target: local Ollama server at localhost:11434 (zero config for most users).
- Streaming support for Gradio chat / generation.
- First-class "Enhance Prompt" using structured reasoning from a small fast model.
- Clean hand-off metadata so downstream generators (image/video) can receive
  beautifully expanded prompts without any copy-paste.
- Optional (opt-in) Ollama Cloud fallback only when user explicitly enables it.

The enhance_prompt method is a major productivity win. It turns vague user ideas
into rich, modality-aware prompts in <2 seconds using the local model.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Generator, Union

import httpx

from .base import GenerationResult, TextGenerator, make_metadata
from .utils import (
    detect_hardware,
    get_ollama_models,
    load_config,
    safe_json,
    truncate_for_log,
)
from .xai_text import XAITextGenerator


def resolve_text_provider(text_cfg: Optional[Dict[str, Any]] = None) -> str:
    """Return 'ollama', 'ollama_cloud', or 'xai'."""
    cfg = text_cfg or {}
    raw = str(cfg.get("provider") or "ollama").strip().lower()
    if raw in ("xai", "spacexai", "grok"):
        return "xai"
    if raw in ("ollama_cloud", "ollama-cloud"):
        return "ollama_cloud"

    key = (
        str(cfg.get("xai_api_key") or "").strip()
        or os.environ.get("XAI_API_KEY", "").strip()
        or str(cfg.get("api_key") or "").strip()
    )
    # Pasting an xAI key into the old Ollama Cloud field should still select xAI.
    if cfg.get("use_cloud") and key.startswith("xai-"):
        return "xai"
    if cfg.get("use_cloud"):
        return "ollama_cloud"
    return "ollama"


def _text_section(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if config is None:
        return dict(load_config().get("text", {}))
    if any(k in config for k in ("app", "image", "video", "voice")) and isinstance(config.get("text"), dict):
        return dict(config["text"])
    if isinstance(config.get("text"), dict) and not any(
        k in config for k in ("provider", "base_url", "use_cloud", "xai_api_key", "xai_base_url")
    ):
        return dict(config["text"])
    return dict(config)


def get_text_generator(config: Optional[Dict[str, Any]] = None):
    text_cfg = _text_section(config)
    provider = resolve_text_provider(text_cfg)
    if provider == "xai":
        return XAITextGenerator(text_cfg)
    adapted = dict(text_cfg)
    adapted["use_cloud"] = provider == "ollama_cloud"
    gen = OllamaTextGenerator(adapted)
    gen.provider = provider
    return gen


class OllamaTextGenerator:
    """Production-grade Ollama client (local or cloud) with streaming and prompt enhancement."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.use_cloud = config.get("use_cloud", False)
        self.provider = "ollama_cloud" if self.use_cloud else "ollama"
        self.hardware = detect_hardware()

        if self.use_cloud:
            self.base_url = config.get("cloud_base_url", "https://api.ollama.com").rstrip("/")
            # Support api_key in config or OLLAMA_API_KEY env var
            self.api_key = config.get("api_key") or os.environ.get("OLLAMA_API_KEY", "")
            if not self.api_key:
                print("[Text] WARNING: use_cloud=True but no API key found (config or OLLAMA_API_KEY env)")
        else:
            self.base_url = config.get("base_url", "http://localhost:11434").rstrip("/")
            self.api_key = None

        self.default_model = config.get("default_model", "llama3.2:3b")
        self.enhancer_model = config.get("enhancer_model", self.default_model)
        self.timeout = config.get("timeout", 120)

        self.client = httpx.Client(timeout=self.timeout)
        self._models_cache: List[str] = []

    def _get_headers(self) -> Dict[str, str]:
        """Return auth headers for cloud requests."""
        if self.use_cloud and self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    # ------------------------- Core Generation -------------------------

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        model: Optional[str] = None,
        seed: Optional[int] = None,
        stream: bool = False,
    ) -> Union[GenerationResult, Generator[str, None, None]]:
        """
        Generate text. When stream=True returns a generator of text chunks
        (ideal for Gradio streaming). Otherwise returns GenerationResult.
        """
        model = model or self.default_model
        start = time.time()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        }
        if seed is not None:
            payload["options"]["seed"] = seed

        url = f"{self.base_url}/api/chat"
        headers = self._get_headers()

        try:
            if stream:
                return self._stream_generate(url, payload, model, prompt, start, headers)

            resp = self.client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            content = data.get("message", {}).get("content", "")
            done = data.get("done", True)

            duration_ms = (time.time() - start) * 1000

            result = GenerationResult(
                modality="text",
                output=content,
                prompt=prompt,
                model_id=model,
                parameters={
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "system_prompt": system_prompt,
                },
                seed=seed,
                duration_ms=duration_ms,
                metadata=make_metadata({
                    "ollama_done": done,
                    "raw_response_keys": list(data.keys()),
                }, self.hardware),
            )
            return result

        except httpx.ConnectError:
            service = "Ollama Cloud" if self.use_cloud else "local Ollama"
            return self._error_result(prompt, model, f"{service} not reachable at " + self.base_url)
        except Exception as e:
            return self._error_result(prompt, model, str(e))

    def _stream_generate(
        self,
        url: str,
        payload: Dict,
        model: str,
        original_prompt: str,
        start_time: float,
        headers: Dict[str, str] = None,
    ) -> Generator[str, None, None]:
        """Yields text chunks. Final chunk is special marker with metadata."""
        full_text = ""
        headers = headers or {}
        try:
            with self.client.stream("POST", url, json=payload, headers=headers) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        if "message" in chunk and "content" in chunk["message"]:
                            delta = chunk["message"]["content"]
                            full_text += delta
                            yield delta
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            yield f"\n[ERROR] {e}"
            return

        # After stream finishes, we can't return a full result easily in generator form.
        # Gradio streaming components usually just need the text.
        # For history saving, callers should reconstruct a result from accumulated text.
        # We yield a final JSON metadata line (prefixed) so the caller can parse it.
        duration_ms = (time.time() - start_time) * 1000
        meta = {
            "__meta__": True,
            "model": model,
            "prompt": original_prompt,
            "duration_ms": duration_ms,
            "full_text": full_text,
        }
        yield "\n__LUMINA_META__" + json.dumps(meta)

    # ------------------------- Prompt Enhancer (Killer Feature) -------------------------

    def enhance_prompt(
        self,
        prompt: str,
        style: str = "detailed cinematic",
        target_modality: str = "image",
    ) -> str:
        """
        Use a fast local model to dramatically improve the user's prompt.

        The enhancer is instructed to output ONLY the improved prompt (no yapping).
        This is the secret sauce for non-expert users to get pro results instantly.
        """
        if not prompt or len(prompt.strip()) < 3:
            return prompt

        model = self.enhancer_model

        modality_instructions = {
            "image": "highly detailed, professional photography or digital art description with lighting, composition, mood, and style",
            "video": "cinematic video prompt with camera motion, pacing, scene description, and visual storytelling",
            "voice": "expressive narration script suitable for high-quality voice synthesis",
            "text": "well-structured, vivid, and actionable creative writing prompt",
        }
        target_desc = modality_instructions.get(target_modality, "detailed creative prompt")

        system = (
            "You are an expert prompt engineer for local open-source generative AI. "
            "Rewrite the user's idea into a single, rich, optimized prompt for "
            f"{target_modality} generation. Focus on {target_desc}. "
            "Be specific about subject, environment, lighting, style, quality. "
            "Keep under 180 words. Output ONLY the improved prompt text. No explanations."
        )

        user = f"Original idea: {prompt}\nStyle guidance: {style}\n\nImproved prompt:"

        try:
            result = self.generate(
                prompt=user,
                system_prompt=system,
                temperature=0.6,
                max_tokens=280,
                model=model,
                stream=False,
            )
            if isinstance(result, GenerationResult) and not result.error:
                improved = result.output.strip()
                # Safety: never return completely empty
                return improved if len(improved) > 10 else prompt
            return prompt
        except Exception:
            return prompt  # graceful degradation

    # ------------------------- Model Management -------------------------

    def list_models(self) -> List[str]:
        try:
            headers = self._get_headers()
            models = get_ollama_models(self.base_url, headers=headers)
            self._models_cache = models
            return models
        except Exception:
            return self._models_cache or [self.default_model]

    def pull_model(self, model_name: str) -> Dict[str, Any]:
        """Trigger pull via Ollama API. Returns progress info.
        Note: Cloud mode does not support pulling models the same way.
        """
        if self.use_cloud:
            return {
                "success": False,
                "error": "Model pulling is not supported when using Ollama Cloud. Models are hosted."
            }
        url = f"{self.base_url}/api/pull"
        headers = self._get_headers()
        try:
            resp = self.client.post(url, json={"name": model_name}, headers=headers, timeout=600)
            resp.raise_for_status()
            return {"success": True, "data": resp.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def delete_model(self, model_name: str) -> bool:
        if self.use_cloud:
            return False  # Not applicable for cloud
        try:
            headers = self._get_headers()
            resp = self.client.delete(f"{self.base_url}/api/delete", json={"name": model_name}, headers=headers)
            return resp.status_code == 200
        except Exception:
            return False

    def validate(self) -> bool:
        try:
            headers = self._get_headers()
            r = self.client.get(f"{self.base_url}/api/tags", headers=headers, timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    # ------------------------- Helpers -------------------------

    def _error_result(self, prompt: str, model: str, error: str) -> GenerationResult:
        return GenerationResult(
            modality="text",
            output=f"[Error] {error}",
            prompt=prompt,
            model_id=model,
            error=error,
            metadata=make_metadata({"error_type": "ollama"}, self.hardware),
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "connected": self.validate(),
            "provider": self.provider,
            "default_model": self.default_model,
            "enhancer_model": self.enhancer_model,
            "base_url": self.base_url,
            "use_cloud": self.use_cloud,
            "available_models": self.list_models(),
            "hardware": self.hardware,
        }