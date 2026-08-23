"""
xAI (Grok) text backend.

OpenAI-compatible Chat Completions at https://api.x.ai/v1
Auth: XAI_API_KEY, config xai_api_key, or an api_key that starts with "xai-".
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Generator, List, Optional, Union

import httpx

from .base import GenerationResult, make_metadata
from .utils import detect_hardware

DEFAULT_XAI_MODELS = [
    "grok-4.6",
    "grok-4.5",
    "grok-4",
    "grok-3",
    "grok-3-mini",
]


def resolve_xai_api_key(config: Dict[str, Any]) -> str:
    direct = (config.get("xai_api_key") or "").strip()
    if direct:
        return direct
    env = (os.environ.get("XAI_API_KEY") or "").strip()
    if env:
        return env
    shared = (config.get("api_key") or "").strip()
    if shared.startswith("xai-"):
        return shared
    return ""


class XAITextGenerator:
    """xAI Grok client using Chat Completions."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider = "xai"
        self.use_cloud = True
        self.hardware = detect_hardware()
        self.base_url = (config.get("xai_base_url") or "https://api.x.ai/v1").rstrip("/")
        self.api_key = resolve_xai_api_key(config)
        if not self.api_key:
            print("[Text] WARNING: xAI selected but no key found (xai_api_key, XAI_API_KEY, or api_key starting with xai-)")

        configured_default = config.get("xai_default_model") or config.get("default_model") or "grok-4.6"
        self.default_model = configured_default
        if self._looks_like_ollama_model(self.default_model):
            self.default_model = "grok-4.6"
        self.enhancer_model = config.get("xai_enhancer_model") or self.default_model
        if self._looks_like_ollama_model(self.enhancer_model):
            self.enhancer_model = self.default_model
        self.timeout = float(config.get("xai_timeout") or config.get("timeout") or 300)
        self.fallback_models = list(config.get("xai_models") or DEFAULT_XAI_MODELS)
        self.client = httpx.Client(timeout=self.timeout)
        self._models_cache: List[str] = []

    @staticmethod
    def _looks_like_ollama_model(name: str) -> bool:
        n = (name or "").lower()
        return ":" in n or n.startswith("llama") or n.startswith("phi") or n.startswith("qwen") or n.startswith("gemma") or n.startswith("mistral")

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

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
        model = model or self.default_model
        if self._looks_like_ollama_model(model):
            model = self.default_model
        start = time.time()

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": int(max_tokens),
            "stream": stream,
        }
        if seed is not None:
            payload["seed"] = int(seed)

        url = f"{self.base_url}/chat/completions"
        try:
            if stream:
                return self._stream_generate(url, payload, model, prompt, start)
            resp = self.client.post(url, json=payload, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()
            content = (
                (data.get("choices") or [{}])[0]
                .get("message", {})
                .get("content", "")
            ) or ""
            duration_ms = (time.time() - start) * 1000
            return GenerationResult(
                modality="text",
                output=content,
                prompt=prompt,
                model_id=model,
                parameters={
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "system_prompt": system_prompt,
                    "provider": "xai",
                },
                seed=seed,
                duration_ms=duration_ms,
                metadata=make_metadata(
                    {
                        "provider": "xai",
                        "finish_reason": (data.get("choices") or [{}])[0].get("finish_reason"),
                        "usage": data.get("usage") or {},
                    },
                    self.hardware,
                ),
            )
        except httpx.ConnectError:
            return self._error_result(prompt, model, f"xAI not reachable at {self.base_url}")
        except httpx.HTTPStatusError as e:
            detail = e.response.text[:400] if e.response is not None else str(e)
            return self._error_result(prompt, model, f"xAI HTTP {e.response.status_code}: {detail}")
        except Exception as e:
            return self._error_result(prompt, model, str(e))

    def _stream_generate(
        self,
        url: str,
        payload: Dict[str, Any],
        model: str,
        original_prompt: str,
        start_time: float,
    ) -> Generator[str, None, None]:
        full_text = ""
        try:
            with self.client.stream("POST", url, json=payload, headers=self._headers()) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if line == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    delta = (chunk.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
                    if delta:
                        full_text += delta
                        yield delta
        except Exception as e:
            yield f"\n[ERROR] {e}"
            return
        duration_ms = (time.time() - start_time) * 1000
        yield "\n__LUMINA_META__" + json.dumps(
            {
                "__meta__": True,
                "model": model,
                "prompt": original_prompt,
                "duration_ms": duration_ms,
                "full_text": full_text,
                "provider": "xai",
            }
        )

    def enhance_prompt(
        self,
        prompt: str,
        style: str = "detailed cinematic",
        target_modality: str = "image",
    ) -> str:
        if not prompt or len(prompt.strip()) < 3:
            return prompt
        modality_instructions = {
            "image": "highly detailed, professional photography or digital art description with lighting, composition, mood, and style",
            "video": "cinematic video prompt with camera motion, pacing, scene description, and visual storytelling",
            "voice": "expressive narration script suitable for high-quality voice synthesis",
            "text": "well-structured, vivid, and actionable creative writing prompt",
        }
        target_desc = modality_instructions.get(target_modality, "detailed creative prompt")
        system = (
            "You are an expert prompt engineer for generative AI. "
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
                model=self.enhancer_model,
                stream=False,
            )
            if isinstance(result, GenerationResult) and not result.error:
                improved = (result.output or "").strip()
                return improved if len(improved) > 10 else prompt
            return prompt
        except Exception:
            return prompt

    def list_models(self) -> List[str]:
        try:
            resp = self.client.get(f"{self.base_url}/models", headers=self._headers(), timeout=8)
            resp.raise_for_status()
            data = resp.json()
            ids = []
            for item in data.get("data") or []:
                mid = item.get("id")
                if isinstance(mid, str) and mid:
                    ids.append(mid)
            grok = [m for m in ids if m.lower().startswith("grok")]
            models = grok or ids or list(self.fallback_models)
            self._models_cache = models
            return models
        except Exception:
            return self._models_cache or list(self.fallback_models)

    def pull_model(self, model_name: str) -> Dict[str, Any]:
        return {
            "success": False,
            "error": "Model pulling is not applicable for xAI. Pick a Grok model from the list.",
        }

    def delete_model(self, model_name: str) -> bool:
        return False

    def validate(self) -> bool:
        if not self.api_key:
            return False
        try:
            r = self.client.get(f"{self.base_url}/models", headers=self._headers(), timeout=8)
            return r.status_code == 200
        except Exception:
            return False

    def _error_result(self, prompt: str, model: str, error: str) -> GenerationResult:
        return GenerationResult(
            modality="text",
            output=f"[Error] {error}",
            prompt=prompt,
            model_id=model,
            error=error,
            metadata=make_metadata({"error_type": "xai", "provider": "xai"}, self.hardware),
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "connected": self.validate(),
            "provider": "xai",
            "default_model": self.default_model,
            "enhancer_model": self.enhancer_model,
            "base_url": self.base_url,
            "use_cloud": True,
            "available_models": self.list_models(),
            "hardware": self.hardware,
            "has_api_key": bool(self.api_key),
        }
