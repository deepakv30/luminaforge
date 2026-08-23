"""
Basic smoke tests for LuminaForge generators.
Run with: pytest
"""

import pytest
from generators.base import GenerationResult
from generators.utils import load_config, detect_hardware


def test_config_loads():
    cfg = load_config("config.yaml")
    assert "text" in cfg
    assert "image" in cfg


def test_hardware_detect():
    hw = detect_hardware()
    assert "device" in hw
    assert "cuda_available" in hw


def test_generation_result_serialization():
    res = GenerationResult(
        modality="text",
        output="Hello world",
        prompt="hi",
        model_id="test",
    )
    d = res.to_dict()
    assert d["modality"] == "text"
    assert "prompt" in d


def test_resolve_text_provider_detects_xai_key():
    from generators import resolve_text_provider

    assert resolve_text_provider({"provider": "xai"}) == "xai"
    assert resolve_text_provider({"provider": "ollama", "use_cloud": False}) == "ollama"
    assert resolve_text_provider({"provider": "ollama", "use_cloud": True, "api_key": "ollama-cloud-key"}) == "ollama_cloud"
    assert resolve_text_provider({"provider": "ollama", "use_cloud": True, "api_key": "xai-not-a-real-key"}) == "xai"


def test_xai_generator_parses_chat_completion():
    from generators.xai_text import XAITextGenerator

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "hello from grok"}, "finish_reason": "stop"}]}

    class FakeClient:
        def post(self, *args, **kwargs):
            return FakeResp()

        def get(self, *args, **kwargs):
            r = FakeResp()
            r.json = lambda: {"data": [{"id": "grok-4.6"}, {"id": "grok-4.5"}]}
            return r

    gen = XAITextGenerator({"xai_api_key": "xai-test", "xai_default_model": "grok-4.6"})
    gen.client = FakeClient()
    res = gen.generate("hi")
    assert res.error is None
    assert res.output == "hello from grok"
    assert res.model_id == "grok-4.6"
    assert gen.list_models() == ["grok-4.6", "grok-4.5"]


def test_get_text_generator_selects_xai():
    from generators import get_text_generator
    from generators.xai_text import XAITextGenerator

    gen = get_text_generator({"provider": "xai", "xai_api_key": "xai-test", "xai_default_model": "grok-4.6"})
    assert isinstance(gen, XAITextGenerator)
    assert gen.default_model == "grok-4.6"


def test_prompt_library_exists():
    from generators.utils import load_prompt_library
    lib = load_prompt_library("data/prompt_library.json")
    assert isinstance(lib, list)
    assert len(lib) >= 5