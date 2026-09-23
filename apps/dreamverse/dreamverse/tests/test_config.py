from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SERVER_DIR = Path(__file__).resolve().parents[1]


def _load_config_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "server_config_test_module",
        SERVER_DIR / "config.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _set_required_prompt_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CEREBRAS_API_KEY", "cerebras-key")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")


def test_config_allows_missing_cerebras_api_key_until_prompt_runtime(monkeypatch):
    monkeypatch.delenv("FASTVIDEO_PROMPT_PROVIDER", raising=False)
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    module = _load_config_module()

    assert module.PROMPT_API_KEYS["cerebras"] is None


def test_config_allows_missing_groq_api_key_until_prompt_runtime(monkeypatch):
    monkeypatch.delenv("FASTVIDEO_PROMPT_PROVIDER", raising=False)
    monkeypatch.setenv("CEREBRAS_API_KEY", "cerebras-key")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    module = _load_config_module()

    assert module.PROMPT_API_KEYS["groq"] is None


def test_config_defaults_to_cerebras_with_parallel_groq_fallback_stage(monkeypatch):
    monkeypatch.delenv("FASTVIDEO_PROMPT_PROVIDER", raising=False)
    _set_required_prompt_keys(monkeypatch)

    module = _load_config_module()

    assert module.PROMPT_PROVIDER == "cerebras"
    assert module.PROMPT_PROVIDER_RUNTIME_STAGES == (("cerebras", "groq"), )
    assert module.PROMPT_PROVIDER_PRIORITY == (
        "cerebras",
        "groq",
    )
    assert module.PROMPT_API_KEY == "cerebras-key"
    assert module.PROMPT_API_BASE_URL is None
    assert module.PROMPT_API_KEYS == {
        "cerebras": "cerebras-key",
        "groq": "groq-key",
    }
    assert module.PROMPT_API_BASE_URLS == {
        "cerebras": None,
        "groq": "https://api.groq.com/openai/v1",
    }
    assert module.PROMPT_MODEL == "gpt-oss-120b"
    assert module.PROMPT_REWRITE_MODEL == "gpt-oss-120b"
    assert module.PROMPT_REWRITE_MODEL_OPTIONS == ["gpt-oss-120b"]
    assert module.PROMPT_PROVIDER_MODELS == {
        "cerebras": "gpt-oss-120b",
        "groq": "openai/gpt-oss-120b",
    }


def test_config_ignores_legacy_groq_primary_override(monkeypatch):
    monkeypatch.setenv("FASTVIDEO_PROMPT_PROVIDER", "groq")
    _set_required_prompt_keys(monkeypatch)

    module = _load_config_module()

    assert module.PROMPT_PROVIDER == "cerebras"
    assert module.PROMPT_PROVIDER_RUNTIME_STAGES == (("cerebras", "groq"), )
    assert module.PROMPT_PROVIDER_PRIORITY == (
        "cerebras",
        "groq",
    )
    assert module.PROMPT_API_KEY == "cerebras-key"
    assert module.PROMPT_API_BASE_URL is None


def test_config_uses_local_overlay_paths_when_devtools_enabled(monkeypatch, tmp_path):
    _set_required_prompt_keys(monkeypatch)
    monkeypatch.setenv("FASTVIDEO_ENABLE_DEVTOOLS", "true")
    monkeypatch.setenv("FASTVIDEO_DREAMVERSE_HOME", str(tmp_path / "dreamverse-state"))

    module = _load_config_module()

    assert module.DEVTOOLS_ENABLED is True
    assert module.FRONTEND_ROOT.as_posix().endswith("apps/dreamverse/web")
    assert module.PROMPT_ENHANCE_SYSTEM_PROMPT_PATH.endswith("dreamverse/prompts.local/next_segment_system_prompt.md")
    assert module.PROMPT_ENHANCE_SYSTEM_PROMPT_FALLBACK_PATH.endswith(
        "dreamverse/prompts/next_segment_system_prompt.md")
    assert module.PROMPT_REWRITE_USER_SYSTEM_PROMPT_PATH.endswith(
        "dreamverse/prompts.local/rewrite_user_system_prompt.md")
    assert module.PROMPT_REWRITE_USER_SYSTEM_PROMPT_FALLBACK_PATH.endswith(
        "dreamverse/prompts/rewrite_user_system_prompt.md")
    assert module.CURATED_PRESETS_FILE_PATH.endswith(
        "apps/dreamverse/web/prompts.local/selected_ltx2_continuation_story_presets.json")
    assert module.CURATED_PRESETS_FALLBACK_FILE_PATH.endswith(
        "apps/dreamverse/web/prompts/selected_ltx2_continuation_story_presets.json")
    assert module.FRONTEND_STATIC_DIR_CANDIDATES[:2] == (
        str(module.FRONTEND_ROOT / "out"),
        str(module.FRONTEND_ROOT / "dist"),
    )


def test_config_includes_docker_checkout_static_frontend_candidates(monkeypatch):
    _set_required_prompt_keys(monkeypatch)

    module = _load_config_module()

    assert "/opt/FastVideo/apps/dreamverse/web/out" in module.FRONTEND_STATIC_DIR_CANDIDATES
    assert "/opt/FastVideo/apps/dreamverse/web/dist" in module.FRONTEND_STATIC_DIR_CANDIDATES


def test_config_enables_prompt_safety_when_requested(monkeypatch):
    _set_required_prompt_keys(monkeypatch)
    monkeypatch.setenv("FASTVIDEO_ENABLE_PROMPT_SAFETY", "true")

    module = _load_config_module()

    assert module.PROMPT_SAFETY_ENABLED is True


def test_config_uses_five_minute_session_timeout(monkeypatch):
    _set_required_prompt_keys(monkeypatch)
    monkeypatch.delenv("DREAMVERSE_MODEL_ID", raising=False)
    monkeypatch.delenv("FASTVIDEO_SESSION_TIMEOUT_SECONDS", raising=False)

    module = _load_config_module()

    assert module.SESSION_TIMEOUT_SECONDS == 300


def test_config_uses_thirty_minute_cosmos25_session_timeout(monkeypatch):
    _set_required_prompt_keys(monkeypatch)
    monkeypatch.setenv("DREAMVERSE_MODEL_ID", "cosmos25-dfd")
    monkeypatch.delenv("FASTVIDEO_SESSION_TIMEOUT_SECONDS", raising=False)

    module = _load_config_module()

    assert module.SESSION_TIMEOUT_SECONDS == 1800


def test_config_allows_session_timeout_override(monkeypatch):
    _set_required_prompt_keys(monkeypatch)
    monkeypatch.setenv("DREAMVERSE_MODEL_ID", "cosmos25-dfd")
    monkeypatch.setenv("FASTVIDEO_SESSION_TIMEOUT_SECONDS", "900")

    module = _load_config_module()

    assert module.SESSION_TIMEOUT_SECONDS == 900


def test_config_rejects_invalid_prompt_provider(monkeypatch):
    monkeypatch.setenv("FASTVIDEO_PROMPT_PROVIDER", "unsupported")
    _set_required_prompt_keys(monkeypatch)

    with pytest.raises(RuntimeError, match="Invalid FASTVIDEO_PROMPT_PROVIDER"):
        _load_config_module()


def test_config_registers_vsa_datafree_fasth3_profile(monkeypatch):
    """The FastH3 registry entry owns the complete fixed Preview recipe."""
    _set_required_prompt_keys(monkeypatch)

    module = _load_config_module()

    assert module.MODEL_REGISTRY["fast-h3"] == {
        "name": "FastH3",
        "generation_backend": "minimax_h3",
        "default_sp_size": 4,
        "model_path": "MiniMaxAI/MiniMax-H3",
        "adapter_repo": "FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA",
        "adapter_filename": "vsa-datafree/adapter_model.safetensors",
        "attention_backend": "VIDEO_SPARSE_ATTN_H3",
        "height": 768,
        "width": 1344,
        "num_frames": 124,
        "num_inference_steps": 5,
        "seed": 1000,
    }


def test_config_uses_fasth3_sequence_parallel_default(monkeypatch):
    """Selecting FastH3 defaults DreamVerse to its four-GPU topology."""
    _set_required_prompt_keys(monkeypatch)
    monkeypatch.setenv("DREAMVERSE_MODEL_ID", "fast-h3")
    monkeypatch.delenv("DREAMVERSE_SP_SIZE", raising=False)

    module = _load_config_module()

    assert module.ACTIVE_MODEL_ID == "fast-h3"
    assert module.MODEL_CONFIG["generation_backend"] == "minimax_h3"
    assert module.DREAMVERSE_SP_SIZE == 4


def test_config_registers_cosmos25_dfd_profile(monkeypatch):
    _set_required_prompt_keys(monkeypatch)

    module = _load_config_module()

    assert module.MODEL_REGISTRY["cosmos25-dfd"] == {
        "name": "Cosmos Predict2.5 DFD",
        "generation_backend": "cosmos25_dfd",
        "default_sp_size": 1,
        "model_path": "FastVideo/Cosmos-Predict2.5-2B-Distilled-TrigFlow",
        "continuation_model_path": "FastVideo/Cosmos-Predict2.5-2B-DFD",
        "attention_backend": "TORCH_SDPA",
        "height": 704,
        "width": 1280,
        "bootstrap_num_frames": 77,
        "continuation_num_frames": 81,
        "fps": 24,
        "num_inference_steps": 4,
        "seed": 42,
        "session_timeout_seconds": 1800,
    }


def test_config_selects_cosmos25_package_roles(monkeypatch, tmp_path):
    _set_required_prompt_keys(monkeypatch)
    bootstrap_path = tmp_path / "cosmos25-t2w"
    continuation_path = tmp_path / "cosmos25-dfd"
    monkeypatch.setenv("DREAMVERSE_MODEL_ID", "cosmos25-dfd")
    monkeypatch.setenv("DREAMVERSE_MODEL_PATH", str(bootstrap_path))
    monkeypatch.setenv("DREAMVERSE_COSMOS25_DFD_MODEL_PATH", str(continuation_path))
    monkeypatch.delenv("DREAMVERSE_SP_SIZE", raising=False)

    module = _load_config_module()

    assert module.ACTIVE_MODEL_ID == "cosmos25-dfd"
    assert module.MODEL_CONFIG["generation_backend"] == "cosmos25_dfd"
    assert module.MODEL_CONFIG["model_path"] == str(bootstrap_path)
    assert module.MODEL_CONFIG["continuation_model_path"] == str(continuation_path)
    assert module.DREAMVERSE_SP_SIZE == 1
