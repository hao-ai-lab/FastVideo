# pyright: reportMissingTypeArgument=false, reportArgumentType=false, reportOptionalSubscript=false, reportOperatorIssue=false
from __future__ import annotations

import asyncio
import os
import re
import time


os.environ.setdefault("CEREBRAS_API_KEY", "dummy")
os.environ.setdefault("GROQ_API_KEY", "dummy")

import dreamverse.prompt_enhancer as prompt_enhancer_module
from dreamverse.prompt_enhancer import (
    ProviderRuntime,
    PromptEnhancer,
    _build_prompt_backup_path,
    _load_prompt_required,
    _parse_json_response,
    _resolve_prompt_save_path,
)


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def model_dump(self, mode: str = "json"):  # noqa: ARG002
        return self._payload


class _FakeSyncCompletions:
    def __init__(self, payload: dict):
        self._payload = payload

    def create(self, **kwargs):  # noqa: ARG002
        return _FakeResponse(self._payload)


class _FakeSyncClient:
    def __init__(self, payload: dict):
        self.chat = type(
            "_FakeChat",
            (),
            {"completions": _FakeSyncCompletions(payload)},
        )()


class _DelayedSyncCompletions:
    def __init__(self, payload: dict, delay_s: float = 0.0, exc: Exception | None = None):
        self._payload = payload
        self._delay_s = delay_s
        self._exc = exc

    def create(self, **kwargs):  # noqa: ARG002
        if self._delay_s > 0:
            time.sleep(self._delay_s)
        if self._exc is not None:
            raise self._exc
        return _FakeResponse(self._payload)


class _DelayedSyncClient:
    def __init__(self, payload: dict, delay_s: float = 0.0, exc: Exception | None = None):
        self.chat = type(
            "_FakeChat",
            (),
            {
                "completions": _DelayedSyncCompletions(
                    payload,
                    delay_s=delay_s,
                    exc=exc,
                )
            },
        )()


def _chat_payload_with_content(content: str) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": content,
                }
            }
        ]
    }


def _build_test_enhancer(payload: dict) -> PromptEnhancer:
    enhancer = PromptEnhancer.__new__(PromptEnhancer)
    enhancer.provider = "cerebras"
    enhancer.provider_label = "cerebras"
    enhancer.model = "gpt-test"
    enhancer.rewrite_default_model = "gpt-test"
    enhancer.rewrite_model_options = ["gpt-test"]
    enhancer.default_timeout_ms = 1200
    enhancer.http_timeout_ms = 1200
    enhancer.temperature = 0.4
    enhancer.rewrite_default_temperature = 0.4
    enhancer.max_completion_tokens = 512
    enhancer.enhance_system_prompt_fallback_path = None
    enhancer.auto_system_prompt_fallback_path = None
    enhancer.rewrite_all_system_prompt_fallback_path = None
    enhancer.rewrite_user_system_prompt_path = "/tmp/rewrite_user_system_prompt.md"
    enhancer.rewrite_user_system_prompt_fallback_path = None
    enhancer.rewrite_all_system_prompt = "system prompt"
    enhancer.rewrite_user_system_prompt = "system prompt"
    enhancer.provider_request_models = {
        "cerebras": "gpt-test",
        "groq": "openai/gpt-test",
    }
    enhancer.provider_success_counts = {
        "cerebras": 0,
        "groq": 0,
    }
    enhancer.client = _FakeSyncClient(payload)
    return enhancer


def _build_staged_enhancer(
    *,
    cerebras_payload: dict,
    groq_payload: dict,
    cerebras_request_model: str = "gpt-test",
    groq_request_model: str = "openai/gpt-test",
    cerebras_delay_s: float = 0.0,
    groq_delay_s: float = 0.0,
    cerebras_exc: Exception | None = None,
    groq_exc: Exception | None = None,
) -> PromptEnhancer:
    enhancer = _build_test_enhancer(cerebras_payload)
    enhancer.provider_request_models = {
        "cerebras": cerebras_request_model,
        "groq": groq_request_model,
    }
    enhancer.enhance_system_prompt = "enhance prompt"
    enhancer.auto_system_prompt = "auto prompt"
    enhancer.provider_runtimes = [
        ProviderRuntime(
            name="cerebras",
            api_key="cerebras-key",
            api_base_url=None,
            request_model=cerebras_request_model,
            client=_DelayedSyncClient(
                cerebras_payload,
                delay_s=cerebras_delay_s,
                exc=cerebras_exc,
            ),
        ),
        ProviderRuntime(
            name="groq",
            api_key="groq-key",
            api_base_url="https://api.groq.com/openai/v1",
            request_model=groq_request_model,
            client=_DelayedSyncClient(
                groq_payload,
                delay_s=groq_delay_s,
                exc=groq_exc,
            ),
        ),
    ]
    enhancer.provider_runtime_stages = [
        list(enhancer.provider_runtimes),
    ]
    enhancer.provider_success_counts = {
        "cerebras": 0,
        "groq": 0,
    }
    enhancer.client = enhancer.provider_runtimes[0].client
    return enhancer


class _FakeOpenAIClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.chat = type(
            "_FakeChat",
            (),
            {"completions": _FakeSyncCompletions(_chat_payload_with_content("{}"))},
        )()


class _FakeCerebrasClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.chat = type(
            "_FakeChat",
            (),
            {"completions": _FakeSyncCompletions(_chat_payload_with_content("{}"))},
        )()


def test_parse_json_response_accepts_fenced_json_with_prose():
    parsed = _parse_json_response(
        "Here is the rewrite:\n```json\n{\"segment_prompts\":[\"A\",\"B\"]}\n```\nThanks."
    )
    assert parsed == {"segment_prompts": ["A", "B"]}


def test_parse_json_response_extracts_first_embedded_object():
    parsed = _parse_json_response(
        "Model output:\n{\"segment_prompts\":[\"A\",\"B\"]}\n(complete)"
    )
    assert parsed == {"segment_prompts": ["A", "B"]}


def test_load_prompt_required_falls_back_to_default_path(tmp_path):
    fallback_path = tmp_path / "next_segment_system_prompt.md"
    fallback_path.write_text("fallback prompt\n", encoding="utf-8")

    loaded_prompt = _load_prompt_required(
        str(tmp_path / "prompts.local" / "next_segment_system_prompt.md"),
        "next-segment",
        str(fallback_path),
    )

    assert loaded_prompt == "fallback prompt"


def test_load_prompt_required_unwraps_python_style_system_prompt_assignment(tmp_path):
    prompt_path = tmp_path / "rewrite_window_system_prompt.md"
    prompt_path.write_text(
        'SYSTEM_PROMPT = """\nfirst line\nsecond line\n"""\n',
        encoding="utf-8",
    )

    loaded_prompt = _load_prompt_required(
        str(prompt_path),
        "rewrite-window",
    )

    assert loaded_prompt == "first line\nsecond line"


def test_build_client_supports_cerebras_provider(monkeypatch):
    monkeypatch.setattr(prompt_enhancer_module, "Cerebras", _FakeCerebrasClient)

    enhancer = PromptEnhancer.__new__(PromptEnhancer)
    enhancer.provider = "cerebras"
    enhancer.api_key = "cerebras-key"
    enhancer.api_base_url = None

    client = PromptEnhancer._build_client(enhancer)

    assert isinstance(client, _FakeCerebrasClient)
    assert client.kwargs == {
        "api_key": "cerebras-key",
    }


def test_build_client_supports_groq_provider(monkeypatch):
    monkeypatch.setattr(prompt_enhancer_module, "OpenAI", _FakeOpenAIClient)

    enhancer = PromptEnhancer.__new__(PromptEnhancer)
    enhancer.provider = "groq"
    enhancer.api_key = "groq-key"
    enhancer.api_base_url = "https://api.groq.com/openai/v1"

    client = PromptEnhancer._build_client(enhancer)

    assert isinstance(client, _FakeOpenAIClient)
    assert client.kwargs == {
        "api_key": "groq-key",
        "base_url": "https://api.groq.com/openai/v1",
    }


def test_rewrite_prompt_sequence_accepts_segment_prompts_output():
    enhancer = _build_test_enhancer(
        _chat_payload_with_content(
            '{"id":"preset_a","label":"Preset A","segment_prompts":["A","B"]}'
        )
    )
    result = asyncio.run(
        enhancer.rewrite_prompt_sequence(
            ["prompt one", "prompt two"],
            rewrite_instruction="make it cinematic",
        )
    )
    assert result.fallback_used is False
    assert result.error is None
    assert result.rollout_id == "preset_a"
    assert result.rollout_label == "Preset A"
    assert result.prompts == ["A", "B"]


def test_rewrite_prompt_sequence_accepts_legacy_rewritten_prompts_output():
    enhancer = _build_test_enhancer(
        _chat_payload_with_content('{"rewritten_prompts":["A","B"]}')
    )
    result = asyncio.run(
        enhancer.rewrite_prompt_sequence(
            ["prompt one", "prompt two"],
            rewrite_instruction="make it cinematic",
        )
    )
    assert result.fallback_used is False
    assert result.error is None
    assert result.rollout_id == "current_rollout"
    assert result.rollout_label == "Current rollout"
    assert result.prompts == ["A", "B"]


def test_rewrite_prompt_sequence_accepts_segment_dicts_without_top_level_rollout_metadata():
    enhancer = _build_test_enhancer(
        _chat_payload_with_content(
            '{"segments":[{"prompt":"A"},{"text":"B"}]}'
        )
    )
    result = asyncio.run(
        enhancer.rewrite_prompt_sequence(
            ["prompt one", "prompt two"],
            preset_id="preset_a",
            preset_label="Preset A",
            rewrite_instruction="make it cinematic",
        )
    )
    assert result.fallback_used is False
    assert result.error is None
    assert result.rollout_id == "preset_a"
    assert result.rollout_label == "Preset A"
    assert result.prompts == ["A", "B"]


def test_rewrite_prompt_sequence_accepts_numbered_prose_output():
    enhancer = _build_test_enhancer(
        _chat_payload_with_content(
            "The user is asking for a cinematic rewrite.\n\n"
            "1. A dog bounds across the moon's dusty surface, kicking up silver regolith as it chases a rabbit beneath the black sky.\n"
            "2. The rabbit darts around a crater rim while the dog lunges after it, Earth glowing blue in the distance.\n"
        )
    )
    result = asyncio.run(
        enhancer.rewrite_prompt_sequence(
            ["prompt one", "prompt two"],
            rewrite_instruction="make it cinematic",
        )
    )
    assert result.fallback_used is False
    assert result.error is None
    assert result.rollout_id == "current_rollout"
    assert result.rollout_label == "Current rollout"
    assert result.prompts == [
        "A dog bounds across the moon's dusty surface, kicking up silver regolith as it chases a rabbit beneath the black sky.",
        "The rabbit darts around a crater rim while the dog lunges after it, Earth glowing blue in the distance.",
    ]


def test_enhance_prompt_prefers_cerebras_before_groq_fallback():
    enhancer = _build_staged_enhancer(
        cerebras_payload=_chat_payload_with_content('{"prompt":"Cerebras prompt"}'),
        groq_payload=_chat_payload_with_content('{"prompt":"Groq prompt"}'),
        cerebras_delay_s=0.01,
        groq_delay_s=0.01,
    )

    result = asyncio.run(
        enhancer.enhance_prompt(
            "A rainy alley at night",
            mode="single_clip",
        )
    )

    assert result.fallback_used is False
    assert result.error is None
    assert result.provider == "cerebras"
    assert result.model == "gpt-test"
    assert result.prompt == "Cerebras prompt"
    assert enhancer.get_provider_success_counts() == {
        "cerebras": 1,
        "groq": 0,
    }


def test_enhance_prompt_uses_groq_when_cerebras_fails():
    enhancer = _build_staged_enhancer(
        cerebras_payload=_chat_payload_with_content("{}"),
        groq_payload=_chat_payload_with_content('{"prompt":"Groq prompt"}'),
        cerebras_exc=RuntimeError("cerebras overloaded"),
        groq_delay_s=0.01,
    )

    result = asyncio.run(
        enhancer.enhance_prompt(
            "A rainy alley at night",
            mode="single_clip",
        )
    )

    assert result.fallback_used is False
    assert result.error is None
    assert result.provider == "groq"
    assert result.prompt == "Groq prompt"
    assert enhancer.get_provider_success_counts() == {
        "cerebras": 0,
        "groq": 1,
    }


def test_enhance_prompt_can_use_groq_when_cerebras_times_out():
    enhancer = _build_staged_enhancer(
        cerebras_payload=_chat_payload_with_content('{"prompt":"Slow Cerebras prompt"}'),
        groq_payload=_chat_payload_with_content('{"prompt":"Groq prompt"}'),
        cerebras_delay_s=0.4,
        groq_delay_s=0.01,
    )
    enhancer.http_timeout_ms = 50
    enhancer.default_timeout_ms = 50

    result = asyncio.run(
        enhancer.enhance_prompt(
            "A rainy alley at night",
            mode="single_clip",
            timeout_ms=50,
        )
    )

    assert result.fallback_used is False
    assert result.error is None
    assert result.provider == "groq"
    assert result.prompt == "Groq prompt"
    assert enhancer.get_provider_success_counts() == {
        "cerebras": 0,
        "groq": 1,
    }


def test_enhance_prompt_can_use_cerebras_when_it_returns_first():
    enhancer = _build_staged_enhancer(
        cerebras_payload=_chat_payload_with_content('{"prompt":"Primary Cerebras prompt"}'),
        groq_payload=_chat_payload_with_content('{"prompt":"Groq prompt"}'),
        cerebras_delay_s=0.01,
        groq_delay_s=0.08,
    )

    result = asyncio.run(
        enhancer.enhance_prompt(
            "A rainy alley at night",
            mode="single_clip",
        )
    )

    assert result.fallback_used is False
    assert result.error is None
    assert result.provider == "cerebras"
    assert result.model == "gpt-test"
    assert result.prompt == "Primary Cerebras prompt"
    assert enhancer.get_provider_success_counts() == {
        "cerebras": 1,
        "groq": 0,
    }


def test_rewrite_prompt_sequence_keeps_raw_output_on_parse_error():
    enhancer = _build_test_enhancer(
        _chat_payload_with_content("I cannot comply with JSON right now.")
    )
    result = asyncio.run(
        enhancer.rewrite_prompt_sequence(
            ["prompt one", "prompt two"],
            rewrite_instruction="make it cinematic",
        )
    )
    assert result.fallback_used is True
    assert "No JSON object found in assistant response." in (result.error or "")
    assert result.raw_response_text == "I cannot comply with JSON right now."
    assert result.rollout_id == "current_rollout"
    assert result.rollout_label == "Current rollout"
    assert result.prompts == ["prompt one", "prompt two"]


def test_rewrite_prompt_sequence_uses_current_rollout_payload_shape():
    enhancer = _build_test_enhancer(
        _chat_payload_with_content(
            '{"id":"rewritten_rollout","label":"Rewritten Rollout","segment_prompts":["A","B"]}'
        )
    )
    captured = {
        "body": None,
        "timeout_seconds": None,
    }

    async def _fake_request_content_with_body(*, body: dict, timeout_seconds: float):
        captured["body"] = body
        captured["timeout_seconds"] = timeout_seconds
        return (
            _chat_payload_with_content(
                '{"id":"rewritten_rollout","label":"Rewritten Rollout","segment_prompts":["A","B"]}'
            ),
            '{"id":"rewritten_rollout","label":"Rewritten Rollout","segment_prompts":["A","B"]}',
        )

    enhancer._request_content_with_body = _fake_request_content_with_body  # type: ignore[attr-defined]

    result = asyncio.run(
        enhancer.rewrite_prompt_sequence(
            ["prompt one", "prompt two"],
            preset_id="preset_a",
            preset_label="Preset A",
            rewrite_instruction="make it cinematic",
            rewrite_model="gpt-test",
            rewrite_temperature=0.2,
            timeout_ms=800,
        )
    )

    assert result.fallback_used is False
    assert captured["body"]["messages"][0] == {
        "role": "system",
        "content": "system prompt",
    }
    assert captured["body"]["messages"][1]["role"] == "user"
    assert prompt_enhancer_module.json.loads(captured["body"]["messages"][1]["content"]) == {
        "mode": "edit_existing_rollout",
        "request": (
            "Rewrite all segment prompts with improved continuity and cinematic detail. "
            "Keep count and ordering identical."
        ),
        "user_instruction": "make it cinematic",
        "current_rollout": {
            "id": "preset_a",
            "label": "Preset A",
            "segment_prompts": ["prompt one", "prompt two"],
        },
    }


def test_rewrite_prompt_sequence_supports_new_rollout_mode():
    enhancer = _build_test_enhancer(
        _chat_payload_with_content(
            '{"id":"custom_editable","label":"Custom rollout","segment_prompts":['
            '"A","B","C","D","E","F"]}'
        )
    )
    captured = {
        "body": None,
    }

    async def _fake_request_content_with_body(*, body: dict, timeout_seconds: float):
        del timeout_seconds
        captured["body"] = body
        return (
            _chat_payload_with_content(
                '{"id":"custom_editable","label":"Custom rollout","segment_prompts":['
                '"A","B","C","D","E","F"]}'
            ),
            '{"id":"custom_editable","label":"Custom rollout","segment_prompts":['
            '"A","B","C","D","E","F"]}',
        )

    enhancer._request_content_with_body = _fake_request_content_with_body  # type: ignore[attr-defined]

    result = asyncio.run(
        enhancer.rewrite_prompt_sequence(
            [],
            preset_id="custom_editable",
            preset_label="Custom rollout",
            rewrite_instruction="A moonbase corridor thriller with flooding and red alarms",
            rewrite_model="gpt-test",
            rewrite_temperature=0.2,
            timeout_ms=800,
        )
    )

    assert result.fallback_used is False
    assert result.prompts == ["A", "B", "C", "D", "E", "F"]
    assert prompt_enhancer_module.json.loads(captured["body"]["messages"][1]["content"]) == {
        "mode": "new_rollout",
        "request": (
            "Rewrite all segment prompts with improved continuity and cinematic detail. "
            "Keep count and ordering identical."
        ),
        "user_instruction": "A moonbase corridor thriller with flooding and red alarms",
        "desired_segment_count": 6,
        "rollout_id_hint": "custom_editable",
        "rollout_label_hint": "Custom rollout",
    }


def test_rewrite_prompt_sequence_uses_session_override_system_prompt():
    enhancer = _build_test_enhancer(
        _chat_payload_with_content(
            '{"id":"preset_a","label":"Preset A","segment_prompts":["A","B"]}'
        )
    )
    enhancer.rewrite_all_system_prompt = "shared system prompt"
    captured = {
        "body": None,
    }

    async def _fake_request_content_with_body(*, body: dict, timeout_seconds: float):
        del timeout_seconds
        captured["body"] = body
        return (
            _chat_payload_with_content(
                '{"id":"preset_a","label":"Preset A","segment_prompts":["A","B"]}'
            ),
            '{"id":"preset_a","label":"Preset A","segment_prompts":["A","B"]}',
        )

    enhancer._request_content_with_body = _fake_request_content_with_body  # type: ignore[attr-defined]

    result = asyncio.run(
        enhancer.rewrite_prompt_sequence(
            ["prompt one", "prompt two"],
            preset_id="preset_a",
            preset_label="Preset A",
            rewrite_instruction="make it cinematic",
            rewrite_model="gpt-test",
            system_prompt_override="session specific system prompt",
        )
    )

    assert result.fallback_used is False
    assert captured["body"]["messages"][0] == {
        "role": "system",
        "content": "session specific system prompt",
    }


def test_resolve_rewrite_new_rollout_system_prompt_uses_dedicated_prompt():
    enhancer = _build_test_enhancer(
        _chat_payload_with_content(
            '{"id":"preset_a","label":"Preset A","segment_prompts":["A","B"]}'
        )
    )
    enhancer.rewrite_all_system_prompt = "shared rewrite system prompt"
    enhancer.rewrite_user_system_prompt = "new rollout rewrite system prompt"

    resolved = enhancer.resolve_rewrite_new_rollout_system_prompt(None)

    assert resolved == "new rollout rewrite system prompt"


def test_resolve_rewrite_new_rollout_system_prompt_prefers_override():
    enhancer = _build_test_enhancer(
        _chat_payload_with_content(
            '{"id":"preset_a","label":"Preset A","segment_prompts":["A","B"]}'
        )
    )
    enhancer.rewrite_all_system_prompt = "shared rewrite system prompt"
    enhancer.rewrite_user_system_prompt = "new rollout rewrite system prompt"

    resolved = enhancer.resolve_rewrite_new_rollout_system_prompt(
        "session specific system prompt"
    )

    assert resolved == "session specific system prompt"


def test_generate_auto_prompt_uses_selected_model():
    enhancer = _build_test_enhancer(
        _chat_payload_with_content('{"next_prompt":"Auto next"}')
    )
    enhancer.auto_system_prompt = "auto system prompt"
    enhancer.rewrite_model_options = ["gpt-test", "gpt-alt"]
    enhancer.rewrite_default_model = "gpt-test"
    captured_model = {"value": None}

    async def _fake_request_content(
        *,
        system_prompt: str,
        user_payload: dict,
        timeout_ms: int,
        model: str | None = None,
        temperature: float | None = None,
    ):
        captured_model["value"] = model
        return (
            _chat_payload_with_content('{"next_prompt":"Auto next"}'),
            '{"next_prompt":"Auto next"}',
        )

    enhancer._request_content = _fake_request_content  # type: ignore[attr-defined]

    result = asyncio.run(
        enhancer.generate_auto_prompt(
            locked_segments=["One"],
            next_segment_idx=2,
            model="gpt-alt",
            timeout_ms=800,
        )
    )
    assert result.fallback_used is False
    assert result.error is None
    assert result.prompt == "Auto next"
    assert captured_model["value"] == "gpt-alt"
    assert result.model == "gpt-alt"


def test_enhance_prompt_uses_selected_model():
    enhancer = _build_test_enhancer(
        _chat_payload_with_content('{"next_prompt":"Enhanced next"}')
    )
    enhancer.enhance_system_prompt = "enhance system prompt"
    enhancer.auto_system_prompt = "auto system prompt"
    enhancer.rewrite_model_options = ["gpt-test", "gpt-alt"]
    enhancer.rewrite_default_model = "gpt-test"
    captured_model = {"value": None}

    async def _fake_request_content(
        *,
        system_prompt: str,
        user_payload: dict,
        timeout_ms: int,
        model: str | None = None,
        temperature: float | None = None,
    ):
        captured_model["value"] = model
        return (
            _chat_payload_with_content('{"next_prompt":"Enhanced next"}'),
            '{"next_prompt":"Enhanced next"}',
        )

    enhancer._request_content = _fake_request_content  # type: ignore[attr-defined]

    result = asyncio.run(
        enhancer.enhance_prompt(
            "A user live prompt",
            locked_segments=["Segment 1"],
            next_segment_idx=2,
            model="gpt-alt",
            timeout_ms=800,
        )
    )
    assert result.fallback_used is False
    assert result.error is None
    assert result.prompt == "Enhanced next"
    assert captured_model["value"] == "gpt-alt"
    assert result.model == "gpt-alt"


def test_enhance_prompt_single_clip_uses_auto_extension_prompt_and_prompt_field():
    enhancer = _build_test_enhancer(
        _chat_payload_with_content('{"prompt":"Extended single clip"}')
    )
    enhancer.enhance_system_prompt = "enhance system prompt"
    enhancer.auto_system_prompt = "auto system prompt"
    enhancer.rewrite_model_options = ["gpt-test", "gpt-alt"]
    enhancer.rewrite_default_model = "gpt-test"
    captured = {
        "model": None,
        "system_prompt": None,
        "user_payload": None,
    }

    async def _fake_request_content(
        *,
        system_prompt: str,
        user_payload: dict,
        timeout_ms: int,
        model: str | None = None,
        temperature: float | None = None,
    ):
        del timeout_ms, temperature
        captured["model"] = model
        captured["system_prompt"] = system_prompt
        captured["user_payload"] = user_payload
        return (
            _chat_payload_with_content('{"prompt":"Extended single clip"}'),
            '{"prompt":"Extended single clip"}',
        )

    enhancer._request_content = _fake_request_content  # type: ignore[attr-defined]

    result = asyncio.run(
        enhancer.enhance_prompt(
            "short 5s idea",
            mode="single_clip",
            model="gpt-alt",
            timeout_ms=800,
        )
    )
    assert result.fallback_used is False
    assert result.error is None
    assert result.prompt == "Extended single clip"
    assert result.model == "gpt-alt"
    assert captured["model"] == "gpt-alt"
    assert captured["system_prompt"] == "auto system prompt"
    assert captured["user_payload"] == {
        "request": (
            "Expand the user prompt into one detailed prompt for a "
            "single 5-second LTX-2.3 video clip. Respond with "
            'valid JSON only as {"prompt": "..."}.'  # noqa: E501
        ),
        "user_prompt": "short 5s idea",
    }


def test_enhance_prompt_single_clip_rejects_plain_text_response():
    enhancer = _build_test_enhancer(
        _chat_payload_with_content(
            "Medium shot of a woman by a rainy cafe window as she lifts her "
            "phone, exhales softly, and the camera makes a slow push in."
        )
    )
    enhancer.auto_system_prompt = "auto system prompt"

    result = asyncio.run(
        enhancer.enhance_prompt(
            "woman in a cafe looking at her phone",
            mode="single_clip",
            model="gpt-test",
            timeout_ms=800,
        )
    )
    assert result.fallback_used is True
    assert "No JSON object found in assistant response." in result.error
    assert result.prompt == ""


def test_enhance_prompt_single_clip_rejects_segment_prompts_json():
    enhancer = _build_test_enhancer(
        _chat_payload_with_content('{"segment_prompts":["A","B"]}')
    )
    enhancer.auto_system_prompt = "auto system prompt"
    enhancer.rewrite_model_options = ["gpt-test"]
    enhancer.rewrite_default_model = "gpt-test"

    result = asyncio.run(
        enhancer.enhance_prompt(
            "short 5s idea",
            mode="single_clip",
            model="gpt-test",
            timeout_ms=800,
        )
    )
    assert result.fallback_used is True
    assert result.prompt == ""
    assert "Missing prompt string." in (result.error or "")


def test_enhance_prompt_requires_json_and_does_not_fallback_to_raw_text():
    enhancer = _build_test_enhancer(
        _chat_payload_with_content("A cinematic continuation with slow dolly movement.")
    )
    enhancer.enhance_system_prompt = "enhance system prompt"
    enhancer.rewrite_model_options = ["gpt-test"]
    enhancer.rewrite_default_model = "gpt-test"

    result = asyncio.run(
        enhancer.enhance_prompt(
            "raw input prompt",
            locked_segments=["Segment 1"],
            next_segment_idx=2,
            model="gpt-test",
            timeout_ms=800,
        )
    )
    assert result.fallback_used is True
    assert result.prompt == ""
    assert "No JSON object found in assistant response." in (result.error or "")


def test_generate_auto_prompt_requires_json_and_does_not_fallback_to_raw_text():
    enhancer = _build_test_enhancer(
        _chat_payload_with_content("A calm, grounded continuation with subtle motion.")
    )
    enhancer.auto_system_prompt = "auto system prompt"
    enhancer.rewrite_model_options = ["gpt-test"]
    enhancer.rewrite_default_model = "gpt-test"

    result = asyncio.run(
        enhancer.generate_auto_prompt(
            locked_segments=["Segment 1"],
            next_segment_idx=2,
            model="gpt-test",
            timeout_ms=800,
        )
    )
    assert result.fallback_used is True
    assert result.prompt == ""
    assert "No JSON object found in assistant response." in (result.error or "")


def test_rewrite_prompt_sequence_includes_raw_json_when_content_empty():
    enhancer = _build_test_enhancer(
        {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "content": [],
                        "refusal": None,
                    },
                }
            ],
            "usage": {"completion_tokens": 0},
        }
    )
    result = asyncio.run(
        enhancer.rewrite_prompt_sequence(
            ["prompt one", "prompt two"],
            rewrite_instruction="make it cinematic",
        )
    )
    assert result.fallback_used is True
    assert "No rewrite segment prompts found in assistant response." in (result.error or "")
    assert isinstance(result.raw_response_text, str)
    assert '"finish_reason": "length"' in result.raw_response_text


def test_get_rewrite_model_config_returns_fixed_defaults():
    enhancer = _build_test_enhancer(
        _chat_payload_with_content(
            '{"id":"preset_a","label":"Preset A","segment_prompts":["A","B"]}'
        )
    )
    enhancer.rewrite_default_model = "gpt-oss-120b"
    enhancer.rewrite_model_options = ["gpt-oss-120b"]

    config = enhancer.get_rewrite_model_config()

    assert config["default_model"] == "gpt-oss-120b"
    assert config["options"] == ["gpt-oss-120b"]


def test_get_prompt_config_includes_auto_extension_prompt():
    enhancer = _build_test_enhancer(
        _chat_payload_with_content(
            '{"id":"preset_a","label":"Preset A","segment_prompts":["A","B"]}'
        )
    )
    enhancer.enhance_system_prompt_path = "/tmp/next.md"
    enhancer.auto_system_prompt_path = "/tmp/auto.md"
    enhancer.rewrite_all_system_prompt_path = "/tmp/rewrite.md"
    enhancer.rewrite_user_system_prompt_path = "/tmp/rewrite_user.md"
    enhancer.enhance_system_prompt = "next prompt"
    enhancer.auto_system_prompt = "auto prompt"
    enhancer.rewrite_all_system_prompt = "rewrite prompt"
    enhancer.rewrite_user_system_prompt = "rewrite user prompt"

    config = enhancer.get_prompt_config()

    assert config["next_segment_system_prompt_path"] == "/tmp/next.md"
    assert config["auto_extension_system_prompt_path"] == "/tmp/auto.md"
    assert config["rewrite_window_system_prompt_path"] == "/tmp/rewrite.md"
    assert config["rewrite_user_system_prompt_path"] == "/tmp/rewrite_user.md"
    assert config["next_segment_system_prompt"] == "next prompt"
    assert config["auto_extension_system_prompt"] == "auto prompt"
    assert config["rewrite_window_system_prompt"] == "rewrite prompt"
    assert config["rewrite_user_system_prompt"] == "rewrite user prompt"
    assert config["rewrite_model"] == "gpt-test"
    assert config["rewrite_model_options"] == ["gpt-test"]
    assert config["rewrite_temperature"] == 0.4


def test_get_prompt_config_reports_loaded_fallback_prompt_path(tmp_path):
    enhancer = _build_test_enhancer(
        _chat_payload_with_content(
            '{"id":"preset_a","label":"Preset A","segment_prompts":["A","B"]}'
        )
    )
    rewrite_fallback_path = tmp_path / "rewrite_window_system_prompt.md"
    rewrite_fallback_path.write_text("rewrite prompt\n", encoding="utf-8")
    next_path = tmp_path / "next.md"
    next_path.write_text("next prompt\n", encoding="utf-8")
    auto_path = tmp_path / "auto.md"
    auto_path.write_text("auto prompt\n", encoding="utf-8")
    enhancer.rewrite_all_system_prompt_path = str(
        tmp_path / "prompts.local" / "rewrite_window_system_prompt.md"
    )
    enhancer.rewrite_all_system_prompt_fallback_path = str(rewrite_fallback_path)
    enhancer.enhance_system_prompt_path = str(next_path)
    enhancer.auto_system_prompt_path = str(auto_path)
    enhancer.enhance_system_prompt_fallback_path = None
    enhancer.auto_system_prompt_fallback_path = None
    enhancer.reload_system_prompts()

    config = enhancer.get_prompt_config()

    assert config["rewrite_window_system_prompt_path"] == str(rewrite_fallback_path)


def test_reload_system_prompts_falls_back_to_rewrite_window_when_user_prompt_empty(
    tmp_path,
):
    enhancer = _build_test_enhancer(
        _chat_payload_with_content(
            '{"id":"preset_a","label":"Preset A","segment_prompts":["A","B"]}'
        )
    )
    next_path = tmp_path / "next.md"
    auto_path = tmp_path / "auto.md"
    rewrite_path = tmp_path / "rewrite_window_system_prompt.md"
    rewrite_user_path = tmp_path / "rewrite_user_system_prompt.md"
    next_path.write_text("next prompt\n", encoding="utf-8")
    auto_path.write_text("auto prompt\n", encoding="utf-8")
    rewrite_path.write_text("rewrite window prompt\n", encoding="utf-8")
    rewrite_user_path.write_text("\n", encoding="utf-8")
    enhancer.enhance_system_prompt_path = str(next_path)
    enhancer.auto_system_prompt_path = str(auto_path)
    enhancer.rewrite_all_system_prompt_path = str(rewrite_path)
    enhancer.rewrite_user_system_prompt_path = str(rewrite_user_path)
    enhancer.enhance_system_prompt_fallback_path = None
    enhancer.auto_system_prompt_fallback_path = None
    enhancer.rewrite_all_system_prompt_fallback_path = None
    enhancer.rewrite_user_system_prompt_fallback_path = None

    enhancer.reload_system_prompts()

    assert enhancer.rewrite_all_system_prompt == "rewrite window prompt"
    assert enhancer.rewrite_user_system_prompt == "rewrite window prompt"
    assert enhancer.rewrite_user_system_prompt_source_path == str(rewrite_path)


def test_save_prompt_config_updates_auto_extension_prompt(tmp_path):
    enhancer = _build_test_enhancer(
        _chat_payload_with_content(
            '{"id":"preset_a","label":"Preset A","segment_prompts":["A","B"]}'
        )
    )
    next_path = tmp_path / "next.md"
    auto_path = tmp_path / "auto.md"
    rewrite_path = tmp_path / "rewrite.md"
    next_path.write_text("next original\n", encoding="utf-8")
    auto_path.write_text("auto original\n", encoding="utf-8")
    rewrite_path.write_text("rewrite original\n", encoding="utf-8")
    enhancer.enhance_system_prompt_path = str(next_path)
    enhancer.auto_system_prompt_path = str(auto_path)
    enhancer.rewrite_all_system_prompt_path = str(rewrite_path)

    config = enhancer.save_prompt_config(
        auto_extension_system_prompt="auto updated",
    )

    assert auto_path.read_text(encoding="utf-8").strip() == "auto updated"
    assert config["auto_extension_system_prompt"] == "auto updated"


def test_save_prompt_config_updates_rewrite_user_prompt(tmp_path):
    enhancer = _build_test_enhancer(
        _chat_payload_with_content(
            '{"id":"preset_a","label":"Preset A","segment_prompts":["A","B"]}'
        )
    )
    next_path = tmp_path / "next.md"
    auto_path = tmp_path / "auto.md"
    rewrite_path = tmp_path / "rewrite.md"
    rewrite_user_path = tmp_path / "rewrite_user.md"
    next_path.write_text("next original\n", encoding="utf-8")
    auto_path.write_text("auto original\n", encoding="utf-8")
    rewrite_path.write_text("rewrite original\n", encoding="utf-8")
    rewrite_user_path.write_text("rewrite user original\n", encoding="utf-8")
    enhancer.enhance_system_prompt_path = str(next_path)
    enhancer.auto_system_prompt_path = str(auto_path)
    enhancer.rewrite_all_system_prompt_path = str(rewrite_path)
    enhancer.rewrite_user_system_prompt_path = str(rewrite_user_path)
    enhancer.enhance_system_prompt_fallback_path = None
    enhancer.auto_system_prompt_fallback_path = None
    enhancer.rewrite_all_system_prompt_fallback_path = None
    enhancer.rewrite_user_system_prompt_fallback_path = None

    config = enhancer.save_prompt_config(
        rewrite_user_system_prompt="rewrite user updated",
    )

    assert rewrite_user_path.read_text(encoding="utf-8").strip() == "rewrite user updated"
    assert config["rewrite_user_system_prompt"] == "rewrite user updated"


def test_save_prompt_config_updates_rewrite_model(tmp_path):
    enhancer = _build_test_enhancer(
        _chat_payload_with_content(
            '{"id":"preset_a","label":"Preset A","segment_prompts":["A","B"]}'
        )
    )
    next_path = tmp_path / "next.md"
    auto_path = tmp_path / "auto.md"
    rewrite_path = tmp_path / "rewrite.md"
    next_path.write_text("next prompt\n", encoding="utf-8")
    auto_path.write_text("auto prompt\n", encoding="utf-8")
    rewrite_path.write_text("rewrite prompt\n", encoding="utf-8")
    enhancer.enhance_system_prompt_path = str(next_path)
    enhancer.auto_system_prompt_path = str(auto_path)
    enhancer.rewrite_all_system_prompt_path = str(rewrite_path)
    enhancer.enhance_system_prompt_fallback_path = None
    enhancer.auto_system_prompt_fallback_path = None
    enhancer.rewrite_all_system_prompt_fallback_path = None
    enhancer.rewrite_default_model = "gpt-test"
    enhancer.rewrite_model_options = ["gpt-test", "gpt-alt"]

    config = enhancer.save_prompt_config(
        rewrite_model="gpt-alt",
    )

    assert enhancer.rewrite_default_model == "gpt-alt"
    assert config["rewrite_model"] == "gpt-alt"
    assert config["rewrite_model_options"] == ["gpt-test", "gpt-alt"]


def test_save_prompt_config_updates_rewrite_temperature(tmp_path):
    enhancer = _build_test_enhancer(
        _chat_payload_with_content(
            '{"id":"preset_a","label":"Preset A","segment_prompts":["A","B"]}'
        )
    )
    next_path = tmp_path / "next.md"
    auto_path = tmp_path / "auto.md"
    rewrite_path = tmp_path / "rewrite.md"
    next_path.write_text("next prompt\n", encoding="utf-8")
    auto_path.write_text("auto prompt\n", encoding="utf-8")
    rewrite_path.write_text("rewrite prompt\n", encoding="utf-8")
    enhancer.enhance_system_prompt_path = str(next_path)
    enhancer.auto_system_prompt_path = str(auto_path)
    enhancer.rewrite_all_system_prompt_path = str(rewrite_path)
    enhancer.enhance_system_prompt_fallback_path = None
    enhancer.auto_system_prompt_fallback_path = None
    enhancer.rewrite_all_system_prompt_fallback_path = None

    config = enhancer.save_prompt_config(
        rewrite_temperature=1.3,
    )

    assert enhancer.rewrite_default_temperature == 1.3
    assert config["rewrite_temperature"] == 1.3


def test_save_prompt_config_creates_versioned_backup_for_existing_prompt(tmp_path):
    enhancer = _build_test_enhancer(
        _chat_payload_with_content(
            '{"id":"preset_a","label":"Preset A","segment_prompts":["A","B"]}'
        )
    )
    next_path = tmp_path / "next.md"
    auto_path = tmp_path / "auto.md"
    rewrite_path = tmp_path / "rewrite_window_system_prompt.md"
    next_path.write_text("next original\n", encoding="utf-8")
    auto_path.write_text("auto original\n", encoding="utf-8")
    rewrite_path.write_text("rewrite original\n", encoding="utf-8")
    enhancer.enhance_system_prompt_path = str(next_path)
    enhancer.auto_system_prompt_path = str(auto_path)
    enhancer.rewrite_all_system_prompt_path = str(rewrite_path)
    enhancer.enhance_system_prompt_fallback_path = None
    enhancer.auto_system_prompt_fallback_path = None
    enhancer.rewrite_all_system_prompt_fallback_path = None

    enhancer.save_prompt_config(
        rewrite_window_system_prompt="rewrite updated",
    )

    backup_paths = sorted(
        tmp_path.glob("rewrite_window_system_prompt.*.bak.md")
    )

    assert rewrite_path.read_text(encoding="utf-8").strip() == "rewrite updated"
    assert len(backup_paths) == 1
    assert backup_paths[0].read_text(encoding="utf-8").strip() == "rewrite original"


def test_resolve_prompt_save_path_prefers_existing_fallback_file(tmp_path):
    primary_path = tmp_path / "prompts.local" / "rewrite.md"
    fallback_path = tmp_path / "rewrite.md"
    fallback_path.write_text("existing\n", encoding="utf-8")

    resolved = _resolve_prompt_save_path(str(primary_path), str(fallback_path))

    assert resolved == str(fallback_path)


def test_build_prompt_backup_path_preserves_extension(tmp_path):
    prompt_path = tmp_path / "rewrite_window_system_prompt.md"

    backup_path = _build_prompt_backup_path(prompt_path)

    assert backup_path.parent == tmp_path
    assert re.match(
        r"rewrite_window_system_prompt\.\d{8}_\d{6}\.bak\.md$",
        backup_path.name,
    )
