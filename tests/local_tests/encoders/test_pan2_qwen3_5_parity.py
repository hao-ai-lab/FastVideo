# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 text-encoder parity for PAN2 against Transformers' ``Qwen3_5TextModel``.

Coverage scope: both. The checkpoint test loads the public tiny PAN2 text
encoder through FastVideo's production ``TextEncoderLoader`` and feeds both
models the exact PAN2 chat-templated token ids. The random-config tests use a
larger hybrid stack (GQA, partial RoPE, more value than key heads in the Gated
DeltaNet, sequences that cross several 64-token delta-rule chunks) and also
cover bf16 and padded batches.

Everything runs on CPU. Set ``PAN2_TINY_MODEL_DIR`` to a local copy of
``wuqing157/tiny-pan2-modular-pipe``; otherwise the HF cache is used and the
checkpoint test skips when it is absent.
"""

from __future__ import annotations

import copy
import os
import socket
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch.testing import assert_close

from fastvideo.configs.models.encoders.qwen3_5 import Qwen3_5TextConfig
from fastvideo.forward_context import set_forward_context
from fastvideo.models.loader.utils import set_default_torch_dtype

PARITY_SCOPE = "both"
TINY_REPO_ID = "wuqing157/tiny-pan2-modular-pipe"
PAN2_SYSTEM_MESSAGE = "You are a helpful assistant. Generate the video with the following instructions."
PAN2_TEXT_ENCODER_LAYERS_TO_SKIP = 2
PAN2_PROMPTS = (
    "A red fox crosses fresh snow at sunrise.",
    "A slow dolly shot glides through a rain-soaked neon market at night. Steam rises from food stalls while "
    "vendors in bright raincoats shout over the hum of generators. Reflections of pink, teal and amber signs ripple "
    "across puddles as a cyclist weaves between crowds, her umbrella catching droplets that sparkle in the light. "
    "In the background, a tram rattles past an old clock tower, its windows glowing warm yellow. The camera rises to "
    "reveal rooftops crowded with antennas and laundry lines, then tilts toward a thunderstorm rolling in over the "
    "harbor, lightning briefly silhouetting cranes and container ships. Cinematic lighting, shallow depth of field, "
    "35mm film grain, realistic motion blur.",
)
FP32_TOLERANCE = {"atol": 1e-5, "rtol": 1e-5}
BF16_TOLERANCE = {"atol": 2e-2, "rtol": 2e-2}


def _hf_qwen3_5() -> tuple[Any, Any]:
    module = pytest.importorskip("transformers.models.qwen3_5", reason="transformers without Qwen3.5 support")
    return module.Qwen3_5TextConfig, module.Qwen3_5TextModel


def _tiny_model_dir() -> Path | None:
    local = os.getenv("PAN2_TINY_MODEL_DIR")
    if local:
        return Path(local)
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(TINY_REPO_ID, local_files_only=True))
    except Exception:
        return None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def cpu_single_rank():
    """Single-rank CPU runtime; the loader would otherwise target ``mps`` on CPU-only Linux hosts."""
    import fastvideo.distributed.parallel_state as parallel_state
    import fastvideo.models.loader.component_loader as component_loader
    from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(parallel_state, "get_local_torch_device", lambda: torch.device("cpu"))
        patch.setattr(component_loader, "get_local_torch_device", lambda: torch.device("cpu"))
        for name, value in (("MASTER_ADDR", "127.0.0.1"), ("MASTER_PORT", str(_free_port())), ("RANK", "0"),
                            ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")):
            patch.setenv(name, value)
        maybe_init_distributed_environment_and_model_parallel(1, 1)
        yield
        cleanup_dist_env_and_memory()


def _loader_args(config: Qwen3_5TextConfig, precision: str = "fp32") -> SimpleNamespace:
    return SimpleNamespace(
        pipeline_config=SimpleNamespace(text_encoder_configs=(config, ), text_encoder_precisions=(precision, )),
        text_encoder_cpu_offload=False,
        override_text_encoder_quant=None,
        override_text_encoder_safetensors=None,
        pin_cpu_memory=False,
        disable_offload_on_unified_memory=lambda device_id, offload_flag=None: False,
    )


def _pan2_input_ids(tokenizer: Any, prompt: str) -> tuple[torch.Tensor, int]:
    """Tokenize exactly like PAN2's ``encode_prompt`` and return ids plus the user-header crop."""
    messages = [{"role": "system", "content": PAN2_SYSTEM_MESSAGE}, {"role": "user", "content": prompt or " "}]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                              return_dict=True).input_ids
    header_ids = tokenizer("<|im_start|>user\n", add_special_tokens=False).input_ids
    crop_start = next(
        (i + len(header_ids) for i in range(len(input_ids)) if input_ids[i:i + len(header_ids)] == header_ids),
        None,
    )
    assert crop_start is not None, "PAN2 chat template has no user header"
    return torch.tensor([input_ids]), crop_start


def _run_both(reference: torch.nn.Module, native: torch.nn.Module, **inputs: Any) -> tuple[Any, Any]:
    with torch.inference_mode():
        expected = reference(**inputs, output_hidden_states=True)
        with set_forward_context(current_timestep=0, attn_metadata=None):
            actual = native(**inputs, output_hidden_states=True)
    return expected, actual


def _assert_hidden_states_close(
    expected: tuple[torch.Tensor, ...],
    actual: tuple[torch.Tensor, ...] | None,
    label: str,
    tolerance: dict[str, float],
    valid: torch.Tensor | None = None,
) -> float:
    assert actual is not None, f"{label}: native encoder returned no hidden_states"
    assert len(actual) == len(expected), f"{label}: {len(actual)} hidden states, reference has {len(expected)}"
    worst = 0.0
    for index, (reference_state, native_state) in enumerate(zip(expected, actual, strict=True)):
        assert native_state.shape == reference_state.shape, f"{label} hidden_states[{index}] shape"
        assert torch.isfinite(native_state).all(), f"{label} hidden_states[{index}] has non-finite values"
        if valid is not None:
            reference_state = reference_state[valid]
            native_state = native_state[valid]
        drift = (native_state.float() - reference_state.float()).abs().max().item()
        worst = max(worst, drift)
        print(f"{label} hidden_states[{index}]: max_abs={drift:.3e}", flush=True)
        assert_close(native_state, reference_state, **tolerance, msg=lambda m: f"{label} hidden_states[{index}]: {m}")
    return worst


def test_pan2_qwen3_5_arch_config_matches_transformers_rope_and_layer_types() -> None:
    """Config.json fields resolve to the same RoPE settings and layer schedule as Transformers."""
    hf_config_cls, _ = _hf_qwen3_5()
    cases = (
        {},
        {
            "rope_theta": 1_000_000.0
        },
        {
            "partial_rotary_factor": 0.5,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 5_000_000.0
            }
        },
        {
            "partial_rotary_factor": 0.5,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 5_000_000.0,
                "partial_rotary_factor": 0.25
            }
        },
        {
            "num_hidden_layers": 6,
            "full_attention_interval": 3
        },
        {
            "num_hidden_layers": 3,
            "layer_types": ["mamba", "attention", "linear_attention"]
        },
    )
    for case in cases:
        reference = hf_config_cls.from_dict(dict(case))
        config = Qwen3_5TextConfig()
        config.update_model_arch(dict(case))
        rope = config.arch_config.resolved_rope_parameters()
        for key in ("rope_type", "rope_theta", "partial_rotary_factor"):
            assert rope[key] == reference.rope_parameters[key], f"{case}: {key}"
        assert config.arch_config.resolved_layer_types() == list(reference.layer_types), case


def test_pan2_qwen3_5_tiny_checkpoint_loader_parity(cpu_single_rank) -> None:
    """Production-loader encoder matches every HF hidden state on real PAN2 prompts (fp32 CPU)."""
    _, hf_model_cls = _hf_qwen3_5()
    model_dir = _tiny_model_dir()
    if model_dir is None or not (model_dir / "text_encoder" / "config.json").is_file():
        pytest.skip(f"{TINY_REPO_ID} is not available locally; set PAN2_TINY_MODEL_DIR")
    if not (model_dir / "tokenizer").is_dir():
        pytest.skip(f"{TINY_REPO_ID} tokenizer is not available locally")
    from transformers import AutoTokenizer

    from fastvideo.models.encoders.qwen3_5 import Qwen3_5TextModel
    from fastvideo.models.loader.component_loader import TextEncoderLoader

    text_encoder_dir = model_dir / "text_encoder"
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer", local_files_only=True)
    reference, loading_info = hf_model_cls.from_pretrained(
        text_encoder_dir,
        local_files_only=True,
        dtype=torch.float32,
        output_loading_info=True,
    )
    load_errors = {
        name: loading_info.get(name)
        for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if loading_info.get(name)
    }
    assert not load_errors, f"Reference Qwen3.5 checkpoint did not load strictly: {load_errors}"
    reference = reference.eval()

    native = TextEncoderLoader().load(str(text_encoder_dir), _loader_args(Qwen3_5TextConfig()))
    assert isinstance(native, Qwen3_5TextModel)
    assert native.layer_types == list(reference.config.layer_types)
    assert next(native.parameters()).dtype == torch.float32
    assert_close(native.rotary_emb.inv_freq, reference.rotary_emb.inv_freq.float(), atol=0.0, rtol=0.0)

    for prompt_index, prompt in enumerate(PAN2_PROMPTS):
        input_ids, crop_start = _pan2_input_ids(tokenizer, prompt)
        label = f"tiny prompt{prompt_index} (L={input_ids.shape[1]})"
        expected, actual = _run_both(reference, native, input_ids=input_ids)
        worst = _assert_hidden_states_close(expected.hidden_states, actual.hidden_states, label, FP32_TOLERANCE)
        assert_close(actual.last_hidden_state, expected.last_hidden_state, **FP32_TOLERANCE)

        skip = PAN2_TEXT_ENCODER_LAYERS_TO_SKIP + 1
        reference_embeds = expected.hidden_states[-skip][:, crop_start:]
        native_embeds = actual.hidden_states[-skip][:, crop_start:]
        assert_close(native_embeds, reference_embeds, **FP32_TOLERANCE)
        print(
            f"{label} PAN2 prompt_embeds hidden_states[-{skip}][:, {crop_start}:] "
            f"max_abs={(native_embeds - reference_embeds).abs().max().item():.3e} all_layers_max_abs={worst:.3e}",
            flush=True,
        )


@pytest.fixture(scope="module")
def random_pair(cpu_single_rank):
    """A randomly initialized 8-layer hybrid reference and the native model loaded from its state dict."""
    hf_config_cls, hf_model_cls = _hf_qwen3_5()
    from fastvideo.models.encoders.qwen3_5 import Qwen3_5TextModel

    torch.manual_seed(0)
    hf_config = hf_config_cls(
        vocab_size=1024,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=24,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1_000_000.0,
            "partial_rotary_factor": 0.25
        },
        initializer_range=0.08,
    )
    reference = hf_model_cls(hf_config).eval()
    with torch.no_grad():
        # Move the zero-centered norms, gates and decay parameters off their
        # init values so every term contributes.
        for name, parameter in reference.named_parameters():
            if name.endswith("norm.weight") or name.endswith(("A_log", "dt_bias")):
                parameter.add_(0.2 * torch.randn_like(parameter))

    config = Qwen3_5TextConfig()
    config.update_model_arch(hf_config.to_dict())
    with set_default_torch_dtype(torch.float32):
        native = Qwen3_5TextModel(config).eval()
    loaded = native.load_weights(reference.state_dict().items())
    expected_names = {name for name, _ in native.named_parameters()}
    assert loaded == expected_names, f"unloaded native parameters: {sorted(expected_names - loaded)}"
    assert set(reference.state_dict()) == expected_names
    return reference, native


def test_pan2_qwen3_5_random_config_fp32_parity(random_pair) -> None:
    """Multi-chunk delta rule (L=301), GQA and partial RoPE match HF in fp32."""
    reference, native = random_pair
    generator = torch.Generator().manual_seed(1)
    input_ids = torch.randint(0, reference.config.vocab_size, (2, 301), generator=generator)
    expected, actual = _run_both(reference, native, input_ids=input_ids)
    worst = _assert_hidden_states_close(expected.hidden_states, actual.hidden_states, "random fp32", FP32_TOLERANCE)
    print(f"random fp32 all_layers_max_abs={worst:.3e}", flush=True)
    # Each layer must move the residual stream, or the comparison is vacuous.
    for index in range(1, len(expected.hidden_states) - 1):
        change = (expected.hidden_states[index] - expected.hidden_states[index - 1]).abs().max().item()
        assert change > 0.1, f"layer {index - 1} barely changes the hidden state ({change:.3e})"


def test_pan2_qwen3_5_random_config_padding_parity(random_pair) -> None:
    """Right- and left-padded rows match HF on every non-padded position."""
    reference, native = random_pair
    generator = torch.Generator().manual_seed(2)
    input_ids = torch.randint(0, reference.config.vocab_size, (2, 200), generator=generator)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[0, 150:] = 0
    attention_mask[1, :37] = 0
    expected, actual = _run_both(reference, native, input_ids=input_ids, attention_mask=attention_mask)
    worst = _assert_hidden_states_close(
        expected.hidden_states,
        actual.hidden_states,
        "random padded",
        FP32_TOLERANCE,
        valid=attention_mask.bool(),
    )
    print(f"random padded all_layers_max_abs={worst:.3e}", flush=True)


def test_pan2_qwen3_5_random_config_bf16_parity(random_pair) -> None:
    """Half-precision dtype handling (fp32 norms, gates and delta rule) follows HF."""
    reference, native = (copy.deepcopy(model).to(torch.bfloat16) for model in random_pair)
    generator = torch.Generator().manual_seed(3)
    input_ids = torch.randint(0, reference.config.vocab_size, (1, 173), generator=generator)
    expected, actual = _run_both(reference, native, input_ids=input_ids)
    assert actual.hidden_states is not None and actual.hidden_states[-1].dtype == torch.bfloat16
    worst = _assert_hidden_states_close(expected.hidden_states, actual.hidden_states, "random bf16", BF16_TOLERANCE)
    print(f"random bf16 all_layers_max_abs={worst:.3e}", flush=True)
