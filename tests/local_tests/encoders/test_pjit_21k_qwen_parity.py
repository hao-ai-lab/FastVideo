# SPDX-License-Identifier: Apache-2.0
"""Compare P-JiT conditioning to the unmodified Linum reference.

Set PJIT_REFERENCE_ROOT to the official linum-ai/pjit-21k checkout and
PJIT_QWEN_MODEL_DIR to a local Qwen/Qwen3.5-4B snapshot. The lightweight test
uses its real tokenizer and a tiny random 28-block Qwen3.5 text tower with
identical reference/native weights. PJIT_ENCODER_DEVICE selects cpu or mps.
Set PJIT_REAL_ENCODER_PARITY=1 to additionally load the real 4B checkpoint.
"""
from __future__ import annotations

import gc
import importlib.util
import json
import os
import socket
from pathlib import Path

import pytest
import torch

from fastvideo.configs.models.encoders.qwen3_5 import Qwen3_5TextConfig
from fastvideo.forward_context import set_forward_context
from fastvideo.models.encoders.pjit_21k import PJiTTextConditioner
from fastvideo.models.encoders.qwen3_5 import Qwen3_5TextModel
from fastvideo.models.loader.utils import set_default_torch_dtype

PARITY_SCOPE = "both"
PROMPTS = ["A red fox in fresh snow.", "A sunlit landscape with a blue lake and distant mountains."]


@pytest.fixture(scope="module")
def runtime():
    import fastvideo.distributed.parallel_state as parallel_state
    from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel

    device = torch.device(os.environ.get("PJIT_ENCODER_DEVICE", "cpu"))
    if device.type == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(parallel_state, "get_local_torch_device", lambda: torch.device("cpu"))
        for name, value in (("MASTER_ADDR", "127.0.0.1"), ("MASTER_PORT", str(port)), ("RANK", "0"),
                            ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")):
            patch.setenv(name, value)
        maybe_init_distributed_environment_and_model_parallel(1, 1)
        yield device
        cleanup_dist_env_and_memory()


def _reference_class():
    root = os.environ.get("PJIT_REFERENCE_ROOT")
    if not root:
        pytest.skip("Set PJIT_REFERENCE_ROOT to the official P-JiT checkout")
    path = Path(root) / "pjit_21k" / "text_encoder.py"
    spec = importlib.util.spec_from_file_location("pjit_reference_text_encoder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.QwenTextEncoder


def _model_dir():
    path = os.environ.get("PJIT_QWEN_MODEL_DIR")
    if not path:
        pytest.skip("Set PJIT_QWEN_MODEL_DIR to local Qwen3.5-4B weights/tokenizer")
    return path


def _compare(reference, native, tokenizer, device):
    # Bypass only the reference constructor so its unmodified __call__ can run
    # with the same small model, real tokenizer, device and extraction recipe.
    reference_class = _reference_class()
    official = reference_class.__new__(reference_class)
    official.model = reference
    official.tokenizer = tokenizer
    official.device = device
    official.max_length = 256
    official.extraction_layers = [7, 15, 27]
    conditioner = PJiTTextConditioner(native, tokenizer, device=device)
    with set_forward_context(current_timestep=0, attn_metadata=None):
        actual = conditioner(PROMPTS)
    expected = official(PROMPTS)
    for index, (value, target) in enumerate(zip(actual, expected, strict=True)):
        drift = (value.float() - target.float()).abs()
        print(f"prompt {index}: shape={tuple(value.shape)}, dtype={value.dtype}, "
              f"max_abs={drift.max().item():.6g}, mean_abs={drift.mean().item():.6g}", flush=True)
        torch.testing.assert_close(value, target, atol=0.02, rtol=0.02)


def test_pjit_qwen_tiny_official_reference_parity(runtime):
    from transformers import AutoTokenizer
    from transformers import Qwen3_5TextConfig as HFConfig
    from transformers import Qwen3_5TextModel as HFModel

    _reference_class()
    tokenizer = AutoTokenizer.from_pretrained(_model_dir(), local_files_only=True)
    torch.manual_seed(2075)
    config = HFConfig(
        vocab_size=248320, hidden_size=32, intermediate_size=64, num_hidden_layers=28,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=1, linear_num_value_heads=2,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.25},
    )
    reference = HFModel(config).eval().to(device=runtime, dtype=torch.bfloat16)
    native_config = Qwen3_5TextConfig()
    native_config.update_model_arch(config.to_dict())
    with set_default_torch_dtype(torch.bfloat16):
        native = Qwen3_5TextModel(native_config).eval().to(runtime)
    loaded = native.load_weights(reference.state_dict().items())
    assert loaded == set(dict(native.named_parameters()))
    _compare(reference, native, tokenizer, runtime)


@pytest.mark.skipif(os.environ.get("PJIT_REAL_ENCODER_PARITY") != "1", reason="4B checkpoint test is opt-in")
def test_pjit_qwen_real_official_reference_parity(runtime):
    from safetensors import safe_open

    model_dir = _model_dir()
    official = _reference_class()(model_dir, device=runtime)
    print(f"Reference backend: {official.model.config._attn_implementation}", flush=True)
    config = Qwen3_5TextConfig()
    config.update_model_arch(official.model.config.to_dict())
    prompts = [os.environ.get("PJIT_REAL_ENCODER_PROMPT", PROMPTS[0]), ""]
    expected = [official([prompt])[0].cpu() for prompt in prompts]
    tokenizer = official.tokenizer
    del official
    gc.collect()
    if runtime.type == "mps":
        torch.mps.empty_cache()
    with set_default_torch_dtype(torch.bfloat16):
        native = Qwen3_5TextModel(config).eval()

    def weights():
        for path in sorted(Path(model_dir).glob("*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    yield name, handle.get_tensor(name)

    loaded = native.load_weights(weights())
    assert loaded == set(dict(native.named_parameters()))
    native = native.to(runtime)
    conditioner = PJiTTextConditioner(native, tokenizer, device=runtime)
    with set_forward_context(current_timestep=0, attn_metadata=None):
        actual = [conditioner([prompt])[0].cpu() for prompt in prompts]
    output_dir = os.environ.get("PJIT_ENCODER_OUTPUT_DIR")
    if output_dir:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"prompts": prompts, "native": actual, "reference": expected}, path / "qwen_conditioning.pt")
        (path / "qwen_prompts.json").write_text(json.dumps(prompts, indent=2))
    for index, (value, target) in enumerate(zip(actual, expected, strict=True)):
        drift = (value.float() - target.float()).abs()
        print(f"real prompt {index}: shape={tuple(value.shape)}, max_abs={drift.max().item():.6g}, "
              f"mean_abs={drift.mean().item():.6g}", flush=True)
        torch.testing.assert_close(value, target, atol=0.02, rtol=0.02)
