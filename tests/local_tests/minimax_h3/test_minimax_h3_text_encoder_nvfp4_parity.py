# SPDX-License-Identifier: Apache-2.0
"""Smoke test for a converted NVFP4 MiniMax-H3 text encoder.

Loads the bf16 conditioner and a checkpoint written by
``scripts/checkpoint_conversion/convert_minimax_h3_text_encoder_nvfp4.py``
through the production ``TextEncoderLoader`` and compares the layer-50 hidden
states on a few prompts. A layout, transposition or scale regression in the
serialized NVFP4 path shows up here as a cosine near zero; genuine 4-bit
noise stays above the thresholds.

Run on a Blackwell GPU::

    MINIMAX_H3_RUN_NVFP4_PARITY=1 \
    MINIMAX_H3_MODEL_ROOT=/path/to/FastH3 \
    MINIMAX_H3_NVFP4_TEXT_ENCODER=/path/to/FastH3-nvfp4/text_encoder \
    pytest tests/local_tests/minimax_h3/test_minimax_h3_text_encoder_nvfp4_parity.py -s
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fastvideo.configs.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLConfig
from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel
from fastvideo.models.encoders.minimax_h3_checkpoint_nvfp4 import MiniMaxH3SerializedNVFP4LinearMethod
from fastvideo.models.loader.component_loader import TextEncoderLoader

PROMPTS = (
    "an alpine lake at sunrise, gentle wind over the water",
    "a red fox crosses fresh snow at sunrise",
    "a busy night market in Taipei, neon signs reflecting on wet pavement, handheld camera",
    # 170 tokens with the Qwen3-VL tokenizer, so the activation path crosses a 128-row FlashInfer tile.
    # Natural text on purpose: a clause repeated three times scores 0.975 mean cosine because attention
    # between identical copies is a tie that any noise can flip, which says nothing about real prompts.
    "A slow cinematic drone shot glides over a coastal town at golden hour. Terracotta rooftops catch the "
    "last light while laundry lines sway between balconies. A fishing boat returns to the harbor with gulls "
    "trailing behind it, and children chase a ball down a cobblestone lane past a bakery whose windows are "
    "fogged with steam. An old man reads on a bench under a lemon tree as waves break softly against the "
    "pier and a church bell rings twice. Warm haze softens the distant hills, a tram crosses a stone bridge, "
    "and a street musician tunes a guitar near a fountain where pigeons drink. The camera finally settles "
    "on a cafe terrace where a waiter pours coffee, a couple shares a newspaper, and a cat stretches in the "
    "last patch of sunlight before the shadows reach the water and the harbor lights flicker on one by one.",
)
MIN_MEAN_COSINE = float(os.environ.get("MINIMAX_H3_NVFP4_MIN_COSINE", "0.99"))
# Individual low-content tokens such as an article or a word fragment rotate easily under 4-bit noise
# while carrying almost no signal, so the per-token criterion is a share, not a minimum.
LOW_TOKEN_COSINE = 0.90
MAX_LOW_TOKEN_SHARE = float(os.environ.get("MINIMAX_H3_NVFP4_MAX_LOW_TOKEN_SHARE", "0.02"))
MAX_RELATIVE_ERROR = float(os.environ.get("MINIMAX_H3_NVFP4_MAX_REL_ERR", "0.10"))


def _require_assets() -> tuple[torch.device, Path, Path]:
    if os.environ.get("MINIMAX_H3_RUN_NVFP4_PARITY") != "1":
        pytest.skip("set MINIMAX_H3_RUN_NVFP4_PARITY=1 on a Blackwell GPU node")
    if not torch.cuda.is_available():
        pytest.fail("MiniMax-H3 NVFP4 parity requires a CUDA GPU", pytrace=False)
    root = os.environ.get("MINIMAX_H3_MODEL_ROOT")
    converted = os.environ.get("MINIMAX_H3_NVFP4_TEXT_ENCODER")
    if not root or not converted:
        pytest.fail("set MINIMAX_H3_MODEL_ROOT and MINIMAX_H3_NVFP4_TEXT_ENCODER", pytrace=False)
    root_path, converted_path = Path(root), Path(converted)
    missing = [str(p) for p in (root_path / "text_encoder", root_path / "tokenizer", converted_path) if not p.is_dir()]
    if missing:
        pytest.fail(f"directories are missing: {missing}", pytrace=False)
    if not (converted_path / "config.json").is_file():
        pytest.fail(f"{converted_path} has no config.json; point at the converted text_encoder directory",
                    pytrace=False)
    return torch.device("cuda"), root_path, converted_path


@pytest.fixture(scope="module", autouse=True)
def _distributed_runtime():
    if os.environ.get("MINIMAX_H3_RUN_NVFP4_PARITY") != "1":
        yield
        return
    for name, value in {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29624",
            "RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_RANK": "0"
    }.items():
        os.environ.setdefault(name, value)
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    yield
    cleanup_dist_env_and_memory()


def _loader_args() -> SimpleNamespace:
    return SimpleNamespace(
        pipeline_config=SimpleNamespace(
            text_encoder_configs=(MiniMaxH3Qwen3VLConfig(), ),
            text_encoder_precisions=("bf16", ),
        ),
        text_encoder_cpu_offload=False,
        override_text_encoder_quant=None,
        override_text_encoder_safetensors=None,
        pin_cpu_memory=False,
        disable_offload_on_unified_memory=lambda device_id=0, *, offload_flag=None: True,
    )


def _encode(text_encoder_dir: Path, tokenizer, device: torch.device, expect_nvfp4: bool,
            prompts: tuple[str, ...]) -> list[torch.Tensor]:
    model = TextEncoderLoader().load(str(text_encoder_dir), _loader_args())
    probe = model.language_model.layers[0].self_attn.q_proj
    is_nvfp4 = isinstance(probe.quant_method, MiniMaxH3SerializedNVFP4LinearMethod) and probe.weight is None
    assert is_nvfp4 == expect_nvfp4, (f"{text_encoder_dir} loaded {'through' if is_nvfp4 else 'without'} the "
                                      "serialized NVFP4 path")
    outputs = []
    for prompt in prompts:
        ids = torch.tensor(tokenizer(prompt, add_special_tokens=False)["input_ids"], dtype=torch.long, device=device)
        with torch.inference_mode():
            outputs.append(model(input_ids=ids).float().cpu())
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return outputs


def test_minimax_h3_text_encoder_nvfp4_parity() -> None:
    from transformers import AutoTokenizer

    device, root, converted = _require_assets()
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
    prompts = PROMPTS
    reference = _encode(root / "text_encoder", tokenizer, device, expect_nvfp4=False, prompts=prompts)
    quantized = _encode(converted, tokenizer, device, expect_nvfp4=True, prompts=prompts)
    assert reference[-1].shape[0] > 128, "the long prompt must cross a 128-row tile"

    print(f"\n{'prompt':<60}{'tokens':>7}{'cos mean':>10}{'cos min':>9}{'low share':>10}{'rel err':>10}",
          flush=True)
    for prompt, expected, actual in zip(prompts, reference, quantized, strict=True):
        assert actual.shape == expected.shape
        cosine = torch.nn.functional.cosine_similarity(expected, actual, dim=-1)
        low_share = (cosine < LOW_TOKEN_COSINE).float().mean().item()
        relative_error = ((expected - actual).norm() / expected.norm()).item()
        print(f"{prompt[:58]:<60}{expected.shape[0]:>7}{cosine.mean().item():>10.4f}{cosine.min().item():>9.4f}"
              f"{low_share:>10.3f}{relative_error:>10.3e}", flush=True)
        assert cosine.mean().item() >= MIN_MEAN_COSINE, f"{prompt[:40]!r}: mean cosine {cosine.mean().item():.4f}"
        assert low_share <= MAX_LOW_TOKEN_SHARE, (f"{prompt[:40]!r}: {low_share:.1%} of tokens below "
                                                  f"{LOW_TOKEN_COSINE} cosine")
        assert relative_error <= MAX_RELATIVE_ERROR, f"{prompt[:40]!r}: relative error {relative_error:.3e}"
