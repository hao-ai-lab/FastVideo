# SPDX-License-Identifier: Apache-2.0
"""Released-weight Qwen-Image-2.1 conditioner parity through the production loader.

Run with QWEN_IMAGE21_RUN_ENCODER_PARITY=1 and QWEN_IMAGE21_MODEL_ROOT set to
the complete local HF snapshot. This gate requires a BF16 CUDA GPU; opting in
turns missing assets or dependencies into failures rather than skipped parity.
Set QWEN_IMAGE21_ENCODER_PARITY_REPORT to save the numerical diagnostics as JSON.
"""

import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch.testing import assert_close

PARITY_SCOPE = "production_loader"
ATOL = 1e-2
RTOL = 1e-2


def _numerical_metrics(actual, expected):
    actual, expected = actual.float(), expected.float()
    difference = actual - expected
    reference_rms = expected.square().mean().sqrt()
    rmse = difference.square().mean().sqrt()
    actual_flat, expected_flat = actual.flatten().double(), expected.flatten().double()
    cosine = torch.nn.functional.cosine_similarity(actual_flat, expected_flat, dim=0).clamp(-1, 1)
    return {
        "shape": list(actual.shape),
        "finite": bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
        "max_abs": difference.abs().max().item(),
        "mean_abs": difference.abs().mean().item(),
        "rmse": rmse.item(),
        "reference_rms": reference_rms.item(),
        "relative_l2": (rmse / reference_rms.clamp_min(torch.finfo(torch.float32).tiny)).item(),
        "cosine": cosine.item(),
        "mismatched_elements": int((difference.abs() > ATOL + RTOL * expected.abs()).sum()),
        "total_elements": actual.numel(),
    }


def _make_cases(processor):
    prompt = "<|im_start|>user\nA red fox on fresh snow.<|im_end|>\n<|im_start|>assistant\n"
    cases = {"text": processor(text=[prompt], padding=True, padding_side="left", return_tensors="pt")}
    long_prompt = "<|im_start|>user\n" + "Small red fox on fresh snow. " * 220 + "<|im_end|>\n"
    cases["long_text"] = processor(text=[long_prompt], return_tensors="pt")
    assert cases["long_text"]["input_ids"].shape[-1] > 1024
    for count in (1, 2, 10):
        images = []
        for index in range(count):
            pixels = np.arange(64 * 96 * 3, dtype=np.uint32).reshape(64, 96, 3)
            images.append(Image.fromarray(((pixels * (index + 1) + index * 31) % 256).astype(np.uint8)))
        image_prompt = "<|im_start|>user\n" + " ".join(
            f"<image{index + 1}><|vision_start|><|image_pad|><|vision_end|>" for index in range(count))
        image_prompt += " Blend these references into a snowy scene.<|im_end|>\n"
        cases[f"images_{count}"] = processor(
            text=[image_prompt],
            images=images,
            min_pixels=4096,
            max_pixels=16384,
            return_tensors="pt",
        )
    cases["left_padding"] = processor(text=[prompt, prompt + " A blue sky."], padding=True,
                                       padding_side="left", return_tensors="pt")
    return cases


def test_qwen_image21_conditioner_real_weight_parity():
    if os.environ.get("QWEN_IMAGE21_RUN_ENCODER_PARITY") != "1":
        pytest.skip("set QWEN_IMAGE21_RUN_ENCODER_PARITY=1 on an allocated CUDA GPU")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.fail("Qwen-Image-2.1 conditioner parity requires a BF16 CUDA GPU", pytrace=False)
    checkpoint = os.environ.get("QWEN_IMAGE21_MODEL_ROOT")
    if not checkpoint:
        pytest.fail("set QWEN_IMAGE21_MODEL_ROOT to the downloaded Qwen-Image-2.1 HF snapshot", pytrace=False)
    root = Path(checkpoint)
    if not (root / "text_encoder" / "config.json").is_file() or not (root / "processor").is_dir():
        pytest.fail("The checkpoint must include text_encoder and processor component directories", pytrace=False)

    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor, __version__ as transformers_version

    from fastvideo.configs.models.encoders.qwen_image21 import QwenImage21Qwen3VLConfig
    from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel
    from fastvideo.models.loader.component_loader import TextEncoderLoader

    device = torch.device("cuda", 0)
    processor = Qwen3VLProcessor.from_pretrained(root / "processor", local_files_only=True)
    cases = _make_cases(processor)
    official, loading_info = Qwen3VLForConditionalGeneration.from_pretrained(
        root / "text_encoder",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    errors = {name: loading_info[name] for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
              if loading_info.get(name)}
    assert not errors, f"Official checkpoint strict loading failed: {errors}"
    official = official.to(device).eval()
    norm = official.model.language_model.norm
    handle = norm.register_forward_hook(lambda _module, args, _output: args[0])
    expected = {}
    try:
        for name, case in cases.items():
            inputs = {key: value.to(device=device, dtype=torch.bfloat16 if key == "pixel_values" else value.dtype)
                      for key, value in case.items()}
            with torch.no_grad():
                output = official(**inputs, output_hidden_states=True, use_cache=False)
            expected[name] = output.hidden_states[-1].detach().cpu()
            del output, inputs
    finally:
        handle.remove()
    del official, norm
    gc.collect()
    torch.cuda.empty_cache()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29625")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    config = QwenImage21Qwen3VLConfig()
    config.update_model_arch(json.loads((root / "text_encoder" / "config.json").read_text()))
    args = SimpleNamespace(
        text_encoder_cpu_offload=True,
        override_text_encoder_quant=None,
        override_text_encoder_safetensors=None,
        pin_cpu_memory=False,
        disable_offload_on_unified_memory=lambda _device_id, offload_flag=None: False,
    )
    report = {
        "component": "qwen_image21_conditioner",
        "checkpoint_revision": root.name,
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "atol": ATOL,
        "rtol": RTOL,
        "cases": {},
    }
    failures = []
    try:
        native = TextEncoderLoader().load_model(str(root / "text_encoder"), config, device, args, dtype="bf16")
        for name, case in cases.items():
            inputs = {key: value.to(device=device, dtype=torch.bfloat16 if key == "pixel_values" else value.dtype)
                      for key, value in case.items()}
            with torch.no_grad():
                actual = native(**inputs).detach().cpu()
            mask = case["attention_mask"].bool()
            metrics = _numerical_metrics(actual[mask], expected[name][mask])
            report["cases"][name] = metrics
            print(f"QWEN_IMAGE21_ENCODER_PARITY {name} {json.dumps(metrics, sort_keys=True)}", flush=True)
            try:
                assert_close(actual[mask], expected[name][mask], rtol=RTOL, atol=ATOL,
                             msg=lambda message: f"{name}: {message}")
            except AssertionError as error:
                failures.append(f"{name}: {error}")
            del actual, inputs
    finally:
        report_path = os.environ.get("QWEN_IMAGE21_ENCODER_PARITY_REPORT")
        if report_path:
            destination = Path(report_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        cleanup_dist_env_and_memory()
    assert not failures, "\n\n".join(failures)
