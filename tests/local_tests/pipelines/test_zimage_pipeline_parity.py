# SPDX-License-Identifier: Apache-2.0
"""Native Z-Image stage contract and pinned-repository pipeline parity."""

from __future__ import annotations

import gc
import importlib
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.testing import assert_close

from fastvideo.configs.pipelines.zimage import ZImagePipelineConfig
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from fastvideo.pipelines.basic.zimage.stages import (
    ZImageConditioningStage,
    ZImageDecodingStage,
    ZImageDenoisingStage,
    ZImageInputValidationStage,
    ZImageLatentPreparationStage,
    ZImageTimestepPreparationStage,
)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch


REPO_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_REVISION = "26f23eda626ffadda020b04ff79488e1d72004cd"
REFERENCE_REPO = Path(os.getenv("ZIMAGE_REFERENCE_REPO", REPO_ROOT / "Z-Image"))
MODEL_DIR = Path(os.getenv("ZIMAGE_MODEL_DIR", REPO_ROOT / "official_weights" / "Z-Image"))


class _ConstantTransformer:
    in_channels = 16

    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]] = []

    def __call__(
        self,
        *,
        hidden_states: torch.Tensor,
        encoder_hidden_states: list[torch.Tensor],
        timestep: torch.Tensor,
    ) -> tuple[list[torch.Tensor], dict]:
        self.calls.append((hidden_states, encoder_hidden_states, timestep))
        return [torch.ones_like(sample) for sample in hidden_states.unbind(0)], {}


class _RecordingVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(scaling_factor=0.3611, shift_factor=0.1159)
        self.decode_input: torch.Tensor | None = None

    def decode(self, latents: torch.Tensor, return_dict: bool = False) -> tuple[torch.Tensor]:
        assert return_dict is False
        self.decode_input = latents.detach().clone()
        batch, _, height, width = latents.shape
        return (torch.zeros(batch, 3, height * 8, width * 8, device=latents.device),)


def _stage_args(*, output_type: str = "pil") -> SimpleNamespace:
    return SimpleNamespace(
        pipeline_config=ZImagePipelineConfig(),
        output_type=output_type,
        vae_cpu_offload=False,
        disable_autocast=False,
    )


def test_zimage_native_default_stage_math(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the pinned native 1-step formulas without model assets."""
    autocast_calls: list[dict[str, object]] = []

    @contextmanager
    def record_autocast(**kwargs):
        autocast_calls.append(kwargs)
        yield

    monkeypatch.setattr(
        "fastvideo.pipelines.basic.zimage.stages.get_local_torch_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr("fastvideo.pipelines.basic.zimage.stages.torch.autocast", record_autocast)
    args = _stage_args()
    transformer = _ConstantTransformer()
    scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
    vae = _RecordingVAE()
    batch = ForwardBatch(
        data_type="image",
        prompt="test prompt",
        negative_prompt="",
        prompt_embeds=[torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)],
        prompt_attention_mask=[torch.tensor([[True, True, False, False]])],
        height=64,
        width=64,
        num_frames=1,
        num_inference_steps=1,
        guidance_scale=0.0,
        seed=42,
        latents=torch.zeros(1, 16, 1, 8, 8),
    )

    ZImageInputValidationStage().forward(batch, args)
    ZImageConditioningStage().forward(batch, args)
    ZImageLatentPreparationStage(transformer).forward(batch, args)
    ZImageTimestepPreparationStage(scheduler).forward(batch, args)

    assert len(batch.extra["zimage_prompt_embeds"]) == 1
    assert batch.extra["zimage_prompt_embeds"][0].shape == (2, 8)
    assert_close(batch.timesteps, torch.tensor([1000.0]))
    assert scheduler.sigma_min == 0.0
    assert scheduler.config.use_reference_discrete_timesteps is True

    ZImageDenoisingStage(transformer, scheduler).forward(batch, args)
    assert autocast_calls == [{"device_type": "cpu", "enabled": False}]
    assert_close(batch.latents, torch.ones_like(batch.latents))
    _, encoded, normalized_timestep = transformer.calls[0]
    assert encoded[0].shape == (2, 8)
    assert_close(normalized_timestep, torch.tensor([0.0]))

    ZImageDecodingStage(vae).forward(batch, args)
    assert vae.decode_input is not None
    expected_vae_input = torch.full((1, 16, 8, 8), 1 / 0.3611 + 0.1159)
    assert_close(vae.decode_input, expected_vae_input)
    assert batch.output is not None
    assert batch.output.shape == (1, 3, 1, 64, 64)
    assert_close(batch.output, torch.full_like(batch.output, 0.5))


def test_zimage_batch_rng_matches_one_native_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "fastvideo.pipelines.basic.zimage.stages.get_local_torch_device",
        lambda: torch.device("cpu"),
    )
    args = _stage_args(output_type="latent")
    transformer = _ConstantTransformer()
    batch = ForwardBatch(
        data_type="image",
        prompt=["one", "two"],
        negative_prompt="",
        height=64,
        width=64,
        num_frames=1,
        num_inference_steps=1,
        guidance_scale=0.0,
        seed=42,
    )
    ZImageInputValidationStage().forward(batch, args)
    batch.extra["zimage_prompt_embeds"] = [torch.zeros(1, 8), torch.zeros(1, 8)]
    ZImageLatentPreparationStage(transformer).forward(batch, args)

    expected = torch.randn(
        (2, 16, 1, 8, 8),
        generator=torch.Generator("cpu").manual_seed(42),
    )
    assert_close(batch.latents, expected)


def test_zimage_text_stage_honors_request_max_length(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastvideo.pipelines.stages.text_encoding import TextEncodingStage

    args = _stage_args()
    stage = TextEncodingStage(text_encoders=[object()], tokenizers=[object()])
    observed: list[int | None] = []

    def fake_encode_text(
        text,
        fastvideo_args,
        encoder_index,
        return_attention_mask,
        max_length=None,
    ):
        del text, fastvideo_args, encoder_index, return_attention_mask
        observed.append(max_length)
        return [torch.zeros(1, 4, 8)], [torch.ones(1, 4, dtype=torch.long)]

    monkeypatch.setattr(stage, "encode_text", fake_encode_text)
    batch = ForwardBatch(
        data_type="image",
        prompt="test",
        negative_prompt="",
        max_sequence_length=64,
        guidance_scale=0.0,
    )
    stage.forward(batch, args)
    assert observed == [64]


def test_zimage_cfg_none_uses_native_empty_negative_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "fastvideo.pipelines.basic.zimage.stages.get_local_torch_device",
        lambda: torch.device("cpu"),
    )
    batch = ForwardBatch(
        data_type="image",
        prompt="test",
        negative_prompt=None,
        height=64,
        width=64,
        num_frames=1,
        num_inference_steps=1,
        guidance_scale=2.0,
        seed=42,
    )
    ZImageInputValidationStage().forward(batch, _stage_args())
    assert batch.negative_prompt == ""


def _require_pinned_reference() -> Path:
    if not REFERENCE_REPO.is_dir():
        pytest.skip(f"Pinned Tongyi Z-Image clone not found: {REFERENCE_REPO}")
    result = subprocess.run(
        ["git", "-C", str(REFERENCE_REPO), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = result.stdout.strip()
    assert actual == REFERENCE_REVISION, (
        f"Tongyi Z-Image oracle must be pinned at {REFERENCE_REVISION}, got {actual}"
    )
    return REFERENCE_REPO / "src"


def _import_reference_modules() -> tuple[object, object]:
    source_root = _require_pinned_reference().resolve()
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    pipeline_module = importlib.import_module("zimage.pipeline")
    utils_module = importlib.import_module("utils")
    for module in (pipeline_module, utils_module):
        module_path = Path(module.__file__ or "").resolve()
        assert module_path.is_relative_to(source_root), (
            f"Z-Image oracle import escaped the pinned repository: {module_path}"
        )
    return pipeline_module, utils_module


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Full Z-Image pipeline parity requires CUDA")
def test_zimage_pipeline_latents_match_pinned_native_repo(tmp_path: Path) -> None:
    """Compare one complete native step; the oracle never imports Diffusers' pipeline."""
    if not (MODEL_DIR / "model_index.json").is_file():
        pytest.skip(f"Pinned Z-Image weights not found: {MODEL_DIR}")
    pipeline_module, utils_module = _import_reference_modules()
    prompt = "Young Chinese woman in red Hanfu, intricate embroidery."

    previous_fastvideo_backend = os.environ.get("FASTVIDEO_ATTENTION_BACKEND")
    try:
        # Exercise the shared PyTorch SDPA path with each process's native
        # kernel selection; the official and FastVideo implementations both
        # pass the same padding mask into scaled_dot_product_attention.
        utils_module.set_attention_backend(None)
        components = utils_module.load_from_local_dir(
            MODEL_DIR,
            device="cuda",
            dtype=torch.bfloat16,
            compile=False,
        )
        reference = pipeline_module.generate(
            prompt=prompt,
            **components,
            height=64,
            width=64,
            num_inference_steps=1,
            guidance_scale=0.0,
            negative_prompt="",
            max_sequence_length=512,
            generator=torch.Generator("cuda").manual_seed(42),
            output_type="latent",
        ).detach().float().cpu()
        del components
        gc.collect()
        torch.cuda.empty_cache()

        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "TORCH_SDPA"
        from fastvideo import VideoGenerator

        generator = VideoGenerator.from_pretrained(
            str(MODEL_DIR),
            num_gpus=1,
            tp_size=1,
            sp_size=1,
            use_fsdp_inference=False,
            dit_cpu_offload=False,
            dit_layerwise_offload=False,
            text_encoder_cpu_offload=False,
            vae_cpu_offload=False,
            pin_cpu_memory=False,
            output_type="latent",
        )
        try:
            result = generator.generate_video(
                prompt,
                negative_prompt="",
                output_path=str(tmp_path),
                save_video=False,
                return_frames=True,
                height=64,
                width=64,
                num_frames=1,
                fps=1,
                num_inference_steps=1,
                guidance_scale=0.0,
                max_sequence_length=512,
                seed=42,
            )
        finally:
            generator.shutdown()
        actual = result["samples"].squeeze(2).float()
        assert actual.shape == reference.shape
        diff = (actual - reference).abs()
        print(f"Z-Image pipeline latent parity: max={diff.max():.6f}, mean={diff.mean():.6f}")
        assert_close(actual, reference, atol=1e-2, rtol=1e-2)
    finally:
        if previous_fastvideo_backend is None:
            os.environ.pop("FASTVIDEO_ATTENTION_BACKEND", None)
        else:
            os.environ["FASTVIDEO_ATTENTION_BACKEND"] = previous_fastvideo_backend
