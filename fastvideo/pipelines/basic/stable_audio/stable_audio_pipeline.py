# SPDX-License-Identifier: Apache-2.0
"""Stable Audio Open 1.0 text-to-audio pipeline (first-class).

All pipeline components are FastVideo-native ports of the official
`Stability-AI/stable-audio-tools` reference — no `from diffusers import`
or `from transformers import <ModelClass>` at runtime (see REVIEW item
30 in `.claude/skills/add-model/REVIEW.md`).

Stages:

    InputValidationStage
      → StableAudioConditioningStage    (T5 + NumberConditioner → cross-attn + global cond, with CFG)
      → StableAudioLatentPreparationStage  (initial Gaussian noise scaled by sigma_max)
      → StableAudioDenoisingStage       (k-diffusion `dpmpp-3m-sde` over the native DiT)
      → StableAudioDecodingStage        (first-class OobleckVAE → waveform)

Components (all FastVideo-native):
  * `fastvideo.models.dits.stable_audio.StableAudioDiT` — vendored from
    upstream `DiffusionTransformer + ContinuousTransformer`.
  * `fastvideo.models.encoders.stable_audio_conditioner.StableAudioMultiConditioner`
    — vendored from upstream `MultiConditioner + T5Conditioner + NumberConditioner`.
  * `fastvideo.models.vaes.oobleck.OobleckVAE` — vendored Oobleck (this
    branch's predecessor commit).

The sampler is `k_diffusion.sampling.sample_dpmpp_3m_sde` + `K.external.VDenoiser`
— pure math/sampling library (no model classes or weights), the same
function the official upstream `generate_diffusion_cond` calls.
"""
from __future__ import annotations

import os
from typing import Any

import torch

from fastvideo.distributed.parallel_state import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.basic.stable_audio.stages import (
    StableAudioConditioningStage,
    StableAudioDecodingStage,
    StableAudioDenoisingStage,
    StableAudioLatentPreparationStage,
)
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.stages import InputValidationStage

logger = init_logger(__name__)

_HF_REPO_ID = "stabilityai/stable-audio-open-1.0"
_OFFICIAL_WEIGHTS_FILE = "model.safetensors"


def _ensure_hf_token_env() -> None:
    for src in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_KEY"):
        v = os.environ.get(src)
        if v:
            os.environ.setdefault("HF_TOKEN", v)
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", v)
            return


class StableAudioPipeline(ComposedPipelineBase):
    """Stable Audio Open 1.0 pipeline (T2A + A2A + RePaint inpainting).

    A single class deliberately handles all three end-user modes — the
    DiT, conditioner, VAE, and sampler are byte-identical across modes,
    and the upstream `generate_diffusion_cond` API is also kwargs-driven
    on a single function. Mode is selected by the kwargs the caller
    passes to `generate_video()`:

      * **Text-to-audio (T2A)** — default. Just `prompt=...` + duration.
        Produces audio from pure noise.

            generator.generate_video(prompt="...", audio_end_in_s=6.0)

      * **Audio-to-audio variation (A2A)** — pass `init_audio` (a `[B, C,
        samples]` tensor at the model's sampling rate) plus an optional
        `init_noise_level` (= upstream `sigma_max`; lower = closer to
        the reference, higher = more freedom). The reference is encoded
        through the VAE, sampled stochastically, then renoised.

            generator.generate_video(
                prompt="...", init_audio=ref, init_noise_level=1.0
            )

      * **Inpainting / outpainting (RePaint blending)** — pass both
        `inpaint_audio` and `inpaint_mask` (1-D `[samples]` tensor, 1 =
        keep / 0 = regenerate, audio-domain at the model sample rate).
        At every k-diffusion step, the kept region is replaced with the
        renoised reference; only the unkept region is freely denoised.
        Works on the published `model_type=diffusion_cond` checkpoint
        (no inpaint-trained model required).

            generator.generate_video(
                prompt="...", inpaint_audio=ref, inpaint_mask=mask,
                audio_end_in_s=12.0,
            )

    See the four `examples/inference/basic/basic_stable_audio*.py`
    scripts for runnable user-story-shaped examples of each mode.

    The split into three pipeline classes was considered and rejected —
    the modes share weights / components / stage chain and the branches
    are ~30 LOC of stage-internal if/else. Triggers to revisit are in
    REVIEW.md (item 30 / 31): inpaint-trained checkpoint published,
    `WorkloadType` gains audio variants, or the per-mode forward
    signatures genuinely diverge.
    """

    _required_config_modules = [
        "vae",
        "transformer",
        "conditioner",
    ]

    def load_modules(
        self,
        fastvideo_args: FastVideoArgs,
        loaded_modules: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build native FV components from the official `model.safetensors`.

        We bypass FastVideo's standard component loader entirely — the
        SA repo isn't in Diffusers layout, so there's nothing to load
        per-subfolder; everything lives in the single
        `model.safetensors` checkpoint.
        """
        _ensure_hf_token_env()
        loaded_modules = loaded_modules or {}
        modules: dict[str, Any] = {}

        device = get_local_torch_device()
        precision = getattr(fastvideo_args.pipeline_config, "precision", "fp32")
        torch_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }.get(precision, torch.float32)

        # --- Locate weights. `model_path` is either the local cache dir
        #     (already downloaded) or we lazy-download via HF Hub.
        from huggingface_hub import hf_hub_download
        local_weights = os.path.join(self.model_path, _OFFICIAL_WEIGHTS_FILE)
        if os.path.isfile(local_weights):
            weights_path = local_weights
        else:
            weights_path = hf_hub_download(repo_id=_HF_REPO_ID,
                                           filename=_OFFICIAL_WEIGHTS_FILE,
                                           token=os.environ.get("HF_TOKEN"))

        from safetensors.torch import load_file
        logger.info("Loading official Stable Audio checkpoint from %s", weights_path)
        full_state = load_file(weights_path)

        # Mirror upstream `generate_diffusion_cond`'s deterministic-math
        # toggles — TF32 matmul / cuDNN nondeterminism amplifies through
        # the A2A renoise-then-denoise SDE trajectory. One-shot here so
        # we don't flip cuDNN's algorithm cache mid-run.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cudnn.benchmark = False

        # --- VAE: first-class FastVideo OobleckVAE via the lazy wrapper.
        if "vae" in loaded_modules:
            modules["vae"] = loaded_modules["vae"]
        else:
            from fastvideo.configs.models.vaes import OobleckVAEConfig
            from fastvideo.models.vaes.sa_audio import SAAudioVAEModel
            cfg = OobleckVAEConfig()
            local_vae_dir = os.path.join(self.model_path, "vae")
            if os.path.isdir(local_vae_dir):
                cfg.pretrained_path = local_vae_dir
                cfg.pretrained_subfolder = None
            else:
                cfg.pretrained_path = _HF_REPO_ID
                cfg.pretrained_subfolder = "vae"
            modules["vae"] = SAAudioVAEModel(cfg)

        # --- DiT: native StableAudioDiT loaded from the same checkpoint.
        if "transformer" in loaded_modules:
            modules["transformer"] = loaded_modules["transformer"]
        else:
            from fastvideo.models.dits.stable_audio import StableAudioDiT
            logger.info("Building native StableAudioDiT from official weights")
            modules["transformer"] = StableAudioDiT.from_official_state_dict(full_state)
            modules["transformer"] = modules["transformer"].to(device=device, dtype=torch_dtype).eval()

        # --- Conditioner: native MultiConditioner (T5 + Number x 2).
        if "conditioner" in loaded_modules:
            modules["conditioner"] = loaded_modules["conditioner"]
        else:
            from fastvideo.models.encoders.stable_audio_conditioner import (
                StableAudioMultiConditioner, )
            logger.info("Building native StableAudioMultiConditioner")
            modules["conditioner"] = StableAudioMultiConditioner.from_official_state_dict(full_state)
            modules["conditioner"] = modules["conditioner"].to(device=device).eval()
            # T5 lives outside the module's own parameters() — move it explicitly.
            if hasattr(modules["conditioner"].conditioners["prompt"], "model"):
                modules["conditioner"].conditioners["prompt"].model.to(device=device)

        return modules

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        pc = fastvideo_args.pipeline_config

        self.add_stage(stage_name="input_validation_stage", stage=InputValidationStage())

        self.add_stage(
            stage_name="conditioning_stage",
            stage=StableAudioConditioningStage(conditioner=self.get_module("conditioner")),
        )

        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=StableAudioLatentPreparationStage(
                io_channels=64,
                # Stable Audio uses fixed sample_size=2097152 (~47.5s) at the
                # latent input; the requested duration is sliced post-decode.
                sample_size=2097152,
                vae=self.get_module("vae"),
                sample_rate=pc.sampling_rate,
                audio_channels=pc.audio_channels,
            ),
        )

        self.add_stage(
            stage_name="denoising_stage",
            stage=StableAudioDenoisingStage(transformer=self.get_module("transformer")),
        )

        self.add_stage(
            stage_name="decoding_stage",
            stage=StableAudioDecodingStage(vae=self.get_module("vae")),
        )


EntryClass = StableAudioPipeline
