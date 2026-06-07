# SPDX-License-Identifier: Apache-2.0
"""FastVideo-native Cosmos3 video pipeline (T2V / I2V / T2I).

This replaces the earlier vllm-omni-derived skeleton with a native, stage-based
:class:`ComposedPipelineBase` pipeline that wires the framework-parity-verified
Cosmos3 components:

* tokenizer: Qwen2 ``Qwen2TokenizerFast`` + chat template (the only allowed
  third-party model-adjacent dependency; tokenizers are explicitly permitted),
* VAE: FastVideo-native ``AutoencoderKLWan`` (Wan2.2) via ``Cosmos3VAEConfig``;
  encode normalizes ``(mu - mean) * inv_std`` and decode denormalizes + clamps,
* sequence-packing: :func:`pack_cosmos3_video_sequence` (native, parity-tested),
* DiT: ``Cosmos3VFMTransformer`` (native, bit-identical to the framework),
* scheduler: FastVideo-native ``UniPCMultistepScheduler`` configured for pure
  flow matching (``flow_prediction`` + ``use_flow_sigmas``), numerically
  equivalent to the framework's ``FlowUniPCMultistepScheduler`` (parity-tested
  in ``test_cosmos3_scheduler_parity``).

The denoise/CFG glue is a faithful port of the framework's
``Cosmos3OmniDiffusersPipeline`` math (mirrored in the framework-equivalent
``diffusers_cosmos3.pipeline``): per UniPC timestep, run a SEQUENTIAL conditional
then unconditional pass (each repacks the sequence with the prompt / negative
prompt token ids, forwards the DiT, and zeros the prediction on conditioning
frames), then combine ``v = uncond + guidance * (cond - uncond)`` and take one
``scheduler.step(model_output=v, timestep, sample=latent)``. ``timestep_scale``
is applied to the per-token timesteps *inside* the DiT (its ``forward`` already
multiplies ``vision_timesteps * timestep_scale`` before the time embedder), so
the loop passes raw scheduler timesteps to the packer.

The pure denoise math lives in :class:`Cosmos3DenoiseEngine` and the free
function :func:`cosmos3_get_cfg_velocity` so it can be unit-/parity-tested
directly against the framework oracle without constructing the full pipeline.

No diffusers/transformers *model* classes are imported at runtime here; only the
Qwen2 tokenizer (loaded by the component loader) and the UniPC scheduler are
third-party, both explicitly allowed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_unipc_multistep import (
    UniPCMultistepScheduler, )
from fastvideo.pipelines.basic.cosmos3.sequence_packing import (
    Cosmos3SampleInputs,
    Cosmos3VisionItem,
    pack_cosmos3_video_sequence,
)
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase

logger = init_logger(__name__)

# System prompts, verbatim from the framework (diffusers_cosmos3.pipeline).
_SYSTEM_PROMPT_IMAGE = "You are a helpful assistant who will generate images from a give prompt."
_SYSTEM_PROMPT_VIDEO = "You are a helpful assistant who will generate videos from a give prompt."

# Default video negative prompt (framework / Cosmos quality prompt).
COSMOS3_VIDEO_NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
    "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
    "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, "
    "jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, "
    "fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. "
    "Overall, the video is of poor quality.")


# ===========================================================================
# Special-token resolution (Qwen2 chat tokenizer)
# ===========================================================================
def cosmos3_special_tokens(tokenizer: Any) -> dict[str, int]:
    """Resolve the Cosmos3 generation special tokens from a Qwen2 tokenizer.

    Mirrors the framework's ``llm_special_tokens``:
    ``start_of_generation=<|vision_start|>``, ``end_of_generation=<|vision_end|>``,
    ``eos_token_id=tokenizer.eos_token_id``.
    """
    return {
        "start_of_generation": int(tokenizer.convert_tokens_to_ids("<|vision_start|>")),
        "end_of_generation": int(tokenizer.convert_tokens_to_ids("<|vision_end|>")),
        "eos_token_id": int(tokenizer.eos_token_id),
    }


def cosmos3_tokenize_caption(
    tokenizer: Any,
    caption: str,
    *,
    is_video: bool = False,
    use_system_prompt: bool = False,
) -> list[int]:
    """Tokenize a caption with the Qwen2 chat template (framework-faithful).

    Optionally prepends an image/video system prompt; always adds the
    generation prompt and disables ``add_vision_id`` (matching the framework's
    ``tokenize_caption``).
    """
    conversations: list[dict[str, str]] = []
    if use_system_prompt:
        conversations.append({
            "role": "system",
            "content": _SYSTEM_PROMPT_VIDEO if is_video else _SYSTEM_PROMPT_IMAGE,
        })
    conversations.append({"role": "user", "content": caption})
    token_ids = tokenizer.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        add_vision_id=False,
    )
    return list(token_ids)


# ===========================================================================
# VAE encode/decode bridge (normalize / denormalize, matching the framework)
# ===========================================================================
@dataclass
class _VaeNorm:
    """Cached ``mean`` / ``inv_std`` for VAE (de)normalization."""

    mean: torch.Tensor  # [z_dim]
    inv_std: torch.Tensor  # [z_dim]

    @classmethod
    def from_vae(cls, vae: Any, dtype: torch.dtype) -> _VaeNorm:
        mean = torch.tensor(list(vae.config.latents_mean), dtype=dtype)
        std = torch.tensor(list(vae.config.latents_std), dtype=dtype)
        return cls(mean=mean, inv_std=1.0 / std)


def cosmos3_vae_encode(vae: Any, video: torch.Tensor, norm: _VaeNorm) -> torch.Tensor:
    """Encode ``[B, 3, T, H, W]`` pixels in [-1, 1] to NORMALIZED latents.

    Matches the framework ``DiffusersWan22VAE.encode``: take the posterior mode
    and apply ``(mu - mean) * inv_std``. FastVideo's ``AutoencoderKLWan.encode``
    returns a ``DiagonalGaussianDistribution``; we read ``.mode()``.
    """
    in_dtype = video.dtype
    device = video.device
    mean = norm.mean.to(device=device, dtype=in_dtype).view(1, -1, 1, 1, 1)
    inv_std = norm.inv_std.to(device=device, dtype=in_dtype).view(1, -1, 1, 1, 1)
    raw_mu = vae.encode(video).mode()
    return ((raw_mu - mean) * inv_std).to(in_dtype)


def cosmos3_vae_decode(vae: Any, latents: torch.Tensor, norm: _VaeNorm) -> torch.Tensor:
    """Decode NORMALIZED latents ``[B, z, T, H, W]`` to pixels ``[B, 3, T, H, W]``.

    Inverts the normalization (``z / inv_std + mean``) then calls
    ``vae.decode`` (which already clamps to [-1, 1]).
    """
    in_dtype = latents.dtype
    device = latents.device
    mean = norm.mean.to(device=device, dtype=in_dtype).view(1, -1, 1, 1, 1)
    inv_std = norm.inv_std.to(device=device, dtype=in_dtype).view(1, -1, 1, 1, 1)
    z_raw = latents / inv_std + mean
    out = vae.decode(z_raw)
    if isinstance(out, tuple):
        out = out[0]
    if hasattr(out, "sample"):
        out = out.sample
    return out.to(in_dtype)


# ===========================================================================
# Per-vision-item packing geometry
# ===========================================================================
@dataclass
class Cosmos3VisionSpec:
    """Geometry + conditioning for one vision item in a denoise run.

    Args:
        clean_latent: VAE-encoded conditioning latent ``[C, T, H, W]`` used to
            keep condition frames clean (I2V / T2I). May be ``None`` for T2V.
        condition_frame_indexes: Latent-frame indices kept clean.
        shape: ``(C, T, H, W)`` of the latent for this item.
    """

    shape: tuple[int, int, int, int]
    condition_frame_indexes: list[int]
    clean_latent: torch.Tensor | None = None

    @property
    def numel(self) -> int:
        return int(math.prod(self.shape))


# ===========================================================================
# Pure denoise/CFG math (parity oracle target)
# ===========================================================================
def _split_flat_latent(flat: torch.Tensor, specs: list[Cosmos3VisionSpec]) -> list[torch.Tensor]:
    """Split a flat latent vector into per-vision-item ``[C, T, H, W]`` tensors."""
    out: list[torch.Tensor] = []
    offset = 0
    for spec in specs:
        out.append(flat[offset:offset + spec.numel].reshape(spec.shape))
        offset += spec.numel
    return out


def cosmos3_get_cfg_velocity(
    *,
    transformer: Any,
    flat_latent: torch.Tensor,
    timestep: torch.Tensor,
    guidance: float,
    specs: list[Cosmos3VisionSpec],
    cond_token_ids: list[int],
    uncond_token_ids: list[int],
    special_tokens: dict[str, int],
    latent_patch_size: int,
    temporal_modality_margin: int,
    reset_spatial_ids: bool,
    enable_fps_modulation: bool,
    base_fps: float,
    temporal_compression_factor: int,
    include_end_of_generation_token: bool = False,
    fps_per_item: list[float] | None = None,
    normalize_cfg: bool = False,
) -> torch.Tensor:
    """Sequential-CFG velocity for one denoise step (framework math).

    Replicates the framework ``get_cfg_velocity``:

    1. split ``flat_latent`` into per-vision-item ``[C, T, H, W]`` latents,
    2. run a conditional pass (prompt tokens) and an unconditional pass
       (negative-prompt tokens); each repacks via
       :func:`pack_cosmos3_video_sequence`, forwards the DiT to obtain
       ``preds_vision`` (a list of ``[1, C, T, H, W]`` unpatchified noisy-frame
       predictions), and zeros the prediction on conditioning frames
       (``pred * (1 - condition_mask)``),
    3. combine ``v = uncond + guidance * (cond - uncond)`` (optionally
       norm-rescaled), returned flattened to match ``flat_latent``.

    ``timestep`` is a scalar tensor (raw scheduler timestep); ``timestep_scale``
    is applied inside the DiT, so it is passed through unscaled here.
    """
    assert timestep.numel() == 1, "timestep must be a scalar"
    timestep_value = float(timestep.reshape(()).item())

    noise_x_vision = _split_flat_latent(flat_latent, specs)

    def _run(token_ids: list[int]) -> torch.Tensor:
        samples = [
            Cosmos3SampleInputs(
                text_ids=list(token_ids),
                vision=Cosmos3VisionItem(
                    latent=latent,
                    condition_frame_indexes=list(spec.condition_frame_indexes),
                    fps=(fps_per_item[i] if fps_per_item is not None else None),
                ),
                timestep=timestep_value,
            ) for i, (latent, spec) in enumerate(zip(noise_x_vision, specs, strict=False))
        ]
        packed = pack_cosmos3_video_sequence(
            samples,
            special_tokens,
            latent_patch_size=latent_patch_size,
            include_end_of_generation_token=include_end_of_generation_token,
            temporal_modality_margin=temporal_modality_margin,
            reset_spatial_ids=reset_spatial_ids,
            enable_fps_modulation=enable_fps_modulation,
            base_fps=base_fps,
            temporal_compression_factor=temporal_compression_factor,
        )
        device = next(transformer.parameters()).device
        out = transformer(**packed.to_dit_kwargs(device=device))
        preds = out.get("preds_vision")
        if preds is None:
            # Fully-conditioned clip: no noisy frames, zero velocity.
            return torch.zeros_like(flat_latent)
        # Zero the prediction on conditioning frames, per item.
        velocity_items: list[torch.Tensor] = []
        for pred, cond_mask in zip(preds, packed.vision_condition_mask, strict=False):
            pred = pred.squeeze(0) if pred.dim() == 5 else pred  # [C, T, H, W]
            keep = (1.0 - cond_mask).to(dtype=pred.dtype, device=pred.device)  # [T,1,1]
            if keep.sum() > 0:
                velocity_items.append(pred * keep)
            else:
                velocity_items.append(torch.zeros_like(pred))
        return torch.cat([v.reshape(-1) for v in velocity_items])

    cond_v = _run(cond_token_ids)
    uncond_v = _run(uncond_token_ids)
    v_pred = uncond_v + guidance * (cond_v - uncond_v)
    if normalize_cfg:
        scale = (torch.norm(cond_v) / (torch.norm(v_pred) + 1e-8)).clamp(min=0.0, max=1.0)
        v_pred = v_pred * scale
    return v_pred


class Cosmos3DenoiseEngine:
    """Stateless denoise driver tying CFG velocity to UniPC stepping.

    Holds the transformer + scheduler + packing constants and runs the full
    UniPC denoise loop. Kept separate from the pipeline so it can be exercised
    in isolation (smoke + parity tests) with stub or real components.
    """

    def __init__(
        self,
        *,
        transformer: Any,
        scheduler: Any,
        special_tokens: dict[str, int],
        latent_patch_size: int,
        temporal_modality_margin: int,
        reset_spatial_ids: bool,
        enable_fps_modulation: bool,
        base_fps: float,
        temporal_compression_factor: int,
        include_end_of_generation_token: bool = False,
    ) -> None:
        self.transformer = transformer
        self.scheduler = scheduler
        self.special_tokens = special_tokens
        self.latent_patch_size = latent_patch_size
        self.temporal_modality_margin = temporal_modality_margin
        self.reset_spatial_ids = reset_spatial_ids
        self.enable_fps_modulation = enable_fps_modulation
        self.base_fps = base_fps
        self.temporal_compression_factor = temporal_compression_factor
        self.include_end_of_generation_token = include_end_of_generation_token

    def velocity(
        self,
        *,
        flat_latent: torch.Tensor,
        timestep: torch.Tensor,
        guidance: float,
        specs: list[Cosmos3VisionSpec],
        cond_token_ids: list[int],
        uncond_token_ids: list[int],
        fps_per_item: list[float] | None = None,
    ) -> torch.Tensor:
        return cosmos3_get_cfg_velocity(
            transformer=self.transformer,
            flat_latent=flat_latent,
            timestep=timestep,
            guidance=guidance,
            specs=specs,
            cond_token_ids=cond_token_ids,
            uncond_token_ids=uncond_token_ids,
            special_tokens=self.special_tokens,
            latent_patch_size=self.latent_patch_size,
            temporal_modality_margin=self.temporal_modality_margin,
            reset_spatial_ids=self.reset_spatial_ids,
            enable_fps_modulation=self.enable_fps_modulation,
            base_fps=self.base_fps,
            temporal_compression_factor=self.temporal_compression_factor,
            include_end_of_generation_token=self.include_end_of_generation_token,
            fps_per_item=fps_per_item,
        )

    def denoise(
        self,
        *,
        flat_latent: torch.Tensor,
        timesteps: torch.Tensor,
        guidance: float,
        specs: list[Cosmos3VisionSpec],
        cond_token_ids: list[int],
        uncond_token_ids: list[int],
        fps_per_item: list[float] | None = None,
        progress_bar: Any | None = None,
    ) -> torch.Tensor:
        """Run the full UniPC denoise loop, returning the final flat latent.

        For each timestep: compute the sequential-CFG velocity, then
        ``scheduler.step(model_output=v, timestep, sample=latent.unsqueeze(0))``
        (the framework steps with a leading batch axis), squeezing back to flat.
        """
        latent = flat_latent
        iterator = progress_bar(timesteps) if progress_bar is not None else timesteps
        for t in iterator:
            v_pred = self.velocity(
                flat_latent=latent,
                timestep=t.reshape(1),
                guidance=guidance,
                specs=specs,
                cond_token_ids=cond_token_ids,
                uncond_token_ids=uncond_token_ids,
                fps_per_item=fps_per_item,
            )
            stepped = self.scheduler.step(
                model_output=v_pred,
                timestep=t,
                sample=latent.unsqueeze(0),
                return_dict=False,
            )[0]
            latent = stepped.squeeze(0)
        return latent


# ===========================================================================
# Pipeline (ComposedPipelineBase)
# ===========================================================================
class Cosmos3OmniDiffusersPipeline(ComposedPipelineBase):
    """Cosmos3 video generation pipeline (T2V / I2V / T2I).

    Stage-based ``ComposedPipelineBase`` pipeline. The required modules
    (``transformer`` / ``vae`` / ``scheduler`` / ``text_tokenizer``) are loaded
    from the ``nvidia/Cosmos3-Nano`` checkpoint by the component loader. The
    class name matches the checkpoint ``model_index.json`` ``_class_name`` so
    the registry resolves it directly.

    The denoise/CFG/VAE math is delegated to module-level helpers
    (:func:`cosmos3_get_cfg_velocity`, :class:`Cosmos3DenoiseEngine`,
    :func:`cosmos3_vae_encode` / :func:`cosmos3_vae_decode`) which are
    framework-parity tested in ``tests/local_tests/cosmos3``.
    """

    is_video_pipeline = True
    # ``vision_encoder`` / ``sound_tokenizer`` ship in the checkpoint but the
    # video path does not need them; they are intentionally omitted here.
    _required_config_modules = ["text_tokenizer", "vae", "transformer", "scheduler"]

    # Engine-init flow_shift (T2V/I2V); T2I overrides to 3.0 per request.
    _engine_init_flow_shift: float = 1.0
    # Class-attribute defaults so ``__new__``-based unit tests can read these
    # before ``initialize_pipeline`` runs.
    scheduler: Any = None
    _base_scheduler_config: Any = None
    _current_flow_shift: float | None = None

    @staticmethod
    def _flow_scheduler_config(config: Any) -> dict[str, Any]:
        """Coerce a loaded UniPC config to the framework's flow-matching setup.

        The checkpoint ``scheduler_config.json`` carries diffusers-style fields
        (``use_karras_sigmas=True``, ``sigma_min``/``sigma_max``, beta schedule)
        that do not describe the framework sampler. The framework uses
        ``FlowUniPCMultistepScheduler`` (pure flow matching: ``shift`` +
        ``num_train_timesteps`` only). FastVideo's vendored UniPC checks
        ``use_karras_sigmas`` *before* ``use_flow_sigmas``, so leaving karras on
        builds diffusion-style sigmas and the denoise diverges to NaN. Force the
        flow config here (parity-verified in ``test_cosmos3_scheduler_parity``).
        """
        cfg = dict(config)
        cfg.update(
            use_karras_sigmas=False,
            use_exponential_sigmas=False,
            use_beta_sigmas=False,
            use_flow_sigmas=True,
            prediction_type="flow_prediction",
            predict_x0=True,
            final_sigmas_type="zero",
        )
        return cfg

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs) -> None:
        """Bind the loaded scheduler + snapshot its config so per-request
        flow_shift rebuilds are cheap and the engine-init shift is applied."""
        pipeline_config = fastvideo_args.pipeline_config
        engine_shift = getattr(pipeline_config, "flow_shift", None)
        if engine_shift is not None:
            self._engine_init_flow_shift = float(engine_shift)
        scheduler = self.get_module("scheduler")
        if scheduler is not None:
            # Rebuild from a flow-coerced config so the runtime scheduler matches
            # the framework sampler (the loaded checkpoint config is diffusers-style).
            flow_config = self._flow_scheduler_config(scheduler.config)
            self.scheduler = UniPCMultistepScheduler.from_config(flow_config)
            if isinstance(self.modules, dict):
                self.modules["scheduler"] = self.scheduler
            self._base_scheduler_config = self.scheduler.config
            self._current_flow_shift = float(getattr(self.scheduler.config, "flow_shift", 1.0))

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        """Wire the Cosmos3 stages.

        The whole text->latent->denoise->decode flow is custom (sequential CFG
        with per-pass repacking), so a single :class:`Cosmos3DenoisingStage`
        owns it. ``InputValidationStage`` runs first for the standard checks.
        """
        from fastvideo.pipelines.stages import InputValidationStage
        from fastvideo.pipelines.stages.cosmos3_stages import Cosmos3DenoisingStage

        self.add_stage(stage_name="input_validation_stage", stage=InputValidationStage())
        self.add_stage(
            stage_name="denoising_stage",
            stage=Cosmos3DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                vae=self.get_module("vae"),
                tokenizer=self.get_module("text_tokenizer"),
                pipeline=self,
            ),
        )

    # -- Scheduler control --------------------------------------------------

    def _set_flow_shift(self, target_shift: float) -> None:
        """Set UniPC ``flow_shift`` to ``target_shift``.

        Lazily builds a default UniPC scheduler when called before
        ``initialize_pipeline`` (e.g. the ``__new__``-based scheduler-parity
        tests); otherwise rebuilds from the snapshotted base config only when
        the target differs from the current shift.
        """
        target = float(target_shift)
        base_config = self._base_scheduler_config
        if base_config is None:
            self.scheduler = UniPCMultistepScheduler(
                num_train_timesteps=1000,
                solver_order=2,
                prediction_type="flow_prediction",
                use_flow_sigmas=True,
                flow_shift=target,
            )
            self._base_scheduler_config = self.scheduler.config
            self._current_flow_shift = target
            return
        current = self._current_flow_shift
        if current is not None and target == float(current):
            return
        self.scheduler = UniPCMultistepScheduler.from_config(base_config, flow_shift=target)
        if isinstance(self.modules, dict):
            self.modules["scheduler"] = self.scheduler
        self._current_flow_shift = target

    # -- Tokenization -------------------------------------------------------

    def tokenize_caption(self, caption: str, *, is_video: bool = False, use_system_prompt: bool = False) -> list[int]:
        return cosmos3_tokenize_caption(self.get_module("text_tokenizer"),
                                        caption,
                                        is_video=is_video,
                                        use_system_prompt=use_system_prompt)


# Entry point for the pipeline registry. The class name matches the checkpoint
# ``model_index.json`` ``_class_name`` so ``resolve_pipeline_cls`` finds it.
EntryClass = Cosmos3OmniDiffusersPipeline
