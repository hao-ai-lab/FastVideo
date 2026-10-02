# SPDX-License-Identifier: Apache-2.0
"""One-forward joint audio/video denoising for MiniMax H3."""

from __future__ import annotations

import contextlib
from typing import Any

import torch

from fastvideo.attention.selector import component_attention_backend, get_attn_backend
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.hooks.activation_trace import trace_step
from fastvideo.layers.pdd import PDDModalitySchedule, PDDSamplingPlan, build_pdd_sampling_plan
from fastvideo.logger import init_logger
from fastvideo.profiler import nvtx_range, profiler_region
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MiniMaxH3PackedLayout,
    build_row_timesteps,
)
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import MINIMAX_H3_LAYOUT_KEY
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.utils import get_compute_dtype

logger = init_logger(__name__)


def scale_pdd_initial_noise(video_rows: torch.Tensor, audio_rows: torch.Tensor, layout: MiniMaxH3PackedLayout,
                            plan: PDDSamplingPlan) -> None:
    """Start PDD targets at the first node's noise level (``0.999 * eps``), as the students are trained.

    Latent preparation draws unit noise; conditioning rows are left untouched.
    """
    video_rows[layout.num_condition_video_rows:] *= float(plan.node_sigmas["video"][0])
    audio_rows[layout.num_condition_audio_rows:] *= float(plan.node_sigmas["audio"][0])


def _h3_vsa_metadata_builder(transformer: Any, fastvideo_args: FastVideoArgs) -> Any:
    """Builder instance when the transformer resolved to VSA-H3, else None.

    Resolves through the same selector record the attention layers used
    (``component_attention_backend``) instead of introspecting module
    internals, mirroring the generic DenoisingStage.
    """
    dit_config = fastvideo_args.pipeline_config.dit_config
    backend = get_attn_backend(
        head_size=dit_config.attention_head_dim,
        dtype=get_compute_dtype(),
        supported_attention_backends=dit_config._supported_attention_backends,
        requested=component_attention_backend(transformer),
    )
    if backend.get_name() != "VIDEO_SPARSE_ATTN_H3":
        return None
    return backend.get_builder_cls()()


def _h3_vsa_single_region_segments(
    layout: MiniMaxH3PackedLayout,
    patch_size: tuple[int, int, int],
) -> tuple[int | tuple[int, int, int], ...]:
    """VSA-H3 packed segments with the generated video as the only sparse region.

    Returns ``(text rows, condition rows, audio rows, target video latent
    shape)`` for ``MiniMaxH3VSAMetadataBuilder.build``, validated against the
    layout; every conditioning row stays dense.
    """
    n_text = int(layout.text_indices.numel())
    n_cond = int(layout.num_condition_video_rows)
    n_audio = int(layout.audio_indices.numel())
    n_video = ((layout.num_video_latent_frames // patch_size[0]) * (layout.latent_height // patch_size[1]) *
               (layout.latent_width // patch_size[2]))
    if n_text + n_cond + n_audio + n_video != layout.sequence_length:
        raise ValueError("VSA-H3 supports the standard [text|cond|audio|video] packing only; "
                         f"segments ({n_text}, {n_cond}, {n_audio}) + video {n_video} do not sum to "
                         f"sequence length {layout.sequence_length}.")
    return n_text, n_cond, n_audio, (layout.num_video_latent_frames, layout.latent_height, layout.latent_width)


def _h3_vsa_ref2va_segments(
    layout: MiniMaxH3PackedLayout,
    patch_size: tuple[int, int, int],
) -> tuple[int | tuple[int, int, int], ...]:
    """VSA-H3 packed segments of a Ref2VA layout with every reference video as a sparse region.

    Returns the segments for ``MiniMaxH3VSAMetadataBuilder.build`` in true
    packed order, ``[text | reference spans... | target audio | target
    video]``: each reference VIDEO is a region (its latent shape); text,
    every reference's audio rows, image references, and the target audio are
    dense row counts; the target video is the last region.
    """
    if not layout.reference_segments:
        raise ValueError("Ref2VA VSA sparsification needs per-reference spans on the layout; "
                         "this layout carries none (T2VA/FL2VA packing).")
    cursor = int(layout.text_indices.numel())
    segments: list[int | tuple[int, int, int]] = [cursor]
    for kind, rows, latent_shape in layout.reference_segments:
        rows = int(rows)
        if kind == "video":
            segments.append((int(latent_shape[0]), int(latent_shape[1]), int(latent_shape[2])))
        else:
            # audio rows and single-frame image latents stay dense
            segments.append(rows)
        cursor += rows
    n_target_audio = int(layout.audio_indices.numel()) - int(layout.num_condition_audio_rows)
    segments.append(n_target_audio)
    cursor += n_target_audio
    segments.append((layout.num_video_latent_frames, layout.latent_height, layout.latent_width))
    n_target_video = ((layout.num_video_latent_frames // patch_size[0]) * (layout.latent_height // patch_size[1]) *
                      (layout.latent_width // patch_size[2]))
    if cursor + n_target_video != layout.sequence_length:
        raise ValueError(f"Ref2VA reference spans {layout.reference_segments} + target segments do not tile the "
                         f"packed sequence: reached row {cursor} + video {n_target_video} != "
                         f"{layout.sequence_length}.")
    return tuple(segments)


class MiniMaxH3DenoisingStage(PipelineStage):
    """Build both schedules and denoise both modalities in one transformer call."""

    performance_component_metric = "dit_time_s"

    def __init__(self, transformer: Any, scheduler: Any, audio_scheduler: Any) -> None:
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler
        self.audio_scheduler = audio_scheduler

    def _set_dmd_schedule(self, steps: list[int], grid_points: int, device: torch.device) -> None:
        """Run the trained rungs, shifting the shared noise clock once per modality."""
        if (not steps or any(type(step) is not int or not 0 < step <= 1000 for step in steps)
                or any(left <= right for left, right in zip(steps, steps[1:], strict=False))):
            raise ValueError("MiniMax-H3 DMD rungs must be strictly decreasing integers in (0, 1000].")
        if grid_points != len(steps) + 1:
            raise ValueError("MiniMax-H3 num_inference_steps counts sigma-grid points: "
                             f"{len(steps)} DMD forwards require {len(steps) + 1} grid points, got {grid_points}.")
        base = torch.tensor([step / 1000.0 for step in steps] + [0.0], dtype=torch.float32)
        for scheduler in (self.scheduler, self.audio_scheduler):
            shift = float(scheduler.shift)
            sigmas = shift * base / (1 + (shift - 1) * base)
            # Explicit sigmas are already shifted. Scheduler timesteps are H3
            # clean time (1 - sigma); passing integer rungs to step() is wrong.
            scheduler.set_timesteps(sigmas=sigmas, device=device)

    def _pdd_sampling_plan(self, fastvideo_args: FastVideoArgs, device: torch.device) -> PDDSamplingPlan | None:
        """The fused-block plan of a PDD checkpoint, or None for any other checkpoint.

        ``pdd_step_indices`` is the checkpoint's trained partition, which
        ``MiniMaxH3PipelineConfig.resolve_checkpoint_settings`` read and checked
        against the transformer's ``pdd_steps``. Each modality's node sigmas
        come from its own scheduler shift on the shared base clock.
        """
        indices = getattr(fastvideo_args.pipeline_config, "pdd_step_indices", None)
        if indices is None:
            return None
        schedules = {
            "video": PDDModalitySchedule(shift=float(self.scheduler.shift)),
            "audio": PDDModalitySchedule(shift=float(self.audio_scheduler.shift)),
        }
        return build_pdd_sampling_plan(indices, schedules, device=device)

    def _pdd_head_fusion(self, plan: PDDSamplingPlan | None, index: int) -> contextlib.AbstractContextManager[None]:
        """Fuse runtime step *index*'s block of heads inside both widened output projections."""
        if plan is None:
            return contextlib.nullcontext()
        block_start, block_end = plan.block(index)
        return self.transformer.fuse_pdd_block(block_start, block_end, plan.integration_weights, torch.float32)

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("layout", batch.extra.get(MINIMAX_H3_LAYOUT_KEY), V.not_none)
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_of_tensors_dims(3))
        result.add_check("latents", batch.latents, V.with_dims(2))
        result.add_check("audio_latents", batch.audio_latents, V.with_dims(2))
        result.add_check("num_inference_steps", batch.num_inference_steps, V.positive_int)
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, V.with_dims(2))
        result.add_check("audio_latents", batch.audio_latents, V.with_dims(2))
        result.add_check("timesteps", batch.timesteps, V.with_dims(1))
        result.add_check("step_index", batch.step_index, V.non_negative_int)
        return result

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        """Denoise the packed H3 video and audio streams over one shared schedule."""
        layout = batch.extra.get(MINIMAX_H3_LAYOUT_KEY)
        if not isinstance(layout, MiniMaxH3PackedLayout):
            raise ValueError("MiniMax-H3 packed layout is missing before denoising.")
        if not batch.prompt_embeds or batch.latents is None or batch.audio_latents is None:
            raise ValueError("MiniMax-H3 conditioning and packed latents must precede denoising.")

        full_cpu_offload = (fastvideo_args.dit_cpu_offload and not fastvideo_args.dit_layerwise_offload
                            and not fastvideo_args.use_fsdp_inference)
        device = get_local_torch_device()

        dmd_steps = fastvideo_args.pipeline_config.dmd_denoising_steps
        pdd_plan = self._pdd_sampling_plan(fastvideo_args, device)
        if pdd_plan is not None:
            # One runtime step fuses one block of heads into their weighted
            # mean velocity; the ordinary Euler step over the block's two node
            # sigmas then applies the block's total increment per modality.
            self.scheduler.set_timesteps(sigmas=pdd_plan.node_sigmas["video"].to(torch.float32), device=device)
            self.audio_scheduler.set_timesteps(sigmas=pdd_plan.node_sigmas["audio"].to(torch.float32), device=device)
            logger.info("MiniMax-H3 PDD denoising: %d fused block(s) %s over a %d-interval fine grid.",
                        pdd_plan.num_steps, pdd_plan.block_sizes, pdd_plan.pdd_steps)
        elif dmd_steps is None:
            self.scheduler.set_timesteps(batch.num_inference_steps, device=device)
            self.audio_scheduler.set_timesteps(batch.num_inference_steps, device=device)
        else:
            self._set_dmd_schedule(dmd_steps, batch.num_inference_steps, device)
        video_timesteps = self.scheduler.timesteps
        audio_timesteps = self.audio_scheduler.timesteps
        if video_timesteps is None or audio_timesteps is None:
            raise ValueError("MiniMax-H3 schedulers did not produce timesteps.")
        if len(video_timesteps) != len(audio_timesteps):
            raise ValueError("MiniMax-H3 video and audio schedules must have the same number of intervals.")

        row_timestep_plan = []
        for video_timestep, audio_timestep in zip(video_timesteps, audio_timesteps, strict=True):
            video_value = float(video_timestep.item())
            audio_value = float(audio_timestep.item())
            unique, inverse = build_row_timesteps(
                layout,
                video_timestep=video_value,
                audio_timestep=audio_value,
                condition_video_timestep=max(video_value, MINIMAX_H3_KEYFRAME_NOISE_AUG),
                condition_audio_timestep=1.0,
            )
            row_timestep_plan.append((unique.to(device), inverse.to(device)))
        batch.timesteps = video_timesteps

        position_ids = layout.position_ids.to(device)
        token_tags = layout.token_tags.to(device)
        video_indices = layout.video_indices.to(device)
        audio_indices = layout.audio_indices.to(device)
        text_indices = layout.text_indices.to(device)
        prompt_embeds = batch.prompt_embeds[0].to(device)

        vsa_metadata_builder = _h3_vsa_metadata_builder(self.transformer, fastvideo_args)
        if vsa_metadata_builder is not None:
            vsa_patch_size = fastvideo_args.pipeline_config.dit_config.patch_size
            # Per-request knobs (sweeps flip these between generate_video calls
            # without respawning workers); mode None defers to the env default.
            vsa_mode = batch.extra.get("vsa_mode", "exempt")
            if vsa_mode not in ("exempt", "compete"):
                raise ValueError(f"vsa_mode must be 'exempt' or 'compete', got {vsa_mode!r}.")
            vsa_exempt = vsa_mode == "exempt"
            # PDD checkpoints set a reference keep rate: each reference video is
            # then its own sparse region keeping that fraction of its tiles.
            # Without one, every conditioning row stays dense.
            vsa_ref_keep_rate = getattr(fastvideo_args.pipeline_config, "vsa_ref_keep_rate", None)
            if vsa_ref_keep_rate is None:
                vsa_packed_segments = _h3_vsa_single_region_segments(layout, vsa_patch_size)
            else:
                vsa_packed_segments = _h3_vsa_ref2va_segments(layout, vsa_patch_size)
            vsa_dense_layers = tuple(batch.extra.get("vsa_dense_layers", ()))
            vsa_dense_first_n = int(batch.extra.get("vsa_dense_first_n_steps", 0))
            # Run-level tile geometry (256 default, 64 = native Triton path,
            # 128 = sm_100a CUDA), plumbed like the run-level sparsity; the
            # builder validates the value against VSA_H3_TILE_SHAPES.
            vsa_tile_size = int(fastvideo_args.VSA_tile_size)
            if vsa_ref_keep_rate is not None:
                num_reference_regions = sum(isinstance(segment, tuple) for segment in vsa_packed_segments) - 1
                run_sparsity = float(batch.VSA_sparsity)
                if run_sparsity <= 0.0:
                    effect = "VSA_sparsity=0 keeps every region dense"
                else:
                    effect = (f"each reference keeps {vsa_ref_keep_rate:g} of its tiles and the target keeps "
                              f"{1.0 - run_sparsity:g}")
                logger.info("MiniMax-H3 VSA-H3: %d reference video region(s); %s; %d-token tiles.",
                            num_reference_regions, effect, vsa_tile_size)

        try:
            if full_cpu_offload:
                self.transformer.to(device)
                batch.latents = batch.latents.to(device)
                batch.audio_latents = batch.audio_latents.to(device)
            if pdd_plan is not None:
                scale_pdd_initial_noise(batch.latents, batch.audio_latents, layout, pdd_plan)

            # The stage range groups the complete denoising loop while the
            # indexed model ranges retain timing detail for every H3 block.
            with profiler_region("inference_denoising"), nvtx_range("minimax_h3.dit"):
                for index, (video_timestep,
                            audio_timestep) in enumerate(zip(video_timesteps, audio_timesteps, strict=True)):
                    unique_timesteps, timestep_indices = row_timestep_plan[index]
                    attn_metadata = None
                    if vsa_metadata_builder is not None:
                        # Optional schedule: run the first N steps dense (sparsity 0
                        # selects every tile — parity-proven ≡ dense ≤2e-4); early
                        # steps set global structure and are the most damage-prone.
                        vsa_sparsity = 0.0 if index < vsa_dense_first_n else float(batch.VSA_sparsity)
                        attn_metadata = vsa_metadata_builder.build(
                            current_timestep=index,
                            patch_size=vsa_patch_size,
                            VSA_sparsity=vsa_sparsity,
                            packed_segments=vsa_packed_segments,
                            device=device,
                            exempt=vsa_exempt,
                            dense_layers=vsa_dense_layers,
                            tile_size=vsa_tile_size,
                            ref_keep_rate=vsa_ref_keep_rate,
                        )
                    # Under torch.compile(mode="reduce-overhead") each denoising
                    # step must be marked, or cudagraph trees flag cross-step
                    # reuse of pooled outputs as "accessing tensor output of
                    # CUDAGraphs that has been overwritten" (surfaces at sp=1;
                    # sp>1 is masked by collective-induced graph breaks).
                    torch.compiler.cudagraph_mark_step_begin()
                    with self._pdd_head_fusion(pdd_plan, index), trace_step(index), set_forward_context(
                            current_timestep=index,
                            attn_metadata=attn_metadata,
                            forward_batch=batch,
                    ):
                        video_velocity, audio_velocity = self.transformer(
                            hidden_states=batch.latents[None],
                            audio_hidden_states=batch.audio_latents[None],
                            encoder_hidden_states=prompt_embeds,
                            timestep=unique_timesteps,
                            timestep_indices=timestep_indices,
                            token_tags=token_tags,
                            position_ids=position_ids,
                            video_indices=video_indices,
                            audio_indices=audio_indices,
                            text_indices=text_indices,
                        )

                    video_start = layout.num_condition_video_rows
                    audio_start = layout.num_condition_audio_rows
                    batch.latents[video_start:] = self.scheduler.step(
                        video_velocity[0, video_start:].float(),
                        video_timestep,
                        batch.latents[video_start:],
                        return_dict=False,
                    )[0]
                    batch.audio_latents[audio_start:] = self.audio_scheduler.step(
                        audio_velocity[0, audio_start:].float(),
                        audio_timestep,
                        batch.audio_latents[audio_start:],
                        return_dict=False,
                    )[0]
                    batch.step_index = index
                    batch.timestep = video_timestep
        finally:
            if bool(getattr(fastvideo_args, "dit_layerwise_offload", False)):
                manager = getattr(self.transformer, "_layerwise_offload_manager", None)
                if manager is not None and getattr(manager, "enabled", False):
                    manager.release_all()
            if full_cpu_offload:
                self.transformer.to("cpu")
        return batch


__all__ = ["MiniMaxH3DenoisingStage", "scale_pdd_initial_noise"]
