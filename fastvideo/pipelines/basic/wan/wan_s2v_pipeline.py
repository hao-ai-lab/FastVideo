# SPDX-License-Identifier: Apache-2.0
"""Wan2.2-S2V speech-to-video pipeline.

Audio-driven generation: a reference image fixes who the subject is, the prompt
sets the scene, and the speech track drives the motion. The audio is encoded
once up front (it is the same for every denoising step) and cross-attended to
inside 12 of the transformer's 40 blocks.

Long requests run as several clips, the way the official runner does: each
clip denoises fresh noise conditioned on the same reference image, its slice of
the audio, and the previous clip's last ``motion_frames`` pixels re-encoded as
motion latents. The clips are concatenated and cut to ``num_frames``.
"""
import numpy as np
import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import FlowUniPCMultistepScheduler
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.lora_pipeline import LoRAPipeline
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

# isort: off
from fastvideo.pipelines.basic.wan.s2v_stages import (EXTRA_AUDIO_FRAMES, EXTRA_INFER_FRAMES, EXTRA_MOTION_LATENTS,
                                                      WARMUP_FRAMES, S2VDecodingStage, S2VRefImageEncodingStage,
                                                      plan_clips)
from fastvideo.pipelines.basic.wan.stages.s2v_denoising import WanS2VDenoisingStage
from fastvideo.pipelines.stages import (AudioEncodingStage, ConditioningStage, InputValidationStage,
                                        LatentPreparationStage, TextEncodingStage, TimestepPreparationStage)
# isort: on

logger = init_logger(__name__)

# Stages that run once per clip, in order. Everything before them runs once.
_PER_CLIP_STAGES = ("timestep_preparation_stage", "latent_preparation_stage", "denoising_stage", "decoding_stage")


def roll_motion_history(history: torch.Tensor | None, frames: torch.Tensor, motion_frames: int) -> torch.Tensor:
    """Shift the newest ``frames`` ([B, 3, T, H, W]) into a ``motion_frames``-long history.

    The official runner starts from a zero history and slides the newest clip
    in, so a clip shorter than ``motion_frames`` keeps older frames in front.
    """
    if history is None:
        history = frames.new_zeros(frames.shape[0], frames.shape[1], motion_frames, *frames.shape[3:])
    keep = min(motion_frames, frames.shape[2])
    return torch.cat([history[:, :, keep:], frames[:, :, -keep:]], dim=2)


def align_audio(audio: np.ndarray, sample_rate: int, fps: int, num_frames: int) -> np.ndarray:
    """Cut the source track to the generated span.

    Output frame ``j`` was generated for audio frame ``j + WARMUP_FRAMES`` (the
    first clip drops its warm-up frames but the audio window starts at 0), so
    the track starts ``WARMUP_FRAMES / fps`` seconds in and lasts exactly
    ``num_frames / fps`` seconds. The MP4 muxer trims to the shorter stream, so
    a track that ends early simply ends early.
    """
    start = int(round(WARMUP_FRAMES / fps * sample_rate))
    stop = start + int(round(num_frames / fps * sample_rate))
    return audio[..., start:stop]


class WanSpeechToVideoPipeline(LoRAPipeline, ComposedPipelineBase):

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler", "audio_encoder", "audio_processor"
    ]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        self.modules["scheduler"] = FlowUniPCMultistepScheduler(shift=fastvideo_args.pipeline_config.flow_shift)

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        self.add_stage(stage_name="input_validation_stage", stage=InputValidationStage())

        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))

        # Audio is constant across denoising steps, so encode it once here
        # rather than inside the loop.
        self.add_stage(stage_name="audio_encoding_stage",
                       stage=AudioEncodingStage(
                           audio_encoder=self.get_module("audio_encoder"),
                           audio_processor=self.get_module("audio_processor"),
                       ))

        self.add_stage(stage_name="conditioning_stage", stage=ConditioningStage())

        # The reference image becomes conditioning tokens appended to the video
        # sequence: one latent frame, encoded alone (not the I2V padded-video
        # format the shared stage produces).
        self.add_stage(stage_name="image_latent_preparation_stage",
                       stage=S2VRefImageEncodingStage(vae=self.get_module("vae")))

        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=TimestepPreparationStage(scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(scheduler=self.get_module("scheduler"),
                                                    transformer=self.get_module("transformer")))

        self.add_stage(stage_name="denoising_stage",
                       stage=WanS2VDenoisingStage(transformer=self.get_module("transformer"),
                                                  scheduler=self.get_module("scheduler")))

        # Official recipe: decode with the reference (or motion) latents
        # prepended so the causal VAE has temporal context, then trim.
        self.add_stage(stage_name="decoding_stage", stage=S2VDecodingStage(vae=self.get_module("vae")))

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if not self.post_init_called:
            self.post_init()
        try:
            return self._generate_clips(batch, fastvideo_args)
        except BaseException:
            # Same contract as the base forward: an aborted run must not leave
            # lazily materialised modules behind for the retry.
            self._release_all_lazy_modules()
            raise

    def _generate_clips(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        config = fastvideo_args.pipeline_config
        assert isinstance(batch.num_frames, int) and isinstance(batch.fps, int)
        plan = plan_clips(batch.num_frames, config.clip_frames)
        if plan.num_clips > 1 and fastvideo_args.output_type == "latent":
            raise ValueError(f"num_frames={batch.num_frames} needs {plan.num_clips} clips, and later clips "
                             "condition on the previous clip's decoded pixels; use output_type='pil' or ask "
                             f"for at most {plan.visible_frames(0)} frames with output_type='latent'.")
        logger.info("Wan-S2V plan: %d frames as %d clip(s) of %d generated frames", plan.num_frames, plan.num_clips,
                    plan.infer_frames)
        batch.extra[EXTRA_INFER_FRAMES] = plan.infer_frames
        batch.extra[EXTRA_AUDIO_FRAMES] = plan.audio_frames
        batch.extra[EXTRA_MOTION_LATENTS] = None

        stages = self._stage_name_mapping
        for name, stage in stages.items():
            if name in _PER_CLIP_STAGES:
                break
            batch = stage(batch, fastvideo_args)

        audio_embeds = batch.audio_embeds
        assert audio_embeds is not None and batch.seeds is not None
        ref_stage = stages["image_latent_preparation_stage"]
        assert isinstance(ref_stage, S2VRefImageEncodingStage)
        motion_frames = config.motion_frames[0]
        history: torch.Tensor | None = None
        clips: list[torch.Tensor] = []
        for clip in range(plan.num_clips):
            start = clip * plan.infer_frames
            batch.audio_embeds = audio_embeds[..., start:start + plan.infer_frames]
            # Official runner: clip r is seeded with seed + r.
            batch.generator = [torch.Generator("cpu").manual_seed(seed + clip) for seed in batch.seeds]
            batch.latents = None
            # LatentPreparationStage sizes noise as (num_frames - 1) // 4 + 1
            # latent frames; the visible count of a first clip lands exactly on
            # infer_frames // 4, which is what the DiT and audio expect.
            batch.num_frames = plan.infer_frames - WARMUP_FRAMES
            for name in _PER_CLIP_STAGES:
                batch = stages[name](batch, fastvideo_args)
            assert batch.output is not None
            frames = batch.output.cpu()
            clips.append(frames)
            if clip + 1 < plan.num_clips:
                history = roll_motion_history(history, frames * 2 - 1, motion_frames)
                batch.extra[EXTRA_MOTION_LATENTS] = ref_stage.encode_pixels(history, fastvideo_args)

        batch.num_frames = plan.num_frames
        batch.audio_embeds = audio_embeds
        batch.extra[EXTRA_MOTION_LATENTS] = None
        batch.output = torch.cat(clips, dim=2)[:, :, :plan.num_frames]
        audio = batch.extra.get("audio")
        if audio is not None:
            batch.extra["audio"] = align_audio(audio, int(batch.extra["audio_sample_rate"]), batch.fps, plan.num_frames)
        return batch


EntryClass = WanSpeechToVideoPipeline
