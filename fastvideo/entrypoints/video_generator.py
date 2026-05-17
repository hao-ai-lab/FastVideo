# SPDX-License-Identifier: Apache-2.0
"""
VideoGenerator module for FastVideo.

This module provides a consolidated interface for generating videos using
diffusion models.
"""

import os
import re
import shutil
import subprocess
import threading
import time
import tempfile
import types
import warnings
from collections.abc import Mapping
from contextlib import suppress
from copy import deepcopy
from typing import Any

import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange

from fastvideo.api.compat import (
    expand_request_prompt_batch,
    generator_config_to_fastvideo_args,
    legacy_from_pretrained_to_config,
    legacy_generate_call_to_request,
    load_generator_config_from_file,
    normalize_generation_request,
    normalize_generator_config,
    request_to_pipeline_overrides,
    request_to_sampling_param,
)
from fastvideo.api.results import (
    GenerationResult,
    VideoFinalEvent,
    VideoProgressEvent,
)
from fastvideo.api.schema import (
    GenerationRequest,
    GeneratorConfig,
    InputConfig,
    OutputConfig,
    SamplingConfig,
)
from fastvideo.api.sampling_param import SamplingParam
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines import ForwardBatch
from fastvideo.utils import align_to, shallow_asdict
from fastvideo.worker.executor import Executor

fcntl: types.ModuleType | None
try:
    import fcntl
except ImportError:
    fcntl = None

logger = init_logger(__name__)
_FFMPEG_ENCODER_OPTION_CACHE: dict[tuple[str, str, str], bool] = {}

_BATCH_EXTRA_PASSTHROUGH_KEYS: tuple[str, ...] = (
    "ltx2_audio_latents",
    "ltx2_audio_clean_latent",
    "ltx2_audio_denoise_mask",
    "audio_num_frames",
    "video_position_offset_sec",
)

_FROM_PRETRAINED_CONVENIENCE_KWARGS = frozenset({
    "num_gpus",
    "revision",
    "trust_remote_code",
    "distributed_executor_backend",
    "tp_size",
    "sp_size",
    "hsdp_replicate_dim",
    "hsdp_shard_dim",
    "dist_timeout",
    "use_fsdp_inference",
    "disable_autocast",
    "enable_stage_verification",
    "dit_cpu_offload",
    "dit_layerwise_offload",
    "text_encoder_cpu_offload",
    "image_encoder_cpu_offload",
    "vae_cpu_offload",
    "pin_cpu_memory",
    "enable_torch_compile",
    "torch_compile_kwargs",
    "output_type",
})


def _infer_latent_batch_size(batch: ForwardBatch) -> int:
    if isinstance(batch.prompt, list):
        latent_batch_size = len(batch.prompt)
    elif batch.prompt is not None:
        latent_batch_size = 1
    elif batch.prompt_embeds is not None and len(batch.prompt_embeds) > 0:
        latent_batch_size = batch.prompt_embeds[0].shape[0]
    else:
        raise ValueError("Cannot infer batch size from batch; no prompt or prompt_embeds found")
    latent_batch_size *= batch.num_videos_per_prompt
    return latent_batch_size


class VideoGenerator:
    """
    A unified class for generating videos using diffusion models.
    
    This class provides a simple interface for video generation with rich
    customization options, similar to popular frameworks like HF Diffusers.
    """

    def __init__(
        self,
        fastvideo_args: FastVideoArgs,
        executor_class: type[Executor],
        log_stats: bool,
        *,
        log_queue=None,
    ):
        """
        Initialize the video generator.

        Args:
            fastvideo_args: The inference arguments
            executor_class: The executor class to use for inference
            log_stats: Whether to log statistics
            log_queue: Optional multiprocessing.Queue to forward worker logs to
        """
        self.config: GeneratorConfig | None = None
        self.fastvideo_args = fastvideo_args
        self.executor = executor_class(fastvideo_args, log_queue=log_queue)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | GeneratorConfig | Mapping[str, Any] | None = None,
        **kwargs,
    ) -> "VideoGenerator":
        """
        Create a video generator from a pretrained model.
        
        Args:
            model_path: Path or identifier for the pretrained model
            pipeline_config: Pipeline config to use for inference
            **kwargs: Additional arguments to customize model loading, set any FastVideoArgs or PipelineConfig attributes here.
                
        Returns:
            The created video generator

        Priority level: Default pipeline config < User's pipeline config < User's kwargs

        Stable convenience kwargs remain supported here for common engine and
        offload settings. Advanced model- or pipeline-specific options should
        move to VideoGenerator.from_config(...).
        """
        log_queue = kwargs.pop("log_queue", None)
        typed_config = kwargs.pop("config", None)
        if typed_config is not None:
            if model_path is not None:
                raise TypeError("Pass either model_path or config to from_pretrained, not both")
            if kwargs:
                unexpected = ", ".join(sorted(kwargs))
                raise TypeError(f"Unexpected keyword arguments with config: {unexpected}")
            return cls.from_config(typed_config, log_queue=log_queue)

        if isinstance(model_path, GeneratorConfig | Mapping):
            if kwargs:
                unexpected = ", ".join(sorted(kwargs))
                raise TypeError(f"Unexpected keyword arguments with typed config: {unexpected}")
            return cls.from_config(model_path, log_queue=log_queue)

        if model_path is None:
            raise TypeError("model_path or config is required")

        legacy_only_kwargs = sorted(set(kwargs) - _FROM_PRETRAINED_CONVENIENCE_KWARGS)
        if legacy_only_kwargs:
            warnings.warn(
                "VideoGenerator.from_pretrained(...) received legacy-only kwargs "
                f"({', '.join(legacy_only_kwargs)}); prefer VideoGenerator.from_config(...) "
                "for advanced configuration.",
                DeprecationWarning,
                stacklevel=2,
            )
        return cls.from_config(
            legacy_from_pretrained_to_config(model_path, kwargs),
            log_queue=log_queue,
        )

    @classmethod
    def from_config(
        cls,
        config: GeneratorConfig | Mapping[str, Any],
        *,
        log_queue=None,
    ) -> "VideoGenerator":
        normalized = normalize_generator_config(config)
        fastvideo_args = generator_config_to_fastvideo_args(normalized)
        generator = cls.from_fastvideo_args(fastvideo_args, log_queue=log_queue)
        generator.config = normalized
        return generator

    @classmethod
    def from_file(
        cls,
        path: str,
        overrides: list[str] | Mapping[str, Any] | None = None,
        *,
        log_queue=None,
    ) -> "VideoGenerator":
        return cls.from_config(
            load_generator_config_from_file(path, overrides=overrides),
            log_queue=log_queue,
        )

    @classmethod
    def from_fastvideo_args(
        cls,
        fastvideo_args: FastVideoArgs,
        *,
        log_queue=None,
    ) -> "VideoGenerator":
        """
        Create a video generator with the specified arguments.

        Args:
            fastvideo_args: The inference arguments
            log_queue: Optional multiprocessing.Queue to forward worker logs to

        Returns:
            The created video generator
        """
        # Initialize distributed environment if needed
        # initialize_distributed_and_parallelism(fastvideo_args)

        executor_class = Executor.get_class(fastvideo_args)
        return cls(
            fastvideo_args=fastvideo_args,
            executor_class=executor_class,
            log_stats=False,  # TODO: implement
            log_queue=log_queue,
        )

    def generate(
        self,
        request: GenerationRequest | Mapping[str, Any],
        *,
        log_queue=None,
    ) -> GenerationResult | list[GenerationResult]:
        """
        Generate video or image outputs from a typed inference request.

        Args:
            request: A `GenerationRequest` instance or a mapping that can be
                parsed into one. This is the primary public inference
                entrypoint for the typed API.
            log_queue: Optional multiprocessing.Queue to forward worker logs to
                during this request.

        Returns:
            A `GenerationResult` for single-request generation, or a list of
            `GenerationResult` objects when the request expands into multiple
            prompts.
        """
        normalized_request = normalize_generation_request(request)
        if log_queue:
            self.executor.set_log_queue(log_queue)

        try:
            return self._generate_request_impl(normalized_request)
        finally:
            if log_queue:
                self.executor.clear_log_queue()

    async def generate_async(
        self,
        request: GenerationRequest | Mapping[str, Any],
        *,
        log_queue=None,
    ):
        """Async generation that yields typed :class:`VideoEvent`s.

        Three consumers share this substrate:

        * Streaming server (:mod:`fastvideo.entrypoints.streaming`) —
          pipes :class:`VideoPartialEvent` frames into fMP4.
        * Stateless OpenAI server — ignores progress events, forwards
          :class:`VideoFinalEvent` as the HTTP response body.
        * Dynamo native backend
          (``components/src/dynamo/fastvideo/``) — wraps each event as
          an ``NvVideosResponse`` chunk.

        The aggregated code path shipped here yields a single
        :class:`VideoProgressEvent` at start and one
        :class:`VideoFinalEvent` at end. Future work will thread
        per-step progress events through the pipeline's denoise loop
        so streaming consumers don't have to wait on a materialized
        final.
        """
        import asyncio

        normalized = normalize_generation_request(request)
        total_steps = max(1, normalized.sampling.num_inference_steps)
        yield VideoProgressEvent(step=0, total_steps=total_steps, stage="denoise")

        if log_queue:
            self.executor.set_log_queue(log_queue)
        try:
            result = await asyncio.to_thread(self._generate_request_impl, normalized)
        finally:
            if log_queue:
                self.executor.clear_log_queue()

        if isinstance(result, list):
            # Prompt-batch expansion — emit one Final per sub-result.
            for sub in result:
                yield await asyncio.to_thread(_final_event_from_result, sub)
            return
        yield await asyncio.to_thread(_final_event_from_result, result)

    @staticmethod
    def default_health_check_request() -> GenerationRequest:
        """Return the minimal typed request Dynamo uses for probes.

        256x256, 8 frames, 1 inference step -- fast enough to be a
        viable liveness check, non-trivial enough to exercise the
        DiT -> VAE -> decode path. Consumers adapt this shape to their
        transport's health-check payload (see
        ``docs/design/server_contracts/dynamo.md``).
        """
        return GenerationRequest(
            prompt="health check",
            inputs=InputConfig(),
            sampling=SamplingConfig(
                num_frames=8,
                height=256,
                width=256,
                fps=24,
                num_inference_steps=1,
                guidance_scale=1.0,
            ),
            output=OutputConfig(save_video=False, return_frames=False),
        )

    def generate_video(
        self,
        prompt: str | None = None,
        sampling_param: SamplingParam | None = None,
        # Action control inputs (Matrix-Game)
        mouse_cond: torch.Tensor | None = None,
        keyboard_cond: torch.Tensor | None = None,
        grid_sizes: tuple[int, int, int] | list[int] | torch.Tensor
        | None = None,
        **kwargs,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Generate a video based on the given prompt.
        
        Args:
            prompt: The prompt to use for generation (optional if prompt_txt is provided)
            negative_prompt: The negative prompt to use (overrides the one in fastvideo_args)
            output_path: Path to save the video (overrides the one in fastvideo_args)
            prompt_path: Path to prompt file
            save_video: Whether to save the video to disk
            return_frames: Whether to include raw frames in the result dict
            num_inference_steps: Number of denoising steps (overrides fastvideo_args)
            guidance_scale: Classifier-free guidance scale (overrides fastvideo_args)
            num_frames: Number of frames to generate (overrides fastvideo_args)
            height: Height of generated video (overrides fastvideo_args)
            width: Width of generated video (overrides fastvideo_args)
            fps: Frames per second for saved video (overrides fastvideo_args)
            seed: Random seed for generation (overrides fastvideo_args)
            callback: Callback function called after each step
            callback_steps: Number of steps between each callback
            
        Returns:
            A metadata dictionary for single-prompt generation, or a list of
            metadata dictionaries for prompt-file batch generation.
        """
        log_queue = kwargs.pop("log_queue", None)
        warnings.warn(
            "VideoGenerator.generate_video(...) is deprecated; use "
            "VideoGenerator.generate(request=...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if log_queue:
            self.executor.set_log_queue(log_queue)

        try:
            extra_overrides: dict[str, Any] = {}
            for _ek in _BATCH_EXTRA_PASSTHROUGH_KEYS:
                if _ek in kwargs:
                    extra_overrides[_ek] = kwargs.pop(_ek)

            request = legacy_generate_call_to_request(
                prompt,
                sampling_param,
                mouse_cond=mouse_cond,
                keyboard_cond=keyboard_cond,
                grid_sizes=grid_sizes,
                legacy_kwargs=kwargs,
            )

            fastvideo_args = self.fastvideo_args
            pipeline_overrides = request_to_pipeline_overrides(request)
            if pipeline_overrides:
                fastvideo_args = deepcopy(self.fastvideo_args)
                for key, value in pipeline_overrides.items():
                    if not hasattr(fastvideo_args.pipeline_config, key):
                        raise ValueError(f"Request field {key!r} is not supported by pipeline config overrides")
                    setattr(fastvideo_args.pipeline_config, key, deepcopy(value))

            resolved_sampling_param = request_to_sampling_param(
                request,
                model_path=self.fastvideo_args.model_path,
            )
            return self._generate_video_impl(
                prompt=request.prompt,
                sampling_param=resolved_sampling_param,
                fastvideo_args=fastvideo_args,
                **extra_overrides,
            )
        finally:
            if log_queue:
                self.executor.clear_log_queue()

    def _generate_request_impl(
        self,
        request: GenerationRequest,
    ) -> GenerationResult | list[GenerationResult]:
        if isinstance(request.prompt, list):
            if request.inputs.prompt_path is not None:
                raise ValueError("request.prompt list cannot be combined with request.inputs.prompt_path")
            results: list[GenerationResult] = []
            for index, single_request in enumerate(expand_request_prompt_batch(request)):
                prompt = single_request.prompt
                wrapped = self._generate_single_request(single_request)
                if isinstance(wrapped, list):
                    results.extend(wrapped)
                    continue
                wrapped.prompt_index = index
                if wrapped.prompt is None:
                    wrapped.prompt = prompt
                results.append(wrapped)
            return results

        return self._generate_single_request(request)

    def _generate_single_request(
        self,
        request: GenerationRequest,
    ) -> GenerationResult | list[GenerationResult]:
        fastvideo_args = self.fastvideo_args
        pipeline_overrides = request_to_pipeline_overrides(request)
        if pipeline_overrides:
            fastvideo_args = deepcopy(self.fastvideo_args)
            for key, value in pipeline_overrides.items():
                if not hasattr(fastvideo_args.pipeline_config, key):
                    raise ValueError(f"Request field {key!r} is not supported by pipeline config overrides")
                setattr(fastvideo_args.pipeline_config, key, deepcopy(value))

        sampling_param = request_to_sampling_param(
            request,
            model_path=self.fastvideo_args.model_path,
        )
        result = self._generate_video_impl(
            prompt=request.prompt,
            sampling_param=sampling_param,
            fastvideo_args=fastvideo_args,
        )
        return self._wrap_legacy_result(result)

    def _generate_video_impl(
        self,
        prompt: str | None = None,
        sampling_param: SamplingParam | None = None,
        mouse_cond: torch.Tensor | None = None,
        keyboard_cond: torch.Tensor | None = None,
        grid_sizes: tuple[int, int, int] | list[int] | torch.Tensor
        | None = None,
        fastvideo_args: FastVideoArgs | None = None,
        **kwargs,
    ) -> dict[str, Any] | list[np.ndarray] | list[dict[str, Any]]:
        """Internal implementation of generate_video."""
        if fastvideo_args is None:
            fastvideo_args = self.fastvideo_args

        # Handle batch processing from text file
        if sampling_param is None:
            sampling_param = SamplingParam.from_pretrained(fastvideo_args.model_path)

        # Add action control inputs to kwargs if provided
        if mouse_cond is not None:
            kwargs['mouse_cond'] = mouse_cond
        if keyboard_cond is not None:
            kwargs['keyboard_cond'] = keyboard_cond
        if grid_sizes is not None:
            kwargs['grid_sizes'] = grid_sizes

        extra_overrides: dict[str, Any] = {}
        for _ek in _BATCH_EXTRA_PASSTHROUGH_KEYS:
            if _ek in kwargs:
                extra_overrides[_ek] = kwargs.pop(_ek)

        sampling_param.update(kwargs)
        kwargs["_extra_overrides"] = extra_overrides

        if fastvideo_args.prompt_txt is not None or sampling_param.prompt_path is not None:
            prompt_txt_path = sampling_param.prompt_path or fastvideo_args.prompt_txt
            if not prompt_txt_path or not os.path.exists(prompt_txt_path):
                raise FileNotFoundError(f"Prompt text file not found: {prompt_txt_path}")

            # Read prompts from file
            with open(prompt_txt_path, encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]

            if not prompts:
                raise ValueError(f"No prompts found in file: {prompt_txt_path}")

            logger.info("Found %d prompts in %s", len(prompts), prompt_txt_path)

            results = []
            for i, batch_prompt in enumerate(prompts):
                logger.info("Processing prompt %d/%d: %s...", i + 1, len(prompts), batch_prompt[:100])
                try:
                    # Generate video for this prompt using the same logic below
                    output_path = self._prepare_output_path(sampling_param.output_path, batch_prompt)
                    kwargs["output_path"] = output_path
                    result = self._generate_single_video(
                        prompt=batch_prompt,
                        sampling_param=sampling_param,
                        fastvideo_args=fastvideo_args,
                        **kwargs,
                    )

                    # Add prompt info to result
                    result["prompt_index"] = i
                    result["prompt"] = batch_prompt

                    results.append(result)
                    logger.info("Successfully generated video for prompt %d", i + 1)

                except Exception as e:
                    logger.error("Failed to generate video for prompt %d: %s", i + 1, e)
                    continue

            logger.info("Completed batch processing. Generated %d videos successfully.", len(results))
            return results

        # Single prompt generation (original behavior)
        if prompt is None:
            raise ValueError("Either prompt or prompt_txt must be provided")
        output_path = self._prepare_output_path(sampling_param.output_path, prompt)
        kwargs["output_path"] = output_path
        return self._generate_single_video(
            prompt=prompt,
            sampling_param=sampling_param,
            fastvideo_args=fastvideo_args,
            **kwargs,
        )

    def _is_image_workload(self) -> bool:
        """Return True when the workload produces a single image (t2i, i2i …)."""
        args = getattr(self, "fastvideo_args", None)
        if args is None:
            return False
        return args.workload_type.value.endswith("2i")

    def _prepare_output_path(
        self,
        output_path: str,
        prompt: str,
    ) -> str:
        """Build a unique, sanitized output file path.

        The file extension is chosen automatically based on the workload type:
        ``.png`` for image workloads (``t2i``, ``i2i``, …) and ``.mp4`` for
        video workloads.

        - If ``output_path`` already carries the correct extension, treat it
          as a file path.
        - Otherwise, treat ``output_path`` as a directory and derive the
          filename from the prompt.
        - Invalid filename characters are removed; if the name changes, a
          warning is logged.
        - If the target path already exists, a numeric suffix is appended.
        """
        target_ext = ".png" if self._is_image_workload() else ".mp4"

        def _sanitize_filename_component(name: str) -> str:
            # Remove characters invalid on common filesystems, strip spaces/dots
            sanitized = re.sub(r'[\\/:*?"<>|]', '', name)
            sanitized = sanitized.strip().strip('.')
            sanitized = re.sub(r'\s+', ' ', sanitized)
            return sanitized or "output"

        base_path, extension = os.path.splitext(output_path)
        extension_lower = extension.lower()

        if extension_lower == target_ext:
            output_dir = os.path.dirname(output_path)
            base_name = os.path.basename(base_path)  # filename without extension
            sanitized_base = _sanitize_filename_component(base_name)
            if sanitized_base != base_name:
                logger.warning(
                    "The output name '%s' contained invalid characters. "
                    "It has been renamed to '%s%s'",
                    os.path.basename(output_path),
                    sanitized_base,
                    target_ext,
                )
            out_name = f"{sanitized_base}{target_ext}"
        else:
            # Treat as directory; inform if an unexpected extension was
            # provided.
            if extension:
                logger.info(
                    "Output path '%s' has extension '%s' which does not "
                    "match the target '%s'; treating it as a directory",
                    output_path,
                    extension,
                    target_ext,
                )
            output_dir = output_path
            prompt_component = _sanitize_filename_component(prompt[:100])
            out_name = f"{prompt_component}{target_ext}"

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        new_output_path = os.path.join(output_dir, out_name)
        counter = 1
        while os.path.exists(new_output_path):
            name_part, ext_part = os.path.splitext(out_name)
            new_name = f"{name_part}_{counter}{ext_part}"
            new_output_path = os.path.join(output_dir, new_name)
            counter += 1
        return new_output_path

    def _generate_single_video(
        self,
        prompt: str,
        sampling_param: SamplingParam | None = None,
        fastvideo_args: FastVideoArgs | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Internal method for single video generation"""
        if fastvideo_args is None:
            fastvideo_args = self.fastvideo_args

        # Validate inputs
        if not isinstance(prompt, str):
            raise TypeError(f"`prompt` must be a string, but got {type(prompt)}")
        prompt = prompt.strip()
        sampling_param = deepcopy(sampling_param)
        output_path = kwargs["output_path"]
        sampling_param.prompt = prompt
        # Process negative prompt
        if sampling_param.negative_prompt is not None:
            sampling_param.negative_prompt = sampling_param.negative_prompt.strip()

        # Validate dimensions
        if (sampling_param.height <= 0 or sampling_param.width <= 0 or sampling_param.num_frames <= 0):
            raise ValueError(f"Height, width, and num_frames must be positive integers, got "
                             f"height={sampling_param.height}, width={sampling_param.width}, "
                             f"num_frames={sampling_param.num_frames}")

        # Calculate sizes
        target_height = align_to(sampling_param.height, 16)
        target_width = align_to(sampling_param.width, 16)

        # Calculate latent sizes
        latents_size = [(sampling_param.num_frames - 1) // 4 + 1, sampling_param.height // 8, sampling_param.width // 8]
        n_tokens = latents_size[0] * latents_size[1] * latents_size[2]

        # Log parameters
        debug_str = f"""
                      height: {target_height}
                       width: {target_width}
                video_length: {sampling_param.num_frames}
                      prompt: {sampling_param.prompt}
                      image_path: {sampling_param.image_path}
                  neg_prompt: {sampling_param.negative_prompt}
                        seed: {sampling_param.seed}
                 infer_steps: {sampling_param.num_inference_steps}
       num_videos_per_prompt: {sampling_param.num_videos_per_prompt}
              guidance_scale: {sampling_param.guidance_scale}
                    n_tokens: {n_tokens}
                  flow_shift: {fastvideo_args.pipeline_config.flow_shift}
     embedded_guidance_scale: {fastvideo_args.pipeline_config.embedded_cfg_scale}
                  save_video: {sampling_param.save_video}
                  output_path: {output_path}
        """ # type: ignore[attr-defined]
        logger.info(debug_str)

        # Prepare batch
        batch = ForwardBatch(
            **shallow_asdict(sampling_param),
            eta=0.0,
            n_tokens=n_tokens,
            VSA_sparsity=fastvideo_args.VSA_sparsity,
        )

        extra_overrides = kwargs.pop("_extra_overrides", {})
        for _ek, _ev in extra_overrides.items():
            batch.extra[_ek] = _ev

        # Run inference
        start_time = time.perf_counter()

        # Execute forward pass in a new thread for non-blocking tensor
        # allocation. Capture thread exceptions so we can surface the true
        # failure in the main thread instead of later hitting None outputs.
        result_container = {"output_batch": ForwardBatch(data_type=batch.data_type)}
        thread_error: dict[str, BaseException | None] = {"error": None}
        thread_error_traceback: dict[str, str] = {"traceback": ""}

        def execute_forward_thread():
            import traceback
            try:
                result_container["output_batch"] = self.executor.execute_forward(batch, fastvideo_args)
            except BaseException as error:  # noqa: BLE001
                thread_error["error"] = error
                thread_error_traceback["traceback"] = traceback.format_exc()

        thread = threading.Thread(target=execute_forward_thread)
        thread.start()
        latent_batch_size = _infer_latent_batch_size(batch)
        # When ``output_type == "latent"`` the forward output has latent
        # shape (e.g. ``[B, C_latent, T_latent, H_latent, W_latent]``)
        # rather than the pre-allocation's pixel shape. Skip the pinned
        # ~50 MB buffer entirely; we always fall through to the
        # ``samples = output_batch.output.cpu()`` branch below in that
        # mode. ``skip_pixel_prealloc`` also gates the slow-path warning.
        skip_pixel_prealloc = fastvideo_args.output_type == "latent"
        if skip_pixel_prealloc:
            samples = torch.empty(0, device='cpu')
        else:
            samples = torch.empty(
                (latent_batch_size, 3, sampling_param.num_frames, sampling_param.height, sampling_param.width),
                device='cpu',
                pin_memory=fastvideo_args.pin_cpu_memory)
        thread.join()

        if thread_error["error"] is not None:
            raise RuntimeError("Forward execution thread failed.\n"
                               f"{thread_error_traceback['traceback']}") from thread_error["error"]

        output_batch = result_container["output_batch"]
        if output_batch.output is None:
            raise RuntimeError("Forward execution returned no output tensor. "
                               "This usually means the executor/pipeline failed earlier.")

        if output_batch.output.shape == samples.shape:
            samples.copy_(output_batch.output)
        else:
            if not skip_pixel_prealloc:
                logger.warning("Output shape %s does not match expected shape %s; use slow path",
                               output_batch.output.shape, samples.shape)
            samples = output_batch.output.cpu()
        logging_info = output_batch.logging_info

        gen_time = time.perf_counter() - start_time
        logger.info("Generated successfully in %.2f seconds", gen_time)

        # Three mutually-exclusive output modes determine whether (a) we
        # build an RGB frame buffer and (b) what file we write to disk:
        #
        #   1. `output_type == "latent"` — VAE is bypassed in DecodingStage
        #      and `samples` holds raw latents (arbitrary channel count).
        #      The RGB grid / uint8 / mp4 / png pipeline below cannot
        #      consume those, so we skip it entirely and let callers work
        #      with the latent tensor directly via `result["samples"]`.
        #   2. Audio-only workload — `samples` is a 1×3×1×8×8 placeholder
        #      no caller will use; skip the grid loop and save a `.wav`.
        #   3. Pixel video / image — the historical happy path.
        is_latent_output = fastvideo_args.output_type == "latent"
        audio_only = bool(output_batch.extra.get("audio_only"))

        postprocess_start = time.perf_counter()
        frames: list[np.ndarray] | None
        if is_latent_output or audio_only:
            frames = None if is_latent_output else []
        else:
            videos = rearrange(samples, "b c t h w -> t b c h w")
            frames = []
            for x in videos:
                x = torchvision.utils.make_grid(x, nrow=6)
                x = x.permute(1, 2, 0).squeeze(-1)
                x = (x * 255).to(torch.uint8)
                frames.append(x.contiguous().cpu().numpy())
        postprocess_time = time.perf_counter() - postprocess_start
        logger.info("PostDecodeFrameProcessStage completed in %.3f s", postprocess_time)
        if logging_info is not None:
            logging_info.add_stage_execution_time("PostDecodeFrameProcessStage", postprocess_time)

        save_to_disk = batch.save_video and not is_latent_output
        save_video_time = 0.0
        audio_mux_time = 0.0
        if save_to_disk:
            if audio_only:
                # Audio-only workload: write a standalone .wav rather than
                # muxing the audio into a placeholder mp4 (which forces
                # ffmpeg to round 8x8 placeholder frames up to 16x16).
                output_path = self._rewrite_extension(output_path, ".wav")
                save_start = time.perf_counter()
                self._write_pcm_wav(
                    output_path,
                    output_batch.extra["audio"],
                    int(output_batch.extra["audio_sample_rate"]),
                )
                save_video_time = time.perf_counter() - save_start
                logger.info("Saved audio to %s", output_path)
            elif self._is_image_workload():
                # Image workloads (t2i, i2i, …): save the first frame as PNG.
                assert frames is not None  # implied by save_to_disk and not audio_only
                save_start = time.perf_counter()
                imageio.imwrite(output_path, frames[0])
                save_video_time = time.perf_counter() - save_start
                logger.info("Saved image to %s", output_path)
            else:
                assert frames is not None  # implied by save_to_disk and not audio_only
                audio = output_batch.extra.get("audio")
                audio_sample_rate = output_batch.extra.get("audio_sample_rate")
                if audio is not None and audio_sample_rate is not None:
                    # Single-pass save path: encode video+audio once, avoiding
                    # second-pass remux overhead. AudioMuxStage remains 0.0 on
                    # success because audio is already present in the saved MP4.
                    save_start = time.perf_counter()
                    save_ok = self._save_video_with_audio_ffmpeg_pipe(
                        output_path=output_path,
                        frames=frames,
                        fps=batch.fps,
                        audio=audio,
                        sample_rate=int(audio_sample_rate),
                    )
                    if not save_ok:
                        logger.warning("ffmpeg pipe save failed; trying PyAV single-pass save.")
                        save_ok = self._save_video_with_audio_single_pass(
                            output_path=output_path,
                            frames=frames,
                            fps=batch.fps,
                            audio=audio,
                            sample_rate=int(audio_sample_rate),
                        )
                    save_video_time = time.perf_counter() - save_start
                    if save_ok:
                        audio_mux_time = 0.0
                    else:
                        logger.warning("Single-pass save failed; falling back to two-step save/mux.")
                        save_start = time.perf_counter()
                        imageio.mimsave(output_path, frames, fps=batch.fps, format="mp4")
                        save_video_time = time.perf_counter() - save_start
                        mux_start = time.perf_counter()
                        mux_ok = self._mux_audio(output_path, audio, int(audio_sample_rate))
                        audio_mux_time = time.perf_counter() - mux_start
                        if not mux_ok:
                            logger.warning("Audio mux failed; saved video without audio.")
                else:
                    save_start = time.perf_counter()
                    imageio.mimsave(output_path, frames, fps=batch.fps, format="mp4")
                    save_video_time = time.perf_counter() - save_start
                    audio_mux_time = 0.0
                logger.info("Saved video to %s", output_path)

            logger.info("VideoSaveStage completed in %.3f s", save_video_time)
            if logging_info is not None:
                logging_info.add_stage_execution_time("VideoSaveStage", save_video_time)
            logger.info("AudioMuxStage completed in %.3f s", audio_mux_time)
            if logging_info is not None:
                logging_info.add_stage_execution_time("AudioMuxStage", audio_mux_time)

        e2e_time = time.perf_counter() - start_time
        logger.info("End-to-end latency: %.2f seconds", e2e_time)

        result: dict[str, Any] = {
            "prompts": prompt,
            "samples": samples if batch.return_frames else None,
            "frames": frames if batch.return_frames else None,
            # Audio is the primary output for audio workloads — return it
            # whenever the pipeline produced one, regardless of
            # `return_frames` (which gates the video-shaped buffers).
            "audio": output_batch.extra.get("audio"),
            "audio_sample_rate": output_batch.extra.get("audio_sample_rate"),
            "ltx2_audio_latents": output_batch.extra.get("ltx2_audio_latents"),
            "size": (target_height, target_width, batch.num_frames),
            "generation_time": gen_time,
            "e2e_latency": e2e_time,
            "logging_info": logging_info,
            "trajectory": output_batch.trajectory_latents,
            "trajectory_timesteps": output_batch.trajectory_timesteps,
            "trajectory_decoded": output_batch.trajectory_decoded,
            "video_path": output_path if save_to_disk else None,
            "peak_memory_mb": output_batch.extra.get("peak_memory_mb"),
        }

        return result

    @staticmethod
    def _wrap_legacy_result(
        result: dict[str, Any] | list[dict[str, Any]], ) -> GenerationResult | list[GenerationResult]:
        if isinstance(result, list):
            return [GenerationResult.from_legacy_result(item) for item in result]
        return GenerationResult.from_legacy_result(result)

    @staticmethod
    def _unwrap_typed_result(
        result: GenerationResult | list[GenerationResult], ) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(result, list):
            return [item.to_legacy_dict() for item in result]
        return result.to_legacy_dict()

    @staticmethod
    def _rewrite_extension(path: str, new_ext: str) -> str:
        root, old_ext = os.path.splitext(path)
        new_path = root + new_ext
        if old_ext and old_ext.lower() != new_ext.lower():
            logger.info("Rewriting output extension %s -> %s.", old_ext, new_ext)
        return new_path

    @staticmethod
    def _audio_to_int16(audio: torch.Tensor | np.ndarray, ) -> tuple[np.ndarray, int]:
        """Normalize `[samples]` / `[samples, channels]` / `[channels,
        samples]` audio in roughly [-1, 1] to a `(int16 [samples,
        channels], num_channels)` pair. Raises `ValueError` for shapes
        we can't classify.
        """
        if torch.is_tensor(audio):
            audio_np = audio.detach().cpu().float().numpy()
        else:
            audio_np = np.asarray(audio, dtype=np.float32)
        if audio_np.ndim == 1:
            audio_np = audio_np[:, None]
        elif audio_np.ndim == 2:
            if audio_np.shape[0] <= 8 and audio_np.shape[1] > audio_np.shape[0]:
                audio_np = audio_np.T
        else:
            raise ValueError(f"Unexpected audio shape {audio_np.shape}.")
        audio_np = np.clip(audio_np, -1.0, 1.0)
        audio_int16 = (audio_np * 32767.0).astype(np.int16)
        return audio_int16, audio_int16.shape[1]

    @classmethod
    def _write_pcm_wav(
        cls,
        wav_path: str,
        audio: torch.Tensor | np.ndarray,
        sample_rate: int,
    ) -> int:
        """Write 16-bit PCM WAV; returns the channel count."""
        import wave
        audio_int16, num_channels = cls._audio_to_int16(audio)
        with wave.open(wav_path, "wb") as f:
            f.setnchannels(num_channels)
            f.setsampwidth(2)
            f.setframerate(sample_rate)
            f.writeframes(audio_int16.tobytes())
        return num_channels

    @classmethod
    def _save_video_with_audio_single_pass(
        cls,
        output_path: str,
        frames: list[np.ndarray],
        fps: int,
        audio: torch.Tensor | np.ndarray,
        sample_rate: int,
    ) -> bool:
        """Encode video+audio into MP4 in one pass using PyAV."""
        try:
            import av
        except ImportError:
            logger.warning("PyAV not installed; cannot use single-pass save.")
            return False

        if not frames:
            return False

        output = None
        try:
            audio_int16, num_channels = cls._audio_to_int16(audio)
            layout = "stereo" if num_channels == 2 else "mono"
            output = av.open(output_path, mode="w")
            video_stream = output.add_stream("libx264", rate=fps)
            video_stream.width = int(frames[0].shape[1])
            video_stream.height = int(frames[0].shape[0])
            video_stream.pix_fmt = "yuv420p"
            video_stream.options = {
                "preset": "ultrafast",
                "tune": "zerolatency",
            }

            audio_stream = output.add_stream("aac", rate=sample_rate, layout=layout)

            for frame_np in frames:
                vframe = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame_np), format="rgb24")
                for packet in video_stream.encode(vframe):
                    output.mux(packet)
            for packet in video_stream.encode():
                output.mux(packet)

            # AAC commonly uses 1024 samples/frame. Pad the tail frame to avoid
            # shape errors in PyAV/FFmpeg.
            chunk_size = 1024
            for start in range(0, audio_int16.shape[0], chunk_size):
                chunk = audio_int16[start:start + chunk_size]
                if chunk.shape[0] < chunk_size:
                    pad = np.zeros((chunk_size - chunk.shape[0], chunk.shape[1]), dtype=chunk.dtype)
                    chunk = np.concatenate([chunk, pad], axis=0)
                chunk_planar = np.ascontiguousarray(chunk.T)
                aframe = av.AudioFrame.from_ndarray(chunk_planar, format="s16p", layout=layout)
                aframe.sample_rate = sample_rate
                for packet in audio_stream.encode(aframe):
                    output.mux(packet)
            for packet in audio_stream.encode():
                output.mux(packet)

            output.close()
            return True
        except Exception as e:
            logger.warning("Single-pass video+audio save failed: %s", e)
            if output is not None:
                with suppress(Exception):
                    output.close()
            return False

    @staticmethod
    def _ffmpeg_encoder_supports_option(ffmpeg_bin: str, codec: str, option_name: str) -> bool:
        """Best-effort check whether ffmpeg encoder supports an option."""
        cache_key = (ffmpeg_bin, codec, option_name)
        cached = _FFMPEG_ENCODER_OPTION_CACHE.get(cache_key)
        if cached is not None:
            return cached

        try:
            result = subprocess.run(
                [
                    ffmpeg_bin,
                    "-hide_banner",
                    "-h",
                    f"encoder={codec}",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                _FFMPEG_ENCODER_OPTION_CACHE[cache_key] = False
                return False
            haystack = f"{result.stdout}\n{result.stderr}"
            supported = f"-{option_name}" in haystack
            _FFMPEG_ENCODER_OPTION_CACHE[cache_key] = supported
            return supported
        except Exception:
            _FFMPEG_ENCODER_OPTION_CACHE[cache_key] = False
            return False

    @classmethod
    def _save_video_with_audio_ffmpeg_pipe(
        cls,
        output_path: str,
        frames: list[np.ndarray],
        fps: int,
        audio: torch.Tensor | np.ndarray,
        sample_rate: int,
    ) -> bool:
        """Encode video+audio using ffmpeg via rawvideo stdin + WAV input."""
        ffmpeg_bin = shutil.which(os.getenv("FASTVIDEO_FFMPEG_BIN", "ffmpeg"))
        if ffmpeg_bin is None:
            logger.warning("ffmpeg not found; cannot use ffmpeg pipe save.")
            return False

        if not frames:
            return False

        height = int(frames[0].shape[0])
        width = int(frames[0].shape[1])
        codec = os.getenv("FASTVIDEO_VIDEO_CODEC", "libx264")

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                wav_path = os.path.join(tmpdir, "audio.wav")
                try:
                    cls._write_pcm_wav(wav_path, audio, sample_rate)
                except ValueError as e:
                    logger.warning("Unexpected audio tensor for ffmpeg pipe save: %s", e)
                    return False

                cmd = [
                    ffmpeg_bin,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s:v",
                    f"{width}x{height}",
                    "-r",
                    str(fps),
                    "-i",
                    "pipe:0",
                    "-i",
                    wav_path,
                    "-c:v",
                    codec,
                ]

                if codec.endswith("_nvenc"):
                    nvenc_options = [
                        ("preset", os.getenv("FASTVIDEO_NVENC_PRESET", "p1")),
                        ("tune", os.getenv("FASTVIDEO_NVENC_TUNE", "ull")),
                        ("rc", os.getenv("FASTVIDEO_NVENC_RC", "constqp")),
                        ("qp", os.getenv("FASTVIDEO_NVENC_QP", "28")),
                        ("bf", os.getenv("FASTVIDEO_NVENC_BF", "0")),
                    ]
                    for option_name, option_value in nvenc_options:
                        if cls._ffmpeg_encoder_supports_option(ffmpeg_bin, codec, option_name):
                            cmd += [f"-{option_name}", option_value]
                else:
                    cmd += ["-preset", os.getenv("FASTVIDEO_X264_PRESET", "ultrafast")]

                cmd += [
                    "-c:a",
                    "aac",
                    "-pix_fmt",
                    os.getenv("FASTVIDEO_OUTPUT_PIX_FMT", "yuv420p"),
                    "-shortest",
                    "-movflags",
                    "+faststart",
                    output_path,
                ]

                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                if proc.stdin is None:
                    proc.kill()
                    proc.wait()
                    logger.warning("ffmpeg stdin unavailable.")
                    return False
                if fcntl is not None and hasattr(fcntl, "F_SETPIPE_SZ"):
                    with suppress(OSError):
                        fcntl.fcntl(proc.stdin.fileno(), fcntl.F_SETPIPE_SZ, 1048576)

                try:
                    ts_start = time.perf_counter()
                    for frame in frames:
                        proc.stdin.write(frame.tobytes())
                    proc.stdin.close()
                    logger.info("Wrote frames to ffmpeg stdin in %.3f s", time.perf_counter() - ts_start)
                    rc = proc.wait()
                    if rc != 0:
                        logger.warning("ffmpeg pipe save failed with return code %d", rc)
                        return False
                    return True
                except Exception:
                    proc.kill()
                    proc.wait()
                    raise
        except Exception as e:
            logger.warning("ffmpeg pipe save failed: %s", e)
            return False

    @classmethod
    def _mux_audio(
        cls,
        video_path: str,
        audio: torch.Tensor | np.ndarray,
        sample_rate: int,
    ) -> bool:
        """Mux audio into video using PyAV."""
        try:
            import av
        except ImportError:
            logger.warning("PyAV not installed; cannot mux audio. "
                           "Install with: uv pip install av")
            return False

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                out_path = os.path.join(tmpdir, "muxed.mp4")
                wav_path = os.path.join(tmpdir, "audio.wav")

                num_channels = cls._write_pcm_wav(wav_path, audio, sample_rate)
                layout = "stereo" if num_channels == 2 else "mono"

                # Open input video and audio
                input_video = av.open(video_path)
                input_audio = av.open(wav_path)

                # Create output with both streams
                output = av.open(out_path, mode="w")

                # Add video stream (copy codec from input)
                in_video_stream = input_video.streams.video[0]
                out_video_stream = output.add_stream(
                    codec_name=in_video_stream.codec_context.name,
                    rate=in_video_stream.average_rate,
                )
                out_video_stream.width = in_video_stream.width
                out_video_stream.height = in_video_stream.height
                out_video_stream.pix_fmt = in_video_stream.pix_fmt

                # Add audio stream (AAC)
                out_audio_stream = output.add_stream("aac", rate=sample_rate)
                out_audio_stream.layout = layout

                # Remux video (decode and re-encode to be safe)
                for frame in input_video.decode(video=0):
                    for packet in out_video_stream.encode(frame):
                        output.mux(packet)
                for packet in out_video_stream.encode():
                    output.mux(packet)

                # Encode audio
                for frame in input_audio.decode(audio=0):
                    frame.pts = None  # Let encoder assign PTS
                    for packet in out_audio_stream.encode(frame):
                        output.mux(packet)
                for packet in out_audio_stream.encode():
                    output.mux(packet)

                input_video.close()
                input_audio.close()
                output.close()
                shutil.move(out_path, video_path)
            return True
        except Exception as e:
            logger.warning("Audio mux failed: %s", e)
            return False

    def set_lora_adapter(self, lora_nickname: str, lora_path: str | None = None) -> None:
        self.executor.set_lora_adapter(lora_nickname, lora_path)

    def unmerge_lora_weights(self) -> None:
        """
        Use unmerged weights for inference to produce videos that align with 
        validation videos generated during training.
        """
        self.executor.unmerge_lora_weights()

    def merge_lora_weights(self) -> None:
        self.executor.merge_lora_weights()

    def shutdown(self):
        """
        Shutdown the video generator.
        """
        self.executor.shutdown()
        del self.executor


def _final_event_from_result(result: GenerationResult) -> VideoFinalEvent:
    """Build a :class:`VideoFinalEvent` from a terminal result.

    Streaming consumers prefer ``frames`` for the MSE-backed path;
    server-side consumers prefer encoded ``video_bytes``. We carry both
    — whichever the pipeline actually produced — and attach the full
    :class:`GenerationResult` so callers that want the complete object
    don't have to keep a second reference.
    """
    video_bytes: bytes | None = None
    if result.video_path and os.path.isfile(result.video_path):
        try:
            with open(result.video_path, "rb") as f:
                video_bytes = f.read()
        except OSError:
            video_bytes = None
    metadata = {
        "generation_time": result.generation_time,
        "peak_memory_mb": result.peak_memory_mb,
        "video_path": result.video_path,
    }
    return VideoFinalEvent(
        video_bytes=video_bytes,
        tensor=result.samples,
        frames=result.frames,
        metadata=metadata,
        continuation_state=result.state,
        result=result,
    )
