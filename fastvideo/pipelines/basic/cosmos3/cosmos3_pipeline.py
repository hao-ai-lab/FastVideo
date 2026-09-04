# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 pipeline (Phase 2f.2 partial).

Upstream reference:
    ``vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py`` from
    vllm-omni HEAD ``8536f5b1421f78c7df06af6d96fa195c1ceb6384``.

Phase 2f.1 landed the checkpoint-remap + scheduler-shift surface:

* ``_remap_ckpt_key`` — ported verbatim from upstream lines 319-409. This is a
  pure key-translation function with no FastVideo-specific adaptations; the
  checkpoint converter at ``scripts/checkpoint_conversion/cosmos3_convert.py``
  (Phase 5) will reuse it.
* ``_set_flow_shift`` — ported from upstream lines 498-512 with lazy scheduler
  construction so the scheduler-default-parity tests can exercise the method on
  ``__new__``-allocated instances before ``__init__`` is wired.
* ``_engine_init_flow_shift`` — exposed as a class attribute (default ``1.0``)
  so ``hasattr(__new__(...), "_engine_init_flow_shift")`` succeeds without
  running ``__init__``.

Phase 2f.2 adds the runtime call graph (this commit):

* ``diffuse`` — sequential 3-mode CFG denoising loop ported from upstream
  lines 883-1033. The CFG-Parallel branch (cfg_parallel=True) is deferred
  until FastVideo's classifier-free-guidance-world-size plumbing lands;
  ``_cfg_parallel_active`` returns False so only the sequential and no-CFG
  paths execute.
* ``forward`` — request parsing + T2I/T2V/I2V mode dispatch + flow-shift
  selection + diffusion driver + decode, ported from upstream lines
  1037-1206 (single-prompt path; the T2I num_outputs_per_prompt > 1 branch
  is preserved but uses the unimplemented ``_prepare_latents`` stub).
* ``_is_t2i_request`` / ``_get_sp_param`` / ``_cfg_parallel_active`` —
  ported from upstream lines 452-496.
* Negative-prompt constants — ported from upstream lines 51-61.

Helper methods ``_format_and_tokenize_prompts``, ``_prepare_latents``,
``_prepare_latents_i2v``, ``_set_scheduler_timesteps``, and
``_decode_latents`` are defined as ``NotImplementedError`` stubs so the
pipeline call-graph tests can monkey-patch them without an
``AttributeError``; their real implementations land in Phase 2c
(tokenizer) and Phase 2f.3+ (latent prep, scheduler timesteps, VAE decode).

``__init__`` and ``create_pipeline_stages`` remain ``NotImplementedError``
stubs — full module/stage wiring is left to a future phase.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from diffusers import UniPCMultistepScheduler

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase

logger = init_logger(__name__)

# Negative-prompt constants ported verbatim from upstream pipeline_cosmos3.py:51-61.
COSMOS3_DEFAULT_NEGATIVE_PROMPT = ""
COSMOS3_VIDEO_NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
    "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
    "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, "
    "jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, "
    "fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. "
    "Overall, the video is of poor quality.")
COSMOS3_T2V_NEGATIVE_PROMPT = COSMOS3_VIDEO_NEGATIVE_PROMPT
COSMOS3_I2V_NEGATIVE_PROMPT = COSMOS3_VIDEO_NEGATIVE_PROMPT


class Cosmos3OmniDiffusersPipeline(nn.Module, ComposedPipelineBase):
    """Cosmos3 T2V/I2V/T2I pipeline (skeleton).

    Phase 2f.1 lands only the checkpoint-remap + scheduler-shift surface needed
    by ``test_cosmos3_state_dict_keys.py`` and
    ``test_cosmos3_scheduler_default_parity.py``. Full wiring (modules,
    stages, ``diffuse()``, mode dispatch) is deferred to Phase 2f.2.

    Inherits from both ``nn.Module`` (matching upstream
    ``pipeline_cosmos3.py:197-199`` — the conftest fixture
    ``make_cosmos3_pipeline`` calls ``nn.Module.__init__`` directly) and
    ``ComposedPipelineBase`` (FastVideo's pipeline convention — sibling
    ``cosmos2_5_pipeline.py``). Phase 2f.2 can collapse either branch if
    one proves unnecessary once the full pipeline lands.

    The scheduler-parity tests build instances via
    ``Cosmos3OmniDiffusersPipeline.__new__(...)`` and rely on:

    * ``hasattr(instance, "scheduler")`` succeeding before ``_set_flow_shift``
      is called (satisfied by the ``scheduler = None`` class attribute below);
    * ``hasattr(instance, "_engine_init_flow_shift")`` succeeding (satisfied
      by the class-attribute default ``1.0``).
    """

    # --- Class attributes ---------------------------------------------------
    #
    # These exist as class attributes so ``hasattr(__new__(cls), name)`` is
    # True without ``__init__`` having run. ``__init__`` (Phase 2f.2) will
    # shadow them with instance attributes derived from the loaded scheduler.
    _engine_init_flow_shift: float = 1.0
    scheduler: Any = None
    _base_scheduler_config: Any = None
    _current_flow_shift: float | None = None

    # -- Weight loading -----------------------------------------------------

    @staticmethod
    def _remap_ckpt_key(key: str) -> str | None:
        """Remap a Diffusers transformer key to the model parameter namespace.

        Checkpoint keys arrive with a synthetic ``transformer.`` prefix from
        ``weights_sources``. The source checkpoint itself uses the Diffusers
        transformer namespace: top-level projections plus ``model.*`` for the
        Qwen3-VL backbone. UND and GEN components share each layer in the
        source and are split into separate module lists here.

        Returns the remapped name under ``transformer.``, or ``None`` to skip.

        Ported verbatim from upstream ``pipeline_cosmos3.py`` lines 319-409.
        """
        k = key
        # Strip the weights_sources prefix
        if k.startswith("transformer."):
            k = k[len("transformer."):]

        # Top-level generation components.
        if k.startswith((
                "vae2llm.",
                "llm2vae.",
                "time_embedder.",
        )):
            return f"transformer.{k}"

        # Skip lm_head
        if k.startswith("lm_head."):
            return None

        # embed_tokens / norm → language_model.*
        if k.startswith("model.embed_tokens."):
            return f"transformer.language_model.{k[len('model.'):]}"
        if k.startswith("model.norm."):
            return f"transformer.language_model.{k[len('model.'):]}"

        # norm_moe_gen → top level
        if k.startswith("model.norm_moe_gen."):
            return f"transformer.{k[len('model.'):]}"

        if not k.startswith("model.layers."):
            return None
        k = k[len("model."):]

        if not k.startswith("layers."):
            return None

        parts = k.split(".", 2)  # ['layers', '{i}', '{rest}']
        if len(parts) != 3:
            return None
        layer_idx = parts[1]
        rest = parts[2]

        und_lp = f"transformer.language_model.layers.{layer_idx}"
        gen_lp = f"transformer.gen_layers.{layer_idx}"

        _LAYER_MAP = {
            # UND attention
            "self_attn.q_proj.": f"{und_lp}.self_attn.q_proj.",
            "self_attn.k_proj.": f"{und_lp}.self_attn.k_proj.",
            "self_attn.v_proj.": f"{und_lp}.self_attn.v_proj.",
            "self_attn.o_proj.": f"{und_lp}.self_attn.o_proj.",
            "self_attn.q_norm.": f"{und_lp}.self_attn.q_norm.",
            "self_attn.k_norm.": f"{und_lp}.self_attn.k_norm.",
            # GEN attention
            "self_attn.q_proj_moe_gen.": f"{gen_lp}.cross_attention.q_proj.",
            "self_attn.k_proj_moe_gen.": f"{gen_lp}.cross_attention.k_proj.",
            "self_attn.v_proj_moe_gen.": f"{gen_lp}.cross_attention.v_proj.",
            "self_attn.o_proj_moe_gen.": f"{gen_lp}.cross_attention.o_proj.",
            "self_attn.q_norm_moe_gen.": f"{gen_lp}.cross_attention.q_norm.",
            "self_attn.k_norm_moe_gen.": f"{gen_lp}.cross_attention.k_norm.",
            # Norms
            "input_layernorm.": f"{und_lp}.input_layernorm.",
            "post_attention_layernorm.": f"{und_lp}.post_attention_layernorm.",
            "input_layernorm_moe_gen.": f"{gen_lp}.input_layernorm.",
            "post_attention_layernorm_moe_gen.": f"{gen_lp}.post_attention_layernorm.",
            # UND MLP
            "mlp.gate_proj.": f"{und_lp}.mlp.gate_proj.",
            "mlp.up_proj.": f"{und_lp}.mlp.up_proj.",
            "mlp.down_proj.": f"{und_lp}.mlp.down_proj.",
            # GEN MLP
            "mlp_moe_gen.gate_proj.": f"{gen_lp}.mlp.gate_proj.",
            "mlp_moe_gen.up_proj.": f"{gen_lp}.mlp.up_proj.",
            "mlp_moe_gen.down_proj.": f"{gen_lp}.mlp.down_proj.",
        }

        for pattern, replacement in _LAYER_MAP.items():
            if rest.startswith(pattern):
                suffix = rest[len(pattern):]
                return replacement + suffix

        return None

    # -- Scheduler control --------------------------------------------------

    def _set_flow_shift(self, target_shift: float) -> None:
        """Set the UniPC ``flow_shift`` to a concrete target value.

        Adapted from upstream ``pipeline_cosmos3.py`` lines 498-512 with one
        FastVideo-specific addition: when called on an instance built via
        ``__new__`` (i.e. ``__init__`` has not run, so
        ``self._base_scheduler_config is None``), this method lazily
        constructs a default ``UniPCMultistepScheduler`` rather than
        rebuilding from a saved config. Phase 2f.2's ``__init__`` will
        replace that lazy path with a checkpoint-loaded scheduler and
        snapshot its config into ``_base_scheduler_config`` per upstream.

        Tracking ``self._current_flow_shift`` explicitly is required because
        the previous mode may have rebuilt the scheduler — we cannot rely on
        ``self.scheduler.config.flow_shift`` reflecting the last requested
        target if a rebuild was skipped via the equality check.
        """
        target = float(target_shift)

        # Lazy path: no checkpoint-loaded base config yet (Phase 2f.1 test
        # entry via __new__). Construct a fresh UniPC scheduler at the
        # requested flow_shift so test assertions can read
        # ``self.scheduler.config.flow_shift``. Phase 2f.2's __init__ will
        # load the real scheduler from the checkpoint and overwrite both
        # ``self.scheduler`` and ``self._base_scheduler_config``.
        if self._base_scheduler_config is None:
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

        # Rebuild only if the target differs from the current shift.
        if self._current_flow_shift is not None and target == float(self._current_flow_shift):
            return
        self.scheduler = UniPCMultistepScheduler.from_config(self._base_scheduler_config, flow_shift=target)
        self._current_flow_shift = target

    # -- Request introspection ----------------------------------------------

    @staticmethod
    def _cfg_parallel_active() -> bool:
        """Return True when CFG-Parallel is enabled in the current topology.

        Upstream ``pipeline_cosmos3.py:452-457`` queries
        ``get_classifier_free_guidance_world_size() > 1``. FastVideo's
        CFG-Parallel plumbing is not wired in Phase 2f.2, so this always
        returns False; ``diffuse()`` therefore exercises only the sequential
        CFG path and the no-CFG path.
        """
        return False

    @staticmethod
    def _get_sp_param(sp: Any, key: str, default: Any = None) -> Any:
        """Read a runtime control from sampling params.

        Ported verbatim from upstream ``pipeline_cosmos3.py:459-481``.

        Order of precedence:
            1. ``sp.extra_args[key]`` — preferred path; the OpenAI image/video
               endpoints surface custom controls there.
            2. direct attribute on ``sp``.
            3. ``default``.
        """
        extra = getattr(sp, "extra_args", None)
        if isinstance(extra, dict) and extra.get(key) is not None:
            return extra[key]
        val = getattr(sp, key, None)
        if val is not None:
            return val
        return default

    @staticmethod
    def _is_t2i_request(req: Any) -> bool:
        """Detect text-to-image mode from request-level prompt modalities.

        Ported verbatim from upstream ``pipeline_cosmos3.py:483-496``.
        Raises ValueError when a prompt requests both image AND video output
        simultaneously.
        """
        if not req.prompts:
            return False
        first_prompt = req.prompts[0]
        modalities = first_prompt.get("modalities", []) if isinstance(first_prompt, dict) else []
        if modalities is None:
            modalities = []
        if isinstance(modalities, str):
            modalities = [modalities]
        if "image" in modalities and "video" in modalities:
            raise ValueError("Cosmos3 prompt modalities cannot request both image and video output.")
        return "image" in modalities

    # -- Helper stubs (real implementations land in later phases) -----------
    #
    # These exist so the pipeline call-graph tests can monkey-patch them on
    # the instance without an ``AttributeError``. Each raises
    # ``NotImplementedError`` if invoked outside a test that replaces it.

    def _format_and_tokenize_prompts(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("_format_and_tokenize_prompts lands in Phase 2c (tokenizer wiring).")

    def _prepare_latents(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("_prepare_latents lands in Phase 2f.3 (latent preparation).")

    def _prepare_latents_i2v(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("_prepare_latents_i2v lands in Phase 2f.3 (I2V latent preparation).")

    def _set_scheduler_timesteps(self, num_inference_steps: int) -> None:
        raise NotImplementedError("_set_scheduler_timesteps lands in Phase 2f.3 (scheduler timesteps).")

    def _decode_latents(self, latents: torch.Tensor) -> Any:
        raise NotImplementedError("_decode_latents lands in Phase 2f.3 (VAE decode wiring).")

    # -- Denoising loop -----------------------------------------------------

    def diffuse(
        self,
        *,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        cond_ids: torch.Tensor,
        cond_mask: torch.Tensor,
        uncond_ids: torch.Tensor,
        uncond_mask: torch.Tensor,
        guidance_scale: float,
        shared_kwargs: dict[str, Any],
        velocity_mask: torch.Tensor | None = None,
        image_latent: torch.Tensor | None = None,
        condition_latents: torch.Tensor | None = None,
        guidance_interval: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Sequential 3-mode CFG denoising loop with optional I2V conditioning.

        Ported from upstream ``pipeline_cosmos3.py:883-1033``. The
        CFG-Parallel branch (upstream lines 953-980) is deferred until
        FastVideo's classifier-free-guidance-world-size plumbing lands;
        ``_cfg_parallel_active`` returns False in Phase 2f.2 so only the
        sequential CFG branch (upstream lines 982-1019) and the no-CFG
        branch (upstream lines 1021-1031) execute.

        Cosmos3's UND pathway is text-dependent, so sequential CFG keeps
        separate K/V caches for the conditional and unconditional text
        forwards and swaps them in before each branch's transformer call.

        I2V conditioning is applied via ``_step``: ``velocity_mask`` zeros
        frame-0 noise predictions before the scheduler step, and
        ``image_latent`` is re-injected into frame 0 after each step
        (UniPC's predictor-corrector rescales the sample, so zero velocity
        alone does not preserve frame 0).

        ``guidance_interval`` (T2I) restricts CFG to timesteps inside the
        closed interval ``[lo, hi]``. Outside the interval the cond/uncond
        delta is dropped and only the cond branch executes — equivalent to
        CFG with scale=1.0 but cheaper.
        """
        do_cfg = guidance_scale > 1.0
        cfg_parallel = self._cfg_parallel_active() and do_cfg
        if cfg_parallel:
            raise NotImplementedError("Cosmos3OmniDiffusersPipeline.diffuse: CFG-Parallel branch is "
                                      "deferred; FastVideo cfg-world-size plumbing is not yet wired.")

        self.transformer.reset_cache()

        def _cfg_active_at(t: torch.Tensor) -> bool:
            if guidance_interval is None:
                return True
            t_scalar = float(t.item()) if torch.is_tensor(t) else float(t)
            lo, hi = guidance_interval
            return lo <= t_scalar <= hi

        def _step(noise_pred: torch.Tensor, t: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
            if isinstance(noise_pred, tuple):
                raise ValueError("Cosmos3 video-only diffusion received tuple predictions.")
            if velocity_mask is not None:
                noise_pred = noise_pred * velocity_mask
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if condition_latents is not None and velocity_mask is not None:
                latents = velocity_mask * latents + (1.0 - velocity_mask) * condition_latents
            elif image_latent is not None:
                latents[:, :, 0:1, :, :] = image_latent
            return latents

        if do_cfg:
            cond_cache: tuple = (None, None)
            uncond_cache: tuple = (None, None)
            for t in self.progress_bar(timesteps):
                timestep = t.unsqueeze(0)
                cfg_active = _cfg_active_at(t)

                self.transformer.cached_kv, self.transformer.cached_freqs_gen = cond_cache
                noise_cond = self.transformer(
                    hidden_states=latents,
                    timestep=timestep,
                    text_ids=cond_ids,
                    text_mask=cond_mask,
                    **shared_kwargs,
                )
                if cond_cache[0] is None:
                    cond_cache = (self.transformer.cached_kv, self.transformer.cached_freqs_gen)

                if cfg_active:
                    self.transformer.cached_kv, self.transformer.cached_freqs_gen = uncond_cache
                    noise_uncond = self.transformer(
                        hidden_states=latents,
                        timestep=timestep,
                        text_ids=uncond_ids,
                        text_mask=uncond_mask,
                        **shared_kwargs,
                    )
                    if uncond_cache[0] is None:
                        uncond_cache = (self.transformer.cached_kv, self.transformer.cached_freqs_gen)
                    noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                else:
                    noise_pred = noise_cond

                latents = _step(noise_pred, t, latents)
        else:
            for t in self.progress_bar(timesteps):
                timestep = t.unsqueeze(0)
                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep,
                    text_ids=cond_ids,
                    text_mask=cond_mask,
                    **shared_kwargs,
                )
                latents = _step(noise_pred, t, latents)

        return latents

    # -- Forward (main generation entry point) ------------------------------

    def forward(self, req: Any) -> SimpleNamespace:
        """Cosmos3 inference: request parse + mode dispatch + diffuse + decode.

        Ported from upstream ``pipeline_cosmos3.py:1037-1206`` (single-prompt
        path). FastVideo does not yet have ``OmniDiffusionRequest`` or
        ``DiffusionOutput`` dataclasses on the production path, so the
        return value is a duck-typed ``SimpleNamespace`` with an ``output``
        attribute mirroring upstream ``DiffusionOutput.output``.

        Mode selection (upstream lines 1059-1093):
          * T2I: ``"image" in modalities`` and no preprocessed image. Defaults
            to ``num_frames=1``, ``flow_shift=3.0``,
            ``guidance_interval=(400.0, 1000.0)``, 50 steps, scale=7.0.
          * I2V: ``preprocessed_image`` present and not T2I. Like T2V but
            with image conditioning.
          * T2V: otherwise. Defaults to ``num_frames=189``,
            ``flow_shift=self._engine_init_flow_shift``, no guidance
            interval, 35 steps, scale=6.0.

        Calls ``_set_flow_shift`` exactly once per request, after defaults
        are resolved and before tokenization.
        """
        if not req.prompts:
            raise ValueError("Cosmos3OmniDiffusersPipeline.forward() requires at least one prompt.")
        if len(req.prompts) > 1:
            raise ValueError("Cosmos3OmniDiffusersPipeline currently supports a single prompt per request.")

        prompt_data = req.prompts[0]
        if isinstance(prompt_data, str):
            prompt = prompt_data
            negative_prompt = None
            image_tensor = None
        else:
            prompt = prompt_data.get("prompt", "")
            negative_prompt = prompt_data.get("negative_prompt")
            additional_info = prompt_data.get("additional_information", {}) or {}
            image_tensor = additional_info.get("preprocessed_image")

        sp = req.sampling_params
        is_t2i = self._is_t2i_request(req)
        is_i2v = image_tensor is not None and not is_t2i
        if negative_prompt is None:
            if is_t2i:
                negative_prompt = COSMOS3_DEFAULT_NEGATIVE_PROMPT
            elif is_i2v:
                negative_prompt = COSMOS3_I2V_NEGATIVE_PROMPT
            else:
                negative_prompt = COSMOS3_T2V_NEGATIVE_PROMPT

        if is_t2i:
            height = sp.height or 1024
            width = sp.width or 1024
            num_frames = 1
            num_inference_steps = sp.num_inference_steps or 50
            guidance_scale = sp.guidance_scale if sp.guidance_scale else 7.0
            default_flow_shift = 3.0
            default_guidance_interval: tuple[float, float] | None = (400.0, 1000.0)
            batch_size = max(1, int(getattr(sp, "num_outputs_per_prompt", None) or 1))
        else:
            height = sp.height or 720
            width = sp.width or 1280
            num_frames = sp.num_frames or 189
            num_inference_steps = sp.num_inference_steps or 35
            guidance_scale = sp.guidance_scale if sp.guidance_scale else 6.0
            default_flow_shift = self._engine_init_flow_shift
            default_guidance_interval = None
            batch_size = 1

        flow_shift_target = float(self._get_sp_param(sp, "flow_shift", default_flow_shift))
        guidance_interval = self._get_sp_param(sp, "guidance_interval", default_guidance_interval)

        frame_rate = (self._get_sp_param(sp, "resolved_frame_rate") or self._get_sp_param(sp, "frame_rate") or 24.0)
        max_sequence_length = self._get_sp_param(sp, "max_sequence_length", 512) or 512
        use_system_prompt = bool(self._get_sp_param(sp, "use_system_prompt", False))

        self._guidance_scale = guidance_scale
        self._num_timesteps = num_inference_steps

        self._set_flow_shift(flow_shift_target)

        generator = sp.generator
        if generator is None:
            seed = sp.seed if sp.seed is not None else 42
            generator = torch.Generator(device=self.device).manual_seed(seed)

        cond_ids, cond_mask, uncond_ids, uncond_mask = self._format_and_tokenize_prompts(
            prompt,
            negative_prompt,
            num_frames,
            frame_rate,
            height,
            width,
            max_sequence_length,
            sp,
            use_system_prompt,
            is_t2i=is_t2i,
        )

        if image_tensor is not None and not is_t2i:
            latents, velocity_mask, image_latent = self._prepare_latents_i2v(
                image_tensor,
                height,
                width,
                num_frames,
                generator,
            )
            condition_latents = None
        else:
            latents = self._prepare_latents(height, width, num_frames, generator)
            velocity_mask = None
            image_latent = None
            condition_latents = None

        video_shape = (latents.shape[2], latents.shape[3], latents.shape[4])
        shared_kwargs: dict[str, Any] = dict(video_shape=video_shape, fps=frame_rate)
        if velocity_mask is not None:
            shared_kwargs["noisy_frame_mask"] = velocity_mask

        def _run_diffusion(start_latents: torch.Tensor) -> torch.Tensor:
            self._set_scheduler_timesteps(num_inference_steps)
            return self.diffuse(
                latents=start_latents,
                timesteps=self.scheduler.timesteps,
                cond_ids=cond_ids,
                cond_mask=cond_mask,
                uncond_ids=uncond_ids,
                uncond_mask=uncond_mask,
                guidance_scale=guidance_scale,
                shared_kwargs=shared_kwargs,
                velocity_mask=velocity_mask,
                image_latent=image_latent,
                condition_latents=condition_latents,
                guidance_interval=guidance_interval,
            )

        if is_t2i and batch_size > 1:
            samples = [_run_diffusion(latents)]
            for _ in range(batch_size - 1):
                next_latents = self._prepare_latents(height, width, num_frames, generator)
                samples.append(_run_diffusion(next_latents))
            latents = torch.cat(samples, dim=0)
        else:
            latents = _run_diffusion(latents)

        video = self._decode_latents(latents)
        return SimpleNamespace(output={"image": video} if is_t2i else {"video": video})

    # -- Construction stubs (full wiring deferred) --------------------------

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Cosmos3OmniDiffusersPipeline.__init__ is wired in a later phase. "
                                  "Use Cosmos3OmniDiffusersPipeline.__new__(cls) for Phase 2f unit "
                                  "tests that only exercise _remap_ckpt_key, _set_flow_shift, "
                                  "diffuse, or forward (with monkey-patched helpers).")

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        raise NotImplementedError("Cosmos3OmniDiffusersPipeline.create_pipeline_stages is wired in a later phase.")


# Entry point for pipeline registry (placeholder; full registry wiring lands
# alongside Phase 2f.2 once create_pipeline_stages and __init__ are real).
EntryClass = Cosmos3OmniDiffusersPipeline
