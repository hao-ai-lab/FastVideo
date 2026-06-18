"""v2 ``VideoGenerator`` — a typed entrypoint mirroring ``fastvideo.entrypoints.VideoGenerator``,
wrapping the v2 ``Engine`` so a ``basic_dmd_new_api.py``-style script runs against the v2
(recipe, runtime) substrate + the real torch backend on a GPU box.

It consumes the OFFICIAL ``fastvideo.api`` typed configs (``GeneratorConfig`` / ``GenerationRequest`` /
``EngineConfig`` / ``OutputConfig`` / ...) — so the only delta from the upstream example is importing
``VideoGenerator`` from ``v2`` instead of ``fastvideo``. ``from_config(GeneratorConfig)`` →
``generate(GenerationRequest)`` → ``GenerationResult`` with ``.video_path``.

All torch / fastvideo / huggingface imports are lazy (inside methods) so ``import v2`` stays CPU-clean.

Model dispatch lives in the SHARED ``v2/registry.py`` (so a CLI / server resolve identically): an
explicit HF-id registry PRIMARY + architecture inference FALLBACK, mirroring fastvideo's
``fastvideo/registry.py``. Wired: Wan2.1, Wan2.2-TI2V-5B, Wan2.2-A14B (MoE), SF-causal Wan, LTX-2
(two-stage distilled + single-stage base/2.3). Multi-GPU / CPU-FSDP offload / VSA are accepted for API
parity but the bring-up runs single-GPU, resident, on TORCH_SDPA (no fastvideo-kernel) — see GPU_BRINGUP.md.
"""
from __future__ import annotations

import os
from typing import Any

# Model dispatch (HF-id registry primary + architecture-inference fallback) lives in the shared
# ``v2/registry.py`` so every entrypoint (this VideoGenerator, a CLI, the server) resolves identically.


class VideoGenerator:
    """Typed t2v entrypoint over a single resident v2 ``ModelInstance`` (mirrors fastvideo's)."""

    def __init__(self, engine: Any, model_id: str) -> None:
        self._engine = engine
        self._model_id = model_id

    # --------------------------------------------------------------------- #
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs: Any) -> "VideoGenerator":
        """Convenience constructor (mirrors fastvideo's): accepts the legacy ``from_pretrained`` kwargs
        (``num_gpus`` / ``use_fsdp_inference`` / ``*_cpu_offload`` / ``pin_cpu_memory`` / ``VSA_sparsity``
        / ...). The v2 bring-up runs single-GPU, resident, on SDPA, so these are accepted for parity but
        not all applied."""
        from fastvideo.api import EngineConfig, GeneratorConfig, OffloadConfig
        engine = EngineConfig(
            num_gpus=int(kwargs.get("num_gpus", 1)),
            use_fsdp_inference=bool(kwargs.get("use_fsdp_inference", False)),
            offload=OffloadConfig(
                text_encoder=bool(kwargs.get("text_encoder_cpu_offload", False)),
                dit=bool(kwargs.get("dit_cpu_offload", False)),
                vae=bool(kwargs.get("vae_cpu_offload", False)),
                pin_cpu_memory=bool(kwargs.get("pin_cpu_memory", False))))
        return cls.from_config(GeneratorConfig(model_path=model_path, engine=engine))

    @classmethod
    def from_config(cls, config: Any) -> "VideoGenerator":
        """``config``: ``fastvideo.api.GeneratorConfig``. Resolves the v2 card from ``model_path``,
        stamps the (downloaded) checkpoint onto it, and registers it on a fresh v2 ``Engine`` whose
        ``Platform.detect()`` returns cuda on a GPU box."""
        # The v2 bring-up runs dense attention via SDPA (fastvideo-kernel / VSA not built). Honor an
        # explicit override but default to SDPA so PipelineSelection.experimental VSA knobs don't break.
        os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")

        model_path = config.model_path
        if getattr(config.engine, "num_gpus", 1) != 1:
            import warnings
            warnings.warn("v2 VideoGenerator bring-up is single-GPU; ignoring num_gpus>1.", stacklevel=2)

        from huggingface_hub import snapshot_download

        from .registry import resolve
        rev = getattr(config, "revision", None)
        local = os.path.isdir(model_path)
        # Resolve card+program via the shared registry (v2/registry.py): a registered HF-id resolves with
        # no download; an unregistered id / local path falls back to architecture inference, which needs
        # the configs (the local dir, or a cheap ``*.json`` snapshot fetched BEFORE the full — possibly
        # 100GB+ — weight download, so an unsupported arch is rejected early).
        try:
            build_card, build_program = resolve(model_path)
        except ValueError:
            cfg_root = model_path if local else snapshot_download(
                model_path, revision=rev, allow_patterns=["*.json", "**/*.json"])
            build_card, build_program = resolve(model_path, cfg_root)
        root = model_path if local else snapshot_download(model_path, revision=rev)   # full weights (cached)
        card, program = build_card(), build_program()

        from .models.wan21 import stamp_wan21_checkpoints   # transformer/vae/text_encoder subfolder layout
        stamp_wan21_checkpoints(card, root)

        from .card import load_card
        from .cache import CacheManager
        from .runtime import Engine
        inst = load_card(card, cache_manager=CacheManager.from_card(card))   # platform auto-detect -> cuda
        eng = Engine()
        eng.register(card.model_id, inst, program)
        return cls(eng, card.model_id)

    def shutdown(self) -> None:
        """API parity with ``fastvideo.VideoGenerator.shutdown()``. The v2 bring-up holds a single
        resident instance that is freed at process exit; there is nothing to tear down explicitly."""

    # --------------------------------------------------------------------- #
    def generate(self, request: Any) -> Any:
        """``request``: ``fastvideo.api.GenerationRequest``. Returns a ``GenerationResult`` (or a list
        for multi-prompt) with ``.video_path`` (saved mp4) and optionally ``.frames``."""
        from .request import DiffusionParams, TaskType, make_request
        s = request.sampling
        prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
        results = []
        for idx, prompt in enumerate(prompts):
            diff = DiffusionParams(
                num_steps=s.num_inference_steps, seed=s.seed, num_frames=s.num_frames,
                height=s.height, width=s.width, guidance_scale=s.guidance_scale,
                negative_prompt=request.negative_prompt or "",
                sigmas=tuple(s.sigmas) if getattr(s, "sigmas", None) else None)
            req = make_request(TaskType.T2V, self._model_id, prompt, diffusion=diff)
            out = self._engine.run(req)
            results.append(self._result(out, request.output, s.fps, idx))
        return results[0] if len(results) == 1 else results

    # --------------------------------------------------------------------- #
    def generate_video(self, prompt: Any = None, sampling_param: Any = None, **kwargs: Any) -> Any:
        """Convenience API mirroring ``fastvideo.VideoGenerator.generate_video``. Builds a
        ``GenerationRequest`` from loose kwargs (+ an optional ``SamplingParam`` override) and calls
        ``generate()``. kwargs: num_inference_steps / seed / num_frames / height / width /
        guidance_scale / fps / sigmas / negative_prompt / output_path / output_video_name /
        save_video / return_frames."""
        from fastvideo.api import GenerationRequest, OutputConfig, SamplingConfig
        sp = sampling_param

        def pick(key: str, default: Any) -> Any:
            if key in kwargs and kwargs[key] is not None:
                return kwargs[key]
            if sp is not None and getattr(sp, key, None) is not None:
                return getattr(sp, key)
            return default

        sampling = SamplingConfig(
            num_inference_steps=int(pick("num_inference_steps", 30)),
            seed=int(pick("seed", 1024)),
            num_frames=int(pick("num_frames", 25)),
            height=int(pick("height", 480)),
            width=int(pick("width", 832)),
            guidance_scale=float(pick("guidance_scale", 5.0)),
            fps=int(pick("fps", 16)),
            sigmas=pick("sigmas", None))
        output = OutputConfig(
            output_path=kwargs.get("output_path", "outputs/"),
            output_video_name=kwargs.get("output_video_name"),
            save_video=bool(kwargs.get("save_video", True)),
            return_frames=bool(kwargs.get("return_frames", False)))
        req = GenerationRequest(prompt=prompt, negative_prompt=pick("negative_prompt", None),
                               sampling=sampling, output=output)
        return self.generate(req)

    # --------------------------------------------------------------------- #
    def _result(self, out: Any, output: Any, fps: int, idx: int) -> Any:
        import numpy as np
        from fastvideo.api import GenerationResult
        frames = np.asarray(out.artifacts["video"].frames, dtype="float32")   # [C,T,H,W] in [-1,1]
        # -> [T,H,W,C] uint8 (the on-disk / return convention)
        vid = ((np.clip(frames, -1.0, 1.0) + 1.0) / 2.0 * 255.0).astype("uint8")
        vid = vid.transpose(1, 2, 3, 0) if vid.ndim == 4 else vid
        video_path = None
        if getattr(output, "save_video", True):
            os.makedirs(output.output_path, exist_ok=True)
            name = output.output_video_name or f"v2_{self._model_id}_{idx}"
            name = name if name.endswith(".mp4") else f"{name}.mp4"
            video_path = os.path.join(output.output_path, name)
            import imageio.v2 as imageio
            imageio.mimsave(video_path, list(vid), fps=int(fps), format="mp4")
        return GenerationResult(frames=(vid if getattr(output, "return_frames", True) else None),
                                video_path=video_path)
