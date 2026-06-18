"""v2 ``VideoGenerator`` — a typed entrypoint mirroring ``fastvideo.entrypoints.VideoGenerator``,
wrapping the v2 ``Engine`` so a ``basic_dmd_new_api.py``-style script runs against the v2
(recipe, runtime) substrate + the real torch backend on a GPU box.

It consumes the OFFICIAL ``fastvideo.api`` typed configs (``GeneratorConfig`` / ``GenerationRequest`` /
``EngineConfig`` / ``OutputConfig`` / ...) — so the only delta from the upstream example is importing
``VideoGenerator`` from ``v2`` instead of ``fastvideo``. ``from_config(GeneratorConfig)`` →
``generate(GenerationRequest)`` → ``GenerationResult`` with ``.video_path``.

All torch / fastvideo / huggingface imports are lazy (inside methods) so ``import v2`` stays CPU-clean.

Scope: the three representative models brought up on GPU (Wan2.1, SF-causal Wan, LTX-2). Multi-GPU
(num_gpus>1), CPU/FSDP offload, and VSA attention are accepted for API parity but the bring-up runs
single-GPU, resident, on the TORCH_SDPA backend (no fastvideo-kernel) — see GPU_BRINGUP.md.
"""
from __future__ import annotations

import os
from typing import Any

# model_path (HF id, from fastvideo/tests/ssim/*) -> v2 model family. Wan2.1 -> wan21; Wan2.2-TI2V-5B
# -> wan2.2-ti2v (same arch classes, 48ch/16x-spatial VAE geometry); SF-causal -> wan_causal; LTX-2 ->
# ltx2. (FastWan/DMD is NOT a generic wan21 reuse — its checkpoint's to_gate_compress param mapping
# differs from the generic WanTransformer3DModel load; see V2_PORTING_STATUS.md.)
_FAMILY_BY_PATH = {
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": "wan21",
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers": "wan2.2-ti2v",
    "wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers": "wan_causal",
    "FastVideo/LTX2-Distilled-Diffusers": "ltx2",
    "FastVideo/LTX-2.3-Distilled-Diffusers": "ltx2",
}


def _resolve_family(model_path: str) -> str:
    if model_path in _FAMILY_BY_PATH:
        return _FAMILY_BY_PATH[model_path]
    p = model_path.lower()
    if "ltx" in p:
        return "ltx2"
    if "sfwan" in p or "self-forcing" in p or "causal" in p:
        return "wan_causal"
    if "wan2.2-ti2v" in p or "ti2v-5b" in p:
        return "wan2.2-ti2v"
    if "wan2.1-t2v-1.3b" in p:
        return "wan21"
    raise ValueError(
        f"v2 VideoGenerator: no v2 card mapped for model_path {model_path!r}. "
        f"Supported: {sorted(_FAMILY_BY_PATH)} (or names containing ltx / sfwan / "
        f"wan2.2-ti2v / wan2.1-t2v-1.3b).")


def _build_card_and_program(family: str):
    if family == "wan21":
        from .models.wan21 import build_wan21_card, build_wan_t2v_program
        return build_wan21_card(), build_wan_t2v_program()
    if family == "wan2.2-ti2v":
        from .models.wan21 import build_wan22_ti2v_card, build_wan_t2v_program
        return build_wan22_ti2v_card(), build_wan_t2v_program()
    if family == "wan_causal":
        from .models.wan_causal import build_wan_causal_card, build_wan_causal_program
        return build_wan_causal_card(), build_wan_causal_program()
    if family == "ltx2":
        from .models.ltx2 import build_ltx2_card, build_ltx2_program
        return build_ltx2_card(), build_ltx2_program()
    raise ValueError(f"unknown family {family!r}")


class VideoGenerator:
    """Typed t2v entrypoint over a single resident v2 ``ModelInstance`` (mirrors fastvideo's)."""

    def __init__(self, engine: Any, model_id: str, family: str) -> None:
        self._engine = engine
        self._model_id = model_id
        self._family = family

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
        family = _resolve_family(model_path)
        if getattr(config.engine, "num_gpus", 1) != 1:
            import warnings
            warnings.warn("v2 VideoGenerator bring-up is single-GPU; ignoring num_gpus>1.", stacklevel=2)

        from huggingface_hub import snapshot_download
        root = snapshot_download(model_path, revision=getattr(config, "revision", None))

        card, program = _build_card_and_program(family)
        from .models.wan21 import stamp_wan21_checkpoints   # transformer/vae/text_encoder subfolder layout
        stamp_wan21_checkpoints(card, root)

        from .card import load_card
        from .cache import CacheManager
        from .runtime import Engine
        inst = load_card(card, cache_manager=CacheManager.from_card(card))   # platform auto-detect -> cuda
        eng = Engine()
        eng.register(card.model_id, inst, program)
        return cls(eng, card.model_id, family)

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
            name = output.output_video_name or f"v2_{self._family}_{idx}"
            name = name if name.endswith(".mp4") else f"{name}.mp4"
            video_path = os.path.join(output.output_path, name)
            import imageio.v2 as imageio
            imageio.mimsave(video_path, list(vid), fps=int(fps), format="mp4")
        return GenerationResult(frames=(vid if getattr(output, "return_frames", True) else None),
                                video_path=video_path)
