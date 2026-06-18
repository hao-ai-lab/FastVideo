"""v2 ``VideoGenerator`` — a typed entrypoint mirroring ``fastvideo.entrypoints.VideoGenerator``,
wrapping the v2 ``Engine`` so a ``basic_dmd_new_api.py``-style script runs against the v2
(recipe, runtime) substrate + the real torch backend on a GPU box.

It consumes the OFFICIAL ``fastvideo.api`` typed configs (``GeneratorConfig`` / ``GenerationRequest`` /
``EngineConfig`` / ``OutputConfig`` / ...) — so the only delta from the upstream example is importing
``VideoGenerator`` from ``v2`` instead of ``fastvideo``. ``from_config(GeneratorConfig)`` →
``generate(GenerationRequest)`` → ``GenerationResult`` with ``.video_path``.

All torch / fastvideo / huggingface imports are lazy (inside methods) so ``import v2`` stays CPU-clean.

Model dispatch is registry-style: ``from_config`` reads the checkpoint's architecture (the pipeline /
transformer / VAE class names) and picks the v2 card — no hardcoded repo-id table (mirrors fastvideo's
``get_pipeline_config_cls_from_name``). Wired families: Wan2.1, Wan2.2-TI2V-5B, Wan2.2-A14B (MoE),
SF-causal Wan, LTX-2. Multi-GPU / CPU-FSDP offload / VSA are accepted for API parity but the bring-up
runs single-GPU, resident, on the TORCH_SDPA backend (no fastvideo-kernel) — see GPU_BRINGUP.md.
"""
from __future__ import annotations

import os
from typing import Any

def _read_arch_signature(root: str) -> dict:
    """Read a diffusers checkpoint's configs into an architecture signature. Dispatch keys are the
    *class names* the model declares (pipeline / transformer / VAE) plus the few config fields that
    distinguish same-class variants (VAE ``z_dim``, a second transformer / ``boundary_ratio`` for MoE)
    — the same information fastvideo's registry (``get_pipeline_config_cls_from_name``) keys on, not
    the repo id. ``root`` is a local diffusers dir (or a snapshot of just the ``*.json`` configs)."""
    import json
    import os

    def _load(*parts: str) -> dict:
        p = os.path.join(root, *parts)
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
        return {}

    mi, tcfg, vcfg = _load("model_index.json"), _load("transformer", "config.json"), _load("vae", "config.json")
    return {
        "pipeline": mi.get("_class_name"),
        "boundary_ratio": mi.get("boundary_ratio"),
        "has_transformer_2": os.path.isdir(os.path.join(root, "transformer_2")),
        "has_spatial_upsampler": os.path.isdir(os.path.join(root, "spatial_upsampler")) or "spatial_upsampler" in mi,
        "transformer_cls": tcfg.get("_class_name"),
        "in_channels": tcfg.get("in_channels"),
        "vae_z_dim": vcfg.get("z_dim", vcfg.get("latent_channels")),
    }


def _select_builders(sig: dict):
    """Map an architecture signature -> (build_card, build_program). Which v2 card a checkpoint uses is
    determined by its transformer/pipeline/VAE classes — exactly like fastvideo's registry — so a local
    path, a renamed repo, or a new distilled variant of a known arch all resolve correctly with no
    hardcoded HF-id table. New families are added here by class, next to the card that handles them."""
    tr, pipe = sig.get("transformer_cls"), sig.get("pipeline")
    if tr == "LTX2Transformer3DModel":
        from .models.ltx2 import (
            build_ltx2_base_card,
            build_ltx2_base_program,
            build_ltx2_card,
            build_ltx2_program,
        )
        if sig.get("has_spatial_upsampler"):
            return build_ltx2_card, build_ltx2_program          # distilled two-stage (base→upsample→refine)
        return build_ltx2_base_card, build_ltx2_base_program    # single-stage base model
    if tr == "CausalWanTransformer3DModel":
        from .models.wan_causal import build_wan_causal_card, build_wan_causal_program
        return build_wan_causal_card, build_wan_causal_program
    if tr == "WanTransformer3DModel":
        from .models.wan21 import (
            build_wan21_card,
            build_wan22_a14b_card,
            build_wan22_ti2v_card,
            build_wan_t2v_program,
        )
        if pipe == "WanDMDPipeline":   # FastWan: detected by pipeline class -> precise, not a load crash
            raise ValueError(
                "v2 VideoGenerator: WanDMD/FastWan is not supported via the generic Wan path — its "
                "checkpoint's to_gate_compress param mapping differs from the generic WanTransformer3DModel "
                "load. See examples/inference/basic/V2_PORTING_STATUS.md.")
        if sig.get("has_transformer_2") or sig.get("boundary_ratio"):
            return build_wan22_a14b_card, build_wan_t2v_program     # Wan2.2 MoE (two experts)
        if sig.get("vae_z_dim") == 48:
            return build_wan22_ti2v_card, build_wan_t2v_program     # Wan2.2-TI2V-5B (z_dim=48 VAE)
        return build_wan21_card, build_wan_t2v_program              # Wan2.1
    raise ValueError(
        f"v2 VideoGenerator: unsupported architecture (transformer={tr!r}, pipeline={pipe!r}). Supported "
        f"transformers: WanTransformer3DModel / CausalWanTransformer3DModel / LTX2Transformer3DModel. "
        f"See examples/inference/basic/V2_PORTING_STATUS.md.")


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
        rev = getattr(config, "revision", None)
        # Registry-style dispatch on the checkpoint's architecture (class names), not the repo id. Read
        # just the configs first (a local dir, or a cheap ``*.json`` snapshot) so an unsupported arch is
        # rejected BEFORE the full (possibly 100GB+) weight download; then fetch the weights (cached).
        if os.path.isdir(model_path):
            root = model_path
        else:
            root = snapshot_download(model_path, revision=rev, allow_patterns=["*.json", "**/*.json"])
        build_card, build_program = _select_builders(_read_arch_signature(root))
        if not os.path.isdir(model_path):
            root = snapshot_download(model_path, revision=rev)   # full weights (idempotent, reuses cache)
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
