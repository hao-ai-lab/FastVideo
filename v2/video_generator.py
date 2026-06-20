"""v2 ``VideoGenerator`` — a typed entrypoint mirroring ``fastvideo.entrypoints.VideoGenerator``,
wrapping the v2 ``Engine`` so an upstream-style script runs against the v2 (recipe, runtime) substrate
and the real torch backend on a GPU box.

It consumes the official ``fastvideo.api`` typed configs (``GeneratorConfig`` / ``GenerationRequest`` /
``EngineConfig`` / ``OutputConfig`` / ...), so the only delta from the upstream example is importing
``VideoGenerator`` from ``v2`` instead of ``fastvideo``. Flow: ``from_config(GeneratorConfig)`` →
``generate(GenerationRequest)`` → ``GenerationResult`` with ``.video_path``.

All torch / fastvideo / huggingface imports are lazy (inside methods) so ``import v2`` stays CPU-clean.

Model dispatch lives in the shared ``v2/registry.py`` (HF-id registry primary + architecture-inference
fallback) so a CLI or server resolve identically. Multi-GPU / CPU-FSDP offload / VSA are accepted for
API parity but the bring-up runs single-GPU, resident, on TORCH_SDPA (no fastvideo-kernel) — see
GPU_BRINGUP.md.
"""
from __future__ import annotations

import os
from typing import Any

# Model dispatch (HF-id registry primary + architecture-inference fallback) lives in the shared
# ``v2/registry.py`` so every entrypoint (this VideoGenerator, a CLI, the server) resolves identically.


def _resolve_default(key: str, generic: Any, kwargs: dict, sp: Any, sd: Any, card_attr: str | None = None) -> Any:
    """Resolve one sampling param with precedence kwargs > SamplingParam override > card
    ``SamplingDefaults`` > generic fallback. ``card_attr`` aliases the card field name when it differs
    from the request key (e.g. ``num_inference_steps`` -> ``SamplingDefaults.num_steps``). Pure and
    torch-free, so it's unit-testable without a GPU."""
    if key in kwargs and kwargs[key] is not None:
        return kwargs[key]
    if sp is not None and getattr(sp, key, None) is not None:
        return getattr(sp, key)
    if sd is not None:
        cv = getattr(sd, card_attr or key, None)
        if cv is not None:
            return cv
    return generic


class VideoGenerator:
    """Typed t2v entrypoint over a single resident v2 ``ModelInstance`` (mirrors fastvideo's)."""

    def __init__(self,
                 engine: Any,
                 model_id: str,
                 *,
                 instance: Any = None,
                 supports_av: bool = False,
                 card: Any = None) -> None:
        self._engine = engine
        self._model_id = model_id
        self._inst = instance  # the resident ModelInstance (to read e.g. the audio sample rate)
        self._supports_av = supports_av  # model emits joint video+sound (LTX-2.3 T2VS) -> auto-request audio
        self._card = card  # the v2 ModelCard (for per-model sampling_defaults)

    # --------------------------------------------------------------------- #
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs: Any) -> VideoGenerator:
        """Convenience constructor (mirrors fastvideo's): accepts the legacy ``from_pretrained`` kwargs
        (``num_gpus`` / ``use_fsdp_inference`` / ``*_cpu_offload`` / ``pin_cpu_memory`` / ``VSA_sparsity``
        / ...). The v2 bring-up runs single-GPU, resident, on SDPA, so these are accepted for parity but
        not all applied."""
        from v2._vendor.api import EngineConfig, GeneratorConfig, OffloadConfig
        engine = EngineConfig(num_gpus=int(kwargs.get("num_gpus", 1)),
                              use_fsdp_inference=bool(kwargs.get("use_fsdp_inference", False)),
                              offload=OffloadConfig(text_encoder=bool(kwargs.get("text_encoder_cpu_offload", False)),
                                                    dit=bool(kwargs.get("dit_cpu_offload", False)),
                                                    vae=bool(kwargs.get("vae_cpu_offload", False)),
                                                    pin_cpu_memory=bool(kwargs.get("pin_cpu_memory", False))))
        return cls.from_config(GeneratorConfig(model_path=model_path, engine=engine))

    @classmethod
    def from_config(cls, config: Any) -> VideoGenerator:
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

        from v2.registry import resolve
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
        root = model_path if local else snapshot_download(model_path, revision=rev)  # full weights (cached)
        card, program = build_card(), build_program()

        from v2.recipes.wan21 import stamp_wan21_checkpoints  # transformer/vae/text_encoder subfolder layout
        stamp_wan21_checkpoints(card, root)

        from v2.core.enums import Capability
        from v2.core.card import load_card
        from v2.runtime.cache import CacheManager
        from v2.runtime import Engine
        inst = load_card(card, cache_manager=CacheManager.from_card(card))  # platform auto-detect -> cuda
        eng = Engine()
        eng.register(card.model_id, inst, program)
        # A model that advertises TEXT_TO_VIDEO_SOUND (LTX-2.3) generates joint video+audio; the
        # entrypoint then issues a T2VS request by default so a plain generate() yields both modalities.
        supports_av = card.capabilities.has(Capability.TEXT_TO_VIDEO_SOUND)
        return cls(eng, card.model_id, instance=inst, supports_av=supports_av, card=card)

    def shutdown(self) -> None:
        """API parity with ``fastvideo.VideoGenerator.shutdown()``. The v2 bring-up holds a single
        resident instance that is freed at process exit; there is nothing to tear down explicitly."""

    # --------------------------------------------------------------------- #
    def generate(self, request: Any, *, want_audio: bool | None = None) -> Any:
        """``request``: ``fastvideo.api.GenerationRequest``. Returns a ``GenerationResult`` (or a list
        for multi-prompt) with ``.video_path`` (saved mp4) and optionally ``.frames``.

        ``want_audio``: ``None`` (default) auto-enables sound for an A/V model (LTX-2.3 → T2VS, both
        video+audio in one joint pass); ``True``/``False`` force it on/off. For a video-only model the
        auto resolves to video-only (T2V), unchanged."""
        from v2.core.request import DiffusionParams, OutputSpec, TaskType, make_request
        s = request.sampling
        prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
        audio = self._supports_av if want_audio is None else bool(want_audio)
        results = []
        for idx, prompt in enumerate(prompts):
            diff = DiffusionParams(num_steps=s.num_inference_steps,
                                   seed=s.seed,
                                   num_frames=s.num_frames,
                                   height=s.height,
                                   width=s.width,
                                   guidance_scale=s.guidance_scale,
                                   negative_prompt=request.negative_prompt or "",
                                   sigmas=tuple(s.sigmas) if getattr(s, "sigmas", None) else None)
            if audio:  # joint text->video+sound: ask for both modalities so the audio branch runs
                req = make_request(TaskType.T2VS,
                                   self._model_id,
                                   prompt,
                                   diffusion=diff,
                                   outputs=OutputSpec(modalities=frozenset({"video", "audio"})))
            else:
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
        save_video / return_frames / audio (None=auto for an A/V model, True/False to force; the audio
        is saved as a sibling ``.wav`` next to the mp4)."""
        from v2._vendor.api import GenerationRequest, OutputConfig, SamplingConfig
        audio = kwargs.pop("audio", None)  # None -> auto (by model capability), True/False -> force
        sp = sampling_param
        # Per-model defaults (LTX-2 wants 8/30 steps + its own res, Wan2.2-TI2V 704x1280@24fps, etc.) come
        # from the resolved card; precedence is kwargs > SamplingParam > card defaults > generic fallback.
        sd = getattr(self._card, "sampling_defaults", None) if self._card is not None else None

        def pick(key: str, default: Any, card_attr: str | None = None) -> Any:
            return _resolve_default(key, default, kwargs, sp, sd, card_attr)

        sampling = SamplingConfig(num_inference_steps=int(pick("num_inference_steps", 30, "num_steps")),
                                  seed=int(pick("seed", 1024)),
                                  num_frames=int(pick("num_frames", 25)),
                                  height=int(pick("height", 480)),
                                  width=int(pick("width", 832)),
                                  guidance_scale=float(pick("guidance_scale", 5.0)),
                                  fps=int(pick("fps", 16)),
                                  sigmas=pick("sigmas", None))
        output = OutputConfig(output_path=kwargs.get("output_path", "outputs/"),
                              output_video_name=kwargs.get("output_video_name"),
                              save_video=bool(kwargs.get("save_video", True)),
                              return_frames=bool(kwargs.get("return_frames", False)))
        req = GenerationRequest(prompt=prompt,
                                negative_prompt=pick("negative_prompt", None),
                                sampling=sampling,
                                output=output)
        return self.generate(req, want_audio=audio)

    # --------------------------------------------------------------------- #
    def _result(self, out: Any, output: Any, fps: int, idx: int) -> Any:
        import numpy as np
        from v2._vendor.api import GenerationResult
        arts = out.artifacts
        save = bool(getattr(output, "save_video", True))
        out_dir = getattr(output, "output_path", "outputs/")
        stem = (output.output_video_name or f"v2_{self._model_id}_{idx}")
        for _ext in (".mp4", ".png", ".wav"):
            stem = stem[:-len(_ext)] if stem.endswith(_ext) else stem

        def _to_uint8(a: Any) -> Any:  # [-1,1] float -> [0,255] uint8
            return ((np.clip(np.asarray(a, dtype="float32"), -1.0, 1.0) + 1.0) / 2.0 * 255.0).astype("uint8")

        # Modality-aware: a model emits exactly one of video / image / (and/or) audio. T2V/I2V/T2VS carry
        # a "video" artifact; T2I (SD3.5 / FLUX.2) an "image" TensorArtifact; audio (Stable Audio) only
        # the AudioArtifact. Guard each so an audio/image-only result does not KeyError on "video".
        vid = video_path = image_path = None
        if "video" in arts:
            vid = _to_uint8(arts["video"].frames)  # [C,T,H,W] -> [T,H,W,C]
            vid = vid.transpose(1, 2, 3, 0) if vid.ndim == 4 else vid
            if save:
                os.makedirs(out_dir, exist_ok=True)
                video_path = os.path.join(out_dir, f"{stem}.mp4")
                import imageio.v2 as imageio
                imageio.mimsave(video_path, list(vid), fps=int(fps), format="mp4")
        elif "image" in arts:  # single decoded image (T2I)
            art = arts["image"]
            img = _to_uint8(getattr(art, "tensor", getattr(art, "frames", art)))
            img = np.squeeze(img)
            if img.ndim == 4:
                img = img[0]
            if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
                img = img.transpose(1, 2, 0)  # [C,H,W] -> [H,W,C]
            vid = img
            if save:
                os.makedirs(out_dir, exist_ok=True)
                image_path = os.path.join(out_dir, f"{stem}.png")
                import imageio.v2 as imageio
                imageio.imwrite(image_path, img)
        # Audio (T2VS / Stable Audio): the AudioArtifact carries the raw stereo waveform; save a sibling
        # .wav at the vocoder's real rate (read from the built audio_vae adapter — the artifact's default
        # rate is a placeholder). A plain T2V request produces no audio artifact, so this is skipped.
        audio = audio_sr = audio_path = None
        art = out.artifacts.get("audio") if hasattr(out.artifacts, "get") else None
        wav = getattr(art, "samples", None) if art is not None else None
        if wav is not None:
            audio = np.asarray(wav, dtype="float32")
            audio_sr = self._audio_sample_rate(default=int(getattr(art, "sample_rate", 24000) or 24000))
            if save:
                audio_path = os.path.join(out_dir, f"{stem}.wav")
                self._write_wav(audio_path, audio, audio_sr)
        res = GenerationResult(frames=(vid if getattr(output, "return_frames", True) else None),
                               video_path=video_path,
                               audio=audio,
                               audio_sample_rate=audio_sr)
        if audio_path:
            res.extra["audio_path"] = audio_path
        if image_path:
            res.extra["image_path"] = image_path
        return res

    def _audio_sample_rate(self, default: int = 24000) -> int:
        """Read the true output rate off the built audio_vae adapter (LTX-2 vocoder = 24000); fall back
        to ``default`` if the model has no audio component or the adapter doesn't expose it."""
        try:
            return int(getattr(self._inst.component("audio_vae"), "sample_rate", default))
        except Exception:
            return int(default)

    @staticmethod
    def _write_wav(path: str, wav: Any, sample_rate: int) -> None:
        """Write ``wav`` ([samples] or [channels, samples] / [samples, channels], float in [-1,1]) to a
        WAV at ``sample_rate`` (scipy: IEEE-float WAV, no extra deps beyond the installed stack)."""
        import numpy as np
        from scipy.io import wavfile
        a = np.asarray(wav, dtype="float32")
        if a.ndim == 2 and a.shape[0] in (1, 2) and a.shape[0] < a.shape[1]:
            a = a.T  # [channels, samples] -> [samples, channels] (scipy convention)
        wavfile.write(path, int(sample_rate), np.clip(a, -1.0, 1.0))
