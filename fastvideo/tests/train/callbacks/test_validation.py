# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for :mod:`fastvideo.train.callbacks.validation`.

Covers the parts of ``ValidationCallback`` that don't need a real
pipeline or distributed init:

* constructor type coercions and defaults,
* ``on_validation_begin`` gating logic (every_steps + modulo),
* ``_find_ema_callback`` lookup via ``_callback_dict``,
* ``state_dict`` / ``load_state_dict`` rng round-trip.

The heavy ``_run_validation`` path needs a real diffusion pipeline plus
distributed init and is exercised by Phase 2/3 tests.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fastvideo.api.sampling_param import SamplingParam
from fastvideo.train.callbacks.callback import CallbackDict
from fastvideo.train.callbacks.ema import EMACallback
from fastvideo.train.callbacks.validation import (
    DEFAULT_VALIDATION_VBENCH_METRICS,
    SYNTHETIC_OPTICAL_FLOW_LOG_KEYS,
    SYNTHETIC_OPTICAL_FLOW_METRIC,
    ValidationCallback,
    _ValidationMetricStats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_PIPE_TARGET = "fastvideo.pipelines.basic.wan.wan_pipeline.WanPipeline"


def _make_callback(
    *,
    every_steps: int = 100,
    sampling_steps: list[int] | None = None,
    guidance_scale: float | None = None,
    num_frames: int | None = None,
    num_videos_per_prompt: int = 1,
    use_validation_media_conditioning: bool = True,
    sampling_timesteps: list[int] | None = None,
    output_dir: str | None = None,
    overlay_actions: bool = False,
    run_at_start: bool = True,
) -> ValidationCallback:
    """Construct a callback with the shared schedule and media-test defaults."""
    return ValidationCallback(
        pipeline_target=_PIPE_TARGET,
        dataset_file="/tmp/does_not_exist.json",
        every_steps=every_steps,
        sampling_steps=sampling_steps,
        guidance_scale=guidance_scale,
        num_frames=num_frames,
        num_videos_per_prompt=num_videos_per_prompt,
        use_validation_media_conditioning=use_validation_media_conditioning,
        sampling_timesteps=sampling_timesteps,
        output_dir=output_dir,
        overlay_actions=overlay_actions,
        run_at_start=run_at_start,
    )


# ---------------------------------------------------------------------------
# A. Constructor coercions / defaults
# ---------------------------------------------------------------------------


class TestConstructor:

    def test_defaults(self) -> None:
        """Verify defaults preserve baseline validation and media conditioning."""
        cb = _make_callback()
        assert cb.pipeline_target == _PIPE_TARGET
        assert cb.dataset_file == "/tmp/does_not_exist.json"
        assert cb.every_steps == 100
        assert cb.sampling_steps == [40]
        assert cb.guidance_scale is None
        assert cb.num_frames is None
        assert cb.num_videos_per_prompt == 1
        assert cb.use_validation_media_conditioning is True
        assert cb.run_at_start is True
        assert cb.sampling_timesteps is None
        assert cb.output_dir is None
        assert cb.overlay_actions is False
        assert cb.offload_training_state is False
        assert cb.unload_pipeline_after_validation is False
        assert cb.attn_qat_infer is False
        # Lazy fields not yet populated.
        assert cb._pipeline is None
        assert cb._sampling_param is None
        assert cb.validation_random_generator is None
        assert cb.metrics_config.enabled is False

    def test_string_inputs_are_coerced(self) -> None:
        """Verify YAML-compatible scalar strings become callback value types."""
        # YAML often produces strings for numeric fields; the
        # constructor must coerce them.
        cb = ValidationCallback(
            pipeline_target=_PIPE_TARGET,
            dataset_file="x.json",
            every_steps="50",  # type: ignore[arg-type]
            sampling_steps=["20", "40"],  # type: ignore[arg-type]
            guidance_scale="4.5",  # type: ignore[arg-type]
            num_frames="77",  # type: ignore[arg-type]
            num_videos_per_prompt="2",  # type: ignore[arg-type]
            use_validation_media_conditioning="false",  # type: ignore[arg-type]
            sampling_timesteps=["1000", "500"],
            run_at_start="false",  # type: ignore[arg-type]
            overlay_actions=1,  # type: ignore[arg-type]
            offload_training_state="1",  # type: ignore[arg-type]
            unload_pipeline_after_validation="false",  # type: ignore[arg-type]
            attn_qat_infer="true",  # type: ignore[arg-type]
        )
        assert cb.every_steps == 50
        assert cb.sampling_steps == [20, 40]
        assert cb.guidance_scale == 4.5
        assert cb.num_frames == 77
        assert cb.num_videos_per_prompt == 2
        assert cb.use_validation_media_conditioning is False
        assert cb.run_at_start is False
        assert cb.sampling_timesteps == [1000, 500]
        assert cb.overlay_actions is True
        assert cb.offload_training_state is True
        assert cb.unload_pipeline_after_validation is False
        assert cb.attn_qat_infer is True

    def test_init_rejects_nonpositive_video_count(self) -> None:
        """Verify every prompt requests at least one generated video."""
        with pytest.raises(ValueError, match="num_videos_per_prompt must be positive"):
            _make_callback(num_videos_per_prompt=0)

    def test_pipeline_kwargs_collected(self) -> None:
        cb = ValidationCallback(
            pipeline_target=_PIPE_TARGET,
            dataset_file="x.json",
            extra_arg=123,
            another="value",
        )
        # Unknown kwargs are stashed for the pipeline factory.
        assert cb.pipeline_kwargs == {
            "extra_arg": 123,
            "another": "value",
        }

    def test_metrics_true_uses_default_vbench_subset(self) -> None:
        cb = ValidationCallback(
            pipeline_target=_PIPE_TARGET,
            dataset_file="x.json",
            metrics=True,
        )
        assert cb.metrics_config.enabled is True
        assert cb.metrics_config.names == DEFAULT_VALIDATION_VBENCH_METRICS
        assert "metrics" not in cb.pipeline_kwargs

    def test_metrics_mapping_is_coerced(self) -> None:
        cb = ValidationCallback(
            pipeline_target=_PIPE_TARGET,
            dataset_file="x.json",
            metrics={
                "names": "vbench.aesthetic_quality",
                "device": "cpu",
                "calibration_path": "/tmp/calibration.json",
                "mouse_pitch_sign": "-1",
                "skip_missing_deps": False,
                "strict": True,
                "unload_after_validation": False,
                "loader_threads": "2",
                "prefetch_factor": "3",
                "log_prefix": "custom/validation",
            },
        )
        assert cb.metrics_config.enabled is True
        assert cb.metrics_config.names == ["vbench.aesthetic_quality"]
        assert cb.metrics_config.device == "cpu"
        assert cb.metrics_config.calibration_path == "/tmp/calibration.json"
        assert cb.metrics_config.mouse_pitch_sign == -1
        assert cb.metrics_config.skip_missing_deps is False
        assert cb.metrics_config.strict is True
        assert cb.metrics_config.unload_after_validation is False
        assert cb.metrics_config.loader_threads == 2
        assert cb.metrics_config.prefetch_factor == 3
        assert cb.metrics_config.log_prefix == "custom/validation"


# ---------------------------------------------------------------------------
# B. on_validation_begin gating
# ---------------------------------------------------------------------------


class _NoRunValidation(ValidationCallback):
    """Subclass that records ``_run_validation`` calls instead of
    actually running them — lets us assert the gating logic without a
    real pipeline."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.run_calls: list[int] = []

    def _run_validation(self, method, step: int) -> None:  # type: ignore[override]
        self.run_calls.append(step)


def _make_recording(**kwargs) -> _NoRunValidation:
    return _NoRunValidation(
        pipeline_target=_PIPE_TARGET,
        dataset_file="x.json",
        **kwargs,
    )


class TestOnValidationBegin:

    def test_skipped_when_every_steps_zero(self) -> None:
        cb = _make_recording(every_steps=0)
        cb.on_validation_begin(method=None, iteration=0)
        cb.on_validation_begin(method=None, iteration=1000)
        assert cb.run_calls == []

    def test_skipped_on_off_iter(self) -> None:
        cb = _make_recording(every_steps=50)
        cb.on_validation_begin(method=None, iteration=49)
        cb.on_validation_begin(method=None, iteration=51)
        assert cb.run_calls == []

    def test_runs_on_match(self) -> None:
        cb = _make_recording(every_steps=50)
        cb.on_validation_begin(method=None, iteration=50)
        cb.on_validation_begin(method=None, iteration=100)
        assert cb.run_calls == [50, 100]

    def test_on_validation_begin_runs_step_zero_by_default(self) -> None:
        """Verify the callback records a pre-training baseline by default."""
        cb = _make_recording(every_steps=50)
        cb.on_validation_begin(method=None, iteration=0)
        assert cb.run_calls == [0]

    def test_on_validation_begin_respects_start_gate_and_twenty_step_cadence(self) -> None:
        """Verify the H3 schedule from baseline through optimizer step 400."""
        cb = _make_recording(every_steps=20, run_at_start=True)

        for iteration in range(401):
            cb.on_validation_begin(method=None, iteration=iteration)

        assert cb.run_calls == [0, *range(20, 401, 20)]

    def test_on_validation_begin_skips_step_zero_when_disabled(self) -> None:
        """Verify disabling the baseline preserves later cadence events."""
        cb = _make_recording(every_steps=20, run_at_start=False)
        cb.on_validation_begin(method=None, iteration=0)
        cb.on_validation_begin(method=None, iteration=20)
        assert cb.run_calls == [20]


class TestH3ValidationContract:
    """Verify synchronized MiniMax H3 validation through Weights & Biases logging."""

    def test_prepare_validation_batch_forwards_video_count(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify that each ForwardBatch receives the configured output count."""
        cb = _make_callback(num_videos_per_prompt=3)
        cb.training_config = SimpleNamespace(
            data=SimpleNamespace(
                num_height=64,
                num_width=96,
                num_latent_t=2,
            ),
            pipeline_config=SimpleNamespace(
                vae_config=SimpleNamespace(
                    arch_config=SimpleNamespace(temporal_compression_ratio=4),
                ),
            ),
            model_path="unused",
            vsa_sparsity=0.0,
        )
        monkeypatch.setattr(
            "fastvideo.train.callbacks.validation.make_inference_args",
            lambda *args, **kwargs: SimpleNamespace(),
        )

        batch = cb._prepare_validation_batch(
            SamplingParam(),
            {"prompt": "Generate synchronized media."},
            num_inference_steps=5,
        )

        assert batch.num_videos_per_prompt == 3

    def test_prepare_validation_batch_ignores_media_for_text_only_generation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ) -> None:
        """Verify text-to-video-with-audio uses captions without source media."""
        video_path = tmp_path / "reference.mp4"
        video_path.touch()
        cb = _make_callback(use_validation_media_conditioning=False)
        cb.training_config = SimpleNamespace(
            data=SimpleNamespace(
                num_height=64,
                num_width=96,
                num_latent_t=2,
            ),
            pipeline_config=SimpleNamespace(
                vae_config=SimpleNamespace(
                    arch_config=SimpleNamespace(temporal_compression_ratio=4),
                ),
            ),
            model_path="unused",
            vsa_sparsity=0.0,
        )
        monkeypatch.setattr(
            "fastvideo.train.callbacks.validation.make_inference_args",
            lambda *args, **kwargs: SimpleNamespace(),
        )

        batch = cb._prepare_validation_batch(
            SamplingParam(),
            {
                "prompt": "Generate synchronized media.",
                "video_path": str(video_path),
            },
            num_inference_steps=5,
        )

        assert batch.image_path is None

    def test_run_validation_for_steps_aligns_h3_audio_with_captions(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify one waveform and sample rate stay aligned with each H3 video."""
        validation_records = [{"caption": f"prompt-{index}"} for index in range(8)]

        class FakeH3Pipeline:
            """Return deterministic video and stereo audio for each prompt."""

            fastvideo_args = SimpleNamespace(pipeline_config=SimpleNamespace())

            def forward(self, batch, inference_args):
                """Produce media whose values identify the source prompt."""
                prompt_index = int(batch.prompt.rsplit("-", 1)[1])
                return SimpleNamespace(
                    output=torch.full((1, 3, 2, 2, 2), prompt_index / 8),
                    extra={
                        "audio": torch.full((32, 2), float(prompt_index)),
                        "audio_sample_rate": 32_000,
                    },
                )

        cb = _make_callback()
        cb.training_config = SimpleNamespace(
            model_path="unused",
            pipeline_config=SimpleNamespace(),
        )
        cb.rank_in_sp_group = 0
        pipeline = FakeH3Pipeline()
        monkeypatch.setattr(
            "fastvideo.train.callbacks.validation.ValidationDataset",
            lambda filename: validation_records,
        )
        monkeypatch.setattr(
            "fastvideo.train.callbacks.validation.make_inference_args",
            lambda *args, **kwargs: SimpleNamespace(
                pipeline_config=SimpleNamespace(dmd_denoising_steps=None),
                dit_cpu_offload=True,
            ),
        )
        monkeypatch.setattr(cb, "_get_pipeline", lambda *, transformer: pipeline)
        monkeypatch.setattr(cb, "_get_sampling_param", SamplingParam)
        monkeypatch.setattr(
            cb,
            "_prepare_validation_batch",
            lambda sampling_param, validation_batch, num_inference_steps: SimpleNamespace(
                prompt=validation_batch["caption"],
            ),
        )

        result = cb._run_validation_for_steps(50, transformer=torch.nn.Identity())

        assert result.captions == [f"prompt-{index}" for index in range(8)]
        assert len(result.videos) == 8
        assert result.audio_sample_rates == [32_000] * 8
        assert len(result.audio_waveforms) == 8
        for prompt_index, waveform in enumerate(result.audio_waveforms):
            assert torch.is_tensor(waveform)
            torch.testing.assert_close(waveform, torch.full((32, 2), float(prompt_index)))

    def test_log_validation_video_artifacts_emits_eight_wandb_videos(self) -> None:
        """Verify a MiniMax H3 event contains eight media objects and scalar evidence."""
        class FakeWandbTracker:
            """Record tracker calls without contacting W&B."""

            def __init__(self) -> None:
                self.video_calls = []
                self.artifact_calls = []

            def video(self, filename, *, caption, fps):
                self.video_calls.append((filename, caption, fps))
                return filename

            def log_artifacts(self, artifacts, step):
                self.artifact_calls.append((artifacts, step))

        cb = _make_callback()
        cb.tracker = FakeWandbTracker()
        filenames = [f"validation-{index}.mp4" for index in range(8)]
        captions = [f"caption-{index}" for index in range(8)]
        scalar_metrics = {
            "validation/50_steps_video_count": 8.0,
            "validation/50_steps_duration_sec": 12.5,
            "validation/50_steps_audio_video_count": 8.0,
        }

        cb._log_validation_video_artifacts(
            filenames,
            captions,
            key="validation_videos_50_steps",
            step=20,
            fps=24,
            scalar_metrics=scalar_metrics,
        )

        assert len(cb.tracker.video_calls) == 8
        assert len(cb.tracker.artifact_calls) == 1
        artifacts, step = cb.tracker.artifact_calls[0]
        assert step == 20
        assert artifacts["validation_videos_50_steps"] == filenames
        assert {key: artifacts[key] for key in scalar_metrics} == scalar_metrics


class TestAttnQatInferValidation:

    def test_context_swaps_and_restores_qat_attention(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from fastvideo.attention.backends import attn_qat_infer
        from fastvideo.attention.backends.attn_qat_infer import (
            AttnQatInferImpl, )
        from fastvideo.attention.backends.attn_qat_train import (
            AttnQatTrainImpl, )
        from fastvideo.platforms import AttentionBackendEnum

        monkeypatch.setattr(attn_qat_infer, "is_attn_qat_infer_available", lambda: True)
        impl = AttnQatTrainImpl(
            num_heads=1,
            head_size=128,
            causal=False,
            softmax_scale=128**-0.5,
        )

        holder = torch.nn.Module()
        holder.attn_impl = impl
        holder.num_heads = 1
        holder.head_size = 128
        holder.num_kv_heads = 1
        holder.softmax_scale = 128**-0.5
        holder.backend = AttentionBackendEnum.ATTN_QAT_TRAIN
        transformer = torch.nn.Module()
        transformer.add_module("attention", holder)
        cb = _make_callback()
        cb.attn_qat_infer = True

        with pytest.raises(RuntimeError, match="stop"):
            with cb._attn_qat_infer_context(transformer):
                assert isinstance(holder.attn_impl, AttnQatInferImpl)
                assert holder.backend is AttentionBackendEnum.ATTN_QAT_INFER
                raise RuntimeError("stop")

        assert holder.attn_impl is impl
        assert holder.backend is AttentionBackendEnum.ATTN_QAT_TRAIN

    def test_context_requires_qat_training_attention(self) -> None:
        cb = _make_callback()
        cb.attn_qat_infer = True

        with pytest.raises(RuntimeError, match="no ATTN_QAT_TRAIN layers"):
            with cb._attn_qat_infer_context(torch.nn.Linear(1, 1)):
                pass


# ---------------------------------------------------------------------------
# C. _find_ema_callback
# ---------------------------------------------------------------------------


class TestFindEmaCallback:

    def test_returns_none_without_callback_dict(self) -> None:
        cb = _make_callback()
        # _callback_dict is not set on bare instances.
        assert cb._find_ema_callback() is None

    def test_returns_none_when_no_ema_registered(self) -> None:
        cb = _make_callback()
        cb_dict = CallbackDict({}, training_config=object())
        cb._callback_dict = cb_dict
        assert cb._find_ema_callback() is None

    def test_finds_ema_callback(self) -> None:
        cb = _make_callback()
        cb_dict = CallbackDict({}, training_config=object())
        ema = EMACallback(decay=0.99)
        cb_dict._callbacks["ema"] = ema
        cb_dict._callbacks["validation"] = cb
        cb._callback_dict = cb_dict

        found = cb._find_ema_callback()
        assert found is ema


# ---------------------------------------------------------------------------
# D. state_dict / load_state_dict (rng round-trip)
# ---------------------------------------------------------------------------


class TestStateDict:

    def test_state_dict_empty_without_generator(self) -> None:
        cb = _make_callback()
        # validation_random_generator is None until on_train_start.
        assert cb.state_dict() == {}

    def test_round_trip_preserves_rng_state(self) -> None:
        cb = _make_callback()
        gen = torch.Generator(device="cpu").manual_seed(123)
        # Advance RNG so a default-init generator on the receiving
        # side is observably different.
        for _ in range(5):
            torch.randn(4, generator=gen)
        cb.validation_random_generator = gen

        state = cb.state_dict()
        assert "validation_rng" in state

        # Receiver: fresh generator with a different seed.
        fresh = _make_callback()
        fresh.validation_random_generator = (
            torch.Generator(device="cpu").manual_seed(999)
        )
        fresh.load_state_dict(state)

        # After load, both generators draw the same next sample.
        a = torch.randn(8, generator=cb.validation_random_generator)
        b = torch.randn(8, generator=fresh.validation_random_generator)
        assert torch.equal(a, b)

    def test_load_without_generator_is_noop(self) -> None:
        cb = _make_callback()
        # Generator is None: load must not raise even when state has
        # an rng entry.
        cb.load_state_dict(
            {"validation_rng": torch.tensor([1, 2, 3], dtype=torch.uint8)}
        )
        assert cb.validation_random_generator is None


# ---------------------------------------------------------------------------
# E. metric result aggregation
# ---------------------------------------------------------------------------


class TestMetricAggregation:

    def test_accumulates_scores_and_scalar_details(self) -> None:
        stats = _ValidationMetricStats()
        row: dict = {"path": "sample.mp4"}
        ValidationCallback._accumulate_metric_results(
            stats,
            row,
            {
                "vbench.aesthetic_quality": SimpleNamespace(
                    name="vbench.aesthetic_quality",
                    score=0.5,
                    details={"ignored": [1, 2, 3]},
                ),
                SYNTHETIC_OPTICAL_FLOW_METRIC: SimpleNamespace(
                    name=SYNTHETIC_OPTICAL_FLOW_METRIC,
                    score=1.5,
                    details={
                        **{key: float(i) for i, key in enumerate(SYNTHETIC_OPTICAL_FLOW_LOG_KEYS)},
                        "pixel_epe_mean_std": 99.0,
                        "pixel_epe_mean_max": 100.0,
                        "pixel_epe_mean_auc": 101.0,
                    },
                ),
            },
        )

        assert stats.sums["vbench.aesthetic_quality"] == 0.5
        assert stats.counts["vbench.aesthetic_quality"] == 1.0
        for i, key in enumerate(SYNTHETIC_OPTICAL_FLOW_LOG_KEYS):
            metric_key = f"{SYNTHETIC_OPTICAL_FLOW_METRIC}.{key}"
            assert stats.sums[metric_key] == float(i)
            assert stats.counts[metric_key] == 1.0
        assert SYNTHETIC_OPTICAL_FLOW_METRIC not in stats.sums
        assert f"{SYNTHETIC_OPTICAL_FLOW_METRIC}.pixel_epe_mean_std" not in stats.sums
        assert f"{SYNTHETIC_OPTICAL_FLOW_METRIC}.pixel_epe_mean_max" not in stats.sums
        assert f"{SYNTHETIC_OPTICAL_FLOW_METRIC}.pixel_epe_mean_auc" not in stats.sums
        assert "vbench.aesthetic_quality.ignored" not in stats.sums
        assert row["vbench.aesthetic_quality"] == 0.5

    def test_merges_metric_stats(self) -> None:
        dst = _ValidationMetricStats(
            sums={"a": 1.0},
            counts={"a": 1.0},
            per_video=[{"path": "a.mp4"}],
            errors=["first"],
        )
        src = _ValidationMetricStats(
            sums={"a": 2.0, "b": 3.0},
            counts={"a": 2.0, "b": 1.0},
            per_video=[{"path": "b.mp4"}],
            errors=["second"],
        )

        ValidationCallback._merge_metric_stats(dst, src)

        assert dst.sums == {"a": 3.0, "b": 3.0}
        assert dst.counts == {"a": 3.0, "b": 1.0}
        assert dst.per_video == [{"path": "a.mp4"}, {"path": "b.mp4"}]
        assert dst.errors == ["first", "second"]

    def test_metric_log_name_replaces_dots(self) -> None:
        cb = ValidationCallback(
            pipeline_target=_PIPE_TARGET,
            dataset_file="x.json",
            metrics={"names": ["vbench.aesthetic_quality"]},
        )

        assert (
            cb._metric_log_name("vbench.aesthetic_quality")
            == "metrics/validation/vbench/aesthetic_quality"
        )

    def test_metric_device_uses_local_rank(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cb = ValidationCallback(
            pipeline_target=_PIPE_TARGET,
            dataset_file="x.json",
            metrics={"names": ["vbench.aesthetic_quality"]},
        )
        cb.world_group = SimpleNamespace(local_rank=5)
        monkeypatch.setattr(
            torch.cuda,
            "is_available",
            lambda: True,
        )

        assert cb._metric_device() == "cuda:5"

    def test_metric_extras_include_actions_and_calibration(self) -> None:
        cb = ValidationCallback(
            pipeline_target=_PIPE_TARGET,
            dataset_file="x.json",
            metrics={
                "names": ["optical_flow.synthetic_optical_flow"],
                "calibration_path": "/tmp/calibration.json",
                "mouse_pitch_sign": -1,
            },
        )
        actions = {
            "keyboard": np.zeros((3, 6)),
            "mouse": np.zeros((3, 2)),
        }

        extras = cb._validation_metric_extras(
            video_filenames=["a.mp4", "b.mp4"],
            actions=[actions, None],
            mouse_pitch_signs=[None, 1],
        )

        assert extras[0]["actions"] is actions
        assert extras[0]["calibration"] == "/tmp/calibration.json"
        assert extras[0]["mouse_pitch_sign"] == -1
        assert "actions" not in extras[1]
        assert extras[1]["mouse_pitch_sign"] == 1

    def test_validation_actions_prefers_loaded_conditions(self) -> None:
        keyboard = np.zeros((4, 6))
        mouse = np.zeros((4, 2))

        actions = ValidationCallback._validation_actions(
            {
                "keyboard_cond": keyboard,
                "mouse_cond": mouse,
            }
        )

        assert actions is not None
        assert np.array_equal(actions["keyboard"], keyboard)
        assert np.array_equal(actions["mouse"], mouse)

    def test_validation_actions_loads_action_path(
        self,
        tmp_path,
    ) -> None:
        path = tmp_path / "actions.npy"
        np.save(
            path,
            {
                "keyboard": np.ones((4, 6)),
                "mouse": np.ones((4, 2)),
            },
            allow_pickle=True,
        )

        actions = ValidationCallback._validation_actions({"action_path": str(path)})

        assert actions is not None
        assert actions["keyboard"].shape == (4, 6)
        assert actions["mouse"].shape == (4, 2)

    def test_available_paths_requires_all_paths(self) -> None:
        assert ValidationCallback._available_paths(["a.mp4", "b.mp4"]) == ["a.mp4", "b.mp4"]
        assert ValidationCallback._available_paths(["a.mp4", None]) is None


# ---------------------------------------------------------------------------
# F. action overlay plumbing
# ---------------------------------------------------------------------------


class TestActionOverlay:

    def test_save_validation_videos_supports_overlay_suffix(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify overlay artifacts use distinct names and the media writer."""
        cb = _make_callback()
        cb.global_rank = 3
        calls: list[tuple[str, int]] = []

        def fake_write_validation_mp4(
            fname: str,
            video: list[np.ndarray],
            *,
            fps: int,
            audio=None,
            audio_sample_rate=None,
        ) -> None:
            """Record media arguments without invoking a system codec."""
            del video, audio, audio_sample_rate
            calls.append((fname, fps))

        monkeypatch.setattr(
            "fastvideo.train.callbacks.validation.write_validation_mp4",
            fake_write_validation_mp4,
        )

        saved = cb._save_validation_videos(
            [[np.zeros((2, 2, 3), dtype=np.uint8)]],
            output_dir=str(tmp_path),
            step=7,
            num_inference_steps=4,
            fps=25,
            suffix="_overlay",
        )

        assert saved.filenames == [
            str(tmp_path / "validation_step_7_inference_steps_4_rank_3_video_0_overlay.mp4")
        ]
        assert saved.indices == [0]
        assert calls == [(saved.filenames[0], 25)]

    def test_save_validation_videos_skips_failed_artifact(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify a failed video-only encode does not hide later artifacts."""
        cb = _make_callback()
        cb.global_rank = 0
        calls: list[str] = []

        def fake_write_validation_mp4(
            fname: str,
            video: list[np.ndarray],
            *,
            fps: int,
            audio=None,
            audio_sample_rate=None,
        ) -> None:
            """Fail the first encode to exercise video-only best-effort logging."""
            del video, fps, audio, audio_sample_rate
            calls.append(fname)
            if len(calls) == 1:
                raise OSError("ffmpeg failed")

        monkeypatch.setattr(
            "fastvideo.train.callbacks.validation.write_validation_mp4",
            fake_write_validation_mp4,
        )

        saved = cb._save_validation_videos(
            [
                [np.zeros((2, 2, 3), dtype=np.uint8)],
                [np.ones((2, 2, 3), dtype=np.uint8)],
            ],
            output_dir=str(tmp_path),
            step=8,
            num_inference_steps=5,
            fps=24,
        )

        assert saved.indices == [1]
        assert saved.filenames == [
            str(tmp_path / "validation_step_8_inference_steps_5_rank_0_video_1.mp4")
        ]
        assert len(calls) == 2

    def test_save_validation_videos_counts_preserved_audio(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify the callback counts only MP4 files with requested waveforms."""
        cb = _make_callback()
        cb.global_rank = 0
        encoded_audio = []

        def fake_write_validation_mp4(
            fname: str,
            video: list[np.ndarray],
            *,
            fps: int,
            audio=None,
            audio_sample_rate=None,
        ) -> None:
            del fname, video, fps
            encoded_audio.append((audio, audio_sample_rate))

        monkeypatch.setattr(
            "fastvideo.train.callbacks.validation.write_validation_mp4",
            fake_write_validation_mp4,
        )
        waveforms = [torch.zeros(32, 2), torch.ones(32, 2)]

        saved = cb._save_validation_videos(
            [[np.zeros((2, 2, 3), dtype=np.uint8)]] * 2,
            output_dir=str(tmp_path),
            step=0,
            num_inference_steps=50,
            fps=24,
            audio_waveforms=waveforms,
            audio_sample_rates=[32_000, 32_000],
        )

        assert saved.audio_video_count == 2
        assert saved.indices == [0, 1]
        assert len(encoded_audio) == 2
        assert encoded_audio[0][0] is waveforms[0]
        assert encoded_audio[1][0] is waveforms[1]
        assert [sample_rate for _, sample_rate in encoded_audio] == [32_000, 32_000]

    def test_save_validation_videos_skips_audio_encode_failure(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify an audio encoding failure is omitted without stopping later writes."""
        cb = _make_callback()
        cb.global_rank = 0
        calls: list[str] = []

        def fail_first_audio_encode(fname: str, *args, **kwargs) -> None:
            """Leave one incomplete file before allowing the second write."""
            del args, kwargs
            calls.append(fname)
            if len(calls) == 1:
                with open(fname, "wb") as incomplete_file:
                    incomplete_file.write(b"incomplete")
                raise RuntimeError("AAC unavailable")

        monkeypatch.setattr(
            "fastvideo.train.callbacks.validation.write_validation_mp4",
            fail_first_audio_encode,
        )

        saved = cb._save_validation_videos(
            [[np.zeros((2, 2, 3), dtype=np.uint8)]] * 2,
            output_dir=str(tmp_path),
            step=0,
            num_inference_steps=50,
            fps=24,
            audio_waveforms=[torch.zeros(32, 2)] * 2,
            audio_sample_rates=[32_000, 32_000],
        )

        assert saved.indices == [1]
        assert saved.audio_video_count == 1
        assert len(calls) == 2
        assert not (tmp_path / "validation_step_0_inference_steps_50_rank_0_video_0.mp4").exists()

    def test_post_process_validation_frames_uses_student_hook(self) -> None:
        cb = _make_callback(overlay_actions=True)
        frames = [np.zeros((2, 2, 3), dtype=np.uint8)]
        action = {
            "keyboard": np.ones((1, 4), dtype=np.float32),
            "mouse": np.zeros((1, 2), dtype=np.float32),
        }

        class Student:

            def post_process_validation_frames(
                self,
                received_frames: list[np.ndarray],
                *,
                action: dict,
            ) -> list[np.ndarray]:
                assert received_frames is frames
                assert action["keyboard"].shape == (1, 4)
                return [np.full((2, 2, 3), 255, dtype=np.uint8)]

        cb.method = SimpleNamespace(student=Student())

        overlay = cb._post_process_validation_frames(
            frames,
            action=action,
        )

        assert overlay is not None
        assert int(overlay[0][0, 0, 0]) == 255

    def test_post_process_validation_frames_without_hook_is_noop(self) -> None:
        cb = _make_callback(overlay_actions=True)
        cb.method = SimpleNamespace(student=object())

        overlay = cb._post_process_validation_frames(
            [np.zeros((2, 2, 3), dtype=np.uint8)],
            action=None,
        )

        assert overlay is None


class TestKeepLoadedEncoderWidths:
    """``_keep_loaded_encoder_widths`` merges the two config sources.

    Validation runs the encoder weights the loader brought in, but reaches
    the stages through configs derived from the training side, which never
    see the checkpoint's ``config.json``. Each side therefore holds the
    real value for a different set of fields: the loader knows the widths,
    training knows the FastVideo-only fields HF configs never carry.
    """

    @staticmethod
    def _encoder(hidden_size: int, text_len: int = 0) -> SimpleNamespace:
        return SimpleNamespace(arch_config=SimpleNamespace(
            hidden_size=hidden_size,
            text_len=text_len,
        ))

    def test_copies_loaded_width(self) -> None:
        # ByT5 at the generic T5 default vs the checkpoint's real width.
        validation = SimpleNamespace(text_encoder_configs=(self._encoder(512), ))
        loaded = SimpleNamespace(text_encoder_configs=(self._encoder(1472), ))

        ValidationCallback._keep_loaded_encoder_widths(validation, loaded)

        assert validation.text_encoder_configs[0].arch_config.hidden_size == 1472

    def test_preserves_training_owned_fields(self) -> None:
        # ``text_len`` is a FastVideo-only field absent from HF configs, so
        # the loader never populates it and the training value must survive.
        validation = SimpleNamespace(text_encoder_configs=(self._encoder(512, text_len=1000), ))
        loaded = SimpleNamespace(text_encoder_configs=(self._encoder(1472, text_len=0), ))

        ValidationCallback._keep_loaded_encoder_widths(validation, loaded)

        assert validation.text_encoder_configs[0].arch_config.text_len == 1000

    def test_keeps_config_objects_unshared(self) -> None:
        # The second call site writes into ``tc.pipeline_config`` itself, so
        # the merge must not alias the loaded encoder objects into it.
        validation = SimpleNamespace(text_encoder_configs=(self._encoder(512), ))
        original = validation.text_encoder_configs
        loaded = SimpleNamespace(text_encoder_configs=(self._encoder(1472), ))

        ValidationCallback._keep_loaded_encoder_widths(validation, loaded)

        assert validation.text_encoder_configs is original
        assert validation.text_encoder_configs[0] is not loaded.text_encoder_configs[0]

    def test_skips_unpopulated_loaded_width(self) -> None:
        # ``TextEncoderArchConfig.hidden_size`` defaults to 0; a loader that
        # never filled it must not clobber a real training-side width.
        validation = SimpleNamespace(text_encoder_configs=(self._encoder(3584), ))
        loaded = SimpleNamespace(text_encoder_configs=(self._encoder(0), ))

        ValidationCallback._keep_loaded_encoder_widths(validation, loaded)

        assert validation.text_encoder_configs[0].arch_config.hidden_size == 3584

    def test_multiple_encoders_are_matched_by_index(self) -> None:
        validation = SimpleNamespace(text_encoder_configs=(
            self._encoder(512),
            self._encoder(512),
        ))
        loaded = SimpleNamespace(text_encoder_configs=(
            self._encoder(3584),
            self._encoder(1472),
        ))

        ValidationCallback._keep_loaded_encoder_widths(validation, loaded)

        widths = [e.arch_config.hidden_size for e in validation.text_encoder_configs]
        assert widths == [3584, 1472]

    @pytest.mark.parametrize(
        "validation, loaded",
        [
            (SimpleNamespace(), SimpleNamespace(text_encoder_configs=())),
            (SimpleNamespace(text_encoder_configs=()), SimpleNamespace()),
            (SimpleNamespace(text_encoder_configs=()), SimpleNamespace(text_encoder_configs=())),
        ],
    )
    def test_missing_or_empty_encoders_is_noop(
        self,
        validation: SimpleNamespace,
        loaded: SimpleNamespace,
    ) -> None:
        # Pipelines without text encoders must not raise here.
        ValidationCallback._keep_loaded_encoder_widths(validation, loaded)
