# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from fastvideo.models.dits.trackwan.track_encoder import TrackEncoder
from fastvideo.models.dits.trackwan.model import _TrackConditioningMixin
from fastvideo.pipelines import TrainingBatch
from fastvideo.train.models.wantrack.control import (
    HandleState,
    StableGridController,
    cosine_radius_weights,
    deform_grid,
)
from fastvideo.train.models.wantrack.runtime import (
    CausalWanTrackSession,
    PreparedWanTrackInput,
    WanTrackInferenceRuntime,
)


def test_track_encoder_window_matches_full_sequence() -> None:
    torch.manual_seed(7)
    encoder = TrackEncoder(
        id_dim=8,
        track_channels=4,
        vae_temporal_compression=4,
    )
    coords = torch.rand(1, 17, 6, 2)
    visibility = (torch.rand(1, 17, 6) > 0.2).float()
    track_ids = torch.arange(6).unsqueeze(0)
    full = encoder(
        coords,
        visibility,
        latent_t=5,
        latent_h=8,
        latent_w=10,
        track_ids=track_ids,
    )

    leading = encoder.forward_window(
        coords[:, :1],
        visibility[:, :1],
        latent_start=0,
        latent_t=1,
        latent_h=8,
        latent_w=10,
        pixel_start=0,
        track_ids=track_ids,
    )
    later = encoder.forward_window(
        coords[:, 1:17],
        visibility[:, 1:17],
        latent_start=1,
        latent_t=4,
        latent_h=8,
        latent_w=10,
        pixel_start=1,
        track_ids=track_ids,
    )
    torch.testing.assert_close(leading, full[:, :, :1])
    torch.testing.assert_close(later, full[:, :, 1:5])


def test_preencoded_track_map_is_latent_aligned_and_exclusive() -> None:
    condition = _TrackConditioningMixin()
    condition.in_channels = 5
    condition.track_channels = 1
    hidden = torch.zeros(1, 4, 2, 3, 3)
    track_map = torch.ones(1, 1, 2, 3, 3)
    combined = condition._append_track_conditioning(
        hidden,
        track_points=None,
        track_visibility=None,
        track_ids=None,
        track_map=track_map,
        start_frame=0,
    )
    assert combined.shape == (1, 5, 2, 3, 3)
    torch.testing.assert_close(combined[:, -1:], track_map)
    with pytest.raises(ValueError, match="mutually exclusive"):
        condition._append_track_conditioning(
            hidden,
            track_points=torch.zeros(1, 5, 1, 2),
            track_visibility=torch.ones(1, 5, 1),
            track_ids=torch.zeros(1, 1, dtype=torch.long),
            track_map=track_map,
            start_frame=0,
        )


def test_radius_falloff_and_overlap_blending() -> None:
    points = np.array([[0.5, 0.5], [0.6, 0.5], [0.9, 0.5]],
                      dtype=np.float32)
    weights = cosine_radius_weights(points, (0.5, 0.5), 0.2)
    assert weights[0] == pytest.approx(1.0)
    assert 0.0 < weights[1] < 1.0
    assert weights[2] == 0.0

    handles = (
        HandleState("left", (0.5, 0.5), (0.7, 0.5)),
        HandleState("right", (0.5, 0.5), (0.3, 0.5)),
    )
    deformed = deform_grid(points[:1], handles, 0.2)
    np.testing.assert_allclose(deformed, points[:1], atol=1e-6)


def test_control_revisions_are_monotonic_resampled_and_boundary_only() -> None:
    controller = StableGridController(
        [{"id": "main", "x": 0.5, "y": 0.5}],
        radius=0.2,
    )
    initial = controller.render_constant(3)
    assert controller.queue_revision(
        2,
        samples=[{
            "id": "main",
            "x": 0.7,
            "y": 0.5,
            "timestamp_ms": 50,
        }],
        add=[{
            "id": "new",
            "x": 0.2,
            "y": 0.2,
        }],
    )
    assert not controller.queue_revision(1, samples=[])
    # Queuing does not mutate the already-rendered/committed prefix.
    np.testing.assert_array_equal(initial.tracks,
                                  controller.render_constant(3).tracks)

    applied = controller.apply_pending(
        3,
        interval_start_ms=0,
        interval_end_ms=100,
    )
    assert applied.revision == 2
    assert applied.active_handle_ids == ("main", "new")
    center_index = int(np.argmin(
        np.linalg.norm(controller.grid - np.array([0.5, 0.5]), axis=1)))
    assert applied.tracks[0, center_index, 0] == pytest.approx(
        controller.grid[center_index, 0], abs=2e-2)
    assert applied.tracks[-1, center_index, 0] > applied.tracks[
        0, center_index, 0]

    assert controller.queue_revision(3, remove=["main"], radius=0.1)
    removed = controller.apply_pending(
        2,
        interval_start_ms=100,
        interval_end_ms=200,
    )
    assert removed.active_handle_ids == ("new", )
    assert removed.radius == pytest.approx(0.1)


@dataclass
class _FakeModel:
    device: torch.device = torch.device("cpu")


class _FakeRuntime:
    fps = 16.0
    chunk_size = 3
    temporal_compression = 4
    dmd_denoising_steps = [1000, 750, 500, 250]
    warp_denoising_step = True

    def __init__(self) -> None:
        self.model = _FakeModel()
        self.clear_count = 0
        self.encoded_windows: list[np.ndarray] = []

    def prepare(self, image, prompt):
        del image
        batch = TrainingBatch(
            latents=torch.zeros(1, 1, 1, 2, 2),
            conditional_dict={
                "track_ids": None,
                "track_map": None,
                "track_points": None,
                "track_visibility": None,
            },
        )
        return PreparedWanTrackInput(
            image=Image.new("RGB", (16, 16)),
            prompt=prompt,
            batch=batch,
            latent_channels=1,
            latent_height=2,
            latent_width=2,
        )

    def clear_state(self):
        self.clear_count += 1

    def new_vae_cache(self):
        return []

    def encode_track_window(self, **kwargs):
        self.encoded_windows.append(kwargs["points"].copy())
        return torch.zeros(1, 1, kwargs["latent_t"], 2, 2)

    def decode_block(self, latents, *, cache, first):
        del first
        frames = np.repeat(
            latents[:, :, :1, :1, :1].float().cpu().numpy().reshape(
                -1, 1, 1, 1),
            3,
            axis=-1,
        )
        return frames.astype(np.float32), cache


def test_session_noise_is_deterministic_and_edits_are_future_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fastvideo.train.models.wantrack.runtime as runtime_module

    monkeypatch.setattr(
        runtime_module,
        "sample_wantrack_block",
        lambda model, batch, latents, **kwargs: latents,
    )

    def run_two_blocks(runtime: _FakeRuntime):
        session = CausalWanTrackSession(runtime)
        session.start(
            runtime.prepare(b"", ""),
            "",
            [{"id": "h", "x": 0.5, "y": 0.5}],
            {"seed": 9, "steps": 1},
            radius=0.2,
        )
        first = session.generate_next_block()
        first_history = session.committed_control_history
        assert session.apply_control_revision(
            1,
            samples=[{
                "id": "h",
                "x": 0.7,
                "y": 0.5,
                "timestamp_ms": 1,
            }],
            received_at_ms=1,
        )
        second = session.generate_next_block()
        np.testing.assert_array_equal(
            first_history[0],
            session.committed_control_history[0],
        )
        session.close()
        return first.pixel_frames, second.pixel_frames

    first_runtime = _FakeRuntime()
    second_runtime = _FakeRuntime()
    first_outputs = run_two_blocks(first_runtime)
    second_outputs = run_two_blocks(second_runtime)
    np.testing.assert_array_equal(first_outputs[0], second_outputs[0])
    np.testing.assert_array_equal(first_outputs[1], second_outputs[1])
    assert first_runtime.encoded_windows[1].shape[0] == 12
    assert first_runtime.clear_count == 2


def test_session_sampling_error_clears_model_and_vae_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fastvideo.train.models.wantrack.runtime as runtime_module

    def fail_sampling(*args, **kwargs):
        raise RuntimeError("sampling failed")

    monkeypatch.setattr(runtime_module, "sample_wantrack_block",
                        fail_sampling)
    runtime = _FakeRuntime()
    session = CausalWanTrackSession(runtime)
    session.start(
        runtime.prepare(b"", ""),
        "",
        [{"id": "h", "x": 0.5, "y": 0.5}],
        {"seed": 9, "steps": 1},
    )
    with pytest.raises(RuntimeError, match="sampling failed"):
        session.generate_next_block()
    assert session.state == "failed"
    assert runtime.clear_count == 2


def test_image_preprocessing_empty_prompt_fps_and_invalid_export(
    tmp_path,
) -> None:
    image = Image.new("RGB", (100, 50), "red")
    processed = WanTrackInferenceRuntime.preprocess_image(
        image,
        width=32,
        height=32,
    )
    assert processed.size == (32, 32)
    assert WanTrackInferenceRuntime._fps_from_yaml({}) == 16.0
    assert WanTrackInferenceRuntime._fps_from_yaml({
        "callbacks": {
            "track_validation": {
                "fps": 24
            }
        }
    }) == 24.0

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model_index.json").write_text(
        '{"transformer": ["diffusers", "TrackWanTransformer3DModel"]}',
        encoding="utf-8",
    )
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("models: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="causal"):
        WanTrackInferenceRuntime.from_export(model_dir, yaml_path)
