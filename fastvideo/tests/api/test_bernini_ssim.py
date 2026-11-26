# SPDX-License-Identifier: Apache-2.0
"""Weight-free contracts for the CI-discovered Bernini video-editing gate."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import fastvideo.envs as envs
from fastvideo.tests.loader.test_bernini_checkpoint import make_source
from fastvideo.tests.ssim import test_bernini_similarity as ssim
from fastvideo.tests.ssim.ci_runner import discover_tasks


def test_bernini_ssim_is_discovered_once_with_one_gpu():
    tasks = discover_tasks(Path(ssim.__file__).parent, [Path(ssim.__file__).name])
    assert len(tasks) == 1
    assert tasks[0].required_gpus == 1
    assert tasks[0].model_id == "Bernini-R-1.3B-Diffusers"
    assert ssim.MIN_SSIM == 0.98
    # SHA-256 of the unchanged official v2v_case1.json prompt.
    assert hashlib.sha256(ssim.TEST_PROMPTS[0].encode()).hexdigest() == (
        "bb6a4e91c0a89a33465d939cae70d0ae3de653d05fba08067221c040bc94dc0c")
    assert ssim.OFFICIAL_CODE_REVISION in ssim.SOURCE_VIDEO_URL


@pytest.mark.parametrize("valid", [False, True])
def test_bernini_ssim_source_download_is_verified_before_publishing(tmp_path, monkeypatch, valid):
    data = b"pinned input video"
    calls = []

    class Response:
        content = data

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            calls.append("checked HTTP status")

    def get(url, *, timeout):
        assert url == ssim.SOURCE_VIDEO_URL and timeout == 120
        return Response()

    monkeypatch.setattr(ssim.requests, "get", get)
    if valid:
        monkeypatch.setattr(ssim, "SOURCE_VIDEO_SHA256", hashlib.sha256(data).hexdigest())
    destination = tmp_path / "input.mp4"
    if valid:
        assert ssim._download_source_video(destination).read_bytes() == data
    else:
        with pytest.raises(ValueError, match="SHA-256"):
            ssim._download_source_video(destination)
        assert not destination.exists()
    assert calls == ["checked HTTP status"]


@pytest.mark.parametrize("layout", ["local", "snapshot"])
@pytest.mark.parametrize("valid", [False, True])
def test_bernini_ssim_checkpoint_revision_is_immutable(tmp_path, layout, valid):
    revision = ssim.SOURCE_REVISION if valid else "0" * 40
    if layout == "local":
        source = tmp_path / "local"
        metadata = source / ".cache/huggingface/download/config.json.metadata"
        metadata.parent.mkdir(parents=True)
        metadata.write_text(f"{revision}\netag\n0\n")
    else:
        source = tmp_path / "snapshots" / revision
        source.mkdir(parents=True)
    if valid:
        ssim._validate_source_revision(source)
    else:
        with pytest.raises(ValueError, match="checkpoint revision"):
            ssim._validate_source_revision(source)


def test_bernini_ssim_rejects_checkpoint_without_provenance(tmp_path):
    with pytest.raises(ValueError, match="pinned Hub snapshot"):
        ssim._validate_source_revision(tmp_path)


def test_bernini_ssim_reports_missing_local_checkpoint(tmp_path):
    with envs.FASTVIDEO_TEST_BERNINI_MODEL_DIR.override(tmp_path / "missing"):
        with pytest.raises(pytest.skip.Exception, match="checkpoint directory does not exist"):
            ssim.bernini_checkpoint.__wrapped__(None, "L40S_reference_videos")


@pytest.mark.parametrize("device,expected", [
    ("NVIDIA GB200", "GB200_reference_videos"),
    ("NVIDIA L40S", "L40S_reference_videos"),
    ("NVIDIA Example GPU 32GB", "ExampleGPU32GB_reference_videos"),
])
def test_bernini_ssim_reference_namespace_tracks_device(monkeypatch, device, expected):
    monkeypatch.setattr(ssim.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(ssim, "get_cuda_device_name", lambda: device)
    assert ssim.bernini_device_reference_folder.__wrapped__() == expected


@pytest.mark.parametrize("local", [False, True])
def test_bernini_ssim_prepares_pinned_weights_without_tensor_conversion(tmp_path, monkeypatch, local):
    source = make_source(tmp_path)
    metadata = source / ".cache/huggingface/download/config.json.metadata"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(f"{ssim.SOURCE_REVISION}\netag\n0\n")
    observed = []

    def download(**kwargs):
        observed.append(kwargs)
        return str(source)

    monkeypatch.setattr(ssim, "snapshot_download", download)
    factory = SimpleNamespace(mktemp=lambda _: tmp_path / "model-output")
    with envs.FASTVIDEO_TEST_BERNINI_MODEL_DIR.override(str(source) if local else None):
        prepared = ssim.bernini_checkpoint.__wrapped__(factory, "GB200_reference_videos")
    assert (prepared / "transformer/model.safetensors").read_bytes() == b"unchanged tensors"
    if local:
        assert not observed
    else:
        assert len(observed) == 1
        assert observed[0]["repo_id"] == ssim.SOURCE_REPO
        assert observed[0]["revision"] == ssim.SOURCE_REVISION
        assert "vae/*" in observed[0]["allow_patterns"]


@pytest.mark.parametrize("failure", [None, "generation", "geometry"])
def test_bernini_ssim_uses_this_runs_video_and_always_shuts_down(tmp_path, monkeypatch, failure):
    observed = {}
    returned_dir = tmp_path / "returned-output"
    returned_dir.mkdir()
    generated = returned_dir / "fresh_candidate_2.mp4"
    generated.write_bytes(b"generated output")
    monkeypatch.setattr(ssim, "build_generated_output_dir", lambda *args: str(tmp_path))
    monkeypatch.setattr(ssim, "_assert_similarity", lambda **kwargs: observed.update(comparison=kwargs))

    class Generator:
        @classmethod
        def from_config(cls, config):
            observed["config"] = config
            return cls()

        def generate(self, *, request):
            observed["request"] = request
            if failure == "generation":
                raise RuntimeError("simulated generation failure")
            size = (480, 848, 80 if failure == "geometry" else 81)
            return SimpleNamespace(video_path=str(generated), size=size)

        def shutdown(self):
            observed["shutdown"] = True

    monkeypatch.setattr(ssim, "VideoGenerator", Generator)
    args = (ssim.TEST_PROMPTS[0], "TORCH_SDPA", "Bernini-R-1.3B-Diffusers", tmp_path / "checkpoint",
            tmp_path / "source.mp4", "GB200_reference_videos")
    if failure == "generation":
        with pytest.raises(RuntimeError, match="simulated generation failure"):
            ssim.test_bernini_video_editing_similarity(*args)
        assert "comparison" not in observed
    elif failure == "geometry":
        with pytest.raises(AssertionError, match="output geometry"):
            ssim.test_bernini_video_editing_similarity(*args)
        assert "comparison" not in observed
    else:
        ssim.test_bernini_video_editing_similarity(*args)
        assert observed["comparison"]["output_media_name"] == generated.name
        assert observed["comparison"]["output_dir"] == str(generated.parent)
        assert observed["comparison"]["min_acceptable_ssim"] == 0.98
    assert observed["shutdown"]
    assert observed["request"].inputs.video_path == str(tmp_path / "source.mp4")
    assert observed["request"].sampling.num_inference_steps == 40
    assert observed["config"].engine.num_gpus == 1
    assert not observed["config"].engine.use_fsdp_inference
