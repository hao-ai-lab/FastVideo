"""CPU contract tests; fake executors do not validate generated-media quality."""

from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from dreamverse.config import MODEL_REGISTRY
from dreamverse.generation_inputs import GenerationAsset, GenerationInputs
from dreamverse.generation_worker import VideoGenerationWorker
from dreamverse.minimax_h3_generation import MiniMaxH3GenerationBackend
from dreamverse.worker_ipc import UserStepPayload


@pytest.fixture
def fastvideo_api(monkeypatch):
    """Use the actual lightweight API schema with only GPU execution replaced."""
    schema_path = Path(__file__).resolve().parents[4] / "fastvideo/api/schema.py"
    spec = importlib.util.spec_from_file_location("dreamverse_test_api_schema", schema_path)
    assert spec is not None and spec.loader is not None
    schema = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, schema)
    spec.loader.exec_module(schema)
    package = ModuleType("fastvideo")
    package.__path__ = []
    package.VideoGenerator = SimpleNamespace(from_config=Mock())
    monkeypatch.setitem(sys.modules, "fastvideo", package)
    monkeypatch.setitem(sys.modules, "fastvideo.api", schema)
    schema.MiniMaxH3Reference = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setattr("dreamverse.minimax_h3_generation.torch.cuda.synchronize", lambda: None)
    monkeypatch.setattr("dreamverse.minimax_h3_generation.torch.cuda.empty_cache", lambda: None)
    return package.VideoGenerator.from_config


class RecordingGenerator:
    def __init__(self):
        self.requests = []
        self.images = []
        self.closed = False

    def shutdown(self):
        self.closed = True

    def generate(self, request):
        self.requests.append(request)
        self.images.append(tuple(None if image is None else np.asarray(image).copy()
                                 for image in (request.inputs.pil_image, request.inputs.last_image)))
        return SimpleNamespace(
            frames=[np.full((2, 3, 3), 7, dtype=np.uint8), np.full((2, 3, 3), 29, dtype=np.uint8)],
            audio=np.zeros((2, 16), dtype=np.float32),
            audio_sample_rate=44100,
            generation_time=0.1,
        )


def prepared_backend(monkeypatch):
    backend = MiniMaxH3GenerationBackend(0)
    backend.model_config = dict(MODEL_REGISTRY["full-h3"])
    backend.generator = RecordingGenerator()
    monkeypatch.setattr(backend, "_gpu_mem", lambda: "fake executor")
    return backend


def test_ipc_preserves_immutable_ordered_references():
    inputs = GenerationInputs("ref2va", (
        GenerationAsset("second", "video", "/assets/second.mp4", "reference"),
        GenerationAsset("first", "image", "/assets/first.png", "reference"),
    ))
    payload = UserStepPayload("follow the references", 1, None, True, inputs)
    restored = pickle.loads(pickle.dumps(payload))
    assert restored == payload
    assert [asset.asset_id for asset in restored.generation_inputs.references] == ["second", "first"]


def test_worker_passes_conditioning_to_selected_backend():
    inputs = GenerationInputs("t2va")
    worker = VideoGenerationWorker(0)
    worker.backend = Mock()
    worker.generate_step("prompt", 1, None, True, inputs)
    worker.backend.generate_step.assert_called_once_with("prompt", 1, None, True, generation_inputs=inputs)


def test_full_h3_uses_full_weights_without_preview_lora(monkeypatch, fastvideo_api):
    backend = prepared_backend(monkeypatch)
    old_generator = backend.generator
    fastvideo_api.return_value = RecordingGenerator()
    monkeypatch.setattr("dreamverse.minimax_h3_generation.DREAMVERSE_SP_SIZE", 4)
    backend.initialize(MODEL_REGISTRY["full-h3"])
    config = fastvideo_api.call_args.args[0]
    assert old_generator.closed
    assert config.pipeline.components.lora_path is None
    assert config.pipeline.components.override_pipeline_cls_name is None
    assert config.engine.use_fsdp_inference
    assert config.engine.num_gpus == 4
    assert not config.pipeline.experimental["inference_torch_compile"]


def test_fl2va_maps_endpoints_only_on_initial_segment(monkeypatch, fastvideo_api, tmp_path):
    first = tmp_path / "first.png"
    last = tmp_path / "last.png"
    Image.new("RGB", (3, 2), (10, 20, 30)).save(first)
    Image.new("RGB", (3, 2), (40, 50, 60)).save(last)
    inputs = GenerationInputs("fl2va", (
        GenerationAsset("first", "image", str(first), "first_frame"),
        GenerationAsset("last", "image", str(last), "last_frame"),
    ))
    backend = prepared_backend(monkeypatch)
    first_result = backend.generate_step("first", 1, None, True, inputs)
    later_result = backend.generate_step("later", 2, None, False, inputs)
    assert backend.generator.images[0][0][0, 0].tolist() == [10, 20, 30]
    assert backend.generator.images[0][1][0, 0].tolist() == [40, 50, 60]
    assert backend.generator.images[1][0][0, 0].tolist() == [29, 29, 29]
    assert backend.generator.images[1][1] is None
    assert first_result.head_trim_frames == 0
    assert later_result.head_trim_frames == 1
    assert backend.generator.requests[0].sampling.num_inference_steps == 50


def test_ref2va_switches_pipeline_and_preserves_reference_order(monkeypatch, fastvideo_api):
    inputs = GenerationInputs("ref2va", (
        GenerationAsset("video", "video", "/assets/reference.mp4", "reference"),
        GenerationAsset("audio", "audio", "/assets/reference.wav", "reference"),
        GenerationAsset("image", "image", "/assets/reference.png", "reference"),
    ))
    backend = prepared_backend(monkeypatch)
    base_generator = backend.generator
    reference_generator = RecordingGenerator()

    def load(config):
        assert base_generator.closed, "Old executor must release memory before loading reference weights"
        assert config.pipeline.components.override_pipeline_cls_name == "MiniMaxH3Ref2VAModularPipeline"
        assert config.pipeline.workload_type == "i2v"
        assert config.pipeline.components.lora_path is None
        return reference_generator

    fastvideo_api.side_effect = load
    backend.generate_step("first", 1, None, True, inputs)
    result = backend.generate_step("second", 2, None, False, inputs)
    assert fastvideo_api.call_count == 1
    for request in reference_generator.requests:
        assert [(reference.media_type, reference.source) for reference in request.inputs.references] == [
            ("video", "/assets/reference.mp4"), ("audio", "/assets/reference.wav"), ("image", "/assets/reference.png")
        ]
        assert request.inputs.pil_image is None
        assert request.inputs.last_image is None
    assert result.head_trim_frames == result.head_trim_audio_frames == 0
    assert backend.continuation_image is None

    fastvideo_api.side_effect = None
    fastvideo_api.return_value = RecordingGenerator()
    backend.generate_step("new project", 1, None, True, GenerationInputs("t2va"))
    assert reference_generator.closed
    config = fastvideo_api.call_args.args[0]
    assert config.pipeline.components.override_pipeline_cls_name is None
    assert backend.pipeline_mode == "base"


def test_ref2va_pipeline_switch_failure_drops_unloaded_executor(monkeypatch, fastvideo_api):
    backend = prepared_backend(monkeypatch)
    old_generator = backend.generator
    fastvideo_api.side_effect = RuntimeError("checkpoint unavailable")
    with pytest.raises(RuntimeError, match="checkpoint unavailable"):
        backend.generate_step("prompt", 1, None, True, GenerationInputs("ref2va"))
    assert old_generator.closed
    assert backend.generator is None


def test_mode_cannot_switch_mid_project(monkeypatch, fastvideo_api):
    backend = prepared_backend(monkeypatch)
    with pytest.raises(ValueError, match="middle of a project"):
        backend.generate_step("prompt", 2, None, False, GenerationInputs("ref2va"))
    fastvideo_api.assert_not_called()


@pytest.mark.parametrize("mode", ["fl2va", "ref2va"])
def test_preview_rejects_unsupported_generation_modes(monkeypatch, fastvideo_api, mode):
    backend = prepared_backend(monkeypatch)
    backend.model_config = dict(MODEL_REGISTRY["fast-h3"])
    with pytest.raises(ValueError, match="full-h3"):
        backend.generate_step("prompt", 1, None, True, GenerationInputs(mode))
    assert backend.generator.requests == []


def test_legacy_h3_continuation_is_preserved(monkeypatch, fastvideo_api):
    backend = prepared_backend(monkeypatch)
    backend.generate_step("first", 1, None, False)
    result = backend.generate_step("second", 2, None, False)
    assert backend.generator.requests[0].inputs.pil_image is None
    assert backend.generator.images[1][0][0, 0].tolist() == [29, 29, 29]
    assert result.head_trim_frames == 1
