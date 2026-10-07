import os
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import fastvideo.envs as envs
import fastvideo.platforms as platforms
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines import ForwardBatch
from fastvideo.worker.gpu_worker import Worker, _log_cuda_device_uuid


@pytest.mark.parametrize("dist_timeout", [0, -1])
def test_fastvideo_args_rejects_non_positive_dist_timeout(dist_timeout) -> None:
    with pytest.raises(ValueError, match="greater than zero seconds"):
        FastVideoArgs(model_path="test", dist_timeout=dist_timeout)


def test_worker_threads_dist_timeout_to_distributed_groups(monkeypatch, env_overrides) -> None:
    init_distributed = Mock()
    # init_device() writes LOCAL_RANK; restore it after the test.
    env_overrides.enter_context(envs.override_external("LOCAL_RANK", "0"))
    monkeypatch.setattr("fastvideo.worker.gpu_worker.get_local_torch_device", lambda: torch.device("cpu"))
    # Fake the platform behind fastvideo.platforms.current_platform, the seam the
    # attention-selector tests use. Patching current_platform itself would fetch
    # it through the module's __getattr__, and monkeypatch's undo would then pin
    # the real platform as a module attribute that hides every later
    # _current_platform fake for the rest of the session.
    monkeypatch.setattr(platforms, "_current_platform",
                        SimpleNamespace(is_mps=lambda: False,
                                        is_cuda_alike=lambda: False,
                                        is_cuda=lambda: False,
                                        has_unified_memory=lambda device_id=0: False))
    monkeypatch.setattr("fastvideo.worker.gpu_worker.maybe_init_distributed_environment_and_model_parallel",
                        init_distributed)
    monkeypatch.setattr("fastvideo.worker.gpu_worker.build_pipeline", lambda args: SimpleNamespace())
    args = FastVideoArgs(model_path="test",
                         num_gpus=2,
                         sp_size=2,
                         dist_timeout=7,
                         distributed_executor_backend="external_launcher")
    worker = Worker(args, local_rank=1, rank=1, distributed_init_method="env://")

    worker.init_device()

    init_distributed.assert_called_once_with(1, 2, "env://", timeout=timedelta(seconds=7))


def test_cuda_device_uuid_receipt_is_disabled_without_nvtx_profiling(monkeypatch, env_overrides) -> None:
    """Avoid NVIDIA property access during ordinary worker initialization."""
    get_device_properties = Mock()
    env_overrides.enter_context(envs.FASTVIDEO_NVTX_PROFILE.override(False))
    monkeypatch.setattr(torch.cuda, "get_device_properties", get_device_properties)

    _log_cuda_device_uuid(0, torch.device("cuda:0"))

    get_device_properties.assert_not_called()


def test_cuda_device_uuid_receipt_identifies_profiled_worker(monkeypatch, env_overrides) -> None:
    """Bind one profiled worker rank to its NVIDIA device UUID in logs."""
    log_info = Mock()
    env_overrides.enter_context(envs.FASTVIDEO_NVTX_PROFILE.override(True))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(uuid="device-uuid"))
    monkeypatch.setattr("fastvideo.worker.gpu_worker.logger.info", log_info)

    _log_cuda_device_uuid(2, torch.device("cuda:0"))

    log_info.assert_called_once_with(
        "Worker %d CUDA device UUID: GPU-%s",
        2,
        "device-uuid",
        local_main_process_only=False,
    )


@pytest.mark.parametrize("executor_backend", ["mp", "ray"])
def test_init_device_applies_offload_policy_after_binding_worker_device(monkeypatch, env_overrides,
                                                                        executor_backend: str) -> None:
    """The runtime probe must see this worker's device, never driver device 0."""
    events = []
    args = FastVideoArgs(model_path="test", num_gpus=1, distributed_executor_backend=executor_backend)
    args.finalize_device_offload_policy = Mock(side_effect=lambda device_id: events.append(("policy", device_id)))
    worker = Worker(args, local_rank=3, rank=3, distributed_init_method="env://")

    env_overrides.enter_context(envs.override_external("LOCAL_RANK", "0"))
    # init_device() also writes RANK and WORLD_SIZE for these backends. A leaked
    # RANK reaches every later child process (the profiler names its per-rank
    # summary after it).
    env_overrides.enter_context(envs.override_external("RANK", None))
    env_overrides.enter_context(envs.override_external("WORLD_SIZE", None))
    monkeypatch.setattr("fastvideo.platforms.current_platform.is_cuda_alike", lambda: True)
    monkeypatch.setattr("fastvideo.platforms.current_platform.is_cuda", lambda: False)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: events.append(("set_device", device.index)))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (123, 456))
    monkeypatch.setattr(
        "fastvideo.worker.gpu_worker.maybe_init_distributed_environment_and_model_parallel",
        lambda *args, **kwargs: events.append(("distributed", None)),
    )
    monkeypatch.setattr("fastvideo.worker.gpu_worker.build_pipeline", lambda args: events.append(("pipeline", None)))

    worker.init_device()

    assert events == [
        ("set_device", 3),
        ("policy", 3),
        ("distributed", None),
        ("pipeline", None),
    ]
    assert os.environ["LOCAL_RANK"] == "3"
    assert worker.device == torch.device("cuda:3")
    assert worker.init_gpu_memory == 123


def _worker_returning(output_batch: ForwardBatch) -> Worker:
    worker = Worker.__new__(Worker)
    worker.fastvideo_args = SimpleNamespace(is_output_rank=True)
    worker.pipeline = SimpleNamespace(forward=lambda batch, args: output_batch)
    return worker


def test_execute_forward_drops_metadata_only_output_before_transport():
    output = torch.ones((1, 3, 2, 4, 4))
    output_batch = ForwardBatch(data_type="video", output=output)
    worker = _worker_returning(output_batch)
    request_batch = ForwardBatch(data_type="video", save_video=False, return_frames=False)

    result = worker.execute_forward(request_batch, FastVideoArgs(model_path="test"))

    assert result.output is not None
    assert result.output.device.type == "cpu"
    assert result.output.numel() == 0


def test_execute_forward_preserves_missing_metadata_only_output():
    output_batch = ForwardBatch(data_type="video", output=None)
    worker = _worker_returning(output_batch)
    request_batch = ForwardBatch(data_type="video", save_video=False, return_frames=False)

    result = worker.execute_forward(request_batch, FastVideoArgs(model_path="test"))

    assert result.output is None


def test_execute_forward_drops_save_only_latent_output():
    output = torch.ones((1, 16, 1, 2, 2))
    output_batch = ForwardBatch(data_type="video", output=output)
    worker = _worker_returning(output_batch)
    request_batch = ForwardBatch(data_type="video", save_video=True, return_frames=False)

    result = worker.execute_forward(request_batch, FastVideoArgs(model_path="test", output_type="latent"))

    assert result.output is not None
    assert result.output.device.type == "cpu"
    assert result.output.numel() == 0


def test_execute_forward_drops_save_only_audio_placeholder():
    output = torch.ones((1, 3, 1, 8, 8))
    output_batch = ForwardBatch(data_type="audio", output=output, extra={"audio_only": True})
    worker = _worker_returning(output_batch)
    request_batch = ForwardBatch(data_type="audio", save_video=True, return_frames=False)

    result = worker.execute_forward(request_batch, FastVideoArgs(model_path="test"))

    assert result.output is not None
    assert result.output.device.type == "cpu"
    assert result.output.numel() == 0


@pytest.mark.parametrize(
    ("save_video", "return_frames"),
    [
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_execute_forward_preserves_requested_output(save_video, return_frames):
    output = torch.ones((1, 3, 2, 4, 4))
    output_batch = ForwardBatch(data_type="video", output=output)
    worker = _worker_returning(output_batch)
    request_batch = ForwardBatch(data_type="video", save_video=save_video, return_frames=return_frames)

    result = worker.execute_forward(request_batch, FastVideoArgs(model_path="test"))

    assert result.output is output
