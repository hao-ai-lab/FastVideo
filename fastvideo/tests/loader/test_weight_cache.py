# SPDX-License-Identifier: Apache-2.0
"""IPC bindings preserve registration/alias identity and reject stale layouts."""
from dataclasses import replace
import multiprocessing as mp
from pathlib import Path
import time

import pytest
import torch

from fastvideo.models.loader.weight_cache import (
    WeightCacheClient, WeightCacheKey, _bind, _inventory, _manifest, serve_weight_cache,
)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2., 3.]))
        self.alias = self.weight
        self.register_buffer("offset", torch.tensor([5., 7.]), persistent=False)
        self.requests = 0

    def forward(self, x):
        self.requests += 1
        return x * self.weight + self.offset


def key():
    return WeightCacheKey("immutable-test-revision", "tiny", "config-sha", "torch.float32", "native-v1")


def test_binding_preserves_ties_and_nonpersistent_buffers():
    source = Tiny().eval()
    tensors, _, _ = _inventory(source)
    with torch.device("meta"):
        target = Tiny()
    _bind(target, tensors, _manifest(source))
    assert target.weight is target.alias
    assert target.weight.data_ptr() == source.weight.data_ptr()
    assert "offset" not in target.state_dict()
    torch.testing.assert_close(target(torch.tensor([11., 13.])), torch.tensor([27., 46.]), rtol=0, atol=0)
    assert source.requests == 0 and target.requests == 1


def test_incompatible_layout_is_rejected_before_binding():
    source = Tiny()
    with torch.device("meta"):
        target = Tiny()
    manifest = _manifest(source)
    manifest["specs"]["weight"]["shape"] = (3,)
    with pytest.raises(ValueError, match="do not match"):
        _bind(target, _inventory(source)[0], manifest)
    assert all(tensor.is_meta for tensor in target.parameters())


def test_missing_cache_never_falls_back_to_disk(tmp_path):
    with pytest.raises(FileNotFoundError):
        with WeightCacheClient(tmp_path / "missing.sock", key(), torch.device("cuda:0")).load_model(Tiny):
            pytest.fail("A missing donor must not load from disk")


def test_identity_requires_every_field():
    with pytest.raises(ValueError, match="nonempty"):
        replace(key(), checkpoint="")


def _donor(path, stop):
    model = Tiny().cuda().eval()
    serve_weight_cache(model, key(), path, stop=stop)
    torch.testing.assert_close(model.weight.cpu(), torch.tensor([2., 3.]), rtol=0, atol=0)


def _consumer(path, queue, mismatched=False):
    client_key = replace(key(), checkpoint="wrong-revision") if mismatched else key()
    try:
        baseline = torch.cuda.memory_allocated()
        with WeightCacheClient(path, client_key).load_model(Tiny) as attached:
            model = attached.model
            assert model.weight is model.alias
            result = model(torch.tensor([11., 13.], device="cuda")).cpu().tolist()
            assert model.requests == 1
            queue.put({"result": result, "pid": attached.donor_pid, "handles": attached.handles,
                       "allocated": torch.cuda.memory_allocated() - baseline})
        assert model.weight.is_meta  # The model cannot be used beyond its lease.
    except ValueError as error:
        if not mismatched:
            raise
        queue.put({"rejected": str(error)})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA IPC requires one GPU")
def test_donor_survives_two_fresh_consumers_and_rejected_identity(tmp_path):
    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    queue = ctx.Queue()
    path = str(tmp_path / "weights.sock")
    donor = ctx.Process(target=_donor, args=(path, stop))
    donor.start()
    records = []
    try:
        deadline = time.monotonic() + 90
        while not Path(path).exists():
            assert donor.is_alive(), "Donor died before readiness"
            assert time.monotonic() < deadline, "Donor readiness timeout"
            time.sleep(0.1)
        for mismatched in (False, True, False):
            worker = ctx.Process(target=_consumer, args=(path, queue, mismatched))
            worker.start()
            try:
                worker.join(90)
                assert worker.exitcode == 0
                records.append(queue.get(timeout=10))
            finally:
                if worker.is_alive():
                    worker.terminate()
                    worker.join(10)
        assert records[0]["result"] == records[2]["result"] == [27., 46.]
        assert records[0]["pid"] == records[2]["pid"] == donor.pid
        assert records[0]["handles"] == records[2]["handles"]
        assert "rejected" in records[1]
        assert donor.is_alive()
    finally:
        stop.set()
        donor.join(10)
        if donor.is_alive():
            donor.terminate()
            donor.join(10)
        queue.close()
    assert donor.exitcode == 0


def test_incomplete_tensor_set_does_not_partially_bind():
    source = Tiny()
    with torch.device("meta"):
        target = Tiny()
    tensors = _inventory(source)[0]
    tensors.pop("offset")
    with pytest.raises(ValueError, match="Incomplete"):
        _bind(target, tensors, _manifest(source))
    assert target.weight.is_meta and target.offset.is_meta


def test_donor_rejects_training_models_before_serving(tmp_path):
    with pytest.raises(ValueError, match="evaluation"):
        serve_weight_cache(Tiny(), key(), tmp_path / "weights.sock")
    assert not (tmp_path / "weights.sock").exists()


def test_donor_rejects_cpu_weights_before_serving(tmp_path):
    with pytest.raises(ValueError, match="one CUDA GPU"):
        serve_weight_cache(Tiny().eval(), key(), tmp_path / "weights.sock")
    assert not (tmp_path / "weights.sock").exists()
