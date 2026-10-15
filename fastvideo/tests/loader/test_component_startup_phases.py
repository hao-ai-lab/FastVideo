# SPDX-License-Identifier: Apache-2.0
"""Lazy checkpoint consumption belongs to hydration, not iterator construction."""
from types import SimpleNamespace

import torch

from fastvideo.models.loader import component_loader, fsdp_load
from fastvideo.models.loader.startup import collect_startup_timings, startup_phase


class TinyModel(torch.nn.Module):
    param_names_mapping = {}

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.weight = torch.nn.Parameter(torch.zeros(2))

    def load_weights(self, weights):
        loaded = set()
        for name, tensor in weights:
            self.weight.data.copy_(tensor)
            loaded.add(name)
        return loaded


def _weights(*args, **kwargs):
    with startup_phase("test.iterator_consumption"):
        yield "weight", torch.tensor([3., 5.])


def _assert_hydration(timings, prefix):
    phases = timings.to_dict()["phases"]
    hydration = next(p for p in phases if p["name"] == f"{prefix}.hydrate")
    consumed = next(p for p in phases if p["name"] == "test.iterator_consumption")
    assert consumed["parent_id"] == hydration["id"]
    assert hydration["status"] == "ok"
    assert any(p["name"] == f"{prefix}.construct" for p in phases)


def test_transformer_records_lazy_consumption_inside_hydration(monkeypatch):
    monkeypatch.setattr(fsdp_load, "safetensors_weights_iterator", _weights)
    with collect_startup_timings() as timings:
        model = fsdp_load.maybe_load_fsdp_model(
            model_cls=TinyModel, init_params={"config": SimpleNamespace(quant_config=None)},
            weight_dir_list=[], device=torch.device("cpu"), hsdp_replicate_dim=1, hsdp_shard_dim=1,
            default_dtype=torch.float32, param_dtype=torch.float32, reduce_dtype=torch.float32,
            fsdp_inference=False, training_mode=False,
        )
    torch.testing.assert_close(model.weight, torch.tensor([3., 5.]), rtol=0, atol=0)
    _assert_hydration(timings, "transformer")


def test_text_encoder_records_lazy_consumption_inside_hydration(monkeypatch, tmp_path):
    monkeypatch.setattr(component_loader, "get_local_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(component_loader.ModelRegistry, "resolve_model_cls", lambda _: (TinyModel, None))
    monkeypatch.setattr(component_loader, "_configure_text_encoder_quantization", lambda *args: None)
    loader = component_loader.TextEncoderLoader()
    monkeypatch.setattr(loader, "_get_all_weights", _weights)
    # The existing counter is normally initialized by the real iterator.
    loader.counter_before_loading_weights = 0.0
    config = SimpleNamespace(architectures=["TinyModel"], quant_config=None, _fsdp_shard_conditions=[])
    args = SimpleNamespace(
        text_encoder_cpu_offload=False, pin_cpu_memory=False,
        disable_offload_on_unified_memory=lambda *args, **kwargs: False,
        override_text_encoder_quant=None, override_text_encoder_safetensors=None,
    )
    with collect_startup_timings() as timings:
        model = loader.load_model(str(tmp_path), config, torch.device("cpu"), args, dtype="fp32")
    torch.testing.assert_close(model.weight, torch.tensor([3., 5.]), rtol=0, atol=0)
    assert not model.training
    _assert_hydration(timings, "text_encoder")
