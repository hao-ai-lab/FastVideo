# SPDX-License-Identifier: Apache-2.0
"""Real safetensors hydration remains equivalent to the combined native loader."""
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from fastvideo.models.loader import component_loader
from fastvideo.models.loader.component_loader import PipelineComponentLoader, VAELoader
from fastvideo.models.loader.startup import collect_startup_timings
from fastvideo.models.loader.utils import set_default_torch_dtype


class TinyVAE(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.weight = torch.nn.Parameter(torch.zeros(2))
        self.register_buffer("scale", torch.zeros(2))

    def forward(self, x):
        return x * self.weight + self.scale


@pytest.mark.parametrize("sharded", [False, True])
def test_native_load_matches_separate_construction_and_hydration(tmp_path, monkeypatch, sharded):
    config = SimpleNamespace(update_model_arch=lambda _: None)
    args = SimpleNamespace(
        model_paths={}, vae_cpu_offload=False,
        pipeline_config=SimpleNamespace(vae_config=config, vae_precision="fp32"),
    )
    monkeypatch.setattr(component_loader, "get_local_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(component_loader, "get_diffusers_config", lambda **_: {"_class_name": "AutoencoderKLWan"})
    monkeypatch.setattr(component_loader.ModelRegistry, "resolve_model_cls", lambda _: (TinyVAE, None))
    tensors = {"weight": torch.tensor([2., 3.]), "scale": torch.tensor([5., 7.])}
    if sharded:
        for key, tensor in tensors.items():
            save_file({key: tensor}, str(tmp_path / f"{key}.safetensors"))
    else:
        save_file(tensors, str(tmp_path / "model.safetensors"))
    loader = VAELoader()
    with collect_startup_timings() as timings:
        combined = PipelineComponentLoader.load_module("vae", str(tmp_path), "diffusers", args)
    with set_default_torch_dtype(torch.float32):
        separate = loader.build_model(TinyVAE, config, torch.device("cpu"))
    assert torch.count_nonzero(separate.weight) == 0
    separate = loader.load_weights(separate, str(tmp_path), class_name="AutoencoderKLWan", config={},
                                   target_device=torch.device("cpu"))
    x = torch.tensor([11., 13.])
    torch.testing.assert_close(combined(x), torch.tensor([27., 46.]), rtol=0, atol=0)
    torch.testing.assert_close(combined(x), separate(x), rtol=0, atol=0)
    assert not combined.training and not separate.training
    for name, tensor in separate.state_dict().items():
        torch.testing.assert_close(tensor, tensors[name], rtol=0, atol=0)
    assert {p["name"] for p in timings.phases} == {
        "component.vae", "vae.construct", "vae.device_transfer", "vae.checkpoint_read", "vae.weight_assignment",
        "vae.finalize",
    }

    root_phase = next(p for p in timings.phases if p["name"] == "component.vae")
    assert all(p["parent_id"] == root_phase["id"] for p in timings.phases if p is not root_phase)


def test_construction_does_not_read_checkpoint(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("construction accessed checkpoint")
    monkeypatch.setattr(component_loader, "safetensors_load_file", unexpected)
    vae = VAELoader().build_model(TinyVAE, None, torch.device("cpu"))
    assert torch.count_nonzero(vae.weight) == 0


def test_hydration_preserves_strict_loading_and_missing_checkpoint_errors(tmp_path):
    loader = VAELoader()
    vae = TinyVAE(None)
    with pytest.raises(ValueError, match="No safetensors"):
        loader.load_weights(vae, str(tmp_path), class_name="AutoencoderKL", config={},
                            target_device=torch.device("cpu"))
    save_file({"weight": torch.ones(2)}, str(tmp_path / "model.safetensors"))
    with pytest.raises(RuntimeError, match="scale"):
        loader.load_weights(vae, str(tmp_path), class_name="AutoencoderKL", config={},
                            target_device=torch.device("cpu"))
