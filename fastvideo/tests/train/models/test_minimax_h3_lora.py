# SPDX-License-Identifier: Apache-2.0
"""Exact adapter-coverage contracts for MiniMax H3 LoRA training."""

from types import SimpleNamespace

import torch

from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3Config
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.lora.linear import BaseLayerWithLoRA
from fastvideo.models.dits.minimax_h3 import MiniMaxH3Transformer3DModel
from fastvideo.train.models.minimax_h3.minimax_h3 import MiniMaxH3LoraModel
from fastvideo.train.models.minimax_h3.minimax_h3_ref2va import MiniMaxH3Ref2VALoraModel
from fastvideo.train.utils.lora import enable_lora_training, finalize_lora_training


_H3_LORA_TARGETS = [
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out",
    "ff.fc_in",
    "ff.fc_out",
]


class _TinyH3Transformer(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(arch_config=SimpleNamespace(exclude_lora_layers=[]))
        self.to_q = ReplicatedLinear(4, 4, bias=False)
        self.patch_size = (1, 2, 2)


def test_h3_plugin_applies_lora_before_component_loader_returns(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def _load_module_from_path(**kwargs):
        transformer = _TinyH3Transformer()
        transform = kwargs["pre_fsdp_model_transform"]
        assert callable(transform)
        transform(transformer)
        observed["wrapped_before_return"] = isinstance(transformer.to_q, BaseLayerWithLoRA)
        return transformer

    monkeypatch.setattr("fastvideo.train.models.minimax_h3.minimax_h3.load_module_from_path", _load_module_from_path)
    training_config = SimpleNamespace(
        pipeline_config=SimpleNamespace(dit_config=SimpleNamespace(uniform_parameter_dtype=False)),
        data=SimpleNamespace(
            train_batch_size=1,
            training_cfg_rate=0.0,
            preprocessed_data_type="t2va",
            seed=42,
        ),
        model=SimpleNamespace(enable_gradient_checkpointing_type=None),
    )

    model = MiniMaxH3LoraModel(
        init_from="unused",
        training_config=training_config,
        lora={
            "enable": True,
            "rank": 2,
            "alpha": 2,
            "target_modules": ["to_q"],
        },
        expected_lora_layers=1,
    )

    assert observed["wrapped_before_return"] is True
    assert model.transformer.to_q.cpu_weight is None
    assert model.transformer.to_q.lora_A.requires_grad
    assert model.transformer.to_q.lora_B.requires_grad
    assert not model.transformer.to_q.base_layer.weight.requires_grad


def test_h3_lora_exact_target_coverage_and_no_cpu_snapshots() -> None:
    config = MiniMaxH3Config()
    with torch.device("meta"):
        transformer = MiniMaxH3Transformer3DModel(config=config, hf_config={})

    converted = enable_lora_training(
        transformer,
        lora_rank=32,
        lora_alpha=32,
        lora_target_modules=_H3_LORA_TARGETS,
        prepare_for_fsdp=True,
        initialization_seed=42,
    )
    wrappers = [module for module in transformer.modules() if isinstance(module, BaseLayerWithLoRA)]

    # Fifty main blocks plus two token-refiner blocks, with six projections
    # selected in each block.
    assert converted == len(wrappers) == 52 * 6 == 312
    assert len(transformer._fastvideo_checkpoint_key_aliases) == 312
    assert all(wrapper.cpu_weight is None for wrapper in wrappers)
    assert all(wrapper.training_mode for wrapper in wrappers)

    assert finalize_lora_training(transformer) == 312
    trainable = [(name, parameter) for name, parameter in transformer.named_parameters() if parameter.requires_grad]
    assert len(trainable) == 624
    assert all(name.endswith((".lora_A", ".lora_B")) for name, _ in trainable)
    assert sum(parameter.numel() for _, parameter in trainable) == 172_949_504
    assert MiniMaxH3Ref2VALoraModel._transformer_module_type == "transformer_ref"
