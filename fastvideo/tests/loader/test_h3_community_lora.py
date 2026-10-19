# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for community MiniMax-H3 LoRA conversion and the loader's coverage check."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fastvideo.models.loader.h3_community_lora import convert_h3_community_lora
from fastvideo.pipelines.lora_pipeline import LoRAPipeline

HIDDEN, PROJ, FFN, RANK = 8, 6, 10, 3


def _pair(out_rows: int, in_cols: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(RANK, in_cols, generator=generator), torch.randn(out_rows, RANK, generator=generator)


def _delta(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return b.double() @ a.double()


def _fused_adapter(prefix: str = "diffusion_model.") -> dict[str, torch.Tensor]:
    qkv_a, qkv_b = _pair(3 * PROJ, HIDDEN, 1)
    out_a, out_b = _pair(HIDDEN, PROJ, 2)
    fc1_a, fc1_b = _pair(2 * FFN, HIDDEN, 3)
    fc2_a, fc2_b = _pair(HIDDEN, FFN, 4)
    tensors: dict[str, torch.Tensor] = {}
    for scope in ("blocks.0", "token_refiner.blocks.1"):
        for name, (a, b) in {
                "attn.qkv_proj": (qkv_a, qkv_b),
                "attn.out_proj": (out_a, out_b),
                "mlp.fc1": (fc1_a, fc1_b),
                "mlp.fc2": (fc2_a, fc2_b),
        }.items():
            tensors[f"{prefix}{scope}.{name}.lora_A.weight"] = a
            tensors[f"{prefix}{scope}.{name}.lora_B.weight"] = b
    return tensors


def test_fused_qkv_split_reproduces_the_fused_delta_exactly() -> None:
    source = _fused_adapter()
    converted = convert_h3_community_lora(source).tensors
    fused = _delta(source["diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight"],
                   source["diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight"])

    parts = [
        _delta(converted[f"transformer_blocks.0.attn.{name}.lora_A.weight"],
               converted[f"transformer_blocks.0.attn.{name}.lora_B.weight"]) for name in ("to_q", "to_k", "to_v")
    ]

    assert [p.shape for p in parts] == [(PROJ, HIDDEN)] * 3
    torch.testing.assert_close(torch.cat(parts, dim=0), fused, rtol=0, atol=0)


def test_qkv_split_shares_lora_a_across_projections() -> None:
    converted = convert_h3_community_lora(_fused_adapter()).tensors
    a_q = converted["transformer_blocks.0.attn.to_q.lora_A.weight"]
    assert torch.equal(a_q, converted["transformer_blocks.0.attn.to_k.lora_A.weight"])
    assert torch.equal(a_q, converted["transformer_blocks.0.attn.to_v.lora_A.weight"])


def test_swiglu_halves_are_swapped_for_fc1_and_not_for_fc2() -> None:
    source = _fused_adapter()
    converted = convert_h3_community_lora(source).tensors
    fused = _delta(source["diffusion_model.blocks.0.mlp.fc1.lora_A.weight"],
                   source["diffusion_model.blocks.0.mlp.fc1.lora_B.weight"])
    got = _delta(converted["transformer_blocks.0.ff.fc_in.lora_A.weight"],
                 converted["transformer_blocks.0.ff.fc_in.lora_B.weight"])
    torch.testing.assert_close(got, torch.cat([fused[FFN:], fused[:FFN]], dim=0), rtol=0, atol=0)
    assert torch.equal(converted["transformer_blocks.0.ff.fc_out.lora_B.weight"],
                       source["diffusion_model.blocks.0.mlp.fc2.lora_B.weight"])


def test_module_names_cover_attention_mlp_and_refiner_blocks() -> None:
    result = convert_h3_community_lora(_fused_adapter())
    modules = {key.rsplit(".lora_A", 1)[0] for key in result.tensors if ".lora_A" in key}
    expected = {
        f"{scope}.{layer}"
        for scope in ("transformer_blocks.0", "token_refiner.refiner_blocks.1")
        for layer in ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out", "ff.fc_in", "ff.fc_out")
    }
    assert modules == expected
    assert result.source_format == "comfy-fused"
    assert result.source_modules == 8
    assert result.target_modules == 12
    assert result.unrecognized == ()


@pytest.mark.parametrize("prefix", ["", "diffusion_model.", "model.diffusion_model.", "transformer.", "base_model.model."])
def test_trainer_prefixes_are_stripped(prefix: str) -> None:
    converted = convert_h3_community_lora(_fused_adapter(prefix))
    assert converted.target_modules == 12
    assert converted.unrecognized == ()


def test_peft_adapter_name_infix_and_lora_down_up_conventions() -> None:
    a, b = _pair(PROJ, HIDDEN, 5)
    peft = convert_h3_community_lora({
        "base_model.model.blocks.2.attn.out_proj.lora_A.default.weight": a,
        "base_model.model.blocks.2.attn.out_proj.lora_B.default.weight": b,
    })
    kohya_style = convert_h3_community_lora({
        "diffusion_model.blocks.2.attn.out_proj.lora_down.weight": a,
        "diffusion_model.blocks.2.attn.out_proj.lora_up.weight": b,
    })
    assert peft.tensors.keys() == kohya_style.tensors.keys() == {
        "transformer_blocks.2.attn.to_out.lora_A.weight",
        "transformer_blocks.2.attn.to_out.lora_B.weight",
    }
    assert torch.equal(peft.tensors["transformer_blocks.2.attn.to_out.lora_B.weight"], b)


def test_kohya_flattened_names_are_resolved() -> None:
    qkv_a, qkv_b = _pair(3 * PROJ, HIDDEN, 6)
    result = convert_h3_community_lora({
        "lora_unet_blocks_3_attn_qkv_proj.lora_down.weight": qkv_a,
        "lora_unet_blocks_3_attn_qkv_proj.lora_up.weight": qkv_b,
        "lora_unet_blocks_3_attn_qkv_proj.alpha": torch.tensor(6.0),
        "lora_unet_token_refiner_blocks_0_mlp_fc2.lora_down.weight": qkv_a,
        "lora_unet_token_refiner_blocks_0_mlp_fc2.lora_up.weight": qkv_b[:HIDDEN],
    })
    assert result.unrecognized == ()
    assert "transformer_blocks.3.attn.to_v.lora_B.weight" in result.tensors
    assert "token_refiner.refiner_blocks.0.ff.fc_out.lora_A.weight" in result.tensors


@pytest.mark.parametrize("module", ["diffusion_model.blocks.4.adaln_proj.linear", "lora_unet_blocks_4_adaln_proj_linear"])
def test_per_block_adaln_projection_is_mapped(module: str) -> None:
    a, b = _pair(6 * HIDDEN, HIDDEN, 9)
    result = convert_h3_community_lora({f"{module}.lora_down.weight": a, f"{module}.lora_up.weight": b})
    assert result.unrecognized == ()
    assert result.tensors.keys() == {
        "transformer_blocks.4.adaln_proj.linear.lora_A.weight",
        "transformer_blocks.4.adaln_proj.linear.lora_B.weight",
    }
    assert torch.equal(result.tensors["transformer_blocks.4.adaln_proj.linear.lora_B.weight"], b)


def test_alpha_is_carried_to_every_split_projection_and_absent_alpha_is_not_invented() -> None:
    qkv_a, qkv_b = _pair(3 * PROJ, HIDDEN, 7)
    with_alpha = convert_h3_community_lora({
        "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": qkv_a,
        "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight": qkv_b,
        "diffusion_model.blocks.0.attn.qkv_proj.alpha": torch.tensor(16.0),
    }).tensors
    without = convert_h3_community_lora(_fused_adapter()).tensors
    assert [float(with_alpha[f"transformer_blocks.0.attn.{n}.alpha"]) for n in ("to_q", "to_k", "to_v")] == [16.0] * 3
    assert not any(key.endswith(".alpha") for key in without)


def test_native_layout_adapter_passes_through() -> None:
    a, b = _pair(PROJ, HIDDEN, 8)
    result = convert_h3_community_lora({
        "transformer.transformer_blocks.4.attn.to_out.0.lora_A.weight": a,
        "transformer.transformer_blocks.4.attn.to_out.0.lora_B.weight": b,
        "transformer_blocks.4.ff.net.2.lora_A.weight": a,
        "transformer_blocks.4.ff.net.2.lora_B.weight": b,
    })
    assert result.source_format == "native"
    assert set(result.tensors) == {
        "transformer_blocks.4.attn.to_out.lora_A.weight",
        "transformer_blocks.4.attn.to_out.lora_B.weight",
        "transformer_blocks.4.ff.fc_out.lora_A.weight",
        "transformer_blocks.4.ff.fc_out.lora_B.weight",
    }


def test_boundary_projection_names_are_mapped() -> None:
    a, b = _pair(PROJ, HIDDEN, 9)
    result = convert_h3_community_lora({
        f"diffusion_model.{name}.lora_{ab}.weight": tensor
        for name in ("audio_patch_proj", "final_layer.audio_out", "final_layer.adaln_proj.linear")
        for ab, tensor in (("A", a), ("B", b))
    })
    assert {key.rsplit(".lora_", 1)[0] for key in result.tensors} == {"audio_proj_in", "audio_proj_out", "norm_out.linear"}


def test_unrecognized_and_incomplete_keys_are_reported_not_dropped_silently() -> None:
    a, b = _pair(PROJ, HIDDEN, 10)
    result = convert_h3_community_lora({
        "diffusion_model.blocks.0.attn.out_proj.lora_A.weight": a,
        "diffusion_model.blocks.0.attn.out_proj.lora_B.weight": b,
        "diffusion_model.blocks.0.cross_attn.q.lora_A.weight": a,
        "diffusion_model.blocks.0.cross_attn.q.lora_B.weight": b,
        "diffusion_model.blocks.1.attn.out_proj.lora_A.weight": a,
        "something_else": a,
    })
    assert result.target_modules == 1
    assert set(result.unrecognized) == {
        "diffusion_model.blocks.0.cross_attn.q.lora_A.weight",
        "diffusion_model.blocks.0.cross_attn.q.lora_B.weight",
        "diffusion_model.blocks.1.attn.out_proj.lora_A.weight",
        "something_else",
    }


def test_dense_payload_keys_are_left_to_the_dense_patch() -> None:
    result = convert_h3_community_lora({"diffusion_model.blocks.0.attn.q_norm.diff": torch.zeros(2)})
    assert result.tensors == {} and result.unrecognized == () and result.source_format == "unknown"


def test_mismatched_rank_fails_loudly() -> None:
    a, b = _pair(PROJ, HIDDEN, 11)
    with pytest.raises(ValueError, match="do not share a rank"):
        convert_h3_community_lora({
            "diffusion_model.blocks.0.attn.out_proj.lora_A.weight": a,
            "diffusion_model.blocks.0.attn.out_proj.lora_B.weight": b[:, :2],
        })


def _pipeline_with_layers(names: list[str], adapter_modules: list[str], min_fraction: float = 0.5) -> SimpleNamespace:
    layers = SimpleNamespace(all_lora_layers=lambda: iter((name, object()) for name in names))
    adapter = {f"{name}.lora_A": torch.zeros(1) for name in adapter_modules}
    adapter.update({f"{name}.lora_B": torch.zeros(1) for name in adapter_modules})
    return SimpleNamespace(lora_adapters={"default": adapter},
                           lora_layers={"transformer": layers},
                           lora_min_match_fraction=min_fraction)


def test_coverage_check_accepts_a_fully_matched_adapter() -> None:
    names = [f"transformer_blocks.{i}.attn.to_q" for i in range(4)]
    pipeline = _pipeline_with_layers(names, names)
    LoRAPipeline._check_adapter_coverage(pipeline, "default", "adapter.safetensors")


def test_coverage_check_fails_loudly_and_lists_unmatched_modules() -> None:
    pipeline = _pipeline_with_layers(["transformer_blocks.0.attn.to_q"], ["blocks.0.attn.q", "blocks.1.attn.q"])
    with pytest.raises(ValueError, match=r"reaches 0 of 2 adapter modules.*blocks\.0\.attn\.q"):
        LoRAPipeline._check_adapter_coverage(pipeline, "default", "adapter.safetensors")


def test_coverage_check_rejects_a_mostly_unmatched_adapter() -> None:
    layers = [f"transformer_blocks.{i}.attn.to_q" for i in range(2)]
    pipeline = _pipeline_with_layers(layers, layers + [f"unknown.{i}" for i in range(6)])
    with pytest.raises(ValueError, match="at least 50% must match"):
        LoRAPipeline._check_adapter_coverage(pipeline, "default", "adapter.safetensors")


def test_coverage_check_rejects_an_empty_adapter() -> None:
    pipeline = _pipeline_with_layers(["transformer_blocks.0.attn.to_q"], [])
    with pytest.raises(ValueError, match="reaches 0 of 0"):
        LoRAPipeline._check_adapter_coverage(pipeline, "default", None)


def test_coverage_check_does_not_count_a_lora_a_without_lora_b_as_applied() -> None:
    names = [f"transformer_blocks.{i}.attn.to_q" for i in range(4)]
    pipeline = _pipeline_with_layers(names, names[:1])
    # Three native A-only entries: they match layers by name but would never be applied.
    pipeline.lora_adapters["default"].update({f"{name}.lora_A": torch.zeros(1) for name in names[1:]})
    with pytest.raises(ValueError, match=r"reaches 1 of 4 adapter modules"):
        LoRAPipeline._check_adapter_coverage(pipeline, "default", "adapter.safetensors")


def test_coverage_check_does_not_count_a_lora_b_without_lora_a_as_applied() -> None:
    names = [f"transformer_blocks.{i}.attn.to_q" for i in range(2)]
    pipeline = _pipeline_with_layers(names, names[:1])
    pipeline.lora_adapters["default"][f"{names[1]}.lora_B"] = torch.zeros(1)
    LoRAPipeline._check_adapter_coverage(pipeline, "default", "adapter.safetensors")  # 1 of 2 meets 50%
    pipeline.lora_min_match_fraction = 0.6
    with pytest.raises(ValueError, match=r"reaches 1 of 2"):
        LoRAPipeline._check_adapter_coverage(pipeline, "default", "adapter.safetensors")


def test_coverage_check_exempts_a_constructor_dense_only_adapter() -> None:
    pipeline = _pipeline_with_layers(["transformer_blocks.0.attn.to_q"], [])
    pipeline._constructor_dense_lora_path = "dense.safetensors"
    LoRAPipeline._check_adapter_coverage(pipeline, "default", "dense.safetensors")
    with pytest.raises(ValueError, match="reaches 0 of 0"):
        LoRAPipeline._check_adapter_coverage(pipeline, "default", "other.safetensors")


def test_dense_only_adapter_converts_to_nothing_instead_of_raising() -> None:
    from fastvideo.pipelines.basic.minimax_h3.minimax_h3_pipeline import MiniMaxH3BasePipeline

    dense = {
        "diffusion_model.blocks.0.attn.q_norm.diff": torch.zeros(2),
        "diffusion_model.blocks.0.attn.out_proj.set_weight": torch.zeros(2, 2),
    }
    result = convert_h3_community_lora(dense)
    assert result.tensors == {} and len(result.dense_keys) == 2
    assert MiniMaxH3BasePipeline.convert_adapter_state_dict(SimpleNamespace(), dense, "dense") == {}


def test_failed_coverage_restores_the_previously_registered_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    import fastvideo.pipelines.lora_pipeline as lp

    previous = {"transformer_blocks.0.attn.to_q.lora_A": torch.ones(1)}
    stale = {"unknown.0.lora_A.weight": torch.zeros(2, 2), "unknown.0.lora_B.weight": torch.zeros(2, 2)}
    monkeypatch.setattr(lp.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(lp, "maybe_download_lora", lambda path: path)
    monkeypatch.setattr(lp, "load_file", lambda path: dict(stale))
    layers = SimpleNamespace(all_lora_layers=lambda: iter([("transformer_blocks.0.attn.to_q", object())]),
                             lora_layers_by_block=lambda: iter(()))
    pipeline = SimpleNamespace(
        lora_adapters={"default": previous},
        lora_adapter_paths={"default": "a.safetensors"},
        lora_layers={"transformer": layers},
        trainable_transformer_modules={},
        modules={"transformer": SimpleNamespace(param_names_mapping={}, lora_param_names_mapping={})},
        cur_adapter_name="default",
        cur_adapter_path="a.safetensors",
        cur_adapter_strength=1.0,
        lora_initialized=True,
        lora_strict_coverage=True,
        lora_min_match_fraction=0.5,
        _setting_constructor_adapter=True,
        _constructor_dense_lora_path=None,
        device=torch.device("cpu"),
        fastvideo_args=SimpleNamespace(training_mode=False),
        convert_adapter_state_dict=lambda state, path: state,
    )
    pipeline._check_adapter_coverage = lambda *a: LoRAPipeline._check_adapter_coverage(pipeline, *a)
    with pytest.raises(ValueError, match="reaches 0 of 1"):
        LoRAPipeline.set_lora_adapter(pipeline, "default", "b.safetensors")
    assert pipeline.lora_adapters["default"] is previous
    assert pipeline.lora_adapter_paths["default"] == "a.safetensors"


def test_failed_conversion_keeps_the_previously_registered_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    import fastvideo.pipelines.lora_pipeline as lp

    previous = {"transformer_blocks.0.attn.to_q.lora_A": torch.ones(1)}
    monkeypatch.setattr(lp.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(lp, "maybe_download_lora", lambda path: path)
    monkeypatch.setattr(lp, "load_file", lambda path: {"unet.down.0.lora_A.weight": torch.zeros(1)})

    def reject(state, path):
        raise ValueError("no MiniMax-H3 LoRA layers")

    pipeline = SimpleNamespace(
        lora_adapters={"default": previous},
        lora_adapter_paths={"default": "a.safetensors"},
        modules={"transformer": SimpleNamespace(param_names_mapping={}, lora_param_names_mapping={})},
        cur_adapter_name="default",
        cur_adapter_path="a.safetensors",
        cur_adapter_strength=1.0,
        lora_initialized=True,
        lora_strict_coverage=True,
        _setting_constructor_adapter=True,
        _constructor_dense_lora_path=None,
        device=torch.device("cpu"),
        fastvideo_args=SimpleNamespace(training_mode=False),
        convert_adapter_state_dict=reject,
    )
    with pytest.raises(ValueError, match="no MiniMax-H3 LoRA layers"):
        LoRAPipeline.set_lora_adapter(pipeline, "default", "b.safetensors")
    assert pipeline.lora_adapters["default"] is previous
    assert pipeline.lora_adapter_paths["default"] == "a.safetensors"


def test_h3_pipeline_converts_and_rejects_foreign_adapters() -> None:
    from fastvideo.pipelines.basic.minimax_h3.minimax_h3_pipeline import MiniMaxH3BasePipeline

    assert MiniMaxH3BasePipeline.lora_strict_coverage
    pipeline = SimpleNamespace()
    converted = MiniMaxH3BasePipeline.convert_adapter_state_dict(pipeline, _fused_adapter(), "adapter")
    assert "transformer_blocks.0.attn.to_k.lora_B.weight" in converted
    with pytest.raises(ValueError, match="no MiniMax-H3 LoRA layers"):
        MiniMaxH3BasePipeline.convert_adapter_state_dict(pipeline, {"unet.down.0.lora_A.weight": torch.zeros(1)},
                                                         "adapter")
