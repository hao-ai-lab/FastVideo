"""Whole-parameter LoRA payload: key classification, application, and reporting.

CPU-only. Nothing here loads a model -- the point is the key algebra and the arithmetic,
which is where a wrong answer is silent rather than loud.
"""
import logging

import pytest
import torch
from safetensors.torch import save_file

from fastvideo.models.loader.lora_patch import DenseLoRAPatch, normalize_lora_key



def write_adapter(tmp_path, tensors, name="adapter_model.safetensors"):
    path = tmp_path / name
    save_file(tensors, str(path))
    return str(path)


# --------------------------------------------------------------------------- keys


@pytest.mark.parametrize(
    "raw, expected",
    [
        # PEFT, the spelling FastVideo's own extractor emits.
        ("blocks.0.attn.to_q.lora_A.weight", "blocks.0.attn.to_q.lora_A.weight"),
        # PEFT with a named adapter: the name sits between the factor and `.weight`.
        ("blocks.0.attn.to_q.lora_A.default.weight", "blocks.0.attn.to_q.lora_A.weight"),
        ("blocks.0.attn.to_q.lora_B.turbo_v2.weight", "blocks.0.attn.to_q.lora_B.weight"),
        # kohya / ComfyUI up-down spelling, with and without `.weight`.
        ("blocks.0.attn.to_q.lora_down.weight", "blocks.0.attn.to_q.lora_A.weight"),
        ("blocks.0.attn.to_q.lora_up.weight", "blocks.0.attn.to_q.lora_B.weight"),
        ("blocks.0.attn.to_q.lora_down", "blocks.0.attn.to_q.lora_A"),
        # kohya alpha.
        ("blocks.0.attn.to_q.alpha", "blocks.0.attn.to_q.lora_alpha"),
        # The `diffusion_model.` prefix ComfyUI-format files carry.
        ("diffusion_model.blocks.0.mlp.fc1.lora_up.weight", "blocks.0.mlp.fc1.lora_B.weight"),
    ],
)
def test_normalize_lora_key_accepts_every_published_spelling(raw, expected):
    assert normalize_lora_key(raw) == expected


@pytest.mark.parametrize("raw", [
    "blocks.0.norm1.diff",
    "blocks.0.attn.to_q.diff_b",
    "blocks.0.attn.to_gate_compress.set_weight",
    "blocks.0.attn.to_q.dora_scale",
])
def test_normalize_lora_key_disclaims_non_low_rank_keys(raw):
    """These belong to the loader-side patch, not the module merge path."""
    assert normalize_lora_key(raw) is None


def test_set_weight_is_not_mangled_into_set():
    """`.replace('.weight', '')` downstream would turn `.set_weight` into `.set`."""
    assert normalize_lora_key("blocks.0.attn.to_gate_compress.set_weight") is None


# --------------------------------------------------------------------- classification


def test_from_adapter_returns_none_for_a_purely_low_rank_adapter(tmp_path):
    path = write_adapter(
        tmp_path, {
            "blocks.0.attn.to_q.lora_A.weight": torch.zeros(4, 8),
            "blocks.0.attn.to_q.lora_B.weight": torch.zeros(8, 4),
        })
    assert DenseLoRAPatch.from_adapter(path) is None


def test_from_adapter_returns_none_without_a_path():
    assert DenseLoRAPatch.from_adapter(None) is None


def test_from_adapter_splits_additive_from_replacement(tmp_path):
    path = write_adapter(
        tmp_path, {
            "blocks.0.attn.to_q.lora_A.weight": torch.zeros(4, 8),
            "blocks.0.norm1.diff": torch.zeros(8),
            "blocks.0.attn.to_q.diff_b": torch.zeros(8),
            "blocks.0.attn.to_gate_compress.set_weight": torch.zeros(8, 8),
        })
    patch = DenseLoRAPatch.from_adapter(path)
    assert patch is not None
    # `.diff` targets `.weight`; `.diff_b` targets `.bias`; `.set_weight` targets `.weight`.
    assert set(patch._additive) == {"blocks.0.norm1.weight", "blocks.0.attn.to_q.bias"}
    assert set(patch._replacement) == {"blocks.0.attn.to_gate_compress.weight"}


def test_param_names_mapping_is_applied_to_dense_keys(tmp_path):
    """An adapter written against the published layout needs no separate rename table."""
    path = write_adapter(tmp_path, {"blocks.0.ff.net.0.proj.diff": torch.zeros(4)})

    def mapping(name):
        return name.replace("ff.net.0.proj", "ff.fc_in"), None, None

    patch = DenseLoRAPatch.from_adapter(path, mapping)
    assert set(patch._additive) == {"blocks.0.ff.fc_in.weight"}


def test_fused_target_is_refused_rather_than_guessed(tmp_path):
    path = write_adapter(tmp_path, {"blocks.0.attn.to_q.diff": torch.zeros(4)})

    def fusing_mapping(name):
        return "blocks.0.attn.to_qkv.weight", 0, 3

    with pytest.raises(NotImplementedError, match="fused parameter"):
        DenseLoRAPatch.from_adapter(path, fusing_mapping)


def test_two_keys_onto_one_parameter_is_an_error(tmp_path):
    path = write_adapter(tmp_path, {
        "blocks.0.norm1.diff": torch.zeros(4),
        "blocks.0.norm1.set_weight": torch.zeros(4),
    })
    # `.diff` and `.set_weight` land in different tables, so they do not collide; two
    # keys of the *same* kind must.
    patch = DenseLoRAPatch.from_adapter(path)
    assert set(patch._additive) == {"blocks.0.norm1.weight"}
    assert set(patch._replacement) == {"blocks.0.norm1.weight"}


# ---------------------------------------------------------------------- application


def test_apply_to_adds_the_delta(tmp_path):
    path = write_adapter(tmp_path, {"blocks.0.norm1.diff": torch.full((4, ), 0.25)})
    patch = DenseLoRAPatch.from_adapter(path)
    out = patch.apply_to("blocks.0.norm1.weight", torch.ones(4))
    assert torch.allclose(out, torch.full((4, ), 1.25))


def test_apply_to_leaves_unrelated_parameters_untouched(tmp_path):
    path = write_adapter(tmp_path, {"blocks.0.norm1.diff": torch.full((4, ), 0.25)})
    patch = DenseLoRAPatch.from_adapter(path)
    base = torch.ones(4)
    assert patch.apply_to("blocks.9.norm1.weight", base) is base


def test_apply_to_sums_in_float32_so_a_small_delta_survives(tmp_path):
    """A bfloat16 add would quantize away a delta far below the base weight.

    The VSA gates and the norm deltas both sit three to four orders of magnitude under
    the weights they modify, which is exactly where bfloat16's 8-bit significand drops
    the change entirely.
    """
    delta = 1e-4
    path = write_adapter(tmp_path, {"blocks.0.norm1.diff": torch.full((4, ), delta, dtype=torch.bfloat16)})
    patch = DenseLoRAPatch.from_adapter(path)
    base = torch.ones(4, dtype=torch.bfloat16)

    out = patch.apply_to("blocks.0.norm1.weight", base)
    assert out.dtype == torch.float32
    assert (out - 1.0).abs().min() > 0, "delta was rounded away"

    naive = (base + torch.full((4, ), delta, dtype=torch.bfloat16))
    assert torch.equal(naive, base), "precondition: the bfloat16 add is what loses it"


def test_apply_to_rejects_a_shape_mismatch(tmp_path):
    path = write_adapter(tmp_path, {"blocks.0.norm1.diff": torch.zeros(8)})
    patch = DenseLoRAPatch.from_adapter(path)
    with pytest.raises(ValueError, match="shape"):
        patch.apply_to("blocks.0.norm1.weight", torch.zeros(4))


def test_replacement_for_returns_the_whole_tensor(tmp_path):
    gate = torch.randn(6, 4)
    path = write_adapter(tmp_path, {"blocks.0.attn.to_gate_compress.set_weight": gate})
    patch = DenseLoRAPatch.from_adapter(path)
    assert torch.equal(patch.replacement_for("blocks.0.attn.to_gate_compress.weight"), gate)
    assert patch.replacement_for("blocks.0.attn.to_q.weight") is None


# ------------------------------------------------------------------------ reporting


def test_unapplied_keys_are_named_in_the_log(tmp_path, caplog):
    path = write_adapter(tmp_path, {
        "blocks.0.norm1.diff": torch.zeros(4),
        "blocks.77.norm1.diff": torch.zeros(4),
    })
    patch = DenseLoRAPatch.from_adapter(path)
    patch.apply_to("blocks.0.norm1.weight", torch.zeros(4))

    with caplog.at_level(logging.WARNING):
        patch.report_unapplied()

    assert "blocks.77.norm1.diff" in caplog.text
    assert "blocks.0.norm1.diff" not in caplog.text


def test_a_fully_applied_payload_warns_about_nothing(tmp_path, caplog):
    path = write_adapter(tmp_path, {"blocks.0.norm1.diff": torch.zeros(4)})
    patch = DenseLoRAPatch.from_adapter(path)
    patch.apply_to("blocks.0.norm1.weight", torch.zeros(4))

    with caplog.at_level(logging.WARNING):
        patch.report_unapplied()

    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
