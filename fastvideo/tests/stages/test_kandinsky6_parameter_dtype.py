# SPDX-License-Identifier: Apache-2.0
"""Regression tests for ``Kandinsky6Transformer3DModel._get_parameter_dtype`` (A13): the distill
checkpoint's safetensors file stores its modulation and time-embedding tensors as F32, and diffusers'
``from_pretrained(torch_dtype=bf16)`` keeps only an *exact* dotted-path-component match against
``visual_modulation``/``text_modulation``/``modulation`` fp32 -- not a substring match, so the
differently-named ``va_modulation``/``av_modulation`` cross-modal gates and the ``*_time_embeddings``
towers are rounded to bf16 despite also being fp32 in the raw checkpoint file.

``_get_parameter_dtype`` is a pure function of its ``name``/``default_dtype`` arguments (it never reads
``self``), so the model is constructed with ``__new__`` and never initialized -- no weights, no
``config`` object, no GPU needed. ``TransformerLoader``/``maybe_load_fsdp_model`` calling this hook is
exercised elsewhere (it is the same generic mechanism ``Kandinsky6SRTransformer3DModel`` already uses);
this file only pins the K6-specific fp32/bf16 split.
"""
from __future__ import annotations

import torch

from fastvideo.models.dits.kandinsky6 import Kandinsky6Transformer3DModel
from fastvideo.models.dits.kandinsky6_sr import Kandinsky6SRTransformer3DModel


def _dit() -> Kandinsky6Transformer3DModel:
    return Kandinsky6Transformer3DModel.__new__(Kandinsky6Transformer3DModel)


def test_exact_modulation_components_stay_fp32():
    model = _dit()
    fp32_names = [
        "text_transformer_blocks.0.text_modulation.out_layer.weight",
        "video_text_transformer_blocks.2.text_modulation.out_layer.bias",
        "audio_text_transformer_blocks.1.text_modulation.out_layer.weight",
        "visual_transformer_blocks.0.visual_modulation.out_layer.weight",  # non-multimodal decoder block
        "visual_transformer_blocks.3.videoT.visual_modulation.out_layer.weight",
        "visual_transformer_blocks.3.audioT.visual_modulation.out_layer.bias",
        "out_layer.modulation.out_layer.weight",
        "audio_out_layer.modulation.out_layer.bias",
    ]
    for name in fp32_names:
        assert model._get_parameter_dtype(name, torch.bfloat16) == torch.float32, name


def test_differently_named_cross_modal_gates_and_time_towers_round_to_bf16():
    # These are fp32 in the raw checkpoint file too, but diffusers' EXACT component match does not
    # keep them fp32 -- "va_modulation" as a whole path component is not "modulation".
    model = _dit()
    bf16_names = [
        "visual_transformer_blocks.5.va_modulation.out_layer.weight",
        "visual_transformer_blocks.5.av_modulation.out_layer.bias",
        "video_time_embeddings.in_layer.weight",
        "audio_time_embeddings.out_layer.bias",
        "time_embeddings.in_layer.weight",
        "visual_embeddings.in_layer.weight",
        "text_embeddings.in_layer.weight",
        "video_text_embeddings.norm.weight",
    ]
    for name in bf16_names:
        assert model._get_parameter_dtype(name, torch.bfloat16) == torch.bfloat16, name


def test_default_dtype_is_returned_unchanged_for_non_fp32_parameters():
    model = _dit()
    assert model._get_parameter_dtype("visual_embeddings.in_layer.weight", torch.float16) == torch.float16


def test_k6_and_sr_dit_hooks_are_independent_and_do_not_leak_into_each_other():
    # The K6 (T2VA) DiT and the SR DiT are separate classes with separate hooks and separate marker
    # sets; adding _get_parameter_dtype to Kandinsky6Transformer3DModel must not change
    # Kandinsky6SRTransformer3DModel's behavior (or vice versa) -- e.g. the SR DiT keeps
    # "time_embeddings." fp32 (single-tower, no video/audio-prefixed names), the K6 DiT does not.
    assert Kandinsky6Transformer3DModel._get_parameter_dtype is not Kandinsky6SRTransformer3DModel._get_parameter_dtype

    k6 = _dit()
    sr = Kandinsky6SRTransformer3DModel.__new__(Kandinsky6SRTransformer3DModel)

    assert sr._get_parameter_dtype("time_embeddings.in_layer.weight", torch.bfloat16) == torch.float32
    assert k6._get_parameter_dtype("time_embeddings.in_layer.weight", torch.bfloat16) == torch.bfloat16


def test_other_dit_classes_have_no_parameter_dtype_hook():
    # Sanity/scope check: the loader's mixed-dtype mechanism (component_loader.py /
    # fsdp_load.py's `getattr(model, "_get_parameter_dtype", None)`) is opt-in per DiT class; a
    # plain DiT untouched by this change must not suddenly grow the hook.
    from fastvideo.models.dits.wanvideo import WanTransformer3DModel

    assert getattr(WanTransformer3DModel, "_get_parameter_dtype", None) is None
