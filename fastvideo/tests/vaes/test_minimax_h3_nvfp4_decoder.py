# SPDX-License-Identifier: Apache-2.0
"""NVFP4 MiniMax-H3 decoder linears: conversion contract, rotation exactness, train/deploy parity."""
import pytest
import torch
import torch.nn as nn

from fastvideo.models.vaes.minimax_h3_nvfp4_decoder import (
    NVFP4DecoderLinear,
    convert_decoder_to_nvfp4,
    nvfp4_decoder_linear_names,
)
from fastvideo.models.vaes.minimax_h3_video import MiniMaxH3VideoViTDecoder3d
from fastvideo.models.vaes.minimax_h3_int8_convrot import rotate_activation


def _tiny_decoder(num_layers: int = 3) -> MiniMaxH3VideoViTDecoder3d:
    torch.manual_seed(0)
    return MiniMaxH3VideoViTDecoder3d(in_channels=24,
                                      out_channels=3,
                                      patch_size=2,
                                      patch_size_t=1,
                                      num_layers=num_layers,
                                      num_attention_heads=4,
                                      attention_head_dim=64,
                                      num_register_tokens=4,
                                      ffn_mult=4,
                                      rope_theta=100.0,
                                      rope_dim_ratio=0.75,
                                      norm_eps=1e-5)


def test_selects_only_block_linears():
    decoder = _tiny_decoder()
    names = nvfp4_decoder_linear_names(decoder)
    assert len(names) == 3 * 6
    assert all(name.startswith("transformer_blocks.") for name in names)
    assert "proj_in" not in names and "proj_out" not in names


def test_skip_blocks_keeps_them_dense():
    decoder = _tiny_decoder()
    names = convert_decoder_to_nvfp4(decoder, skip_blocks=(0, 2))
    assert {name.split(".")[1] for name in names} == {"1"}
    assert type(decoder.transformer_blocks[0].attn.to_q) is nn.Linear
    assert isinstance(decoder.transformer_blocks[1].attn.to_q, NVFP4DecoderLinear)
    assert isinstance(decoder.transformer_blocks[1].ff.net[2], NVFP4DecoderLinear)
    assert isinstance(decoder.transformer_blocks[1].attn.to_out[0], NVFP4DecoderLinear)


@pytest.mark.parametrize("group", [16, 64, 256])
def test_rotation_is_exact_in_full_precision(group):
    torch.manual_seed(0)
    linear = nn.Linear(256, 96)
    rotated = NVFP4DecoderLinear.from_linear(linear, rotation_group=group, compute_dtype=torch.float32)
    x = torch.randn(5, 7, 256, dtype=torch.float64)
    expected = x @ linear.weight.double().T + linear.bias.double()
    actual = rotate_activation(x, group) @ rotated.weight.double().T + rotated.bias.double()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_rejects_non_blockable_input():
    with pytest.raises(ValueError, match="divisible"):
        NVFP4DecoderLinear.from_linear(nn.Linear(24, 32), rotation_group=None, compute_dtype=torch.bfloat16)


def _fp4_available() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        return False
    try:
        import flashinfer  # noqa: F401
    except ImportError:
        return False
    return True


def test_rejects_unknown_activation_scale():
    with pytest.raises(ValueError, match="act_scale"):
        NVFP4DecoderLinear.from_linear(nn.Linear(32, 32), rotation_group=None, compute_dtype=torch.bfloat16,
                                       act_scale="per-token")


@pytest.mark.skipif(not _fp4_available(), reason="NVFP4 GEMM needs a Blackwell GPU and flashinfer")
@pytest.mark.parametrize("act_scale", ["dynamic", "unit", "static"])
@pytest.mark.parametrize("group", [None, 256])
def test_inference_matches_training_forward_bitwise(group, act_scale):
    from fastvideo.models.vaes.minimax_h3_nvfp4_decoder import calibrate_static_scales

    torch.manual_seed(0)
    linear = nn.Linear(2048, 768).cuda()
    layer = NVFP4DecoderLinear.from_linear(linear, rotation_group=group, compute_dtype=torch.bfloat16,
                                           act_scale=act_scale).cuda()
    x = torch.randn(3, 300, 2048, device="cuda", dtype=torch.bfloat16)
    if act_scale == "static":
        assert calibrate_static_scales(layer, lambda: layer(x)) == 1
        assert layer.input_amax.item() > 0
    with torch.enable_grad():
        train_out = layer(x)
    with torch.no_grad():
        deploy_out = layer(x)
    assert torch.equal(train_out.detach(), deploy_out)
    relative = (deploy_out.float() - linear(x.float())).norm() / linear(x.float()).norm()
    assert relative < 0.15


@pytest.mark.skipif(not _fp4_available(), reason="NVFP4 GEMM needs a Blackwell GPU and flashinfer")
def test_training_forward_has_gradients():
    torch.manual_seed(0)
    layer = NVFP4DecoderLinear.from_linear(nn.Linear(256, 128), rotation_group=64,
                                           compute_dtype=torch.bfloat16).cuda()
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    layer(x).float().square().mean().backward()
    assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_global_scale_matches_reference_reduction(dtype):
    from fastvideo.layers.fp4linear import _global_sf

    torch.manual_seed(0)
    x = (torch.randn(1797, 256) * 7).to(dtype)
    expected = (448.0 * 6.0) / x.float().abs().nan_to_num().max().clamp(min=1e-12)
    assert torch.equal(_global_sf(x), expected)
    with_inf = x.clone()
    with_inf[3, 5] = float("inf")
    expected_inf = (448.0 * 6.0) / with_inf.float().abs().nan_to_num().max().clamp(min=1e-12)
    assert torch.equal(_global_sf(with_inf), expected_inf)


# Fused inference kernels: every one must match the eager op sequence bit for bit.
_needs_fp4 = pytest.mark.skipif(not _fp4_available(), reason="NVFP4 GEMM needs a Blackwell GPU and flashinfer")


def _flashinfer_quantize(x, global_sf):
    import flashinfer

    from fastvideo.layers.quantization.nvfp4_config import nvfp4_quantize_fenced

    return nvfp4_quantize_fenced(x, global_sf, flashinfer.SfLayout.layout_128x4.value)


def _assert_quantized_equal(actual, expected):
    assert actual[0].shape == expected[0].shape and actual[1].shape == expected[1].shape
    assert torch.equal(actual[0].view(torch.uint8), expected[0].view(torch.uint8))
    assert torch.equal(actual[1].view(torch.uint8), expected[1].view(torch.uint8))


@_needs_fp4
@pytest.mark.parametrize("global_sf", [1.0, 2688.0 / 40.0])
@pytest.mark.parametrize("magnitude", [3.0, 3000.0])
def test_fused_quantize_matches_flashinfer(global_sf, magnitude):
    from fastvideo.models.vaes.minimax_h3_nvfp4_fused import quantize

    torch.manual_seed(0)
    x = (torch.randn(1797 * 2 + 5, 2048, device="cuda") * magnitude).bfloat16()
    x[7, :16] = 0  # an all-zero block
    sf = torch.tensor(global_sf, device="cuda")
    _assert_quantized_equal(quantize(x, sf), _flashinfer_quantize(x, sf))


@_needs_fp4
@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("norm", [False, True])
def test_fused_residual_norm_quantize_matches_eager(residual, norm):
    import torch.nn.functional as F

    from fastvideo.models.vaes.minimax_h3_nvfp4_fused import residual_norm_quantize

    if not (residual or norm):
        pytest.skip("nothing to fuse")
    torch.manual_seed(0)
    rows, dim, eps = 1797 * 2 + 5, 2048, 1e-5
    h = torch.randn(rows, dim, device="cuda") * 5
    o = (torch.randn(rows, dim, device="cuda") * 2).bfloat16()
    bias = torch.randn(dim, device="cuda") * 0.1
    scale = torch.randn(dim, device="cuda") * 0.3
    gamma = torch.rand(dim, device="cuda") + 0.5
    sf = torch.ones((), device="cuda")
    expected_h = h
    if residual:
        biased = o.clone()
        biased.add_(bias.to(torch.bfloat16))
        expected_h = h + biased * scale
    actual_h = h.clone()
    quantized = residual_norm_quantize(actual_h, (o, bias.to(torch.bfloat16), scale) if residual else None,
                                       (gamma, eps, sf) if norm else None)
    assert torch.equal(actual_h, expected_h)
    if norm:
        normed = F.rms_norm(expected_h, (dim, ), gamma, eps).to(torch.bfloat16)
        _assert_quantized_equal(quantized, _flashinfer_quantize(normed, sf))


@_needs_fp4
def test_fused_swiglu_quantize_matches_eager():
    import torch.nn.functional as F

    from fastvideo.models.vaes.minimax_h3_nvfp4_fused import swiglu_quantize

    torch.manual_seed(0)
    rows, ffn = 1797 + 5, 8192
    o = (torch.randn(rows, 2 * ffn, device="cuda") * 2).bfloat16()
    bias = torch.randn(2 * ffn, device="cuda").bfloat16()
    sf = torch.ones((), device="cuda")
    biased = o.clone()
    biased.add_(bias)
    value, gate = biased.chunk(2, dim=-1)
    expected = _flashinfer_quantize((value * F.silu(gate)).contiguous(), sf)
    _assert_quantized_equal(swiglu_quantize(o, bias, sf), expected)


@_needs_fp4
def test_fused_qkv_epilogue_matches_eager_attention_prologue():
    import torch.nn.functional as F

    from fastvideo.models.vaes.minimax_h3_nvfp4_fused import qkv_epilogue

    torch.manual_seed(0)
    batch, seq, heads, head_dim, rotary, eps = 2, 1797, 32, 64, 48, 1e-5
    rows = batch * seq
    projections = [(torch.randn(rows, heads * head_dim, device="cuda") * 2).bfloat16() for _ in range(3)]
    biases = tuple((torch.randn(heads * head_dim, device="cuda") * 0.2).bfloat16() for _ in range(3))
    cos = torch.randn(batch, seq, 1, rotary, device="cuda").cos().to(torch.bfloat16)
    sin = torch.randn(batch, seq, 1, rotary, device="cuda").sin().to(torch.bfloat16)
    expected = []
    for projection, bias in zip(projections, biases, strict=True):
        biased = projection.clone()
        biased.add_(bias)
        expected.append(biased.view(batch, seq, heads, head_dim))
    for index in range(2):  # the attention module's q/k RMSNorm and partial RoPE, op by op
        x = F.rms_norm(expected[index].float(), (head_dim, ), None, eps).to(torch.bfloat16)
        x_rotary, x_pass = x[..., :rotary], x[..., rotary:]
        first, second = x_rotary.chunk(2, dim=-1)
        rotated = torch.cat([-second, first], dim=-1)
        expected[index] = torch.cat([x_rotary * cos + rotated * sin, x_pass], dim=-1)
    actual = qkv_epilogue(*projections, biases, cos.reshape(rows, rotary), sin.reshape(rows, rotary), heads, eps)
    for got, want in zip(actual, expected, strict=True):
        assert torch.equal(got.view(batch, seq, heads, head_dim), want)


def _fused_test_decoder(act_scale: str, num_layers: int = 2, convert: bool = True) -> MiniMaxH3VideoViTDecoder3d:
    torch.manual_seed(0)
    decoder = MiniMaxH3VideoViTDecoder3d(in_channels=24,
                                         out_channels=3,
                                         patch_size=2,
                                         patch_size_t=1,
                                         num_layers=num_layers,
                                         num_attention_heads=8,
                                         attention_head_dim=64,
                                         num_register_tokens=4,
                                         ffn_mult=4,
                                         rope_theta=100.0,
                                         rope_dim_ratio=0.75,
                                         norm_eps=1e-5).cuda()
    with torch.no_grad():
        for block in decoder.transformer_blocks:
            block.scale1.normal_(0, 0.5)
            block.scale2.normal_(0, 0.5)
        decoder.register_tokens.normal_()
    if convert:
        convert_decoder_to_nvfp4(decoder, compute_dtype=torch.bfloat16, act_scale=act_scale)
    return decoder.requires_grad_(False).eval()


@_needs_fp4
@pytest.mark.parametrize("act_scale", ["unit", "static"])
@pytest.mark.parametrize("autocast", [True, False])
def test_fused_decoder_matches_eager_decoder_bitwise(act_scale, autocast):
    from fastvideo.models.vaes.minimax_h3_nvfp4_decoder import calibrate_static_scales

    decoder = _fused_test_decoder(act_scale)
    latents = torch.randn(3, 24, 3, 8, 8, device="cuda")
    fused = decoder.fused_blocks_forward
    assert fused is not None

    def run():
        context = torch.autocast("cuda", dtype=torch.bfloat16) if autocast else torch.autocast("cuda", enabled=False)
        with torch.no_grad(), context:
            return decoder(latents)

    if act_scale == "static":
        decoder.fused_blocks_forward = None
        calibrate_static_scales(decoder, run)
        decoder.fused_blocks_forward = fused
    fused_out = run()
    assert all(block._nvfp4_fused_plan[1] is not None for block in decoder.transformer_blocks)
    decoder.fused_blocks_forward = None
    eager_out = run()
    assert torch.equal(fused_out, eager_out)


@_needs_fp4
def test_fused_decoder_falls_back_when_inapplicable():
    from fastvideo.models.vaes.minimax_h3_nvfp4_fused import fused_nvfp4_blocks_forward

    hidden = torch.randn(1, 64, 512, device="cuda")
    rotary = (torch.ones(1, 64, 1, 48, device="cuda"), torch.zeros(1, 64, 1, 48, device="cuda"))
    dynamic = _fused_test_decoder("dynamic", num_layers=1)
    with torch.no_grad():
        assert fused_nvfp4_blocks_forward(dynamic.transformer_blocks, hidden, rotary) is None
    unit = _fused_test_decoder("unit", num_layers=1)
    with torch.no_grad():
        assert fused_nvfp4_blocks_forward(unit.transformer_blocks, hidden, rotary) is not None
        assert fused_nvfp4_blocks_forward(unit.transformer_blocks, hidden.bfloat16(), rotary) is None
    with torch.enable_grad():
        assert fused_nvfp4_blocks_forward(unit.transformer_blocks, hidden, rotary) is None


@_needs_fp4
@pytest.mark.parametrize("rows,out_features,in_features", [(1797, 2048, 2048), (3 * 1797, 4096, 2048),
                                                             (1797, 2048, 8192)])
def test_cutlass_fp4_tactics_are_bit_identical(rows, out_features, in_features):
    import flashinfer

    from fastvideo.models.vaes.minimax_h3_nvfp4_decoder import NVFP4FusedState, _quantized_weight
    from fastvideo.models.vaes.minimax_h3_nvfp4_fused import (_gemm, _gemm_workspace, _sm100_cutlass_fp4_module,
                                                              quantize)

    torch.manual_seed(0)
    sf = torch.ones((), device="cuda")
    quantized = quantize(torch.randn(rows, in_features, device="cuda").bfloat16(), sf)
    packed, inv_scale, global_sf_w = _quantized_weight(
        (torch.randn(out_features, in_features, device="cuda") * 0.02).bfloat16())
    alpha = 1.0 / (sf * global_sf_w)
    expected = torch.empty(rows, out_features, device="cuda", dtype=torch.bfloat16)
    flashinfer.mm_fp4(quantized[0], packed.T, quantized[1], inv_scale.T, alpha, torch.bfloat16, expected,
                      block_size=16, use_8x4_sf_layout=False, backend="cutlass")
    state = NVFP4FusedState(packed, inv_scale, sf, alpha, torch.zeros(out_features, device="cuda"), out_features)
    assert torch.equal(_gemm(quantized, state), expected)
    module = _sm100_cutlass_fp4_module(torch.cuda.current_device())
    if module is None:
        pytest.skip("the direct CUTLASS path is sm_100 only")
    for tactic in range(module.fp4_gemm_tactic_num()):
        out = torch.empty_like(expected)
        module.fp4_gemm(quantized[0], packed, quantized[1], inv_scale, alpha, out, _gemm_workspace(out.device.index),
                        tactic)
        assert torch.equal(out, expected), f"tactic {tactic}"


@_needs_fp4
def test_fused_decoder_cuda_graph_replay_matches_eager():
    decoder = _fused_test_decoder("unit")
    fused = decoder.fused_blocks_forward
    inputs = [torch.randn(2, 24, 3, 8, 8, device="cuda") for _ in range(4)]

    def run(latents):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return decoder(latents)

    # Call 1 warms up eagerly, call 2 captures the graph, calls 3-4 replay it on new inputs.
    fused_outs = [run(latents) for latents in inputs]
    assert len(decoder.transformer_blocks.__dict__["_nvfp4_cuda_graphs"]) == 2
    decoder.fused_blocks_forward = None
    for latents, fused_out in zip(inputs, fused_outs, strict=True):
        assert torch.equal(fused_out, run(latents))
    decoder.fused_blocks_forward = fused


def test_dense_bypass_runs_the_master_weight():
    torch.manual_seed(0)
    linear = nn.Linear(64, 32)
    layer = NVFP4DecoderLinear.from_linear(linear, rotation_group=None, compute_dtype=torch.float32, act_scale="unit")
    layer.dense_bypass = True
    x = torch.randn(5, 64)
    torch.testing.assert_close(layer(x), linear(x))
    assert layer.fused_state() is None


# Inference-only freeze: the packed weight replaces the master, outputs stay bit-identical.
@_needs_fp4
@pytest.mark.parametrize("fused", [True, False])
def test_frozen_decoder_matches_unfrozen_bitwise(fused):
    from fastvideo.models.vaes.minimax_h3_nvfp4_decoder import freeze_nvfp4_linears, nvfp4_linears

    decoder = _fused_test_decoder("unit")
    if not fused:
        decoder.fused_blocks_forward = None
    latents = torch.randn(2, 24, 3, 8, 8, device="cuda")

    def run():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return decoder(latents)

    before = run()
    master_bytes = sum(layer.weight.numel() * layer.weight.element_size() for layer in nvfp4_linears(decoder))
    assert freeze_nvfp4_linears(decoder) == 2 * 6
    layers = nvfp4_linears(decoder)
    assert all(layer.frozen and layer.weight is None for layer in layers)
    names = {name for name, module in decoder.named_modules() if isinstance(module, NVFP4DecoderLinear)}
    assert not names & {key.rpartition(".")[0] for key in decoder.state_dict() if key.endswith(".weight")}
    packed_bytes = sum(buf.numel() * buf.element_size() for layer in layers for buf in layer.buffers())
    assert packed_bytes < master_bytes / 6
    assert torch.equal(run(), before)
    if fused:
        assert all(block._nvfp4_fused_plan[1] is not None for block in decoder.transformer_blocks)
    with pytest.raises(RuntimeError, match="frozen"):
        layers[0].invalidate()


@_needs_fp4
def test_frozen_linear_survives_module_to():
    torch.manual_seed(0)
    layer = NVFP4DecoderLinear.from_linear(nn.Linear(256, 128), rotation_group=None, compute_dtype=torch.bfloat16,
                                           act_scale="unit").cuda().requires_grad_(False)
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        expected = layer(x)
        state = layer.fused_state()
        layer.freeze()
        assert torch.equal(layer(x), expected)
        assert torch.equal(layer.fused_state().packed, state.packed)
        layer.cpu().cuda()
        assert torch.equal(layer(x), expected)
        with torch.enable_grad():
            assert torch.equal(layer(x), expected)


# Loading an exported decoder (export_deploy.py format) into a dense H3 VAE.
_TINY_HEAD_DIM = 16


def _tiny_vae_arch(decoder_layers: int, heads: int = 2) -> dict:
    return {
        "_class_name": "AutoencoderKLMiniMaxH3",
        "latent_channels": 4,
        "block_out_channels": [32, 32],
        "layers_per_block": 1,
        "spatial_downsample_factors": [2, 2],
        "temporal_downsample_factors": [2, 2],
        "decoder_num_layers": decoder_layers,
        "decoder_num_attention_heads": heads,
        "decoder_attention_head_dim": _TINY_HEAD_DIM,
        "decoder_num_register_tokens": 2,
        "decoder_ffn_mult": 2,
        "latents_mean": [0.0] * 4,
        "latents_std": [1.0] * 4,
    }


def _tiny_h3_vae(arch: dict):
    from fastvideo.configs.models.vaes.minimax_h3_video import MiniMaxH3VideoVAEArchConfig, MiniMaxH3VideoVAEConfig
    from fastvideo.models.vaes.minimax_h3_video import AutoencoderKLMiniMaxH3

    fields = {key: tuple(value) if isinstance(value, list) else value for key, value in arch.items() if key[0] != "_"}
    return AutoencoderKLMiniMaxH3(MiniMaxH3VideoVAEConfig(arch_config=MiniMaxH3VideoVAEArchConfig(**fields))).eval()


def _write_vae_dir(path, decoder_layers: int):
    from safetensors.torch import save_file

    torch.manual_seed(1)
    path.mkdir()
    arch = _tiny_vae_arch(decoder_layers)
    (path / "config.json").write_text(__import__("json").dumps(arch))
    save_file({key: value.contiguous() for key, value in _tiny_h3_vae(arch).state_dict().items()},
              str(path / "diffusion_pytorch_model.safetensors"))


def _write_deploy_checkpoint(path, student_layers: int, base_layers: int, heads: int = 2, **metadata):
    from fastvideo.models.vaes.minimax_h3_nvfp4_decoder import (NVFP4_DECODER_DEPLOY_FORMAT,
                                                                keep_evenly_spaced_blocks)

    torch.manual_seed(2)
    vae = _tiny_h3_vae(_tiny_vae_arch(base_layers, heads))
    keep_evenly_spaced_blocks(vae.decoder, student_layers)
    names = convert_decoder_to_nvfp4(vae.decoder, compute_dtype=torch.bfloat16, act_scale="unit")
    with torch.no_grad():
        for parameter in vae.decoder.parameters():
            parameter.normal_()
    decoder = {}
    for key, value in vae.decoder.state_dict().items():
        module, _, param = key.rpartition(".")
        decoder[key] = value.bfloat16() if module in names and param in ("weight", "bias") else value
    torch.save(
        {
            "format": NVFP4_DECODER_DEPLOY_FORMAT,
            "decoder": decoder,
            "post_quant_conv": vae.post_quant_conv.state_dict(),
            "metadata": {
                "act_scale": "unit",
                "rotation_group": None,
                "skip_blocks": [],
                "num_linears": len(names),
                "student_layers": student_layers,
                **metadata,
            },
        }, path)
    return decoder


def _load_h3_vae(vae_dir, cpu: bool = True, decoder: str | None = None, pipeline_config=None):
    from fastvideo.configs.models.vaes.minimax_h3_video import MiniMaxH3VideoVAEConfig
    from fastvideo.configs.pipelines import PipelineConfig
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.models.loader.component_loader import VAELoader

    args = FastVideoArgs(
        model_path=str(vae_dir),
        pipeline_config=pipeline_config or PipelineConfig(vae_config=MiniMaxH3VideoVAEConfig(), vae_precision="fp32"),
        h3_vae_nvfp4_decoder=decoder)
    args.vae_cpu_offload = cpu  # load on CPU
    return VAELoader().load(str(vae_dir), args)


def test_open_h3_regenerate_2k_vae_load_applies_the_exported_decoder(tmp_path):
    """Regenerate-2K loads its ``base_model/vae`` through the same swap, so the option applies there too."""
    from fastvideo.configs.pipelines.open_h3_regenerate_2k import OpenH3Regenerate2KPipelineConfig

    _write_vae_dir(tmp_path / "vae", decoder_layers=5)
    _write_deploy_checkpoint(tmp_path / "d3.pt", student_layers=3, base_layers=5)
    config = OpenH3Regenerate2KPipelineConfig()
    assert len(_load_h3_vae(tmp_path / "vae", pipeline_config=config).decoder.transformer_blocks) == 5
    vae = _load_h3_vae(tmp_path / "vae", decoder=str(tmp_path / "d3.pt"), pipeline_config=config)
    assert len(vae.decoder.transformer_blocks) == 3
    assert vae.decode_autocast_dtype == torch.bfloat16
    assert isinstance(vae.decoder.transformer_blocks[1].ff.net[2], NVFP4DecoderLinear)


def test_loader_applies_exported_nvfp4_decoder(tmp_path):
    import fastvideo.envs as envs

    _write_vae_dir(tmp_path / "vae", decoder_layers=5)
    expected = _write_deploy_checkpoint(tmp_path / "d3.pt", student_layers=3, base_layers=5)
    dense = _load_h3_vae(tmp_path / "vae")
    assert len(dense.decoder.transformer_blocks) == 5 and dense.decode_autocast_dtype == torch.float16
    with envs.FASTVIDEO_H3_VAE_NVFP4_DECODER.override(str(tmp_path / "d3.pt")):
        vae = _load_h3_vae(tmp_path / "vae")
    assert len(vae.decoder.transformer_blocks) == 3
    assert vae.decode_autocast_dtype == torch.bfloat16
    layer = vae.decoder.transformer_blocks[1].ff.net[2]
    assert isinstance(layer, NVFP4DecoderLinear) and layer.act_scale == "unit" and not layer.frozen  # CPU
    state = vae.decoder.state_dict()
    for key, value in expected.items():
        assert torch.equal(state[key].to(value.dtype).cpu(), value), key
    assert not any(parameter.requires_grad for parameter in vae.parameters())


def test_loader_applies_nvfp4_decoder_from_config(tmp_path):
    import fastvideo.envs as envs

    _write_vae_dir(tmp_path / "vae", decoder_layers=5)
    _write_deploy_checkpoint(tmp_path / "d3.pt", student_layers=3, base_layers=5)
    _write_deploy_checkpoint(tmp_path / "d4.pt", student_layers=4, base_layers=5)
    vae = _load_h3_vae(tmp_path / "vae", decoder=str(tmp_path / "d3.pt"))
    assert len(vae.decoder.transformer_blocks) == 3
    # The serve-config field wins over the environment variable.
    with envs.FASTVIDEO_H3_VAE_NVFP4_DECODER.override(str(tmp_path / "d4.pt")):
        vae = _load_h3_vae(tmp_path / "vae", decoder=str(tmp_path / "d3.pt"))
    assert len(vae.decoder.transformer_blocks) == 3


@pytest.mark.parametrize("student_layers,base_layers,heads,match", [(6, 6, 2, "needs 6 decoder blocks"),
                                                                     (2, 2, 4, "does not fit")])
def test_loader_refuses_incompatible_nvfp4_decoder(tmp_path, student_layers, base_layers, heads, match):
    import fastvideo.envs as envs

    _write_vae_dir(tmp_path / "vae", decoder_layers=4)
    _write_deploy_checkpoint(tmp_path / "bad.pt", student_layers, base_layers, heads)
    with envs.FASTVIDEO_H3_VAE_NVFP4_DECODER.override(str(tmp_path / "bad.pt")), pytest.raises(ValueError,
                                                                                               match=match):
        _load_h3_vae(tmp_path / "vae")


def test_loader_rejects_non_deploy_checkpoint(tmp_path):
    import fastvideo.envs as envs

    _write_vae_dir(tmp_path / "vae", decoder_layers=2)
    torch.save({"decoder": {}, "step": 3}, tmp_path / "train.pt")
    with envs.FASTVIDEO_H3_VAE_NVFP4_DECODER.override(str(tmp_path / "train.pt")), pytest.raises(ValueError,
                                                                                                 match="format"):
        _load_h3_vae(tmp_path / "vae")
    with envs.FASTVIDEO_H3_VAE_NVFP4_DECODER.override(str(tmp_path / "missing.pt")), pytest.raises(
            FileNotFoundError):
        _load_h3_vae(tmp_path / "vae")


def test_keep_evenly_spaced_blocks_matches_the_f8_cut():
    from fastvideo.models.vaes.minimax_h3_nvfp4_decoder import keep_evenly_spaced_blocks

    decoder = _tiny_decoder(num_layers=26)
    blocks = list(decoder.transformer_blocks)
    assert keep_evenly_spaced_blocks(decoder, 8) == [0, 4, 7, 11, 14, 18, 21, 25]
    assert list(decoder.transformer_blocks) == [blocks[i] for i in (0, 4, 7, 11, 14, 18, 21, 25)]


@_needs_fp4
@pytest.mark.parametrize("rows,out_features,in_features", [(1797, 2048, 2048), (12 * 1797, 16384, 2048),
                                                             (12 * 1797, 2048, 8192)])
def test_decoder_gemm_backend_matches_cutlass(rows, out_features, in_features):
    import flashinfer

    from fastvideo.models.vaes.minimax_h3_nvfp4_decoder import _quantized_weight, mm_fp4_backend
    from fastvideo.models.vaes.minimax_h3_nvfp4_fused import quantize

    backend = mm_fp4_backend(torch.device("cuda"))
    if backend == "cutlass":
        pytest.skip("this GPU decodes with the CUTLASS backend")
    torch.manual_seed(0)
    sf = torch.ones((), device="cuda")
    x_fp4, x_scale = quantize(torch.randn(rows, in_features, device="cuda").bfloat16(), sf)
    packed, inv_scale, global_sf_w = _quantized_weight(
        (torch.randn(out_features, in_features, device="cuda") * 0.02).bfloat16())
    outs = []
    for name in ("cutlass", backend):
        out = torch.empty(rows, out_features, device="cuda", dtype=torch.bfloat16)
        flashinfer.mm_fp4(x_fp4, packed.T, x_scale, inv_scale.T, 1.0 / (sf * global_sf_w), torch.bfloat16, out,
                          block_size=16, use_8x4_sf_layout=False, backend=name)
        outs.append(out)
    assert torch.equal(outs[0], outs[1])


# Safetensors exports: bf16 masters read directly, and pre-packed NVFP4 weights.
def _deploy_to_safetensors(pt_path, out_path):
    from fastvideo.models.vaes.minimax_h3_nvfp4_checkpoint import write_nvfp4_decoder_safetensors
    from fastvideo.models.vaes.minimax_h3_nvfp4_decoder import load_nvfp4_decoder_checkpoint

    write_nvfp4_decoder_safetensors(str(out_path), load_nvfp4_decoder_checkpoint(str(pt_path)))


def test_loader_reads_bf16_safetensors_like_the_pt(tmp_path):
    import fastvideo.envs as envs

    _write_vae_dir(tmp_path / "vae", decoder_layers=5)
    _write_deploy_checkpoint(tmp_path / "d3.pt", student_layers=3, base_layers=5)
    _deploy_to_safetensors(tmp_path / "d3.pt", tmp_path / "d3.safetensors")
    states = []
    for name in ("d3.pt", "d3.safetensors"):
        with envs.FASTVIDEO_H3_VAE_NVFP4_DECODER.override(str(tmp_path / name)):
            vae = _load_h3_vae(tmp_path / "vae")
        assert len(vae.decoder.transformer_blocks) == 3 and vae.decode_autocast_dtype == torch.bfloat16
        states.append({**vae.decoder.state_dict(), **vae.post_quant_conv.state_dict()})
    assert states[0].keys() == states[1].keys()
    assert all(torch.equal(states[0][key], states[1][key]) for key in states[0])


def test_safetensors_reader_rejects_foreign_files(tmp_path):
    from safetensors.torch import save_file

    from fastvideo.models.vaes.minimax_h3_nvfp4_decoder import load_nvfp4_decoder_checkpoint

    save_file({"decoder.x": torch.zeros(2)}, str(tmp_path / "plain.safetensors"))
    with pytest.raises(ValueError, match="format"):
        load_nvfp4_decoder_checkpoint(str(tmp_path / "plain.safetensors"))
    save_file({"decoder.x": torch.zeros(2)}, str(tmp_path / "nometa.safetensors"),
              metadata={"format": "fastvideo_h3_decoder_nvfp4_packed_v1"})
    with pytest.raises(ValueError, match="nvfp4_metadata"):
        load_nvfp4_decoder_checkpoint(str(tmp_path / "nometa.safetensors"))


def _fake_packed_checkpoint(vae, act_scale="unit"):
    """A packed-format checkpoint with arbitrary (not GEMM-meaningful) codes, for CPU-side contract tests."""
    from fastvideo.models.vaes.minimax_h3_nvfp4_checkpoint import NVFP4_DECODER_PACKED_FORMAT

    names = nvfp4_decoder_linear_names(vae.decoder)
    state = {}
    generator = torch.Generator().manual_seed(3)
    for key, value in vae.decoder.state_dict().items():
        module, _, param = key.rpartition(".")
        if module in names and param == "weight":
            out_features, in_features = value.shape
            state[f"{module}.packed_weight"] = torch.randint(0, 256, (out_features, in_features // 2),
                                                             dtype=torch.uint8, generator=generator)
            state[f"{module}.weight_inv_scale"] = torch.randint(0, 256, (out_features * in_features // 16, ),
                                                                dtype=torch.uint8, generator=generator)
            state[f"{module}.weight_global_sf"] = torch.rand((), generator=generator)
            state[f"{module}.input_amax"] = torch.zeros(())
        elif module in names:
            state[key] = value.bfloat16()
        else:
            state[key] = torch.randn(value.shape, generator=generator).to(value.dtype)
    return {
        "format": NVFP4_DECODER_PACKED_FORMAT,
        "metadata": {
            "act_scale": act_scale,
            "rotation_group": None,
            "skip_blocks": [],
            "num_linears": len(names),
            "student_layers": len(vae.decoder.transformer_blocks),
        },
        "decoder": state,
        "post_quant_conv": vae.post_quant_conv.state_dict(),
    }


def test_loader_applies_packed_safetensors_without_master_weights(tmp_path):
    import fastvideo.envs as envs
    from fastvideo.models.vaes.minimax_h3_nvfp4_checkpoint import write_nvfp4_decoder_safetensors
    from fastvideo.models.vaes.minimax_h3_nvfp4_decoder import keep_evenly_spaced_blocks, nvfp4_linears

    _write_vae_dir(tmp_path / "vae", decoder_layers=5)
    student = _tiny_h3_vae(_tiny_vae_arch(5))
    keep_evenly_spaced_blocks(student.decoder, 3)
    checkpoint = _fake_packed_checkpoint(student)
    write_nvfp4_decoder_safetensors(str(tmp_path / "p3.safetensors"), checkpoint)
    with envs.FASTVIDEO_H3_VAE_NVFP4_DECODER.override(str(tmp_path / "p3.safetensors")):
        vae = _load_h3_vae(tmp_path / "vae")
    layers = nvfp4_linears(vae.decoder)
    assert len(vae.decoder.transformer_blocks) == 3 and len(layers) == checkpoint["metadata"]["num_linears"]
    assert all(layer.frozen and layer.weight is None and layer.act_scale == "unit" for layer in layers)
    assert vae.decode_autocast_dtype == torch.bfloat16
    named = dict(vae.decoder.named_modules())
    for key, value in checkpoint["decoder"].items():
        module, _, param = key.rpartition(".")
        actual = getattr(named[module], param)
        assert actual.dtype == value.dtype and torch.equal(actual.cpu(), value), key
    assert not any(parameter.requires_grad for parameter in vae.parameters())


def test_packed_loader_refuses_mismatched_architecture(tmp_path):
    from fastvideo.models.vaes.minimax_h3_nvfp4_decoder import apply_nvfp4_decoder_checkpoint

    checkpoint = _fake_packed_checkpoint(_tiny_h3_vae(_tiny_vae_arch(2, heads=4)))
    with pytest.raises(ValueError, match="does not fit"):
        apply_nvfp4_decoder_checkpoint(_tiny_h3_vae(_tiny_vae_arch(2, heads=2)), checkpoint, freeze=True)
    checkpoint = _fake_packed_checkpoint(_tiny_h3_vae(_tiny_vae_arch(2)))
    del checkpoint["decoder"]["transformer_blocks.0.ff.net.2.packed_weight"]
    with pytest.raises(ValueError, match="missing"):
        apply_nvfp4_decoder_checkpoint(_tiny_h3_vae(_tiny_vae_arch(2)), checkpoint, freeze=True)


@pytest.mark.parametrize("act_scale", ["unit", "dynamic", "static"])
def test_packed_loader_needs_input_amax_only_for_static_scales(act_scale):
    from fastvideo.models.vaes.minimax_h3_nvfp4_decoder import apply_nvfp4_decoder_checkpoint, nvfp4_linears

    checkpoint = _fake_packed_checkpoint(_tiny_h3_vae(_tiny_vae_arch(2)), act_scale=act_scale)
    for key in [key for key in checkpoint["decoder"] if key.endswith(".input_amax")]:
        del checkpoint["decoder"][key]
    vae = _tiny_h3_vae(_tiny_vae_arch(2))
    if act_scale == "static":
        with pytest.raises(ValueError, match="missing.*input_amax"):
            apply_nvfp4_decoder_checkpoint(vae, checkpoint, freeze=True)
        return
    apply_nvfp4_decoder_checkpoint(vae, checkpoint, freeze=True)
    layers = nvfp4_linears(vae.decoder)
    assert layers and all(layer.input_amax.item() == 0 for layer in layers)


def test_export_deploy_strips_base_layer_from_wrapped_training_keys(tmp_path, monkeypatch):
    import importlib.util
    import json
    import sys
    from pathlib import Path

    expected = _write_deploy_checkpoint(tmp_path / "reference.pt", student_layers=2, base_layers=2)
    reference = torch.load(tmp_path / "reference.pt", weights_only=False)
    names = nvfp4_decoder_linear_names(_tiny_h3_vae(_tiny_vae_arch(2)).decoder)
    wrapped = {}
    for key, value in expected.items():
        module, _, param = key.rpartition(".")
        # Store the training master in fp32 under a wrapper's base_layer, as a QAD run may.
        wrapped[f"{module}.base_layer.{param}" if module in names else key] = value.float()
    torch.save({"decoder": wrapped, "post_quant_conv": reference["post_quant_conv"], "step": 7}, tmp_path / "last.pt")
    (tmp_path / "metadata.json").write_text(json.dumps(reference["metadata"]))
    script = Path(__file__).parents[3] / "scripts/distill/minimax_h3_nvfp4_decoder/export_deploy.py"
    spec = importlib.util.spec_from_file_location("export_deploy", script)
    export_deploy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export_deploy)
    monkeypatch.setattr(sys, "argv", ["export_deploy", str(tmp_path / "last.pt"), str(tmp_path / "deploy.pt")])
    export_deploy.main()

    exported = torch.load(tmp_path / "deploy.pt", weights_only=False)["decoder"]
    assert exported.keys() == expected.keys()
    for key, value in expected.items():
        module, _, param = key.rpartition(".")
        dtype = torch.bfloat16 if module in names and param in ("weight", "bias") else torch.float32
        assert exported[key].dtype == dtype and torch.equal(exported[key], value.float().to(dtype)), key


class _DecoderHost(nn.Module):
    """The two VAE submodules an NVFP4 decoder export covers."""

    def __init__(self, decoder: nn.Module) -> None:
        super().__init__()
        self.decoder = decoder
        self.post_quant_conv = nn.Conv3d(24, 24, 1)


@_needs_fp4
@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("act_scale", ["unit", "static"])
def test_packed_export_round_trip_is_bit_identical(tmp_path, fused, act_scale):
    from fastvideo.models.vaes.minimax_h3_nvfp4_checkpoint import (packed_nvfp4_decoder_checkpoint,
                                                                   read_nvfp4_decoder_safetensors,
                                                                   write_nvfp4_decoder_safetensors)
    from fastvideo.models.vaes.minimax_h3_nvfp4_decoder import (apply_nvfp4_decoder_checkpoint,
                                                                freeze_nvfp4_linears, nvfp4_linears)

    reference = _fused_test_decoder(act_scale)
    with torch.no_grad():
        for layer in nvfp4_linears(reference):
            layer.input_amax.fill_(4.0)
    freeze_nvfp4_linears(reference)
    host = _DecoderHost(reference).cuda()
    metadata = {"act_scale": act_scale, "rotation_group": None, "skip_blocks": [], "student_layers": 2}
    write_nvfp4_decoder_safetensors(str(tmp_path / "p.safetensors"), packed_nvfp4_decoder_checkpoint(host, metadata))

    loaded = _DecoderHost(_fused_test_decoder(act_scale, convert=False)).cuda()
    with torch.no_grad():
        for parameter in loaded.parameters():
            parameter.zero_()  # every value must come from the file
    checkpoint = read_nvfp4_decoder_safetensors(str(tmp_path / "p.safetensors"), "cuda")
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    apply_nvfp4_decoder_checkpoint(loaded, checkpoint, freeze=True)
    # No master weight: the dense linears are only released; at most a few scalars are allocated,
    # far below one bf16 copy of the smallest quantized weight (512 x 512).
    assert torch.cuda.max_memory_allocated() - before < 512 * 512 * 2 // 16
    assert all(layer.frozen for layer in nvfp4_linears(loaded.decoder))
    if not fused:
        reference.fused_blocks_forward = None
        loaded.decoder.fused_blocks_forward = None
    latents = torch.randn(2, 24, 3, 8, 8, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        assert torch.equal(loaded.decoder(latents), reference(latents))
