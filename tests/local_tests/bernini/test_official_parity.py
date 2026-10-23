# SPDX-License-Identifier: Apache-2.0
import json
import re
import subprocess

import pytest
import torch
from safetensors.torch import load_file

from fastvideo.configs.models.dits.bernini import BerniniRConfig
from fastvideo.forward_context import set_forward_context
from fastvideo.models.dits.bernini import BerniniRTransformer3DModel, source_rotary_embedding
from fastvideo.models.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from fastvideo.pipelines.basic.bernini.stages import projected_guidance
from reference import REFERENCE, REFERENCE_SHA, WEIGHTS, official_modules, official_scheduler

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def oracle():
    if not REFERENCE.is_dir():
        pytest.skip("Clone official Bernini into reference_repos/bernini")
    actual = subprocess.check_output(["git", "-C", str(REFERENCE), "rev-parse", "HEAD"], text=True).strip()
    assert actual == REFERENCE_SHA, f"Reference must be pinned to {REFERENCE_SHA}, got {actual}"
    return official_modules()


def mapped_state_dict(state, config):
    mapped = {}
    for key, value in state.items():
        target = key
        for pattern, replacement in config.param_names_mapping.items():
            if re.match(pattern, key):
                target = re.sub(pattern, replacement, key)
                break
        assert target not in mapped, f"Mapping collision: {target}"
        mapped[target] = value
    return mapped


@pytest.mark.parametrize("grid,sid", [((2, 3, 4), 0), ((3, 2, 5), 1), ((1, 4, 6), 1)])
def test_rope_against_official(oracle, grid, sid):
    module, _ = oracle
    rope = module.WanRotaryPosEmbed(128, (1, 2, 2), 1024, use_src_id_rotary_emb=True)
    expected = rope(torch.empty(1, 16, grid[0], grid[1]*2, grid[2]*2), source_id=sid)[0, 0]
    cos, sin = source_rotary_embedding(grid, 128, sid)
    torch.testing.assert_close(cos[:, ::2].double(), expected.real, rtol=1e-7, atol=4e-8)
    torch.testing.assert_close(sin[:, ::2].double(), expected.imag, rtol=1e-7, atol=4e-8)


def test_apg_against_official(oracle):
    _, module = oracle
    torch.manual_seed(14)
    x = torch.randn(1, 16, 3, 4, 6)
    cond, uncond = torch.randn_like(x)*30, torch.randn_like(x)*30
    sigma = torch.tensor(0.73)
    expected_x = module.normalized_guidance(x-sigma*cond, x-sigma*uncond, 4.0,
                                           module.MomentumBuffer(0.0), eta=0.5, norm_threshold=50.0)
    torch.testing.assert_close(projected_guidance(cond, uncond, x, sigma, 4.0),
                               (x-expected_x)/sigma, rtol=0, atol=0)


@pytest.mark.parametrize("steps", [3, 10, 50])
def test_scheduler_against_official(steps):
    # Copied from the released scheduler/config.json, not from native config.
    config = dict(num_train_timesteps=1000, solver_order=2, prediction_type="flow_prediction",
                  use_flow_sigmas=True, flow_shift=3.0, timestep_spacing="linspace")
    try:
        scheduler_cls = official_scheduler()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    reference, native = scheduler_cls(**config), UniPCMultistepScheduler(**config)
    reference.set_timesteps(steps)
    native.set_timesteps(steps)
    torch.testing.assert_close(native.timesteps, reference.timesteps, rtol=0, atol=0)
    torch.testing.assert_close(native.sigmas, reference.sigmas, rtol=0, atol=0)
    torch.manual_seed(12)
    a = torch.randn(1, 18, 64)
    b = a.clone()
    for t in reference.timesteps:
        pred = torch.randn_like(a)
        a = reference.step(pred, t, a, return_dict=False)[0]
        b = native.step(pred, t, b, return_dict=False)[0]
        torch.testing.assert_close(b, a, rtol=0, atol=0)


@pytest.mark.parametrize("real_weights", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_transformer_against_official(oracle, distributed_setup, real_weights, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    module, _ = oracle
    if real_weights:
        config_path = WEIGHTS / "transformer/config.json"
        if not config_path.is_file():
            pytest.skip("Download the pinned official checkpoint first")
        kwargs = json.loads(config_path.read_text())
        kwargs = {k:v for k,v in kwargs.items() if not k.startswith("_")}
    else:
        kwargs = dict(num_attention_heads=2, attention_head_dim=16, num_layers=2,
                      ffn_dim=64, text_dim=24, freq_dim=8, in_channels=16, out_channels=16,
                      cross_attn_norm=True)
    torch.manual_seed(13)
    ref = module.WanTransformer3DModel.from_config(kwargs, use_src_id_rotary_emb=True).eval()
    if real_weights:
        state = {}
        for path in sorted((WEIGHTS / "transformer").glob("*.safetensors")):
            state.update(load_file(path))
        ref.load_state_dict(state, strict=True)
    config = BerniniRConfig()
    config.update_model_arch(kwargs)
    native = BerniniRTransformer3DModel(config, kwargs).eval()
    native.load_state_dict(mapped_state_dict(ref.state_dict(), config), strict=True)
    # Preserve the reference loader's mixed dtypes without first rounding its
    # FP32 exceptions to BF16. Direct model.to(dtype) destroys that information.
    for name, parameter in ref.named_parameters():
        keep = any(part in ref._keep_in_fp32_modules for part in name.split("."))
        parameter.data = parameter.data.to(device="cuda", dtype=torch.float32 if keep else dtype)
    for name, parameter in native.named_parameters():
        parameter.data = parameter.data.to(device="cuda", dtype=native._get_parameter_dtype(name, dtype))
    torch.manual_seed(24)
    target = torch.randn(1, 16, 2, 4, 6, device="cuda", dtype=dtype)
    source = torch.randn(1, 16, 3, 6, 4, device="cuda", dtype=dtype)
    text = torch.randn(1, 9, config.text_dim, device="cuda", dtype=dtype)
    t = torch.tensor([713.0], device="cuda")
    activations = ({}, {})
    handles = []
    probes = [
        ("time", ref.condition_embedder.time_embedder, native.condition_embedder.time_embedder.mlp),
        ("modulation", ref.condition_embedder.time_proj, native.condition_embedder.time_modulation),
        ("text", ref.condition_embedder.text_embedder, native.condition_embedder.text_embedder),
        ("norm1", ref.blocks[0].norm1, native.blocks[0].norm1),
        ("q", ref.blocks[0].attn1.to_q, native.blocks[0].to_q),
        ("q_norm", ref.blocks[0].attn1.norm_q, native.blocks[0].norm_q),
        ("attention_out", ref.blocks[0].attn1.to_out[0], native.blocks[0].to_out),
    ]
    probes += [(f"block_{i+1}", ref.blocks[i], native.blocks[i])
               for i in sorted({0, min(4, len(ref.blocks)-1), min(9, len(ref.blocks)-1),
                                min(19, len(ref.blocks)-1), len(ref.blocks)-1})]
    for label, a, b in probes:
        for bucket, layer in zip(activations, (a, b)):
            def capture(module, inputs, output, bucket=bucket, label=label):
                value = output[0] if isinstance(output, tuple) else output
                bucket[label] = value.detach().float().clone()
            handles.append(layer.register_forward_hook(capture))
    with torch.no_grad():
        src, sr = ref.patch_vae_latent(source, source_id=1)
        dst, dr = ref.patch_vae_latent(target, source_id=0)
        tokens, rope = torch.cat([src, dst], dim=1), torch.cat([sr, dr], dim=2)
        expected_tokens = ref(tokens, t, text, rope, [tokens.shape[1]], [text.shape[1]]).sample[:, -dst.shape[1]:]
        # Oracle owns its unpatchification; native unpacker is tested separately.
        expected = official_modules()[1]._to_spatial(expected_tokens, tuple(target.shape))
        with set_forward_context(current_timestep=713, attn_metadata=None):
            actual = native(target, text, t, source_latents=source)
    for handle in handles:
        handle.remove()
    for label in activations[0]:
        a, b = activations[0][label], activations[1][label]
        error = ((a-b).norm()/a.norm().clamp_min(1e-8)).item()
        print(f"  {label}: rel_L2={error:.8f}, mismatches={(a!=b).sum().item()}/{a.numel()}")
        assert error < (1e-3 if dtype == torch.bfloat16 else 3e-5), label
    relative_l2 = ((actual.float()-expected.float()).norm()/expected.float().norm()).item()
    print(f"real_weights={real_weights}, dtype={dtype}, relative_L2={relative_l2:.8f}, "
          f"max_abs={(actual-expected).abs().max().item():.8f}")
    assert torch.isfinite(actual).all()
    if dtype == torch.float32:
        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-5)
    else:
        assert relative_l2 < 1e-3
