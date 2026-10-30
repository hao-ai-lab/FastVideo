# SPDX-License-Identifier: Apache-2.0
"""PAN2's fused Q/K RMSNorm + RoPE and LayerNorm + modulation ops (``fastvideo.models.dits.pan2_fusions``).

The PyTorch references must reproduce the eager arithmetic they replace, their backward formulas must be the true
gradients (fp64 gradcheck), the Triton kernels must reproduce the references, and a PAN2 model with fused kernels must
match the eager model forward and backward. On a GPU-less host the Triton kernels run under the Triton interpreter in
a subprocess, in FP32: the interpreter truncates FP32 to BF16 instead of rounding to nearest even, so the BF16 kernel
comparison needs a GPU.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import fastvideo.envs as envs
from fastvideo.configs.models.dits.pan2 import PAN2ArchConfig, PAN2VideoConfig
from fastvideo.forward_context import set_forward_context
from fastvideo.layers.layernorm import RMSNorm
from fastvideo.layers.rotary_embedding import _apply_rotary_emb, get_1d_rotary_pos_embed
from fastvideo.models.dits import pan2 as pan2_module
from fastvideo.models.dits import pan2_fusions
from fastvideo.models.dits.pan2_fusions import layer_norm_modulate, qk_rmsnorm_rope
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

REPO_ROOT = Path(__file__).resolve().parents[3]

LATENT_CHANNELS = 4
TINY_ARCH = dict(
    patch_size=(1, 2, 2),
    in_channels=2 * LATENT_CHANNELS + 1,
    out_channels=LATENT_CHANNELS,
    num_attention_heads=4,
    attention_head_dim=16,
    num_layers=2,
    num_refiner_layers=1,
    mlp_ratio=2.0,
    context_mlp_ratio=2.0,
    refiner_mlp_ratio=2.0,
    text_embed_dim=16,
    rope_axes_dim=(4, 6, 6),
)


@pytest.fixture()
def cpu_attention(env_overrides):
    env_overrides.enter_context(envs.FASTVIDEO_ATTENTION_BACKEND.override("TORCH_SDPA"))


@pytest.fixture()
def single_process_group(env_overrides):
    """A 1x1 sequence/tensor-parallel group, which DistributedAttention needs even on one CPU process."""
    from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel

    env_overrides.enter_context(envs.override_external("MASTER_ADDR", os.environ.get("MASTER_ADDR") or "127.0.0.1"))
    env_overrides.enter_context(envs.override_external("MASTER_PORT", os.environ.get("MASTER_PORT") or "29593"))
    env_overrides.enter_context(envs.override_external("RANK", "0"))
    env_overrides.enter_context(envs.override_external("WORLD_SIZE", "1"))
    env_overrides.enter_context(envs.override_external("LOCAL_RANK", "0"))
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    try:
        yield
    finally:
        cleanup_dist_env_and_memory()


def _model() -> pan2_module.PAN2Transformer3DModel:
    return pan2_module.PAN2Transformer3DModel(PAN2VideoConfig(arch_config=PAN2ArchConfig(**TINY_ARCH)), hf_config={})


def _inputs(seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return dict(
        hidden_states=torch.randn(1, TINY_ARCH["in_channels"], 3, 4, 6, generator=generator),
        encoder_hidden_states=torch.randn(1, 5, TINY_ARCH["text_embed_dim"], generator=generator),
        timestep=torch.tensor([600.0]),
    )


def _randomize(model: torch.nn.Module, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)



def _qkv(batch: int, seq: int, heads: int, dim: int, dtype: torch.dtype, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    qkv = torch.randn(batch, seq, 3 * heads * dim, generator=generator).to(dtype)
    return qkv.unflatten(-1, (3, heads, -1)).unbind(2)


def _rope_tables(seq: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(seq, dtype=torch.float32) * (16 / 24)
    return get_1d_rotary_pos_embed(dim, positions, theta=256.0, use_real=False)


def _eager_qk(x: torch.Tensor, weight: torch.Tensor, eps: float, rope: tuple[torch.Tensor, torch.Tensor] | None):
    norm = RMSNorm(x.shape[-1], eps=eps)
    with torch.no_grad():
        norm.weight.copy_(weight)
    y = norm.forward_native(x).to(x.dtype)
    if rope is None:
        return y
    rows = y.flatten(0, 1)
    cos, sin = (table.repeat(x.shape[0], 1) for table in rope)
    return _apply_rotary_emb(rows, cos, sin, is_neox_style=False).unflatten(0, x.shape[:2])


# ---- references against the eager arithmetic -----------------------------------------------------------------------


@pytest.mark.parametrize("with_rope", [False, True])
def test_qk_reference_is_the_eager_chain_bit_for_bit(with_rope: bool) -> None:
    q, k, _ = _qkv(2, 13, 3, 16, torch.bfloat16)
    weights = torch.rand(16) + 0.5, torch.rand(16) + 0.5
    rope = _rope_tables(13, 16) if with_rope else None
    fused_q, fused_k = qk_rmsnorm_rope(q, k, *weights, 1e-6, *(rope or (None, None)))
    assert torch.equal(fused_q, _eager_qk(q, weights[0], 1e-6, rope))
    assert torch.equal(fused_k, _eager_qk(k, weights[1], 1e-6, rope))
    assert fused_q.is_contiguous() and fused_q.dtype == torch.bfloat16


def test_layer_norm_modulate_reference_is_the_autocast_arithmetic() -> None:
    generator = torch.Generator().manual_seed(1)
    x = (torch.randn(2, 11, 48, generator=generator) * 4 + 2).to(torch.bfloat16)
    scale = (1 + torch.randn(2, 48, generator=generator)).to(torch.bfloat16)
    shift = torch.randn(48, generator=generator).to(torch.bfloat16)
    norm = torch.nn.LayerNorm(48, elementwise_affine=False, eps=1e-6)
    # Under autocast the eager LayerNorm and modulation run in FP32 and the next linear rounds the result once.
    expected = (norm(x.float()) * scale.float()[:, None] + shift.float()).to(torch.bfloat16)
    actual = layer_norm_modulate(x, scale, shift, 1e-6)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, atol=0, rtol=2**-7)


def test_fused_ops_have_true_gradients() -> None:
    q, k, _ = _qkv(2, 5, 2, 8, torch.float64, seed=2)
    q, k = q.detach().requires_grad_(), k.detach().requires_grad_()
    weights = [(torch.rand(8, dtype=torch.float64) + 0.5).requires_grad_() for _ in range(2)]
    cos, sin = (table.double() for table in _rope_tables(5, 8))
    assert torch.autograd.gradcheck(lambda *args: qk_rmsnorm_rope(*args, 1e-6, cos, sin), (q, k, *weights))
    assert torch.autograd.gradcheck(lambda *args: qk_rmsnorm_rope(*args, 1e-6), (q, k, *weights))

    x = torch.randn(2, 4, 12, dtype=torch.float64, requires_grad=True)
    scale = torch.randn(2, 12, dtype=torch.float64, requires_grad=True)
    shift = torch.randn(12, dtype=torch.float64, requires_grad=True)  # shared by the batch, as PAN2's text tables
    assert torch.autograd.gradcheck(lambda *args: layer_norm_modulate(*args, 1e-6), (x, scale, shift))


@pytest.mark.gpu
def test_fused_ops_compile_with_their_backwards_on_gpu() -> None:
    """Inside a fullgraph compiled region, as regional compile captures PAN2's blocks, both fused ops and their
    backwards are opaque ops: tracing never launches a Triton kernel, and the values and gradients are eager's."""
    if not torch.cuda.is_available() or not pan2_fusions.HAVE_TRITON:
        pytest.skip("needs CUDA and Triton")
    q, k, _ = (t.cuda() for t in _qkv(2, 21, 3, 16, torch.bfloat16, seed=6))
    weights = [(torch.rand(16) + 0.5).cuda() for _ in range(2)]
    cos, sin = (table.cuda() for table in _rope_tables(21, 16))
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(2, 9, 40, generator=generator).cuda().to(torch.bfloat16)
    scale = (1 + torch.randn(2, 40, generator=generator)).cuda()
    shift = torch.randn(40, generator=generator).cuda()

    def fn(q, k, q_weight, k_weight, x, scale, shift):
        q_out, k_out = qk_rmsnorm_rope(q, k, q_weight, k_weight, 1e-6, cos, sin)
        return (q_out.float().square().sum() + k_out.float().square().sum() +
                layer_norm_modulate(x, scale, shift, 1e-6).float().square().sum())

    def run(function):
        leaves = [t.detach().clone().requires_grad_() for t in (q, k, *weights, x, scale, shift)]
        loss = function(*leaves)
        loss.backward()
        return loss.detach(), [leaf.grad for leaf in leaves]

    eager_loss, eager_grads = run(fn)
    loss, grads = run(torch.compile(fn, fullgraph=True, dynamic=True))
    torch.testing.assert_close(loss, eager_loss, atol=0, rtol=1e-5)
    for actual, expected in zip(grads, eager_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


# ---- the Triton kernels against the references ----------------------------------------------------------------------


def check_triton_kernels(device: str, dtypes: tuple[torch.dtype, ...]) -> None:
    """Every Triton kernel against its reference, forward and backward; raises on a mismatch."""
    for dtype in dtypes:
        tol = dict(atol=1e-5, rtol=1e-5) if dtype == torch.float32 else dict(atol=0, rtol=2**-6)
        # 21 rows: a partial row block. q and k are strided views of one projection, as in the model.
        q, k, _ = (t.to(device) for t in _qkv(2, 21, 3, 16, dtype, seed=3))
        weight = (torch.rand(16) + 0.5).to(device)
        cos, sin = (table.to(device) for table in _rope_tables(21, 16))
        grad = torch.randn(2, 21, 3, 16, generator=torch.Generator().manual_seed(4)).to(device=device, dtype=dtype)
        for tables in ((cos, sin), (None, None)):
            expected, expected_rstd = pan2_fusions._qk_forward_reference(q, weight, *tables, 1e-6)
            actual, rstd = pan2_fusions._qk_forward_triton(q, weight, *tables, 1e-6)
            torch.testing.assert_close(actual.float(), expected.float(), **tol)
            torch.testing.assert_close(rstd, expected_rstd, atol=1e-6, rtol=1e-6)
            expected_grads = pan2_fusions._qk_backward_reference(grad, q, weight, *tables, expected_rstd)
            actual_grads = pan2_fusions._qk_backward_triton(grad, q, weight, *tables, rstd)
            for a, e in zip(actual_grads, expected_grads, strict=True):
                torch.testing.assert_close(a.float(), e.float(), atol=1e-4, rtol=1e-3 if dtype != torch.float32 else 1e-5)

        generator = torch.Generator().manual_seed(5)
        x = (torch.randn(2, 9, 40, generator=generator) * 3 + 1).to(device=device, dtype=dtype)
        scale = (1 + torch.randn(2, 40, generator=generator)).to(device=device, dtype=dtype)
        shift = torch.randn(40, generator=generator).to(device=device, dtype=dtype)
        grad = torch.randn(2, 9, 40, generator=generator).to(device=device, dtype=dtype)
        expected = pan2_fusions._ln_forward_reference(x, scale, shift, 1e-6)
        actual = pan2_fusions._ln_forward_triton(x, scale, shift, 1e-6)
        torch.testing.assert_close(actual[0].float(), expected[0].float(), **tol)
        torch.testing.assert_close(actual[1:], expected[1:], atol=1e-5, rtol=1e-5)
        expected_grads = pan2_fusions._ln_backward_reference(grad, x, scale, expected[1], expected[2])
        actual_grads = pan2_fusions._ln_backward_triton(grad, x, scale, actual[1], actual[2])
        for a, e in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(a.float(), e.float(), atol=1e-4, rtol=1e-3 if dtype != torch.float32 else 1e-5)


def test_triton_kernels_match_their_references(env_overrides) -> None:
    if not pan2_fusions.HAVE_TRITON:
        pytest.skip("Triton is not installed")
    if torch.cuda.is_available():
        check_triton_kernels("cuda", (torch.float32, torch.bfloat16))
        return
    code = (f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r}); "
            "import fastvideo.models.dits.pan2_fusions as fusions; fusions._FORCE_TRITON = True; "
            "import torch; from fastvideo.tests.transformers.test_pan2_fusions import check_triton_kernels; "
            "check_triton_kernels('cpu', (torch.float32, ))")
    # The child inherits this environment with the Triton interpreter switched on.
    env_overrides.enter_context(envs.override_external("TRITON_INTERPRET", "1"))
    process = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert process.returncode == 0, f"Triton interpreter check failed:\n{process.stdout}\n{process.stderr}"


# ---- the fused model against the eager model -----------------------------------------------------------------------


def _forward_backward(model: torch.nn.Module, inputs: dict[str, torch.Tensor]):
    model.zero_grad(set_to_none=True)
    with set_forward_context(current_timestep=0, attn_metadata=None, forward_batch=ForwardBatch(data_type="dummy")):
        output = model(**inputs)
    weight = torch.randn(output.shape, generator=torch.Generator().manual_seed(6), dtype=output.dtype)
    (output * weight).sum().backward()
    return output.detach(), {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}


def test_fused_model_matches_the_eager_model(cpu_attention, single_process_group) -> None:
    model = _model()
    _randomize(model, seed=7)
    inputs = _inputs(seed=8)
    eager_out, eager_grads = _forward_backward(model, inputs)
    model.set_fused_kernels(True)
    assert all(block.fused_kernels and block.attn.fused_kernels for block in model.transformer_blocks)
    fused_out, fused_grads = _forward_backward(model, inputs)
    torch.testing.assert_close(fused_out, eager_out, atol=1e-5, rtol=1e-5)
    assert fused_grads.keys() == eager_grads.keys()
    for name, grad in eager_grads.items():
        torch.testing.assert_close(fused_grads[name], grad, atol=1e-5, rtol=1e-4, msg=name)


@pytest.mark.gpu
def test_fused_model_matches_the_eager_model_in_bf16_on_gpu(cpu_attention, single_process_group) -> None:
    """The sampling numerics: BF16 activations under autocast, Triton kernels, on one GPU."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    model = _model().cuda()
    _randomize(model, seed=7)
    inputs = {name: value.cuda() for name, value in _inputs(seed=8).items()}

    def run():
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16), set_forward_context(
                current_timestep=0, attn_metadata=None, forward_batch=ForwardBatch(data_type="dummy")):
            output = model(**inputs)
        output.float().square().mean().backward()
        return output.float(), {name: p.grad.float() for name, p in model.named_parameters() if p.grad is not None}

    eager_out, eager_grads = run()
    model.set_fused_kernels(True)
    fused_out, fused_grads = run()
    scale = eager_out.abs().mean()
    assert float((fused_out - eager_out).abs().mean() / scale) < 1e-2
    for name, grad in eager_grads.items():
        error = float((fused_grads[name] - grad).norm() / grad.norm().clamp_min(1e-12))
        assert error < 5e-2, f"{name}: relative gradient error {error:.3e}"
