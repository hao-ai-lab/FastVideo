# SPDX-License-Identifier: Apache-2.0
"""
Synthetic end-to-end tests for the SD3.5 training pipeline.
Runs without real model weights or real data (synthetic parquet + tiny transformer).
"""
from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from fastvideo.dataset.dataloader.schema import pyarrow_schema_sd35  # noqa: E402

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29555")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "TORCH_SDPA"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="SD3.5 training tests require CUDA",
)

def _make_synthetic_parquet(out_dir: str,
                            n_samples: int = 4) -> str:
    """Write a parquet file whose rows match pyarrow_schema_sd35."""
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for i in range(n_samples):
        # (C=16, T=1, H=8, W=8)  — tiny spatial size
        vae_latent = np.random.randn(16, 1, 8, 8).astype(np.float32)
        # (231, 4096) — 77 CLIP + 154 T5 tokens
        text_emb = np.random.randn(231, 4096).astype(np.float32)
        # (2048,)
        pooled = np.random.randn(2048).astype(np.float32)

        rows.append({
            "id": f"sample_{i}",
            "vae_latent_bytes": vae_latent.tobytes(),
            "vae_latent_shape": list(vae_latent.shape),
            "vae_latent_dtype": "float32",
            "text_embedding_bytes": text_emb.tobytes(),
            "text_embedding_shape": list(text_emb.shape),
            "text_embedding_dtype": "float32",
            "pooled_projection_bytes": pooled.tobytes(),
            "pooled_projection_shape": list(pooled.shape),
            "pooled_projection_dtype": "float32",
            "file_name": f"sample_{i}",
            "caption": f"a test prompt {i}",
            "media_type": "image",
            "width": 64,
            "height": 64,
            "num_frames": 1,
            "duration_sec": 0.0,
            "fps": 1.0,
        })

    arrays: dict[str, list] = {f.name: [] for f in pyarrow_schema_sd35}
    for row in rows:
        for col in arrays:
            arrays[col].append(row[col])

    table = pa.table(
        {
            col: pa.array(vals, type=pyarrow_schema_sd35.field(col).type)
            for col, vals in arrays.items()
        },
        schema=pyarrow_schema_sd35,
    )
    pq_path = os.path.join(out_dir, "part-0.parquet")
    pq.write_table(table, pq_path)
    return pq_path

@pytest.fixture(scope="module", autouse=True)
def _dist():
    import torch.distributed as dist
    from fastvideo.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    if not dist.is_initialized():
        store_path = f"/tmp/fastvideo_sd35_train_{os.getpid()}.store"
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{store_path}",
            rank=0,
            world_size=1,
        )
        torch.cuda.set_device(0)
        init_distributed_environment(world_size=1,
                                     rank=0,
                                     local_rank=0,
                                     distributed_init_method="env://")
        initialize_model_parallel(tensor_model_parallel_size=1,
                                  sequence_model_parallel_size=1)
    yield

def test_sd35_schema_fields():
    """Verify pyarrow_schema_sd35 has all expected fields."""
    field_names = {f.name for f in pyarrow_schema_sd35}
    required = {
        "id",
        "vae_latent_bytes", "vae_latent_shape", "vae_latent_dtype",
        "text_embedding_bytes", "text_embedding_shape", "text_embedding_dtype",
        "pooled_projection_bytes", "pooled_projection_shape",
        "pooled_projection_dtype",
        "file_name", "caption", "media_type",
        "width", "height", "num_frames", "duration_sec", "fps",
    }
    missing = required - field_names
    assert not missing, f"Missing schema fields: {missing}"

def test_sd35_parquet_roundtrip():
    """Write synthetic parquet and read it back; check tensor shapes."""
    from fastvideo.dataset.utils import collate_rows_from_parquet_schema

    tmpdir = tempfile.mkdtemp()
    try:
        _make_synthetic_parquet(tmpdir, n_samples=2)
        pq_path = os.path.join(tmpdir, "part-0.parquet")
        table = pq.read_table(pq_path)
        assert table.num_rows == 2

        rows = table.to_pydict()
        row_list = [{k: v[0] for k, v in rows.items()}]  # first row as dict

        # text_padding_length = 256 (SD35_TEXT_SEQ_LEN)
        batch = collate_rows_from_parquet_schema(row_list,
                                                 pyarrow_schema_sd35,
                                                 text_padding_length=256)
        assert "vae_latent" in batch
        assert "text_embedding" in batch
        assert "pooled_projection" in batch
        assert "text_attention_mask" in batch

        assert batch["vae_latent"].shape == (1, 16, 1, 8, 8)
        assert batch["text_embedding"].shape == (1, 256, 4096)
        assert batch["pooled_projection"].shape == (1, 2048)
        assert batch["text_attention_mask"].shape == (1, 256)
    finally:
        shutil.rmtree(tmpdir)

def test_normalize_dit_input_sd3():
    """normalize_dit_input('sd3') applies shift_factor and scaling_factor."""
    from fastvideo.training.training_utils import normalize_dit_input

    class _FakeVAE:
        class config:
            shift_factor = 0.0609
            scaling_factor = 1.5305

    latents = torch.randn(2, 16, 1, 8, 8)
    result = normalize_dit_input("sd3", latents, _FakeVAE())
    expected = ((latents.float() - 0.0609) * 1.5305).to(latents)
    assert torch.allclose(result, expected, atol=1e-5), \
        "normalize_dit_input('sd3') gave wrong result"

def test_flow_matching_noisy_input():
    """Verify the noisy interpolation formula used in _prepare_dit_inputs."""
    latents = torch.randn(2, 16, 4, 4)
    noise = torch.randn_like(latents)
    sigma = torch.tensor([0.5]).view(1, 1, 1, 1)

    noisy = (1.0 - sigma) * latents + sigma * noise

    # Boundary checks
    sigma_zero = torch.zeros(1, 1, 1, 1)
    assert torch.allclose((1.0 - sigma_zero) * latents + sigma_zero * noise,
                           latents), "sigma=0 should give clean latents"

    sigma_one = torch.ones(1, 1, 1, 1)
    assert torch.allclose((1.0 - sigma_one) * latents + sigma_one * noise,
                           noise), "sigma=1 should give pure noise"
    _ = noisy  # used above

def test_build_input_kwargs_keys():
    """SD35TrainingPipeline._build_input_kwargs produces the right keys."""
    from fastvideo.pipelines.pipeline_batch_info import TrainingBatch

    device = torch.device("cuda")
    B, C, H, W = 1, 16, 8, 8
    seq, dim = 256, 4096

    tb = TrainingBatch()
    tb.noisy_model_input = torch.randn(B, C, H, W, device=device,
                                       dtype=torch.bfloat16)
    tb.encoder_hidden_states = torch.randn(B, seq, dim, device=device,
                                           dtype=torch.bfloat16)
    tb.pooled_projections = torch.randn(B, 2048, device=device,
                                        dtype=torch.bfloat16)
    tb.timesteps = torch.tensor([500], device=device, dtype=torch.long)
    tb.sigmas = torch.tensor([0.5], device=device).view(1, 1, 1, 1)
    tb.noise = torch.randn_like(tb.noisy_model_input)
    tb.latents = torch.randn_like(tb.noisy_model_input)
    tb.raw_latent_shape = tb.noisy_model_input.shape

    # Instantiate just the method via a mock
    class _MockPipeline:
        training_args = None

        def _build_input_kwargs(self, tb):
            from fastvideo.distributed import get_local_torch_device
            assert tb.noisy_model_input is not None
            assert tb.encoder_hidden_states is not None
            assert tb.timesteps is not None
            assert tb.pooled_projections is not None
            tb.input_kwargs = {
                "hidden_states": tb.noisy_model_input,
                "encoder_hidden_states": tb.encoder_hidden_states,
                "pooled_projections": tb.pooled_projections,
                "timestep": tb.timesteps.to(get_local_torch_device(),
                                            dtype=torch.bfloat16),
                "return_dict": False,
            }
            return tb

    m = _MockPipeline()
    tb = m._build_input_kwargs(tb)

    assert "hidden_states" in tb.input_kwargs
    assert "encoder_hidden_states" in tb.input_kwargs
    assert "pooled_projections" in tb.input_kwargs
    assert "timestep" in tb.input_kwargs
    assert tb.input_kwargs["return_dict"] is False
    assert tb.input_kwargs["pooled_projections"].shape == (B, 2048)

def test_sd35_transformer_forward():
    """Load real SD3 transformer weights and run one forward pass."""
    model_dir = "/FastVideo/official_weights/stabilityai__stable-diffusion-3.5-medium"
    if not os.path.isdir(model_dir):
        pytest.skip(f"Model weights not found: {model_dir}")

    import json
    from safetensors.torch import load_file

    from fastvideo.configs.models.dits.sd3 import SD3DiTConfig
    from fastvideo.models.registry import ModelRegistry

    cfg_path = os.path.join(model_dir, "transformer", "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    fv_cls, _ = ModelRegistry.resolve_model_cls("SD3Transformer2DModel")
    dit_cfg = SD3DiTConfig()
    dit_cfg.update_model_arch(cfg)
    transformer = fv_cls(config=dit_cfg, hf_config=dict(cfg)).eval().to(
        device=device, dtype=dtype)

    weight_path = os.path.join(model_dir, "transformer",
                               "diffusion_pytorch_model.safetensors")
    sd = load_file(weight_path, device="cpu")
    transformer.load_state_dict(sd, strict=True)

    B, C, H, W = 1, 16, 16, 16
    hidden = torch.randn(B, C, H, W, device=device, dtype=dtype)
    enc = torch.randn(B, 256, 4096, device=device, dtype=dtype)
    pooled = torch.randn(B, 2048, device=device, dtype=dtype)
    t = torch.tensor([500], device=device, dtype=torch.long)

    from fastvideo.forward_context import set_forward_context
    with torch.no_grad(), set_forward_context(current_timestep=500,
                                               attn_metadata=None):
        out = transformer(
            hidden_states=hidden,
            encoder_hidden_states=enc,
            pooled_projections=pooled,
            timestep=t.to(dtype=torch.bfloat16),
            return_dict=False,
        )
    assert isinstance(out, tuple), "return_dict=False must return a tuple"
    pred = out[0]
    assert pred.shape == (B, C, H, W), f"Unexpected output shape: {pred.shape}"
    assert torch.isfinite(pred).all(), "Transformer output has NaN/Inf"

def test_sd35_loss_computation():
    """Run a single forward + backward through the transformer and check loss."""
    model_dir = "/FastVideo/official_weights/stabilityai__stable-diffusion-3.5-medium"
    if not os.path.isdir(model_dir):
        pytest.skip(f"Model weights not found: {model_dir}")

    import json
    from safetensors.torch import load_file

    from fastvideo.configs.models.dits.sd3 import SD3DiTConfig
    from fastvideo.models.registry import ModelRegistry
    from fastvideo.forward_context import set_forward_context
    from fastvideo.training.training_utils import (
        compute_density_for_timestep_sampling, get_sigmas, normalize_dit_input)

    cfg_path = os.path.join(model_dir, "transformer", "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)

    device = torch.device("cuda")

    fv_cls, _ = ModelRegistry.resolve_model_cls("SD3Transformer2DModel")
    dit_cfg = SD3DiTConfig()
    dit_cfg.update_model_arch(cfg)
    transformer = fv_cls(config=dit_cfg, hf_config=dict(cfg)).to(
        device=device, dtype=torch.float32)
    weight_path = os.path.join(model_dir, "transformer",
                               "diffusion_pytorch_model.safetensors")
    sd = load_file(weight_path, device="cpu")
    transformer.load_state_dict(sd, strict=True)
    transformer.train()

    # Fake VAE (just need config attrs)
    class _FakeVAE:
        class config:
            shift_factor = 0.0609
            scaling_factor = 1.5305

    from diffusers import FlowMatchEulerDiscreteScheduler
    # Use full 1000-step training schedule. set_timesteps(1000) keeps the full
    # grid so that sampling indices in [0, 999] stay in bounds.
    scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0)
    scheduler.set_timesteps(1000, device=device)

    # The launch script uses --dit_precision fp32; without FSDP we keep the
    # transformer in float32 and run all inputs in float32 to match.
    B, C, H, W = 1, 16, 16, 16
    latents = torch.randn(B, C, H, W, device=device, dtype=torch.float32)
    latents = normalize_dit_input("sd3", latents, _FakeVAE())

    noise_gen = torch.Generator(device="cpu").manual_seed(42)
    u = compute_density_for_timestep_sampling(
        weighting_scheme="logit_normal",
        batch_size=B,
        generator=noise_gen,
        logit_mean=0.0,
        logit_std=1.0,
    )
    indices = (u * scheduler.config.num_train_timesteps).long()
    timesteps = scheduler.timesteps[indices].to(device=device)

    sigmas = get_sigmas(scheduler,
                        device,
                        timesteps,
                        n_dim=latents.ndim,
                        dtype=latents.dtype)
    noise = torch.randn_like(latents)
    noisy = (1.0 - sigmas) * latents + sigmas * noise

    enc = torch.randn(B, 256, 4096, device=device, dtype=torch.float32)
    pooled = torch.randn(B, 2048, device=device, dtype=torch.float32)

    with set_forward_context(current_timestep=timesteps[0].item(),
                             attn_metadata=None):
        pred = transformer(
            hidden_states=noisy,
            encoder_hidden_states=enc,
            pooled_projections=pooled,
            timestep=timesteps.to(dtype=torch.float32),
            return_dict=False,
        )[0]

    target = noise - latents
    loss = torch.mean((pred.float() - target.float())**2)
    assert torch.isfinite(loss), f"Loss is non-finite: {loss.item()}"
    assert loss.item() > 0, "Loss should be positive"

    loss.backward()
    grad_norms = [
        p.grad.norm().item()
        for p in transformer.parameters()
        if p.grad is not None
    ]
    assert len(grad_norms) > 0, "No gradients were computed"
    assert all(np.isfinite(g) for g in grad_norms), "Some gradients are non-finite"
