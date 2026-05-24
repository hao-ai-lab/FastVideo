# SPDX-License-Identifier: Apache-2.0
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportMissingTypeArgument=false
"""Flux2 component parity tests.

Compares FastVideo's Flux2 components (DiT, VAE, Qwen3 text encoder)
against Diffusers/transformers references using the published
``black-forest-labs/FLUX.2-klein-4B`` weights.

All tests are skip-marked (CUDA + weight directory required).
Run locally with:

    FLUX2_MODEL_DIR=/path/to/weights pytest tests/local_tests/flux2/ -v -s
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import gc

import pytest
import torch
import torch.distributed as dist

from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import safe_open
from torch.testing import assert_close

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Flux2 component parity tests require CUDA",
    ),
    pytest.mark.filterwarnings(
        "ignore:.*torch.jit.script_method.*:DeprecationWarning",
    ),
]

os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")

MODEL_DIR = Path(
    os.getenv(
        "FLUX2_MODEL_DIR",
        "/FastVideo/official_weights/black-forest-labs__FLUX.2-klein-4B",
    )
)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pick_device_and_dtype() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.device("cuda"), torch.bfloat16
        return torch.device("cuda"), torch.float32
    return torch.device("cpu"), torch.float32


def _print_assert_close_means(
    label: str,
    expected: torch.Tensor,
    actual: torch.Tensor,
) -> None:
    expected_f32 = expected.detach().float()
    actual_f32 = actual.detach().float()
    print(
        f"[{label}] assert_close means "
        f"expected_mean={expected_f32.mean().item():.6f} "
        f"actual_mean={actual_f32.mean().item():.6f} "
        f"expected_abs_mean={expected_f32.abs().mean().item():.6f} "
        f"actual_abs_mean={actual_f32.abs().mean().item():.6f}"
    )


def _iter_safetensors(path: str):
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            yield k, f.get_tensor(k)


def _iter_pretrained_safetensors(model_dir: Path):
    candidates = [
        ("model.safetensors", "model.safetensors.index.json"),
        ("diffusion_pytorch_model.safetensors",
         "diffusion_pytorch_model.safetensors.index.json"),
    ]
    for single_name, index_name in candidates:
        single = model_dir / single_name
        if single.exists():
            yield from _iter_safetensors(str(single))
            return
        index = model_dir / index_name
        if index.exists():
            idx = _load_json(index)
            shard_names = sorted(set(idx["weight_map"].values()))
            for shard in shard_names:
                yield from _iter_safetensors(str(model_dir / shard))
            return

    raise FileNotFoundError(
        f"Missing safetensors checkpoint in {model_dir} "
        "(expected model.safetensors or diffusion_pytorch_model.safetensors)"
    )


# -----------------------------------------------------------------
# dist / TP fixture (single-GPU stub)
# -----------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _init_dist_and_tp_groups():
    if not torch.cuda.is_available():
        yield
        return

    import fastvideo.distributed.parallel_state as ps
    from fastvideo.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )

    created_dist = False
    created_tp = False

    try:
        _ = get_tp_group()
        yield
        return
    except Exception:
        pass

    stubbed = False
    old_tp = old_sp = old_dp = old_world = None

    try:
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "29500")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            os.environ.setdefault("LOCAL_RANK", "0")
            os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")

            if torch.cuda.is_available():
                torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

            backend = "nccl" if torch.cuda.is_available() else "gloo"
            store_path = f"/tmp/fastvideo_flux2_pg_{os.getpid()}.store"
            dist.init_process_group(
                backend=backend,
                init_method=f"file://{store_path}",
                rank=int(os.environ["RANK"]),
                world_size=int(os.environ["WORLD_SIZE"]),
            )
            created_dist = True

            init_distributed_environment(
                world_size=int(os.environ["WORLD_SIZE"]),
                rank=int(os.environ["RANK"]),
                local_rank=int(os.environ["LOCAL_RANK"]),
                distributed_init_method="env://",
            )
        else:
            init_distributed_environment(
                world_size=dist.get_world_size(),
                rank=dist.get_rank(),
                local_rank=int(os.environ.get("LOCAL_RANK", "0")),
                distributed_init_method="env://",
            )

        try:
            _ = get_tp_group()
        except Exception:
            initialize_model_parallel(
                tensor_model_parallel_size=1,
                sequence_model_parallel_size=1,
                data_parallel_size=(
                    dist.get_world_size() if dist.is_initialized() else 1
                ),
            )
            created_tp = True
    except Exception:
        old_tp = getattr(ps, "_TP", None)
        old_sp = getattr(ps, "_SP", None)
        old_dp = getattr(ps, "_DP", None)
        old_world = getattr(ps, "_WORLD", None)

        class _NoOpGroup:
            world_size = 1
            rank_in_group = 0
            local_rank = 0
            device_group = None

            def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
                return x

            def all_gather(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
                return x

            def all_to_all_4D(self, x: torch.Tensor, *_a, **_kw) -> torch.Tensor:
                return x

            def barrier(self) -> None:
                return None

            def destroy(self) -> None:
                return None

        ps._WORLD = _NoOpGroup()
        ps._TP = _NoOpGroup()
        ps._SP = _NoOpGroup()
        ps._DP = _NoOpGroup()
        stubbed = True

    yield

    if stubbed:
        ps._TP = old_tp
        ps._SP = old_sp
        ps._DP = old_dp
        ps._WORLD = old_world
    else:
        if created_tp:
            destroy_model_parallel()
        if created_dist:
            destroy_distributed_environment()


# -----------------------------------------------------------------
# DiT transformer parity
# -----------------------------------------------------------------

def test_flux2_transformer_parity():
    """Numerical forward parity: Diffusers Flux2Transformer2DModel vs FastVideo Flux2.

    The two implementations expose slightly different public forward surfaces.
    This test adapts both to the same denoising-step inputs: image tokens, text
    tokens, timestep, and explicit Flux2 text/image RoPE ids for Diffusers.
    Klein does not use guidance; Diffusers' pooled projection input is provided
    as zeros because FastVideo's native Flux2 embedding intentionally ignores
    pooled text projections for this distilled path.
    """
    transformer_dir = MODEL_DIR / "transformer"
    if not transformer_dir.exists():
        pytest.skip(f"Flux2 transformer dir not found: {transformer_dir}")

    from diffusers import Flux2Transformer2DModel as RefTransformer

    from fastvideo.configs.models.dits.flux_2 import Flux2Config
    from fastvideo.forward_context import set_forward_context
    from fastvideo.models.registry import ModelRegistry

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    device, dtype = _pick_device_and_dtype()
    torch.manual_seed(0)

    cfg = _load_json(transformer_dir / "config.json")
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)

    fv_cls, _ = ModelRegistry.resolve_model_cls("Flux2Transformer2DModel")
    dit_cfg = Flux2Config()
    dit_cfg.update_model_arch(cfg)

    in_channels = dit_cfg.in_channels
    joint_dim = dit_cfg.joint_attention_dim
    B, img_h, img_w, txt_len = 1, 8, 8, 16
    seq_len = img_h * img_w
    hidden_cpu = torch.randn(B, seq_len, in_channels, dtype=torch.float32)
    enc_cpu = torch.randn(B, txt_len, joint_dim, dtype=torch.float32)
    timestep_cpu = torch.tensor([0.5], dtype=torch.float32)
    txt_ids_cpu = torch.cartesian_prod(
        torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(txt_len),
    )
    img_ids_cpu = torch.cartesian_prod(
        torch.arange(1), torch.arange(img_h), torch.arange(img_w), torch.arange(1),
    )

    ref = RefTransformer.from_pretrained(
        str(transformer_dir),
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    ).eval().to(device)
    ref.time_guidance_embed.guidance_embedder = None
    with torch.no_grad():
        ref_out = ref(
            hidden_states=hidden_cpu.to(device=device, dtype=dtype),
            encoder_hidden_states=enc_cpu.to(device=device, dtype=dtype),
            timestep=timestep_cpu.to(device=device, dtype=dtype),
            img_ids=img_ids_cpu.to(device=device),
            txt_ids=txt_ids_cpu.to(device=device),
            guidance=torch.zeros_like(timestep_cpu).to(device=device, dtype=dtype),
            return_dict=False,
        )[0].detach().float().cpu()

    del ref
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    fv = fv_cls(config=dit_cfg, hf_config=dict(cfg)).eval()
    fv_sd = {}
    for k, v in _iter_pretrained_safetensors(transformer_dir):
        fv_sd[k] = v
    fv.load_state_dict(fv_sd, strict=True)
    fv = fv.to(device=device, dtype=dtype)

    with torch.no_grad():
        with set_forward_context(current_timestep=0, attn_metadata=None):
            fv_out = fv(
                hidden_states=hidden_cpu.to(device=device, dtype=dtype),
                encoder_hidden_states=enc_cpu.to(device=device, dtype=dtype),
                timestep=timestep_cpu.to(device=device, dtype=dtype),
            ).detach().float().cpu()

    assert fv_out.shape == (B, seq_len, in_channels), (
        f"Expected output shape {(B, seq_len, in_channels)}, got {fv_out.shape}"
    )
    assert torch.isfinite(fv_out).all(), "FastVideo DiT output contains non-finite values"
    assert torch.isfinite(ref_out).all(), "Diffusers DiT output contains non-finite values"

    diff = (ref_out - fv_out).abs()
    print(
        f"[FLUX2 DIT] diff max={diff.max().item():.6f} "
        f"mean={diff.mean().item():.6f} median={diff.median().item():.6f}"
    )
    print(
        "[FLUX2 DIT] abs-mean drift "
        f"diffusers={ref_out.abs().mean().item():.6f} "
        f"fastvideo={fv_out.abs().mean().item():.6f}"
    )
    _print_assert_close_means("FLUX2 DIT", ref_out, fv_out)
    assert_close(ref_out, fv_out, atol=1e-5, rtol=1e-5)

    hidden_5d = hidden_cpu.reshape(B, img_h, img_w, in_channels).permute(
        0, 3, 1, 2
    ).unsqueeze(2).contiguous()
    with torch.no_grad():
        with set_forward_context(current_timestep=0, attn_metadata=None):
            fv_out_5d = fv(
                hidden_states=hidden_5d.to(device=device, dtype=dtype),
                encoder_hidden_states=enc_cpu.to(device=device, dtype=dtype),
                timestep=timestep_cpu.to(device=device, dtype=dtype),
            ).detach().float().cpu()
    fv_out_5d_seq = fv_out_5d.squeeze(2).permute(0, 2, 3, 1).reshape(
        B, seq_len, in_channels
    )
    diff_5d = (ref_out - fv_out_5d_seq).abs()
    print(
        f"[FLUX2 DIT 5D] diff max={diff_5d.max().item():.6f} "
        f"mean={diff_5d.mean().item():.6f} median={diff_5d.median().item():.6f}"
    )
    _print_assert_close_means("FLUX2 DIT 5D", ref_out, fv_out_5d_seq)
    assert_close(ref_out, fv_out_5d_seq, atol=1e-5, rtol=1e-5)

    del fv
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# -----------------------------------------------------------------
# VAE parity
# -----------------------------------------------------------------

def test_flux2_vae_encode_decode_parity():
    """Encode/decode parity: Diffusers AutoencoderKLFlux2 vs FastVideo Flux2 VAE."""
    vae_dir = MODEL_DIR / "vae"
    if not vae_dir.exists():
        pytest.skip(f"Flux2 VAE dir not found: {vae_dir}")

    from diffusers import AutoencoderKLFlux2 as RefVAE

    from fastvideo.configs.models.vaes.flux2vae import Flux2VAEConfig
    from fastvideo.models.registry import ModelRegistry

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float32
    torch.manual_seed(0)

    ref = RefVAE.from_pretrained(
        str(vae_dir), local_files_only=True, torch_dtype=dtype,
    ).eval().to(device)

    x = torch.randn(1, 3, 64, 64, device=device, dtype=dtype)

    with torch.no_grad():
        ref_latents = ref.encode(x).latent_dist.mean.detach().float().cpu()
        ref_dec = ref.decode(
            ref_latents.to(device=device, dtype=dtype)
        ).sample.detach().float().cpu()

    del ref
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    cfg = _load_json(vae_dir / "config.json")
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)

    fv_cls, _ = ModelRegistry.resolve_model_cls("AutoencoderKLFlux2")
    vae_cfg = Flux2VAEConfig()
    vae_cfg.update_model_arch(cfg)
    fv = fv_cls(vae_cfg).eval()

    weight_path = vae_dir / "diffusion_pytorch_model.safetensors"
    fv_sd = safetensors_load_file(str(weight_path), device="cpu")
    fv.load_state_dict(fv_sd, strict=True)
    fv = fv.to(device=device, dtype=dtype)

    with torch.no_grad():
        fv_latents = fv.encode(x).mean.detach().float().cpu()
        fv_dec_output = fv.decode(
            fv_latents.to(device=device, dtype=dtype)
        )
        fv_dec_sample = getattr(fv_dec_output, "sample", fv_dec_output)
        fv_dec = fv_dec_sample.detach().float().cpu()

    _print_assert_close_means("FLUX2 VAE encode", ref_latents, fv_latents)
    assert_close(ref_latents, fv_latents, atol=1e-4, rtol=1e-4)
    _print_assert_close_means("FLUX2 VAE decode", ref_dec, fv_dec)
    assert_close(ref_dec, fv_dec, atol=1e-4, rtol=1e-4)

    del fv
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# -----------------------------------------------------------------
# Qwen3 text encoder parity
# -----------------------------------------------------------------

def test_flux2_qwen3_text_encoder_parity():
    """Hidden-state parity for the Flux2 Qwen3 loader path.

    Flux2 uses the HuggingFace Qwen3 module through FastVideo's component
    loader, so this validates that passthrough path against a direct HF load.
    The native TP-aware Qwen3 class is intentionally not used by the Flux2
    pipeline until it can provide strict hidden-state parity.
    """
    text_encoder_dir = MODEL_DIR / "text_encoder"
    if not text_encoder_dir.exists():
        pytest.skip(f"Flux2 text_encoder dir not found: {text_encoder_dir}")

    from transformers import AutoTokenizer, AutoModelForCausalLM

    device, dtype = _pick_device_and_dtype()
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR / "tokenizer"), local_files_only=True,
    )
    prompt = "a photo of a cat"
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    toks = tokenizer(
        [formatted],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    input_ids = toks["input_ids"].to(device=device)
    attention_mask = toks["attention_mask"].to(device=device)

    ref = AutoModelForCausalLM.from_pretrained(
        str(text_encoder_dir),
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).eval().to(device)

    with torch.no_grad():
        ref_out = ref(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        ref_embeds = torch.stack(
            [ref_out.hidden_states[k] for k in (9, 18, 27)], dim=1
        )
        ref_embeds = ref_embeds.permute(0, 2, 1, 3).reshape(
            input_ids.shape[0],
            input_ids.shape[1],
            -1,
        ).detach().float().cpu()

    del ref
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    from fastvideo.configs.models.encoders.qwen3 import Qwen3TextConfig
    from fastvideo.models.encoders.qwen3 import Qwen3ForCausalLM

    cfg_raw = _load_json(text_encoder_dir / "config.json")
    for k in ("_name_or_path", "transformers_version", "model_type", "torch_dtype"):
        cfg_raw.pop(k, None)

    fv_cfg = Qwen3TextConfig()
    fv_cfg.update_model_arch(cfg_raw)
    fv = Qwen3ForCausalLM.from_pretrained_local(
        str(text_encoder_dir),
        fv_cfg,
        dtype=dtype,
        device=device,
    )
    assert fv.__class__.__module__.startswith("transformers"), (
        "Flux2 Qwen3 should load through the exact HuggingFace passthrough path"
    )

    with torch.no_grad():
        fv_out = fv(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        fv_embeds = torch.stack(
            [fv_out.hidden_states[k] for k in (9, 18, 27)], dim=1
        )
        fv_embeds = fv_embeds.permute(0, 2, 1, 3).reshape(
            input_ids.shape[0],
            input_ids.shape[1],
            -1,
        ).detach().float().cpu()

    diff = (ref_embeds - fv_embeds).abs()
    print(
        f"[FLUX2 QWEN3] diff max={diff.max().item():.6f} "
        f"mean={diff.mean().item():.6f} median={diff.median().item():.6f}"
    )
    _print_assert_close_means("FLUX2 QWEN3", ref_embeds, fv_embeds)
    assert_close(ref_embeds, fv_embeds, atol=1e-5, rtol=1e-5)

    del fv
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
