# SPDX-License-Identifier: Apache-2.0
"""Numeric parity test: FastVideo's Kandinsky6Transformer3DModel vs. the
diffusers reference (see tests/local_tests/kandinsky6/README.md).

The reference classes must be importable (from the installed `diffusers` if it
provides Kandinsky6, else from a `diffusers` checkout at
KANDINSKY6_DIFFUSERS_REPO_PATH), and the transformer weights of an official
Diffusers repo (default: kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers) must
exist on disk. Absent either, this skips -- a skip is not a verified pass (see
.agents/skills/add-model/shared/common_rules.md).

The reference's forward() uses a packed/ragged (sum_T, H, W, C) layout keyed
by cu_seqlens (batch size folded into the leading dim); FastVideo's port uses
an ordinary batched (B, T, H, W, C) tensor. This test's batch size is 1, so
the two are related by a plain squeeze/unsqueeze.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
from torch.testing import assert_close

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29516")
os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
os.environ.setdefault("DIFFUSERS_ATTN_BACKEND", "native")


_DEFAULT_WEIGHTS_DIR = "official_weights/kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers"


def _resolve_transformer_path() -> Path:
    root = Path(os.getenv("KANDINSKY6_DIFFUSERS_PATH", _DEFAULT_WEIGHTS_DIR))
    return Path(os.getenv("KANDINSKY6_TRANSFORMER_PATH", str(root / "transformer")))


def _import_diffusers_kandinsky6():
    """Try the installed `diffusers` first, then a diffusers checkout.

    KANDINSKY6_DIFFUSERS_REPO_PATH should point at a diffusers checkout that
    provides Kandinsky6 -- its src/ directory is prepended to sys.path so
    `import diffusers` resolves there instead of the installed copy.
    """
    try:
        from diffusers import Kandinsky6Transformer3DModel
        return Kandinsky6Transformer3DModel
    except ImportError:
        pass

    repo_path = os.getenv("KANDINSKY6_DIFFUSERS_REPO_PATH")
    if not repo_path:
        return None
    src_path = str(Path(repo_path) / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    # Drop any partially-imported `diffusers` from the earlier failed
    # attempt so the retry actually re-resolves against the new sys.path.
    sys.modules.pop("diffusers", None)
    try:
        from diffusers import Kandinsky6Transformer3DModel
        return Kandinsky6Transformer3DModel
    except ImportError:
        return None


def test_kandinsky6_transformer_parity():
    transformer_path = _resolve_transformer_path()
    if not transformer_path.exists():
        pytest.skip(f"Kandinsky6 transformer weights not found at {transformer_path}")

    if not torch.cuda.is_available():
        pytest.skip("Kandinsky6 transformer parity test requires CUDA for practical runtime.")

    DiffusersKandinsky6 = _import_diffusers_kandinsky6()
    if DiffusersKandinsky6 is None:
        pytest.skip("diffusers.Kandinsky6Transformer3DModel is not importable: the installed diffusers lacks it and "
                   "KANDINSKY6_DIFFUSERS_REPO_PATH is unset or doesn't resolve. See README.md.")

    try:
        from fastvideo.configs.models.dits import Kandinsky6VideoAudioConfig
        from fastvideo.configs.pipelines import PipelineConfig
        from fastvideo.fastvideo_args import FastVideoArgs
        from fastvideo.models.loader.component_loader import TransformerLoader
    except Exception as exc:
        pytest.skip(f"FastVideo imports unavailable for parity run: {exc}")

    torch.manual_seed(42)
    device = torch.device("cuda:0")
    precision = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    precision_str = "bf16" if precision == torch.bfloat16 else "fp16"

    reference_model = DiffusersKandinsky6.from_pretrained(transformer_path).to(device=device, dtype=precision)
    reference_model.eval()

    config = Kandinsky6VideoAudioConfig()
    args = FastVideoArgs(
        model_path=str(transformer_path),
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        use_fsdp_inference=False,
        pipeline_config=PipelineConfig(dit_config=config, dit_precision=precision_str),
    )
    args.device = device
    # The official transformer/config.json has no _class_name; the pipeline loader takes it from model_index.json
    # (composed_pipeline_base.load_modules), so hand the loader the same hint when loading the component directly.
    args._model_index_class_names = {"transformer": "Kandinsky6Transformer3DModel"}
    fastvideo_model = TransformerLoader().load(str(transformer_path), args).to(device=device, dtype=precision)
    fastvideo_model.eval()

    arch = reference_model.config
    in_visual_dim = arch.in_visual_dim
    visual_cond = bool(getattr(arch, "visual_cond", False))
    in_text_dim = arch.in_text_dim
    in_text_dim2 = arch.in_text_dim2
    in_audio_dim = arch.in_audio_dim
    patch_size = arch.patch_size

    batch_size = 1
    grid_t, grid_h, grid_w = 2, 4, 4
    latent_t = grid_t * patch_size[0]
    latent_h = grid_h * patch_size[1]
    latent_w = grid_w * patch_size[2]
    audio_len = 12
    text_len = 8

    base_video = torch.randn(batch_size, latent_t, latent_h, latent_w, in_visual_dim, device=device, dtype=precision)
    if visual_cond:
        cond = torch.zeros_like(base_video)
        mask = torch.zeros(batch_size, latent_t, latent_h, latent_w, 1, device=device, dtype=precision)
        video = torch.cat([base_video, cond, mask], dim=-1)
    else:
        video = base_video
    audio = torch.randn(batch_size, audio_len, in_audio_dim, device=device, dtype=precision)
    encoder_hidden_states = torch.randn(batch_size, text_len, in_text_dim, device=device, dtype=precision)
    pooled_projections = torch.randn(batch_size, in_text_dim2, device=device, dtype=precision)
    timestep = torch.tensor([500.0], device=device, dtype=precision)

    visual_rope_pos = [
        torch.arange(grid_t, device=device),
        torch.arange(grid_h, device=device),
        torch.arange(grid_w, device=device),
    ]
    text_rope_pos = torch.arange(text_len, device=device)

    sdpa_math_ctx = torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
    with torch.no_grad(), sdpa_math_ctx:
        fv_video_out, fv_audio_out = fastvideo_model(
            hidden_states=video,
            hidden_states_audio=audio,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            visual_rope_pos=visual_rope_pos,
            text_rope_pos=text_rope_pos,
            scale_factor=(1.0, 1.0, 1.0),
            sparse_params=None,
            return_dict=False,
        )

        # Reference forward: packed (sum_T, H, W, C) / (sum_A, D) layout with
        # batch_size folded into the leading dim -- batch_size=1 here, so
        # this is a pure squeeze of the batch axis, no reshuffling needed.
        ref_video_in = video.squeeze(0)
        ref_audio_in = audio.squeeze(0)
        ref_out = reference_model(
            x_video=ref_video_in,
            x_audio=ref_audio_in,
            text_embed=encoder_hidden_states.squeeze(0),
            pooled_text_embed=pooled_projections.squeeze(0),
            time=timestep,
            visual_rope=reference_model.visual_rope_embeddings((grid_t, grid_h, grid_w), visual_rope_pos,
                                                               (1.0, 1.0, 1.0)),
            audio_rope=reference_model.audio_rope_embeddings(torch.arange(audio_len, device=device)),
            text_rope=reference_model.video_text_rope_embeddings(text_rope_pos),
            sparse_params=None,
            return_dict=False,
        )
        ref_video_out, ref_audio_out = ref_out

    assert fv_video_out.shape == ref_video_out.unsqueeze(0).shape
    assert fv_audio_out.shape == ref_audio_out.unsqueeze(0).shape
    tol = 1e-4 if precision == torch.bfloat16 else 2e-4
    assert_close(fv_video_out.squeeze(0), ref_video_out, atol=tol, rtol=tol)
    assert_close(fv_audio_out.squeeze(0), ref_audio_out, atol=tol, rtol=tol)
