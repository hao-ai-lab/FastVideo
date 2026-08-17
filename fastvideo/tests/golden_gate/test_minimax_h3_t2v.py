# SPDX-License-Identifier: Apache-2.0
"""Golden-gate for the MiniMax H3 T2V DiT: single-layer bitwise fingerprint.

Loads ONE transformer block of the 62 GB checkpoint (the safetensors index
locates the ~1.2 GB of layer tensors; only the shards containing them are
fetched), feeds a fixed seeded packed batch exercising all three modalities,
and compares the output **bitwise** against a device-keyed golden latent.

Zero tolerance is deliberate, not optimistic. Verified on GB200 2026-08-05:
  * this block reproduces bit-identically across independent processes,
  * the full 50-layer forward holds ``atol=0`` against pinned Diffusers
    (tests/local_tests/transformers/test_minimax_h3_transformer_parity.py),
  * two end-to-end seeded T2V generations produced md5-identical videos.
Any drift here is therefore a real change to the compute path — the gate
stands in front of the ~13-minute full H3 SSIM generation at ~40 seconds.

The golden's environment fingerprint (torch/CUDA/cuDNN/flash-attn/TF32/
backend/FASTVIDEO_FA4) is asserted before the tensor compare, so a legitimate
environment change fails as "environment changed — re-mint deliberately",
never as unexplained bit drift.

Goldens are device-keyed like SSIM references and live in the same HF dataset
(``FastVideo/ssim-reference-videos`` under ``golden_gates/<device>/``); a
local ``goldens/`` directory or ``MINIMAX_H3_GATE_GOLDEN_DIR`` wins over the
download. A missing golden is seeded locally and the test fails with upload
instructions, mirroring the SSIM missing-reference convention.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
import torch
from torch.testing import assert_close

GOLDEN_REPO_ID = os.environ.get("FASTVIDEO_SSIM_REFERENCE_HF_REPO", "FastVideo/ssim-reference-videos")
MODEL_REPO_ID = "MiniMaxAI/MiniMax-H3"
LAYER = int(os.environ.get("MINIMAX_H3_GATE_LAYER", "0"))
SEED = 20260805

# Pin the same attention path the H3 SSIM test runs, before any fastvideo
# import can cache a backend decision. CUBLAS workspace must be set before
# cuBLAS initializes for use_deterministic_algorithms to hold.
os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29661")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

# Checkpoint -> FastVideo renames, the block-level subset of
# MiniMaxH3Config.param_names_mapping (fastvideo/configs/models/dits/minimax_h3.py).
_RENAMES = (
    (re.compile(r"\.ff\.net\.0\.proj\."), ".ff.fc_in."),
    (re.compile(r"\.ff\.net\.2\."), ".ff.fc_out."),
    (re.compile(r"\.attn\.to_out\.0\."), ".attn.to_out."),
)


@pytest.fixture(scope="module", autouse=True)
def _distributed_runtime():
    if not torch.cuda.is_available():
        yield
        return
    from fastvideo.distributed import (
        cleanup_dist_env_and_memory,
        maybe_init_distributed_environment_and_model_parallel,
    )
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    yield
    cleanup_dist_env_and_memory()


def _env_fingerprint() -> dict[str, str]:
    """Everything that legitimately changes numerics. A golden minted under a
    different fingerprint must fail as 'environment changed', never as
    unexplained bit drift."""
    try:
        import flash_attn
        fa_version = str(getattr(flash_attn, "__version__", "?"))
    except ImportError:
        fa_version = "<not installed>"
    return {
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "cudnn": str(torch.backends.cudnn.version()),
        "flash_attn": fa_version,
        "device_name": torch.cuda.get_device_name(0),
        "attention_backend": os.environ.get("FASTVIDEO_ATTENTION_BACKEND", "<unset>"),
        "fastvideo_fa4": os.environ.get("FASTVIDEO_FA4", "<unset>"),
        "tf32_matmul": str(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": str(torch.backends.cudnn.allow_tf32),
    }


def _component_dir() -> Path:
    """Local checkout wins; otherwise fetch only the index + the shards that
    hold this layer's tensors (~4.5 GB of the 62 GB checkpoint)."""
    root = os.environ.get("MINIMAX_H3_MODEL_ROOT")
    if root:
        component = Path(root) / "transformer"
        if not (component / "diffusion_pytorch_model.safetensors.index.json").is_file():
            pytest.fail(f"MINIMAX_H3_MODEL_ROOT set but index missing under {component}", pytrace=False)
        return component

    from huggingface_hub import hf_hub_download

    index_path = Path(hf_hub_download(MODEL_REPO_ID, "transformer/diffusion_pytorch_model.safetensors.index.json"))
    weight_map = json.loads(index_path.read_text())["weight_map"]
    prefix = f"transformer_blocks.{LAYER}."
    shards = sorted({shard for name, shard in weight_map.items() if name.startswith(prefix)})
    if not shards:
        pytest.fail(f"no shards found for prefix {prefix} in {MODEL_REPO_ID}", pytrace=False)
    for shard in shards:
        hf_hub_download(MODEL_REPO_ID, f"transformer/{shard}")
    return index_path.parent


def _load_layer_state(component_dir: Path, layer: int) -> dict[str, torch.Tensor]:
    """Load only ``transformer_blocks.{layer}.*`` tensors via the index."""
    from safetensors import safe_open

    index = json.loads((component_dir / "diffusion_pytorch_model.safetensors.index.json").read_text())
    prefix = f"transformer_blocks.{layer}."
    by_shard: dict[str, list[str]] = {}
    for name, shard in index["weight_map"].items():
        if name.startswith(prefix):
            by_shard.setdefault(shard, []).append(name)
    assert by_shard, f"no parameters found for prefix {prefix}"

    state: dict[str, torch.Tensor] = {}
    for shard, names in sorted(by_shard.items()):
        with safe_open(component_dir / shard, framework="pt", device="cpu") as f:
            for name in names:
                mapped = f".{name[len(prefix):]}"
                for pattern, repl in _RENAMES:
                    mapped = pattern.sub(repl, mapped)
                state[mapped[1:]] = f.get_tensor(name)
    return state


def _build_block(layer: int) -> torch.nn.Module:
    from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3ArchConfig
    from fastvideo.models.dits.minimax_h3 import MiniMaxH3TransformerBlock

    arch = MiniMaxH3ArchConfig()
    return MiniMaxH3TransformerBlock(
        arch.hidden_size,
        arch.num_attention_heads,
        arch.attention_head_dim,
        arch.ffn_dim,
        arch.time_embed_dim,
        arch.norm_eps,
        arch.qk_norm_eps,
        arch._supported_attention_backends,
        None,
        prefix=f"minimax_h3.transformer_blocks.{layer}",
    )


def _make_inputs(device: torch.device) -> dict:
    """Fixed packed batch: 32 text + 24 audio + 200 video rows, 3 timesteps."""
    from fastvideo.models.dits.minimax_h3 import MiniMaxH3RotaryPosEmbed

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    n_text, n_audio, n_video = 32, 24, 200
    seq = n_text + n_audio + n_video

    token_tags = torch.cat([
        torch.full((n_text, ), 1, dtype=torch.long),
        torch.full((n_audio, ), 2, dtype=torch.long),
        torch.full((n_video, ), 0, dtype=torch.long),
    ])
    timestep_indices = torch.zeros(seq, dtype=torch.long)
    timestep_indices[n_text:n_text + n_audio] = 1
    timestep_indices[n_text + n_audio:n_text + n_audio + 100] = 2
    adaln_indices = timestep_indices * 3 + token_tags  # MINIMAX_H3_MODALITY_NUM = 3

    position_ids = torch.zeros(seq, 3, dtype=torch.float64)
    position_ids[:, 0] = torch.arange(seq, dtype=torch.float64)
    vid = torch.arange(n_video, dtype=torch.float64)
    position_ids[n_text + n_audio:, 1] = vid % 10
    position_ids[n_text + n_audio:, 2] = vid // 10

    rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=16, rope_theta=10000.0)
    cos, sin = rope(position_ids.to(torch.float32))

    hidden = torch.randn(1, seq, 5376, generator=generator, dtype=torch.float32)
    temb = torch.randn(9, 2688, generator=generator, dtype=torch.float32)  # 3 timesteps x 3 modalities

    return {
        "hidden_states": hidden.to(device=device, dtype=torch.bfloat16),
        "temb": temb.to(device=device, dtype=torch.bfloat16),
        "adaln_indices": adaln_indices.to(device),
        "rotary_emb": (cos.to(device), sin.to(device)),
        "original_seq_len": seq,
    }


def _device_slug() -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", torch.cuda.get_device_name(0)).strip("_")


def _golden_filename() -> str:
    backend = os.environ.get("FASTVIDEO_ATTENTION_BACKEND", "default")
    return f"minimax_h3_t2v_layer{LAYER}_{backend}_seed{SEED}.pt"


def _resolve_golden() -> tuple[Path, bool]:
    """Return (path, exists). Local dir wins; falls back to the HF dataset."""
    local_root = Path(os.environ.get("MINIMAX_H3_GATE_GOLDEN_DIR", Path(__file__).resolve().parent / "goldens"))
    local = local_root / _device_slug() / _golden_filename()
    if local.exists():
        return local, True

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    try:
        remote = hf_hub_download(
            GOLDEN_REPO_ID,
            f"golden_gates/{_device_slug()}/{_golden_filename()}",
            repo_type="dataset",
        )
        return Path(remote), True
    except EntryNotFoundError:
        return local, False


def test_minimax_h3_t2v_golden_gate() -> None:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("golden-gate requires a bf16-capable CUDA GPU")

    # Tripwire: any nondeterministic op added to the block errors immediately
    # instead of flaking the gate. benchmark=False kills cuDNN autotune
    # variance (no-op today — the block has no convs — but the gate may grow).
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0")
    component_dir = _component_dir()

    from fastvideo.forward_context import set_forward_context

    block = _build_block(LAYER)
    state = _load_layer_state(component_dir, LAYER)
    missing, unexpected = block.load_state_dict(state, strict=True)
    assert not missing and not unexpected
    block = block.to(device=device, dtype=torch.bfloat16).eval()

    inputs = _make_inputs(device)
    with torch.inference_mode(), set_forward_context(current_timestep=0, attn_metadata=None):
        out = block(**inputs)
    out = out.detach().float().cpu()

    golden_path, golden_exists = _resolve_golden()
    meta = {
        "layer": LAYER,
        "seed": SEED,
        "shape": list(out.shape),
        "env": _env_fingerprint(),
    }
    if not golden_exists:
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"output": out, "metadata": meta}, golden_path)
        pytest.fail(
            f"golden latent seeded at {golden_path} ({meta}). Verify with a second "
            f"run, then upload to {GOLDEN_REPO_ID} at "
            f"golden_gates/{_device_slug()}/{_golden_filename()}",
            pytrace=False,
        )

    golden = torch.load(golden_path, weights_only=True)
    mismatches = {
        key: (value, meta["env"].get(key))
        for key, value in golden["metadata"]["env"].items() if meta["env"].get(key) != value
    }
    assert not mismatches, (f"environment changed since golden was minted — re-mint deliberately rather "
                            f"than chasing bit drift. golden -> current: {mismatches}")

    drift = (out - golden["output"]).abs()
    print(f"golden-gate drift: max_abs={drift.max().item():.10f} "
          f"mean_abs={drift.mean().item():.10f}", flush=True)
    assert_close(out, golden["output"], atol=0.0, rtol=0.0)
