# SPDX-License-Identifier: Apache-2.0
import os
import re
from pathlib import Path
import sys

import pytest
import torch
from safetensors.torch import load_file
from torch.testing import assert_close

os.environ.setdefault("FASTVIDEO_LIGHT_IMPORT", "1")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29513")

repo_root = Path(__file__).resolve().parents[3]
ltx_core_path = repo_root / "LTX-2" / "packages" / "ltx-core" / "src"
if ltx_core_path.exists() and str(ltx_core_path) not in sys.path:
    sys.path.insert(0, str(ltx_core_path))

from fastvideo.configs.models.dits import LTX2VideoConfig
from fastvideo.configs.pipelines import PipelineConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.models.loader.component_loader import TransformerLoader


def _load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    if path.is_file():
        print(f"[LTX2 TEST] Loading weights from file: {path}")
        return load_file(str(path))
    if not path.is_dir():
        raise FileNotFoundError(f"LTX-2 weights not found at {path}")

    model_file = path / "model.safetensors"
    if model_file.exists():
        print(f"[LTX2 TEST] Loading weights from file: {model_file}")
        return load_file(str(model_file))

    index_files = list(path.glob("*.safetensors.index.json"))
    if index_files:
        print(f"[LTX2 TEST] Loading weights from index: {index_files[0]}")
        index = index_files[0].read_text(encoding="utf-8")
        weight_map = __import__("json").loads(index)["weight_map"]
        shards = sorted({path / shard for shard in weight_map.values()})
    else:
        shards = sorted(Path(p) for p in path.glob("*.safetensors"))
    print(f"[LTX2 TEST] Loading {len(shards)} shard(s) from {path}")
    if not shards:
        raise FileNotFoundError(f"No safetensors found in {path}")
    weights: dict[str, torch.Tensor] = {}
    for shard in shards:
        print(f"[LTX2 TEST] Loading shard: {shard}")
        weights.update(load_file(str(shard)))
    print(f"[LTX2 TEST] Loaded {len(weights)} total tensors")
    return weights


def _normalize_keys(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(k.startswith("model.diffusion_model.") for k in weights):
        normalized = {
            k.replace("model.diffusion_model.", ""): v
            for k, v in weights.items()
            if k.startswith("model.diffusion_model.")
        }
        print(f"[LTX2 TEST] Normalized model.diffusion_model.* -> {len(normalized)} tensors")
        return normalized
    if any(k.startswith("diffusion_model.") for k in weights):
        normalized = {
            k.replace("diffusion_model.", ""): v
            for k, v in weights.items()
            if k.startswith("diffusion_model.")
        }
        print(f"[LTX2 TEST] Normalized diffusion_model.* -> {len(normalized)} tensors")
        return normalized
    if any(k.startswith("model.") for k in weights):
        normalized = {k.replace("model.", ""): v for k, v in weights.items()}
        print(f"[LTX2 TEST] Normalized model.* -> {len(normalized)} tensors")
        return normalized
    print(f"[LTX2 TEST] No normalization applied; {len(weights)} tensors")
    return weights


def _select_transformer_weights(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    allowed_prefixes = (
        "patchify_proj.",
        "adaln_single.",
        "caption_projection.",
        "audio_patchify_proj.",
        "audio_adaln_single.",
        "audio_caption_projection.",
        "transformer_blocks.",
        "scale_shift_table",
        "audio_scale_shift_table",
        "av_ca_",
        "norm_out.",
        "audio_norm_out.",
        "proj_out.",
        "audio_proj_out.",
        "video_args_preprocessor.",
    )
    filtered = {
        k: v
        for k, v in weights.items()
        if k.startswith(allowed_prefixes)
        and not k.startswith("audio_embeddings_connector.")
        and not k.startswith("video_embeddings_connector.")
    }
    print(f"[LTX2 TEST] Selected transformer tensors: {len(filtered)}")
    return filtered


def _infer_arch(weights: dict[str, torch.Tensor]) -> dict[str, int]:
    patch_key = None
    if "patchify_proj.weight" in weights:
        patch_key = "patchify_proj.weight"
    else:
        patch_key = next(
            (k for k in weights if k.endswith(".patchify_proj.weight") and "audio_" not in k),
            None,
        )
    if patch_key is None:
        raise KeyError("patchify_proj.weight not found in LTX-2 weights.")
    in_channels = weights[patch_key].shape[1]
    inner_dim = weights[patch_key].shape[0]

    caption_key = next((k for k in weights if k.endswith("caption_projection.linear_1.weight")), None)
    caption_channels = weights[caption_key].shape[1] if caption_key else inner_dim

    block_indices = [
        int(m.group(1))
        for k in weights
        if (m := re.match(r"transformer_blocks\.(\d+)\.", k))
    ]
    num_layers = max(block_indices) + 1 if block_indices else 0

    num_heads_candidates = [32, 16, 64, 8]
    num_attention_heads = next((h for h in num_heads_candidates if inner_dim % h == 0), 32)
    attention_head_dim = inner_dim // num_attention_heads

    patch_size = 2
    num_channels_latents = 16
    for candidate in (16, 32, 64, 128):
        if in_channels % candidate != 0:
            continue
        patch_volume = in_channels // candidate
        root = int(round(patch_volume**0.5))
        if root * root == patch_volume:
            patch_size = root
            num_channels_latents = candidate
            break

    arch = {
        "in_channels": in_channels,
        "out_channels": in_channels,
        "inner_dim": inner_dim,
        "caption_channels": caption_channels,
        "num_layers": num_layers,
        "num_attention_heads": num_attention_heads,
        "attention_head_dim": attention_head_dim,
        "num_channels_latents": num_channels_latents,
        "patch_size": patch_size,
    }
    print(f"[LTX2 TEST] Inferred arch: {arch}")
    return arch


def _load_into_model(model: torch.nn.Module, weights: dict[str, torch.Tensor]) -> int:
    model_state = model.state_dict()
    filtered = {
        k: v
        for k, v in weights.items()
        if k in model_state and model_state[k].shape == v.shape
    }
    print(
        f"[LTX2 TEST] Loading {len(filtered)} / {len(model_state)} tensors "
        f"from {len(weights)} available weights"
    )
    if not filtered:
        return 0
    model.load_state_dict(filtered, strict=False)
    return len(filtered)


def _attach_block_sum_logging(
    model: torch.nn.Module,
    log_path: Path,
    label: str,
    enabled: bool,
) -> None:
    if not enabled:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    def _format_sum(tensor: torch.Tensor | None) -> str:
        if tensor is None:
            return "None"
        return f"{tensor.float().sum().item():.6f}"

    def _hook(module, inputs, outputs):  # noqa: ANN001
        if isinstance(outputs, tuple):
            video_args, audio_args = outputs
            video_sum = _format_sum(video_args.x if video_args is not None else None)
            audio_sum = _format_sum(audio_args.x if audio_args is not None else None)
        else:
            video_sum = _format_sum(outputs)
            audio_sum = "None"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"{label}:{module.idx}:video_sum={video_sum},audio_sum={audio_sum}\n")

    for block in model.transformer_blocks:
        block.register_forward_hook(_hook)


def _attach_block_detail_logging(
    model: torch.nn.Module,
    log_path: Path,
    label: str,
    enabled: bool,
) -> None:
    if not enabled:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    def _format_sum(tensor: torch.Tensor | None) -> str:
        if tensor is None:
            return "None"
        return f"{tensor.float().sum().item():.6f}"

    def _hook_factory(block_idx: int, name: str):
        def _hook(_module, _inputs, outputs):  # noqa: ANN001
            if isinstance(outputs, tuple):
                out = outputs[0]
            else:
                out = outputs
            out_sum = _format_sum(out if torch.is_tensor(out) else None)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"{label}:{block_idx}:{name}:out_sum={out_sum}\n")
        return _hook

    for block in model.transformer_blocks:
        idx = block.idx
        for name in (
            "attn1",
            "attn2",
            "ff",
            "audio_attn1",
            "audio_attn2",
            "audio_ff",
            "audio_to_video_attn",
            "video_to_audio_attn",
        ):
            if hasattr(block, name):
                getattr(block, name).register_forward_hook(_hook_factory(idx, name))


def test_ltx2_transformer_parity():
    torch.manual_seed(42)
    official_path = Path(
        os.getenv(
            "LTX2_OFFICIAL_PATH",
            "official_ltx_weights/ltx-2-19b-distilled.safetensors",
        )
    )
    fastvideo_path = Path(
        os.getenv(
            "LTX2_FASTVIDEO_PATH",
            "converted/ltx2/transformer",
        )
    )
    if not official_path.exists():
        pytest.skip(f"LTX-2 official weights not found at {official_path}")
    if not fastvideo_path.exists():
        pytest.skip(f"FastVideo converted weights not found at {fastvideo_path}")

    try:
        from ltx_core.components.patchifiers import VideoLatentPatchifier
        from ltx_core.guidance.perturbations import BatchedPerturbationConfig
        from ltx_core.model.transformer import LTXModel
        from ltx_core.model.transformer.model import LTXModelType
        from ltx_core.model.transformer.modality import Modality
        from ltx_core.model.transformer.attention import AttentionFunction
        from ltx_core.model.transformer.rope import LTXRopeType
        from ltx_core.types import VideoLatentShape
    except ImportError as exc:
        pytest.skip(f"LTX-2 import failed: {exc}")

    ref_raw_weights = _load_safetensors(official_path)
    ref_weights = _normalize_keys(ref_raw_weights)
    ref_weights = _select_transformer_weights(ref_weights)
    if not ref_weights:
        pytest.skip("No transformer weights found in safetensors file.")
    arch = _infer_arch(ref_weights)
    if arch["num_layers"] == 0:
        pytest.skip("Could not infer transformer block count from LTX-2 weights.")

    config = LTX2VideoConfig()
    cfg = config.arch_config
    cfg.in_channels = arch["in_channels"]
    cfg.out_channels = arch["out_channels"]
    cfg.num_layers = arch["num_layers"]
    cfg.num_attention_heads = arch["num_attention_heads"]
    cfg.attention_head_dim = arch["attention_head_dim"]
    cfg.cross_attention_dim = arch["inner_dim"]
    cfg.caption_channels = arch["caption_channels"]
    cfg.num_channels_latents = arch["num_channels_latents"]
    cfg.patch_size = (1, arch["patch_size"], arch["patch_size"])

    if not torch.cuda.is_available():
        pytest.skip("LTX-2 transformer parity test requires CUDA for attention backends.")

    device = torch.device("cuda:0")
    precision = torch.bfloat16
    precision_str = "bf16"

    args = FastVideoArgs(
        model_path=str(fastvideo_path),
        dit_cpu_offload=True,
        use_fsdp_inference=False,
        pipeline_config=PipelineConfig(dit_config=config, dit_precision=precision_str),
    )
    args.device = device

    loader = TransformerLoader()
    fastvideo_model = loader.load(str(fastvideo_path), args).to(device=device, dtype=precision)

    reference_attention = AttentionFunction.PYTORCH
    reference_attn_env = os.getenv("LTX2_REFERENCE_ATTN")
    if reference_attn_env:
        reference_attention = AttentionFunction(reference_attn_env)
    else:
        fastvideo_attn_backend = os.getenv("FASTVIDEO_ATTENTION_BACKEND")
        if fastvideo_attn_backend == "FLASH_ATTN":
            reference_attention = AttentionFunction.FLASH_ATTENTION_3

    reference_model = LTXModel(
        model_type=LTXModelType.AudioVideo,
        num_attention_heads=cfg.num_attention_heads,
        attention_head_dim=cfg.attention_head_dim,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        num_layers=cfg.num_layers,
        cross_attention_dim=cfg.cross_attention_dim,
        norm_eps=cfg.norm_eps,
        caption_channels=cfg.caption_channels,
        positional_embedding_theta=cfg.positional_embedding_theta,
        positional_embedding_max_pos=cfg.positional_embedding_max_pos,
        timestep_scale_multiplier=cfg.timestep_scale_multiplier,
        use_middle_indices_grid=cfg.use_middle_indices_grid,
        attention_type=reference_attention,
        rope_type=LTXRopeType(cfg.rope_type),
        double_precision_rope=cfg.double_precision_rope,
        audio_num_attention_heads=cfg.audio_num_attention_heads,
        audio_attention_head_dim=cfg.audio_attention_head_dim,
        audio_in_channels=cfg.audio_in_channels,
        audio_out_channels=cfg.audio_out_channels,
        audio_cross_attention_dim=cfg.audio_cross_attention_dim,
        audio_positional_embedding_max_pos=cfg.audio_positional_embedding_max_pos,
        av_ca_timestep_scale_multiplier=cfg.av_ca_timestep_scale_multiplier,
    ).to(device=device, dtype=precision)

    loaded_reference = _load_into_model(reference_model, ref_weights)
    if loaded_reference == 0:
        pytest.skip("No matching keys loaded into LTX-2 transformer models.")

    fastvideo_model.eval()
    reference_model.eval()

    debug_logs = os.getenv("LTX2_DEBUG_LOGS", "0") == "1"
    _attach_block_sum_logging(
        fastvideo_model.model,
        repo_root / "ltx2_debug" / "fastvideo.log",
        "fastvideo",
        debug_logs,
    )
    _attach_block_sum_logging(
        reference_model,
        repo_root / "ltx2_debug" / "reference.log",
        "reference",
        debug_logs,
    )
    _attach_block_detail_logging(
        fastvideo_model.model,
        repo_root / "ltx2_debug" / "fastvideo_detail.log",
        "fastvideo",
        os.getenv("LTX2_DEBUG_DETAIL", "0") == "1",
    )
    _attach_block_detail_logging(
        reference_model,
        repo_root / "ltx2_debug" / "reference_detail.log",
        "reference",
        os.getenv("LTX2_DEBUG_DETAIL", "0") == "1",
    )

    patchifier = VideoLatentPatchifier(patch_size=cfg.patch_size[1])
    batch_size = 1
    frames = 4
    height = cfg.patch_size[1] * 4
    width = cfg.patch_size[2] * 4
    hidden_states = torch.randn(
        batch_size,
        cfg.num_channels_latents,
        frames,
        height,
        width,
        device=device,
        dtype=precision,
    )
    encoder_hidden_states = torch.randn(
        batch_size,
        16,
        cfg.caption_channels,
        device=device,
        dtype=precision,
    )
    timestep = torch.tensor([500], device=device, dtype=precision)

    video_shape = VideoLatentShape.from_torch_shape(hidden_states.shape)
    positions = patchifier.get_patch_grid_bounds(video_shape, device=hidden_states.device)
    latents = patchifier.patchify(hidden_states)

    video = Modality(
        enabled=True,
        latent=latents,
        timesteps=timestep,
        positions=positions,
        context=encoder_hidden_states,
        context_mask=None,
    )

    with torch.no_grad():
        ref_out, _ = reference_model(
            video=video,
            audio=None,
            perturbations=BatchedPerturbationConfig.empty(batch_size),
        )
        ref_out = patchifier.unpatchify(ref_out, output_shape=video_shape)
        print(f"[LTX2 TEST] Reference model output shape: {ref_out.shape}")
        with set_forward_context(
            current_timestep=0,
            attn_metadata=None,
            forward_batch=None,
        ):
            fastvideo_out = fastvideo_model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
            )
            print(f"[LTX2 TEST] FastVideo model output shape: {fastvideo_out.shape}")
    assert ref_out.shape == fastvideo_out.shape
    assert ref_out.dtype == fastvideo_out.dtype
    assert_close(ref_out, fastvideo_out, atol=1e-2, rtol=1e-2)
