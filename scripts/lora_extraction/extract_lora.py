"""Extract FastVideo-style LoRA adapters from a fine-tuned model by SVDing (FT - base).

Usage:
    python scripts/lora_extraction/extract_lora.py \\
        --base <base_model> --ft <fine_tuned_model> --out adapter.safetensors --rank 16
        
Example for models with architectural differences (fallback is automatic):
    python extract_lora.py \\
        --base Wan-AI/Wan2.1-T2V-1.3B-Diffusers \\
        --ft FastVideo/FastWan2.1-T2V-1.3B-Diffusers \\
        --out fastvideo_adapter.safetensors \\
        --rank 32
"""

from __future__ import annotations

import os
# Set distributed env BEFORE any fastvideo imports
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")

import argparse
import sys
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
from tqdm import tqdm

# Optional safetensors support
_HAVE_SAFETENSORS = True
try:
    from safetensors.torch import save_file as safetensors_save  # type: ignore
    from safetensors import safe_open  # type: ignore
except Exception:
    _HAVE_SAFETENSORS = False
    safe_open = None  # type: ignore


def load_transformer_state_dict_from_safetensors(model_path: str) -> Dict[str, torch.Tensor]:
    """Load transformer weights directly from safetensors files.
    
    This bypasses the pipeline loader and works even when the model has
    architectural differences (e.g., extra layers in fine-tuned model).
    
    Args:
        model_path: HuggingFace model ID or local path
        
    Returns:
        State dict with all transformer weights
    """
    from huggingface_hub import snapshot_download
    import os
    
    # Download or locate the model
    if os.path.isdir(model_path):
        local_path = model_path
    else:
        local_path = snapshot_download(model_path)
    
    # Find transformer directory
    transformer_dir = os.path.join(local_path, "transformer")
    if not os.path.isdir(transformer_dir):
        raise FileNotFoundError(f"Transformer directory not found at {transformer_dir}")
    
    # Load all safetensors files
    state_dict: Dict[str, torch.Tensor] = {}
    for fname in sorted(os.listdir(transformer_dir)):
        if fname.endswith('.safetensors'):
            fpath = os.path.join(transformer_dir, fname)
            with safe_open(fpath, framework='pt', device='cpu') as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
    
    if not state_dict:
        raise ValueError(f"No safetensors files found in {transformer_dir}")
    
    LOG.info("Loaded %d keys directly from safetensors", len(state_dict))
    return state_dict

# Configure minimal logging
LOG = logging.getLogger("extract_lora")


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    fmt = "%(asctime)s %(levelname)s %(message)s"
    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    LOG.addHandler(handler)
    LOG.setLevel(level)


def get_pipeline_class_for_model(model_path: str):
    """Return appropriate FastVideo Pipeline class for the model."""
    from fastvideo.utils import maybe_download_model_index  # local import
    from fastvideo.pipelines.pipeline_registry import get_pipeline_registry, PipelineType
    from fastvideo.fastvideo_args import WorkloadType

    config = maybe_download_model_index(model_path)
    pipeline_name = config.get("_class_name")
    if pipeline_name is None:
        raise ValueError(f"Model config for {model_path} missing _class_name (diffusers format expected).")

    pipeline_registry = get_pipeline_registry(PipelineType.BASIC)
    pipeline_cls = pipeline_registry.resolve_pipeline_cls(pipeline_name, PipelineType.BASIC, WorkloadType.T2V)
    return pipeline_cls


def load_transformer_state_dict_from_model(
    model_path: str,
    num_gpus: int = 1,
    dit_cpu_offload: bool = True,
    vae_cpu_offload: bool = True,
    text_encoder_cpu_offload: bool = True,
    pin_cpu_memory: bool = True,
) -> Dict[str, torch.Tensor]:
    """Load pipeline and extract transformer.state_dict as CPU tensors."""
    pipeline_cls = get_pipeline_class_for_model(model_path)
    pipeline = pipeline_cls.from_pretrained(
        model_path,
        num_gpus=num_gpus,
        inference_mode=True,
        dit_cpu_offload=dit_cpu_offload,
        vae_cpu_offload=vae_cpu_offload,
        text_encoder_cpu_offload=text_encoder_cpu_offload,
        pin_cpu_memory=pin_cpu_memory,
    )

    # Try to locate transformer in several typical attributes
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        modules = getattr(pipeline, "modules", None)
        if isinstance(modules, dict):
            transformer = modules.get("transformer")
    if transformer is None:
        pipeline_attr = getattr(pipeline, "pipeline", None)
        transformer = getattr(pipeline_attr, "transformer", None) if pipeline_attr else None
    if transformer is None:
        raise RuntimeError("Transformer not found in pipeline. Expected pipeline.transformer or pipeline.modules['transformer'].")

    state_dict = transformer.state_dict()

    # DTensor safe handling
    try:
        from torch.distributed.tensor import DTensor  # type: ignore
        _HAS_DTENSOR = True
    except Exception:
        DTensor = None  # type: ignore
        _HAS_DTENSOR = False

    state_dict_cpu: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if _HAS_DTENSOR and isinstance(v, DTensor):  # type: ignore
            state_dict_cpu[k] = v.to_local().detach().cpu().contiguous()
        else:
            state_dict_cpu[k] = v.detach().cpu().contiguous()

    # cleanup
    try:
        del pipeline, transformer
    except Exception:
        pass
    torch.cuda.empty_cache()
    return state_dict_cpu


def is_extractable_weight(key: str) -> bool:
    """Return True if key represents a weight suitable for LoRA extraction."""
    if not key.endswith("weight"):
        return False
    low = key.lower()
    for skip in ("norm", "bias", "embedding"):
        if skip in low:
            return False
    return True


def save_adapter_state(adapter_state: Dict[str, torch.Tensor], out_path: Path) -> None:
    """Save adapter state dict to safetensors (if available) or torch.save."""
    cleaned = {k: v.detach().cpu().contiguous() for k, v in adapter_state.items()}
    out_str = str(out_path)
    if out_path.suffix == ".safetensors" and _HAVE_SAFETENSORS:
        safetensors_save(cleaned, out_str)
    else:
        torch.save(cleaned, out_str)


def build_adapter_from_states(
    base_sd: Dict[str, torch.Tensor],
    ft_sd: Dict[str, torch.Tensor],
    rank: int,
    full_rank: bool,
    min_delta: float,
    checkpoint_interval: int,
    checkpoint_path: Optional[Path],
    resume_from: int = 0,
) -> Dict[str, torch.Tensor]:
    """Compute low-rank LoRA adapters by SVD on (ft - base) for extractable weights."""
    # DTensor detection
    try:
        from torch.distributed.tensor import DTensor  # type: ignore
        _HAS_DTENSOR = True
    except Exception:
        DTensor = None  # type: ignore
        _HAS_DTENSOR = False

    keys = sorted(ft_sd.keys())
    adapter_state: Dict[str, torch.Tensor] = {}
    processed = 0
    mean_deltas = []

    for idx, key in enumerate(tqdm(keys, desc="scanning keys", unit="keys")):
        if idx < resume_from:
            continue
        if not is_extractable_weight(key):
            continue
        if key not in base_sd:
            continue

        Wb_raw = base_sd[key]
        Wf_raw = ft_sd[key]

        # Convert DTensor if present
        if _HAS_DTENSOR and isinstance(Wb_raw, DTensor):  # type: ignore
            Wb = Wb_raw.to_local().detach().cpu().to(torch.float32).contiguous()
        else:
            Wb = Wb_raw.detach().cpu().to(torch.float32).contiguous()

        if _HAS_DTENSOR and isinstance(Wf_raw, DTensor):  # type: ignore
            Wf = Wf_raw.to_local().detach().cpu().to(torch.float32).contiguous()
        else:
            Wf = Wf_raw.detach().cpu().to(torch.float32).contiguous()

        if Wb.shape != Wf.shape:
            continue

        delta = (Wf - Wb).contiguous()
        mean_abs = float(delta.abs().mean().item())
        mean_deltas.append(mean_abs)
        if mean_abs < min_delta:
            continue

        # SVD (CPU)
        try:
            U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        except RuntimeError:
            # skip layers that fail SVD
            continue

        max_rank = S.numel()
        chosen_rank = max_rank if full_rank or rank <= 0 else min(rank, max_rank)
        if chosen_rank == 0:
            continue

        S_sqrt = torch.sqrt(S[:chosen_rank].to(torch.float32))
        U_r = U[:, :chosen_rank].to(torch.float32)    # (out, r)
        Vh_r = Vh[:chosen_rank, :].to(torch.float32)  # (r, in)

        lora_B = (U_r * S_sqrt.unsqueeze(0)).contiguous()        # (out, r)
        tmp = (Vh_r.T * S_sqrt.unsqueeze(0)).contiguous()       # (in, r)
        lora_A = tmp.T.contiguous()                              # (r, in)

        base_name = key[:-len(".weight")]
        a_key = f"{base_name}.lora_A.weight"
        b_key = f"{base_name}.lora_B.weight"
        rank_key = f"{base_name}.lora_rank"
        alpha_key = f"{base_name}.lora_alpha"

        adapter_state[a_key] = lora_A.cpu()
        adapter_state[b_key] = lora_B.cpu()
        adapter_state[rank_key] = torch.tensor([chosen_rank], dtype=torch.int32)
        adapter_state[alpha_key] = torch.tensor([float(chosen_rank)], dtype=torch.float32)

        processed += 1

        # checkpoint periodically
        if checkpoint_path and checkpoint_interval > 0 and (idx + 1) % checkpoint_interval == 0:
            try:
                torch.save({"index": idx + 1, "adapter": adapter_state}, str(checkpoint_path))
            except Exception:
                # non-fatal; continue
                pass

        # free local large tensors
        del delta, U, S, Vh, U_r, Vh_r, tmp, lora_A, lora_B

    # final checkpoint
    if checkpoint_path:
        try:
            torch.save({"index": len(keys), "adapter": adapter_state}, str(checkpoint_path))
        except Exception:
            pass

    avg_delta = (sum(mean_deltas) / len(mean_deltas)) if mean_deltas else 0.0
    LOG.info("Extraction complete: processed_keys=%d, extracted_layers=%d, avg_abs_delta=%.6e",
             len(keys), processed, avg_delta)
    return adapter_state


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract FastVideo-style LoRA adapter (CPU SVD).")
    p.add_argument("--base", required=True, help="Base model id or local path")
    p.add_argument("--ft", required=True, help="Fine-tuned model id or local path")
    p.add_argument("--out", default="fastvideo_adapter.safetensors", help="Output adapter file (.safetensors or .pt)")
    p.add_argument("--rank", type=int, default=16, help="Truncated SVD rank; <=0 for full rank")
    p.add_argument("--full-rank", action="store_true", help="Use full SVD rank for every layer")
    p.add_argument("--min-delta", type=float, default=1e-8, help="Minimum mean abs delta to consider a layer changed")
    p.add_argument("--checkpoint", default="extract_lora_checkpoint.pt", help="Checkpoint path to resume/save progress")
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint if available")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return p.parse_args()


def extract_lora_adapter(
    base: str,
    ft: str,
    out: str,
    rank: int = 32,
    full_rank: bool = False,
    min_delta: float = 1e-6,
    checkpoint: Optional[str] = None,
    resume: bool = False,
    log_level: str = "INFO",
) -> None:
    """Extract LoRA adapter from fine-tuned model.
    
    Args:
        base: Base model path or HuggingFace ID
        ft: Fine-tuned model path or HuggingFace ID
        out: Output adapter file path
        rank: LoRA rank (default: 32)
        full_rank: Extract full-rank adapter
        min_delta: Minimum delta for extraction
        checkpoint: Checkpoint file path
        resume: Resume from checkpoint
        log_level: Logging level
    """
    configure_logging(log_level)

    out_path = Path(out)
    checkpoint_path = Path(checkpoint) if checkpoint else None

    # Load both models - ensure consistent loading method for matching keys
    try:
        import fastvideo  # noqa: F401
        LOG.info("Loading base model via pipeline: %s", base)
        base_sd = load_transformer_state_dict_from_model(base)
        LOG.info("Loading fine-tuned model via pipeline: %s", ft)
        ft_sd = load_transformer_state_dict_from_model(ft)
    except Exception as exc:
        LOG.warning("Pipeline loading failed: %s", exc)
        LOG.info("Falling back to direct safetensors loading for BOTH models...")
        # Direct loading - both models use same method for consistent keys
        LOG.info("Loading base model directly from safetensors: %s", base)
        base_sd = load_transformer_state_dict_from_safetensors(base)
        LOG.info("Loading fine-tuned model directly from safetensors: %s", ft)
        ft_sd = load_transformer_state_dict_from_safetensors(ft)


    resume_idx = 0
    adapter_existing: Dict[str, torch.Tensor] = {}
    if resume and checkpoint_path and checkpoint_path.exists():
        try:
            ck = torch.load(str(checkpoint_path), map_location="cpu")
            adapter_existing = ck.get("adapter", {}) or {}
            resume_idx = int(ck.get("index", 0) or 0)
            LOG.info("Resuming from checkpoint index=%d with %d existing entries", resume_idx, len(adapter_existing))
        except Exception:
            adapter_existing = {}

    adapter_state = dict(adapter_existing) if adapter_existing else {}
    new_adapter = build_adapter_from_states(
        base_sd=base_sd,
        ft_sd=ft_sd,
        rank=rank,
        full_rank=full_rank,
        min_delta=min_delta,
        checkpoint_interval=50,
        checkpoint_path=checkpoint_path,
        resume_from=resume_idx,
    )
    adapter_state.update(new_adapter)

    # final save
    save_adapter_state(adapter_state, out_path)

    # cleanup checkpoint if present
    if checkpoint_path and checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
        except Exception:
            pass

    LOG.info("Saved adapter to %s (entries=%d)", str(out_path), len(adapter_state) // 4)


def main() -> None:
    """CLI wrapper for extract_lora_adapter."""
    args = parse_args()
    extract_lora_adapter(
        base=args.base,
        ft=args.ft,
        out=args.out,
        rank=args.rank,
        full_rank=args.full_rank,
        min_delta=args.min_delta,
        checkpoint=args.checkpoint,
        resume=args.resume,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
