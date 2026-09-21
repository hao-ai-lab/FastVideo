# SPDX-License-Identifier: Apache-2.0
"""Component-level pipeline parallel for MiniMax H3: dedicated text-encoder ranks.

Enabled with ``FASTVIDEO_H3_ENCODER_SPLIT=1`` (or ``FastVideoArgs.h3_encoder_split``)
on the Ray backend. The workers on the first ``FASTVIDEO_H3_ENCODER_NODES`` nodes
(ranks ``0..n-1``) load only the Qwen3-VL conditioner, run the condition stages, and
NCCL-broadcast ``prompt_embeds`` to the remaining denoising ranks over the world
group. The denoising ranks never load the text encoder, so their peak resident
memory drops by the encoder size (~48 GiB on GB10), which is what makes 720p fit in
128 GiB of unified memory.

The denoising ranks form the sequence-parallel group (``sp_size`` is recomputed per
rank as ``num_gpus - h3_encoder_workers``); encoder ranks hold singleton SP/DP
groups so every rank still belongs to exactly one group of every kind, keeping
``GroupCoordinator`` construction uniform.

Constraints and semantics:
- The denoise group runs SP, so its size must divide the DiT attention head count
  (56 in the released FastH3 checkpoints). With 8 nodes the legal encoder-node
  counts are therefore 1, 4, 6, 7 (SP = 7, 4, 2, 1). This is validated early — in
  ``FastVideoArgs.check_fastvideo_args``, in the Ray executor against the
  placement-derived worker count, and again here before any weights load.
- One request packs a single prompt presentation, so **world rank 0 is the only
  encoder that computes**; encoder ranks ``1..n-1`` mirror the broadcast
  collectives and stay idle. Extra encoder nodes are reserved capacity (e.g. for
  future multi-prompt parallel encoding), not a speedup for single requests.
"""

from __future__ import annotations

from typing import Any

import torch

from fastvideo.distributed import get_local_torch_device, get_world_group
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

logger = init_logger(__name__)

# Modules the encoder group owns. Everything else in the H3 manifest stays off
# these ranks; schedulers are cheap CPU objects and both groups need them.
H3_ENCODER_MODULE_NAMES = frozenset({"text_encoder", "tokenizer", "processor", "scheduler", "audio_scheduler"})
H3_DENOISE_MODULE_NAMES = frozenset(
    {"tokenizer", "processor", "vae", "audio_vae", "transformer", "scheduler", "audio_scheduler"})

_DTYPE_BY_NAME = {
    "torch.bfloat16": torch.bfloat16,
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
}


def h3_encoder_split_enabled(fastvideo_args: FastVideoArgs | None) -> bool:
    return bool(getattr(fastvideo_args, "h3_encoder_split", False))


def h3_encoder_worker_count(fastvideo_args: FastVideoArgs) -> int:
    """Number of world ranks reserved for the encoder group.

    The Ray executor stamps the placement-derived worker count onto args; other
    entrypoints fall back to the node count (one worker per node, which is the
    DGX Spark / single-GPU-per-node topology this targets).
    """
    world_size = get_world_group().world_size
    count = int(getattr(fastvideo_args, "h3_encoder_workers", 0) or 0)
    if count <= 0:
        count = max(1, int(getattr(fastvideo_args, "h3_encoder_nodes", 1) or 1))
    if not 1 <= count < world_size:
        raise ValueError(f"MiniMax-H3 encoder split needs 1 <= encoder workers < world size, got "
                         f"encoder_workers={count}, world_size={world_size}.")
    return count


def h3_is_encoder_worker(fastvideo_args: FastVideoArgs) -> bool:
    return get_world_group().rank < h3_encoder_worker_count(fastvideo_args)


def h3_is_primary_encoder_worker(fastvideo_args: FastVideoArgs) -> bool:
    """Rank 0 encodes every request; ranks 1..n-1 only mirror the broadcast."""
    return get_world_group().rank == 0


def h3_is_output_worker(fastvideo_args: FastVideoArgs) -> bool:
    """Rank that owns the decoded output: world rank 0 normally, first denoise rank in split."""
    world_group = get_world_group()
    if not h3_encoder_split_enabled(fastvideo_args):
        return world_group.is_first_rank
    return world_group.rank == h3_encoder_worker_count(fastvideo_args)


def h3_prepare_split_worker_parallelism(fastvideo_args: FastVideoArgs, rank: int,
                                        world_size: int) -> tuple[int, list[list[int]], list[list[int]]]:
    """Resolve per-rank SP size and the custom SP/DP group layout for split mode.

    Called by the worker before ``maybe_init_distributed_environment_and_model_parallel``,
    and stamped onto the worker-local ``fastvideo_args`` so the consistency check in
    ``ComposedPipelineBase.__init__`` sees the same per-rank SP size.
    """
    if fastvideo_args.distributed_executor_backend != "ray":
        raise ValueError("MiniMax-H3 encoder split (h3_encoder_split) requires distributed_executor_backend='ray'.")
    count = int(getattr(fastvideo_args, "h3_encoder_workers", 0) or 0)
    if count <= 0:
        count = max(1, int(getattr(fastvideo_args, "h3_encoder_nodes", 1) or 1))
    if not 1 <= count < world_size:
        raise ValueError(f"MiniMax-H3 encoder split needs 1 <= encoder nodes < num_gpus, got "
                         f"encoder={count}, world={world_size}.")
    from fastvideo.fastvideo_args import h3_split_sp_error, probe_h3_attention_heads

    heads = probe_h3_attention_heads(fastvideo_args.model_path)
    if heads and heads % (world_size - count):
        raise ValueError(h3_split_sp_error(heads, world_size - count, world_size))
    if rank < count:
        sp_size = 1
    else:
        sp_size = world_size - count
        if int(getattr(fastvideo_args, "h3_sequential_load", False) or False):
            logger.info("MiniMax-H3 encoder split: sequential module load is redundant on denoise ranks and stays off")
    sp_group_ranks = [[r] for r in range(count)] + [list(range(count, world_size))]
    dp_group_ranks = [[r] for r in range(world_size)]
    fastvideo_args.sp_size = sp_size
    fastvideo_args.h3_encoder_workers = count
    logger.info("MiniMax-H3 split: rank %d is an %s worker (sp_size=%d, denoise ranks %d..%d)",
                rank,
                "encoder" if rank < count else "denoise",
                sp_size,
                count,
                world_size - 1,
                local_main_process_only=False)
    return sp_size, sp_group_ranks, dp_group_ranks


def _embed_specs(embeds: list[torch.Tensor]) -> list[dict[str, Any]]:
    return [{
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    } for tensor in embeds]


def h3_broadcast_condition(batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> None:
    """Send the conditioning output to every rank over the world NCCL group.

    Only world rank 0 encodes a request (one prompt presentation per request),
    so this runs on rank 0 as the source. Other encoder ranks mirror the same
    collectives through ``h3_receive_condition`` and drop the payload; skipping
    them entirely would desynchronize the world-group broadcasts for the
    denoise ranks.
    """
    # Imported lazily: stages/__init__ imports the decoding stage, which imports
    # this module, so a top-level import would close that cycle mid-init.
    from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_conditioning import MINIMAX_H3_TEXT_TOKEN_TAGS_KEY

    world_group = get_world_group()
    if world_group.rank != 0:
        raise RuntimeError("h3_broadcast_condition must run on the primary encoder worker (rank 0).")
    if not batch.prompt_embeds:
        raise RuntimeError("MiniMax-H3 encoder worker has no prompt_embeds to send.")
    tags = batch.extra.get(MINIMAX_H3_TEXT_TOKEN_TAGS_KEY)
    if tags is None:
        raise RuntimeError("MiniMax-H3 encoder worker has no text token tags to send.")

    device = get_local_torch_device()
    meta = {
        "embeds": _embed_specs(batch.prompt_embeds),
        "num_tags": int(tags.numel()),
    }
    world_group.broadcast_object(meta, src=0)
    for tensor in batch.prompt_embeds:
        world_group.broadcast(tensor.contiguous(), src=0)
    tags_on_device = tags.to(device=device, dtype=torch.long).contiguous()
    world_group.broadcast(tags_on_device, src=0)
    logger.info("MiniMax-H3 encoder worker broadcast prompt_embeds: %s", _embed_specs(batch.prompt_embeds))


def h3_receive_condition(batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> None:
    """Materialize the encoder worker's conditioning output on a denoise rank."""
    from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_conditioning import MINIMAX_H3_TEXT_TOKEN_TAGS_KEY

    world_group = get_world_group()
    device = get_local_torch_device()
    meta = world_group.broadcast_object(None, src=0)
    embeds: list[torch.Tensor] = []
    for spec in meta["embeds"]:
        dtype = _DTYPE_BY_NAME.get(spec["dtype"])
        if dtype is None:
            raise RuntimeError(f"Unsupported prompt_embeds dtype from encoder worker: {spec['dtype']}")
        tensor = torch.empty(tuple(spec["shape"]), dtype=dtype, device=device)
        world_group.broadcast(tensor, src=0)
        embeds.append(tensor)
    tags = torch.empty(int(meta["num_tags"]), dtype=torch.long, device=device)
    world_group.broadcast(tags, src=0)
    batch.prompt_embeds = embeds
    batch.extra[MINIMAX_H3_TEXT_TOKEN_TAGS_KEY] = tags.cpu()
    logger.info("MiniMax-H3 split worker received prompt_embeds: %s",
                _embed_specs(embeds),
                local_main_process_only=False)


__all__ = [
    "H3_DENOISE_MODULE_NAMES",
    "H3_ENCODER_MODULE_NAMES",
    "h3_broadcast_condition",
    "h3_encoder_split_enabled",
    "h3_encoder_worker_count",
    "h3_is_encoder_worker",
    "h3_is_output_worker",
    "h3_is_primary_encoder_worker",
    "h3_prepare_split_worker_parallelism",
    "h3_receive_condition",
]
