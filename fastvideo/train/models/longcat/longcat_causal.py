# SPDX-License-Identifier: Apache-2.0
"""LongCat streaming model plugin for self-forcing rollouts.

KV cache management mirrors :class:`WanCausalModel` 's pattern: a
fixed-size buffer per layer, pre-allocated to fit the full rollout
(``training.data.num_frames``), with a write pointer that advances
chunk-by-chunk via in-place ``.copy_()``. Old positions are never
overwritten, so backward (under gradient checkpointing) can safely
re-read the same slice even after later forwards have advanced the
pointer — no per-forward clone of the K/V tensors is needed.

Switching from a growing dict + ``torch.cat`` per chunk to a
pre-allocated buffer keeps peak memory constant at
``num_frames * tokens_per_frame * 2 (K+V) * num_layers`` instead of
growing linearly with the number of chunks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TYPE_CHECKING

import torch

from fastvideo.forward_context import set_forward_context
from fastvideo.train.models.base import CausalModelBase
from fastvideo.train.models.longcat.longcat import LongCatModel

if TYPE_CHECKING:
    from fastvideo.pipelines import TrainingBatch
    from fastvideo.train.utils.training_config import TrainingConfig


@dataclass(slots=True)
class _LongCatStreamingCache:
    """Pre-allocated KV cache buffer for one ``cache_tag``.

    Mirrors the ``WanCausalModel`` pattern: each transformer layer
    gets a fixed-size ``[B, H, max_tokens, D]`` buffer, and a single
    ``write_idx`` (0-d long tensor, like Wan's ``global_end_index``)
    tracks the next write position.
    """

    # layer_idx -> {"k": [B, H, max_tokens, D], "v": [B, H, max_tokens, D]}
    buffers: dict[int, dict[str, torch.Tensor]]
    # Next write position, in tokens. 0-d tensor for cheap snapshotting.
    write_idx: torch.Tensor
    tokens_per_frame: int
    max_tokens: int


class LongCatCausalModel(LongCatModel, CausalModelBase):
    """LongCat model with chunk-wise KV cache accumulation."""

    def __init__(
        self,
        *,
        init_from: str,
        training_config: TrainingConfig,
        trainable: bool = True,
        disable_custom_init_weights: bool = False,
        flow_shift: float = 12.0,
        enable_gradient_checkpointing_type: str | None = None,
        transformer_override_safetensor: str | None = None,
    ) -> None:
        super().__init__(
            init_from=init_from,
            training_config=training_config,
            trainable=trainable,
            disable_custom_init_weights=disable_custom_init_weights,
            flow_shift=flow_shift,
            enable_gradient_checkpointing_type=enable_gradient_checkpointing_type,
            transformer_override_safetensor=transformer_override_safetensor,
        )
        # Lazy-allocated per cache_tag on the first store_kv call.
        self._streaming_caches: dict[str, _LongCatStreamingCache] = {}

    def clear_caches(self, *, cache_tag: str = "pos") -> None:
        self._streaming_caches.pop(str(cache_tag), None)

    def predict_noise_streaming(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
        *,
        conditional: bool,
        cache_tag: str = "pos",
        store_kv: bool = False,
        cur_start_frame: int = 0,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> torch.Tensor | None:
        if attn_kind not in {"dense", "vsa"}:
            raise ValueError(f"Unknown attn_kind: {attn_kind!r}")
        # SelfForcingMethod passes "vsa" for student rollout, but LongCat
        # selects sparse behavior through its native BSA config, not VSA
        # metadata.
        cache_tag = str(cache_tag)
        cur_start_frame = int(cur_start_frame)
        if cur_start_frame < 0:
            raise ValueError("cur_start_frame must be >= 0")

        if conditional:
            text_dict = batch.conditional_dict
            if text_dict is None:
                raise RuntimeError("Missing conditional_dict in TrainingBatch")
        else:
            text_dict = self._get_uncond_text_dict(
                batch, cfg_uncond=cfg_uncond)

        transformer = self.transformer
        dtype = noisy_latents.dtype

        cache_state = self._streaming_caches.get(cache_tag)
        kv_cache_view, cached_frames = self._cache_view(cache_state)

        with torch.autocast(self.device.type,
                            dtype=dtype), set_forward_context(
                                current_timestep=batch.timesteps,
                                attn_metadata=None,
                            ):
            if store_kv:
                empty_embeds = self._empty_kv_cache_inputs(
                    transformer=transformer,
                    batch_size=int(noisy_latents.shape[0]),
                    device=noisy_latents.device,
                    dtype=dtype,
                )
                _, new_chunk = transformer(
                    hidden_states=noisy_latents.permute(0, 2, 1, 3, 4),
                    encoder_hidden_states=empty_embeds,
                    timestep=timestep,
                    num_cond_latents=cached_frames,
                    return_kv=True,
                    skip_crs_attn=True,
                )
                if new_chunk is None:
                    raise RuntimeError(
                        "LongCat transformer returned no KV cache")
                self._streaming_caches[cache_tag] = (
                    self._write_chunk_to_buffer(
                        cache_state=cache_state,
                        new_chunk=new_chunk,
                        new_frames=int(noisy_latents.shape[1]),
                        device=noisy_latents.device,
                    ))
                return None

            pred_noise = transformer(
                hidden_states=noisy_latents.permute(0, 2, 1, 3, 4),
                encoder_hidden_states=text_dict["encoder_hidden_states"],
                encoder_attention_mask=text_dict["encoder_attention_mask"],
                timestep=timestep,
                num_cond_latents=cached_frames,
                kv_cache_dict=kv_cache_view,
                # Buffer never evicts; cached frames always start at frame 0.
                kv_cache_start_frame=0,
            )
            if isinstance(pred_noise, tuple):
                raise RuntimeError("LongCat transformer returned a tuple "
                                   "when return_kv=False")
        return pred_noise.permute(0, 2, 1, 3, 4)

    # ------------------------------------------------------------------
    # KV cache buffer
    # ------------------------------------------------------------------

    def _cache_view(
        self,
        cache_state: _LongCatStreamingCache | None,
    ) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor]] | None, int]:
        """Return ``(kv_cache_dict view, cached_frames)`` for the transformer.

        The slice ``buf[..., :write_idx, :]`` is a view into the
        pre-allocated buffer. Because positions ``[0, write_idx)`` are
        never overwritten, this view stays valid across later writes —
        which is what makes the buffer pattern grad-checkpoint safe
        without cloning the full K/V every forward.
        """
        if cache_state is None:
            return None, 0

        widx = int(cache_state.write_idx.item())
        if widx == 0:
            return None, 0

        view = {
            idx: (buf["k"][:, :, :widx, :], buf["v"][:, :, :widx, :])
            for idx, buf in cache_state.buffers.items()
        }
        cached_frames = widx // cache_state.tokens_per_frame
        return view, cached_frames

    def _write_chunk_to_buffer(
        self,
        *,
        cache_state: _LongCatStreamingCache | None,
        new_chunk: dict[int, tuple[torch.Tensor, torch.Tensor]],
        new_frames: int,
        device: torch.device,
    ) -> _LongCatStreamingCache:
        """Copy the new chunk's K/V into the pre-allocated buffer.

        Lazy-allocates the buffer on the first call from the first
        chunk's K/V shape and ``training_config.data.num_frames``.
        Subsequent calls are pure in-place ``.copy_()`` into the next
        slot, plus an advance of ``write_idx`` — no allocation, no
        ``torch.cat``, no eviction.
        """
        if int(new_frames) <= 0:
            raise ValueError("new_frames must be > 0")
        if not new_chunk:
            raise ValueError("LongCat transformer returned an empty KV cache")

        sample_k, _ = next(iter(new_chunk.values()))
        if sample_k.ndim != 4:
            raise ValueError(
                "Unexpected LongCat KV cache shape; expected [B, H, N, D], "
                f"got ndim={sample_k.ndim}")
        new_tokens = int(sample_k.shape[2])
        if new_tokens % new_frames != 0:
            raise ValueError(
                "LongCat KV cache token count is not divisible by the "
                "number of frames in the cached chunk: "
                f"{new_tokens} vs {new_frames}")
        tokens_per_frame = new_tokens // new_frames

        if cache_state is None:
            # First chunk: allocate a fixed-size buffer sized for the
            # full rollout. We use training_config.data.num_frames as
            # the upper bound.
            tc = self.training_config
            assert tc is not None
            total_frames = int(tc.data.num_frames)
            if total_frames <= 0:
                raise ValueError(
                    "training.num_frames must be > 0 for streaming "
                    f"KV cache; got {total_frames}")
            max_tokens = tokens_per_frame * total_frames
            B = int(sample_k.shape[0])
            H = int(sample_k.shape[1])
            D = int(sample_k.shape[3])

            buffers: dict[int, dict[str, torch.Tensor]] = {}
            for idx, (k, v) in new_chunk.items():
                if (k.shape[0] != B or k.shape[1] != H or k.shape[3] != D):
                    raise ValueError(
                        "LongCat KV cache shape mismatch across layers: "
                        f"layer {idx} has {tuple(k.shape)}, expected "
                        f"({B}, {H}, *, {D})")
                buffers[idx] = {
                    "k":
                    torch.zeros(
                        (B, H, max_tokens, D),
                        dtype=k.dtype,
                        device=k.device,
                    ),
                    "v":
                    torch.zeros(
                        (B, H, max_tokens, D),
                        dtype=v.dtype,
                        device=v.device,
                    ),
                }

            cache_state = _LongCatStreamingCache(
                buffers=buffers,
                write_idx=torch.zeros((), dtype=torch.long, device=device),
                tokens_per_frame=tokens_per_frame,
                max_tokens=max_tokens,
            )
        else:
            if cache_state.tokens_per_frame != tokens_per_frame:
                raise ValueError(
                    "LongCat KV cache token density changed between chunks: "
                    f"{cache_state.tokens_per_frame} vs {tokens_per_frame}")

        widx = int(cache_state.write_idx.item())
        new_widx = widx + new_tokens
        if new_widx > cache_state.max_tokens:
            raise ValueError(
                "LongCat KV cache buffer overflow: tried to write up to "
                f"{new_widx} tokens, buffer capacity is "
                f"{cache_state.max_tokens} "
                f"({cache_state.max_tokens // cache_state.tokens_per_frame} "
                "frames). Increase training.data.num_frames or reduce "
                "the rollout length.")

        for idx, (k, v) in new_chunk.items():
            buf = cache_state.buffers.get(idx)
            if buf is None:
                raise ValueError(
                    f"LongCat returned new K/V for layer {idx} not present "
                    "in the pre-allocated buffer; the layer set must be "
                    "consistent across chunks")
            # In-place write into the next slot. .detach() ensures the
            # buffer is a constant from the next chunk's perspective —
            # gradients flow through the live transformer call, not back
            # through previously-cached K/V (matches WanCausalModel).
            buf["k"][:, :, widx:new_widx, :].copy_(k.detach())
            buf["v"][:, :, widx:new_widx, :].copy_(v.detach())

        cache_state.write_idx.fill_(new_widx)
        return cache_state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _empty_kv_cache_inputs(
        self,
        *,
        transformer: torch.nn.Module,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        base_transformer = getattr(transformer, "module", transformer)
        config = getattr(base_transformer, "config", None)
        caption_dim = int(getattr(config, "caption_channels", 4096))
        text_len = self._get_text_len(default=512)
        return torch.zeros(
            batch_size,
            text_len,
            caption_dim,
            device=device,
            dtype=dtype,
        )

    def _get_text_len(self, *, default: int) -> int:
        pipeline_config = getattr(self.training_config, "pipeline_config",
                                  None)
        text_configs = getattr(pipeline_config, "text_encoder_configs", None)
        if text_configs:
            arch_config = getattr(text_configs[0], "arch_config", None)
            value = getattr(arch_config, "text_len", None)
            if value is not None:
                return int(value)
        return int(default)
