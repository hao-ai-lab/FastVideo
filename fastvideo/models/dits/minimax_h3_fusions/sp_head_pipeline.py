# SPDX-License-Identifier: Apache-2.0
"""Overlap independent H3 attention head groups with FP8 Ulysses exchanges."""

import torch

from fastvideo.distributed.async_collectives import (
    PeerSlices, all_gather_tensor_async, all_to_all_tensor_async, exchange_peer_tensor_async,
)

from fastvideo.attention.backends.video_sparse_attn_h3 import _build_block_mask, _pool_tiles
from fastvideo.models.dits import minimax_h3_vsa_fp4 as base
from fastvideo.models.dits.minimax_h3_fusions.sp_exchange import unpack_fp8_gate
from fastvideo.models.dits.minimax_h3_fusions.sp_qkv_pack import pack_normalized_qkv_fp8


def _start_exchange(payload, scale, group):
    if payload.shape[0] == 2 and payload.shape[-1] == 128:
        packet = exchange_peer_tensor_async(payload, group)
    else:
        packet = all_to_all_tensor_async(payload.view(torch.uint8), group)
    return packet, all_to_all_tensor_async(scale, group, align_bytes=16)


def _finish_exchange(packet):
    payload, scale = packet
    received = payload.wait()
    if not isinstance(received, PeerSlices):
        received = received.view(torch.float8_e4m3fn)
    return received, scale.wait()


def _output_quant_method(attn):
    from fastvideo.layers.quantization.nvfp4_config import NVFP4QuantizeMethod

    method = getattr(attn.to_out, "quant_method", None)
    if (type(method) is NVFP4QuantizeMethod and method.wants_prequantized_input()
            and method.uses_unit_activation_scale(attn.to_out) and attn.attention_head_dim % 64 == 0):
        return method
    return None


def _prepare_tiles(api, layout, payload, scale, query_start):
    if (payload.shape[-1] == 128 and not torch.is_grad_enabled() and not torch.compiler.is_compiling()
            and torch.cuda.get_device_capability(payload.device)[0] >= 10
            and all(hasattr(api, name) for name in ("blockscaled_fp4_attn_sparse", "_zero_delta_s",
                                                    "scale_and_quant_fp4_permute", "scale_and_quant_fp4_transpose"))):
        return layout.tiles_from_fp8(payload, scale, query_start=query_start)
    return layout.tiles_from_fp8(payload, scale), None


def _prepare_values(api, value, valid, tile, logical):
    if (tile == 64 and logical == value.shape[1] and value.shape[-1] == 128 and value.stride(-1) == 1
            and not torch.is_grad_enabled() and not torch.compiler.is_compiling()
            and torch.cuda.get_device_capability(value.device)[0] >= 10
            and all(hasattr(api, name) for name in ("blockscaled_fp4_attn_sparse", "_zero_delta_s",
                                                    "scale_and_quant_fp4", "scale_and_quant_fp4_permute"))):
        from fastvideo.models.dits.minimax_h3_fusions.value_quant import pool_quantize_values
        return pool_quantize_values(value, valid)
    return None, None


def _prepare_received(api, layout, payload, scale, query_start, valid, *, prefix_values=False, direct_keys=False):
    template = payload.template if isinstance(payload, PeerSlices) else payload
    if (layout.tile == 64 and layout.n_tiles * layout.tile == layout.rows and template.shape[-1] == 128
            and not torch.is_grad_enabled() and not torch.compiler.is_compiling()
            and torch.cuda.get_device_capability(template.device)[0] >= 10
            and all(hasattr(api, name) for name in ("blockscaled_fp4_attn_sparse", "_zero_delta_s",
                                                    "scale_and_quant_fp4", "scale_and_quant_fp4_permute",
                                                    "scale_and_quant_fp4_transpose"))):
        from .sp_receive import unpack_pool_quantize_fp8
        inverse, qkv = layout.fp8_buffers(template)
        prefix_value, prefix = None, None
        # The dense branch consumes gathered values and the sparse branch
        # consumes native V codes, so raw BF16 V need not be materialized.
        if prefix_values:
            valid_index, prefix_inverse = layout.valid_keys_from_tiles(valid, api.HALF_N)
            prefix_value = torch.empty((1, valid_index.numel(), *qkv.shape[2:]),
                                       device=qkv.device, dtype=qkv.dtype)
            prefix = (prefix_inverse, prefix_value)
        # Native FP4 query codes supply the suffix; the raw query suffix is
        # used only for shape/dtype metadata. Keep a full block beyond the
        # aligned start to cover a dense prefix ending halfway through it.
        pooled, prepared_value, prepared_query = unpack_pool_quantize_fp8(
            payload, scale, inverse, valid, qkv, query_start,
            query_rows=min(query_start + base._BLOCK, layout.rows), prefix_values=prefix, store_keys=not direct_keys,
        )
        return qkv, prepared_query, pooled, prepared_value, prefix_value
    if isinstance(payload, PeerSlices):
        payload = payload.materialize()
    qkv, prepared_query = _prepare_tiles(api, layout, payload, scale, query_start)
    return qkv, prepared_query, None, None, None


def _pack_output(out, layout, world, rows):
    if isinstance(out, tuple):
        from .sp_output_pack import pack_split_fp8
        return pack_split_fp8(out, layout.untile, world, rows)
    return base._pack_seq_fp8(out, layout.untile, world, rows)


def attention_pipelined(attn, hidden_states, rotary_emb, meta, sp_group, chunks, pre_quantized=None):
    """Pipeline head groups so communication overlaps independent attention work."""
    world, rank = sp_group.world_size, sp_group.rank_in_group
    rows, heads, dim = hidden_states.shape[1], attn.num_attention_heads, attn.attention_head_dim
    assert heads % (world * chunks) == 0
    assert attn.to_gate_compress is not None and attn._gate_active()
    layouts = getattr(meta, '_h3_fp4_pipeline_layouts', None)
    if layouts is None or len(layouts) != chunks:
        layouts = [base._SPTileLayout(meta, rank, rows) for _ in range(chunks)]
        meta._h3_fp4_pipeline_layouts = layouts
    options = getattr(attn, '_vsa_fp4_options', {})
    early_sends = (options.get('sp_projection_overlap', False) and world == 2 and dim == 128
                   and hidden_states.is_cuda and hidden_states.dtype == torch.bfloat16
                   and not torch.is_grad_enabled() and not torch.compiler.is_compiling()
                   and base._shared_qkv_gate_scale(attn, hidden_states) is not None)
    if early_sends:
        from .sp_projection import project_and_exchange
        with base.STAGES.span('pipe_project_pack'):
            packets, gate_pre = project_and_exchange(attn, hidden_states, rotary_emb, sp_group.device_group,
                                                     chunks, pre_quantized)
    else:
        with base.STAGES.span('pipe_project_pack'):
            shared = base._shared_qkv_gate_projections(attn, hidden_states, pre_quantized=pre_quantized)
            if shared is None:
                projected = base._shared_input_projections((attn.to_q, attn.to_k, attn.to_v), hidden_states)
                gate_pre = None
            else:
                projected, gate_pre = shared
            query, key, value = (t.unflatten(-1, (heads, dim)) for t in projected)
            cos, sin = (t.to(query.dtype) for t in rotary_emb)
            payload, scale = pack_normalized_qkv_fp8(
                query[0],
                key[0],
                value[0],
                attn.norm_q.weight,
                attn.norm_k.weight,
                cos,
                sin,
                attn.norm_q.eps,
                attn.norm_k.eps,
                world,
                chunks,
            )
        packets = [_start_exchange(payload[chunk], scale[chunk], sp_group.device_group) for chunk in range(chunks)]
    with base.STAGES.span('gate_proj'):
        if gate_pre is None:
            gate, _ = attn.to_gate_compress(hidden_states)
        else:
            linear = attn.to_gate_compress
            gate = linear.quant_method.apply(linear, hidden_states, linear.bias, pre_quantized=gate_pre)
    vbs = meta.variable_block_sizes
    api = base._api()
    pending = []
    for chunk, layout in enumerate(layouts):
        prefix_mode, prefix_rows = base._prefix_settings(attn, meta, layout.n_tiles)
        direct_keys = bool(options.get('sp_direct_keys', False) and options.get('sp_pooled_key_mean', False)
                           and options.get('smooth_k') is True and prefix_rows and prefix_mode == 'fp8')
        with base.STAGES.span('pipe_receive'):
            received, received_scale = _finish_exchange(packets[chunk])
            qkv, prepared_query, pooled_qkv, prepared_value, prefix_value = _prepare_received(
                api, layout, received, received_scale, prefix_rows // base._BLOCK * base._BLOCK, vbs,
                prefix_values=bool(prefix_rows and prefix_mode == 'fp8' and options.get('smooth_k') is True),
                direct_keys=direct_keys,
            )
        q_t, k_t, v_t = qkv[0:1], qkv[1:2], qkv[2:3]
        logical = layout.n_tiles * layout.tile
        with base.STAGES.span('pipe_attention'):
            q_pool = pooled_qkv[0:1] if pooled_qkv is not None else _pool_tiles(q_t[:, :logical], vbs, layout.tile)
            k_pool = pooled_qkv[1:2] if pooled_qkv is not None else _pool_tiles(k_t[:, :logical], vbs, layout.tile)
            scores = torch.matmul(q_pool, k_pool.transpose(-2, -1)) / (dim**0.5)
            sparsity = 0.0 if attn._layer_idx in meta.dense_layers else meta.VSA_sparsity
            mask = _build_block_mask(scores, meta.num_prefix_tiles, sparsity, meta.exempt, meta.video_tile_spans,
                                     meta.span_sparsities)
            idx, num, valid, quad = base._tile_mask_to_fp4_blocks(api, mask, layout.tile, vbs)
            if pooled_qkv is None:
                pooled, prepared_value = _prepare_values(api, v_t, vbs, layout.tile, logical)
            else:
                pooled = pooled_qkv[2:3]
            prepared_keys = None
            if ((options.get('sp_pooled_key_mean', False) or prefix_value is not None)
                    and options.get('smooth_k') is True
                    and pooled_qkv is not None and prefix_rows and prefix_mode == 'fp8'):
                from .key_quant import quantize_smoothed_keys
                from .sp_key_mean import key_mean_from_tiles
                valid_index = layout.valid_key_index(valid, api.HALF_N)
                inverse = layout.valid_key_inverse(valid, api.HALF_N)
                mean = (key_mean_from_tiles(pooled_qkv[1:2], vbs, logical)
                        if options.get('sp_pooled_key_mean', False) else k_t.mean(dim=1, keepdim=True))
                key_packet = None
                if direct_keys and prefix_value is not None:
                    template = received.template if isinstance(received, PeerSlices) else received
                    tile_inverse, _ = layout.fp8_buffers(template)
                    key_packet = (received, received_scale, tile_inverse)
                prepared_keys = quantize_smoothed_keys(k_t, v_t, mean, inverse, valid_index.numel(),
                                                       prefix_value=prefix_value, received_keys=key_packet)
            out = base._fp4_fine_attention(api,
                                           q_t,
                                           k_t,
                                           v_t,
                                           idx,
                                           num,
                                           valid,
                                           quad,
                                           logical=logical,
                                           prefix_rows=prefix_rows,
                                           precision=prefix_mode,
                                           options=options,
                                           layout=layout, prepared_query=prepared_query, prepared_value=prepared_value,
                                           prepared_keys=prepared_keys, return_split=True)
        with base.STAGES.span('pipe_output_pack'):
            out_payload, out_scale = _pack_output(out, layout, world, rows)
        # Queue this output before computing the next head group. The pending
        # handles retain their inputs until the final wait and gate application.
        out_packet = _start_exchange(out_payload, out_scale, sp_group.device_group)
        with base.STAGES.span('gate_compress'):
            if pooled is None:
                pooled = _pool_tiles(v_t[:, :logical], vbs, layout.tile)
            compressed = torch.matmul(torch.softmax(scores, dim=-1), pooled)[0].to(qkv.dtype).contiguous()
        gathered = all_gather_tensor_async(compressed, sp_group.device_group, world)
        pending.append((out_packet, gathered))
    full_gate = gate[0].view(rows, heads, dim)
    method = _output_quant_method(attn)
    if method is None:
        full_output = torch.empty_like(full_gate)
    else:
        width = heads * dim
        codes = torch.empty((rows, width // 2), device=gate.device, dtype=torch.uint8)
        scales = torch.zeros((((rows + 127) // 128) * 128, width // 16), device=gate.device, dtype=torch.uint8)
    for chunk, (packet, gathered) in enumerate(pending):
        with base.STAGES.span('pipe_output_wait_gate'):
            received, received_scale = _finish_exchange(packet)
            compression = gathered.wait()
            if method is None:
                unpack_fp8_gate(received, received_scale, compression, full_gate, layouts[chunk].local_row_tile,
                                 head_chunk=chunk, head_chunks=chunks, output=full_output)
            else:
                unpack_fp8_gate(received, received_scale, compression, full_gate, layouts[chunk].local_row_tile,
                                 method.x_global_sf, head_chunk=chunk, head_chunks=chunks, fp4_output=(codes, scales))
    if method is not None:
        return gate, (codes, scales, method.x_global_sf)
    return full_output.flatten(1, 2).unsqueeze(0)
