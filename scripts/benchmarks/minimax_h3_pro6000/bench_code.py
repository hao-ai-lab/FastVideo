"""Per-component timing of one MiniMax-H3 block on the current GPU. Imported inside the Modal container."""
import json
import math
import time

import torch
import torch.nn.functional as F

HID, HEADS, HD, FFN = 5376, 56, 128, 14336


def timed(fn, iters=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return round(start.elapsed_time(end) / iters, 3)


def smooth_tokens(prefix_len, shape, dev, gen, noise=0.5):
    """Token features with spatial-temporal correlation (proxy for real q/k), packed [prefix | video]."""
    t, h, w = shape
    low = torch.randn(1, HEADS * HD, max(2, t // 6), max(2, h // 6), max(2, w // 6), device=dev, generator=gen)
    vid = F.interpolate(low, size=(t, h, w), mode="trilinear", align_corners=False)
    vid = vid.reshape(HEADS, HD, -1).permute(2, 0, 1)
    vid = vid + noise * torch.randn(vid.shape, device=dev, generator=gen)
    pre = torch.randn(prefix_len, HEADS, HD, device=dev, generator=gen)
    return torch.cat([pre, vid]).to(torch.bfloat16)[None]  # [1, L, H, D]


def run(shapes):
    from flashinfer import SfLayout, mm_fp4, nvfp4_quantize
    from fastvideo.attention.backends.video_sparse_attn_h3 import (_build_block_mask, _h3_tile_geometry, _pool_tiles)
    from fastvideo_kernel.block_sparse_attn import block_sparse_attn as bsa64
    from fastvideo.models.dits.minimax_h3_fusions import (fused_qknorm_rope, fused_residual_gate_rmsnorm_modulate,
                                                          fused_rmsnorm_modulate, minimax_h3_swiglu)
    import attn_qat_infer.api as fa
    from attn_qat_infer.api import BLOCK_M, sageattn_blackwell, sageattn_blackwell_sparse

    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(0)
    report = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__}
    unit = torch.tensor(1.0, device=dev)

    def fp4_weight(n, k):
        w = torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02
        gsf = (448 * 6) / w.float().abs().max()
        wq, ws = nvfp4_quantize(w, gsf, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
        return w, wq, ws, (1.0 / gsf).float()

    for name, prefix_segments, video_shape in shapes:
        prefix_segments = tuple(prefix_segments)
        video_shape = tuple(video_shape)
        L = sum(prefix_segments) + math.prod(video_shape)
        r = {"tokens": L}
        torch.cuda.empty_cache()

        # ---------------- linears ----------------
        lin = {}
        for lname, n, k in (("qkv_one", 7168, HID), ("out", HID, 7168), ("fc_in", 2 * FFN, HID),
                            ("fc_out", HID, FFN)):
            w, wq, ws, alpha = fp4_weight(n, k)
            x = torch.randn(L, k, device=dev, dtype=torch.bfloat16)
            xq, xs = nvfp4_quantize(x, unit, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
            entry = {"quant_ms": timed(lambda: nvfp4_quantize(x, unit, sfLayout=SfLayout.layout_128x4,
                                                              do_shuffle=False))}
            for backend in ("cudnn", "cutlass", "auto"):
                try:
                    entry[f"mm_{backend}_ms"] = timed(
                        lambda: mm_fp4(xq, wq.T, xs, ws.T, alpha, torch.bfloat16, None, backend=backend))
                except Exception as exc:  # noqa: BLE001
                    entry[f"mm_{backend}_ms"] = f"ERR {type(exc).__name__}: {str(exc)[:120]}"
            entry["bf16_ms"] = timed(lambda: F.linear(x, w))
            entry["tflops_fp4_auto"] = (round(2 * L * n * k / entry["mm_auto_ms"] / 1e9, 1)
                                        if isinstance(entry["mm_auto_ms"], float) else None)
            lin[lname] = entry
            del w, wq, ws, x, xq, xs
        r["linear"] = lin

        # ---------------- attention ----------------
        geom = _h3_tile_geometry(prefix_segments, video_shape, dev, (4, 4, 4))
        _, vbs, untile, n_prefix, n_video = geom
        n_tiles = vbs.numel()
        Lpad = n_tiles * 64
        att = {"tiles64": n_tiles, "prefix_tiles": n_prefix}
        prefix_len = sum(prefix_segments)
        q = smooth_tokens(prefix_len, video_shape, dev, gen)
        k = (q + 0.5 * torch.randn(q.shape, device=dev, generator=gen, dtype=torch.bfloat16)).to(torch.bfloat16)
        v = torch.randn(q.shape, device=dev, generator=gen, dtype=torch.bfloat16)
        g = torch.randn(q.shape, device=dev, generator=gen, dtype=torch.bfloat16) * 0.1

        def tile(x):
            buf = torch.zeros((1, Lpad, HEADS, HD), device=dev, dtype=x.dtype)
            buf[:, untile] = x
            return buf

        qt, kt, vt, gt = tile(q), tile(k), tile(v), tile(g)
        att["tile_scatter4_gather1_ms"] = timed(lambda: (tile(q), tile(k), tile(v), tile(g), qt[:, untile]))

        def vsa_full(sparsity=0.8, with_gate=True):
            qp = _pool_tiles(qt, vbs, 64)
            kp = _pool_tiles(kt, vbs, 64)
            scores = torch.matmul(qp, kp.transpose(-2, -1)) / HD**0.5
            mask = _build_block_mask(scores, n_prefix, sparsity, True, ((n_prefix, n_prefix + n_video), ), (sparsity, ))
            out, _ = bsa64(qt.transpose(1, 2).contiguous(), kt.transpose(1, 2).contiguous(),
                           vt.transpose(1, 2).contiguous(), mask, vbs)
            out = out.transpose(1, 2).contiguous()
            if with_gate:
                vp = _pool_tiles(vt, vbs, 64)
                oc = torch.matmul(torch.softmax(scores, dim=-1), vp).permute(0, 2, 1, 3).to(out.dtype)
                out = (out.view(1, n_tiles, 64, HEADS, HD) + oc.unsqueeze(2) * gt.view(1, n_tiles, 64, HEADS, HD)).view(
                    1, Lpad, HEADS, HD)
            return out, mask

        _, mask = vsa_full()
        att["tile_density"] = round(mask.float().mean().item(), 4)
        att["vsa_triton_total_ms"] = timed(vsa_full)
        qb, kb, vb = (x.transpose(1, 2).contiguous() for x in (qt, kt, vt))
        att["vsa_triton_kernel_only_ms"] = timed(lambda: bsa64(qb, kb, vb, mask, vbs))

        # dense FP4 on the unpadded packed sequence
        qd, kd, vd = (x.transpose(1, 2).contiguous() for x in (q, k, v))
        att["sage3_dense_fp4_ms"] = timed(lambda: sageattn_blackwell(qd, kd, vd), iters=3)
        # same with delta_s cached + per_block_mean=False (no 9.5 GB memset)
        zero_ds = {}

        def sage_nods(qx, kx, vx):
            QL, KL = qx.size(2), kx.size(2)
            qx, kx, vx = (F.pad(x, (0, 0, 0, (BLOCK_M - x.size(2) % BLOCK_M) % BLOCK_M)).contiguous() for x in (qx, kx, vx))
            key = (qx.shape[0], qx.shape[1], kx.shape[2])
            if key not in zero_ds:
                zero_ds[key] = torch.zeros((qx.shape[0], qx.shape[1], 1, kx.shape[2]), device=dev, dtype=torch.float32)
            ql, kl, vl = fa.scale_and_quant_fp4(qx), fa.scale_and_quant_fp4_permute(kx), fa.scale_and_quant_fp4_transpose(vx)
            return fa.blockscaled_fp4_attn(ql, kl, vl, zero_ds[key], KL, False, False, True, True, None)[0][:, :, :QL]

        att["sage3_dense_fp4_no_deltas_ms"] = timed(lambda: sage_nods(qd, kd, vd), iters=3)
        out_a = sageattn_blackwell(qd, kd, vd)
        out_b = sage_nods(qd, kd, vd)
        att["no_deltas_max_abs_diff"] = (out_a.float() - out_b.float()).abs().max().item()
        Lq = qd.shape[2]
        att["dense_fp4_tflops"] = round(4 * Lq * Lq * HD * HEADS / att["sage3_dense_fp4_no_deltas_ms"] / 1e9, 1)
        if L < 30000:
            att["sdpa_bf16_dense_ms"] = timed(lambda: F.scaled_dot_product_attention(qd, kd, vd), iters=3)

        # sparse FP4 on 128x128 blocks with 64x64 quadrant masks (exact VSA tile-64 semantics)
        nt2 = n_tiles + (n_tiles % 2)
        qs, ks, vs = (F.pad(x.transpose(1, 2), (0, 0, 0, nt2 * 64 - Lpad)).contiguous() for x in (qt, kt, vt))
        q2k_idx, q2k_num, kvv, quad = fa.vsa_tile_mask_to_fp4_blocks(mask, 64, vbs, validate=True)
        att["block128_density"] = round(q2k_num.float().mean().item() / (nt2 // 2), 4)
        att["sparse_fp4_quad_ms"] = timed(lambda: sageattn_blackwell_sparse(qs, ks, vs, q2k_idx, q2k_num, kvv, quad))
        att["mask_to_blocks_ms"] = timed(lambda: fa.vsa_tile_mask_to_fp4_blocks(mask, 64, vbs))
        o_fp4 = sageattn_blackwell_sparse(qs, ks, vs, q2k_idx, q2k_num, kvv, quad)[:, :, :Lpad]
        o_tri, _ = bsa64(qb, kb, vb, mask, vbs)
        rows = untile  # valid (non-pad) rows of the tile buffer
        diff = (o_fp4[:, :, rows].float() - o_tri[:, :, rows].float())
        att["quad_vs_triton_rel_l2"] = round((diff.norm() / o_tri[:, :, rows].float().norm()).item(), 4)
        od = sageattn_blackwell(qd, kd, vd)
        from torch.nn.attention import SDPBackend, sdpa_kernel
        if L < 30000:
            ref_dense = F.scaled_dot_product_attention(qd, kd, vd)
            att["dense_fp4_vs_bf16_rel_l2"] = round(((od.float() - ref_dense.float()).norm() /
                                                     ref_dense.float().norm()).item(), 4)
        att["quad_finite"] = bool(torch.isfinite(o_fp4[:, :, rows]).all())
        r["attention"] = att
        del q, k, v, g, qt, kt, vt, gt, qd, kd, vd, qs, ks, vs, qb, kb, vb, out_a, out_b, zero_ds, o_fp4, o_tri, od

        # ---------------- elementwise ----------------
        ew = {}
        x = torch.randn(1, L, HID, device=dev, dtype=torch.bfloat16)
        br = torch.randn_like(x)
        nw = torch.randn(HID, device=dev, dtype=torch.bfloat16)
        tab = [torch.randn(3, HID, device=dev, dtype=torch.bfloat16) * 0.1 for _ in range(6)]
        idx = torch.randint(0, 3, (L,), device=dev)
        norm = torch.nn.RMSNorm(HID, eps=1e-6, device=dev, dtype=torch.bfloat16)

        def eager_mod():
            n1 = norm(x) * (1.0 + tab[1].index_select(0, idx)) + tab[0].index_select(0, idx)
            h = x + tab[2].index_select(0, idx) * br
            n2 = norm(h) * (1.0 + tab[4].index_select(0, idx)) + tab[3].index_select(0, idx)
            out = h + tab[5].index_select(0, idx) * br
            return n1, n2, out

        def fused_mod():
            n1 = fused_rmsnorm_modulate(x, nw, tab[1], tab[0], idx, 1e-6)
            h, n2 = fused_residual_gate_rmsnorm_modulate(x, br, tab[2], nw, tab[4], tab[3], idx, 1e-6)
            out = h + tab[5].index_select(0, idx) * br
            return n1, n2, out

        with torch.no_grad():
            ew["modulate_eager_ms"] = timed(eager_mod)
            ew["modulate_fused_ms"] = timed(fused_mod)
            ew["modulate_compiled_ms"] = timed(torch.compile(eager_mod))
            packed = torch.randn(1, L, 2 * FFN, device=dev, dtype=torch.bfloat16)

            def eager_swiglu():
                a, b = packed.chunk(2, dim=-1)
                return a * F.silu(b)

            ew["swiglu_eager_ms"] = timed(eager_swiglu)
            ew["swiglu_fused_ms"] = timed(lambda: minimax_h3_swiglu(packed))
            del packed
            qq = torch.randn(1, L, HEADS, HD, device=dev, dtype=torch.bfloat16)
            cos = torch.randn(L, 96, device=dev, dtype=torch.bfloat16)
            sin = torch.randn(L, 96, device=dev, dtype=torch.bfloat16)
            qn = torch.nn.RMSNorm(HD, eps=1e-6, device=dev, dtype=torch.bfloat16)

            def eager_rope():
                outs = []
                for t in (qq, qq):
                    t = qn(t)
                    rot, pas = t[..., :96], t[..., 96:]
                    c, s = cos[None, :, None, :], sin[None, :, None, :]
                    a, b = rot.chunk(2, dim=-1)
                    outs.append(torch.cat((rot * c + torch.cat((-b, a), -1) * s, pas), -1).contiguous())
                return outs

            ew["qknorm_rope_eager_ms"] = timed(eager_rope)
            ew["qknorm_rope_fused_ms"] = timed(
                lambda: (fused_qknorm_rope(qq, qn.weight, cos, sin, 1e-6), fused_qknorm_rope(qq, qn.weight, cos, sin, 1e-6)))
        r["elementwise"] = ew
        report[name] = r
        print("BENCH", name, json.dumps(r), flush=True)
    return report


def check_tile64():
    """Exact semantics check of the quadrant path on small H3-style layouts."""
    from fastvideo.attention.backends.video_sparse_attn_h3 import _build_block_mask, _h3_tile_geometry, _pool_tiles
    import attn_qat_infer.api as fa
    from attn_qat_infer.api import sageattn_blackwell, sageattn_blackwell_sparse

    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(1)
    out = {}
    cases = [("odd_tiles", (77, 46), (9, 6, 10), 64), ("even_tiles", (64, 40), (8, 8, 8), 64),
             ("tile256", (100, 70), (8, 8, 16), 256)]
    for name, prefix, vshape, tt in cases:
        H = 4
        shape = {64: (4, 4, 4), 256: (4, 8, 8)}[tt]
        _, vbs, untile, n_prefix, n_video = _h3_tile_geometry(prefix, vshape, dev, shape)
        n_tiles = vbs.numel()
        Lpad = n_tiles * tt
        L = sum(prefix) + math.prod(vshape)
        q = torch.randn(1, L, H, HD, device=dev, generator=gen, dtype=torch.bfloat16)
        k = torch.randn(1, L, H, HD, device=dev, generator=gen, dtype=torch.bfloat16)
        v = torch.randn(1, L, H, HD, device=dev, generator=gen, dtype=torch.bfloat16)

        def tile(x):
            buf = torch.zeros((1, Lpad, H, HD), device=dev, dtype=x.dtype)
            buf[:, untile] = x
            return buf.transpose(1, 2).contiguous()  # BHSD

        qt, kt, vt = tile(q), tile(k), tile(v)
        scores = torch.matmul(_pool_tiles(qt.transpose(1, 2), vbs, tt), _pool_tiles(kt.transpose(1, 2), vbs, tt).transpose(-2, -1))
        mask = _build_block_mask(scores, n_prefix, 0.8, True, ((n_prefix, n_prefix + n_video), ), (0.8, ))
        q2k_idx, q2k_num, kvv, quad = fa.vsa_tile_mask_to_fp4_blocks(mask, tt, vbs, validate=True)
        Lk = q2k_idx.shape[2] * 128
        pad = lambda x: F.pad(x, (0, 0, 0, Lk - Lpad)).contiguous()
        o = sageattn_blackwell_sparse(pad(qt), pad(kt), pad(vt), q2k_idx, q2k_num, kvv, quad)[:, :, :Lpad]
        tok_tile = torch.arange(n_tiles, device=dev).repeat_interleave(tt)
        tok_valid = torch.zeros(Lpad, dtype=torch.bool, device=dev)
        tok_valid[untile] = True
        tm = mask[:, :, tok_tile][:, :, :, tok_tile] & tok_valid[None, None, None, :]
        ref = F.scaled_dot_product_attention(qt.float(), kt.float(), vt.float(), attn_mask=tm)
        rows = untile
        err = (o.float() - ref)[:, :, rows]
        # FP4 noise floor: dense FP4 vs dense fp32 on the same valid tokens
        qv, kv_, vv = (x[:, :, rows].contiguous() for x in (qt, kt, vt))
        od = sageattn_blackwell(qv, kv_, vv)
        rd = F.scaled_dot_product_attention(qv.float(), kv_.float(), vv.float())
        out[name] = {
            "tiles": n_tiles, "density": round(mask.float().mean().item(), 3),
            "rel_l2": round((err.norm() / ref[:, :, rows].norm()).item(), 4),
            "dense_fp4_rel_l2_floor": round(((od.float() - rd).norm() / rd.norm()).item(), 4),
            "cosine": round(F.cosine_similarity(o[:, :, rows].float().flatten(), ref[:, :, rows].flatten(), dim=0).item(), 5),
            "finite": bool(torch.isfinite(o[:, :, rows]).all()),
        }
        if tt == 64:
            from fastvideo_kernel.block_sparse_attn import block_sparse_attn as bsa64
            ot, _ = bsa64(qt, kt, vt, mask, vbs)
            out[name]["triton_bf16_rel_l2"] = round(((ot.float() - ref)[:, :, rows].norm() / ref[:, :, rows].norm()).item(), 4)
        print("CHECK", name, json.dumps(out[name]), flush=True)
    return out


def density_study(prefix_segments=(256, 810), video_shape=(72, 24, 42)):
    """Block densities each kernel granularity would compute, and FP4 sparse timing at 0.8 / 0.9."""
    from fastvideo.attention.backends.video_sparse_attn_h3 import _build_block_mask, _h3_tile_geometry, _pool_tiles
    import attn_qat_infer.api as fa

    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(0)
    _, vbs, untile, n_prefix, n_video = _h3_tile_geometry(tuple(prefix_segments), tuple(video_shape), dev, (4, 4, 4))
    n_tiles = vbs.numel()
    Lpad = n_tiles * 64
    q = smooth_tokens(sum(prefix_segments), tuple(video_shape), dev, gen)
    k = (q + 0.5 * torch.randn(q.shape, device=dev, generator=gen, dtype=torch.bfloat16)).to(torch.bfloat16)
    v = torch.randn(q.shape, device=dev, generator=gen, dtype=torch.bfloat16)

    def tile(x):
        buf = torch.zeros((1, Lpad, HEADS, HD), device=dev, dtype=x.dtype)
        buf[:, untile] = x
        return buf

    qt, kt, vt = tile(q), tile(k), tile(v)
    scores = torch.matmul(_pool_tiles(qt, vbs, 64), _pool_tiles(kt, vbs, 64).transpose(-2, -1))
    nt2 = n_tiles + n_tiles % 2
    rows = nt2 * 64
    qs, ks, vs = (F.pad(x, (0, 0, 0, 0, 0, rows - Lpad)).contiguous() for x in (qt, kt, vt))
    out = {}
    for sparsity in (0.8, 0.9):
        mask = _build_block_mask(scores, n_prefix, sparsity, True, ((n_prefix, n_prefix + n_video), ), (sparsity, ))
        m = F.pad(mask, (0, nt2 - n_tiles, 0, nt2 - n_tiles), value=False)
        B, H = m.shape[:2]
        d = {"tile64x64": m.float().mean().item()}
        q2 = m.view(B, H, nt2 // 2, 2, nt2)  # query pairs
        d["q128_k64"] = q2.any(3).float().mean().item()
        k2 = m.view(B, H, nt2, nt2 // 2, 2)
        d["q64_k128"] = k2.any(4).float().mean().item()
        d["q128_k128"] = m.view(B, H, nt2 // 2, 2, nt2 // 2, 2).any(5).any(3).float().mean().item()
        idx, num, kvv, quad = fa.vsa_tile_mask_to_fp4_blocks(mask, 64, vbs)
        def bshd():
            qh, kh, vh = (x.transpose(1, 2) for x in (qs, ks, vs))
            ds = fa._zero_delta_s(1, HEADS, rows, dev)
            return fa.blockscaled_fp4_attn_sparse(fa.scale_and_quant_fp4(qh), fa.scale_and_quant_fp4_permute(kh),
                                                  fa.scale_and_quant_fp4_transpose(vh), ds, rows, idx, num, kvv, quad,
                                                  False, True, True, None)[0]

        d["sparse_fp4_quad_ms"] = timed(bshd, iters=3)
        out[str(sparsity)] = {key: round(val, 4) for key, val in d.items()}
        print("DENSITY", sparsity, json.dumps(out[str(sparsity)]), flush=True)
    out["dense_fp4_bshd_ms"] = timed(lambda: fa.sageattn_blackwell(*(x.transpose(1, 2) for x in (qs, ks, vs))), iters=2)
    return out
