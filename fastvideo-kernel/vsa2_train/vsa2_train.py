"""VSA2 attention on sm_100a or sm_103a (B200 / GB200, B300), for training and for FastPAN2 sampling: the forward
and the backward of the joint [text; video] attention as torch.library custom ops with a registered backward, built
from solution/ on first use.

The layout is VSA-H3's at 128-token tiles (reference.py defines it exactly):
  - q, k, v, gate: [B, H, S, 128] bf16, contiguous, S = N * 128. The first P = ceil(text / 128) blocks hold the text
    in order, the last one padded at its end; then the N_v video tiles of 4 x 4 x 8 tokens, tiles in raster order of
    the tile grid and tokens in raster order of the tile, the latent padded up to whole tiles.
  - Padding rows of q, k, v and gate are zero, and gate is zero on the text rows.
  - t, h, w: the video latent's extent in tokens; text: the text tokens; topk: key tiles per video query tile on
    average (VSA-H3's compute_topk: ceil(0.1 * N_v) at sparsity 0.9).

The router takes no gradient (the paper's router is gradient-free), so `selection` is returned without one.

The kernels are custom ops (``torch.ops.vsa2_train.forward`` and ``.backward``) so that torch.compile, fullgraph
included, keeps them as opaque calls: their fake implementations give the outputs' shapes, and the geometry may be
symbolic. Load this file once per process (as ``vsa2_train``): the ops register when it is imported.
"""

import functools
import importlib.util
import os
import sys

import torch
from torch import Tensor

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = 128


def _own(path, name):
    """The module at path, imported under a name of this kernel's own. fastvideo-kernel's other kernels
    (wan22_vae_decode) keep tools and references of the same file names; a bare import would hand whichever loaded
    first to both."""
    name = f"vsa2_train_{name}"
    module = sys.modules.get(name)
    if module is None:
        module_spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[name] = module
        try:
            module_spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    return module


@functools.lru_cache(maxsize=None)
def _module():
    """The extension built from solution/, once per process. Builds are kept in VSA2_TRAIN_BUILD_DIR (default
    ~/.cache/fastvideo/vsa2_train), one directory per local rank, and reused while the sources and flags match."""
    loader = _own(os.path.join(HERE, "tools", "loader.py"), "loader")
    base = os.environ.get("VSA2_TRAIN_BUILD_DIR", os.path.expanduser("~/.cache/fastvideo/vsa2_train"))
    build_dir = os.path.join(base, f"rank{os.environ.get('LOCAL_RANK', '0')}")
    os.makedirs(base, exist_ok=True)
    return loader.load(os.path.join(HERE, "solution"), build_dir=build_dir, chdir=False).module


@functools.lru_cache(maxsize=64)
def _valid_rows(t, h, w, text, device):
    """[S] bool: the rows that hold a token, as reference.row_valid defines them."""
    reference = _own(os.path.join(HERE, "reference.py"), "reference")
    lay = reference.Layout(text, (t, h, w))
    return reference.row_valid(lay, device).reshape(-1)


def _video_blocks(seq: int, text: int) -> int:
    """N_v of a sequence of ``seq`` rows: every block that is not text (symbolic sizes welcome)."""
    return seq // ROWS - (text + ROWS - 1) // ROWS


@torch.library.custom_op("vsa2_train::forward", mutates_args=())
def _forward(q: Tensor, k: Tensor, v: Tensor, gate: Tensor, t: int, h: int, w: int, text: int,
             topk: int) -> tuple[Tensor, Tensor, Tensor]:
    """(out, selection, lse) of one VSA2 forward."""
    q, k, v, gate = (x.contiguous() for x in (q, k, v, gate))
    b, heads, s, _ = q.shape
    nv = _video_blocks(s, text)
    out = q.new_empty(q.shape)
    selection = q.new_empty((b, heads, nv, nv), dtype=torch.float32)
    lse = q.new_empty((b, heads, s), dtype=torch.float32)
    _module().forward(q, k, v, gate, t, h, w, text, topk, out, selection, lse)
    return out, selection, lse


@_forward.register_fake
def _forward_fake(q, k, v, gate, t, h, w, text, topk):
    b, heads, s, _ = q.shape
    nv = _video_blocks(s, text)
    return (q.new_empty(q.shape), q.new_empty((b, heads, nv, nv), dtype=torch.float32),
            q.new_empty((b, heads, s), dtype=torch.float32))


@torch.library.custom_op("vsa2_train::backward", mutates_args=())
def _backward(q: Tensor, k: Tensor, v: Tensor, gate: Tensor, out: Tensor, lse: Tensor, selection: Tensor,
              dout: Tensor, t: int, h: int, w: int, text: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """(dq, dk, dv, dgate) of the gradient ``dout`` of ``out``."""
    q, k, v, gate, out = (x.contiguous() for x in (q, k, v, gate, out))
    # The kernel reads dout only where a row holds a token; the contract is zero elsewhere.
    valid = _valid_rows(t, h, w, text, q.device)
    dout = dout.to(q.dtype).masked_fill(~valid[None, None, :, None], 0).contiguous()
    dq, dk, dv, dgate = (q.new_empty(q.shape) for _ in range(4))
    _module().backward(q, k, v, gate, out, lse, selection, dout, t, h, w, text, dq, dk, dv, dgate)
    return dq, dk, dv, dgate


@_backward.register_fake
def _backward_fake(q, k, v, gate, out, lse, selection, dout, t, h, w, text):
    return tuple(q.new_empty(q.shape) for _ in range(4))


def _setup_context(ctx, inputs, output):
    q, k, v, gate, t, h, w, text, _topk = inputs
    out, selection, lse = output
    ctx.save_for_backward(q, k, v, gate, out, lse, selection)
    ctx.geometry = (t, h, w, text)


def _backward_rule(ctx, dout, _dselection, _dlse):
    q, k, v, gate, out, lse, selection = ctx.saved_tensors
    dq, dk, dv, dgate = torch.ops.vsa2_train.backward(q, k, v, gate, out, lse, selection, dout, *ctx.geometry)
    return dq, dk, dv, dgate, None, None, None, None, None


_forward.register_autograd(_backward_rule, setup_context=_setup_context)


def vsa2_attention(q, k, v, gate, t, h, w, text, topk):
    """VSA2 attention over the joint [text; video] layout: (out, selection). out carries the gradient to q, k, v and
    gate; selection [B, H, N_v, N_v] float32 is 1 for each video tile pair the router kept. The geometry may be
    symbolic under torch.compile."""
    out, selection, _ = torch.ops.vsa2_train.forward(q, k, v, gate, t, h, w, text, topk)
    return out, selection
