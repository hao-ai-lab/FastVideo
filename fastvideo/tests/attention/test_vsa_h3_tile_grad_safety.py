# SPDX-License-Identifier: Apache-2.0
"""CPU regression: a grad-tracking VSA-H3 ``tile()`` must not leave the step's
graph on the builder-owned tile buffer.

``MiniMaxH3VSAMetadataBuilder`` owns ``_tile_buf_holder`` and one builder serves
the whole training run, so every step's metadata references the same holder.
``tile()`` writes one shared buffer in place from every VSA layer, so if the
holder keeps the *graph-tracked* tiled tensor, that single tensor anchors all
of the step's in-place autograd edges -- and, through their ``next_functions``,
the activations they saved -- for the rest of training. This is the VSA-H3
counterpart of the Wan tile-cache OOM (#1423, fixed by #1434/#1444); on the H3
backend the buffer is owned by the builder rather than by per-step metadata, so
the retention spans steps instead of one denoising pass.

Four tests, one property each: the holder never keeps graph state across
grad-tracking steps; the tiled rows still round-trip; inference still reuses the
buffer; and a builder alternating grad and non-grad steps corrupts neither.

The holder invariant is implementation-agnostic -- it passes both for a fresh
buffer when gradients are tracked (nothing is cached) and for a reused buffer
that stores only a detached alias -- so it does not pin either fix. What the
first test does pin is that the fix may not be achieved by detaching the
*returned* tensor, which would silently starve the model of gradients.
"""

import torch

from fastvideo.attention.backends.video_sparse_attn_h3 import (_TILE_ELEMS, MiniMaxH3VSAImpl,
                                                               MiniMaxH3VSAMetadataBuilder)

_SPEC = dict(raw_latent_shape=(8, 8, 12), patch_size=(1, 2, 2), prefix_segments=(7, 5, 3))


def _build(builder=None, step=0):
    builder = builder if builder is not None else MiniMaxH3VSAMetadataBuilder()
    # A fresh metadata per step, all sharing the builder's holder.
    return builder.build(
        current_timestep=step,
        raw_latent_shape=_SPEC["raw_latent_shape"],
        patch_size=_SPEC["patch_size"],
        VSA_sparsity=0.9,
        prefix_segments=_SPEC["prefix_segments"],
        device=torch.device("cpu"),
        tile_size=_TILE_ELEMS,
    ), builder


def _impl():
    return MiniMaxH3VSAImpl(num_heads=2, head_size=8, causal=False, softmax_scale=8**-0.5)


def _assert_holder_is_graph_free(holder, where):
    assert holder is not None, where
    held = holder.buffer
    # Caching nothing is fine; caching something that carries autograd state is
    # what survives the step and anchors the graph.
    assert held is None or (not held.requires_grad and held.grad_fn is None), (
        f"{where}: the builder-owned tile buffer holds a graph-tracked tensor; the "
        "holder outlives the step, so it anchors every VSA layer's in-place "
        "autograd edge (and the activations they saved) for the rest of training")


def test_grad_tracking_tile_keeps_the_holder_graph_free_across_steps():
    """Consecutive grad-tracking steps through one builder leave no graph behind.

    Step 0 additionally checks that the tiled tensor is genuinely connected to
    the input, so the invariant cannot be met by detaching the returned tensor.
    """
    _, builder = _build()
    impl = _impl()

    for step in range(3):
        meta, _ = _build(builder, step=step)
        x = torch.randn(1, meta.total_seq_length, 2, 8, requires_grad=True)
        tiled = impl.tile(x, meta)

        _assert_holder_is_graph_free(builder._tile_buf_holder, f"after grad step {step}")

        if step == 0:
            # The tiled tensor itself is in the graph -- the VSA layers consume
            # it and gradients must reach ``x``. A real backward, not just grad_fn.
            assert tiled.requires_grad and tiled.grad_fn is not None
            tiled.sum().backward()
            assert x.grad is not None, "gradients must reach the tiled input"
            assert bool((x.grad > 0).all()), "every packed row must be written at least once"


def test_grad_tracking_tile_still_populates_the_buffer():
    """The guard must not cost correctness: the tiled rows still round-trip."""
    meta, _ = _build()
    x = torch.randn(1, meta.total_seq_length, 2, 8, requires_grad=True)
    tiled = _impl().tile(x, meta)
    assert torch.equal(tiled[:, meta.untile_combined_index], x)


def test_inference_tile_still_reuses_the_buffer():
    """The fix must not cost the inference optimization.

    With grad tracking off, the builder-owned buffer is still allocated once and
    reused across calls -- which here means across the 50 VSA layers that share
    one metadata.
    """
    meta, _ = _build()
    impl = _impl()
    x = torch.randn(1, meta.total_seq_length, 2, 8)

    with torch.no_grad():
        first = impl.tile(x, meta)
        assert meta.tile_buf_holder.buffer is not None, ("inference must still cache into the builder-owned buffer")
        second = impl.tile(x, meta)

    assert first is second, "the builder-owned buffer must still be reused when gradients are not tracked"
    assert torch.equal(second[:, meta.untile_combined_index], x)


def test_grad_and_inference_steps_share_one_builder_safely():
    """Grad-tracking and non-grad steps share a builder; neither may corrupt the other.

    Covers what the single-mode tests cannot: after a grad step has run, an
    inference step must still reuse the builder's buffer, and a grad step after
    inference must not overwrite that cached buffer with a graph-tracked one.
    (The grad-step invariant itself is asserted by the first test.)
    """
    builder = MiniMaxH3VSAMetadataBuilder()
    impl = _impl()

    # A grad step first, on a builder that has never cached anything.
    meta_grad, _ = _build(builder, step=0)
    xg = torch.randn(1, meta_grad.total_seq_length, 2, 8, requires_grad=True)
    impl.tile(xg, meta_grad)

    # Then an inference step: the buffer is allocated and reused as usual.
    meta_inf, _ = _build(builder, step=1)
    x = torch.randn(1, meta_inf.total_seq_length, 2, 8)
    with torch.no_grad():
        first = impl.tile(x, meta_inf)
        second = impl.tile(x, meta_inf)
    assert first is second, "inference must still reuse the buffer after a grad step"
    _assert_holder_is_graph_free(builder._tile_buf_holder, "after inference on a builder that saw a grad step")

    # Reverse order: a grad step after inference must not overwrite the cached
    # (grad-free) inference buffer with a graph-tracked one.
    meta_grad2, _ = _build(builder, step=2)
    xg2 = torch.randn(1, meta_grad2.total_seq_length, 2, 8, requires_grad=True)
    impl.tile(xg2, meta_grad2)
    _assert_holder_is_graph_free(builder._tile_buf_holder, "after a grad step that followed inference")


if __name__ == "__main__":
    test_grad_tracking_tile_keeps_the_holder_graph_free_across_steps()
    test_grad_tracking_tile_still_populates_the_buffer()
    test_inference_tile_still_reuses_the_buffer()
    test_grad_and_inference_steps_share_one_builder_safely()
    print("all VSA-H3 tile grad-safety checks passed")
