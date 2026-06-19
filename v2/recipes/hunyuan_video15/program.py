"""HunyuanVideo 1.5 t2v program: text_encode → diffusion_denoise → vae_decode.

Same inline shape as the Wan/Cosmos t2v programs — the arch specifics (two text embeds, the image-embed
list, the 33ch i2v concat) live entirely in the ``HunyuanVideo15DiT`` adapter, so the node graph differs
only in the text-encode node, which must produce BOTH text embeddings.

HunyuanVideo 1.5 conditions on two encoders (faithful to ``HunyuanVideo15Pipeline``'s
``TextEncodingStage(text_encoders=[text_encoder, text_encoder_2])``):
  * ``text_encoder``   = Qwen2.5-VL  -> ``hidden_states[-3]`` cropped of the 108 template tokens (3584-d)
  * ``text_encoder_2`` = ByT5/Glyph  -> ``last_hidden_state`` of the quoted-glyph text (1472-d), or a
    zero-length embedding when the prompt carries no quoted text.

The text-encode node packs them as a ``(qwen, byt5)`` tuple in the slot; ``HunyuanVideo15DiT`` unpacks
that into the two-element ``encoder_hidden_states`` list. On the CPU toy backend there is a single
``text_encoder`` component (``ToyTextEncoder``) and no ``text_encoder_2``: the node then writes the plain
toy embedding (a single numpy array), which ``ToyDiT`` consumes directly — so the program CPU-verifies
unchanged. The feature cache still applies to the Qwen path via ``cached_text_encode``.

BRINGUP (i2v): the i2v path adds a VAE-encode node that writes the 33-channel first-frame conditioning
latent into ``i2v_cond`` and zero image embeds into ``i2v_img_embeds`` (the slots the loop already reads);
``Hy15ImageEncodingStage`` builds those (image_embeds=zeros[1,729,1152]; image_latent=[expanded VAE
first-frame ⊕ mask] -> 33ch). No CLIP/SigLIP vision encoder is loaded (image_embeds are placeholder
zeros), so do NOT add an image_encoder component. Left as a documented hook for the t2v registration.
"""
from __future__ import annotations

from typing import Any

from v2.program import ComponentNode, ModelLoopNode, Program, ProgramKind
from v2.recipes.common import cached_text_encode


def _ensure_byt5_checkpoint(instance: Any) -> None:
    """Stamp ``text_encoder_2``'s weights subfolder if the shared stamp left it empty.

    ``stamp_wan21_checkpoints`` (run by ``VideoGenerator.from_pretrained``) is Wan-centric and does NOT cover
    HunyuanVideo 1.5's second text encoder (ByT5 lives in ``text_encoder_2/``). Rather than edit the shared
    stamp, derive the model root from the already-stamped primary ``text_encoder`` (sibling layout) and stamp
    here at first use. Idempotent: a no-op once ``checkpoint`` is set."""
    import os
    spec2 = instance.card.components.get("text_encoder_2")
    if spec2 is None or spec2.checkpoint:
        return
    primary = instance.card.components["text_encoder"].checkpoint  # <root>/text_encoder (already stamped)
    model_root = os.path.dirname(os.path.normpath(primary))
    spec2.checkpoint = os.path.join(model_root, "text_encoder_2")


def _text_encode(instance: Any, slots: dict, request: Any, ctx: Any) -> None:
    """Encode prompt + negative prompt. When a second text encoder (ByT5) exists and the prompt carries
    quoted glyph text, pack the two embeds; otherwise write the single Qwen embedding.

    Why not a plain tuple: ``WanDenoiseLoop`` does numpy bookkeeping on the conditioning slot
    (``np.asarray(e).nbytes`` for resident-bytes, ``np.array(e)`` to allocate the CUDA-graph workspace), and
    a tuple of (qwen[seq,3584], byt5[bseq,1472]) is INHOMOGENEOUS so both raise. For the common t2v case the
    ByT5 stream is a zero-length ``[0,1472]`` embedding (no quoted text), a 0-token DiT no-op, so we pass ONLY
    the Qwen embedding and the adapter rebuilds the empty ByT5 default (its ``byt5_e is None`` branch). When
    glyph text IS present we pack the pair as a numpy OBJECT array (a single ndarray, so the np.asarray /
    np.array calls succeed without stacking) and the adapter unpacks it. This keeps the t2v path capturable
    (single ndarray) while still threading real ByT5 embeddings for glyph prompts (which eager-break)."""
    has_byt5 = "text_encoder_2" in instance.card.components
    # The dual-encoder packing is a GPU-path concern: the CPU toy backend's ToyDiT takes a single
    # [seq,dim] float embedding (and its ToyTextEncoder has no glyph/ByT5 stream), so on the toy we pass
    # the Qwen embedding alone. Gate on the real cuda backend.
    on_cuda = getattr(getattr(instance, "platform", None), "device", "cpu") == "cuda"
    dual = has_byt5 and on_cuda
    if dual:
        _ensure_byt5_checkpoint(instance)

    def _encode(text: str) -> Any:
        qwen = cached_text_encode(instance, text)  # primary (Qwen2.5-VL) — feature-cached
        if not dual:
            return qwen  # toy backend / no ByT5: single embedding
        byt5 = instance.component("text_encoder_2").encode(text)
        if byt5 is None or getattr(byt5, "shape", (0, ))[0] == 0:
            return qwen  # no glyph text -> empty ByT5 (the DiT no-op); pass Qwen alone (capturable)
        import numpy as np
        pair: np.ndarray = np.empty(2, dtype=object)
        pair[0], pair[1] = qwen, byt5
        return pair

    slots["text_embeds"] = _encode(request.prompt())
    slots["neg_text_embeds"] = _encode(request.diffusion.negative_prompt)


def _vae_decode(instance: Any, slots: dict, request: Any, ctx: Any) -> None:
    slots["video"] = instance.component("vae").decode(slots["denoise_out"]["latents"])


def build_hunyuan_video15_program() -> Program:
    return Program(
        program_id="hunyuan_video15.t2v.inline",
        kind=ProgramKind.INLINE,
        nodes=[
            ComponentNode("text_encode", fn=_text_encode, writes=("text_embeds", "neg_text_embeds")),
            ModelLoopNode("denoise",
                          loop_id="diffusion_denoise",
                          output_slot="denoise_out",
                          reads=("text_embeds", "neg_text_embeds"),
                          writes=("denoise_out", )),
            ComponentNode("vae_decode", fn=_vae_decode, reads=("denoise_out", ), writes=("video", )),
        ],
        output_artifacts={
            "video": "video",
            "latents": "denoise_out"
        },
    ).validate()
