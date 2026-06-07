# SPDX-License-Identifier: Apache-2.0
"""Numerical-parity test: FastVideo Cosmos3 sound decoder vs the framework AVAE.

The Cosmos3 ``sound_tokenizer`` is an AVAE (audio VAE). Its shipped diffusers
checkpoint is **decoder-only** (``decoder.*``; the SpectrogramConvNeXt encoder is
not exported) in diffusers ``AutoencoderOobleck`` naming, but with **SnakeBeta**
activations (alpha+beta, logscale) and ``weight_g``/``weight_v`` weight-norm —
i.e. exactly FastVideo's existing native ``OobleckVAE`` decoder
(``fastvideo/models/vaes/oobleck.py``). t2vs only needs DECODE (generate sound
latents -> waveform), so this pins the decoder.

The framework decoder
(``cosmos_framework.model.vfm.tokenizers.audio.avae_utils.models.OobleckDecoder``,
``nn.Sequential`` naming, ``output_padding=stride%2`` on the transpose convs) is
the parity ORACLE. We build a tiny framework decoder, map its weights into the
FastVideo decoder (Sequential -> conv1/block.N/res_unitM/snake1/conv2), and
assert bit-exact decode. Strides include an ODD value (5, as in the real config
``[2,4,5,6,8]``) to exercise the ``output_padding`` path that diverged before.

CPU / float32. Run:
    cd <worktree> && <fv-cosmos3 python> -m pytest \
        tests/local_tests/cosmos3/test_cosmos3_avae_parity.py -q
"""
from __future__ import annotations

import pytest
import torch

# The official framework provides the parity oracle.
_fw_models = pytest.importorskip(
    "cosmos_framework.model.vfm.tokenizers.audio.avae_utils.models",
    reason="cosmos_framework not installed; run in fv-cosmos3 env.",
)
from cosmos_framework.model.vfm.tokenizers.audio.avae_utils.env import (  # noqa: E402
    AttrDict,
)

from fastvideo.models.vaes.oobleck import OobleckDecoder as FvOobleckDecoder  # noqa: E402

pytestmark = [pytest.mark.local]

FwOobleckDecoder = _fw_models.OobleckDecoder


def _framework_decoder(dec_dim, vocoder_input_dim, dec_c_mults, dec_strides):
    """Framework OobleckDecoder (the parity oracle), non-causal / no-antialias."""
    h = AttrDict({
        "vocoder_input_dim": vocoder_input_dim,
        "input_channels": 1,
        "stereo": True,  # 2 audio channels
        "dec_dim": dec_dim,
        "dec_c_mults": dec_c_mults,
        "dec_strides": dec_strides,
        "dec_use_snake": True,
        "dec_use_nearest_upsample": False,
        "dec_anti_aliasing": False,
        "causal": False,
        "dec_use_tanh_at_final": False,
        "padding_mode": "zeros",
    })
    return FwOobleckDecoder(h).eval()


def _framework_to_fastvideo_decoder_state(fw_decoder, num_blocks):
    """Map framework Sequential decoder weights -> FastVideo decoder names.

    framework: layers.0=first conv; layers.{1..K}=OobleckDecoderBlock
    (.layers.0 snake, .1 conv_t, .{2,3,4} ResidualUnit{.layers.0 snake,
    .1 conv, .2 snake, .3 conv}); layers.{1+K}=final snake; layers.{2+K}=final conv.
    FastVideo: conv1; block.{b}.{snake1,conv_t1,res_unit{1,2,3}.{snake1,conv1,snake2,conv2}};
    snake1; conv2. Snake alpha/beta: framework [C] -> FastVideo [1,C,1].
    """
    out = {}
    for k, v in fw_decoder.state_dict().items():
        p = k.split(".")
        li = int(p[1])
        if li == 0:
            nk = "conv1." + ".".join(p[2:])
        elif li == 1 + num_blocks:
            nk = "snake1." + ".".join(p[2:])
        elif li == 2 + num_blocks:
            nk = "conv2." + ".".join(p[2:])
        else:
            b = li - 1
            sub = int(p[3])
            if sub == 0:
                nk = f"block.{b}.snake1." + ".".join(p[4:])
            elif sub == 1:
                nk = f"block.{b}.conv_t1." + ".".join(p[4:])
            else:
                r = sub - 2  # ResidualUnit index 0..2
                m = {0: "snake1", 1: "conv1", 2: "snake2", 3: "conv2"}[int(p[5])]
                nk = f"block.{b}.res_unit{r + 1}.{m}." + ".".join(p[6:])
        if nk.endswith(".alpha") or nk.endswith(".beta"):
            v = v.reshape(1, -1, 1)
        out[nk] = v
    return out


# (dec_dim, vocoder_input_dim, dec_c_mults, dec_strides) — tiny; strides incl odd.
_CASES = [
    pytest.param(4, 8, [1, 2], [5, 2], id="odd_stride5"),
    pytest.param(6, 8, [1, 2, 4], [2, 5, 6], id="real_stride_pattern_tiny"),
    pytest.param(4, 4, [1, 2], [4, 8], id="even_strides"),
]


class TestCosmos3AVAEParity:

    @pytest.mark.parametrize(("dec_dim", "vin", "cmults", "strides"), _CASES)
    def test_decode_matches_framework(self, dec_dim, vin, cmults, strides):
        torch.manual_seed(0)
        fw = _framework_decoder(dec_dim, vin, cmults, strides)
        fv = FvOobleckDecoder(
            channels=dec_dim,
            input_channels=vin,
            audio_channels=2,
            upsampling_ratios=list(reversed(strides)),  # framework reverses dec_strides
            channel_multiples=cmults,
        ).eval()
        state = _framework_to_fastvideo_decoder_state(fw, num_blocks=len(strides))
        fv.load_state_dict(state, strict=True)  # exact name + shape match

        z = torch.randn(1, vin, 5)
        with torch.no_grad():
            a = fw(z)
            b = fv(z)
        assert a.shape == b.shape, f"shape: fw={a.shape} fv={b.shape}"
        max_abs = (a - b).abs().max().item()
        print(f"\n[avae_decode dim={dec_dim} strides={strides}] max abs diff = {max_abs:.3e}")
        torch.testing.assert_close(b, a, atol=1e-6, rtol=1e-5)
