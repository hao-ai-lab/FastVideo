# SPDX-License-Identifier: Apache-2.0
"""CPU smoke test for the HunyuanVideo 1.5 forward contract.

Builds the kwargs dict that ``Hunyuan15Model._build_distill_input_kwargs``
feeds to ``HunyuanVideo15Transformer3DModel.forward`` and checks it against
the transformer's real signature.  No weights are loaded and no forward
pass runs, so this executes on CPU in well under a second -- it exists to
catch signature/shape drift between the training plugin and the inference
DiT without needing a GPU.

Covered:
  - every key we pass is a real forward parameter
  - every required forward parameter is provided
  - the kwargs that crash HY1.5 stay absent
  - encoder_hidden_states is [qwen, byt5], including the zero-token case
  - encoder_hidden_states_image is a one-element all-zero list (T2V branch)
"""

from __future__ import annotations

import inspect

import pytest
import torch

from fastvideo.models.dits.hunyuanvideo15 import (
    HunyuanVideo15Transformer3DModel, )
from fastvideo.train.models.hunyuan15 import Hunyuan15Model

_QWEN_DIM = 3584
_BYT5_DIM = 1472
_IMAGE_TOKENS = 729
_IMAGE_DIM = 1152

# Kwargs that break the HY1.5 forward: absent from its signature, or only
# valid when the arch config enables meanflow.
_FORBIDDEN = ("encoder_attention_mask", "return_dict", "timestep_r")


def _build_kwargs(byt5_tokens: int = 12, batch_size: int = 1) -> dict:
    """Call the plugin helper on synthetic inputs.

    ``_build_distill_input_kwargs`` reads no instance state, so an
    uninitialised instance avoids loading an 8B checkpoint here.
    """
    model = object.__new__(Hunyuan15Model)
    noise_input = torch.zeros(batch_size, 32, 4, 8, 8)
    timestep = torch.full((batch_size, ), 500.0)
    text_dict = {
        "encoder_hidden_states": torch.zeros(batch_size, 20, _QWEN_DIM),
        "encoder_hidden_states_2": torch.zeros(batch_size, byt5_tokens, _BYT5_DIM),
    }
    return model._build_distill_input_kwargs(noise_input, timestep, text_dict)


def test_kwargs_match_transformer_signature():
    kwargs = _build_kwargs()
    params = inspect.signature(HunyuanVideo15Transformer3DModel.forward).parameters

    unknown = set(kwargs) - set(params)
    assert not unknown, ("kwargs rejected by the HY1.5 forward signature: "
                         f"{sorted(unknown)}")

    required = {
        name
        for name, param in params.items()
        if name != "self" and param.default is inspect.Parameter.empty and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    missing = required - set(kwargs)
    assert not missing, f"required forward params not provided: {sorted(missing)}"


@pytest.mark.parametrize("key", _FORBIDDEN)
def test_forbidden_kwarg_absent(key):
    assert key not in _build_kwargs()


def test_dual_text_embeddings_are_a_two_element_list():
    encoder_hidden_states = _build_kwargs()["encoder_hidden_states"]
    assert isinstance(encoder_hidden_states, list)
    assert len(encoder_hidden_states) == 2
    qwen, byt5 = encoder_hidden_states
    assert qwen.shape[-1] == _QWEN_DIM
    assert byt5.shape[-1] == _BYT5_DIM


def test_zero_token_byt5_is_preserved():
    """Captions without glyph text carry a zero-length ByT5 stream."""
    byt5 = _build_kwargs(byt5_tokens=0)["encoder_hidden_states"][1]
    assert byt5.shape[1] == 0, "zero tokens, not zero values"
    assert byt5.shape[-1] == _BYT5_DIM


def test_t2v_image_placeholder_is_all_zero():
    kwargs = _build_kwargs(batch_size=2)
    image = kwargs["encoder_hidden_states_image"]
    assert isinstance(image, list)
    assert len(image) == 1
    assert image[0].shape == (2, _IMAGE_TOKENS, _IMAGE_DIM)
    assert torch.all(image[0] == 0), "all-zero content selects the T2V branch"


def test_hidden_states_are_packed_to_65_channels():
    """32 latent + 1 conditioning mask + 32 conditioning latent.

    predict_noise owns the (B,T,C,H,W) -> (B,C,T,H,W) permute; this
    helper only packs the conditioning channels HY1.5's img_in expects.
    """
    hidden_states = _build_kwargs()["hidden_states"]
    assert hidden_states.shape == (1, 65, 4, 8, 8)
    # T2V leaves both conditioning blocks zero.
    assert torch.all(hidden_states[:, 32:] == 0)
