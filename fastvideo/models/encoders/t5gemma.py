# SPDX-License-Identifier: Apache-2.0
"""T5-Gemma encoder wrapper for daVinci-MagiHuman.

MagiHuman uses `transformers.models.t5gemma.T5GemmaEncoderModel` on
`google/t5gemma-9b-9b-ul2` (a gated Google repo). This wrapper follows the
same lazy-loading pattern as `fastvideo/models/encoders/gemma.py`: we keep
the HF module under `self._t5gemma_model` and exclude it from
`named_parameters` so FastVideo's weight loader does not try to load
encoder shards from the converted repo directory.

For the base MagiHuman T2V port there are no additional connector layers
on top — the pipeline prompt-preprocessing stage handles pad-or-trim to
`text_len` and exposes both the padded embedding and the original length.
"""
from __future__ import annotations

import os

import torch
from torch import nn

from fastvideo.configs.models.encoders import BaseEncoderOutput, TextEncoderConfig
from fastvideo.models.encoders.base import TextEncoder
from fastvideo.platforms import AttentionBackendEnum


class T5GemmaEncoderModel(TextEncoder):
    """Thin wrapper over HuggingFace's `T5GemmaEncoderModel`.

    On first `forward`, the wrapper lazily instantiates the upstream encoder
    from `t5gemma_model_path` (defaulting to `google/t5gemma-9b-9b-ul2`).
    Afterwards, forward returns a `BaseEncoderOutput` with
    `last_hidden_state = [B, L, 3584]` matching MagiHuman's
    `context.half()` output.
    """

    _supported_attention_backends = (
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.TORCH_SDPA,
    )

    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__(config)
        arch = config.arch_config
        self.t5gemma_model_path: str = arch.t5gemma_model_path
        self.t5gemma_dtype: str = arch.t5gemma_dtype
        self._t5gemma_model = None

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        # The upstream encoder is loaded lazily and its parameters are
        # managed by HF, not FastVideo's loader. Hide them from the parent
        # module-tree traversal so Diffusers-repo weight loading does not
        # try to match them.
        for name, param in super().named_parameters(prefix=prefix, recurse=recurse):
            if name.startswith("_t5gemma_model.") or name == "_t5gemma_model":
                continue
            yield name, param

    def _build_t5gemma_model(self, device: torch.device | None = None):
        from transformers.models.t5gemma import T5GemmaEncoderModel as HFEncoder

        path = self.t5gemma_model_path
        if not path:
            raise ValueError(
                "t5gemma_model_path must be set. Expected "
                "`google/t5gemma-9b-9b-ul2` or a local path to an "
                "equivalent T5-Gemma encoder."
            )
        dtype = getattr(torch, self.t5gemma_dtype, torch.bfloat16)
        model = HFEncoder.from_pretrained(
            path,
            is_encoder_decoder=False,
            dtype=dtype,
        )
        if os.getenv("FASTVIDEO_ATTENTION_BACKEND") == "TORCH_SDPA":
            if hasattr(model.config, "attn_implementation"):
                model.config.attn_implementation = "sdpa"
            if hasattr(model.config, "_attn_implementation"):
                model.config._attn_implementation = "sdpa"
        if device is not None:
            model = model.to(device=device)
        model.eval()
        return model

    @property
    def t5gemma_model(self):
        if self._t5gemma_model is None:
            # Lazy-load on CPU if no device is known yet; `forward` will
            # move the model to the input's device on first call.
            self._t5gemma_model = self._build_t5gemma_model()
        return self._t5gemma_model

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> BaseEncoderOutput:
        # Ensure the lazy-loaded encoder lives on the same device as the
        # input; lazy-loading leaves it on CPU until the first forward.
        ref = input_ids if input_ids is not None else inputs_embeds
        target_device = ref.device if ref is not None else None
        model = self.t5gemma_model
        if target_device is not None:
            first_param = next(model.parameters(), None)
            if first_param is not None and first_param.device != target_device:
                model = model.to(device=target_device)
                self._t5gemma_model = model
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=bool(output_hidden_states),
        )
        # MagiHuman casts to fp16 at this point; keep the raw dtype here and
        # leave precision management to the pipeline's postprocess stage.
        return BaseEncoderOutput(
            last_hidden_state=outputs["last_hidden_state"],
            hidden_states=getattr(outputs, "hidden_states", None),
            attention_mask=attention_mask,
        )


EntryClass = T5GemmaEncoderModel
