# SPDX-License-Identifier: Apache-2.0
"""Native Qwen3-VL conditioner returning Qwen-Image-2.1's pre-norm features."""

from collections.abc import Iterable

import torch

from fastvideo.configs.models.encoders.qwen_image21 import QwenImage21Qwen3VLConfig
from fastvideo.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLConditioner
from fastvideo.models.loader.weight_utils import default_weight_loader


class QwenImage21Qwen3VLConditioner(MiniMaxH3Qwen3VLConditioner):
    """Qwen3-VL text and image features, without autoregressive generation.

    The shared native stack stops immediately after the configured last
    decoder layer. Its final RMSNorm is loaded for checkpoint completeness
    but never applied to the image transformer's conditioning features.
    Sequence parallel execution is not implemented for this conditioner.
    """

    supported_checkpoint_quantization_methods: frozenset[str] = frozenset()
    _checkpoint_quantization_configs: dict = {}

    def __init__(self, config: QwenImage21Qwen3VLConfig) -> None:
        super().__init__(config)
        # CPU construction of this constant avoids device-dependent power
        # rounding accumulating across the complete 36-layer decoder stack.
        rotary = self.language_model.rotary_emb
        exponents = torch.arange(0, config.head_dim, 2, dtype=torch.float32, device="cpu") / config.head_dim
        rotary.inv_freq = (1.0 / (config.rope_theta**exponents)).to(rotary.inv_freq.device)

    def _visual_features(
        self,
        pixels: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if grid_thw.ndim != 2 or grid_thw.shape[1] != 3 or grid_thw.shape[0] == 0:
            raise ValueError("image_grid_thw must contain one nonempty [time, height, width] row per image")
        grids = grid_thw.to(device="cpu", dtype=torch.long)
        merge = self.config.vision_spatial_merge_size
        if bool((grids <= 0).any()) or bool((grids[:, 1:] % merge != 0).any()):
            raise ValueError("Image grids must be positive and spatially divisible by the vision merge size")
        patch_counts = grids.prod(dim=1).tolist()
        if sum(patch_counts) != pixels.shape[0]:
            raise ValueError("pixel_values patch count does not match image_grid_thw")

        # Keep packed projection shapes: splitting images changes BF16 GEMM
        # rounding before the complete text decoder amplifies the difference.
        # Vision attention still executes separately for each image grid.
        return self.visual(pixels.to(self.visual.patch_embed.proj.weight.dtype), grid_thw)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``[batch, tokens, width]``, or ``[tokens, width]`` for 1-D IDs.

        Padding is zeroed and removed by the prompt stage using the original
        attention mask before stripping its template.
        """
        single_sequence = input_ids.ndim == 1
        if single_sequence:
            input_ids = input_ids.unsqueeze(0)
            if attention_mask is not None and attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)
            if mm_token_type_ids is not None and mm_token_type_ids.ndim == 1:
                mm_token_type_ids = mm_token_type_ids.unsqueeze(0)
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("Qwen-Image-2.1 input_ids must have shape [batch, nonempty sequence] or [sequence]")
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have the same shape as input_ids")
            if bool((attention_mask.to(torch.bool).sum(dim=1) == 0).any()):
                raise ValueError("Each conditioning sequence must contain an unmasked token")
        if (pixel_values is None) != (image_grid_thw is None):
            raise ValueError("pixel_values and image_grid_thw must be provided together")
        if bool((input_ids == self.config.video_token_id).any()):
            raise ValueError("Qwen-Image-2.1 conditioning accepts image references, not video tokens")
        if mm_token_type_ids is not None:
            expected = (input_ids == self.config.image_token_id).to(mm_token_type_ids.dtype)
            if mm_token_type_ids.shape != input_ids.shape or not torch.equal(mm_token_type_ids, expected):
                raise ValueError("mm_token_type_ids must mark image placeholders as 1 and other tokens as 0")

        inputs_embeds = self.language_model.embed_tokens(input_ids)
        visual_mask = None
        deepstack = None
        if pixel_values is not None:
            features, deepstack = self._visual_features(pixel_values, image_grid_thw)
            features = features.to(inputs_embeds.device, inputs_embeds.dtype)
            visual_mask = self._placeholder_mask(input_ids, inputs_embeds, self.config.image_token_id,
                                                features, "image")
            inputs_embeds = inputs_embeds.masked_scatter(visual_mask.unsqueeze(-1), features)
        elif bool((input_ids == self.config.image_token_id).any()):
            raise ValueError("Image placeholder tokens require pixel_values and image_grid_thw")

        # Text-only prefills retain positions in the padded sequence. A
        # constant RoPE shift is mathematically equivalent, but changes BF16
        # rounding; image prefills need each sample's multimodal layout.
        positions = self._get_rope_index(input_ids, image_grid_thw, None,
                                         attention_mask if image_grid_thw is not None else None)
        # Unpadding changes projection and attention shapes in reduced
        # precision. Preserve the batch through every decoder layer instead.
        hidden_states = self.language_model(inputs_embeds, positions, attention_mask, visual_mask, deepstack)
        if attention_mask is not None:
            hidden_states = hidden_states.masked_fill(~attention_mask.to(torch.bool).unsqueeze(-1), 0)
        return hidden_states[0] if single_sequence else hidden_states

    @torch.no_grad()
    def encode_ids(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode one sequence without imposing a tokenizer truncation limit."""
        if input_ids.ndim != 1:
            raise ValueError("encode_ids requires 1-D input_ids; use forward for a padded batch")
        return self.forward(
            input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        parameters = dict(self.named_parameters())
        loaded: set[str] = set()
        for source_name, tensor in weights:
            if source_name == "lm_head.weight":
                continue
            name = source_name.removeprefix("model.")
            if name not in parameters:
                raise ValueError(f"Unexpected Qwen-Image-2.1 Qwen3-VL checkpoint key: {source_name}")
            if name in loaded:
                raise ValueError(f"Duplicate Qwen-Image-2.1 Qwen3-VL checkpoint key: {source_name}")
            parameter = parameters[name]
            loader = getattr(parameter, "weight_loader", default_weight_loader)
            loader(parameter, tensor)
            loaded.add(name)
        return loaded


EntryClass = QwenImage21Qwen3VLConditioner
