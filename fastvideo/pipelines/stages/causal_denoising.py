import torch  # type: ignore

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.denoising import DenoisingStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult


class CausalDMDDenosingStage(DenoisingStage):
    """
    Denoising stage for causal diffusion.
    """

    def __init__(self,
                 transformer,
                 scheduler,
                 transformer_2=None,
                 vae=None) -> None:
        super().__init__(transformer, scheduler, transformer_2)
        # KV and cross-attention cache state (initialized on first forward)
        self.transformer = transformer
        self.transformer_2 = transformer_2
        self.vae = vae
        # Model-dependent constants (aligned with causal_inference.py assumptions)
        self.num_transformer_blocks = len(self.transformer.blocks)
        self.num_frames_per_block = self.transformer.config.arch_config.num_frames_per_block
        self.sliding_window_num_frames = self.transformer.config.arch_config.sliding_window_num_frames

        try:
            self.local_attn_size = getattr(self.transformer.model,
                                           "local_attn_size",
                                           -1)  # type: ignore
        except Exception:
            self.local_attn_size = -1

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        from fastvideo.pipelines.stages.denoising_engine import DenoisingEngine
        from fastvideo.pipelines.stages.denoising_causal_strategy import (
            CausalBlockStrategy)

        engine = DenoisingEngine(CausalBlockStrategy(self))
        return engine.run(batch, fastvideo_args)

    def _initialize_kv_cache(self, batch_size, dtype, device) -> list[dict]:
        """
        Initialize a Per-GPU KV cache aligned with the Wan model assumptions.
        """
        kv_cache1 = []
        num_attention_heads = self.transformer.num_attention_heads
        attention_head_dim = self.transformer.attention_head_dim
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            kv_cache_size = self.frame_seq_length * self.sliding_window_num_frames

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k":
                torch.zeros([
                    batch_size, kv_cache_size, num_attention_heads,
                    attention_head_dim
                ],
                            dtype=dtype,
                            device=device),
                "v":
                torch.zeros([
                    batch_size, kv_cache_size, num_attention_heads,
                    attention_head_dim
                ],
                            dtype=dtype,
                            device=device),
                "global_end_index":
                torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index":
                torch.tensor([0], dtype=torch.long, device=device),
            })

        return kv_cache1

    def _initialize_crossattn_cache(self, batch_size, max_text_len, dtype,
                                    device) -> list[dict]:
        """
        Initialize a Per-GPU cross-attention cache aligned with the Wan model assumptions.
        """
        crossattn_cache = []
        num_attention_heads = self.transformer.num_attention_heads
        attention_head_dim = self.transformer.attention_head_dim
        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k":
                torch.zeros([
                    batch_size, max_text_len, num_attention_heads,
                    attention_head_dim
                ],
                            dtype=dtype,
                            device=device),
                "v":
                torch.zeros([
                    batch_size, max_text_len, num_attention_heads,
                    attention_head_dim
                ],
                            dtype=dtype,
                            device=device),
                "is_init":
                False,
            })
        return crossattn_cache

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify denoising stage inputs."""
        result = VerificationResult()
        result.add_check("latents", batch.latents,
                         [V.is_tensor, V.with_dims(5)])
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)
        result.add_check("image_embeds", batch.image_embeds, V.is_list)
        result.add_check("image_latent", batch.image_latent,
                         V.none_or_tensor_with_dims(5))
        result.add_check("num_inference_steps", batch.num_inference_steps,
                         V.positive_int)
        result.add_check("guidance_scale", batch.guidance_scale,
                         V.positive_float)
        result.add_check("eta", batch.eta, V.non_negative_float)
        result.add_check("generator", batch.generator,
                         V.generator_or_list_generators)
        result.add_check("do_classifier_free_guidance",
                         batch.do_classifier_free_guidance, V.bool_value)
        result.add_check(
            "negative_prompt_embeds", batch.negative_prompt_embeds, lambda x:
            not batch.do_classifier_free_guidance or V.list_not_empty(x))
        return result
