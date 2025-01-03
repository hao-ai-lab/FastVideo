import os
from transformers import AutoTokenizer, T5EncoderModel
from diffusers import AutoencoderKLHunyuanVideo
import torch

init_dict =  {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 4,
            "down_block_types": (
                "HunyuanVideoDownBlock3D",
                "HunyuanVideoDownBlock3D",
                "HunyuanVideoDownBlock3D",
                "HunyuanVideoDownBlock3D",
            ),
            "up_block_types": (
                "HunyuanVideoUpBlock3D",
                "HunyuanVideoUpBlock3D",
                "HunyuanVideoUpBlock3D",
                "HunyuanVideoUpBlock3D",
            ),
            "block_out_channels": (8, 8, 8, 8),
            "layers_per_block": 1,
            "act_fn": "silu",
            "norm_num_groups": 4,
            "scaling_factor": 0.476986,
            "spatial_compression_ratio": 8,
            "temporal_compression_ratio": 4,
            "mid_block_add_attention": True,
        }
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
text_encoder = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")
tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

model = AutoencoderKLHunyuanVideo(**init_dict)

input_tensor = torch.rand(1, 3, 9, 16, 16)

vae_encoder_output =  model.encoder(input_tensor)

# vae_decoder_output =  model.decoder(vae_encoder_output)

assert vae_encoder_output.shape == (1,8,3,2,2)

# print(vae_decoder_output.shape)