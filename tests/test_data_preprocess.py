import os
import unittest

import torch
from transformers import AutoTokenizer, T5EncoderModel

from fastvideo.models.hunyuan.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D


class TestAutoencoderKLCausal3D(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        setUpClass is called once, before any test is run.
        We can set environment variables or load heavy resources here.
        """
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

        # Load tokenizer/model that can be reused across all tests
        cls.tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")
        cls.text_encoder = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")

    def setUp(self):
        """
        setUp is called before each test method to prepare fresh state.
        """
        self.batch_size = 1
        self.init_time_len = 9
        self.init_height = 16
        self.init_width = 16
        self.latent_channels = 4
        self.spatial_compression_ratio = 8
        self.time_compression_ratio = 4

        # Model initialization config
        self.init_dict = {
            "in_channels":
            3,
            "out_channels":
            3,
            "latent_channels":
            self.latent_channels,
            "down_block_types": (
                "DownEncoderBlockCausal3D",
                "DownEncoderBlockCausal3D",
                "DownEncoderBlockCausal3D",
                "DownEncoderBlockCausal3D",
            ),
            "up_block_types": (
                "UpDecoderBlockCausal3D",
                "UpDecoderBlockCausal3D",
                "UpDecoderBlockCausal3D",
                "UpDecoderBlockCausal3D",
            ),
            "block_out_channels": (8, 8, 8, 8),
            "layers_per_block":
            1,
            "act_fn":
            "silu",
            "norm_num_groups":
            4,
            "scaling_factor":
            0.476986,
            "spatial_compression_ratio":
            self.spatial_compression_ratio,
            "time_compression_ratio":
            self.time_compression_ratio,
            "mid_block_add_attention":
            True,
        }

        # Instantiate the model
        self.model = AutoencoderKLCausal3D(**self.init_dict)

        # Create a random input tensor
        self.input_tensor = torch.rand(self.batch_size, 3, self.init_time_len, self.init_height, self.init_width)

    def test_encode_shape(self):
        """
        Check that the shape of the encoded output matches expectations.
        """
        vae_encoder_output = self.model.encode(self.input_tensor)

        # The distribution from the VAE has a .sample() method
        # so we verify the shape of that sample.
        sample_shape = vae_encoder_output["latent_dist"].sample().shape

        # We expect shape: [batch_size, latent_channels,
        #                   (init_time_len // time_compression_ratio) + 1,
        #                   init_height // spatial_compression_ratio,
        #                   init_width // spatial_compression_ratio]
        expected_shape = (
            self.batch_size,
            self.latent_channels,
            (self.init_time_len // self.time_compression_ratio) + 1,
            self.init_height // self.spatial_compression_ratio,
            self.init_width // self.spatial_compression_ratio,
        )

        # (Optional) Print them if you like, or just rely on assertions:
        print(f"sample_shape: {sample_shape}")
        print(f"expected_shape: {expected_shape}")

        self.assertEqual(
            sample_shape,
            expected_shape,
            f"Encoded sample shape {sample_shape} does not match {expected_shape}.",
        )


if __name__ == "__main__":
    unittest.main()
