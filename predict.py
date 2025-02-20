# Prediction interface for Cog ⚙️
# https://cog.run/python

import argparse
import os
import subprocess
import time

import imageio
import numpy as np
import torch
import torchvision
from cog import BasePredictor, Input, Path
from einops import rearrange

from fastvideo.models.hunyuan.inference import HunyuanVideoSampler

MODEL_CACHE = 'FastHunyuan'
os.environ['MODEL_BASE'] = './' + MODEL_CACHE

MODEL_URL = "https://weights.replicate.delivery/default/FastVideo/FastHunyuan/model.tar"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):

    def setup(self):
        """Load the model into memory"""
        print("Model Base: " + os.environ['MODEL_BASE'])
        # Download weights
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args = argparse.Namespace(
            num_frames=125,
            height=720,
            width=1280,
            num_inference_steps=6,
            fps=24,
            denoise_type='flow',
            seed=1024,
            neg_prompt=None,
            guidance_scale=1.0,
            embedded_cfg_scale=6.0,
            flow_shift=17,
            batch_size=1,
            num_videos=1,
            load_key='module',
            use_cpu_offload=False,
            dit_weight='FastHunyuan/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt',
            reproduce=True,
            disable_autocast=False,
            flow_reverse=True,
            flow_solver='euler',
            use_linear_quadratic_schedule=False,
            linear_schedule_end=25,
            model='HYVideo-T/2-cfgdistill',
            latent_channels=16,
            precision='bf16',
            rope_theta=256,
            vae='884-16c-hy',
            vae_precision='fp16',
            vae_tiling=True,
            text_encoder='llm',
            text_encoder_precision='fp16',
            text_states_dim=4096,
            text_len=256,
            tokenizer='llm',
            prompt_template='dit-llm-encode',
            prompt_template_video='dit-llm-encode-video',
            hidden_state_skip_layer=2,
            apply_final_norm=False,
            text_encoder_2='clipL',
            text_encoder_precision_2='fp16',
            text_states_dim_2=768,
            tokenizer_2='clipL',
            text_len_2=77,
            model_path=MODEL_CACHE,
        )
        self.model = HunyuanVideoSampler.from_pretrained(MODEL_CACHE, args=args)

    def predict(
            self,
            prompt: str = Input(description="Text prompt for video generation",
                                default="A cat walks on the grass, realistic style."),
            negative_prompt: str = Input(description="Text prompt to specify what you don't want in the video.",
                                         default=""),
            width: int = Input(description="Width of output video", default=1280, ge=256),
            height: int = Input(description="Height of output video", default=720, ge=256),
            num_frames: int = Input(description="Number of frames to generate", default=125, ge=16),
            num_inference_steps: int = Input(description="Number of denoising steps", default=6, ge=1, le=50),
            guidance_scale: float = Input(description="Classifier free guidance scale", default=1.0, ge=0.1, le=10.0),
            embedded_cfg_scale: float = Input(description="Embedded classifier free guidance scale",
                                              default=6.0,
                                              ge=0.1,
                                              le=10.0),
            flow_shift: int = Input(description="Flow shift parameter", default=17, ge=1, le=20),
            fps: int = Input(description="Frames per second of output video", default=24, ge=1, le=60),
            seed: int = Input(description="0 for Random seed. Set for reproducible generation", default=0),
    ) -> Path:
        """Run video generation"""
        if seed <= 0:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        outputs = self.model.predict(
            prompt=prompt,
            height=height,
            width=width,
            video_length=num_frames,
            seed=seed,
            negative_prompt=negative_prompt,
            infer_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            embedded_guidance_scale=embedded_cfg_scale,
            flow_shift=flow_shift,
            flow_reverse=True,
            batch_size=1,
            num_videos_per_prompt=1,
        )

        # Process output video
        videos = rearrange(outputs["samples"], "b c t h w -> t b c h w")
        frames = []
        for x in videos:
            x = torchvision.utils.make_grid(x, nrow=6)
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
            frames.append((x * 255).numpy().astype(np.uint8))

        # Save video
        output_path = Path("/tmp/output.mp4")
        imageio.mimsave(str(output_path), frames, fps=fps)
        return Path(output_path)
