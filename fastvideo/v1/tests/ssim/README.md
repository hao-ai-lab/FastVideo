The reference videos in the `reference_videos` directory are used as part of an e2e test to ensure consistency in video generation quality across code changes. `test_inference_similarity.py` compares newly generated videos against these references using Structural Similarity Index (SSIM) metrics to detect any regressions in visual quality across code changes.

`reference_videos/FLASH_ATTN/` videos were generated on commit `66107fd5b8469fed25972feb632cd48887dac451`.
`reference_videos/TORCH_SDPA/` videos were generated on commit `4ea008b8a16d7f5678a44b187ebdd7d9d0416ff1`.

## Generation Details

2 x NVIDIA A40 GPUs

## Generation Parameters

{
"num_gpus": 2,
"model_path": "data/FastHunyuan-diffusers",
"height": 720,
"width": 1280,
"num_frames": 45,
"num_inference_steps": 6,
"guidance_scale": 1,
"embedded_cfg_scale": 6,
"flow_shift": 17,
"seed": 1024,
"sp_size": 2,
"tp_size": 2,
"vae_sp": true,
"fps": 24
}

#### Prompts

1. Will Smith casually eats noodles, his relaxed demeanor contrasting with the energetic background of a bustling street food market. The scene captures a mix of humor and authenticity. Mid-shot framing, vibrant lighting."

2. "A lone hiker stands atop a towering cliff, silhouetted against the vast horizon. The rugged landscape stretches endlessly beneath, its earthy tones blending into the soft blues of the sky. The scene captures the spirit of exploration and human resilience. High angle, dynamic framing, with soft natural lighting emphasizing the grandeur of nature."
