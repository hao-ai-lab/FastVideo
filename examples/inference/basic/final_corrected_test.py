#!/usr/bin/env python3
"""
FINAL TEST: Generate with EXACT SGLang parameters (with corrected dimensions!)

CRITICAL FIX: SGLang generated 1280×720 (landscape), but we were testing 720×1280 (portrait)!
This test uses the CORRECT dimensions that match SGLang's output.
"""
import sys
sys.path.insert(0, '/FastVideo')

import os
import numpy as np
from PIL import Image

def main():
    os.environ["HF_HOME"] = "/tmp/huggingface_cache"
    os.environ["_FLASH_ATTN_DISABLED"] = "1"

    print("="*80)
    print("FINAL TEST: FastVideo with CORRECTED SGLang-matching parameters")
    print("="*80)

    # SGLang's actual output dimensions
    PROMPT = "A cinematic portrait of a fox, 35mm film, soft light, gentle grain."
    WIDTH = 1280   # SGLang image is 1280 wide
    HEIGHT = 720   # SGLang image is 720 tall
    SEED = 42      # Need to determine SGLang's actual seed
    STEPS = 50     # Need to determine SGLang's actual steps

    print(f"\nParameters:")
    print(f"  prompt: {PROMPT[:50]}...")
    print(f"  width:  {WIDTH} (CORRECTED from 720)")
    print(f"  height: {HEIGHT} (CORRECTED from 1280)")
    print(f"  seed:   {SEED} (assumed)")
    print(f"  steps:  {STEPS} (assumed)")

    print("\n" + "="*80)
    print("Loading FastVideo and generating...")
    print("="*80)

    from fastvideo import VideoGenerator

    gen = VideoGenerator.from_pretrained(
        model_path="black-forest-labs/FLUX.1-dev",
        num_gpus=1,
        text_encoder_cpu_offload=True,
    )

    result = gen.generate_video(
        prompt=PROMPT,
        width=WIDTH,
        height=HEIGHT,
        num_frames=1,
        seed=SEED,
        num_inference_steps=STEPS,
        guidance_scale=1.0,  # Match SGLang's guidance_scale
        save_video=False,
        return_frames=True,
    )

    fv_frame = np.array(result[0])

    print("\n" + "="*80)
    print("FastVideo Output")
    print("="*80)
    print(f"Shape: {fv_frame.shape}")
    print(f"  Width×Height: {fv_frame.shape[1]}×{fv_frame.shape[0]}")
    print(f"  Expected: {WIDTH}×{HEIGHT}")
    print(f"  Match: {fv_frame.shape[1] == WIDTH and fv_frame.shape[0] == HEIGHT}")
    print(f"\nPixel stats:")
    print(f"  min={fv_frame.min()}, max={fv_frame.max()}, mean={fv_frame.mean():.2f}")
    print(f"  Top-left corner: {fv_frame[:2, :2, :]}")

    # Save for comparison
    output_path = "fastvideo_corrected_dimensions.png"
    Image.fromarray(fv_frame).save(output_path)
    print(f"\nSaved to: {output_path}")

    # Load and compare with SGLang
    sg_path = "A_cinematic_portrait_of_a_fox_35mm_film_soft_light_gentle_grain._20260215-132609_878dfa3f.png"
    print("\n" + "="*80)
    print("SGLang Output (pre-generated)")
    print("="*80)
    sg_image = Image.open(sg_path)
    sg_frame = np.array(sg_image)
    print(f"Shape: {sg_frame.shape}")
    print(f"  Width×Height: {sg_frame.shape[1]}×{sg_frame.shape[0]}")
    print(f"\nPixel stats:")
    print(f"  min={sg_frame.min()}, max={sg_frame.max()}, mean={sg_frame.mean():.2f}")
    print(f"  Top-left corner: {sg_frame[:2, :2, :]}")

    # Compare
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    if fv_frame.shape == sg_frame.shape:
        diff = np.abs(fv_frame.astype(float) - sg_frame.astype(float))
        print(f"✅ Shapes match: {fv_frame.shape}")
        print(f"\nPixel difference:")
        print(f"  max abs diff:  {diff.max():.2f}")
        print(f"  mean abs diff: {diff.mean():.2f}")
        print(f"  std abs diff:  {diff.std():.2f}")
        
        # Pixel-wise comparison
        close_pixels = (diff.mean(axis=2) < 5).sum()
        total_pixels = diff.shape[0] * diff.shape[1]
        print(f"\n  Close pixels (diff<5): {close_pixels}/{total_pixels} ({100*close_pixels/total_pixels:.1f}%)")
        
        if diff.mean() < 1.0:
            print("\n🎉 EXCELLENT MATCH (mean diff < 1 pixel)")
        elif diff.mean() < 5.0:
            print("\n✅ GOOD MATCH (mean diff < 5 pixels)")
        elif diff.mean() < 20.0:
            print("\n⚠️  MODERATE DIFFERENCE (mean diff < 20 pixels)")
        else:
            print("\n❌ SIGNIFICANT DIFFERENCE (mean diff >= 20 pixels)")
            print("\nPossible causes:")
            print("  - Different random seed (SGLang may use different default)")
            print("  - Different num_inference_steps (SGLang may use different default)")
            print("  - Different guidance parameters")
            print("  - Different model precision or attention backend")
    else:
        print(f"❌ Shape mismatch!")
        print(f"  FastVideo: {fv_frame.shape}")
        print(f"  SGLang:    {sg_frame.shape}")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("""
We corrected the critical dimension error:
  - Original test: 720×1280 (portrait) 
  - Corrected test: 1280×720 (landscape) ✅

This test will show whether dimensions were the only issue, or if there
are other parameter differences (seed, steps, etc.) between FastVideo
and SGLang's defaults that we still need to match.
""")

if __name__ == '__main__':
    main()




# (venv) root@b4bd8f10de13:/FastVideo# python final_corrected_test.py
# python: can't open file '/FastVideo/final_corrected_test.py': [Errno 2] No such file or directory
# (venv) root@b4bd8f10de13:/FastVideo# cd examples/inference/basic/
# (venv) root@b4bd8f10de13:/FastVideo/examples/inference/basic# python final_corrected_test.py
# ================================================================================
# FINAL TEST: FastVideo with CORRECTED SGLang-matching parameters
# ================================================================================

# Parameters:
#   prompt: A cinematic portrait of a fox, 35mm film, soft lig...
#   width:  1280 (CORRECTED from 720)
#   height: 720 (CORRECTED from 1280)
#   seed:   42 (assumed)
#   steps:  50 (assumed)

# ================================================================================
# Loading FastVideo and generating...
# ================================================================================
# INFO 02-17 20:03:55 [__init__.py:109] ROCm platform is unavailable: No module named 'amdsmi'
# WARNING 02-17 20:03:55 [logger.py:122]  By default, logger.info(..) will only log from the local main process. Set logger.info(..., is_local_main_process=False) to log from all processes.
# INFO 02-17 20:03:55 [__init__.py:47] CUDA is available
# WARNING 02-17 20:03:56 [fastvideo_args.py:705] dit_layerwise_offload is enabled, automatically disabling dit_cpu_offload.
# INFO 02-17 20:03:56 [multiproc_executor.py:83] Use master port: 36603
# INFO 02-17 20:04:01 [__init__.py:109] ROCm platform is unavailable: No module named 'amdsmi'
# WARNING 02-17 20:04:01 [logger.py:122]  By default, logger.info(..) will only log from the local main process. Set logger.info(..., is_local_main_process=False) to log from all processes.
# INFO 02-17 20:04:01 [__init__.py:47] CUDA is available
# INFO 02-17 20:04:03 [parallel_state.py:976] Initializing distributed environment with world_size=1, device=cuda:0
# INFO 02-17 20:04:03 [parallel_state.py:788] Using nccl backend for CUDA platform
# [W217 20:04:03.784057796 ProcessGroupNCCL.cpp:924] Warning: TORCH_NCCL_AVOID_RECORD_STREAMS is the default now, this environment variable is thus deprecated. (function operator())
# [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
# [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
# [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
# [Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
# INFO 02-17 20:04:03 [utils.py:517] Downloading model snapshot from HF Hub for black-forest-labs/FLUX.1-dev...
# INFO 02-17 20:04:03 [utils.py:524] Downloaded model to /tmp/huggingface_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21
# INFO 02-17 20:04:03 [__init__.py:43] Model path: /tmp/huggingface_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21
# INFO 02-17 20:04:03 [utils.py:600] Diffusers version: 0.30.0.dev0
# INFO 02-17 20:04:03 [__init__.py:59] Building pipeline of type: basic
# INFO 02-17 20:04:03 [pipeline_registry.py:174] Loading pipelines for types: ['basic']
# INFO 02-17 20:04:03 [pipeline_registry.py:230] Loaded 20 pipeline classes across 1 types
# INFO 02-17 20:04:03 [profiler.py:191] Torch profiler disabled; returning no-op controller
# INFO 02-17 20:04:03 [composed_pipeline_base.py:86] Loading pipeline modules...
# INFO 02-17 20:04:03 [utils.py:512] Model already exists locally at /tmp/huggingface_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21
# INFO 02-17 20:04:03 [composed_pipeline_base.py:225] Model path: /tmp/huggingface_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21
# INFO 02-17 20:04:03 [utils.py:600] Diffusers version: 0.30.0.dev0
# INFO 02-17 20:04:03 [composed_pipeline_base.py:285] Loading pipeline modules from config: {'_class_name': 'FluxPipeline', '_diffusers_version': '0.30.0.dev0', 'scheduler': ['diffusers', 'FlowMatchEulerDiscreteScheduler'], 'text_encoder': ['transformers', 'CLIPTextModel'], 'text_encoder_2': ['transformers', 'T5EncoderModel'], 'tokenizer': ['transformers', 'CLIPTokenizer'], 'tokenizer_2': ['transformers', 'T5TokenizerFast'], 'transformer': ['diffusers', 'FluxTransformer2DModel'], 'vae': ['diffusers', 'AutoencoderKL']}
# INFO 02-17 20:04:03 [composed_pipeline_base.py:329] Loading required modules: ['text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'vae', 'transformer', 'scheduler']
# INFO 02-17 20:04:03 [component_loader.py:972] Loading scheduler using diffusers from /tmp/huggingface_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/scheduler
# INFO 02-17 20:04:03 [composed_pipeline_base.py:363] Loaded module scheduler from /tmp/huggingface_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/scheduler
# INFO 02-17 20:04:03 [component_loader.py:972] Loading text_encoder using transformers from /tmp/huggingface_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/text_encoder
# INFO 02-17 20:04:03 [component_loader.py:308] HF Model config: {'architectures': ['CLIPTextModel'], 'attention_dropout': 0.0, 'bos_token_id': 0, 'dropout': 0.0, 'eos_token_id': 2, 'hidden_act': 'quick_gelu', 'hidden_size': 768, 'initializer_factor': 1.0, 'initializer_range': 0.02, 'intermediate_size': 3072, 'layer_norm_eps': 1e-05, 'max_position_embeddings': 77, 'num_attention_heads': 12, 'num_hidden_layers': 12, 'pad_token_id': 1, 'projection_dim': 768, 'vocab_size': 49408}
# INFO 02-17 20:04:04 [cuda.py:124] Trying FASTVIDEO_ATTENTION_BACKEND=None
# INFO 02-17 20:04:04 [cuda.py:126] Selected backend: None
# INFO 02-17 20:04:04 [cuda.py:267] Cannot use FlashAttention-2 backend because the flash_attn package is not found. Make sure that flash_attn was built and installed (on by default).
# INFO 02-17 20:04:04 [cuda.py:274] Using Torch SDPA backend.
# Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
# Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.11it/s]
# Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  2.11it/s]

# INFO 02-17 20:04:05 [component_loader.py:401] Loading weights took 0.48 seconds
# INFO 02-17 20:04:07 [composed_pipeline_base.py:363] Loaded module text_encoder from /tmp/huggingface_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/text_encoder
# INFO 02-17 20:04:07 [component_loader.py:972] Loading text_encoder_2 using transformers from /tmp/huggingface_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/text_encoder_2
# INFO 02-17 20:04:07 [component_loader.py:308] HF Model config: {'architectures': ['T5EncoderModel'], 'classifier_dropout': 0.0, 'd_ff': 10240, 'd_kv': 64, 'd_model': 4096, 'decoder_start_token_id': 0, 'dense_act_fn': 'gelu_new', 'dropout_rate': 0.1, 'eos_token_id': 1, 'feed_forward_proj': 'gated-gelu', 'initializer_factor': 1.0, 'is_encoder_decoder': True, 'is_gated_act': True, 'layer_norm_epsilon': 1e-06, 'num_decoder_layers': 24, 'num_heads': 64, 'num_layers': 24, 'output_past': True, 'pad_token_id': 0, 'relative_attention_max_distance': 128, 'relative_attention_num_buckets': 32, 'tie_word_embeddings': False, 'use_cache': True, 'vocab_size': 32128}
# Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]
# Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:04<00:04,  4.28s/it]
# Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:08<00:00,  4.12s/it]
# Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:08<00:00,  4.14s/it]

# INFO 02-17 20:04:15 [component_loader.py:401] Loading weights took 8.45 seconds
# INFO 02-17 20:04:42 [composed_pipeline_base.py:363] Loaded module text_encoder_2 from /tmp/huggingface_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/text_encoder_2
# INFO 02-17 20:04:42 [component_loader.py:972] Loading tokenizer using transformers from /tmp/huggingface_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/tokenizer
# INFO 02-17 20:04:42 [component_loader.py:522] Loading tokenizer from /tmp/huggingface_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/tokenizer
# INFO 02-17 20:04:42 [component_loader.py:570] Loaded tokenizer: CLIPTokenizerFast
# INFO 02-17 20:04:42 [composed_pipeline_base.py:363] Loaded module tokenizer from /tmp/huggingface_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/tokenizer
# INFO 02-17 20:04:42 [component_loader.py:972] Loading tokenizer_2 using transformers from /tmp/huggingface_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/tokenizer_2
# INFO 02-17 20:04:42 [component_loader.py:522] Loading tokenizer from /tmp/huggingface_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/tokenizer_2
# You set `add_prefix_space`. The tokenizer needs to be converted from the slow tokenizers
# INFO 02-17 20:04:43 [component_loader.py:570] Loaded tokenizer: T5TokenizerFast
# INFO 02-17 20:04:43 [composed_pipeline_base.py:363] Loaded module tokenizer_2 from /tmp/huggingface_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/tokenizer_2
# INFO 02-17 20:04:43 [component_loader.py:972] Loading transformer using diffusers from /tmp/huggingface_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/transformer
# INFO 02-17 20:04:43 [component_loader.py:765] transformer cls_name: FluxTransformer2DModel
# INFO 02-17 20:04:43 [component_loader.py:814] Loading model from 3 safetensors files: ['/tmp/huggingface_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/transformer/diffusion_pytorch_model-00001-of-00003.safetensors', '/tmp/huggingface_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/transformer/diffusion_pytorch_model-00002-of-00003.safetensors', '/tmp/huggingface_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/transformer/diffusion_pytorch_model-00003-of-00003.safetensors']
# INFO 02-17 20:04:43 [component_loader.py:825] Loading model from FluxTransformer2DModel, default_dtype: torch.bfloat16
# INFO 02-17 20:04:43 [fsdp_load.py:96] Loading model with default_dtype: torch.bfloat16
# INFO 02-17 20:04:43 [cuda.py:124] Trying FASTVIDEO_ATTENTION_BACKEND=None
# INFO 02-17 20:04:43 [cuda.py:126] Selected backend: None
# INFO 02-17 20:04:43 [cuda.py:267] Cannot use FlashAttention-2 backend because the flash_attn package is not found. Make sure that flash_attn was built and installed (on by default).
# INFO 02-17 20:04:43 [cuda.py:274] Using Torch SDPA backend.
# Loading safetensors checkpoint shards:   0% Completed | 0/3 [00:00<?, ?it/s]
# Loading safetensors checkpoint shards:  67% Completed | 2/3 [00:00<00:00, 11.15it/s]
# Loading safetensors checkpoint shards: 100% Completed | 3/3 [00:00<00:00, 15.12it/s]

# INFO 02-17 20:04:50 [component_loader.py:858] Loaded model with 11.90B parameters
# INFO 02-17 20:05:23 [composed_pipeline_base.py:363] Loaded module transformer from /tmp/huggingface_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/transformer
# INFO 02-17 20:05:23 [component_loader.py:972] Loading vae using diffusers from /tmp/huggingface_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/vae
# INFO 02-17 20:05:24 [composed_pipeline_base.py:363] Loaded module vae from /tmp/huggingface_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/vae
# INFO 02-17 20:05:24 [__init__.py:73] Pipelines instantiated
# (Worker pid=28357) INFO 02-17 20:05:24 [multiproc_executor.py:627] Worker 0 starting event loop...
# INFO 02-17 20:05:24 [multiproc_executor.py:618] 1 workers ready
# INFO 02-17 20:05:24 [video_generator.py:363] 
# INFO 02-17 20:05:24 [video_generator.py:363]                       height: 720
# INFO 02-17 20:05:24 [video_generator.py:363]                        width: 1280
# INFO 02-17 20:05:24 [video_generator.py:363]                 video_length: 1
# INFO 02-17 20:05:24 [video_generator.py:363]                       prompt: A cinematic portrait of a fox, 35mm film, soft light, gentle grain.
# INFO 02-17 20:05:24 [video_generator.py:363]                       image_path: None
# INFO 02-17 20:05:24 [video_generator.py:363]                   neg_prompt: 
# INFO 02-17 20:05:24 [video_generator.py:363]                         seed: 42
# INFO 02-17 20:05:24 [video_generator.py:363]                  infer_steps: 50
# INFO 02-17 20:05:24 [video_generator.py:363]        num_videos_per_prompt: 1
# INFO 02-17 20:05:24 [video_generator.py:363]               guidance_scale: 1.0
# INFO 02-17 20:05:24 [video_generator.py:363]                     n_tokens: 14400
# INFO 02-17 20:05:24 [video_generator.py:363]                   flow_shift: 3.0
# INFO 02-17 20:05:24 [video_generator.py:363]      embedded_guidance_scale: 0.0035
# INFO 02-17 20:05:24 [video_generator.py:363]                   save_video: False
# INFO 02-17 20:05:24 [video_generator.py:363]                   output_path: outputs/A cinematic portrait of a fox, 35mm film, soft light, gentle grain.mp4
# INFO 02-17 20:05:24 [video_generator.py:363]         
# (Worker pid=28357) INFO 02-17 20:05:24 [composed_pipeline_base.py:157] Creating pipeline stages...
# (Worker pid=28357) INFO 02-17 20:05:24 [cuda.py:124] Trying FASTVIDEO_ATTENTION_BACKEND=None
# (Worker pid=28357) INFO 02-17 20:05:24 [cuda.py:126] Selected backend: None
# (Worker pid=28357) INFO 02-17 20:05:24 [cuda.py:267] Cannot use FlashAttention-2 backend because the flash_attn package is not found. Make sure that flash_attn was built and installed (on by default).
# (Worker pid=28357) INFO 02-17 20:05:24 [cuda.py:274] Using Torch SDPA backend.
# (Worker pid=28357) INFO 02-17 20:05:24 [composed_pipeline_base.py:417] Running pipeline stages: dict_keys(['input_validation_stage', 'prompt_encoding_stage_primary', 'conditioning_stage', 'timestep_preparation_stage', 'latent_preparation_stage', 'denoising_stage', 'decoding_stage'])
# 100%|███████████████████████████████████████████████████████| 50/50 [00:56<00:00,  1.14s/it]
# (Worker pid=28357) INFO 02-17 20:06:23 [multiproc_executor.py:627] Worker 0 starting event loop...
# INFO 02-17 20:06:23 [video_generator.py:380] Generated successfully in 59.25 seconds

# ================================================================================
# FastVideo Output
# ================================================================================
# Shape: (720, 1280, 3)
#   Width×Height: 1280×720
#   Expected: 1280×720
#   Match: True

# Pixel stats:
#   min=0, max=255, mean=126.59
#   Top-left corner: [[[115 159 121]
#   [113 161 119]]

#  [[108 160 108]
#   [108 162 107]]]

# Saved to: fastvideo_corrected_dimensions.png

# ================================================================================
# SGLang Output (pre-generated)
# ================================================================================
# Shape: (720, 1280, 3)
#   Width×Height: 1280×720

# Pixel stats:
#   min=0, max=255, mean=89.84
#   Top-left corner: [[[68 62 46]
#   [65 56 40]]

#  [[65 58 40]
#   [67 59 40]]]

# ================================================================================
# COMPARISON
# ================================================================================
# ✅ Shapes match: (720, 1280, 3)

# Pixel difference:
#   max abs diff:  255.00
#   mean abs diff: 77.31
#   std abs diff:  55.41

#   Close pixels (diff<5): 1301/921600 (0.1%)

# ❌ SIGNIFICANT DIFFERENCE (mean diff >= 20 pixels)

# Possible causes:
#   - Different random seed (SGLang may use different default)
#   - Different num_inference_steps (SGLang may use different default)
#   - Different guidance parameters
#   - Different model precision or attention backend

# ================================================================================
# CONCLUSION
# ================================================================================

# We corrected the critical dimension error:
#   - Original test: 720×1280 (portrait) 
#   - Corrected test: 1280×720 (landscape) ✅

# This test will show whether dimensions were the only issue, or if there
# are other parameter differences (seed, steps, etc.) between FastVideo
# and SGLang's defaults that we still need to match.

# INFO 02-17 20:06:24 [multiproc_executor.py:329] Shutting down MultiprocExecutor...
# (Worker pid=28357) INFO 02-17 20:06:24 [gpu_worker.py:85] Worker 0 shutting down...
# (Worker pid=28357) INFO 02-17 20:06:24 [gpu_worker.py:96] Worker 0 shutdown complete
# (Worker pid=28357) INFO 02-17 20:06:24 [gpu_worker.py:85] Worker 0 shutting down...
# (Worker pid=28357) INFO 02-17 20:06:24 [gpu_worker.py:96] Worker 0 shutdown complete
# INFO 02-17 20:06:31 [multiproc_executor.py:382] MultiprocExecutor shutdown complete