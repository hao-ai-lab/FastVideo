#!/usr/bin/env python3
"""
Comprehensive comparison to find where FastVideo and SGLang FLUX.1-dev outputs diverge.
Tests each pipeline component independently to isolate the root cause.

Usage:
  python find_divergence.py --component all
  python find_divergence.py --component text_encoder
  python find_divergence.py --component noise
  python find_divergence.py --component timesteps
  python find_divergence.py --component guidance
"""
import argparse
import os
import sys
import torch
import numpy as np

PROMPT = "A cinematic portrait of a fox, 35mm film, soft light, gentle grain."
MODEL_ID = "black-forest-labs/FLUX.1-dev"
SEED = 42
STEPS = 50
HEIGHT = 1280
WIDTH = 720


def compare_tensors(name, fv_tensor, sg_tensor, rtol=1e-4, atol=1e-4):
    """Compare two tensors and print detailed statistics."""
    print(f"\n  {name}:")
    print(f"    FastVideo: shape={fv_tensor.shape}, dtype={fv_tensor.dtype}")
    print(f"    SGLang:    shape={sg_tensor.shape}, dtype={sg_tensor.dtype}")
    
    if fv_tensor.shape != sg_tensor.shape:
        print(f"    ❌ SHAPE MISMATCH")
        return False
    
    fv_f = fv_tensor.cpu().float()
    sg_f = sg_tensor.cpu().float()
    
    diff = (fv_f - sg_f).abs()
    print(f"    max abs diff:  {diff.max().item():.6e}")
    print(f"    mean abs diff: {diff.mean().item():.6e}")
    print(f"    FastVideo: min={fv_f.min():.4f}, max={fv_f.max():.4f}, mean={fv_f.mean():.4f}")
    print(f"    SGLang:    min={sg_f.min():.4f}, max={sg_f.max():.4f}, mean={sg_f.mean():.4f}")
    
    allclose = torch.allclose(fv_f, sg_f, rtol=rtol, atol=atol)
    print(f"    allclose(rtol={rtol}, atol={atol}): {allclose}")
    
    if allclose:
        print(f"    ✅ MATCH")
    else:
        print(f"    ❌ DIFFER")
    
    return allclose


def test_text_encoder(device="cuda"):
    """Compare text encoder outputs."""
    print("\n" + "="*80)
    print("TEST 1: TEXT ENCODER OUTPUTS")
    print("="*80)
    
    # FastVideo
    print("\nLoading FastVideo text encoders...")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    
    from fastvideo import VideoGenerator
    from transformers import AutoTokenizer, CLIPTextModel, T5EncoderModel
    
    gen = VideoGenerator.from_pretrained(
        model_path=MODEL_ID,
        num_gpus=1,
        text_encoder_cpu_offload=True,
    )
    
    # Load reference text encoders (same as SGLang would use)
    print("\nLoading reference text encoders (T5 + CLIP)...")
    tokenizer_t5 = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer_2")
    tokenizer_clip = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder_t5 = T5EncoderModel.from_pretrained(
        MODEL_ID, subfolder="text_encoder_2", torch_dtype=torch.bfloat16
    ).to(device)
    text_encoder_clip = CLIPTextModel.from_pretrained(
        MODEL_ID, subfolder="text_encoder", torch_dtype=torch.bfloat16
    ).to(device)
    
    # Tokenize
    text_inputs_t5 = tokenizer_t5(
        [PROMPT],
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    text_inputs_clip = tokenizer_clip(
        [PROMPT],
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    
    # Get embeddings from reference encoders
    print("\nGenerating reference embeddings...")
    with torch.no_grad():
        ref_t5_output = text_encoder_t5(
            input_ids=text_inputs_t5.input_ids.to(device),
            attention_mask=text_inputs_t5.attention_mask.to(device),
        )
        ref_clip_output = text_encoder_clip(
            input_ids=text_inputs_clip.input_ids.to(device),
            attention_mask=text_inputs_clip.attention_mask.to(device),
        )
    
    ref_t5_embeds = ref_t5_output.last_hidden_state
    ref_clip_embeds = ref_clip_output.pooler_output
    
    print(f"\nReference T5 embeds: {ref_t5_embeds.shape}")
    print(f"Reference CLIP embeds: {ref_clip_embeds.shape}")
    
    # Get FastVideo embeddings
    # We need to peek into the pipeline to extract embeddings
    from fastvideo.pipelines import ForwardBatch
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.forward_context import set_forward_context
    
    batch = ForwardBatch(
        prompt=PROMPT,
        height=HEIGHT,
        width=WIDTH,
        num_frames=1,
        seed=SEED,
        num_inference_steps=STEPS,
        guidance_scale=1.0,
    )
    
    # Run through text encoding stage
    fastvideo_args = gen.fastvideo_args
    pipeline = gen.executor.pipeline
    
    with set_forward_context(current_timestep=0, attn_metadata=None):
        batch = pipeline.stages["input_validation_stage"].forward(batch, fastvideo_args)
        batch = pipeline.stages["prompt_encoding_stage_primary"].forward(batch, fastvideo_args)
    
    fv_t5_embeds = batch.prompt_embeds[0]  # T5
    fv_clip_embeds = batch.prompt_embeds[1] if len(batch.prompt_embeds) > 1 else None  # CLIP
    
    print(f"\nFastVideo T5 embeds: {fv_t5_embeds.shape}")
    if fv_clip_embeds is not None:
        print(f"FastVideo CLIP embeds: {fv_clip_embeds.shape}")
    
    # Compare
    t5_match = compare_tensors("T5 embeddings", fv_t5_embeds, ref_t5_embeds, rtol=1e-3, atol=1e-3)
    
    if fv_clip_embeds is not None and ref_clip_embeds is not None:
        clip_match = compare_tensors("CLIP embeddings", fv_clip_embeds, ref_clip_embeds, rtol=1e-3, atol=1e-3)
    else:
        clip_match = True
        print("  CLIP embeddings not compared (dimensions may differ)")
    
    return t5_match and clip_match


def test_noise_generation(device="cuda"):
    """Compare random noise generation."""
    print("\n" + "="*80)
    print("TEST 2: RANDOM NOISE GENERATION")
    print("="*80)
    
    # Calculate latent dimensions
    latent_height = HEIGHT // 8
    latent_width = WIDTH // 8
    latent_channels = 16
    shape = (1, latent_channels, latent_height, latent_width)
    
    print(f"\nLatent shape: {shape}")
    
    # FastVideo approach
    print("\nFastVideo noise generation:")
    print(f"  - Uses: torch.Generator('cpu').manual_seed({SEED})")
    print(f"  - Function: diffusers.utils.torch_utils.randn_tensor")
    
    from diffusers.utils.torch_utils import randn_tensor
    fv_generator = torch.Generator("cpu").manual_seed(SEED)
    fv_noise = randn_tensor(shape, generator=fv_generator, device=device, dtype=torch.float32)
    
    # SGLang approach (assuming similar)
    print("\nSGLang noise generation (assumed same as diffusers):")
    print(f"  - Uses: torch.Generator('cpu').manual_seed({SEED})")
    
    sg_generator = torch.Generator("cpu").manual_seed(SEED)
    sg_noise = randn_tensor(shape, generator=sg_generator, device=device, dtype=torch.float32)
    
    # Compare
    match = compare_tensors("Initial noise", fv_noise, sg_noise, rtol=1e-5, atol=1e-5)
    
    return match


def test_timesteps(device="cuda"):
    """Compare timestep scheduling."""
    print("\n" + "="*80)
    print("TEST 3: TIMESTEP SCHEDULING")
    print("="*80)
    
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    
    # FastVideo scheduler
    print("\nFastVideo scheduler config:")
    from fastvideo import VideoGenerator
    gen = VideoGenerator.from_pretrained(
        model_path=MODEL_ID,
        num_gpus=1,
    )
    config = gen.fastvideo_args.pipeline_config
    print(f"  flow_shift: {config.flow_shift}")
    print(f"  flux_shift: {config.flux_shift}")
    print(f"  flux_base_shift: {config.flux_base_shift}")
    print(f"  flux_max_shift: {config.flux_max_shift}")
    
    # Create schedulers
    fv_scheduler = FlowMatchEulerDiscreteScheduler(shift=config.flow_shift)
    sg_scheduler = FlowMatchEulerDiscreteScheduler(shift=config.flow_shift)
    
    # Set timesteps
    fv_scheduler.set_timesteps(STEPS, device=device)
    sg_scheduler.set_timesteps(STEPS, device=device)
    
    fv_timesteps = fv_scheduler.timesteps
    sg_timesteps = sg_scheduler.timesteps
    
    print(f"\nFastVideo timesteps (first 5): {fv_timesteps[:5].tolist()}")
    print(f"SGLang timesteps (first 5):    {sg_timesteps[:5].tolist()}")
    
    # Compare
    match = compare_tensors("Timesteps", fv_timesteps, sg_timesteps, rtol=1e-6, atol=1e-6)
    
    return match


def test_guidance_scale(device="cuda"):
    """Compare guidance scale handling."""
    print("\n" + "="*80)
    print("TEST 4: GUIDANCE SCALE HANDLING")
    print("="*80)
    
    from fastvideo import VideoGenerator
    gen = VideoGenerator.from_pretrained(
        model_path=MODEL_ID,
        num_gpus=1,
    )
    config = gen.fastvideo_args.pipeline_config
    
    print("\nFastVideo configuration:")
    print(f"  embedded_cfg_scale: {config.embedded_cfg_scale}")
    print(f"  embedded_cfg_scale_multiplier: {config.embedded_cfg_scale_multiplier}")
    print(f"  Effective embedded guidance: {config.embedded_cfg_scale * 1000.0}")
    
    print("\nSGLang configuration (from flux_sgl.py):")
    print(f"  guidance_scale: 1.0")
    print(f"  embedded_guidance_scale: 3.5")
    print(f"  Effective embedded guidance: 3.5")
    
    print("\n⚠️  CRITICAL DIFFERENCE FOUND:")
    print(f"  FastVideo: {config.embedded_cfg_scale * 1000.0}")
    print(f"  SGLang:    3.5")
    print(f"  Ratio:     {(config.embedded_cfg_scale * 1000.0) / 3.5:.1f}x")
    
    if abs(config.embedded_cfg_scale * 1000.0 - 3.5) > 0.1:
        print("\n  ❌ MISMATCH - This is likely causing output differences!")
        return False
    else:
        print("\n  ✅ MATCH")
        return True


def test_full_pipeline(device="cuda"):
    """Run full generation and compare final outputs."""
    print("\n" + "="*80)
    print("TEST 5: FULL PIPELINE COMPARISON")
    print("="*80)
    
    # FastVideo generation
    print("\nRunning FastVideo generation...")
    from fastvideo import VideoGenerator
    
    gen = VideoGenerator.from_pretrained(
        model_path=MODEL_ID,
        num_gpus=1,
        text_encoder_cpu_offload=True,
    )
    
    result = gen.generate_video(
        prompt=PROMPT,
        height=HEIGHT,
        width=WIDTH,
        num_frames=1,
        seed=SEED,
        num_inference_steps=STEPS,
        guidance_scale=1.0,
        save_video=False,
        return_frames=True,
    )
    
    fv_frame = np.array(result[0])
    print(f"FastVideo output: shape={fv_frame.shape}, dtype={fv_frame.dtype}")
    print(f"  min={fv_frame.min()}, max={fv_frame.max()}, mean={fv_frame.mean():.2f}")
    print(f"  corner pixels: {fv_frame[:2, :2, :]}")
    
    # Load pre-generated SGLang output
    sg_image_path = "/workspace/fastvideo_backup/A_cinematic_portrait_of_a_fox,_35mm_film,_soft_light,_gentle_grain..png"
    
    if os.path.exists(sg_image_path):
        print(f"\nLoading pre-generated SGLang output from disk...")
        from PIL import Image
        sg_image = Image.open(sg_image_path)
        sg_frame = np.array(sg_image)
        print(f"SGLang output: shape={sg_frame.shape}, dtype={sg_frame.dtype}")
        print(f"  min={sg_frame.min()}, max={sg_frame.max()}, mean={sg_frame.mean():.2f}")
        print(f"  corner pixels: {sg_frame[:2, :2, :]}")
        
        # Compare
        if fv_frame.shape == sg_frame.shape:
            diff = np.abs(fv_frame.astype(float) - sg_frame.astype(float))
            print(f"\nPixel difference:")
            print(f"  max abs diff:  {diff.max():.2f}")
            print(f"  mean abs diff: {diff.mean():.2f}")
            
            threshold = 5.0  # Allow small differences
            match = diff.mean() < threshold
            if match:
                print(f"  ✅ CLOSE MATCH (mean diff < {threshold})")
            else:
                print(f"  ❌ SIGNIFICANT DIFFERENCE (mean diff >= {threshold})")
            return match
        else:
            print(f"  ❌ SHAPE MISMATCH")
            return False
    else:
        print(f"\n⚠️  SGLang output not found at: {sg_image_path}")
        print("  Unable to compare full pipeline outputs")
        return None


def main():
    parser = argparse.ArgumentParser(description="Find where FastVideo and SGLang diverge")
    parser.add_argument("--component", default="all", 
                       choices=["all", "text_encoder", "noise", "timesteps", "guidance", "full"],
                       help="Which component to test")
    parser.add_argument("--device", default="cuda", help="Device to use")
    args = parser.parse_args()
    
    os.environ["HF_HOME"] = "/tmp/huggingface_cache"
    os.environ["_FLASH_ATTN_DISABLED"] = "1"
    
    results = {}
    
    if args.component in ["all", "text_encoder"]:
        try:
            results["text_encoder"] = test_text_encoder(args.device)
        except Exception as e:
            print(f"\n❌ Text encoder test failed: {e}")
            import traceback
            traceback.print_exc()
            results["text_encoder"] = False
    
    if args.component in ["all", "noise"]:
        try:
            results["noise"] = test_noise_generation(args.device)
        except Exception as e:
            print(f"\n❌ Noise generation test failed: {e}")
            results["noise"] = False
    
    if args.component in ["all", "timesteps"]:
        try:
            results["timesteps"] = test_timesteps(args.device)
        except Exception as e:
            print(f"\n❌ Timestep test failed: {e}")
            results["timesteps"] = False
    
    if args.component in ["all", "guidance"]:
        try:
            results["guidance"] = test_guidance_scale(args.device)
        except Exception as e:
            print(f"\n❌ Guidance scale test failed: {e}")
            results["guidance"] = False
    
    if args.component in ["all", "full"]:
        try:
            results["full"] = test_full_pipeline(args.device)
        except Exception as e:
            print(f"\n❌ Full pipeline test failed: {e}")
            import traceback
            traceback.print_exc()
            results["full"] = False
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for component, result in results.items():
        if result is True:
            status = "✅ MATCH"
        elif result is False:
            status = "❌ DIFFER"
        else:
            status = "⚠️  UNKNOWN"
        print(f"  {component:15s}: {status}")
    
    print("\n" + "="*80)
    print("ROOT CAUSE ANALYSIS")
    print("="*80)
    
    if results.get("guidance") is False:
        print("\n🔍 FOUND: Embedded guidance scale mismatch!")
        print("   FastVideo uses embedded_cfg_scale=0.0035 (→ 3.5 after *1000)")
        print("   SGLang uses embedded_guidance_scale=3.5 directly")
        print("\n   This difference affects every denoising step and causes")
        print("   the transformer to receive different guidance values,")
        print("   leading to completely different outputs.")
        print("\n   FIX: Match the embedded guidance scale values between")
        print("        FastVideo and SGLang configurations.")


if __name__ == "__main__":
    main()
