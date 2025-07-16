#!/usr/bin/env python3
"""
Simple log comparison script for Cosmos2 Video2World.


This script runs both diffusers and FastVideo implementations and compares
their logs to identify any discrepancies in behavior.
"""


import re
import json
import subprocess
import sys
import os
import time
from typing import Dict, List, Any


def run_diffusers_test():
    """Run the diffusers implementation and capture logs."""
    print("=" * 80)
    print("RUNNING DIFFUSERS IMPLEMENTATION")
    print("=" * 80)
   
    # Check if test_diffusers.py exists
    if not os.path.exists("test_diffusers.py"):
        print("❌ test_diffusers.py not found. Please ensure the file exists.")
        return None
   
    try:
        result = subprocess.run(
            [sys.executable, "test_diffusers.py"],
            capture_output=True,
            text=True,
            timeout=600  # 5 minutes timeout
        )
       
        # Always print the logs for debugging
        print("--- DIFFUSERS STDOUT ---")
        print(result.stdout)
        print("--- DIFFUSERS STDERR ---")
        print(result.stderr)
        print(f"--- EXIT CODE: {result.returncode} ---")
       
        if result.returncode == 0:
            print("✅ Diffusers test completed successfully")
            # Combine stdout and stderr for log processing
            combined_logs = result.stdout + result.stderr
            return combined_logs
        else:
            print(f"❌ Diffusers test failed (exit code {result.returncode})")
            return None
           
    except subprocess.TimeoutExpired:
        print("❌ Diffusers test timed out")
        return None
    except Exception as e:
        print(f"❌ Diffusers test error: {e}")
        return None


def run_fastvideo_test():
    """Run the FastVideo implementation and capture logs."""
    print("=" * 80)
    print("RUNNING FASTVIDEO IMPLEMENTATION")
    print("=" * 80)
   
    # Check if test_fastvideo.py exists
    if not os.path.exists("test_fastvideo.py"):
        print("❌ test_fastvideo.py not found. Please ensure the file exists.")
        return None
   
    try:
        result = subprocess.run(
            [sys.executable, "test_fastvideo.py"],
            capture_output=True,
            text=True,
            timeout=600  # 5 minutes timeout
        )
       
        # Always print the logs for debugging
        print("--- FASTVIDEO STDOUT ---")
        print(result.stdout)
        print("--- FASTVIDEO STDERR ---")
        print(result.stderr)
        print(f"--- EXIT CODE: {result.returncode} ---")
       
        if result.returncode == 0:
            print("✅ FastVideo test completed successfully")
            # Combine stdout and stderr for log processing
            combined_logs = result.stdout + result.stderr
            return combined_logs
        else:
            print(f"❌ FastVideo test failed (exit code {result.returncode})")
            return None
           
    except subprocess.TimeoutExpired:
        print("❌ FastVideo test timed out")
        return None
    except Exception as e:
        print(f"❌ FastVideo test error: {e}")
        return None


def parse_logs(log_output: str, implementation: str) -> Dict[str, Any]:
    """Parse logs to extract key information for comparison."""
    parsed = {
        'implementation': implementation,
        'text_encoding': {},
        'vae_encoding': {},
        'denoising': {},
        'decoding': {},
        'timing': {},
        'conditioning': {},
        'latent_stats': {},
        'diagnostics': {},  # New section for diagnostic logs
        'errors': []
    }
   
    lines = log_output.split('\n')
   
    for line in lines:
        # Extract text encoding info
        if '[TEXT]' in line:
            if 'Prompt embeddings:' in line and 'shape:' in line:
                parsed['text_encoding']['prompt_shape'] = line.split('shape:')[1].strip()
            elif 'Negative prompt embeddings:' in line and 'shape:' in line:
                parsed['text_encoding']['negative_shape'] = line.split('shape:')[1].strip()
            elif 'range:' in line:
                if 'Prompt embeddings range:' in line:
                    parsed['text_encoding']['prompt_range'] = line.split('range:')[1].strip()
                elif 'Negative prompt embeddings range:' in line:
                    parsed['text_encoding']['negative_range'] = line.split('range:')[1].strip()
            elif 'mean:' in line:
                if 'Prompt embeddings mean:' in line:
                    parsed['text_encoding']['prompt_mean'] = line.split('mean:')[1].strip()
                elif 'Negative prompt embeddings mean:' in line:
                    parsed['text_encoding']['negative_mean'] = line.split('mean:')[1].strip()
            elif 'std:' in line:
                if 'Prompt embeddings std:' in line:
                    parsed['text_encoding']['prompt_std'] = line.split('std:')[1].strip()
                elif 'Negative prompt embeddings std:' in line:
                    parsed['text_encoding']['negative_std'] = line.split('std:')[1].strip()
            elif 'sum:' in line:
                if 'Prompt embeddings sum:' in line:
                    parsed['text_encoding']['prompt_sum'] = line.split('sum:')[1].strip()
                elif 'Negative prompt embeddings sum:' in line:
                    parsed['text_encoding']['negative_sum'] = line.split('sum:')[1].strip()
            elif 'abs_sum:' in line:
                if 'Prompt embeddings abs_sum:' in line:
                    parsed['text_encoding']['prompt_abs_sum'] = line.split('abs_sum:')[1].strip()
                elif 'Negative prompt embeddings abs_sum:' in line:
                    parsed['text_encoding']['negative_abs_sum'] = line.split('abs_sum:')[1].strip()
            elif 'abs_max:' in line:
                if 'Prompt embeddings abs_max:' in line:
                    parsed['text_encoding']['prompt_abs_max'] = line.split('abs_max:')[1].strip()
                elif 'Negative prompt embeddings abs_max:' in line:
                    parsed['text_encoding']['negative_abs_max'] = line.split('abs_max:')[1].strip()
            elif 'norm:' in line:
                if 'Prompt embeddings norm:' in line:
                    parsed['text_encoding']['prompt_norm'] = line.split('norm:')[1].strip()
                elif 'Negative prompt embeddings norm:' in line:
                    parsed['text_encoding']['negative_norm'] = line.split('norm:')[1].strip()
       
        # Extract VAE encoding info
        elif '[VAE]' in line:
            if 'Input video stats:' in line and 'shape:' in line:
                parsed['vae_encoding']['input_shape'] = line.split('shape:')[1].strip()
            elif 'Encoded latent shape:' in line:
                parsed['vae_encoding']['latent_shape'] = line.split('shape:')[1].strip()
            elif 'Conditioning first' in line:
                parsed['vae_encoding']['conditioning_frames'] = line.split('Conditioning first')[1].split('latent frames')[0].strip()
            elif 'Encoded latent stats:' in line:
                parsed['vae_encoding']['encoded_stats'] = line.split('stats:')[1].strip()
            elif 'Input video stats:' in line and 'mean:' in line:
                # Extract mean and std from input video stats
                stats_part = line.split('mean:')[1].strip()
                if 'std:' in stats_part:
                    mean_val = stats_part.split(',')[0].strip()
                    std_val = stats_part.split('std:')[1].strip()
                    parsed['vae_encoding']['input_mean'] = mean_val
                    parsed['vae_encoding']['input_std'] = std_val
            elif 'Input video range:' in line:
                parsed['vae_encoding']['input_range'] = line.split('range:')[1].strip()
            elif 'Input video sum:' in line:
                parsed['vae_encoding']['input_sum'] = line.split('sum:')[1].strip()
            elif 'Input video abs_sum:' in line:
                parsed['vae_encoding']['input_abs_sum'] = line.split('abs_sum:')[1].strip()
            elif 'Input video abs_max:' in line:
                parsed['vae_encoding']['input_abs_max'] = line.split('abs_max:')[1].strip()
            elif 'Input video norm:' in line:
                parsed['vae_encoding']['input_norm'] = line.split('norm:')[1].strip()
            elif 'Encoded latent range:' in line:
                parsed['vae_encoding']['encoded_range'] = line.split('range:')[1].strip()
            elif 'Encoded latent sum:' in line:
                parsed['vae_encoding']['encoded_sum'] = line.split('sum:')[1].strip()
            elif 'Encoded latent abs_sum:' in line:
                parsed['vae_encoding']['encoded_abs_sum'] = line.split('abs_sum:')[1].strip()
            elif 'Encoded latent abs_max:' in line:
                parsed['vae_encoding']['encoded_abs_max'] = line.split('abs_max:')[1].strip()
            elif 'Encoded latent norm:' in line:
                parsed['vae_encoding']['encoded_norm'] = line.split('norm:')[1].strip()
            elif 'Conditioning indicator:' in line:
                parsed['vae_encoding']['conditioning_indicator'] = line.split('Conditioning indicator:')[1].strip()
       
        # Extract denoising info
        elif '[DENOISE]' in line:
            if 'latents shape:' in line:
                parsed['denoising']['latents_shape'] = line.split('shape:')[1].strip()
            elif 'noise_pred shape:' in line:
                parsed['denoising']['noise_pred_shape'] = line.split('shape:')[1].strip()
            elif 'guidance_scale:' in line:
                parsed['denoising']['guidance_scale'] = line.split('guidance_scale:')[1].strip()
            elif 'noise_pred stats:' in line:
                parsed['denoising']['noise_pred_stats'] = line.split('stats:')[1].strip()
            elif 'noise_pred range:' in line:
                parsed['denoising']['noise_pred_range'] = line.split('range:')[1].strip()
            elif 'noise_pred sum:' in line:
                parsed['denoising']['noise_pred_sum'] = line.split('sum:')[1].strip()
            elif 'noise_pred abs_sum:' in line:
                parsed['denoising']['noise_pred_abs_sum'] = line.split('abs_sum:')[1].strip()
            elif 'noise_pred abs_max:' in line:
                parsed['denoising']['noise_pred_abs_max'] = line.split('abs_max:')[1].strip()
            elif 'noise_pred norm:' in line:
                parsed['denoising']['noise_pred_norm'] = line.split('norm:')[1].strip()
            elif 'latents range:' in line:
                parsed['denoising']['latents_range'] = line.split('range:')[1].strip()
            elif 'latents mean:' in line:
                parsed['denoising']['latents_mean'] = line.split('mean:')[1].strip()
            elif 'latents std:' in line:
                parsed['denoising']['latents_std'] = line.split('std:')[1].strip()
            elif 'latents sum:' in line:
                parsed['denoising']['latents_sum'] = line.split('sum:')[1].strip()
            elif 'latents abs_sum:' in line:
                parsed['denoising']['latents_abs_sum'] = line.split('abs_sum:')[1].strip()
            elif 'latents abs_max:' in line:
                parsed['denoising']['latents_abs_max'] = line.split('abs_max:')[1].strip()
            elif 'latents norm:' in line:
                parsed['denoising']['latents_norm'] = line.split('norm:')[1].strip()
            elif 'c_in=' in line and 'c_skip=' in line and 'c_out=' in line:
                parsed['denoising']['coefficients'] = line.split('c_in=')[1].strip()
            elif 'latents per frame means:' in line:
                parsed['denoising']['per_frame_means'] = line.split('latents per frame means:')[1].strip()
       
        # Extract conditioning info
        elif 'Conditioning indicator:' in line:
            parsed['conditioning']['indicator'] = line.split('Conditioning indicator:')[1].strip()
        elif 'cond_indicator values:' in line:
            parsed['conditioning']['indicator_values'] = line.split('cond_indicator values:')[1].strip()
        elif 'conditioning_latents mean:' in line:
            parsed['conditioning']['latents_mean'] = line.split('conditioning_latents mean:')[1].strip()
        elif 'latents (noise) mean:' in line:
            parsed['conditioning']['noise_mean'] = line.split('latents (noise) mean:')[1].strip()
        elif 'cond_latent mean:' in line:
            parsed['conditioning']['cond_latent_mean'] = line.split('cond_latent mean:')[1].strip()
        elif 'final noise_pred mean:' in line:
            parsed['conditioning']['final_noise_pred_mean'] = line.split('final noise_pred mean:')[1].strip()
        elif 'final latents mean:' in line:
            parsed['conditioning']['final_latents_mean'] = line.split('final latents mean:')[1].strip()
        elif 'First frame cond_latent mean:' in line:
            parsed['conditioning']['first_frame_cond_mean'] = line.split('mean:')[1].strip()
        elif 'Other frames cond_latent mean:' in line:
            parsed['conditioning']['other_frames_cond_mean'] = line.split('mean:')[1].strip()
        elif 'Conditioning difference:' in line:
            parsed['conditioning']['conditioning_difference'] = line.split('difference:')[1].strip()
        elif 'First frame final mean:' in line:
            parsed['conditioning']['first_frame_final_mean'] = line.split('mean:')[1].strip()
        elif 'Other frames final mean:' in line:
            parsed['conditioning']['other_frames_final_mean'] = line.split('mean:')[1].strip()
        elif 'Final conditioning difference:' in line:
            parsed['conditioning']['final_conditioning_difference'] = line.split('difference:')[1].strip()
       
        # Extract latent statistics
        elif 'latents per frame means:' in line:
            parsed['latent_stats']['per_frame_means'] = line.split('latents per frame means:')[1].strip()
        elif 'noise_pred stats:' in line:
            parsed['latent_stats']['noise_pred_stats'] = line.split('noise_pred stats:')[1].strip()
        elif 'c_in=' in line and 'c_skip=' in line and 'c_out=' in line:
            parsed['latent_stats']['coefficients'] = line.split('c_in=')[1].strip()
        elif 'Per frame means:' in line:
            parsed['latent_stats']['per_frame_means_alt'] = line.split('means:')[1].strip()
        elif 'First frame mean:' in line:
            parsed['latent_stats']['first_frame_mean'] = line.split('mean:')[1].strip()
        elif 'Other frames mean:' in line:
            parsed['latent_stats']['other_frames_mean'] = line.split('mean:')[1].strip()
        elif 'Conditioning difference:' in line and 'conditioning' in line.lower():
            parsed['latent_stats']['conditioning_difference'] = line.split('difference:')[1].strip()
        elif 'Samples mean:' in line:
            parsed['latent_stats']['samples_mean'] = line.split('mean:')[1].strip()
        elif 'Samples std:' in line:
            parsed['latent_stats']['samples_std'] = line.split('std:')[1].strip()
       
        # Extract timing info
        elif 'seconds' in line and ('Generated successfully' in line or 'Generation completed' in line):
            parsed['timing']['generation_time'] = line.split('seconds')[0].split()[-1]
       
        # Extract diagnostic logs
        elif '[PATCH]' in line and 'shape:' in line:
            parsed['diagnostics']['patch_shape'] = line.split('shape:')[1].strip()
        elif '[ROPE]' in line and 'cos mean:' in line:
            parsed['diagnostics']['rope_stats'] = line.split('[ROPE]')[1].strip()
        elif '[ATTN]' in line and ('mean:' in line or 'attention_weights:' in line):
            parsed['diagnostics']['attention_stats'] = line.split('[ATTN]')[1].strip()
        elif '[BLOCK]' in line and 'hidden state mean:' in line:
            if 'Pre-attn' in line:
                parsed['diagnostics']['pre_attn_stats'] = line.split('[BLOCK]')[1].strip()
            elif 'Post-attn' in line:
                parsed['diagnostics']['post_attn_stats'] = line.split('[BLOCK]')[1].strip()
            elif 'Post-ff' in line:
                parsed['diagnostics']['post_ff_stats'] = line.split('[BLOCK]')[1].strip()
        elif '[COND_MASK]' in line and 'mean:' in line:
            parsed['diagnostics']['cond_mask_stats'] = line.split('[COND_MASK]')[1].strip()
        elif '[TEMPORAL]' in line and 'diffs:' in line:
            parsed['diagnostics']['temporal_diffs'] = line.split('diffs:')[1].strip()
        elif '[CFG]' in line and 'mean:' in line:
            parsed['diagnostics']['cfg_stats'] = line.split('[CFG]')[1].strip()
        elif '[VAE_DECODE]' in line:
            if 'Input latents shape:' in line:
                parsed['decoding']['input_latents_shape'] = line.split('shape:')[1].strip()
            elif 'Input latents stats:' in line:
                parsed['decoding']['input_latents_stats'] = line.split('stats:')[1].strip()
            elif 'Input latents range:' in line:
                parsed['decoding']['input_latents_range'] = line.split('range:')[1].strip()
            elif 'Input latents sum:' in line:
                parsed['decoding']['input_latents_sum'] = line.split('sum:')[1].strip()
            elif 'Input latents abs_sum:' in line:
                parsed['decoding']['input_latents_abs_sum'] = line.split('abs_sum:')[1].strip()
            elif 'Input latents abs_max:' in line:
                parsed['decoding']['input_latents_abs_max'] = line.split('abs_max:')[1].strip()
            elif 'Input latents norm:' in line:
                parsed['decoding']['input_latents_norm'] = line.split('norm:')[1].strip()
            elif 'After normalization stats:' in line:
                parsed['decoding']['normalized_stats'] = line.split('stats:')[1].strip()
            elif 'After normalization range:' in line:
                parsed['decoding']['normalized_range'] = line.split('range:')[1].strip()
            elif 'After normalization sum:' in line:
                parsed['decoding']['normalized_sum'] = line.split('sum:')[1].strip()
            elif 'After normalization abs_sum:' in line:
                parsed['decoding']['normalized_abs_sum'] = line.split('abs_sum:')[1].strip()
            elif 'After normalization abs_max:' in line:
                parsed['decoding']['normalized_abs_max'] = line.split('abs_max:')[1].strip()
            elif 'After normalization norm:' in line:
                parsed['decoding']['normalized_norm'] = line.split('norm:')[1].strip()
            elif 'VAE output shape:' in line:
                parsed['decoding']['vae_output_shape'] = line.split('shape:')[1].strip()
            elif 'VAE output stats:' in line:
                parsed['decoding']['vae_output_stats'] = line.split('stats:')[1].strip()
            elif 'VAE output range:' in line:
                parsed['decoding']['vae_output_range'] = line.split('range:')[1].strip()
            elif 'VAE output sum:' in line:
                parsed['decoding']['vae_output_sum'] = line.split('sum:')[1].strip()
            elif 'VAE output abs_sum:' in line:
                parsed['decoding']['vae_output_abs_sum'] = line.split('abs_sum:')[1].strip()
            elif 'VAE output abs_max:' in line:
                parsed['decoding']['vae_output_abs_max'] = line.split('abs_max:')[1].strip()
            elif 'VAE output norm:' in line:
                parsed['decoding']['vae_output_norm'] = line.split('norm:')[1].strip()
            elif 'Final video shape:' in line:
                parsed['decoding']['final_video_shape'] = line.split('shape:')[1].strip()
            elif 'Final video stats:' in line:
                parsed['decoding']['final_video_stats'] = line.split('stats:')[1].strip()
            elif 'Final video range:' in line:
                parsed['decoding']['final_video_range'] = line.split('range:')[1].strip()
            elif 'Final video sum:' in line:
                parsed['decoding']['final_video_sum'] = line.split('sum:')[1].strip()
            elif 'Final video abs_sum:' in line:
                parsed['decoding']['final_video_abs_sum'] = line.split('abs_sum:')[1].strip()
            elif 'Final video abs_max:' in line:
                parsed['decoding']['final_video_abs_max'] = line.split('abs_max:')[1].strip()
            elif 'Final video norm:' in line:
                parsed['decoding']['final_video_norm'] = line.split('norm:')[1].strip()
       
        # Extract step-specific debugging info
        elif 'Step ' in line and 'cond_latent mean:' in line:
            parsed['conditioning']['step_cond_latent_mean'] = line.split('mean:')[1].strip()
        elif 'Step ' in line and 'cond_latent range:' in line:
            parsed['conditioning']['step_cond_latent_range'] = line.split('range:')[1].strip()
        elif 'Step ' in line and 'cond_latent std:' in line:
            parsed['conditioning']['step_cond_latent_std'] = line.split('std:')[1].strip()
        elif 'Step ' in line and 'cond_latent abs_max:' in line:
            parsed['conditioning']['step_cond_latent_abs_max'] = line.split('abs_max:')[1].strip()
        elif 'Step ' in line and 'raw transformer output mean:' in line:
            parsed['conditioning']['step_raw_transformer_mean'] = line.split('mean:')[1].strip()
        elif 'Step ' in line and 'raw transformer output range:' in line:
            parsed['conditioning']['step_raw_transformer_range'] = line.split('range:')[1].strip()
        elif 'Step ' in line and 'raw transformer output std:' in line:
            parsed['conditioning']['step_raw_transformer_std'] = line.split('std:')[1].strip()
        elif 'Step ' in line and 'after coefficients mean:' in line:
            parsed['conditioning']['step_after_coefficients_mean'] = line.split('mean:')[1].strip()
        elif 'Step ' in line and 'after coefficients range:' in line:
            parsed['conditioning']['step_after_coefficients_range'] = line.split('range:')[1].strip()
        elif 'Step ' in line and 'final noise_pred mean:' in line:
            parsed['conditioning']['step_final_noise_pred_mean'] = line.split('mean:')[1].strip()
        elif 'Step ' in line and 'final noise_pred range:' in line:
            parsed['conditioning']['step_final_noise_pred_range'] = line.split('range:')[1].strip()
        elif 'Step ' in line and 'final noise_pred std:' in line:
            parsed['conditioning']['step_final_noise_pred_std'] = line.split('std:')[1].strip()
        elif 'Step ' in line and 'noise_pred_final mean:' in line:
            parsed['conditioning']['step_noise_pred_final_mean'] = line.split('mean:')[1].strip()
        elif 'Step ' in line and 'final latents mean:' in line:
            parsed['conditioning']['step_final_latents_mean'] = line.split('mean:')[1].strip()
        elif 'Step ' in line and 'latents mean:' in line and 'noise_pred mean:' in line:
            # Extract both latents and noise_pred means from the same line
            parts = line.split('latents mean:')[1].split(',')
            latents_mean = parts[0].strip()
            noise_pred_mean = parts[1].split('noise_pred mean:')[1].strip()
            parsed['conditioning']['step_latents_mean'] = latents_mean
            parsed['conditioning']['step_noise_pred_mean'] = noise_pred_mean
        elif 'Step ' in line and 'current_sigma:' in line:
            parsed['conditioning']['step_current_sigma'] = line.split('current_sigma:')[1].strip()
       
        # Extract errors
        elif 'ERROR' in line or '❌' in line:
            parsed['errors'].append(line.strip())
   
    return parsed


def analyze_logs(parsed_logs: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the parsed logs and identify potential issues."""
    analysis = {
        'issues': [],
        'recommendations': [],
        'metrics': {}
    }
   
    # Check text encoding
    if parsed_logs['text_encoding']:
        analysis['metrics']['text_encoding'] = {
            'prompt_shape': parsed_logs['text_encoding'].get('prompt_shape'),
            'negative_shape': parsed_logs['text_encoding'].get('negative_shape'),
            'prompt_range': parsed_logs['text_encoding'].get('prompt_range'),
            'negative_range': parsed_logs['text_encoding'].get('negative_range')
        }
       
        # Check if shapes match
        if ('prompt_shape' in parsed_logs['text_encoding'] and
            'negative_shape' in parsed_logs['text_encoding']):
            if parsed_logs['text_encoding']['prompt_shape'] != parsed_logs['text_encoding']['negative_shape']:
                analysis['issues'].append("Prompt and negative prompt embeddings have different shapes")
   
    # Check VAE encoding
    if parsed_logs['vae_encoding']:
        analysis['metrics']['vae_encoding'] = {
            'input_shape': parsed_logs['vae_encoding'].get('input_shape'),
            'latent_shape': parsed_logs['vae_encoding'].get('latent_shape'),
            'conditioning_frames': parsed_logs['vae_encoding'].get('conditioning_frames'),
            'cond_indicator': parsed_logs['vae_encoding'].get('cond_indicator')
        }
       
        # Check conditioning frames
        if 'conditioning_frames' in parsed_logs['vae_encoding']:
            conditioning_frames = parsed_logs['vae_encoding']['conditioning_frames']
            if conditioning_frames == 1:
                analysis['issues'].append(f"Only {conditioning_frames} frame is being conditioned - this might be insufficient for Video2World")
                analysis['recommendations'].append("Consider conditioning more frames for better temporal coherence")
       
        # Check conditioning indicator
        if 'cond_indicator' in parsed_logs['vae_encoding']:
            cond_indicator = parsed_logs['vae_encoding']['cond_indicator']
            if cond_indicator and cond_indicator[0] == 1.0 and sum(cond_indicator[1:]) == 0:
                analysis['issues'].append("Only the first frame is being conditioned (indicator: [1, 0, 0, 0, 0, 0])")
   
    # Check denoising
    if parsed_logs['denoising']:
        analysis['metrics']['denoising'] = {
            'latents_shape': parsed_logs['denoising'].get('latents_shape'),
            'noise_pred_shape': parsed_logs['denoising'].get('noise_pred_shape'),
            'guidance_scale': parsed_logs['denoising'].get('guidance_scale'),
            'per_frame_means': parsed_logs['denoising'].get('per_frame_means')
        }
       
        # Check guidance scale
        if 'guidance_scale' in parsed_logs['denoising']:
            try:
                guidance_scale = float(parsed_logs['denoising']['guidance_scale'])
                if guidance_scale == 1.0:
                    analysis['issues'].append("Guidance scale is 1.0 - CFG is disabled")
                    analysis['recommendations'].append("Enable CFG with guidance_scale > 1.0 for better results")
                elif guidance_scale > 1.0:
                    analysis['recommendations'].append(f"CFG is enabled with guidance_scale={guidance_scale} - this is good")
            except (ValueError, TypeError):
                # Skip guidance scale analysis if we can't parse it
                pass
   
    # Check conditioning
    if parsed_logs['conditioning']:
        analysis['metrics']['conditioning'] = parsed_logs['conditioning']
        
        # Check conditioning indicator
        if 'indicator' in parsed_logs['conditioning']:
            indicator = parsed_logs['conditioning']['indicator']
            if indicator and indicator.startswith('[1, 0, 0, 0, 0, 0]'):
                analysis['issues'].append("Only first frame is being conditioned - this explains the noise in other frames!")
                analysis['recommendations'].append("Check conditioning logic - should condition more frames for Video2World")
        
        # Check latent means
        if 'latents_mean' in parsed_logs['conditioning'] and 'noise_mean' in parsed_logs['conditioning']:
            try:
                latents_mean = float(parsed_logs['conditioning']['latents_mean'])
                noise_mean = float(parsed_logs['conditioning']['noise_mean'])
                if abs(latents_mean - noise_mean) < 0.1:
                    analysis['issues'].append("Conditioning latents and noise latents have very similar means - conditioning may not be working")
            except ValueError:
                pass
    
    # Check latent statistics
    if parsed_logs['latent_stats']:
        analysis['metrics']['latent_stats'] = parsed_logs['latent_stats']
        
        # Check per-frame means
        if 'per_frame_means' in parsed_logs['latent_stats']:
            per_frame_means = parsed_logs['latent_stats']['per_frame_means']
            if per_frame_means:
                try:
                    # Parse the list of means
                    means_str = per_frame_means.strip('[]')
                    means = [float(x.strip()) for x in means_str.split(',')]
                    if len(means) > 1:
                        first_frame_mean = means[0]
                        other_frames_mean = sum(means[1:]) / len(means[1:])
                        if abs(first_frame_mean - other_frames_mean) < 0.1:
                            analysis['issues'].append("First frame and other frames have very similar means - conditioning may not be working")
                        elif abs(other_frames_mean) > 0.5:
                            analysis['issues'].append("Other frames have high mean values - they may be noise")
                except (ValueError, IndexError):
                    pass
   
    # Check diagnostic logs
    if parsed_logs['diagnostics']:
        analysis['metrics']['diagnostics'] = parsed_logs['diagnostics']
        
        # Check attention statistics
        if 'attention_stats' in parsed_logs['diagnostics']:
            attention_stats = parsed_logs['diagnostics']['attention_stats']
            if 'N/A' in attention_stats:
                analysis['issues'].append("Attention weights are not available - temporal attention may be broken")
            elif 'mean:' in attention_stats:
                try:
                    mean_val = float(attention_stats.split('mean:')[1].split(',')[0].strip())
                    if abs(mean_val) < 0.01:
                        analysis['issues'].append("Attention weights have very low mean - temporal attention may be weak")
                except (ValueError, IndexError):
                    pass
        
        # Check hidden state progression
        if 'pre_attn_stats' in parsed_logs['diagnostics'] and 'post_attn_stats' in parsed_logs['diagnostics']:
            try:
                pre_mean = float(parsed_logs['diagnostics']['pre_attn_stats'].split('mean:')[1].split(',')[0].strip())
                post_mean = float(parsed_logs['diagnostics']['post_attn_stats'].split('mean:')[1].split(',')[0].strip())
                if abs(pre_mean - post_mean) < 0.01:
                    analysis['issues'].append("Hidden states before and after attention are very similar - attention may not be working")
            except (ValueError, IndexError):
                pass
        
        # Check temporal consistency
        if 'temporal_diffs' in parsed_logs['diagnostics']:
            temporal_diffs = parsed_logs['diagnostics']['temporal_diffs']
            if temporal_diffs and 'N/A' not in temporal_diffs:
                try:
                    diffs = [float(x.strip()) for x in temporal_diffs.strip('[]').split(',')]
                    avg_diff = sum(diffs) / len(diffs)
                    if avg_diff < 0.01:
                        analysis['issues'].append("Very low temporal differences between frames - temporal progression may be weak")
                except (ValueError, IndexError):
                    pass
   
    # Check timing
    if parsed_logs['timing']:
        analysis['metrics']['timing'] = parsed_logs['timing']
   
    # Check warnings and errors
    if parsed_logs['errors']:
        analysis['errors'] = parsed_logs['errors']
   
    return analysis


def compare_implementations(diffusers_logs: str, fastvideo_logs: str):
    """Compare the two implementations and report differences."""
    print("=" * 80)
    print("COMPARING IMPLEMENTATIONS")
    print("=" * 80)
   
    # Parse logs
    diffusers_parsed = parse_logs(diffusers_logs, "diffusers")
    fastvideo_parsed = parse_logs(fastvideo_logs, "fastvideo")
   
    # Compare key metrics
    comparisons = []
   
    # Text encoding comparison
    if diffusers_parsed['text_encoding'] and fastvideo_parsed['text_encoding']:
        print("📊 Text Encoding Comparison:")
        for key in ['prompt_shape', 'negative_shape', 'prompt_range', 'negative_range', 
                   'prompt_mean', 'negative_mean', 'prompt_std', 'negative_std',
                   'prompt_sum', 'negative_sum', 'prompt_abs_sum', 'negative_abs_sum',
                   'prompt_abs_max', 'negative_abs_max', 'prompt_norm', 'negative_norm']:
            if key in diffusers_parsed['text_encoding'] and key in fastvideo_parsed['text_encoding']:
                diff = diffusers_parsed['text_encoding'][key]
                fv = fastvideo_parsed['text_encoding'][key]
                match = diff == fv
                print(f"  {key}: {'✅' if match else '❌'}")
                print(f"    Diffusers: {diff}")
                print(f"    FastVideo: {fv}")
                comparisons.append(('text_encoding', key, match))
   
    # VAE encoding comparison
    if diffusers_parsed['vae_encoding'] and fastvideo_parsed['vae_encoding']:
        print("📊 VAE Encoding Comparison:")
        for key in ['latent_shape', 'conditioning_frames', 'encoded_stats', 'input_mean', 'input_std']:
            if key in diffusers_parsed['vae_encoding'] and key in fastvideo_parsed['vae_encoding']:
                diff = diffusers_parsed['vae_encoding'][key]
                fv = fastvideo_parsed['vae_encoding'][key]
                match = diff == fv
                print(f"  {key}: {'✅' if match else '❌'}")
                print(f"    Diffusers: {diff}")
                print(f"    FastVideo: {fv}")
                comparisons.append(('vae_encoding', key, match))
   
    # Denoising comparison
    if diffusers_parsed['denoising'] and fastvideo_parsed['denoising']:
        print("📊 Denoising Comparison:")
        for key in ['latents_shape', 'noise_pred_shape', 'guidance_scale', 'noise_pred_stats', 'coefficients',
                   'noise_pred_range', 'noise_pred_sum', 'noise_pred_abs_sum', 'noise_pred_abs_max', 'noise_pred_norm',
                   'latents_range', 'latents_mean', 'latents_std', 'latents_sum', 'latents_abs_sum', 'latents_abs_max', 'latents_norm']:
            if key in diffusers_parsed['denoising'] and key in fastvideo_parsed['denoising']:
                diff = diffusers_parsed['denoising'][key]
                fv = fastvideo_parsed['denoising'][key]
                match = diff == fv
                print(f"  {key}: {'✅' if match else '❌'}")
                print(f"    Diffusers: {diff}")
                print(f"    FastVideo: {fv}")
                comparisons.append(('denoising', key, match))
   
    # Conditioning comparison
    if diffusers_parsed['conditioning'] and fastvideo_parsed['conditioning']:
        print("📊 Conditioning Comparison:")
        for key in ['indicator', 'latents_mean', 'noise_mean', 'cond_latent_mean', 'first_frame_cond_mean', 
                   'other_frames_cond_mean', 'conditioning_difference', 'first_frame_final_mean', 
                   'other_frames_final_mean', 'final_conditioning_difference', 'step_cond_latent_mean',
                   'step_raw_transformer_mean', 'step_after_coefficients_mean', 'step_final_noise_pred_mean',
                   'step_noise_pred_final_mean', 'step_final_latents_mean', 'step_latents_mean',
                   'step_noise_pred_mean', 'step_current_sigma', 'step_cond_latent_range', 'step_cond_latent_std',
                   'step_cond_latent_abs_max', 'step_raw_transformer_range', 'step_raw_transformer_std',
                   'step_after_coefficients_range', 'step_final_noise_pred_range', 'step_final_noise_pred_std']:
            if key in diffusers_parsed['conditioning'] and key in fastvideo_parsed['conditioning']:
                diff = diffusers_parsed['conditioning'][key]
                fv = fastvideo_parsed['conditioning'][key]
                match = diff == fv
                print(f"  {key}: {'✅' if match else '❌'}")
                print(f"    Diffusers: {diff}")
                print(f"    FastVideo: {fv}")
                comparisons.append(('conditioning', key, match))
   
    # Latent statistics comparison
    if diffusers_parsed['latent_stats'] and fastvideo_parsed['latent_stats']:
        print("📊 Latent Statistics Comparison:")
        for key in ['per_frame_means', 'coefficients', 'per_frame_means_alt', 'first_frame_mean', 
                   'other_frames_mean', 'conditioning_difference', 'samples_mean', 'samples_std']:
            if key in diffusers_parsed['latent_stats'] and key in fastvideo_parsed['latent_stats']:
                diff = diffusers_parsed['latent_stats'][key]
                fv = fastvideo_parsed['latent_stats'][key]
                match = diff == fv
                print(f"  {key}: {'✅' if match else '❌'}")
                print(f"    Diffusers: {diff}")
                print(f"    FastVideo: {fv}")
                comparisons.append(('latent_stats', key, match))
   
    # Decoding comparison
    if diffusers_parsed['decoding'] and fastvideo_parsed['decoding']:
        print("📊 Decoding Comparison:")
        for key in ['input_latents_shape', 'input_latents_stats', 'input_latents_range', 'normalized_stats',
                   'normalized_range', 'vae_output_shape', 'vae_output_stats', 'vae_output_range',
                   'final_video_shape', 'final_video_stats', 'final_video_range']:
            if key in diffusers_parsed['decoding'] and key in fastvideo_parsed['decoding']:
                diff = diffusers_parsed['decoding'][key]
                fv = fastvideo_parsed['decoding'][key]
                match = diff == fv
                print(f"  {key}: {'✅' if match else '❌'}")
                print(f"    Diffusers: {diff}")
                print(f"    FastVideo: {fv}")
                comparisons.append(('decoding', key, match))
   
    # Diagnostic logs comparison
    if diffusers_parsed['diagnostics'] and fastvideo_parsed['diagnostics']:
        print("📊 Diagnostic Logs Comparison:")
        for key in ['patch_shape', 'rope_stats', 'attention_stats', 'pre_attn_stats', 'post_attn_stats', 'post_ff_stats', 'cond_mask_stats', 'temporal_diffs', 'cfg_stats']:
            if key in diffusers_parsed['diagnostics'] and key in fastvideo_parsed['diagnostics']:
                diff = diffusers_parsed['diagnostics'][key]
                fv = fastvideo_parsed['diagnostics'][key]
                match = diff == fv
                print(f"  {key}: {'✅' if match else '❌'}")
                print(f"    Diffusers: {diff}")
                print(f"    FastVideo: {fv}")
                comparisons.append(('diagnostics', key, match))
            elif key in diffusers_parsed['diagnostics'] or key in fastvideo_parsed['diagnostics']:
                print(f"  {key}: {'⚠️' if key in diffusers_parsed['diagnostics'] else '⚠️'}")
                print(f"    Diffusers: {diffusers_parsed['diagnostics'].get(key, 'N/A')}")
                print(f"    FastVideo: {fastvideo_parsed['diagnostics'].get(key, 'N/A')}")
                comparisons.append(('diagnostics', key, False))
   
    # Timing comparison
    if diffusers_parsed['timing'] and fastvideo_parsed['timing']:
        print("📊 Timing Comparison:")
        if 'generation_time' in diffusers_parsed['timing'] and 'generation_time' in fastvideo_parsed['timing']:
            diff_time = float(diffusers_parsed['timing']['generation_time'])
            fv_time = float(fastvideo_parsed['timing']['generation_time'])
            ratio = fv_time / diff_time if diff_time > 0 else float('inf')
            print(f"  Generation time:")
            print(f"    Diffusers: {diff_time:.2f}s")
            print(f"    FastVideo: {fv_time:.2f}s")
            print(f"    Ratio (FV/Diff): {ratio:.2f}x")
   
    # Error comparison
    print("📊 Error Comparison:")
    print(f"  Diffusers errors: {len(diffusers_parsed['errors'])}")
    print(f"  FastVideo errors: {len(fastvideo_parsed['errors'])}")
   
    if diffusers_parsed['errors']:
        print("  Diffusers errors:")
        for error in diffusers_parsed['errors']:
            print(f"    {error}")
   
    if fastvideo_parsed['errors']:
        print("  FastVideo errors:")
        for error in fastvideo_parsed['errors']:
            print(f"    {error}")
   
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
   
    total_comparisons = len(comparisons)
    matching_comparisons = sum(1 for _, _, match in comparisons if match)
   
    print(f"Total comparisons: {total_comparisons}")
    print(f"Matching comparisons: {matching_comparisons}")
    if total_comparisons > 0:
        print(f"Success rate: {matching_comparisons/total_comparisons*100:.1f}%")
    else:
        print("No pipeline stage comparisons found - check if test scripts log detailed pipeline information")
   
    if matching_comparisons == total_comparisons:
        print("✅ All comparisons match! Implementations appear to be equivalent.")
    else:
        print("⚠️  Some comparisons don't match. Check the detailed logs above.")
   
    return {
        'diffusers_parsed': diffusers_parsed,
        'fastvideo_parsed': fastvideo_parsed,
        'comparisons': comparisons,
        'success_rate': matching_comparisons/total_comparisons if total_comparisons > 0 else 0
    }


def print_analysis(analysis: Dict[str, Any]):
    """Print the analysis results in a readable format."""
    print("=" * 80)
    print("COSMOS2 VIDEO2WORLD LOG ANALYSIS")
    print("=" * 80)
   
    # Print metrics
    print("\n📊 METRICS:")
    for category, metrics in analysis['metrics'].items():
        print(f"\n{category.upper()}:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
   
    # Print issues
    if analysis['issues']:
        print(f"\n⚠️  ISSUES FOUND ({len(analysis['issues'])}):")
        for i, issue in enumerate(analysis['issues'], 1):
            print(f"  {i}. {issue}")
    else:
        print("\n✅ No issues found!")
   
    # Print recommendations
    if analysis['recommendations']:
        print(f"\n💡 RECOMMENDATIONS ({len(analysis['recommendations'])}):")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"  {i}. {rec}")
   
    # Print warnings
    if analysis.get('warnings'):
        print(f"\n⚠️  WARNINGS ({len(analysis['warnings'])}):")
        for warning in analysis['warnings']:
            print(f"  {warning}")
   
    # Print errors
    if analysis.get('errors'):
        print(f"\n❌ ERRORS ({len(analysis['errors'])}):")
        for error in analysis['errors']:
            print(f"  {error}")
   
    print("\n" + "=" * 80)


def main():
    """Main function to run both implementations and compare logs."""
    print("🚀 Starting Cosmos2 Video2World implementation comparison test")
   
    # Create output directories
    os.makedirs("outputs/diffusers", exist_ok=True)
    os.makedirs("outputs/fastvideo", exist_ok=True)
   
    # Check if test image exists
    if not os.path.exists("tennis.jpg"):
        print("❌ Test image tennis.jpg not found. Please provide a test image.")
        return
   
    # Run diffusers test
    diffusers_logs = run_diffusers_test()
    if not diffusers_logs:
        print("❌ Diffusers test failed, cannot continue comparison")
        return
   
    # Run FastVideo test
    fastvideo_logs = run_fastvideo_test()
    if not fastvideo_logs:
        print("❌ FastVideo test failed, cannot continue comparison")
        return
   
    # Compare implementations
    comparison_results = compare_implementations(diffusers_logs, fastvideo_logs)
   
    # Analyze each implementation separately
    print("\n" + "=" * 80)
    print("ANALYZING DIFFUSERS IMPLEMENTATION")
    print("=" * 80)
    diffusers_analysis = analyze_logs(comparison_results['diffusers_parsed'])
    print_analysis(diffusers_analysis)
   
    print("\n" + "=" * 80)
    print("ANALYZING FASTVIDEO IMPLEMENTATION")
    print("=" * 80)
    fastvideo_analysis = analyze_logs(comparison_results['fastvideo_parsed'])
    print_analysis(fastvideo_analysis)
   
    # Save detailed results
    with open("comparison_results.json", "w") as f:
        json.dump({
            'diffusers_parsed': comparison_results['diffusers_parsed'],
            'fastvideo_parsed': comparison_results['fastvideo_parsed'],
            'comparisons': comparison_results['comparisons'],
            'success_rate': comparison_results['success_rate'],
            'diffusers_analysis': diffusers_analysis,
            'fastvideo_analysis': fastvideo_analysis
        }, f, indent=2, default=str)
   
    print("✅ Comparison test completed. Results saved to comparison_results.json")
   
    # No cleanup needed since both test files are now standalone
    print("✅ Both test files (test_diffusers.py and test_fastvideo.py) are preserved as standalone files")


if __name__ == "__main__":
    main()

