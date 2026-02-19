import numpy as np
import torch
from sglang.multimodal_gen import DiffGenerator
from collections import OrderedDict
import os

OUTPUT_PATH = "image_samples"
DEBUG_DIR = "debug_outputs"
os.makedirs(DEBUG_DIR, exist_ok=True)

def _print_frame_matrix(frames, label: str) -> None:
    if not frames:
        print(f"[{label}] No frames returned")
        return
    frame0 = frames[0]
    if isinstance(frame0, np.ndarray):
        arr = frame0
    else:
        arr = np.array(frame0)

    print(
        f"[{label}] frame0 shape={arr.shape} dtype={arr.dtype} "
        f"min={arr.min()} max={arr.max()} mean={arr.mean()}"
    )

    if arr.ndim >= 2:
        h = min(4, arr.shape[0])
        w = min(4, arr.shape[1])
        if arr.ndim == 3:
            c = min(3, arr.shape[2])
            print(f"[{label}] frame0 slice (H{h}xW{w}xC{c}):\n{arr[:h, :w, :c]}")
        else:
            print(f"[{label}] frame0 slice (H{h}xW{w}):\n{arr[:h, :w]}")


class DebugHook:
    """Capture intermediate activations during forward pass."""
    def __init__(self):
        self.activations = OrderedDict()
        self.latents_per_step = []
        self.hooks = []
        
    def register_hooks(self, transformer):
        """Register hooks on key transformer layers."""
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.activations[name] = output.detach().cpu()
                elif isinstance(output, tuple) and len(output) > 0:
                    if isinstance(output[0], torch.Tensor):
                        self.activations[name] = output[0].detach().cpu()
            return hook
        
        # Hook embedders
        self.hooks.append(transformer.x_embedder.register_forward_hook(hook_fn("x_embedder")))
        self.hooks.append(transformer.context_embedder.register_forward_hook(hook_fn("context_embedder")))
        self.hooks.append(transformer.time_text_embed.register_forward_hook(hook_fn("time_text_embed")))
        
        # Hook first and last transformer blocks
        self.hooks.append(transformer.transformer_blocks[0].register_forward_hook(hook_fn("transformer_block_00")))
        self.hooks.append(transformer.transformer_blocks[-1].register_forward_hook(hook_fn("transformer_block_last")))
        
        # Hook first and last single blocks
        self.hooks.append(transformer.single_transformer_blocks[0].register_forward_hook(hook_fn("single_block_00")))
        self.hooks.append(transformer.single_transformer_blocks[-1].register_forward_hook(hook_fn("single_block_last")))
        
        # Hook output layers
        self.hooks.append(transformer.norm_out.register_forward_hook(hook_fn("norm_out")))
        self.hooks.append(transformer.proj_out.register_forward_hook(hook_fn("proj_out")))
        
    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
    
    def save_stats(self, prefix: str):
        """Save activation statistics."""
        stats_file = os.path.join(DEBUG_DIR, f"{prefix}_stats.txt")
        with open(stats_file, "w") as f:
            f.write(f"=== Activation Statistics for {prefix} ===\n\n")
            for name, act in self.activations.items():
                if isinstance(act, torch.Tensor):
                    f.write(f"{name}:\n")
                    f.write(f"  Shape: {tuple(act.shape)}\n")
                    f.write(f"  Min: {act.min().item():.6e}\n")
                    f.write(f"  Max: {act.max().item():.6e}\n")
                    f.write(f"  Mean: {act.mean().item():.6e}\n")
                    f.write(f"  Std: {act.std().item():.6e}\n")
                    f.write("\n")
        print(f"Saved activation stats to {stats_file}")
        
    def save_latents(self, prefix: str):
        """Save all captured latents."""
        if self.latents_per_step:
            latents_file = os.path.join(DEBUG_DIR, f"{prefix}_latents.pt")
            torch.save({
                'latents': self.latents_per_step,
                'num_steps': len(self.latents_per_step)
            }, latents_file)
            print(f"Saved {len(self.latents_per_step)} latent steps to {latents_file}")


def monkey_patch_denoising(generator, debug_hook: DebugHook, prompt_id: str):
    """Patch the denoising stage to capture intermediate latents."""
    original_denoise = None
    
    # Find the denoising stage
    for stage in generator.pipeline.stages:
        if stage.__class__.__name__ == 'DenoisingStage':
            original_denoise = stage.run
            break
    
    if original_denoise is None:
        print("Warning: Could not find DenoisingStage to patch")
        return
    
    step_counter = [0]  # mutable counter
    
    def patched_denoise(batch, *args, **kwargs):
        result = original_denoise(batch, *args, **kwargs)
        
        # Capture latent at this step
        if hasattr(batch, 'latents') and batch.latents is not None:
            step_counter[0] += 1
            debug_hook.latents_per_step.append({
                'step': step_counter[0],
                'latent': batch.latents.detach().cpu().clone()
            })
            
            # Print stats at key steps
            if step_counter[0] in [1, 10, 25, 50] or step_counter[0] % 10 == 0:
                lat = batch.latents
                print(f"  Step {step_counter[0]:3d}: latent shape={lat.shape} "
                      f"min={lat.min():.4f} max={lat.max():.4f} mean={lat.mean():.4f}")
        
        return result
    
    stage.run = patched_denoise


def main():
    # Initialize DiffGenerator
    generator = DiffGenerator.from_pretrained(
        model_path="black-forest-labs/FLUX.1-dev",
        num_gpus=1,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )
    
    # Get transformer from pipeline
    transformer = None
    for stage in generator.pipeline.stages:
        if hasattr(stage, 'transformer'):
            transformer = stage.transformer
            break
    
    if transformer is None:
        print("Warning: Could not find transformer in pipeline")
        return
    
    # ===== First generation with debugging =====
    print("\n" + "="*80)
    print("GENERATION 1: Fox Portrait")
    print("="*80)
    
    debug_hook1 = DebugHook()
    debug_hook1.register_hooks(transformer)
    monkey_patch_denoising(generator, debug_hook1, "prompt1")
    
    prompt = "A cinematic portrait of a fox, 35mm film, soft light, gentle grain."
    result = generator.generate(
        sampling_params_kwargs=dict(
            prompt=prompt,
            return_frames=True,
            save_output=False,
            output_path=OUTPUT_PATH,
            seed=42,  # Fixed seed for reproducibility
        )
    )
    
    debug_hook1.remove_hooks()
    debug_hook1.save_stats("prompt1")
    debug_hook1.save_latents("prompt1")
    
    frames = result if isinstance(result, list) else result.get("frames", [])
    _print_frame_matrix(frames, "prompt1")
    
    # ===== Second generation with debugging =====
    print("\n" + "="*80)
    print("GENERATION 2: Lion in Savanna")
    print("="*80)
    
    debug_hook2 = DebugHook()
    debug_hook2.register_hooks(transformer)
    monkey_patch_denoising(generator, debug_hook2, "prompt2")
    
    prompt2 = (
        "A majestic lion strides across the golden savanna, its powerful frame "
        "glistening under the warm afternoon sun. The tall grass ripples gently in "
        "the breeze, enhancing the lion's commanding presence. The tone is vibrant, "
        "embodying the raw energy of the wild. Low angle, steady tracking shot, "
        "cinematic."
    )
    result2 = generator.generate(
        sampling_params_kwargs=dict(
            prompt=prompt2,
            return_frames=True,
            save_output=False,
            output_path=OUTPUT_PATH,
            seed=42,  # Same seed for comparison
        )
    )
    
    debug_hook2.remove_hooks()
    debug_hook2.save_stats("prompt2")
    debug_hook2.save_latents("prompt2")
    
    frames2 = result2 if isinstance(result2, list) else result2.get("frames", [])
    _print_frame_matrix(frames2, "prompt2")
    
    # ===== Compare activations =====
    print("\n" + "="*80)
    print("ACTIVATION COMPARISON")
    print("="*80)
    
    common_layers = set(debug_hook1.activations.keys()) & set(debug_hook2.activations.keys())
    
    for layer_name in sorted(common_layers):
        act1 = debug_hook1.activations[layer_name]
        act2 = debug_hook2.activations[layer_name]
        
        if act1.shape != act2.shape:
            print(f"{layer_name:30s} SHAPE MISMATCH: {act1.shape} vs {act2.shape}")
            continue
        
        diff = (act1.float() - act2.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"{layer_name:30s} max_diff={max_diff:.6e} mean_diff={mean_diff:.6e}")
    
    print(f"\nDebug outputs saved to: {DEBUG_DIR}/")


if __name__ == "__main__":
    main()
