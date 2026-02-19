"""
Debug script to compare layer activations between two generations using native FastVideo API.
"""
import torch
import numpy as np
from pathlib import Path
from fastvideo import VideoGenerator

OUTPUT_PATH = "debug_outputs"
Path(OUTPUT_PATH).mkdir(exist_ok=True)


class DebugHook:
    """Captures activations from specific layers."""
    
    def __init__(self):
        self.activations = {}
        self.hooks = []
    
    def create_hook(self, name):
        """Create a forward hook for a specific layer."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            # Store on CPU to avoid memory issues
            self.activations[name] = {
                'shape': tuple(output.shape),
                'dtype': str(output.dtype),
                'min': float(output.min().cpu()),
                'max': float(output.max().cpu()),
                'mean': float(output.mean().cpu()),
                'std': float(output.std().cpu()),
                # Store a small sample for comparison
                'sample': output.flatten()[:1000].detach().cpu().clone()
            }
        return hook_fn
    
    def register_hooks(self, generator):
        """Register hooks on key transformer layers."""
        # Access the transformer model
        if hasattr(generator, 'runner'):
            # Multi-GPU setup
            transformer = generator.runner.worker.transformer
        else:
            # Single GPU setup
            transformer = generator.transformer
        
        layers_to_monitor = {}
        
        # Embedders
        if hasattr(transformer, 'x_embedder'):
            layers_to_monitor['x_embedder'] = transformer.x_embedder
        if hasattr(transformer, 'context_embedder'):
            layers_to_monitor['context_embedder'] = transformer.context_embedder
        if hasattr(transformer, 'time_text_embed'):
            layers_to_monitor['time_text_embed'] = transformer.time_text_embed
        
        # First few transformer blocks
        if hasattr(transformer, 'transformer_blocks'):
            for i in [0, 1, 2]:
                if i < len(transformer.transformer_blocks):
                    layers_to_monitor[f'transformer_block_{i}'] = transformer.transformer_blocks[i]
        
        # Last few transformer blocks
        if hasattr(transformer, 'transformer_blocks'):
            num_blocks = len(transformer.transformer_blocks)
            for i in range(max(0, num_blocks - 3), num_blocks):
                layers_to_monitor[f'transformer_block_{i}'] = transformer.transformer_blocks[i]
        
        # First few single blocks
        if hasattr(transformer, 'single_transformer_blocks'):
            for i in [0, 1, 2]:
                if i < len(transformer.single_transformer_blocks):
                    layers_to_monitor[f'single_block_{i}'] = transformer.single_transformer_blocks[i]
        
        # Last few single blocks
        if hasattr(transformer, 'single_transformer_blocks'):
            num_single = len(transformer.single_transformer_blocks)
            for i in range(max(0, num_single - 3), num_single):
                layers_to_monitor[f'single_block_{i}'] = transformer.single_transformer_blocks[i]
        
        # Output layers
        if hasattr(transformer, 'norm_out'):
            layers_to_monitor['norm_out'] = transformer.norm_out
        if hasattr(transformer, 'proj_out'):
            layers_to_monitor['proj_out'] = transformer.proj_out
        
        # Register hooks
        for name, layer in layers_to_monitor.items():
            handle = layer.register_forward_hook(self.create_hook(name))
            self.hooks.append(handle)
        
        print(f"Registered {len(self.hooks)} hooks on layers: {list(layers_to_monitor.keys())}")
        return layers_to_monitor.keys()
    
    def clear(self):
        """Clear stored activations."""
        self.activations = {}
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []


def save_stats(activations, filename):
    """Save activation statistics to a text file."""
    with open(filename, 'w') as f:
        f.write("Layer Activation Statistics\n")
        f.write("=" * 80 + "\n\n")
        
        for name, stats in sorted(activations.items()):
            f.write(f"{name}:\n")
            f.write(f"  Shape: {stats['shape']}\n")
            f.write(f"  Dtype: {stats['dtype']}\n")
            f.write(f"  Min: {stats['min']:.6f}\n")
            f.write(f"  Max: {stats['max']:.6f}\n")
            f.write(f"  Mean: {stats['mean']:.6f}\n")
            f.write(f"  Std: {stats['std']:.6f}\n")
            f.write("\n")


def compare_activations(acts1, acts2, label1="prompt1", label2="prompt2"):
    """Compare activations between two generations."""
    print(f"\nComparing activations: {label1} vs {label2}")
    print("=" * 80)
    
    all_layers = set(acts1.keys()) | set(acts2.keys())
    
    differences = {}
    for layer_name in sorted(all_layers):
        if layer_name not in acts1:
            print(f"{layer_name}: Only in {label2}")
            continue
        if layer_name not in acts2:
            print(f"{layer_name}: Only in {label1}")
            continue
        
        stats1 = acts1[layer_name]
        stats2 = acts2[layer_name]
        
        # Compare samples
        sample1 = stats1['sample']
        sample2 = stats2['sample']
        
        if sample1.shape != sample2.shape:
            print(f"{layer_name}: Shape mismatch {sample1.shape} vs {sample2.shape}")
            continue
        
        # Calculate differences
        abs_diff = torch.abs(sample1 - sample2)
        max_diff = float(abs_diff.max())
        mean_diff = float(abs_diff.mean())
        rel_diff = float((abs_diff / (torch.abs(sample1) + 1e-8)).mean())
        
        differences[layer_name] = {
            'max_abs_diff': max_diff,
            'mean_abs_diff': mean_diff,
            'mean_rel_diff': rel_diff,
            'shape1': stats1['shape'],
            'shape2': stats2['shape'],
        }
        
        print(f"{layer_name}:")
        print(f"  Max abs diff: {max_diff:.6e}")
        print(f"  Mean abs diff: {mean_diff:.6e}")
        print(f"  Mean rel diff: {rel_diff:.6%}")
        print(f"  Shape: {stats1['shape']}")
    
    return differences


def _print_frame_matrix(frames, label: str) -> None:
    """Print frame statistics."""
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


def main():
    print("Initializing VideoGenerator...")
    generator = VideoGenerator.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )
    
    # Set up debugging hooks
    debug_hook = DebugHook()
    monitored_layers = debug_hook.register_hooks(generator)
    
    # First generation
    print("\n" + "=" * 80)
    print("GENERATION 1: Fox portrait")
    print("=" * 80)
    prompt1 = "A cinematic portrait of a fox, 35mm film, soft light, gentle grain."
    
    debug_hook.clear()
    video1 = generator.generate_video(
        prompt1,
        output_path=OUTPUT_PATH,
        save_video=False,
        return_frames=True,
    )
    
    frames1 = video1.get("frames", []) if isinstance(video1, dict) else video1
    _print_frame_matrix(frames1, "prompt1")
    
    # Save first generation stats
    acts1 = debug_hook.activations.copy()
    save_stats(acts1, f"{OUTPUT_PATH}/prompt1_stats.txt")
    print(f"\nSaved prompt1 stats to {OUTPUT_PATH}/prompt1_stats.txt")
    
    # Second generation
    print("\n" + "=" * 80)
    print("GENERATION 2: Lion savanna")
    print("=" * 80)
    prompt2 = (
        "A majestic lion strides across the golden savanna, its powerful frame "
        "glistening under the warm afternoon sun. The tall grass ripples gently in "
        "the breeze, enhancing the lion's commanding presence. The tone is vibrant, "
        "embodying the raw energy of the wild. Low angle, steady tracking shot, "
        "cinematic."
    )
    
    debug_hook.clear()
    video2 = generator.generate_video(
        prompt2,
        output_path=OUTPUT_PATH,
        save_video=False,
        return_frames=True,
    )
    
    frames2 = video2.get("frames", []) if isinstance(video2, dict) else video2
    _print_frame_matrix(frames2, "prompt2")
    
    # Save second generation stats
    acts2 = debug_hook.activations.copy()
    save_stats(acts2, f"{OUTPUT_PATH}/prompt2_stats.txt")
    print(f"\nSaved prompt2 stats to {OUTPUT_PATH}/prompt2_stats.txt")
    
    # Compare activations
    differences = compare_activations(acts1, acts2)
    
    # Save comparison
    with open(f"{OUTPUT_PATH}/comparison.txt", 'w') as f:
        f.write("Activation Comparison: prompt1 vs prompt2\n")
        f.write("=" * 80 + "\n\n")
        
        for layer_name, diff_stats in sorted(differences.items()):
            f.write(f"{layer_name}:\n")
            f.write(f"  Max abs diff: {diff_stats['max_abs_diff']:.6e}\n")
            f.write(f"  Mean abs diff: {diff_stats['mean_abs_diff']:.6e}\n")
            f.write(f"  Mean rel diff: {diff_stats['mean_rel_diff']:.6%}\n")
            f.write(f"  Shape: {diff_stats['shape1']}\n")
            f.write("\n")
    
    print(f"\nSaved comparison to {OUTPUT_PATH}/comparison.txt")
    
    # Clean up
    debug_hook.remove_hooks()
    print("\nDebug analysis complete!")


if __name__ == "__main__":
    main()
