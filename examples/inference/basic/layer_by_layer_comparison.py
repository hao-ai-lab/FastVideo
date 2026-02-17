"""
Layer-by-layer comparison between SGLang and FastVideo pipelines.

Captures intermediate outputs from each pipeline stage and model layer to identify divergence points.
"""

import numpy as np
import torch
from collections import OrderedDict
from typing import Dict, Any, List
import json


class LayerOutputCapture:
    """Captures output from each layer during forward pass."""
    
    def __init__(self, name: str):
        self.name = name
        self.captures = OrderedDict()
        self.hooks = []
        self.call_counter = {}
    
    def register_hooks_recursive(self, model: torch.nn.Module, prefix: str = ""):
        """Register forward hooks on all submodules."""
        for child_name, child_module in model.named_children():
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            
            # Register hook on this module
            hook = child_module.register_forward_hook(
                self._create_hook(full_name)
            )
            self.hooks.append(hook)
            
            # Recurse to children
            if len(list(child_module.children())) > 0:
                self.register_hooks_recursive(child_module, full_name)
    
    def _create_hook(self, layer_name: str):
        """Create a hook function for a specific layer."""
        def hook_fn(module, input, output):
            # Count calls to this layer
            if layer_name not in self.call_counter:
                self.call_counter[layer_name] = 0
            self.call_counter[layer_name] += 1
            
            call_id = f"{layer_name}#call{self.call_counter[layer_name]}"
            
            # Extract tensor statistics
            stats = self._extract_stats(output)
            self.captures[call_id] = stats
            
            # Print real-time for first few calls of important layers
            if self.call_counter[layer_name] <= 2 and any(x in layer_name.lower() for x in ['encoder', 'decoder', 'attention', 'block', 'layer']):
                print(f"  [{self.name}] {call_id}: {stats.get('summary', 'N/A')}")
        
        return hook_fn
    
    def _extract_stats(self, output: Any) -> Dict:
        """Extract statistics from layer output."""
        if isinstance(output, torch.Tensor):
            t = output.detach().cpu()
            return {
                'type': 'tensor',
                'shape': list(t.shape),
                'dtype': str(t.dtype),
                'min': float(t.min()) if t.numel() > 0 else None,
                'max': float(t.max()) if t.numel() > 0 else None,
                'mean': float(t.mean()) if t.numel() > 0 else None,
                'std': float(t.std()) if t.numel() > 0 else None,
                'summary': f"Tensor{list(t.shape)} [{float(t.min()):.3f}, {float(t.max()):.3f}] mean={float(t.mean()):.3f}"
            }
        elif isinstance(output, (tuple, list)):
            items = []
            for i, item in enumerate(output[:2]):  # First 2 items
                if isinstance(item, torch.Tensor):
                    items.append(self._extract_stats(item))
            return {
                'type': 'sequence',
                'length': len(output),
                'items': items,
                'summary': f"Sequence[{len(output)}]"
            }
        else:
            return {
                'type': type(output).__name__,
                'summary': f"Type: {type(output).__name__}"
            }
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def save_json(self, filepath: str):
        """Save captures to JSON."""
        with open(filepath, 'w') as f:
            json.dump(dict(self.captures), f, indent=2)


def run_fastvideo_with_capture(prompt: str, seed: int = 42, width: int = 720, height: int = 1280, 
                                steps: int = 50) -> tuple:
    """Run FastVideo with layer capture."""
    from fastvideo import VideoGenerator
    
    print("\n" + "="*100)
    print("FASTVIDEO GENERATION WITH LAYER CAPTURE")
    print("="*100)
    print(f"Prompt: {prompt[:60]}...")
    print(f"Seed: {seed}, Resolution: {width}x{height}, Steps: {steps}")
    
    capture = LayerOutputCapture("FastVideo")
    
    # Initialize generator
    generator = VideoGenerator.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        num_gpus=1,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )
    
    # Hook into pipeline models
    if hasattr(generator.executor, 'pipeline'):
        pipeline = generator.executor.pipeline
        if hasattr(pipeline, 'models'):
            print(f"\nHooking into {len(pipeline.models)} models:")
            for model_name, model in pipeline.models.items():
                if isinstance(model, torch.nn.Module):
                    print(f"  - {model_name}")
                    capture.register_hooks_recursive(model, f"FV_{model_name}")
    
    print(f"\nRegistered {len(capture.hooks)} hooks")
    print("\nGenerating...")
    
    # Generate with matching parameters
    result = generator.generate_video(
        prompt,
        seed=seed,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=1.0,  # SGLang uses guidance_scale=1.0
        save_video=False,
        return_frames=True,
    )
    
    frames = result if isinstance(result, list) else result.get("frames", [])
    
    print(f"\n✓ Generated! Captured {len(capture.captures)} layer outputs")
    
    # Print output
    if frames:
        frame = frames[0]
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        else:
            frame = np.array(frame)
        
        print(f"\nOutput frame: shape={frame.shape}, dtype={frame.dtype}")
        print(f"  min={frame.min()}, max={frame.max()}, mean={frame.mean():.2f}")
        print(f"  Top-left corner:\n{frame[:4, :4, :3]}")
    
    return capture, frames


def run_sglang_with_capture(prompt: str, seed: int = 42) -> tuple:
    """Run SGLang with layer capture."""
    try:
        from sglang.multimodal_gen import DiffGenerator
    except ImportError:
        print("\n✗ SGLang not available")
        return None, None
    
    print("\n" + "="*100)
    print("SGLANG GENERATION WITH LAYER CAPTURE")
    print("="*100)
    print(f"Prompt: {prompt[:60]}...")
    print(f"Seed: {seed}")
    
    capture = LayerOutputCapture("SGLang")
    
    # Initialize generator
    generator = DiffGenerator.from_pretrained(
        model_path="black-forest-labs/FLUX.1-dev",
        num_gpus=1,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )
    
    # Hook into pipeline models
    if hasattr(generator, 'pipeline'):
        pipeline = generator.pipeline
        if hasattr(pipeline, 'models'):
            print(f"\nHooking into {len(pipeline.models)} models:")
            for model_name, model in pipeline.models.items():
                if isinstance(model, torch.nn.Module):
                    print(f"  - {model_name}")
                    capture.register_hooks_recursive(model, f"SG_{model_name}")
    
    print(f"\nRegistered {len(capture.hooks)} hooks")
    print("\nGenerating...")
    
    # Generate with matching parameters
    result = generator.generate(
        sampling_params_kwargs=dict(
            prompt=prompt,
            seed=seed,
            infer_steps=50,
            guidance_scale=1.0,
            embedded_guidance_scale=3.5,
            return_frames=True,
            save_output=False,
        )
    )
    
    frames = result if isinstance(result, list) else result.get("frames", [])
    
    print(f"\n✓ Generated! Captured {len(capture.captures)} layer outputs")
    
    # Print output
    if frames:
        frame = frames[0]
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        else:
            frame = np.array(frame)
        
        print(f"\nOutput frame: shape={frame.shape}, dtype={frame.dtype}")
        print(f"  min={frame.min()}, max={frame.max()}, mean={frame.mean():.2f}")
        print(f"  Top-left corner:\n{frame[:4, :4, :3]}")
    
    return capture, frames


def compare_layer_outputs(fv_capture: LayerOutputCapture, sg_capture: LayerOutputCapture):
    """Compare captured layer outputs to find divergence."""
    print("\n" + "="*100)
    print("LAYER-BY-LAYER COMPARISON")
    print("="*100)
    
    if not fv_capture or not sg_capture:
        print("Missing capture data")
        return
    
    fv_layers = list(fv_capture.captures.keys())
    sg_layers = list(sg_capture.captures.keys())
    
    print(f"\nFastVideo layers: {len(fv_layers)}")
    print(f"SGLang layers: {len(sg_layers)}")
    
    # Compare layer by layer
    matching = 0
    diverging = 0
    first_divergence = None
    
    # Match by index (both should have similar pipeline structure)
    max_layers = min(len(fv_layers), len(sg_layers))
    
    for i in range(max_layers):
        fv_layer = fv_layers[i]
        sg_layer = sg_layers[i]
        
        fv_data = fv_capture.captures[fv_layer]
        sg_data = sg_capture.captures[sg_layer]
        
        if fv_data['type'] == 'tensor' and sg_data['type'] == 'tensor':
            fv_mean = fv_data.get('mean')
            sg_mean = sg_data.get('mean')
            
            if fv_mean is not None and sg_mean is not None:
                diff = abs(fv_mean - sg_mean)
                rel_diff = diff / (abs(sg_mean) + 1e-8)
                
                if rel_diff < 0.01:  # Within 1%
                    matching += 1
                    status = "✓"
                else:
                    diverging += 1
                    status = "✗"
                    if first_divergence is None:
                        first_divergence = (i, fv_layer, sg_layer, diff, rel_diff)
                
                # Print first 10 and any diverging layers
                if i < 10 or status == "✗":
                    print(f"\n{status} Layer {i}:")
                    print(f"  FV: {fv_layer}")
                    print(f"      {fv_data.get('summary', 'N/A')}")
                    print(f"  SG: {sg_layer}")
                    print(f"      {sg_data.get('summary', 'N/A')}")
                    if status == "✗":
                        print(f"  → Mean diff: {diff:.6f} ({rel_diff*100:.2f}%)")
    
    print("\n" + "="*100)
    print(f"SUMMARY: {matching} matching, {diverging} diverging")
    
    if first_divergence:
        i, fv_layer, sg_layer, diff, rel_diff = first_divergence
        print(f"\n🔴 FIRST DIVERGENCE at layer {i}:")
        print(f"  FastVideo: {fv_layer}")
        print(f"  SGLang:    {sg_layer}")
        print(f"  Difference: {diff:.6f} ({rel_diff*100:.2f}%)")
    
    print("="*100)


def main():
    """Main comparison."""
    print("\n" + "="*100)
    print("LAYER-BY-LAYER PIPELINE COMPARISON: FASTVIDEO vs SGLANG")
    print("="*100)
    
    prompt = "A cinematic portrait of a fox, 35mm film, soft light, gentle grain."
    seed = 42
    
    # Run FastVideo
    try:
        fv_capture, fv_frames = run_fastvideo_with_capture(
            prompt, seed=seed, width=720, height=1280, steps=50
        )
        fv_capture.save_json("/FastVideo/examples/inference/basic/fastvideo_layers.json")
        print("Saved FastVideo layers to fastvideo_layers.json")
    except Exception as e:
        print(f"\n✗ FastVideo error: {e}")
        import traceback
        traceback.print_exc()
        fv_capture, fv_frames = None, None
    
    # Run SGLang
    try:
        sg_capture, sg_frames = run_sglang_with_capture(prompt, seed=seed)
        if sg_capture:
            sg_capture.save_json("/FastVideo/examples/inference/basic/sglang_layers.json")
            print("Saved SGLang layers to sglang_layers.json")
    except Exception as e:
        print(f"\n✗ SGLang error: {e}")
        import traceback
        traceback.print_exc()
        sg_capture, sg_frames = None, None
    
    # Compare
    if fv_capture and sg_capture:
        compare_layer_outputs(fv_capture, sg_capture)
    else:
        print("\n⚠ Cannot compare - missing capture data")
    
    print("\n" + "="*100)
    print("DONE")
    print("="*100)


if __name__ == "__main__":
    main()
