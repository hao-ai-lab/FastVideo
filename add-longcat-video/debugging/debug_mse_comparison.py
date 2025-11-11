"""
Calculate MSE and other metrics between saved layer outputs from original and native implementations.
"""

import torch
from pathlib import Path

def calculate_metrics(name, orig, native):
    """Calculate comprehensive metrics between two tensors."""
    # Handle shape mismatches
    if orig.shape != native.shape:
        # Try to handle batch dimension differences
        if len(orig.shape) == len(native.shape) + 1 and orig.shape[0] == 1:
            orig = orig.squeeze(0)
        elif len(native.shape) == len(orig.shape) + 1 and native.shape[0] == 1:
            native = native.squeeze(0)
        
        if orig.shape != native.shape:
            print(f"  ❌ {name}: SHAPE MISMATCH! Orig={orig.shape}, Native={native.shape}")
            return None
    
    orig_f = orig.float()
    native_f = native.float()
    
    # Calculate various metrics
    diff = orig_f - native_f
    abs_diff = torch.abs(diff)
    rel_diff = abs_diff / (orig_f.abs() + 1e-8)
    
    metrics = {
        'mse': torch.mean(diff ** 2).item(),
        'rmse': torch.sqrt(torch.mean(diff ** 2)).item(),
        'mae': abs_diff.mean().item(),  # Mean Absolute Error
        'max_abs_diff': abs_diff.max().item(),
        'mean_abs_diff': abs_diff.mean().item(),
        'max_rel_diff': rel_diff.max().item(),
        'mean_rel_diff': rel_diff.mean().item(),
    }
    
    # Calculate cosine similarity
    orig_flat = orig_f.reshape(-1)
    native_flat = native_f.reshape(-1)
    cos_sim = torch.nn.functional.cosine_similarity(orig_flat.unsqueeze(0), native_flat.unsqueeze(0)).item()
    metrics['cosine_sim'] = cos_sim
    
    # Calculate correlation
    if orig_flat.numel() > 1:
        correlation = torch.corrcoef(torch.stack([orig_flat, native_flat]))[0, 1].item()
        metrics['correlation'] = correlation
    else:
        metrics['correlation'] = 1.0
    
    # Calculate relative MSE
    orig_power = torch.mean(orig_f ** 2).item()
    if orig_power > 0:
        metrics['relative_mse'] = metrics['mse'] / orig_power
    else:
        metrics['relative_mse'] = float('inf')
    
    # Classify divergence level
    if metrics['mse'] < 1e-8:
        status = "✅ IDENTICAL"
    elif metrics['mse'] < 1e-4:
        status = "✅ VERY CLOSE"
    elif metrics['mse'] < 1e-2:
        status = "⚠️  MODERATE"
    elif metrics['mse'] < 1.0:
        status = "❌ DIVERGED"
    else:
        status = "❌ LARGE DIVERGENCE"
    
    metrics['status'] = status
    
    return metrics

def main():
    print("="*100)
    print("MSE & METRICS COMPARISON - LONGCAT ORIGINAL vs NATIVE")
    print("="*100)
    
    output_dir = Path("outputs/debug_layers")
    
    print("\n[1] Loading saved activations...")
    orig_acts = torch.load(output_dir / "orig_activations.pt")
    native_acts = torch.load(output_dir / "native_activations.pt")
    
    print(f"  ✓ Original: {len(orig_acts)} activations")
    print(f"  ✓ Native: {len(native_acts)} activations")
    
    print("\n[2] Computing metrics for each layer...")
    print("="*100)
    
    all_metrics = {}
    
    # Components to compare (in forward order)
    components = [
        ("patch_embed", "Patch Embedding"),
        ("time_embed", "Time Embedding"),
        ("caption_embed", "Caption Embedding"),
    ]
    
    # Add block components
    for block_idx in [0, 11, 23, 35, 47]:
        components.extend([
            (f"block_{block_idx}_self_attn", f"Block {block_idx} Self-Attention"),
            (f"block_{block_idx}_cross_attn", f"Block {block_idx} Cross-Attention"),
            (f"block_{block_idx}_ffn", f"Block {block_idx} FFN"),
            (f"block_{block_idx}", f"Block {block_idx} Output"),
        ])
    
    components.extend([
        ("final_layer", "Final Layer"),
        ("output", "Final Output"),
    ])
    
    # Calculate metrics for each component
    for key, name in components:
        if key not in orig_acts or key not in native_acts:
            print(f"\n⚠️  {name}: Missing in saved activations")
            continue
        
        metrics = calculate_metrics(name, orig_acts[key], native_acts[key])
        if metrics is not None:
            all_metrics[name] = metrics
            
            print(f"\n{metrics['status']} {name}")
            print(f"  MSE:          {metrics['mse']:.6e}")
            print(f"  RMSE:         {metrics['rmse']:.6e}")
            print(f"  MAE:          {metrics['mae']:.6e}")
            print(f"  Relative MSE: {metrics['relative_mse']:.6e}")
            print(f"  Max Abs Diff: {metrics['max_abs_diff']:.6e}")
            print(f"  Mean Abs Diff:{metrics['mean_abs_diff']:.6e}")
            print(f"  Cosine Sim:   {metrics['cosine_sim']:.6f}")
            print(f"  Correlation:  {metrics['correlation']:.6f}")
    
    # Summary table
    print("\n" + "="*100)
    print("SUMMARY TABLE (sorted by MSE)")
    print("="*100)
    
    sorted_metrics = sorted(all_metrics.items(), key=lambda x: x[1]['mse'], reverse=True)
    
    print(f"\n{'Component':<45} {'MSE':>12} {'RMSE':>12} {'MAE':>12} {'Cos Sim':>10} {'Status':<20}")
    print("-"*115)
    
    for name, metrics in sorted_metrics:
        print(f"{name:<45} {metrics['mse']:>12.4e} {metrics['rmse']:>12.4e} {metrics['mae']:>12.4e} {metrics['cosine_sim']:>10.6f} {metrics['status']:<20}")
    
    # Find first significant divergence
    print("\n" + "="*100)
    print("FIRST SIGNIFICANT DIVERGENCE")
    print("="*100)
    
    mse_threshold = 1e-4
    for key, name in components:
        if name in all_metrics:
            metrics = all_metrics[name]
            if metrics['mse'] > mse_threshold:
                print(f"\n❌ First component with MSE > {mse_threshold}: {name}")
                print(f"   MSE:           {metrics['mse']:.6e}")
                print(f"   RMSE:          {metrics['rmse']:.6e}")
                print(f"   Relative MSE:  {metrics['relative_mse']:.6e}")
                print(f"   Max Abs Diff:  {metrics['max_abs_diff']:.6e}")
                print(f"   Cosine Sim:    {metrics['cosine_sim']:.6f}")
                break
    else:
        print(f"\n✅ All components have MSE < {mse_threshold}")
    
    # Statistics
    print("\n" + "="*100)
    print("OVERALL STATISTICS")
    print("="*100)
    
    mses = [m['mse'] for m in all_metrics.values()]
    cos_sims = [m['cosine_sim'] for m in all_metrics.values()]
    
    print(f"\nMSE Statistics:")
    print(f"  Min:     {min(mses):.6e}")
    print(f"  Max:     {max(mses):.6e}")
    print(f"  Mean:    {sum(mses)/len(mses):.6e}")
    print(f"  Median:  {sorted(mses)[len(mses)//2]:.6e}")
    
    print(f"\nCosine Similarity Statistics:")
    print(f"  Min:     {min(cos_sims):.6f}")
    print(f"  Max:     {max(cos_sims):.6f}")
    print(f"  Mean:    {sum(cos_sims)/len(cos_sims):.6f}")
    
    # Count by status
    status_counts = {}
    for metrics in all_metrics.values():
        status = metrics['status']
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print(f"\nDivergence Classification:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    
    print("\n✅ Metrics calculation complete!")
    
    return all_metrics

if __name__ == "__main__":
    metrics = main()






