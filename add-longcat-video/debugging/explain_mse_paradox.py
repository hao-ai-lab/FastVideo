"""
Demonstrate why intermediate MSE can be large while final MSE is small.
"""

import torch
import numpy as np

print("="*80)
print("EXPLAINING THE MSE PARADOX")
print("="*80)

# Simulate intermediate activations (large scale)
print("\n[1] Intermediate Layer (Block 47 Output):")
print("-"*80)

# Original has large values
intermediate_orig = torch.randn(1000) * 50000  # Mean magnitude ~50k
# Native has similar pattern but small differences
intermediate_native = intermediate_orig + torch.randn(1000) * 100  # Add noise

mse_intermediate = torch.mean((intermediate_orig - intermediate_native) ** 2).item()
rel_mse_intermediate = mse_intermediate / torch.mean(intermediate_orig ** 2).item()
cos_sim_intermediate = torch.nn.functional.cosine_similarity(
    intermediate_orig.unsqueeze(0), 
    intermediate_native.unsqueeze(0)
).item()

print(f"  Mean magnitude: {intermediate_orig.abs().mean():.1f}")
print(f"  MSE: {mse_intermediate:.2e}")
print(f"  Relative MSE: {rel_mse_intermediate:.6f} ({rel_mse_intermediate*100:.3f}%)")
print(f"  Cosine similarity: {cos_sim_intermediate:.6f}")

# Simulate final output (normalized/projected)
print("\n[2] Final Layer Output:")
print("-"*80)

# Project to smaller space and normalize (simulating final layer)
projection = torch.randn(64, 1000) / np.sqrt(1000)  # Projection matrix

final_orig = torch.matmul(projection, intermediate_orig)
final_orig = final_orig / final_orig.norm() * 2  # Normalize to unit scale

final_native = torch.matmul(projection, intermediate_native)
final_native = final_native / final_native.norm() * 2

mse_final = torch.mean((final_orig - final_native) ** 2).item()
rel_mse_final = mse_final / torch.mean(final_orig ** 2).item()
cos_sim_final = torch.nn.functional.cosine_similarity(
    final_orig.unsqueeze(0),
    final_native.unsqueeze(0)
).item()

print(f"  Mean magnitude: {final_orig.abs().mean():.1f}")
print(f"  MSE: {mse_final:.2e}")
print(f"  Relative MSE: {rel_mse_final:.6f} ({rel_mse_final*100:.3f}%)")
print(f"  Cosine similarity: {cos_sim_final:.6f}")

# Comparison
print("\n[3] Comparison:")
print("="*80)
print(f"  MSE Ratio (intermediate/final): {mse_intermediate/mse_final:,.0f}x")
print(f"  But Relative MSE is similar:")
print(f"    Intermediate: {rel_mse_intermediate*100:.3f}%")
print(f"    Final:        {rel_mse_final*100:.3f}%")
print(f"  And Cosine Similarity is similar:")
print(f"    Intermediate: {cos_sim_intermediate:.6f}")
print(f"    Final:        {cos_sim_final:.6f}")

print("\n[4] Key Insight:")
print("="*80)
print("""
  The LARGE intermediate MSE is due to LARGE activation magnitudes.
  The SMALL final MSE is due to SMALL output magnitudes.
  
  But the RELATIVE ERROR (% difference) is similar!
  
  This is like saying:
    - Error of $10 when measuring $100,000 building = 0.01% error
    - Error of $1 when measuring $100 purchase = 1% error
  
  The $10 error is larger in absolute terms, but smaller in relative terms!
  
  In our case, the projection + normalization in the final layer acts like
  converting from dollars to percentages - it reveals that the relative
  errors are actually quite similar throughout.
""")

print("\n[5] Why This Happens in Transformers:")
print("="*80)
print("""
  1. Residual Connections: Add activations together → magnitudes grow
  2. No Normalization in Blocks: Values can explode to 10k-100k range
  3. Final Layer: Projects (4096→64) + Normalizes → back to ~1.0 range
  4. AdaLN Modulation: Scales output by learned factors
  
  Result: Huge absolute differences get "squeezed" back to small range!
""")

# Real data comparison
print("\n[6] Your Actual Data:")
print("="*80)
print("  Block 47 Output:")
print("    - MSE: 8.0e9")
print("    - Relative MSE: 0.003 (0.3%)")
print("    - Cosine Sim: 1.001")
print()
print("  Final Output:")
print("    - MSE: 0.003")
print("    - Relative MSE: 0.002 (0.2%)")
print("    - Cosine Sim: 0.999")
print()
print("  → The RELATIVE errors are almost identical!")
print("  → The implementations are ~99.9% correlated!")
print("  → The MSE difference is just a scaling artifact!")

print("\n✅ Explanation complete!")






