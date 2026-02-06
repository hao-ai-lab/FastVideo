# FastVideo Development Guidelines

## Model Porting Checklist

When porting models from SGLang or other frameworks:

### 🔴 CRITICAL: Numerical Alignment
Before claiming a port is complete, verify **each component is numerically aligned** against the reference implementation:

1. **Side-by-side component testing**: Run each stage (prompt encoding, latent prep, timesteps, denoising) with identical inputs and compare intermediate outputs tensor-by-tensor.

2. **Key areas to check**:
   - Prompt encoding method (full prompt vs extracted text)
   - Random number generation (use same RNG method: `randn_tensor` vs `torch.randn`)
   - Data types (float32 vs bfloat16)
   - Scheduler/timestep calculation (sigmas, mu, shift values)
   - Generation parameters (sampling params, model defaults)

3. **Create alignment test scripts**: Write scripts that load both implementations and compare outputs at each stage.

4. **SSIM is not enough**: Similar SSIM scores don't guarantee identical behavior. Components can diverge while still producing aesthetically acceptable outputs.

### Testing
- Run SSIM tests after confirming numerical alignment
- Use identical seeds, prompts, and parameters for comparison
- Log intermediate tensors for debugging

### Documentation
- Document any intentional deviations from reference implementation
- Note dependency version requirements (e.g., `transformers==5.0.0rc3`)
