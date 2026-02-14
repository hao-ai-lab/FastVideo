# Compatibility Matrix

This page summarizes common model + optimization combinations.

For the canonical, code-level list of model IDs recognized by
`VideoGenerator.from_pretrained(...)`, see the registrations in
`fastvideo/registry.py` (`register_configs(...)` entries).

!!! note
    The full STA integration in `fastvideo/` is archived from `main` and kept
    in `sta_do_not_delete`:
    https://github.com/hao-ai-lab/FastVideo/tree/sta_do_not_delete
    We do this because we believe VSA is strictly better than STA for the
    actively maintained `main` inference path.

The symbols used have the following meanings:

- ✅ = Full compatibility
- ❌ = No compatibility
- ⭕ = Does not apply to this model

## Models x Optimization

The `HuggingFace Model ID` can be passed directly to
`from_pretrained()`. FastVideo then uses model-specific default settings for
pipeline initialization and sampling.

<style>
  /* Target tables in this section */
  #models-x-optimization + p + table {
    display: block;
    overflow-x: auto;
    width: 100%;
    font-size: 0.85rem;
  }
  
  #models-x-optimization + p + table td,
  #models-x-optimization + p + table th {
    text-align: center;
    white-space: nowrap;
    padding: 0.5em;
  }
  
  /* First two columns can wrap */
  #models-x-optimization + p + table td:nth-child(1),
  #models-x-optimization + p + table td:nth-child(2) {
    white-space: normal;
    min-width: 120px;
  }
  
  #models-x-optimization + p + table td:nth-child(2) code {
    font-size: 0.75rem;
  }
</style>

| Model Name | HuggingFace Model ID | Resolutions | TeaCache | Sliding Tile Attn (Legacy Branch) | Sage Attn | VSA | BSA |
|------------|---------------------|-------------|----------|-------------------|-----------|-----|-----|
| FastWan2.1 T2V 1.3B | `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` | 480P | ⭕ | ⭕ | ⭕ | ✅ | ⭕ |
| FastWan2.2 TI2V 5B Full Attn* | `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers` | 720P | ⭕ | ⭕ | ⭕ | ✅ | ⭕ |
| Wan2.2 TI2V 5B | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | 720P | ⭕ | ⭕ | ✅ | ⭕ | ⭕ |
| Wan2.2 T2V A14B | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | 480P<br>720P | ❌ | ❌ | ✅ | ⭕ | ⭕ |
| Wan2.2 I2V A14B | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | 480P<br>720P | ❌ | ❌ | ✅ | ⭕ | ⭕ |
| HunyuanVideo | `hunyuanvideo-community/HunyuanVideo` | 720px1280p<br>544px960p | ❌ | ✅ | ✅ | ⭕ | ⭕ |
| FastHunyuan | `FastVideo/FastHunyuan-diffusers` | 720px1280p<br>544px960p | ❌ | ✅ | ✅ | ⭕ | ⭕ |
| Wan2.1 T2V 1.3B | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | 480P | ✅ | ✅* | ✅ | ⭕ | ⭕ |
| Wan2.1 T2V 14B | `Wan-AI/Wan2.1-T2V-14B-Diffusers` | 480P, 720P | ✅ | ✅* | ✅ | ⭕ | ⭕ |
| Wan2.1 I2V 480P | `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` | 480P | ✅ | ✅* | ✅ | ⭕ | ⭕ |
| Wan2.1 I2V 720P | `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` | 720P | ✅ | ✅ | ✅ | ⭕ | ⭕ |
| TurboWan2.1 T2V 1.3B | `loayrashid/TurboWan2.1-T2V-1.3B-Diffusers` | 480P | ⭕ | ⭕ | ⭕ | ⭕ | ⭕ |
| TurboWan2.1 T2V 14B | `loayrashid/TurboWan2.1-T2V-14B-Diffusers` | 480P, 720P | ⭕ | ⭕ | ⭕ | ⭕ | ⭕ |
| TurboWan2.2 I2V A14B | `loayrashid/TurboWan2.2-I2V-A14B-Diffusers` | 480P<br>720P | ⭕ | ⭕ | ⭕ | ⭕ | ⭕ |
| LongCat T2V 13.6B | See note** | 480P<br>720P | ❌ | ❌ | ❌ | ⭕ | ✅ |
| Matrix Game 2.0 Base | `FastVideo/Matrix-Game-2.0-Base-Diffusers` | 352x640 | ⭕ | ⭕ | ⭕ | ⭕ | ⭕ |
| Matrix Game 2.0 GTA | `FastVideo/Matrix-Game-2.0-GTA-Diffusers` | 352x640 | ⭕ | ⭕ | ⭕ | ⭕ | ⭕ |
| Matrix Game 2.0 TempleRun | `FastVideo/Matrix-Game-2.0-TempleRun-Diffusers` | 352x640 | ⭕ | ⭕ | ⭕ | ⭕ | ⭕ |
| GEN3C Cosmos 7B | `nvidia/GEN3C-Cosmos-7B` | 720px1280p | ❌ | ❌ | ❌ | ⭕ | ⭕ |

**Note**: Wan2.2 TI2V 5B has some quality issues when performing I2V generation. We are working on fixing this issue.

`Sliding Tile Attn (Legacy Branch)` entries refer to the archived
`sta_do_not_delete` branch workflow, not active `main` inference wiring.

## Canonical Supported IDs

The authoritative source for model-ID recognition is
`fastvideo/registry.py`. If a model ID is registered there, FastVideo can
resolve default pipeline and sampling configuration for it.

## Special requirements

### Sliding Tile Attention
- Full STA pipeline usage is on the archived branch:
  https://github.com/hao-ai-lab/FastVideo/tree/sta_do_not_delete
- STA currently requires Hopper GPUs (H100s).

### TurboWan2.1 (TurboDiffusion)
- Uses TurboDiffusionPipeline with RCM scheduler for 1-4 step generation
- Requires SLA attention backend: `export FASTVIDEO_ATTENTION_BACKEND=SLA_ATTN`
- Uses `guidance_scale=1.0` (no classifier-free guidance)

### Matrix Game 2.0
- Image-to-video game world models with keyboard/mouse control input
- Three variants available: Base (universal), GTA, and TempleRun
- Each variant has different keyboard dimensions for control inputs
