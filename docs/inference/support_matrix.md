# Compatibility Matrix

The table below shows every supported model and optimizations supported for them.

The symbols used have the following meanings:

- ✅ = Full compatibility
- ❌ = No compatibility
- ⭕ = Does not apply to this model

## Models x Optimization

The `HuggingFace Model ID` can be directly pass to `from_pretrained()` methods and FastVideo will use the optimal default parameters when initializing and generating videos.

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

| Model Name | HuggingFace Model ID | Resolutions | TeaCache | Sliding Tile Attn | Sage Attn | VSA |
|------------|---------------------|-------------|----------|-------------------|-----------|-----|
| FastWan2.1 T2V 1.3B | `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` | 480P | ⭕ | ⭕ | ⭕ | ✅ |
| FastWan2.2 TI2V 5B Full Attn* | `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers` | 720P | ⭕ | ⭕ | ⭕ | ✅ |
| Wan2.2 TI2V 5B | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | 720P | ⭕ | ⭕ | ✅ | ⭕ |
| Wan2.2 T2V A14B | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | 480P<br>720P | ❌ | ❌ | ✅ | ⭕ |
| Wan2.2 I2V A14B | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | 480P<br>720P | ❌ | ❌ | ✅ | ⭕ |
| HunyuanVideo | `hunyuanvideo-community/HunyuanVideo` | 720px1280p<br>544px960p | ❌ | ✅ | ✅ | ⭕ |
| FastHunyuan | `FastVideo/FastHunyuan-diffusers` | 720px1280p<br>544px960p | ❌ | ✅ | ✅ | ⭕ |
| Wan2.1 T2V 1.3B | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | 480P | ✅ | ✅* | ✅ | ⭕ |
| Wan2.1 T2V 14B | `Wan-AI/Wan2.1-T2V-14B-Diffusers` | 480P, 720P | ✅ | ✅* | ✅ | ⭕ |
| Wan2.1 I2V 480P | `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` | 480P | ✅ | ✅* | ✅ | ⭕ |
| Wan2.1 I2V 720P | `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` | 720P | ✅ | ✅ | ✅ | ⭕ |
| StepVideo T2V | `FastVideo/stepvideo-t2v-diffusers` | 768px768px204f<br>544px992px204f<br>544px992px136f | ❌ | ❌ | ✅ | ⭕ |

**Note**: Wan2.2 TI2V 5B has some quality issues when performing I2V generation. We are working on fixing this issue.

## Special requirements

### StepVideo T2V
- The self-attention in text-encoder (step_llm) only supports CUDA capabilities sm_80 sm_86 and sm_90

### Sliding Tile Attention
- Currently only Hopper GPUs (H100s) are supported.