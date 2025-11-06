(support-matrix)=
# Compatibility Matrix
The table below shows every supported model and optimizations supported for them.

The symbols used have the following meanings:

- ✅ = Full compatibility
- ❌ = No compatibility
- ⭕ = Does not apply to this model

## Models x Optimization
The `HuggingFace Model ID` can be directly pass to `from_pretrained()` methods and FastVideo will use the optimal default parameters when initializing and generating videos.

:::{raw} html
<style>
  /* Make smaller to try to improve readability  */
  td {
    font-size: 0.9rem;
    text-align: center;
  }

  th {
    text-align: center;
    font-size: 0.9rem;
  }
</style>
:::

:::{list-table}
:header-rows: 1
:stub-columns: 3
:widths: auto
:class: vertical-table-header

- * Model Name
  * HuggingFace Model ID
  * Resolutions
  * TeaCache
  * Sliding Tile Attn
  * Sage Attn
  * Video Sparse Attention (VSA)
- * FastWan2.1 T2V 1.3B
  * `FastVideo/FastWan2.1-T2V-1.3B-Diffusers`
  * 480P
  * ⭕
  * ⭕
  * ⭕
  * ✅
- * FastWan2.2 TI2V 5B Full Attn*
  * `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers`
  * 720P
  * ⭕
  * ⭕
  * ⭕
  * ✅
- * Wan2.2 TI2V 5B 
  * `Wan-AI/Wan2.2-TI2V-5B-Diffusers`
  * 720P
  * ⭕
  * ⭕
  * ✅
  * ⭕
- * Wan2.2 T2V A14B
  * `Wan-AI/Wan2.2-T2V-A14B-Diffusers`
  * 480P<br>720P
  * ❌
  * ❌
  * ✅
  * ⭕
- * Wan2.2 I2V A14B
  * `Wan-AI/Wan2.2-I2V-A14B-Diffusers`
  * 480P<br>720P
  * ❌
  * ❌
  * ✅
  * ⭕
- * HunyuanVideo
  * `hunyuanvideo-community/HunyuanVideo`
  * 720px1280p<br>544px960p
  * ❌
  * ✅
  * ✅
  * ⭕
- * FastHunyuan
  * `FastVideo/FastHunyuan-diffusers`
  * 720px1280p<br>544px960p
  * ❌
  * ✅
  * ✅
  * ⭕
- * Wan2.1 T2V 1.3B
  * `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
  * 480P
  * ✅
  * ✅*
  * ✅
  * ⭕
- * Wan2.1 T2V 14B
  * `Wan-AI/Wan2.1-T2V-14B-Diffusers`
  * 480P, 720P
  * ✅
  * ✅*
  * ✅
  * ⭕
- * Wan2.1 I2V 480P
  * `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`
  * 480P
  * ✅
  * ✅*
  * ✅
  * ⭕
- * Wan2.1 I2V 720P
  * `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers`
  * 720P
  * ✅
  * ✅
  * ✅
  * ⭕
- * StepVideo T2V
  * `FastVideo/stepvideo-t2v-diffusers`
  * 768px768px204f<br>544px992px204f<br>544px992px136f
  * ❌
  * ❌
  * ✅
  * ⭕
:::

**Note**: Wan2.2 TI2V 5B has some quality issues when performing I2V generation. We are working on fixing this issue.

## Special requirements

### StepVideo T2V
- The self-attention in text-encoder (step_llm) only supports CUDA capabilities sm_80 sm_86 and sm_90

### Sliding Tile Attention
- Currently only Hopper GPUs (H100s) are supported.
