# LongCat LoRA Structure Findings

## Key Discoveries

### Naming Convention
- Uses `___lorahyphen___` as the separator (replaces `.` in module names)
- Format: `lora___lorahyphen___module___lorahyphen___submodule.{lora_down|lora_up}.weight`
- Also includes `.alpha_scale` for each LoRA module

### File Comparison

| Aspect | cfg_step_lora | refinement_lora |
|--------|--------------|------------------|
| **Total keys** | 1,152 | 1,543 |
| **LoRA rank** | 128 | 128 |
| **LoRA alpha** | 64 (from alpha_scale) | 64 |
| **Layers** | Attention + FFN | Attention + FFN + AdaLN + FinalLayer |
| **Use case** | 16-step distilled generation | 480p→720p refinement |

### LoRA Coverage

#### cfg_step_lora (48 blocks × modules)
1. **Self-Attention**:
   - `attn.qkv`: Fused Q/K/V with n_separate=3 (384 → 4096×3)
   - `attn.proj`: Output projection (4096 → 4096)

2. **Cross-Attention**:
   - `cross_attn.q_linear`: Query (4096 → 4096)
   - `cross_attn.kv_linear`: Fused K/V with n_separate=2 (256 → 4096×2)

3. **FFN**:
   - `ffn.w1`: Gate projection (4096 → 11008)
   - `ffn.w2`: Down projection (11008 → 4096)
   - `ffn.w3`: Up projection (4096 → 11008)

#### refinement_lora (Additional layers)
4. **AdaLN Modulation** (48 blocks):
   - `adaLN_modulation.1`: n_separate=6 for (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)

5. **Final Layer** (1 block):
   - `final_layer.adaLN_modulation.1`: n_separate=2 for (shift, scale)
   - `final_layer.linear`: Output projection

### Detailed Patterns

#### Self-Attention QKV (n_separate=3)
```
lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_down.weight
Shape: [384, 4096]  # 3 × rank → in_dim

lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_up.blocks.0.weight
lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_up.blocks.1.weight
lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_up.blocks.2.weight
Each shape: [4096, 128]  # out_dim/3 → rank
```

#### Cross-Attention KV (n_separate=2)
```
lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___cross_attn___lorahyphen___kv_linear.lora_down.weight
Shape: [256, 4096]  # 2 × rank → in_dim

lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___cross_attn___lorahyphen___kv_linear.lora_up.blocks.0.weight
lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___cross_attn___lorahyphen___kv_linear.lora_up.blocks.1.weight
Each shape: [4096, 128]  # out_dim/2 → rank
```

#### AdaLN Modulation (n_separate=6, refinement only)
```
lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___adaLN_modulation___lorahyphen___1.lora_down.weight
Shape: [768, 512]  # 6 × rank → in_dim

lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___adaLN_modulation___lorahyphen___1.lora_up.blocks.0.weight
lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___adaLN_modulation___lorahyphen___1.lora_up.blocks.1.weight
...
lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___adaLN_modulation___lorahyphen___1.lora_up.blocks.5.weight
Each shape: [4096, 128]  # (6*4096)/6 → rank
```

### Mapping Strategy

FastVideo's LoRA system expects:
- `.lora_A` weights: [rank, in_dim]
- `.lora_B` weights: [out_dim, rank]

LongCat provides:
- `.lora_down.weight`: [rank × n_separate, in_dim] or [rank, in_dim]
- `.lora_up.weight` or `.lora_up.blocks.X.weight`: [out_dim/n_separate, rank]

**Handling n_separate > 1**:
1. For `.lora_down`: Already correct shape if we treat it as single matrix
2. For `.lora_up.blocks.*`: Need to concatenate blocks along out_dim

Example for QKV (n_separate=3):
```python
# Load
lora_up_0 = lora_dict["...qkv.lora_up.blocks.0.weight"]  # [4096, 128]
lora_up_1 = lora_dict["...qkv.lora_up.blocks.1.weight"]  # [4096, 128]
lora_up_2 = lora_dict["...qkv.lora_up.blocks.2.weight"]  # [4096, 128]

# Concatenate to [12288, 128] = [4096*3, 128]
lora_B = torch.cat([lora_up_0, lora_up_1, lora_up_2], dim=0)

# Transpose to FastVideo format: [out_dim, rank] → [rank, in_dim] for lora_A
lora_A = lora_dict["...qkv.lora_down.weight"]  # [384, 4096] - already correct!
```

### Implementation Notes

1. **FastVideo LoRA expects**:
   - Layer name format: `blocks.0.self_attn.to_q.lora_A`
   - Not the LongCat format: `lora___lorahyphen___blocks___lorahyphen___0...`

2. **Name mapping required**:
   ```
   lora___lorahyphen___blocks___lorahyphen___X___lorahyphen___attn___lorahyphen___qkv
   →
   blocks.X.self_attn.to_q (and to_k, to_v)
   ```

3. **Special handling needed for**:
   - Fused QKV → separate Q/K/V LoRAs
   - Fused KV → separate K/V LoRAs  
   - n_separate blocks → concatenated tensors

4. **Alpha scale**:
   - LongCat stores as scalar tensor
   - FastVideo uses `lora_alpha` parameter
   - Read from alpha_scale and pass to LoRA initialization

## Regex Patterns for Mapping

See updated `LORA_IMPLEMENTATION_PLAN.md` for the complete regex patterns.

## Verification Steps

1. Load LoRA weights
2. Check all 48 blocks have LoRA layers
3. Verify shapes match FastVideo expectations
4. Test with small inference to ensure no crashes
5. Compare output quality with reference implementation






## Key Discoveries

### Naming Convention
- Uses `___lorahyphen___` as the separator (replaces `.` in module names)
- Format: `lora___lorahyphen___module___lorahyphen___submodule.{lora_down|lora_up}.weight`
- Also includes `.alpha_scale` for each LoRA module

### File Comparison

| Aspect | cfg_step_lora | refinement_lora |
|--------|--------------|------------------|
| **Total keys** | 1,152 | 1,543 |
| **LoRA rank** | 128 | 128 |
| **LoRA alpha** | 64 (from alpha_scale) | 64 |
| **Layers** | Attention + FFN | Attention + FFN + AdaLN + FinalLayer |
| **Use case** | 16-step distilled generation | 480p→720p refinement |

### LoRA Coverage

#### cfg_step_lora (48 blocks × modules)
1. **Self-Attention**:
   - `attn.qkv`: Fused Q/K/V with n_separate=3 (384 → 4096×3)
   - `attn.proj`: Output projection (4096 → 4096)

2. **Cross-Attention**:
   - `cross_attn.q_linear`: Query (4096 → 4096)
   - `cross_attn.kv_linear`: Fused K/V with n_separate=2 (256 → 4096×2)

3. **FFN**:
   - `ffn.w1`: Gate projection (4096 → 11008)
   - `ffn.w2`: Down projection (11008 → 4096)
   - `ffn.w3`: Up projection (4096 → 11008)

#### refinement_lora (Additional layers)
4. **AdaLN Modulation** (48 blocks):
   - `adaLN_modulation.1`: n_separate=6 for (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)

5. **Final Layer** (1 block):
   - `final_layer.adaLN_modulation.1`: n_separate=2 for (shift, scale)
   - `final_layer.linear`: Output projection

### Detailed Patterns

#### Self-Attention QKV (n_separate=3)
```
lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_down.weight
Shape: [384, 4096]  # 3 × rank → in_dim

lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_up.blocks.0.weight
lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_up.blocks.1.weight
lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_up.blocks.2.weight
Each shape: [4096, 128]  # out_dim/3 → rank
```

#### Cross-Attention KV (n_separate=2)
```
lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___cross_attn___lorahyphen___kv_linear.lora_down.weight
Shape: [256, 4096]  # 2 × rank → in_dim

lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___cross_attn___lorahyphen___kv_linear.lora_up.blocks.0.weight
lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___cross_attn___lorahyphen___kv_linear.lora_up.blocks.1.weight
Each shape: [4096, 128]  # out_dim/2 → rank
```

#### AdaLN Modulation (n_separate=6, refinement only)
```
lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___adaLN_modulation___lorahyphen___1.lora_down.weight
Shape: [768, 512]  # 6 × rank → in_dim

lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___adaLN_modulation___lorahyphen___1.lora_up.blocks.0.weight
lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___adaLN_modulation___lorahyphen___1.lora_up.blocks.1.weight
...
lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___adaLN_modulation___lorahyphen___1.lora_up.blocks.5.weight
Each shape: [4096, 128]  # (6*4096)/6 → rank
```

### Mapping Strategy

FastVideo's LoRA system expects:
- `.lora_A` weights: [rank, in_dim]
- `.lora_B` weights: [out_dim, rank]

LongCat provides:
- `.lora_down.weight`: [rank × n_separate, in_dim] or [rank, in_dim]
- `.lora_up.weight` or `.lora_up.blocks.X.weight`: [out_dim/n_separate, rank]

**Handling n_separate > 1**:
1. For `.lora_down`: Already correct shape if we treat it as single matrix
2. For `.lora_up.blocks.*`: Need to concatenate blocks along out_dim

Example for QKV (n_separate=3):
```python
# Load
lora_up_0 = lora_dict["...qkv.lora_up.blocks.0.weight"]  # [4096, 128]
lora_up_1 = lora_dict["...qkv.lora_up.blocks.1.weight"]  # [4096, 128]
lora_up_2 = lora_dict["...qkv.lora_up.blocks.2.weight"]  # [4096, 128]

# Concatenate to [12288, 128] = [4096*3, 128]
lora_B = torch.cat([lora_up_0, lora_up_1, lora_up_2], dim=0)

# Transpose to FastVideo format: [out_dim, rank] → [rank, in_dim] for lora_A
lora_A = lora_dict["...qkv.lora_down.weight"]  # [384, 4096] - already correct!
```

### Implementation Notes

1. **FastVideo LoRA expects**:
   - Layer name format: `blocks.0.self_attn.to_q.lora_A`
   - Not the LongCat format: `lora___lorahyphen___blocks___lorahyphen___0...`

2. **Name mapping required**:
   ```
   lora___lorahyphen___blocks___lorahyphen___X___lorahyphen___attn___lorahyphen___qkv
   →
   blocks.X.self_attn.to_q (and to_k, to_v)
   ```

3. **Special handling needed for**:
   - Fused QKV → separate Q/K/V LoRAs
   - Fused KV → separate K/V LoRAs  
   - n_separate blocks → concatenated tensors

4. **Alpha scale**:
   - LongCat stores as scalar tensor
   - FastVideo uses `lora_alpha` parameter
   - Read from alpha_scale and pass to LoRA initialization

## Regex Patterns for Mapping

See updated `LORA_IMPLEMENTATION_PLAN.md` for the complete regex patterns.

## Verification Steps

1. Load LoRA weights
2. Check all 48 blocks have LoRA layers
3. Verify shapes match FastVideo expectations
4. Test with small inference to ensure no crashes
5. Compare output quality with reference implementation








