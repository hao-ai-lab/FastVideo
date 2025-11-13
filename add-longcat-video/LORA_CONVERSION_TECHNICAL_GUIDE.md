# LongCat LoRA to FastVideo Conversion - Technical Guide

This document explains the LoRA weight formats, FastVideo's expectations, and the conversion implementation.

## Table of Contents
1. [LoRA Background](#lora-background)
2. [LongCat LoRA Format](#longcat-lora-format)
3. [FastVideo LoRA Format](#fastvideo-lora-format)
4. [Conversion Implementation](#conversion-implementation)
5. [Usage Examples](#usage-examples)

---

## LoRA Background

### What is LoRA?

LoRA (Low-Rank Adaptation) adds trainable low-rank matrices to frozen model weights:

```
Original: y = Wx
LoRA:     y = Wx + BAx
```

Where:
- `W`: Frozen pretrained weights [out_dim, in_dim]
- `A`: LoRA down-projection [rank, in_dim] (called `lora_down` or `lora_A`)
- `B`: LoRA up-projection [out_dim, rank] (called `lora_up` or `lora_B`)
- `rank`: LoRA rank (typically 8-128)
- `alpha`: Scaling factor (typically rank/2)

**Final output**: `y = Wx + (alpha/rank) * BAx`

### Why LoRA?

- ✅ **Small file size**: 1-10MB vs 16GB for full model
- ✅ **Fast training**: Only train A and B matrices
- ✅ **Composable**: Can switch between different LoRAs
- ✅ **Preserves base model**: Base weights stay frozen

---

## LongCat LoRA Format

### File Structure

LongCat uses the **original LoRA implementation** format with some special features:

#### 1. Key Naming Convention

Keys use `___lorahyphen___` as separator (replaces `.`):

```
Original module path:      blocks.0.attn.qkv
LongCat LoRA key:          lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_down.weight
```

#### 2. Weight Components

Each LoRA module has 3 components:

```python
# Example: blocks.0.attn.qkv
{
    "lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_down.weight": 
        torch.Tensor([384, 4096]),  # rank*n_separate × in_dim
    
    "lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_up.blocks.0.weight": 
        torch.Tensor([4096, 128]),  # out_dim/n_separate × rank
    "lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_up.blocks.1.weight": 
        torch.Tensor([4096, 128]),
    "lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_up.blocks.2.weight": 
        torch.Tensor([4096, 128]),
    
    "lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.alpha_scale": 
        torch.Tensor([])  # Scalar: alpha/rank = 64/128 = 0.5
}
```

#### 3. n_separate Feature

**Key innovation**: LongCat uses `n_separate` for fused projections.

Instead of:
```python
# Standard LoRA
qkv_weight = base_qkv + lora_B @ lora_A
```

LongCat uses:
```python
# n_separate=3 for QKV
lora_A = [rank*3, in_dim]           # Fused down-projection
lora_B_q = [out_dim, rank]          # Separate up-projections
lora_B_k = [out_dim, rank]
lora_B_v = [out_dim, rank]

q = base_q + lora_B_q @ lora_A[0:rank]
k = base_k + lora_B_k @ lora_A[rank:2*rank]
v = base_v + lora_B_v @ lora_A[2*rank:3*rank]
```

**Why?** This allows independent adaptation of Q/K/V while sharing the input projection.

### Examples from LongCat LoRA Files

#### cfg_step_lora.safetensors (1,152 keys)

Used for **distilled 16-step generation**:

```
Self-Attention:
  - blocks.{0-47}.attn.qkv           (n_separate=3: Q, K, V)
  - blocks.{0-47}.attn.proj          (single output)

Cross-Attention:
  - blocks.{0-47}.cross_attn.q_linear       (single Q)
  - blocks.{0-47}.cross_attn.kv_linear      (n_separate=2: K, V)

FFN:
  - blocks.{0-47}.ffn.w1             (gate)
  - blocks.{0-47}.ffn.w2             (down)
  - blocks.{0-47}.ffn.w3             (up)
```

#### refinement_lora.safetensors (1,543 keys)

Used for **480p→720p refinement**:

Everything in cfg_step_lora **PLUS**:

```
AdaLN Modulation:
  - blocks.{0-47}.adaLN_modulation.1  (n_separate=6: shifts/scales/gates)

Final Layer:
  - final_layer.adaLN_modulation.1    (n_separate=2: shift, scale)
  - final_layer.linear                (output projection)
```

### LoRA Parameters

Both files use:
- **Rank**: 128
- **Alpha**: 64
- **Alpha scale**: 0.5 (= 64/128)

---

## FastVideo LoRA Format

### FastVideo's LoRA Architecture

FastVideo uses `BaseLayerWithLoRA` from SGLang:

```python
class BaseLayerWithLoRA(nn.Module):
    def __init__(self, base_layer, lora_rank, lora_alpha):
        self.base_layer = base_layer  # Original linear layer
        self.lora_A = nn.Parameter(...)  # [rank, in_dim]
        self.lora_B = nn.Parameter(...)  # [out_dim, rank]
    
    def forward(self, x):
        out, bias = self.base_layer(x)
        delta = x @ self.lora_A.T @ self.lora_B.T
        delta = delta * (self.lora_alpha / self.lora_rank)
        return out + delta, bias
```

### Key Expectations

#### 1. Key Naming

FastVideo expects **clean module paths** with `.lora_A` and `.lora_B`:

```
blocks.0.self_attn.to_q.lora_A      # [rank, in_dim]
blocks.0.self_attn.to_q.lora_B      # [out_dim, rank]
blocks.0.self_attn.to_k.lora_A
blocks.0.self_attn.to_k.lora_B
...
```

**No** `lora___lorahyphen___` separators, **no** `.lora_down`/`.lora_up` naming.

#### 2. Separate Projections

FastVideo's native LongCat uses **separate** Q/K/V projections (not fused):

```python
# Native implementation
class LongCatSelfAttention(nn.Module):
    def __init__(self, ...):
        self.to_q = ReplicatedLinear(dim, dim)  # Separate
        self.to_k = ReplicatedLinear(dim, dim)  # Separate
        self.to_v = ReplicatedLinear(dim, dim)  # Separate
```

So LoRA must provide **separate** `to_q`, `to_k`, `to_v` LoRAs.

#### 3. Weight Shapes

```python
# For a layer with in_dim=4096, out_dim=4096, rank=128:
lora_A: torch.Tensor([128, 4096])      # [rank, in_dim]
lora_B: torch.Tensor([4096, 128])      # [out_dim, rank]
```

#### 4. Loading Process

```python
# In LoRAPipeline.set_lora_adapter():
for name, layer in self.lora_layers.items():
    # name: "blocks.0.self_attn.to_q"
    lora_A = lora_dict[f"{name}.lora_A"]  # [rank, in_dim]
    lora_B = lora_dict[f"{name}.lora_B"]  # [out_dim, rank]
    layer.set_lora_weights(lora_A, lora_B)
```

### Module Naming

FastVideo native LongCat uses:

| Component | Original LongCat | FastVideo Native |
|-----------|-----------------|------------------|
| Self-attn QKV | `attn.qkv` | `self_attn.to_q`, `self_attn.to_k`, `self_attn.to_v` |
| Self-attn output | `attn.proj` | `self_attn.to_out` |
| Cross-attn Q | `cross_attn.q_linear` | `cross_attn.to_q` |
| Cross-attn KV | `cross_attn.kv_linear` | `cross_attn.to_k`, `cross_attn.to_v` |
| FFN | `ffn.w1/w2/w3` | `ffn.w1/w2/w3` (same) |
| AdaLN | `adaLN_modulation.1` | `adaln_linear_1` |

---

## Conversion Implementation

### Overview

The conversion in `longcat_to_fastvideo.py` handles:

1. ✅ **Key parsing**: Extract module path from `lora___lorahyphen___` format
2. ✅ **Module mapping**: Map LongCat paths to FastVideo paths
3. ✅ **n_separate handling**: Concatenate blocks and split fused projections
4. ✅ **Weight renaming**: Convert to `.lora_A` / `.lora_B` format

### Step-by-Step Process

#### Step 1: Parse LongCat Keys

```python
def parse_lora_key(key: str) -> tuple[str, str]:
    """
    Parse: "lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_down.weight"
    Return: ("blocks.0.attn.qkv", "lora_down.weight")
    """
    # Remove prefix
    if key.startswith("lora___lorahyphen___"):
        key = key[len("lora___lorahyphen___"):]
    
    # Replace separators: ___lorahyphen___ → .
    key = key.replace("___lorahyphen___", ".")
    
    # Extract module path and weight type
    if ".lora_down.weight" in key:
        return key.replace(".lora_down.weight", ""), "lora_down.weight"
    elif ".lora_up.blocks.0.weight" in key:
        # Handle n_separate blocks
        match = re.match(r"(.+)\.lora_up\.blocks\.(\d+)\.weight", key)
        return match.group(1), f"lora_up.blocks.{match.group(2)}.weight"
    # ... etc
```

#### Step 2: Map to FastVideo Paths

```python
def map_lora_module(module_path: str) -> list[tuple[str, str]]:
    """
    Map LongCat module path to FastVideo paths.
    Returns list because fused projections → multiple targets.
    """
    # QKV (fused) → separate Q, K, V
    if re.match(r"blocks\.(\d+)\.attn\.qkv", module_path):
        block_idx = match.group(1)
        return [
            (f"blocks.{block_idx}.self_attn.to_q", "q"),
            (f"blocks.{block_idx}.self_attn.to_k", "k"),
            (f"blocks.{block_idx}.self_attn.to_v", "v"),
        ]
    
    # KV (fused) → separate K, V
    if re.match(r"blocks\.(\d+)\.cross_attn\.kv_linear", module_path):
        block_idx = match.group(1)
        return [
            (f"blocks.{block_idx}.cross_attn.to_k", "k"),
            (f"blocks.{block_idx}.cross_attn.to_v", "v"),
        ]
    
    # Single output (no splitting)
    if re.match(r"blocks\.(\d+)\.attn\.proj", module_path):
        block_idx = match.group(1)
        return [(f"blocks.{block_idx}.self_attn.to_out", "single")]
    
    # ... etc
```

#### Step 3: Handle n_separate

```python
def convert_lora_weights(source_weights, lora_name):
    # Group weights by module
    modules = {}  # {module_path: {weight_type: key}}
    
    for key in source_weights.keys():
        module_path, weight_type = parse_lora_key(key)
        if module_path not in modules:
            modules[module_path] = {}
        modules[module_path][weight_type] = key
    
    converted = OrderedDict()
    
    for module_path, weight_keys in modules.items():
        targets = map_lora_module(module_path)  # May return multiple targets
        
        # === Handle lora_down (becomes lora_A) ===
        lora_down = source_weights[weight_keys["lora_down.weight"]]
        # Shape: [rank * n_separate, in_dim]
        
        if len(targets) == 1:
            # Single target: no splitting needed
            converted[f"{targets[0][0]}.lora_A"] = lora_down
        else:
            # Multiple targets: split by rank
            n = len(targets)
            rank = lora_down.shape[0] // n
            for i, (path, _) in enumerate(targets):
                # Extract rank slice for this target
                converted[f"{path}.lora_A"] = lora_down[i*rank:(i+1)*rank, :]
        
        # === Handle lora_up blocks (becomes lora_B) ===
        lora_up_blocks = []
        i = 0
        while f"lora_up.blocks.{i}.weight" in weight_keys:
            lora_up_blocks.append(source_weights[weight_keys[f"lora_up.blocks.{i}.weight"]])
            i += 1
        
        if lora_up_blocks:
            # Concatenate n_separate blocks along output dimension
            lora_up = torch.cat(lora_up_blocks, dim=0)
            # Shape: [out_dim_total, rank]
        else:
            # Single lora_up (no n_separate)
            lora_up = source_weights[weight_keys["lora_up.weight"]]
        
        if len(targets) == 1:
            # Single target
            converted[f"{targets[0][0]}.lora_B"] = lora_up
        else:
            # Multiple targets: split by output dimension
            n = len(targets)
            out_dim = lora_up.shape[0] // n
            for i, (path, _) in enumerate(targets):
                converted[f"{path}.lora_B"] = lora_up[i*out_dim:(i+1)*out_dim, :]
    
    return converted
```

### Detailed Example: QKV Conversion

**Input** (LongCat format):
```python
{
    # lora_down: [384, 4096] = [rank*3, in_dim]
    "lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_down.weight": 
        torch.randn([384, 4096]),
    
    # lora_up blocks: 3 × [4096, 128] = 3 × [out_dim, rank]
    "lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_up.blocks.0.weight": 
        torch.randn([4096, 128]),
    "lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_up.blocks.1.weight": 
        torch.randn([4096, 128]),
    "lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_up.blocks.2.weight": 
        torch.randn([4096, 128]),
}
```

**Processing**:
```python
# 1. Parse: "blocks.0.attn.qkv"
module_path = "blocks.0.attn.qkv"

# 2. Map to targets
targets = [
    ("blocks.0.self_attn.to_q", "q"),
    ("blocks.0.self_attn.to_k", "k"),
    ("blocks.0.self_attn.to_v", "v"),
]

# 3. Split lora_down [384, 4096] → 3 × [128, 4096]
lora_down = source["...qkv.lora_down.weight"]  # [384, 4096]
rank = 384 // 3 = 128
q_lora_A = lora_down[0:128, :]    # [128, 4096]
k_lora_A = lora_down[128:256, :]  # [128, 4096]
v_lora_A = lora_down[256:384, :]  # [128, 4096]

# 4. Concatenate lora_up blocks [4096, 128] → [12288, 128]
lora_up = torch.cat([
    source["...qkv.lora_up.blocks.0.weight"],  # [4096, 128]
    source["...qkv.lora_up.blocks.1.weight"],  # [4096, 128]
    source["...qkv.lora_up.blocks.2.weight"],  # [4096, 128]
], dim=0)  # → [12288, 128]

# 5. Split lora_up [12288, 128] → 3 × [4096, 128]
out_dim = 12288 // 3 = 4096
q_lora_B = lora_up[0:4096, :]      # [4096, 128]
k_lora_B = lora_up[4096:8192, :]   # [4096, 128]
v_lora_B = lora_up[8192:12288, :]  # [4096, 128]
```

**Output** (FastVideo format):
```python
{
    "blocks.0.self_attn.to_q.lora_A": torch.Tensor([128, 4096]),
    "blocks.0.self_attn.to_q.lora_B": torch.Tensor([4096, 128]),
    
    "blocks.0.self_attn.to_k.lora_A": torch.Tensor([128, 4096]),
    "blocks.0.self_attn.to_k.lora_B": torch.Tensor([4096, 128]),
    
    "blocks.0.self_attn.to_v.lora_A": torch.Tensor([128, 4096]),
    "blocks.0.self_attn.to_v.lora_B": torch.Tensor([4096, 128]),
}
```

### Why This Works

The key insight is that LongCat's n_separate design is **mathematically equivalent** to separate LoRAs:

```python
# LongCat n_separate (what we receive):
lora_A_fused = [rank*3, in_dim]
lora_B_q = [out_dim, rank]
lora_B_k = [out_dim, rank]
lora_B_v = [out_dim, rank]

q_out = base_q + lora_B_q @ lora_A_fused[0:rank]       @ x
k_out = base_k + lora_B_k @ lora_A_fused[rank:2*rank]  @ x
v_out = base_v + lora_B_v @ lora_A_fused[2*rank:3*rank] @ x

# FastVideo separate LoRAs (what we create):
q_lora_A = lora_A_fused[0:rank]
k_lora_A = lora_A_fused[rank:2*rank]
v_lora_A = lora_A_fused[2*rank:3*rank]

q_out = base_q + lora_B_q @ q_lora_A @ x  # Same!
k_out = base_k + lora_B_k @ k_lora_A @ x  # Same!
v_out = base_v + lora_B_v @ v_lora_A @ x  # Same!
```

---

## Usage Examples

### Running Conversion

```bash
python scripts/checkpoint_conversion/longcat_to_fastvideo.py \
    --source /path/to/LongCat-Video/weights/LongCat-Video \
    --output weights/longcat-native \
    --validate
```

**Output structure**:
```
weights/longcat-native/
├── transformer/
│   └── model.safetensors          # Converted base model
├── lora/
│   ├── cfg_step_lora.safetensors   # Converted distilled LoRA
│   └── refinement_lora.safetensors # Converted refinement LoRA
├── vae/
├── text_encoder/
├── tokenizer/
├── scheduler/
└── model_index.json
```

### Loading in FastVideo

Once you implement `LongCatLoRAPipeline`:

```python
from fastvideo import VideoGenerator

# Option 1: Load with LoRA
generator = VideoGenerator.from_pretrained(
    "weights/longcat-native",
    pipeline="longcat_lora",  # LoRA-enabled pipeline
    lora_path="weights/longcat-native/lora/cfg_step_lora.safetensors",
    lora_nickname="distilled"
)

# Generate with distilled model (16 steps)
video = generator.generate_video(
    prompt="A cat playing piano",
    num_inference_steps=16,
    guidance_scale=1.0,  # Distilled uses CFG-free
)

# Option 2: Switch LoRAs
generator.set_lora_adapter(
    lora_nickname="refinement",
    lora_path="weights/longcat-native/lora/refinement_lora.safetensors"
)
```

---

## Validation

The conversion includes validation:

```python
def validate_conversion(original, converted):
    # 1. Check parameter count (accounting for splits)
    orig_params = sum(p.numel() for k, p in original.items() 
                     if "alpha_scale" not in k)
    conv_params = sum(p.numel() for k, p in converted.items() 
                     if "alpha_scale" not in k)
    assert orig_params == conv_params
    
    # 2. Verify QKV split reconstruction
    qkv_up_blocks = [original[f"...qkv.lora_up.blocks.{i}.weight"] 
                     for i in range(3)]
    orig_qkv_up = torch.cat(qkv_up_blocks, dim=0)
    
    conv_qkv_up = torch.cat([
        converted["blocks.0.self_attn.to_q.lora_B"],
        converted["blocks.0.self_attn.to_k.lora_B"],
        converted["blocks.0.self_attn.to_v.lora_B"],
    ], dim=0)
    
    assert torch.allclose(orig_qkv_up, conv_qkv_up)
```

---

## Technical Notes

### Why Not Use lora_param_names_mapping?

FastVideo has a `lora_param_names_mapping` config option, but we **don't use it** for LongCat because:

1. **n_separate complexity**: The mapping would need to handle concatenation and splitting, which is beyond simple regex
2. **Cleaner separation**: Conversion script handles all format differences
3. **One-time cost**: Conversion happens once, not at every load
4. **Validation**: Can validate the conversion thoroughly before use

### Memory Efficiency

The conversion:
- ✅ Processes one LoRA file at a time
- ✅ No unnecessary copies (uses views where possible)
- ✅ Outputs directly to safetensors (no intermediate storage)

Typical memory usage: ~2GB peak for refinement LoRA conversion

### Future Extensions

If you want to support **runtime loading** of original LongCat LoRAs (without pre-conversion):

1. Implement `lora_param_names_mapping` in config
2. Add n_separate handling in `LoRAPipeline.set_lora_adapter()`
3. Trade-off: More complex loading logic vs pre-conversion simplicity

---

## Summary

| Aspect | LongCat LoRA | FastVideo LoRA | Conversion |
|--------|--------------|----------------|------------|
| **Key format** | `lora___lorahyphen___path` | `path.lora_A/B` | Parse & rename |
| **Down-proj** | `lora_down.weight` | `lora_A` | Rename & split |
| **Up-proj** | `lora_up.blocks.X.weight` | `lora_B` | Concat & split |
| **Fused QKV** | Single module (n_separate=3) | 3 modules (Q/K/V) | Split 3-way |
| **Fused KV** | Single module (n_separate=2) | 2 modules (K/V) | Split 2-way |
| **Alpha** | `.alpha_scale` tensor | Constructor param | Extract value |

**Key achievement**: Converts LongCat's n_separate design to FastVideo's separate-projection architecture while preserving mathematical equivalence.




This document explains the LoRA weight formats, FastVideo's expectations, and the conversion implementation.

## Table of Contents
1. [LoRA Background](#lora-background)
2. [LongCat LoRA Format](#longcat-lora-format)
3. [FastVideo LoRA Format](#fastvideo-lora-format)
4. [Conversion Implementation](#conversion-implementation)
5. [Usage Examples](#usage-examples)

---

## LoRA Background

### What is LoRA?

LoRA (Low-Rank Adaptation) adds trainable low-rank matrices to frozen model weights:

```
Original: y = Wx
LoRA:     y = Wx + BAx
```

Where:
- `W`: Frozen pretrained weights [out_dim, in_dim]
- `A`: LoRA down-projection [rank, in_dim] (called `lora_down` or `lora_A`)
- `B`: LoRA up-projection [out_dim, rank] (called `lora_up` or `lora_B`)
- `rank`: LoRA rank (typically 8-128)
- `alpha`: Scaling factor (typically rank/2)

**Final output**: `y = Wx + (alpha/rank) * BAx`

### Why LoRA?

- ✅ **Small file size**: 1-10MB vs 16GB for full model
- ✅ **Fast training**: Only train A and B matrices
- ✅ **Composable**: Can switch between different LoRAs
- ✅ **Preserves base model**: Base weights stay frozen

---

## LongCat LoRA Format

### File Structure

LongCat uses the **original LoRA implementation** format with some special features:

#### 1. Key Naming Convention

Keys use `___lorahyphen___` as separator (replaces `.`):

```
Original module path:      blocks.0.attn.qkv
LongCat LoRA key:          lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_down.weight
```

#### 2. Weight Components

Each LoRA module has 3 components:

```python
# Example: blocks.0.attn.qkv
{
    "lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_down.weight": 
        torch.Tensor([384, 4096]),  # rank*n_separate × in_dim
    
    "lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_up.blocks.0.weight": 
        torch.Tensor([4096, 128]),  # out_dim/n_separate × rank
    "lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_up.blocks.1.weight": 
        torch.Tensor([4096, 128]),
    "lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_up.blocks.2.weight": 
        torch.Tensor([4096, 128]),
    
    "lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.alpha_scale": 
        torch.Tensor([])  # Scalar: alpha/rank = 64/128 = 0.5
}
```

#### 3. n_separate Feature

**Key innovation**: LongCat uses `n_separate` for fused projections.

Instead of:
```python
# Standard LoRA
qkv_weight = base_qkv + lora_B @ lora_A
```

LongCat uses:
```python
# n_separate=3 for QKV
lora_A = [rank*3, in_dim]           # Fused down-projection
lora_B_q = [out_dim, rank]          # Separate up-projections
lora_B_k = [out_dim, rank]
lora_B_v = [out_dim, rank]

q = base_q + lora_B_q @ lora_A[0:rank]
k = base_k + lora_B_k @ lora_A[rank:2*rank]
v = base_v + lora_B_v @ lora_A[2*rank:3*rank]
```

**Why?** This allows independent adaptation of Q/K/V while sharing the input projection.

### Examples from LongCat LoRA Files

#### cfg_step_lora.safetensors (1,152 keys)

Used for **distilled 16-step generation**:

```
Self-Attention:
  - blocks.{0-47}.attn.qkv           (n_separate=3: Q, K, V)
  - blocks.{0-47}.attn.proj          (single output)

Cross-Attention:
  - blocks.{0-47}.cross_attn.q_linear       (single Q)
  - blocks.{0-47}.cross_attn.kv_linear      (n_separate=2: K, V)

FFN:
  - blocks.{0-47}.ffn.w1             (gate)
  - blocks.{0-47}.ffn.w2             (down)
  - blocks.{0-47}.ffn.w3             (up)
```

#### refinement_lora.safetensors (1,543 keys)

Used for **480p→720p refinement**:

Everything in cfg_step_lora **PLUS**:

```
AdaLN Modulation:
  - blocks.{0-47}.adaLN_modulation.1  (n_separate=6: shifts/scales/gates)

Final Layer:
  - final_layer.adaLN_modulation.1    (n_separate=2: shift, scale)
  - final_layer.linear                (output projection)
```

### LoRA Parameters

Both files use:
- **Rank**: 128
- **Alpha**: 64
- **Alpha scale**: 0.5 (= 64/128)

---

## FastVideo LoRA Format

### FastVideo's LoRA Architecture

FastVideo uses `BaseLayerWithLoRA` from SGLang:

```python
class BaseLayerWithLoRA(nn.Module):
    def __init__(self, base_layer, lora_rank, lora_alpha):
        self.base_layer = base_layer  # Original linear layer
        self.lora_A = nn.Parameter(...)  # [rank, in_dim]
        self.lora_B = nn.Parameter(...)  # [out_dim, rank]
    
    def forward(self, x):
        out, bias = self.base_layer(x)
        delta = x @ self.lora_A.T @ self.lora_B.T
        delta = delta * (self.lora_alpha / self.lora_rank)
        return out + delta, bias
```

### Key Expectations

#### 1. Key Naming

FastVideo expects **clean module paths** with `.lora_A` and `.lora_B`:

```
blocks.0.self_attn.to_q.lora_A      # [rank, in_dim]
blocks.0.self_attn.to_q.lora_B      # [out_dim, rank]
blocks.0.self_attn.to_k.lora_A
blocks.0.self_attn.to_k.lora_B
...
```

**No** `lora___lorahyphen___` separators, **no** `.lora_down`/`.lora_up` naming.

#### 2. Separate Projections

FastVideo's native LongCat uses **separate** Q/K/V projections (not fused):

```python
# Native implementation
class LongCatSelfAttention(nn.Module):
    def __init__(self, ...):
        self.to_q = ReplicatedLinear(dim, dim)  # Separate
        self.to_k = ReplicatedLinear(dim, dim)  # Separate
        self.to_v = ReplicatedLinear(dim, dim)  # Separate
```

So LoRA must provide **separate** `to_q`, `to_k`, `to_v` LoRAs.

#### 3. Weight Shapes

```python
# For a layer with in_dim=4096, out_dim=4096, rank=128:
lora_A: torch.Tensor([128, 4096])      # [rank, in_dim]
lora_B: torch.Tensor([4096, 128])      # [out_dim, rank]
```

#### 4. Loading Process

```python
# In LoRAPipeline.set_lora_adapter():
for name, layer in self.lora_layers.items():
    # name: "blocks.0.self_attn.to_q"
    lora_A = lora_dict[f"{name}.lora_A"]  # [rank, in_dim]
    lora_B = lora_dict[f"{name}.lora_B"]  # [out_dim, rank]
    layer.set_lora_weights(lora_A, lora_B)
```

### Module Naming

FastVideo native LongCat uses:

| Component | Original LongCat | FastVideo Native |
|-----------|-----------------|------------------|
| Self-attn QKV | `attn.qkv` | `self_attn.to_q`, `self_attn.to_k`, `self_attn.to_v` |
| Self-attn output | `attn.proj` | `self_attn.to_out` |
| Cross-attn Q | `cross_attn.q_linear` | `cross_attn.to_q` |
| Cross-attn KV | `cross_attn.kv_linear` | `cross_attn.to_k`, `cross_attn.to_v` |
| FFN | `ffn.w1/w2/w3` | `ffn.w1/w2/w3` (same) |
| AdaLN | `adaLN_modulation.1` | `adaln_linear_1` |

---

## Conversion Implementation

### Overview

The conversion in `longcat_to_fastvideo.py` handles:

1. ✅ **Key parsing**: Extract module path from `lora___lorahyphen___` format
2. ✅ **Module mapping**: Map LongCat paths to FastVideo paths
3. ✅ **n_separate handling**: Concatenate blocks and split fused projections
4. ✅ **Weight renaming**: Convert to `.lora_A` / `.lora_B` format

### Step-by-Step Process

#### Step 1: Parse LongCat Keys

```python
def parse_lora_key(key: str) -> tuple[str, str]:
    """
    Parse: "lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_down.weight"
    Return: ("blocks.0.attn.qkv", "lora_down.weight")
    """
    # Remove prefix
    if key.startswith("lora___lorahyphen___"):
        key = key[len("lora___lorahyphen___"):]
    
    # Replace separators: ___lorahyphen___ → .
    key = key.replace("___lorahyphen___", ".")
    
    # Extract module path and weight type
    if ".lora_down.weight" in key:
        return key.replace(".lora_down.weight", ""), "lora_down.weight"
    elif ".lora_up.blocks.0.weight" in key:
        # Handle n_separate blocks
        match = re.match(r"(.+)\.lora_up\.blocks\.(\d+)\.weight", key)
        return match.group(1), f"lora_up.blocks.{match.group(2)}.weight"
    # ... etc
```

#### Step 2: Map to FastVideo Paths

```python
def map_lora_module(module_path: str) -> list[tuple[str, str]]:
    """
    Map LongCat module path to FastVideo paths.
    Returns list because fused projections → multiple targets.
    """
    # QKV (fused) → separate Q, K, V
    if re.match(r"blocks\.(\d+)\.attn\.qkv", module_path):
        block_idx = match.group(1)
        return [
            (f"blocks.{block_idx}.self_attn.to_q", "q"),
            (f"blocks.{block_idx}.self_attn.to_k", "k"),
            (f"blocks.{block_idx}.self_attn.to_v", "v"),
        ]
    
    # KV (fused) → separate K, V
    if re.match(r"blocks\.(\d+)\.cross_attn\.kv_linear", module_path):
        block_idx = match.group(1)
        return [
            (f"blocks.{block_idx}.cross_attn.to_k", "k"),
            (f"blocks.{block_idx}.cross_attn.to_v", "v"),
        ]
    
    # Single output (no splitting)
    if re.match(r"blocks\.(\d+)\.attn\.proj", module_path):
        block_idx = match.group(1)
        return [(f"blocks.{block_idx}.self_attn.to_out", "single")]
    
    # ... etc
```

#### Step 3: Handle n_separate

```python
def convert_lora_weights(source_weights, lora_name):
    # Group weights by module
    modules = {}  # {module_path: {weight_type: key}}
    
    for key in source_weights.keys():
        module_path, weight_type = parse_lora_key(key)
        if module_path not in modules:
            modules[module_path] = {}
        modules[module_path][weight_type] = key
    
    converted = OrderedDict()
    
    for module_path, weight_keys in modules.items():
        targets = map_lora_module(module_path)  # May return multiple targets
        
        # === Handle lora_down (becomes lora_A) ===
        lora_down = source_weights[weight_keys["lora_down.weight"]]
        # Shape: [rank * n_separate, in_dim]
        
        if len(targets) == 1:
            # Single target: no splitting needed
            converted[f"{targets[0][0]}.lora_A"] = lora_down
        else:
            # Multiple targets: split by rank
            n = len(targets)
            rank = lora_down.shape[0] // n
            for i, (path, _) in enumerate(targets):
                # Extract rank slice for this target
                converted[f"{path}.lora_A"] = lora_down[i*rank:(i+1)*rank, :]
        
        # === Handle lora_up blocks (becomes lora_B) ===
        lora_up_blocks = []
        i = 0
        while f"lora_up.blocks.{i}.weight" in weight_keys:
            lora_up_blocks.append(source_weights[weight_keys[f"lora_up.blocks.{i}.weight"]])
            i += 1
        
        if lora_up_blocks:
            # Concatenate n_separate blocks along output dimension
            lora_up = torch.cat(lora_up_blocks, dim=0)
            # Shape: [out_dim_total, rank]
        else:
            # Single lora_up (no n_separate)
            lora_up = source_weights[weight_keys["lora_up.weight"]]
        
        if len(targets) == 1:
            # Single target
            converted[f"{targets[0][0]}.lora_B"] = lora_up
        else:
            # Multiple targets: split by output dimension
            n = len(targets)
            out_dim = lora_up.shape[0] // n
            for i, (path, _) in enumerate(targets):
                converted[f"{path}.lora_B"] = lora_up[i*out_dim:(i+1)*out_dim, :]
    
    return converted
```

### Detailed Example: QKV Conversion

**Input** (LongCat format):
```python
{
    # lora_down: [384, 4096] = [rank*3, in_dim]
    "lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_down.weight": 
        torch.randn([384, 4096]),
    
    # lora_up blocks: 3 × [4096, 128] = 3 × [out_dim, rank]
    "lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_up.blocks.0.weight": 
        torch.randn([4096, 128]),
    "lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_up.blocks.1.weight": 
        torch.randn([4096, 128]),
    "lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_up.blocks.2.weight": 
        torch.randn([4096, 128]),
}
```

**Processing**:
```python
# 1. Parse: "blocks.0.attn.qkv"
module_path = "blocks.0.attn.qkv"

# 2. Map to targets
targets = [
    ("blocks.0.self_attn.to_q", "q"),
    ("blocks.0.self_attn.to_k", "k"),
    ("blocks.0.self_attn.to_v", "v"),
]

# 3. Split lora_down [384, 4096] → 3 × [128, 4096]
lora_down = source["...qkv.lora_down.weight"]  # [384, 4096]
rank = 384 // 3 = 128
q_lora_A = lora_down[0:128, :]    # [128, 4096]
k_lora_A = lora_down[128:256, :]  # [128, 4096]
v_lora_A = lora_down[256:384, :]  # [128, 4096]

# 4. Concatenate lora_up blocks [4096, 128] → [12288, 128]
lora_up = torch.cat([
    source["...qkv.lora_up.blocks.0.weight"],  # [4096, 128]
    source["...qkv.lora_up.blocks.1.weight"],  # [4096, 128]
    source["...qkv.lora_up.blocks.2.weight"],  # [4096, 128]
], dim=0)  # → [12288, 128]

# 5. Split lora_up [12288, 128] → 3 × [4096, 128]
out_dim = 12288 // 3 = 4096
q_lora_B = lora_up[0:4096, :]      # [4096, 128]
k_lora_B = lora_up[4096:8192, :]   # [4096, 128]
v_lora_B = lora_up[8192:12288, :]  # [4096, 128]
```

**Output** (FastVideo format):
```python
{
    "blocks.0.self_attn.to_q.lora_A": torch.Tensor([128, 4096]),
    "blocks.0.self_attn.to_q.lora_B": torch.Tensor([4096, 128]),
    
    "blocks.0.self_attn.to_k.lora_A": torch.Tensor([128, 4096]),
    "blocks.0.self_attn.to_k.lora_B": torch.Tensor([4096, 128]),
    
    "blocks.0.self_attn.to_v.lora_A": torch.Tensor([128, 4096]),
    "blocks.0.self_attn.to_v.lora_B": torch.Tensor([4096, 128]),
}
```

### Why This Works

The key insight is that LongCat's n_separate design is **mathematically equivalent** to separate LoRAs:

```python
# LongCat n_separate (what we receive):
lora_A_fused = [rank*3, in_dim]
lora_B_q = [out_dim, rank]
lora_B_k = [out_dim, rank]
lora_B_v = [out_dim, rank]

q_out = base_q + lora_B_q @ lora_A_fused[0:rank]       @ x
k_out = base_k + lora_B_k @ lora_A_fused[rank:2*rank]  @ x
v_out = base_v + lora_B_v @ lora_A_fused[2*rank:3*rank] @ x

# FastVideo separate LoRAs (what we create):
q_lora_A = lora_A_fused[0:rank]
k_lora_A = lora_A_fused[rank:2*rank]
v_lora_A = lora_A_fused[2*rank:3*rank]

q_out = base_q + lora_B_q @ q_lora_A @ x  # Same!
k_out = base_k + lora_B_k @ k_lora_A @ x  # Same!
v_out = base_v + lora_B_v @ v_lora_A @ x  # Same!
```

---

## Usage Examples

### Running Conversion

```bash
python scripts/checkpoint_conversion/longcat_to_fastvideo.py \
    --source /path/to/LongCat-Video/weights/LongCat-Video \
    --output weights/longcat-native \
    --validate
```

**Output structure**:
```
weights/longcat-native/
├── transformer/
│   └── model.safetensors          # Converted base model
├── lora/
│   ├── cfg_step_lora.safetensors   # Converted distilled LoRA
│   └── refinement_lora.safetensors # Converted refinement LoRA
├── vae/
├── text_encoder/
├── tokenizer/
├── scheduler/
└── model_index.json
```

### Loading in FastVideo

Once you implement `LongCatLoRAPipeline`:

```python
from fastvideo import VideoGenerator

# Option 1: Load with LoRA
generator = VideoGenerator.from_pretrained(
    "weights/longcat-native",
    pipeline="longcat_lora",  # LoRA-enabled pipeline
    lora_path="weights/longcat-native/lora/cfg_step_lora.safetensors",
    lora_nickname="distilled"
)

# Generate with distilled model (16 steps)
video = generator.generate_video(
    prompt="A cat playing piano",
    num_inference_steps=16,
    guidance_scale=1.0,  # Distilled uses CFG-free
)

# Option 2: Switch LoRAs
generator.set_lora_adapter(
    lora_nickname="refinement",
    lora_path="weights/longcat-native/lora/refinement_lora.safetensors"
)
```

---

## Validation

The conversion includes validation:

```python
def validate_conversion(original, converted):
    # 1. Check parameter count (accounting for splits)
    orig_params = sum(p.numel() for k, p in original.items() 
                     if "alpha_scale" not in k)
    conv_params = sum(p.numel() for k, p in converted.items() 
                     if "alpha_scale" not in k)
    assert orig_params == conv_params
    
    # 2. Verify QKV split reconstruction
    qkv_up_blocks = [original[f"...qkv.lora_up.blocks.{i}.weight"] 
                     for i in range(3)]
    orig_qkv_up = torch.cat(qkv_up_blocks, dim=0)
    
    conv_qkv_up = torch.cat([
        converted["blocks.0.self_attn.to_q.lora_B"],
        converted["blocks.0.self_attn.to_k.lora_B"],
        converted["blocks.0.self_attn.to_v.lora_B"],
    ], dim=0)
    
    assert torch.allclose(orig_qkv_up, conv_qkv_up)
```

---

## Technical Notes

### Why Not Use lora_param_names_mapping?

FastVideo has a `lora_param_names_mapping` config option, but we **don't use it** for LongCat because:

1. **n_separate complexity**: The mapping would need to handle concatenation and splitting, which is beyond simple regex
2. **Cleaner separation**: Conversion script handles all format differences
3. **One-time cost**: Conversion happens once, not at every load
4. **Validation**: Can validate the conversion thoroughly before use

### Memory Efficiency

The conversion:
- ✅ Processes one LoRA file at a time
- ✅ No unnecessary copies (uses views where possible)
- ✅ Outputs directly to safetensors (no intermediate storage)

Typical memory usage: ~2GB peak for refinement LoRA conversion

### Future Extensions

If you want to support **runtime loading** of original LongCat LoRAs (without pre-conversion):

1. Implement `lora_param_names_mapping` in config
2. Add n_separate handling in `LoRAPipeline.set_lora_adapter()`
3. Trade-off: More complex loading logic vs pre-conversion simplicity

---

## Summary

| Aspect | LongCat LoRA | FastVideo LoRA | Conversion |
|--------|--------------|----------------|------------|
| **Key format** | `lora___lorahyphen___path` | `path.lora_A/B` | Parse & rename |
| **Down-proj** | `lora_down.weight` | `lora_A` | Rename & split |
| **Up-proj** | `lora_up.blocks.X.weight` | `lora_B` | Concat & split |
| **Fused QKV** | Single module (n_separate=3) | 3 modules (Q/K/V) | Split 3-way |
| **Fused KV** | Single module (n_separate=2) | 2 modules (K/V) | Split 2-way |
| **Alpha** | `.alpha_scale` tensor | Constructor param | Extract value |

**Key achievement**: Converts LongCat's n_separate design to FastVideo's separate-projection architecture while preserving mathematical equivalence.






