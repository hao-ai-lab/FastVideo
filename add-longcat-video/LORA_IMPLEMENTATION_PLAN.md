# LoRA Implementation Plan for Native LongCat-Video

## Overview

This document outlines the plan to add LoRA (Low-Rank Adaptation) support to the native FastVideo LongCat implementation. LoRA is needed to run:
1. **Distilled model** (`cfg_step_lora.safetensors`) - 16-step fast generation
2. **Refinement model** (`refinement_lora.safetensors`) - 480p→720p upscaling

## FastVideo LoRA Architecture Analysis

### Key Components

1. **BaseLayerWithLoRA** (`fastvideo/layers/lora/linear.py`)
   - Wraps any `LinearBase` layer (ReplicatedLinear, ColumnParallelLinear, etc.)
   - Stores `lora_A` (rank × in_dim) and `lora_B` (out_dim × rank) matrices
   - Forward: `output = base_layer(x) + (x @ lora_A.T @ lora_B.T) * (alpha/rank)`
   - Supports weight merging for inference (permanently adds LoRA to base weights)
   - Handles distributed tensors (DTensor) for FSDP

2. **LoRAPipeline** (`fastvideo/pipelines/lora_pipeline.py`)
   - Mixin class that extends `ComposedPipelineBase`
   - Methods:
     - `convert_to_lora_layers()`: Wraps target linear layers with LoRA
     - `set_lora_adapter(nickname, path)`: Loads and applies LoRA weights
     - `merge_lora_weights()`: Permanently merges LoRA into base weights
   - Uses `lora_target_modules` to filter which layers get LoRA (e.g., ["to_q", "to_k", "to_v", "to_out"])
   - Uses `exclude_lora_layers` to exclude embeddings/etc.

3. **Model Integration Points**
   - Each model defines:
     - `param_names_mapping`: HF names → FastVideo names
     - `lora_param_names_mapping`: Official names → HF names (for LoRA weights)
     - `exclude_lora_layers`: Layers to skip
   - Models inherit from `CachableDiT` which has `lora_param_names_mapping` attribute

### Mapping System

FastVideo uses a two-stage mapping for LoRA weights:
1. `lora_param_names_mapping`: Original model naming → HF diffusers naming
2. `param_names_mapping`: HF naming → FastVideo internal naming

Example from WanVideo:
```python
lora_param_names_mapping = {
    r"^blocks\.(\d+)\.self_attn\.q\.(.*)$": r"blocks.\1.attn1.to_q.\2",
    r"^blocks\.(\d+)\.self_attn\.k\.(.*)$": r"blocks.\1.attn1.to_k.\2",
    # ... etc
}
```

## Implementation Plan

### Phase 1: Update Configuration ✅

**File**: `fastvideo/configs/models/dits/longcat.py`

Add `lora_param_names_mapping` to `LongCatVideoArchConfig`:

```python
lora_param_names_mapping: dict = field(
    default_factory=lambda: {
        # Map official LongCat LoRA naming to our internal naming
        # Official LongCat uses patterns like:
        # "lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_down.weight"
        
        # Self-attention Q/K/V (if LoRA uses fused QKV)
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___attn___lorahyphen___qkv\.lora_A\.(.*)$": 
            r"blocks.\1.self_attn.to_qkv.\2",  # Will need special handling
        
        # Self-attention Q/K/V (separate)
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___attn___lorahyphen___to_q\.lora_A\.(.*)$": 
            r"blocks.\1.self_attn.to_q.lora_A",
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___attn___lorahyphen___to_k\.lora_A\.(.*)$": 
            r"blocks.\1.self_attn.to_k.lora_A",
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___attn___lorahyphen___to_v\.lora_A\.(.*)$": 
            r"blocks.\1.self_attn.to_v.lora_A",
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___attn___lorahyphen___proj\.lora_A\.(.*)$": 
            r"blocks.\1.self_attn.to_out.lora_A",
            
        # Self-attention B matrices
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___attn___lorahyphen___to_q\.lora_B\.(.*)$": 
            r"blocks.\1.self_attn.to_q.lora_B",
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___attn___lorahyphen___to_k\.lora_B\.(.*)$": 
            r"blocks.\1.self_attn.to_k.lora_B",
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___attn___lorahyphen___to_v\.lora_B\.(.*)$": 
            r"blocks.\1.self_attn.to_v.lora_B",
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___attn___lorahyphen___proj\.lora_B\.(.*)$": 
            r"blocks.\1.self_attn.to_out.lora_B",
        
        # Cross-attention Q/K/V
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___cross_attn___lorahyphen___q_linear\.lora_A\.(.*)$": 
            r"blocks.\1.cross_attn.to_q.lora_A",
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___cross_attn___lorahyphen___kv_linear\.lora_A\.(.*)$": 
            r"blocks.\1.cross_attn.to_kv.lora_A",  # Will need special handling for K/V split
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___cross_attn___lorahyphen___proj\.lora_A\.(.*)$": 
            r"blocks.\1.cross_attn.to_out.lora_A",
            
        # Cross-attention B matrices
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___cross_attn___lorahyphen___q_linear\.lora_B\.(.*)$": 
            r"blocks.\1.cross_attn.to_q.lora_B",
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___cross_attn___lorahyphen___kv_linear\.lora_B\.(.*)$": 
            r"blocks.\1.cross_attn.to_kv.lora_B",
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___cross_attn___lorahyphen___proj\.lora_B\.(.*)$": 
            r"blocks.\1.cross_attn.to_out.lora_B",
        
        # FFN layers (w1=gate, w2=down, w3=up)
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___ffn___lorahyphen___w1\.lora_A\.(.*)$": 
            r"blocks.\1.ffn.w1.lora_A",
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___ffn___lorahyphen___w2\.lora_A\.(.*)$": 
            r"blocks.\1.ffn.w2.lora_A",
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___ffn___lorahyphen___w3\.lora_A\.(.*)$": 
            r"blocks.\1.ffn.w3.lora_A",
            
        # FFN B matrices
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___ffn___lorahyphen___w1\.lora_B\.(.*)$": 
            r"blocks.\1.ffn.w1.lora_B",
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___ffn___lorahyphen___w2\.lora_B\.(.*)$": 
            r"blocks.\1.ffn.w2.lora_B",
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___ffn___lorahyphen___w3\.lora_B\.(.*)$": 
            r"blocks.\1.ffn.w3.lora_B",
    })

exclude_lora_layers: list[str] = field(default_factory=lambda: ["embedder", "embed"])
```

**Note**: The exact LoRA naming pattern needs to be verified by inspecting the actual `.safetensors` files.

### Phase 2: Update Model Definition ✅

**File**: `fastvideo/models/dits/longcat.py`

Update class attributes:
```python
class LongCatTransformer3DModel(CachableDiT):
    # ... existing code ...
    
    # Add LoRA mapping (read from config)
    lora_param_names_mapping = {}  # Will be set from config
    
    def __init__(self, config: LongCatVideoConfig, hf_config: dict[str, Any]):
        super().__init__(config=config, hf_config=hf_config)
        # Set class-level mapping from config
        self.__class__.lora_param_names_mapping = config.arch_config.lora_param_names_mapping
        # ... rest of init ...
```

### Phase 3: Inspect LoRA Files

**Action**: Examine the actual LoRA safetensors files to understand naming convention

```python
from safetensors.torch import load_file

# Load and inspect LoRA files
cfg_lora = load_file("/path/to/cfg_step_lora.safetensors")
refinement_lora = load_file("/path/to/refinement_lora.safetensors")

# Print all keys to understand structure
print("CFG LoRA keys:")
for key in sorted(cfg_lora.keys())[:20]:
    print(f"  {key}: {cfg_lora[key].shape}")

print("\nRefinement LoRA keys:")
for key in sorted(refinement_lora.keys())[:20]:
    print(f"  {key}: {refinement_lora[key].shape}")
```

### Phase 4: Create LoRA-enabled Pipeline

**File**: `fastvideo/pipelines/basic/longcat/longcat_lora_pipeline.py` (new)

```python
from fastvideo.pipelines.lora_pipeline import LoRAPipeline
from fastvideo.pipelines.basic.longcat.longcat_pipeline import LongCatPipeline


class LongCatLoRAPipeline(LoRAPipeline, LongCatPipeline):
    """
    LongCat pipeline with LoRA support for distilled and refinement models.
    """
    
    def __init__(self, *args, **kwargs):
        # LoRAPipeline.__init__ will call convert_to_lora_layers if needed
        super().__init__(*args, **kwargs)
    
    # Inherit all methods from both parent classes
    # LoRAPipeline provides: set_lora_adapter, merge_lora_weights, etc.
    # LongCatPipeline provides: generate_t2v, etc.
```

### Phase 5: Register Pipeline

**File**: `fastvideo/pipelines/__init__.py`

Add import:
```python
from fastvideo.pipelines.basic.longcat.longcat_lora_pipeline import LongCatLoRAPipeline
```

**File**: `fastvideo/pipelines/pipeline_registry.py`

Register pipeline:
```python
"longcat_lora": {
    "class": LongCatLoRAPipeline,
    "config_cls": LongCatVideoConfig,
},
```

### Phase 6: Test LoRA Loading

**File**: `test_longcat_lora_inference.py` (new)

```python
import torch
from fastvideo import VideoGenerator
from safetensors.torch import load_file

def test_lora_loading():
    """Test loading LoRA weights without inference."""
    
    # Initialize generator
    generator = VideoGenerator.from_pretrained(
        "/path/to/longcat/weights",
        pipeline="longcat_lora",  # Use LoRA-enabled pipeline
        num_gpus=1,
        dit_cpu_offload=False,
        lora_path="/path/to/cfg_step_lora.safetensors",
        lora_nickname="cfg_distill"
    )
    
    print("✅ LoRA loaded successfully!")
    
    # Check which layers got LoRA
    pipeline = generator.pipeline
    print(f"LoRA layers count: {len(pipeline.lora_layers)}")
    for name, layer in list(pipeline.lora_layers.items())[:5]:
        print(f"  {name}: lora_A={layer.lora_A.shape}, lora_B={layer.lora_B.shape}")


def test_lora_switching():
    """Test switching between different LoRA adapters."""
    
    generator = VideoGenerator.from_pretrained(
        "/path/to/longcat/weights",
        pipeline="longcat_lora",
        num_gpus=1,
    )
    
    # Load first LoRA
    generator.set_lora_adapter(
        lora_nickname="cfg_distill",
        lora_path="/path/to/cfg_step_lora.safetensors"
    )
    print("✅ Loaded distill LoRA")
    
    # Switch to refinement LoRA
    generator.set_lora_adapter(
        lora_nickname="refinement",
        lora_path="/path/to/refinement_lora.safetensors"
    )
    print("✅ Switched to refinement LoRA")


if __name__ == "__main__":
    test_lora_loading()
    test_lora_switching()
```

### Phase 7: Inference with LoRA

**File**: `test_longcat_lora_generation.py` (new)

```python
from fastvideo import VideoGenerator

def test_distill_generation():
    """Test 16-step distilled generation."""
    
    generator = VideoGenerator.from_pretrained(
        "/path/to/longcat/weights",
        pipeline="longcat_lora",
        num_gpus=1,
        lora_path="/path/to/cfg_step_lora.safetensors",
        lora_nickname="cfg_distill"
    )
    
    prompt = "A cat playing piano"
    
    video = generator.generate_video(
        prompt,
        height=480,
        width=832,
        num_frames=93,
        num_inference_steps=16,  # Distilled uses 16 steps
        guidance_scale=1.0,       # Distilled uses CFG-free (guidance=1.0)
        output_path="./output_distill.mp4",
        save_video=True,
    )
    print("✅ Distilled generation complete!")


def test_refinement():
    """Test 480p→720p refinement."""
    
    generator = VideoGenerator.from_pretrained(
        "/path/to/longcat/weights",
        pipeline="longcat_lora",
        num_gpus=1,
        lora_path="/path/to/refinement_lora.safetensors",
        lora_nickname="refinement"
    )
    
    # First generate 480p video (without LoRA)
    generator.set_lora_adapter(lora_nickname=None)  # Disable LoRA
    stage1_video = generator.generate_video(
        prompt="A beautiful sunset",
        height=480,
        width=832,
        num_frames=93,
        num_inference_steps=50,
        guidance_scale=4.0,
    )
    
    # Then refine to 720p (with refinement LoRA)
    generator.set_lora_adapter(
        lora_nickname="refinement",
        lora_path="/path/to/refinement_lora.safetensors"
    )
    
    # Refinement uses stage1 video as conditioning
    # TODO: Implement i2v/refinement mode in pipeline
    refined_video = generator.generate_video_refine(
        prompt="A beautiful sunset",
        stage1_video=stage1_video,
        num_inference_steps=50,
        output_path="./output_refined_720p.mp4",
        save_video=True,
    )
    print("✅ Refinement complete!")


if __name__ == "__main__":
    test_distill_generation()
    # test_refinement()  # Enable after i2v mode is implemented
```

## Implementation Checklist

### Configuration
- [ ] Add `lora_param_names_mapping` to `LongCatVideoArchConfig`
- [ ] Set `exclude_lora_layers` appropriately
- [ ] Update model class to use config mapping

### Pipeline
- [ ] Create `LongCatLoRAPipeline` class
- [ ] Register in pipeline registry
- [ ] Test basic pipeline initialization

### LoRA Files Investigation
- [ ] Load and inspect `cfg_step_lora.safetensors`
- [ ] Load and inspect `refinement_lora.safetensors`
- [ ] Document actual key naming patterns
- [ ] Update `lora_param_names_mapping` based on findings

### Testing
- [ ] Test LoRA layer conversion (without loading weights)
- [ ] Test loading LoRA weights
- [ ] Test LoRA adapter switching
- [ ] Test distilled generation (16 steps, guidance=1.0)
- [ ] Test refinement generation (requires i2v mode)

### Integration
- [ ] Update documentation with LoRA usage examples
- [ ] Add LoRA examples to `examples/inference/lora/`
- [ ] Verify compatibility with distributed inference

## Key Differences: FastVideo vs LongCat Original LoRA

| Aspect | LongCat Original | FastVideo |
|--------|------------------|-----------|
| **Implementation** | Manual forward wrapping | BaseLayerWithLoRA wrapper class |
| **Activation** | enable_loras()/disable_all_loras() | set_lora_adapter()/merge |
| **Weight Merging** | Runtime addition | Permanent merge option |
| **Multi-LoRA** | Multiple LoRAs active simultaneously | One LoRA at a time, switchable |
| **Storage** | LoRA dict with network objects | Adapter state dicts |
| **Distributed** | Manual device/dtype management | DTensor support built-in |

## Notes

1. **Target Modules**: LongCat LoRAs likely only target attention layers (Q, K, V, O) and possibly FFN layers. Check actual files to confirm.

2. **QKV Fusion**: Original LongCat uses fused QKV projection. Need to handle if LoRA weights are also fused.

3. **Refinement Mode**: The refinement LoRA also needs the BSA (block sparse attention) feature, which is a separate implementation task.

4. **Testing Strategy**:
   - Start with distilled model (simpler, no BSA needed)
   - Verify output quality matches reference
   - Then add refinement support

5. **Performance**: FastVideo's weight merging can improve inference speed by permanently adding LoRA weights to base model.

## Next Steps

1. **Immediate**: Inspect actual LoRA safetensors files
2. **Phase 1-2**: Update config and model definition
3. **Phase 3-4**: Create pipeline and test loading
4. **Phase 5**: Test distilled generation
5. **Phase 6**: Implement refinement mode (requires i2v pipeline + BSA)






## Overview

This document outlines the plan to add LoRA (Low-Rank Adaptation) support to the native FastVideo LongCat implementation. LoRA is needed to run:
1. **Distilled model** (`cfg_step_lora.safetensors`) - 16-step fast generation
2. **Refinement model** (`refinement_lora.safetensors`) - 480p→720p upscaling

## FastVideo LoRA Architecture Analysis

### Key Components

1. **BaseLayerWithLoRA** (`fastvideo/layers/lora/linear.py`)
   - Wraps any `LinearBase` layer (ReplicatedLinear, ColumnParallelLinear, etc.)
   - Stores `lora_A` (rank × in_dim) and `lora_B` (out_dim × rank) matrices
   - Forward: `output = base_layer(x) + (x @ lora_A.T @ lora_B.T) * (alpha/rank)`
   - Supports weight merging for inference (permanently adds LoRA to base weights)
   - Handles distributed tensors (DTensor) for FSDP

2. **LoRAPipeline** (`fastvideo/pipelines/lora_pipeline.py`)
   - Mixin class that extends `ComposedPipelineBase`
   - Methods:
     - `convert_to_lora_layers()`: Wraps target linear layers with LoRA
     - `set_lora_adapter(nickname, path)`: Loads and applies LoRA weights
     - `merge_lora_weights()`: Permanently merges LoRA into base weights
   - Uses `lora_target_modules` to filter which layers get LoRA (e.g., ["to_q", "to_k", "to_v", "to_out"])
   - Uses `exclude_lora_layers` to exclude embeddings/etc.

3. **Model Integration Points**
   - Each model defines:
     - `param_names_mapping`: HF names → FastVideo names
     - `lora_param_names_mapping`: Official names → HF names (for LoRA weights)
     - `exclude_lora_layers`: Layers to skip
   - Models inherit from `CachableDiT` which has `lora_param_names_mapping` attribute

### Mapping System

FastVideo uses a two-stage mapping for LoRA weights:
1. `lora_param_names_mapping`: Original model naming → HF diffusers naming
2. `param_names_mapping`: HF naming → FastVideo internal naming

Example from WanVideo:
```python
lora_param_names_mapping = {
    r"^blocks\.(\d+)\.self_attn\.q\.(.*)$": r"blocks.\1.attn1.to_q.\2",
    r"^blocks\.(\d+)\.self_attn\.k\.(.*)$": r"blocks.\1.attn1.to_k.\2",
    # ... etc
}
```

## Implementation Plan

### Phase 1: Update Configuration ✅

**File**: `fastvideo/configs/models/dits/longcat.py`

Add `lora_param_names_mapping` to `LongCatVideoArchConfig`:

```python
lora_param_names_mapping: dict = field(
    default_factory=lambda: {
        # Map official LongCat LoRA naming to our internal naming
        # Official LongCat uses patterns like:
        # "lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_down.weight"
        
        # Self-attention Q/K/V (if LoRA uses fused QKV)
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___attn___lorahyphen___qkv\.lora_A\.(.*)$": 
            r"blocks.\1.self_attn.to_qkv.\2",  # Will need special handling
        
        # Self-attention Q/K/V (separate)
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___attn___lorahyphen___to_q\.lora_A\.(.*)$": 
            r"blocks.\1.self_attn.to_q.lora_A",
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___attn___lorahyphen___to_k\.lora_A\.(.*)$": 
            r"blocks.\1.self_attn.to_k.lora_A",
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___attn___lorahyphen___to_v\.lora_A\.(.*)$": 
            r"blocks.\1.self_attn.to_v.lora_A",
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___attn___lorahyphen___proj\.lora_A\.(.*)$": 
            r"blocks.\1.self_attn.to_out.lora_A",
            
        # Self-attention B matrices
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___attn___lorahyphen___to_q\.lora_B\.(.*)$": 
            r"blocks.\1.self_attn.to_q.lora_B",
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___attn___lorahyphen___to_k\.lora_B\.(.*)$": 
            r"blocks.\1.self_attn.to_k.lora_B",
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___attn___lorahyphen___to_v\.lora_B\.(.*)$": 
            r"blocks.\1.self_attn.to_v.lora_B",
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___attn___lorahyphen___proj\.lora_B\.(.*)$": 
            r"blocks.\1.self_attn.to_out.lora_B",
        
        # Cross-attention Q/K/V
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___cross_attn___lorahyphen___q_linear\.lora_A\.(.*)$": 
            r"blocks.\1.cross_attn.to_q.lora_A",
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___cross_attn___lorahyphen___kv_linear\.lora_A\.(.*)$": 
            r"blocks.\1.cross_attn.to_kv.lora_A",  # Will need special handling for K/V split
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___cross_attn___lorahyphen___proj\.lora_A\.(.*)$": 
            r"blocks.\1.cross_attn.to_out.lora_A",
            
        # Cross-attention B matrices
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___cross_attn___lorahyphen___q_linear\.lora_B\.(.*)$": 
            r"blocks.\1.cross_attn.to_q.lora_B",
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___cross_attn___lorahyphen___kv_linear\.lora_B\.(.*)$": 
            r"blocks.\1.cross_attn.to_kv.lora_B",
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___cross_attn___lorahyphen___proj\.lora_B\.(.*)$": 
            r"blocks.\1.cross_attn.to_out.lora_B",
        
        # FFN layers (w1=gate, w2=down, w3=up)
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___ffn___lorahyphen___w1\.lora_A\.(.*)$": 
            r"blocks.\1.ffn.w1.lora_A",
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___ffn___lorahyphen___w2\.lora_A\.(.*)$": 
            r"blocks.\1.ffn.w2.lora_A",
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___ffn___lorahyphen___w3\.lora_A\.(.*)$": 
            r"blocks.\1.ffn.w3.lora_A",
            
        # FFN B matrices
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___ffn___lorahyphen___w1\.lora_B\.(.*)$": 
            r"blocks.\1.ffn.w1.lora_B",
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___ffn___lorahyphen___w2\.lora_B\.(.*)$": 
            r"blocks.\1.ffn.w2.lora_B",
        r"^lora___lorahyphen___blocks___lorahyphen___(\d+)___lorahyphen___ffn___lorahyphen___w3\.lora_B\.(.*)$": 
            r"blocks.\1.ffn.w3.lora_B",
    })

exclude_lora_layers: list[str] = field(default_factory=lambda: ["embedder", "embed"])
```

**Note**: The exact LoRA naming pattern needs to be verified by inspecting the actual `.safetensors` files.

### Phase 2: Update Model Definition ✅

**File**: `fastvideo/models/dits/longcat.py`

Update class attributes:
```python
class LongCatTransformer3DModel(CachableDiT):
    # ... existing code ...
    
    # Add LoRA mapping (read from config)
    lora_param_names_mapping = {}  # Will be set from config
    
    def __init__(self, config: LongCatVideoConfig, hf_config: dict[str, Any]):
        super().__init__(config=config, hf_config=hf_config)
        # Set class-level mapping from config
        self.__class__.lora_param_names_mapping = config.arch_config.lora_param_names_mapping
        # ... rest of init ...
```

### Phase 3: Inspect LoRA Files

**Action**: Examine the actual LoRA safetensors files to understand naming convention

```python
from safetensors.torch import load_file

# Load and inspect LoRA files
cfg_lora = load_file("/path/to/cfg_step_lora.safetensors")
refinement_lora = load_file("/path/to/refinement_lora.safetensors")

# Print all keys to understand structure
print("CFG LoRA keys:")
for key in sorted(cfg_lora.keys())[:20]:
    print(f"  {key}: {cfg_lora[key].shape}")

print("\nRefinement LoRA keys:")
for key in sorted(refinement_lora.keys())[:20]:
    print(f"  {key}: {refinement_lora[key].shape}")
```

### Phase 4: Create LoRA-enabled Pipeline

**File**: `fastvideo/pipelines/basic/longcat/longcat_lora_pipeline.py` (new)

```python
from fastvideo.pipelines.lora_pipeline import LoRAPipeline
from fastvideo.pipelines.basic.longcat.longcat_pipeline import LongCatPipeline


class LongCatLoRAPipeline(LoRAPipeline, LongCatPipeline):
    """
    LongCat pipeline with LoRA support for distilled and refinement models.
    """
    
    def __init__(self, *args, **kwargs):
        # LoRAPipeline.__init__ will call convert_to_lora_layers if needed
        super().__init__(*args, **kwargs)
    
    # Inherit all methods from both parent classes
    # LoRAPipeline provides: set_lora_adapter, merge_lora_weights, etc.
    # LongCatPipeline provides: generate_t2v, etc.
```

### Phase 5: Register Pipeline

**File**: `fastvideo/pipelines/__init__.py`

Add import:
```python
from fastvideo.pipelines.basic.longcat.longcat_lora_pipeline import LongCatLoRAPipeline
```

**File**: `fastvideo/pipelines/pipeline_registry.py`

Register pipeline:
```python
"longcat_lora": {
    "class": LongCatLoRAPipeline,
    "config_cls": LongCatVideoConfig,
},
```

### Phase 6: Test LoRA Loading

**File**: `test_longcat_lora_inference.py` (new)

```python
import torch
from fastvideo import VideoGenerator
from safetensors.torch import load_file

def test_lora_loading():
    """Test loading LoRA weights without inference."""
    
    # Initialize generator
    generator = VideoGenerator.from_pretrained(
        "/path/to/longcat/weights",
        pipeline="longcat_lora",  # Use LoRA-enabled pipeline
        num_gpus=1,
        dit_cpu_offload=False,
        lora_path="/path/to/cfg_step_lora.safetensors",
        lora_nickname="cfg_distill"
    )
    
    print("✅ LoRA loaded successfully!")
    
    # Check which layers got LoRA
    pipeline = generator.pipeline
    print(f"LoRA layers count: {len(pipeline.lora_layers)}")
    for name, layer in list(pipeline.lora_layers.items())[:5]:
        print(f"  {name}: lora_A={layer.lora_A.shape}, lora_B={layer.lora_B.shape}")


def test_lora_switching():
    """Test switching between different LoRA adapters."""
    
    generator = VideoGenerator.from_pretrained(
        "/path/to/longcat/weights",
        pipeline="longcat_lora",
        num_gpus=1,
    )
    
    # Load first LoRA
    generator.set_lora_adapter(
        lora_nickname="cfg_distill",
        lora_path="/path/to/cfg_step_lora.safetensors"
    )
    print("✅ Loaded distill LoRA")
    
    # Switch to refinement LoRA
    generator.set_lora_adapter(
        lora_nickname="refinement",
        lora_path="/path/to/refinement_lora.safetensors"
    )
    print("✅ Switched to refinement LoRA")


if __name__ == "__main__":
    test_lora_loading()
    test_lora_switching()
```

### Phase 7: Inference with LoRA

**File**: `test_longcat_lora_generation.py` (new)

```python
from fastvideo import VideoGenerator

def test_distill_generation():
    """Test 16-step distilled generation."""
    
    generator = VideoGenerator.from_pretrained(
        "/path/to/longcat/weights",
        pipeline="longcat_lora",
        num_gpus=1,
        lora_path="/path/to/cfg_step_lora.safetensors",
        lora_nickname="cfg_distill"
    )
    
    prompt = "A cat playing piano"
    
    video = generator.generate_video(
        prompt,
        height=480,
        width=832,
        num_frames=93,
        num_inference_steps=16,  # Distilled uses 16 steps
        guidance_scale=1.0,       # Distilled uses CFG-free (guidance=1.0)
        output_path="./output_distill.mp4",
        save_video=True,
    )
    print("✅ Distilled generation complete!")


def test_refinement():
    """Test 480p→720p refinement."""
    
    generator = VideoGenerator.from_pretrained(
        "/path/to/longcat/weights",
        pipeline="longcat_lora",
        num_gpus=1,
        lora_path="/path/to/refinement_lora.safetensors",
        lora_nickname="refinement"
    )
    
    # First generate 480p video (without LoRA)
    generator.set_lora_adapter(lora_nickname=None)  # Disable LoRA
    stage1_video = generator.generate_video(
        prompt="A beautiful sunset",
        height=480,
        width=832,
        num_frames=93,
        num_inference_steps=50,
        guidance_scale=4.0,
    )
    
    # Then refine to 720p (with refinement LoRA)
    generator.set_lora_adapter(
        lora_nickname="refinement",
        lora_path="/path/to/refinement_lora.safetensors"
    )
    
    # Refinement uses stage1 video as conditioning
    # TODO: Implement i2v/refinement mode in pipeline
    refined_video = generator.generate_video_refine(
        prompt="A beautiful sunset",
        stage1_video=stage1_video,
        num_inference_steps=50,
        output_path="./output_refined_720p.mp4",
        save_video=True,
    )
    print("✅ Refinement complete!")


if __name__ == "__main__":
    test_distill_generation()
    # test_refinement()  # Enable after i2v mode is implemented
```

## Implementation Checklist

### Configuration
- [ ] Add `lora_param_names_mapping` to `LongCatVideoArchConfig`
- [ ] Set `exclude_lora_layers` appropriately
- [ ] Update model class to use config mapping

### Pipeline
- [ ] Create `LongCatLoRAPipeline` class
- [ ] Register in pipeline registry
- [ ] Test basic pipeline initialization

### LoRA Files Investigation
- [ ] Load and inspect `cfg_step_lora.safetensors`
- [ ] Load and inspect `refinement_lora.safetensors`
- [ ] Document actual key naming patterns
- [ ] Update `lora_param_names_mapping` based on findings

### Testing
- [ ] Test LoRA layer conversion (without loading weights)
- [ ] Test loading LoRA weights
- [ ] Test LoRA adapter switching
- [ ] Test distilled generation (16 steps, guidance=1.0)
- [ ] Test refinement generation (requires i2v mode)

### Integration
- [ ] Update documentation with LoRA usage examples
- [ ] Add LoRA examples to `examples/inference/lora/`
- [ ] Verify compatibility with distributed inference

## Key Differences: FastVideo vs LongCat Original LoRA

| Aspect | LongCat Original | FastVideo |
|--------|------------------|-----------|
| **Implementation** | Manual forward wrapping | BaseLayerWithLoRA wrapper class |
| **Activation** | enable_loras()/disable_all_loras() | set_lora_adapter()/merge |
| **Weight Merging** | Runtime addition | Permanent merge option |
| **Multi-LoRA** | Multiple LoRAs active simultaneously | One LoRA at a time, switchable |
| **Storage** | LoRA dict with network objects | Adapter state dicts |
| **Distributed** | Manual device/dtype management | DTensor support built-in |

## Notes

1. **Target Modules**: LongCat LoRAs likely only target attention layers (Q, K, V, O) and possibly FFN layers. Check actual files to confirm.

2. **QKV Fusion**: Original LongCat uses fused QKV projection. Need to handle if LoRA weights are also fused.

3. **Refinement Mode**: The refinement LoRA also needs the BSA (block sparse attention) feature, which is a separate implementation task.

4. **Testing Strategy**:
   - Start with distilled model (simpler, no BSA needed)
   - Verify output quality matches reference
   - Then add refinement support

5. **Performance**: FastVideo's weight merging can improve inference speed by permanently adding LoRA weights to base model.

## Next Steps

1. **Immediate**: Inspect actual LoRA safetensors files
2. **Phase 1-2**: Update config and model definition
3. **Phase 3-4**: Create pipeline and test loading
4. **Phase 5**: Test distilled generation
5. **Phase 6**: Implement refinement mode (requires i2v pipeline + BSA)








