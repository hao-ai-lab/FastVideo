---
date: 2026-04-24
experiment: daVinci-MagiHuman inference port
category: porting
severity: critical
---

# Non-Diffusers checkpoints require full pipeline overrides

## What Happened
daVinci-MagiHuman ships raw safetensors shards with no `model_index.json` and no
Diffusers-style subdirectories (`transformer/`, `vae/`, `text_encoder/`).
FastVideo's loading stack assumes Diffusers format in at least three places:
1. `registry.get_model_info` — downloads/reads `model_index.json` to get `_class_name`
2. `ComposedPipelineBase._load_config` — calls `verify_model_config_and_directory` which requires `model_index.json`
3. `ComposedPipelineBase.load_modules` — iterates entries in `model_index.json` to load sub-components

All three fail for a raw safetensors checkpoint.

## Root Cause
FastVideo was built around Diffusers-format repos. Non-Diffusers checkpoints
(raw safetensors shards at repo root, no sub-component directories) have no
supported loading path without overrides.

## Fix / Workaround
Three changes are required:

### 1. Register with `pipeline_cls_name` so `get_model_info` skips model_index lookup
```python
register_configs(
    ...,
    pipeline_cls_name="MyModelPipeline",  # skips model_index.json download
)
```
And patch `get_model_info` in registry.py to check for `pipeline_cls_name` before
calling `verify_model_config_and_directory`:
```python
_pre = _get_config_info(model_path, raise_on_missing=False)
if _pre is not None and _pre.pipeline_cls_name is not None:
    pipeline_name = _pre.pipeline_cls_name
else:
    config = verify_model_config_and_directory(model_path)  # original path
    pipeline_name = config.get("_class_name")
```

### 2. Override `_load_config` in the pipeline to return a stub dict
```python
def _load_config(self, model_path):
    return {"_class_name": "MyModelPipeline", "_diffusers_version": "0.0.0"}
```

### 3. Override `load_modules` entirely to load each component manually
Load the DiT from safetensors directly (applying `param_names_mapping` via
`re.sub`), load text encoder via `AutoModel.from_pretrained`, load tokenizer
via `AutoTokenizer.from_pretrained`, instantiate scheduler directly. No
`PipelineComponentLoader` calls needed.

```python
def load_modules(self, fastvideo_args, loaded_modules=None):
    modules = {}
    modules["scheduler"] = FlowMatchEulerDiscreteScheduler(shift=5.0)
    modules["tokenizer"] = AutoTokenizer.from_pretrained(encoder_path, local_files_only=True)
    modules["text_encoder"] = AutoModel.from_pretrained(encoder_path, ...).cuda().eval()
    # Load DiT from shards with param remapping
    sd = {}
    for f in sorted(glob.glob(os.path.join(self.model_path, "*.safetensors"))):
        sd.update(safetensors_load_file(f, device="cpu"))
    for pattern, replacement in transformer.param_names_mapping.items():
        sd = {re.sub(pattern, replacement, k): v for k, v in sd.items()}
    transformer.load_state_dict(sd, strict=False)
    modules["transformer"] = transformer.cuda().eval()
    # VAE: load separately or stub
    modules["vae"] = ...
    return modules
```

Also set `_required_config_modules = []` to prevent the base class from
asserting that discovered modules match a list.

## Prevention
During Phase 0 recon, check the HF repo root for `model_index.json`. If absent:
- Flag the checkpoint as non-Diffusers format
- Plan to override `_load_config` and `load_modules` in the pipeline class
- Add `pipeline_cls_name` to the `register_configs` call
- Note that sub-component paths (VAE, text encoder) must be known separately
  since they won't be in `model_index.json`
