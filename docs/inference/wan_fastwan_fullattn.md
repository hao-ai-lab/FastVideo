# FastWan2.2 TI2V 5B FullAttn

`FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers` is a dense text-to-video variant of
the FastWan 5B checkpoint. It shares weights with the sparse TI2V alias
`FastVideo/FastWan2.2-TI2V-5B-Diffusers` but routes through dense attention only.

## Supported usage

- **Workload:** text-to-video (`t2v`) only. Image conditioning and TI2V are disabled.
- **Attention:** dense backends (`TORCH_SDPA`, `FLASH_ATTN`, `SAGE_ATTN`, ...).
  `VIDEO_SPARSE_ATTN` is rejected at load time.
- **Resolution:** 720P registry preset (`fast_wan_2_2_ti2v_5b`).

## Example

```bash
FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN fastvideo generate \
  --config scripts/inference/inference_wan_VSA_DMD_5B_720P.yaml
```

The existing config filename includes `VSA` for historical reasons; its model
path is FullAttn and the command above selects dense `FLASH_ATTN`. The
`basic_dmd.py` example is a separate 1.3B recipe and does not read 5B CLI
arguments.

## Registry routing

Both hub IDs, FullAttn and the legacy sparse alias, map to this dense config by
name. Any other path is matched by its manifest: `match_model_index` on the Wan
definition routes every `model_index.json` with
`_class_name: WanDMDPipeline` and `expand_timesteps: true` to the FullAttn
config, before the path/name heuristics run. The manifest check cannot tell
FullAttn from the sparse alias, because both ship the same manifest.

### Checkpoints trained with the FullAttn-to-VSA LoRA recipe

Checkpoints exported from that recipe copy the base model's `model_index.json`,
so they also route to the dense FullAttn config. It rejects
`VIDEO_SPARSE_ATTN` and image inputs. Loading the FullAttn base weights with the
recipe's LoRA and the VSA backend is rejected the same way.

To run such a checkpoint with VSA or TI2V, pass the sparse-capable 5B config
explicitly as a `PipelineConfig` object:

```python
from fastvideo import VideoGenerator
from fastvideo.models.wan.pipeline_config import FastWan2_2_TI2V_5B_Config

generator = VideoGenerator.from_pretrained(
    "path/to/exported_checkpoint",
    pipeline_config=FastWan2_2_TI2V_5B_Config(),
    attention_backend="VIDEO_SPARSE_ATTN",
)
```

A pipeline-config JSON path is not enough: FastVideo loads the JSON into the
config class that the path resolves to, which is still the FullAttn config.

## Limitations

- No sparse (VSA) path for this inference config. The separate FullAttn-to-VSA
  LoRA training recipe uses the VSA-capable 5B training config for its student.
- No TI2V or image-to-video workload on the FullAttn config class.
