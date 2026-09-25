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

Both FullAttn and the legacy sparse alias ship `WanDMDPipeline` manifests with
`expand_timesteps=true`. FastVideo disambiguates them with
`model_index_detectors` on the Wan definition before falling back to path/name
heuristics.

## Limitations

- No sparse (VSA) path for this inference config. The separate FullAttn-to-VSA
  LoRA training recipe uses the VSA-capable 5B training config for its student.
- No TI2V or image-to-video workload on the FullAttn config class.
