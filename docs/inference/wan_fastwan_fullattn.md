# FastWan2.2 TI2V 5B FullAttn

`FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers` is a dense text-to-video variant of
the FastWan 5B checkpoint. It shares weights with the sparse TI2V alias
`FastVideo/FastWan2.2-TI2V-5B-Diffusers` but routes through dense attention only.

## Supported usage

- **Workload:** text-to-video (`t2v`) only. Image conditioning and TI2V are disabled.
- **Attention:** dense backends (`TORCH_SDPA`, `FLASH_ATTN`, `SAGE_ATTN`, ...).
  `VIDEO_SPARSE_ATTN` is rejected at load time.
- **Resolution:** 720P preset (`fastwan_5b_fullattn`).

## Example

```bash
python examples/inference/basic/basic_dmd.py \
  --model_path FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers \
  --attention_backend TORCH_SDPA
```

## Registry routing

Both FullAttn and the legacy sparse alias ship `WanDMDPipeline` manifests with
`expand_timesteps=true`. FastVideo disambiguates them with
`model_index_detectors` on the Wan definition before falling back to path/name
heuristics.

## Limitations

- No sparse (VSA) path for this config.
- No TI2V or image-to-video workload on the FullAttn config class.
- Teacher/critic loads during DMD distillation skip workload validation because
  they are built under a narrowed dense scope.
