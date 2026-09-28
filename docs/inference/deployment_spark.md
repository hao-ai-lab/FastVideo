# MiniMax H3 on NVIDIA DGX Spark (GB10)

This guide covers running MiniMax H3 / FastH3 inference on DGX Spark clusters
(GB10, 128 GB unified memory per node) and the **encoder split** mode that
makes 720p fit without host offload.

## Why a dedicated encoder node group

Measured resident sizes on GB10 (NVFP4 checkpoint, lazy load):

| component | size |
|---|---|
| text_encoder (Qwen3-VL) | 47.98 GiB |
| transformer (DiT) | 65.53 GiB |
| video VAE | 9.74 GiB |
| audio VAE | 0.58 GiB |

All three heavy groups resident at once is ~124 GiB, above the ~115 GiB usable
on a 128 GB GB10, so 720p (768x1344x124) OOMs on a single node no matter what
offload knobs are set. Sequential/lazy loading already avoids encoding and
denoising at the same time; the encoder split goes one step further and runs
them on **different nodes**:

- the first `N` nodes (world ranks `0..N-1`) load only the Qwen3-VL
  conditioner, run `input_preparation` + `conditioning`, and NCCL-broadcast
  `prompt_embeds` to everyone over the world group;
- the remaining nodes load only DiT + VAEs (never the text encoder) and form
  the single sequence-parallel group, `sp_size = num_gpus - encoder_workers`.

A denoise node then peaks around ~76 GiB plus activations, which leaves plenty
of headroom for 720p.

## Enabling

The feature is off by default and only valid with the Ray backend.

```bash
export FASTVIDEO_H3_ENCODER_SPLIT=1   # or FastVideoArgs.h3_encoder_split / --h3-encoder-split
export FASTVIDEO_H3_ENCODER_NODES=1   # nodes dedicated to the encoder; default 1
```

Cluster-side prerequisites (all nodes): the FastVideo venv, the model path,
and per-node fabric exports (`NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` /
`NCCL_IB_HCA` set before `ray start`, since the executor deliberately does not
overwrite these per-node values). Start one Ray head and join the workers as
usual; pass `--num-gpus <total nodes>` and `--execution-backend ray`.

## Choosing the encoder-node count

The denoise group runs sequence parallelism, so its size must divide the DiT
attention head count (56 in the released FastH3 checkpoints). Valid encoder
node counts are therefore constrained: with 8 nodes only `1, 4, 6, 7` are
legal (SP = 7, 4, 2, 1). This is validated before any weights load, with the
legal values listed in the error.

One request packs a single prompt presentation, so world rank 0 is the only
encoder that computes; extra encoder ranks mirror the broadcast collectives
and stay idle (reserved capacity, not a speedup for single-prompt requests).

## Verified run (8x GB10, NVFP4 checkpoint, TAEH3 preview decode)

Encoder node = rank 0, denoise SP = 7, 5 DMD steps:

| geometry | result | wall |
|---|---|---|
| 256x448x124 | ok | 29.2 s cold |
| 768x1344x124 (720p) | ok | 51.6 s cold / 36.9 s steady |

Without the split, the same 720p request reaches ~104 GiB while loading the
DiT after conditioning and OOMs inside safetensors `persistent_load`.

## Troubleshooting

- `attention heads (56) must be divisible by sequence parallel size`: the
  encoder-node count left an illegal denoise-group size; use one of the values
  in the message.
- Drivers hang with no stage events and node memory stuck mid-load: check the
  cluster is actually free (`ray status` should show `0.0/8.0 GPU` and no
  actors). A leftover job holding the placement group starves the new run, and
  a full VAE decode of 720p can die at the NVRM level (`dmesg | grep NVRM`)
  with no Python traceback - prefer `--video-decode-backend taeh3` at 720p on
  128 GB nodes.
- The `prompt_embeds` broadcast travels over the world NCCL group, not the Ray
  object store (whose 4 GB cap spills large tensors to disk).
