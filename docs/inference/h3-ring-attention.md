# MiniMax H3 Ring Attention

H3 dense inference supports pure Ring (`ring_size=sp_size`) and hybrid
Ring × Ulysses (`1 < ring_size < sp_size`). Select `FLASH_ATTN` explicitly.
The Ring kernel currently uses FlashAttention 2, even when the ordinary dense
backend would select another FlashAttention version.

## Usage

Two GPUs, pure Ring:

```bash
FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN python examples/inference/basic/basic_minimax_h3_t2v.py \
  --model-path /path/to/MiniMax-H3 --num-gpus 2 --ring-size 2 \
  --prompt "A red panda walking through a forest" \
  --height 256 --width 256 --num-frames 22 --steps 4 \
  --output outputs/h3-ring
```

Four GPUs, hybrid USP: use `--num-gpus 4 --ring-size 2`. For a pure Ulysses
reference use the same settings and seed with `--ring-size 1`. Four steps are
only a smoke test; use the checkpoint's normal schedule for quality comparisons.

The Python configuration equivalent is
`PipelineSelection(experimental={"ring_size": 2})`, together with
`ParallelismConfig(sp_size=4, tp_size=1)` for the hybrid case. The example sets
SP size to `--num-gpus`.

## Data flow and limits

- The text refiner shards the text embeddings, attends without RoPE, then
  gathers and removes padding before packing with video and audio.
- The caller's `text_indices`, `video_indices`, and `audio_indices` define
  the packed order. No fixed text/video/audio concatenation is assumed.
- Main blocks shard packed embeddings and the corresponding RoPE tables with
  the same trailing padding. H3 normalizes Q/K and rotates only their first
  96 of 128 channels locally. `freqs_cis=None` prevents a second rotation in
  distributed attention.
- Hybrid USP first assembles contiguous SP shards within each Ulysses subgroup.
  Ring sends equal-sized KV buffers and excludes invalid suffixes from each
  FlashAttention call. Empty chunks still participate in communication. Query
  output is padded with zeros before the inverse Ulysses exchange. The model's
  final all-gather removes padding before selecting video/audio outputs.
- Heads must divide evenly by `ulysses_size=sp_size/ring_size`; pure Ring
  imposes no SP head divisibility constraint. Ring size must divide SP size.
- Ring supports non-causal fp16/bf16 inference with equal Q/K/V head counts.
  Training, GQA, replicated Q/K/V, and non-FlashAttention backends are rejected.
  VSA-H3 with Ring is rejected; select dense `FLASH_ATTN` explicitly.
- Regional fullgraph compilation falls back to eager with an explanatory
  message. Ordinary compilation keeps the existing attention graph-break
  boundary; compiling Ring P2P into a full graph is unsupported.

## Validation

Run from an activated FastVideo Python environment:

```bash
pytest -q fastvideo/tests/attention/test_h3_ring_attention.py \
  fastvideo/tests/attention/test_ring_attention_config.py \
  fastvideo/tests/attention/test_ring_attention_rope.py \
  fastvideo/tests/distributed/test_ring_attention.py \
  fastvideo/tests/distributed/test_h3_ring_attention_parity.py \
  fastvideo/tests/transformers/test_minimax_h3_fusion_routing.py
```

The H3 distributed tests use small random models with real refiner and main
blocks, H3's 96/128 partial RoPE, interleaved video/audio indices, and packed
lengths 13 and 129. They compare against the same unsharded H3 model using full
FlashAttention. Text lengths 1 and 5 also cover empty refiner shards. Pure Ring
uses 3 heads on 2 GPUs; hybrid USP uses 6 heads on 4 GPUs, catching the old SP
head-divisibility restriction. GPU comparison tolerance is `atol=rtol=0.03`.
CPU simulated-transport tests poison padded KV values and check all Ring chunk
owners, empty chunks, and restoration of transport shapes.

Local validation on one NVIDIA GB10 (PyTorch 2.12.0+cu130): **65 passed,
5 skipped**. Four skips require 2/4 GPUs; the remaining skip is the optional
FA4 fusion test. This includes real FlashAttention padding checks, a single-GPU
H3 model smoke test, and a Dynamo graph-break smoke test (`backend="eager"`,
mocked Ring transport). It does not establish multi-GPU Inductor compatibility.
Changed-file pre-commit hooks and `git diff --check` passed.

Real-checkpoint multi-GPU output quality, peak memory, latency, and communication
cost remain unmeasured on the development host (one GB10, no cached H3 checkpoint).
For acceptance, run both topologies and the Ulysses reference with the same
checkpoint, prompt, seed, dimensions, and schedule; check finite latent outputs
and video quality, and report numerical differences. After warmup, record peak
CUDA memory and synchronized inference latency, and use a profiler trace to
measure NCCL P2P and all-to-all time separately. Do not infer performance from
the random-model parity tests.
