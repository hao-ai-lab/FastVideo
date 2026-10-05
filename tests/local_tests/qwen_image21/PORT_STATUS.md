# Qwen-Image-2.1 port status

## Implemented

| Surface | Implementation | Validation so far |
| --- | --- | --- |
| DiT | Native 32-block single-stream transformer, segmented block-causal SDPA, RoPE, CPU/GPU prefix cache and safe host-allocation fallback | Released-weight BF16 production-loader parity PASS; all 32 small-grid blocks/final/cache exactly match |
| RGBA VAE | Native single-frame encode/decode, posterior mode, latent statistics, slicing and overlap tiling | Released-weight FP32 production-loader parity PASS; actual-size BF16 tiled encode/decode exactly match |
| Qwen3-VL | Native shared H3 graph, all 36 text layers, pre-final-norm tap, packed vision projections and image-isolated attention | Released-weight BF16 production-loader parity PASS; text/long text/1/2/10 images/padded batch exactly match |
| Pipeline | Separate input/text/latent/schedule/denoise/decode stages, strict loading, registry and presets | Exact upstream parity: 40-step plain I2I/editing and 4-step T2I/two-reference generation; functional I2I/editing visually confirmed |
| Inputs/outputs | Ordered 0–10 refs, edit/ref shortcuts, RGBA preservation, white vision compositing, four-channel PNG allocation | CPU alpha/layout checks; real PNG export gate written |
| API/CLI | `GenerationRequest`, legacy `generate_video`, two YAML presets and a Python example | Public Python generation PASS; CLI GPU smoke pending |
| Optional torch.compile | Existing Inductor compilation of the 32 DiT blocks; text/VAE remain eager | 40-step T2I PASS after a full warmup; 14.56 s core / 15.20 s including saving; compiled-versus-eager/upstream parity remains unverified |

The current CUDA results, exact model/reference revisions and persistent run
logs are recorded in [GPU_VALIDATION.md](GPU_VALIDATION.md). Component parity
does not replace end-to-end pipeline parity, output-quality inspection or
validation of the original 24 GB memory target.

The parity sweep was paused at the user's request after the T2I / two-reference
run. The subsequent requested non-layerwise benchmark has completed: a 40-step
1024 x 1024 plain I2I request took 42.44 s, with a sampled peak of 16.85 GiB
GPU use and exact pixel equality to the validated image. Text/VAE offload and
CPU prefix caches remained enabled. The confirmed runs used this editable fork
on `qwen-img-2.1` with the validation corrections; no further GPU runs
are scheduled.

The additional warm T2I benchmark used preloaded models, no whole/layerwise
DiT offload and CUDA prefix KV storage. Two 40-step 1024 x 1024 requests
averaged 15.38 s for encoder + DiT + decoder and 15.64 s including image save.
The four-step warmup matched the previously validated upstream baseline
exactly; forty-step outputs matched each other. Logs and numerical reports are
preserved separately in Git. Run images/tensors and the upstream source clone
were local to the original pod and are excluded from this archive.

## torch.compile benchmark (2026-10-05)

One full 1024 x 1024, 40-step T2I warmup was followed by one timed request
using the same prompt, seed 42 and model instance. With `backend=inductor`,
`fullgraph=False`, SDPA, resident DiT, CUDA KV cache, text/VAE CPU offload
and CPU pinning enabled, the measured times were:

- **14.56447 s** for encoder + DiT + decoder, including **13.15578 s** DiT.
- **15.19754 s** request wall time including PNG saving.
- **19.57909 s** warmup, reusing existing Inductor disk-cache artifacts.
- **58.29764 s** model initialization, excluded from both request timings.

The worker confirmed all 32 DiT blocks compiled, three captured graphs and
zero graph breaks; no new graphs were captured in the timed request. Warmup
and measured pixel tensors were exactly equal, and the red-teapot output was
visually checked. This is a successful compiled T2I run, not a matched
eager/upstream parity gate or a controlled speedup comparison.

An initial text-encoder pinning failure and a later shared-GPU OOM are retained
in the run logs. The latter completed warmup before failing the timed request
while another process held 13.88 GiB. The final retry on the empty GPU passed
with CPU pinning enabled. Compile was restored to **`enabled: false`** in the
T2I YAML, and the worker exited. The YAML change and compile text records are
committed separately and have not been pushed. Detailed timings and attempt logs are linked
from [GPU_VALIDATION.md](GPU_VALIDATION.md#torchcompile-t2i-benchmark-2026-10-05).

## Original Mac verification

The initial CPU suite contains native component tests and pipeline tests with
substituted components. Those results alone did not prove production imports,
released-weight parity, offload memory use or rendered-image quality.

Final local verification: **62 passed, 22 skipped** in the documented pytest
command. All 35 changed/new Python files compile; TOML/YAML parsing and
`git diff --check` pass. The skipped cases require the installed tokenizer,
pinned reference checkout, CUDA or released weights.

Changed-file formatting, lint, spelling and Markdown hooks were run through
pre-commit with the repository's exclusions. The mypy environment could not be
installed because the Mac ran out of disk space; its check remains pending.

No model weights were downloaded and no GPU was used during that Mac run. The source
reference is Diffusers commit `578c9b2c6636ab2424a0e56186268b83623656b2`;
small official source/config/index files were inspected. Temporary source-isolated
reference experiments also found exact tiny VAE results and matching DiT control
flow, with independent normalization dtype checks. They are supplemental evidence,
not the production parity gates.

## Remaining GPU gates

1. Run the outstanding 1024 pipeline cases: true CFG, caching disabled and ten refs.
2. Verify CLI PNG output, dedicated transparent generation, RGBA editing,
   masks and annotations. No tolerance has been widened in this session.
3. Measure peak memory with text/DiT/VAE offload and CPU KV caches on the planned
   24 GB GPU with >=96 GB host RAM. Validate interrupted-request cleanup there.
4. Run 2K T2I separately and assess forty-step two-reference output quality.
   Seed approved image/latent regression artifacts after
   review; none exist yet for this new model family.

Commands and environment variables are in [README.md](README.md).

## Validation issues

| ID | Status | Evidence / resolution |
| --- | --- | --- |
| I001 | Resolved | Existing environment now installs this fork editable; CUDA 13 PyTorch supports RTX 5090. |
| I002 | Resolved | Unsupported explicit preset removed from Python/YAML examples and the pipeline parity helper; typed parser checks pass. |
| I003 | Resolved | Encoder batching and text-only RoPE corrections produce exact upstream BF16 results for all six gate cases. |
| I004 | Resolved | Full-resolution I2I drift came from VAE singleton input strides; matching upstream preprocessing yields exact 4/40-step plain I2I and 40-step edit pixels/latents. Failed logs retained. |
| I005 | Resolved | NumPy scalar promotion caused tiny sigma drift; canonical math.exp preserves FP32 and makes 4/40-step schedules exact at both tested token counts. |

## Explicit first-release limits

One GPU, one prompt/image per request, one frame, unquantized weights. SP/TP,
LoRA/custom attention processors, prompt rewriting and HTTP serving are outside
this implementation. The separate 9B rewrite checkpoints are deferred.

Annotations and masks are ordered reference images interpreted from the prompt.
The pipeline has no hard inpainting preservation constraint. The initial memory
gate does not include 2K generation with ten references.
