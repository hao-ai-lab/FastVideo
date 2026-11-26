# Capacity estimation

Add an optional `capacity` section to an existing cookbook model manifest, reuse
shared component and hardware entries, then validate and preview the result.

## Walkthrough: FastWan

The [FastWan manifest](../cookbook/recipes/fastwan21.yaml) already declares its
[serving baseline](../../examples/serving/openai_fastwan21_1_3b.yaml) and launch
environment. Keep its title and deployments; append these capacity facts:

```yaml
capacity:
  memory_rules: wan
  components:
    transformer:
      ref: fastwan2.1-t2v-1.3b-vsa-dit
    text_encoder:
      ref: umt5-xxl-encoder
    vae:
      ref: wan2.1-vae-decoder
  reference_runs: []
```

`memory_rules` selects a calculator family that can serve several models; `wan`
alone does not establish coverage of FastWan's VSA execution. Reuse entries
whose architecture and loaded subset match, or add entries to these shared files:

| File | Contents |
| --- | --- |
| [hardware.yaml](../cookbook/capacity/hardware.yaml) | Specific hardware variants and nominal capacity |
| [transformers.yaml](../cookbook/capacity/components/transformers.yaml) | Transformer counts and geometry, including FastWan's VSA gate weights |
| [encoders.yaml](../cookbook/capacity/components/encoders.yaml) | Encoder counts and loaded subsets |
| [vaes.yaml](../cookbook/capacity/components/vaes.yaml) | VAE counts, loaded subsets, and geometry |

The catalog derives `id: fastwan21` from the manifest key, `name` from its title,
and `model_id` from the baseline's `generator.model_path`. Every deployment
becomes an example, preserving its ID, `defaults` configuration, and `env`.
Do not duplicate `id`, `name`, `model_id`, or `examples` inside `capacity`.
Models without `capacity` remain valid cookbook entries. Omit `reference_runs`
or leave it empty until you have successful-run evidence.

From the repository root, use the Node version in the
[cookbook contributor guide](cookbook_configuration.md):

```bash
npm ci --prefix docs
node docs/js/build-cookbook-capacity.mjs --check
node docs/js/build-cookbook-capacity.mjs --preview fastwan21
```

The check currently prints `Validated 1 capacity model(s) and 52 hardware variant(s).`
The preview prints the model with component references expanded and each
example's baseline under `base_config`. Inspect its identity, counts, geometry,
and evidence. Both commands validate deployments and capacity metadata without
writing files, downloading weights, or starting FastVideo; no GPU is needed.
Unknown fields, missing/wrong-type references, invalid counts, and unused shared
components fail validation.

`npm run build:catalog --prefix docs` exports `index.json`, `hardware.json`, and
`models/fastwan21.json` under `docs/assets/cookbook-config/capacity/`.
Commit authored sources; generated assets are build output and must not be hand-edited.

## Authoring reference

| Record | Required fields | Optional fields |
| --- | --- | --- |
| Model `capacity` | `memory_rules`, nonempty `components` | `sources`, `reference_runs` |
| `components.<role>` | `ref` | `alternatives` (other registered refs) |
| Shared component | `name`, `parameter_count`, `sources` | `block_count` + `block_parameter_count`, `loaded_subset`, `geometry`; VAE-only `decode_feature_multiplier` |
| Hardware variant | `name`, `vendor`, `kind`, `memory`, `backend`, `sources` | `aliases`, `family` + `variant`, `form_factor`, `gpu_count`, `host_memory`, `partitionable`, `partitions`, `supported_dtypes`, `notes` |

Roles are `transformer`, `transformer_2`, `text_encoder`, `text_encoder_2`,
`image_encoder`, `vae`, or `audio_vae`; refs must match the role's component type.
Alternatives are documented compatible choices; catalog resolution does not
select one or change the serving configuration.

Count the module actually loaded, such as an encoder-only text model or a
decoder-only VAE. Describe partial loading in `loaded_subset`, and count shared
or tied storage once. Supply both block fields together; block parameters must
not exceed the total. Count fields accept separators such as `1_489_821_760`,
normalized to numbers during the build; scalar geometry dimensions require ordinary integers.
Do not copy component counts into individual model manifests.

Geometry belongs to the shared component. For `memory_rules: wan`, transformer
geometry requires `hidden_size`, `ffn_size`, and `patch_size` (three dimensions);
VAE geometry requires `temporal_compression`, `spatial_compression`, and
`vae_decoder_channels`, including for alternatives. Shared definitions contain
no selected weight/activation dtype or offload settings: those depend on execution.
Quantized storage also needs accounting beyond a parameter count.

Optional activation geometry supplies additional facts for future memory rules:

For the initial encoder inventory, keep parameter/block counts, the loaded subset,
and sources. The shipped UMT5 entry omits detailed activation geometry. Add encoder
geometry when a memory rule needs to model those activations explicitly; it is
not required for the initial inventory. Encoder weights still contribute to
stage residency. A coarse activation allowance must be declared and calibrated
rather than treating missing geometry as zero activation memory.

| Component | Additional allowed fields | Meaning |
| --- | --- | --- |
| Transformer and encoder | `gated_ffn` | Boolean indicating a gated FFN; separate from attention gating |
| Encoder | `hidden_size`, `ffn_size`, `num_attention_heads`, `attention_head_dim` | Positive integer dimensions for encoder activations |
| VAE | `latent_channels`, `output_channels` | Positive integer channel counts for the compressed video and final decoded frames |

The geometry allowlist also includes transformer `num_attention_heads` and
`attention_head_dim`. `patch_size` is exactly three positive integers;
`gated_ffn` is a boolean. Optional fields can be omitted; unknown fields and
arbitrary nested geometry mappings are rejected.

The compression ratios determine the latent grid; `latent_channels: 16` sizes
the compressed video buffer, and `output_channels: 3` sizes the final RGB frames.
Channels multiply buffer storage, not the number of video tokens. Decoder width
scales the rough decode-memory estimate.

### Inputs for VAE working memory

VAE working memory can be a large part of the generation peak. Supply enough
information to account for all of its parts:

| Part | Inputs |
| --- | --- |
| Compressed video | Batch, video dimensions, compression ratios, and latent channels |
| Feature/cache allowance | Optional `decode_feature_multiplier`, decoder width, batch, spatial extent, and actual activation dtype |
| Other temporary features and workspace | Actual spatial/time chunk sizes, dtype, and a model-specific memory rule |
| Decoded frames and merge/cast buffers | Batch, final output channels, video dimensions, retained copies, dtype, and GPU/CPU placement |

Sum buffers that are alive together to estimate the decode peak, then add the
weights still on that GPU. An approximate feature/cache allowance alone does
not establish the whole working-memory peak; a reviewed rule must account for
the remaining allocations without double counting.

Larger frames increase image-sized buffers. More frames increase retained video
buffers, but a fixed history cache does not keep growing after it fills up. If
finished chunks move to the CPU and their GPU storage is released, GPU output
memory behaves differently. Do not infer buffer placement from an API option
such as `return_frames`.

### VAE decode feature multiplier

The optional VAE-only top-level field `decode_feature_multiplier` is a positive
finite number. The Wan2.1 entry supplies `48`, an **uncalibrated heuristic**
borrowed from the [calculator proposal in PR #1978](https://github.com/hao-ai-lab/FastVideo/pull/1978/changes):

```yaml
decode_feature_multiplier: 48
```

It represents an equivalent allowance for internal decoder feature/cache
buffers, not 48 RGB frames and not a value from the architecture configuration.
A future rule can use:

```text
feature_cache_bytes ≈ decode_feature_multiplier * batch * vae_decoder_channels
                    * height * width * activation_bytes
```

For batch 1, decoder width 96, output size 480 × 832, and two-byte BF16
activations, this gives about **3.43 GiB for the feature/cache allowance only**.
It is not a total decode or generation peak. Actual cache mode, dtype, spatial
and temporal tiling, and buffer lifetimes affect whether this approximation
applies; validate and calibrate it for the execution being modeled.

Decoded output and merge/cast buffers are separate. For example, PR #1978 uses
an additional term equivalent to:

```text
output_buffer_bytes ≈ batch * frames * output_channels * height * width
                    * (2 * activation_bytes + 4)
```

For RGB output, this assumes decoded frames are retained twice at decode dtype
and once as FP32. Retention, placement, and execution paths can differ; this is
not a universal guarantee. Resident weights and other workspace are additional
terms in the peak. The builder exports the authored multiplier without
calculating memory; omission remains omission, with no automatic `48` fallback.

### Execution and hardware facts

Tile sizes/strides, feature-cache choices, dtypes, and offload are execution
settings. Keep them in the native serving configuration, where applicable, rather
than component geometry. For example, `generator.pipeline.vae_tiling` is preserved
in an exported example's `base_config`; that does not prove it changes the
selected decoder path. A future estimator must resolve runtime defaults and
effective behavior before applying a tiling rule. VAE tiling changes the decode
chunks and their working memory, not the full transformer's video-token count.
Component geometry is not a
complete layer schedule, cache/workspace model, or GPU-fit calculation.

For example, Wan accepts these literal dotted keys under `experimental`:

```yaml
generator:
  pipeline:
    vae_tiling: true
    experimental:
      "vae_config.use_feature_cache": false
      "vae_config.tile_sample_min_height": 512
      "vae_config.tile_sample_min_width": 512
      "vae_config.tile_sample_stride_height": 384
      "vae_config.tile_sample_stride_width": 384
```

This selects spatial tiles of up to 512 by 512 output pixels, with a 384-pixel
stride and 128-pixel overlap. With compression 8, that is up to 64 by 64 latent
positions and a 48-position stride. Edge tiles can be smaller.

Disabling feature caching lets this Wan path reach spatial tiling; the default
cached path bypasses it. This example does not enable temporal tiling, so each
spatial tile still contains the full clip in time. The estimate must use that
actual extent and include retained tile outputs and merge buffers. Do not apply
a fixed memory discount from `vae_tiling: true`, or assume this mode uses less
memory than the default cached decoder. Other model families can use different
tiling controls. This setting flow was checked in code, not by GPU generation.

Hardware IDs describe variants such as H100 SXM 80 GB. The authored
`memory.nominal` preserves `GB` or `GiB`; the build exports normalized decimal
`memory_gb` (`1 GB = 1e9 bytes`), converted to GiB with
`memory_gb * 1e9 / 1024**3`. Use `memory.type: dedicated` for discrete VRAM or
`unified` for capacity shared with the CPU. Neither means measured free memory.
`supported_dtypes` describes hardware format capability, not FastVideo kernel
support; omission means unknown. `aliases` connect a variant to existing UI IDs
and search names.

Use checkpoint `model_index.json`/component `config.json` files, tensor headers
or indexes, and the FastVideo loader to establish composition, counts, and geometry.
An index's total bytes are insufficient to count mixed/quantized weights.
Prefer one pinned checkpoint-folder link covering config and weights, plus an
implementation link when it explains loading or architecture behavior. Keep
model-specific checkpoint references in the model manifest instead of repeating
them for each shared component. Similar model
names do not establish that components can be shared. Offline scripts or agents
may prepare candidate metadata for review; builds and browser loaders do not
run them, import model code, or download checkpoints.

## Hardware catalog

[`hardware.yaml`](../cookbook/capacity/hardware.yaml) is the authored lookup
catalog. It records device specifications, not a runtime support matrix: a
searchable variant can still be unsupported by a model, backend, or kernel.
Every entry requires a vendor source URL and the date it was checked. The
[coverage checklist](../cookbook/capacity/hardware-coverage.md) tracks what is
present, what is next, and what is deliberately excluded. The catalog is for
discovery and reuse, not a prerequisite for running a model: a missing variant
uses the custom-device path, and `unknown` means no estimate, not that FastVideo
cannot attempt the recipe.

```yaml
h100-sxm-80gb:
  name: NVIDIA H100 SXM 80 GB
  vendor: nvidia
  kind: gpu
  family: nvidia-h100
  variant: SXM 80 GB
  form_factor: sxm
  memory:
    type: dedicated
    nominal: { value: 80, unit: GB }
  backend: cuda
  partitionable: true
  partitions:
    - id: 1g.10gb
      label: 1g.10gb
      memory: { value: 10, unit: GB }
  supported_dtypes: [fp32, fp16, bf16, fp8]
  aliases: [h100]
  sources:
    - url: https://www.nvidia.com/en-us/data-center/h100/
      checked: 2026-10-19
```

`memory.type` is `dedicated` or `unified`. `memory.nominal` preserves the
vendor's value and unit (`GB` or `GiB`). Unified memory is shared system memory,
not VRAM; do not describe it as a GPU-only allocation. Add `memory.usable` plus a
`memory.policy` only when a documented policy derives an application budget.
Without that policy the estimator uses a labeled 75% application default
(`DEFAULT_UNIFIED_BUDGET_FRACTION`); `memoryBudgetGiB` overrides it. Nominal
unified memory is never treated as available VRAM, and the assumption is
reported in `memory.budgetSource` and the result notes.

`family` groups exact `variant` records, so board, form-factor, and capacity
differences stay separate entries. `partitions` list selectable MIG/vGPU-style
allocations; selecting one never silently inherits full-card capacity. `gpu_count`
describes the device record and does not authorize summing GPUs into one pool.
`host_memory` is host RAM for CPU offload, separate from dedicated VRAM and from
unified memory. `aliases` preserve old links and search names; canonical IDs stay
stable.

Custom devices use the same memory fields and must declare `backend` and
`supported_dtypes`. They are not catalog-verified:

```javascript
const custom = {
  id: "custom",
  name: "Custom accelerator",
  custom: true,
  memory: { type: "dedicated", nominal: { value: 48, unit: "GB" } },
  backend: "cuda",
  supported_dtypes: ["fp32", "fp16", "bf16", "fp8"],
};
const result = estimateMemory({
  config,
  workload,
  model,
  hardware: custom,
  memoryBudgetGiB: 40, // optional application cap, in GiB
  partitionId: "1g.10gb", // optional catalog partition
  allocationMemoryGb: 10, // optional provisioned allocation, in GB
  hostMemoryGb: 64, // optional host RAM for offload
});
```

An unknown backend or missing memory type/capacity returns `unknown`. A missing
format list is accepted for fp32/fp16/bf16 storage and called out as an
assumption; fp8 or NVFP4 storage still returns `unknown` because capability was
not cataloged. The estimate
reports `memory` (type, nominal, budget, partition) and `host` (required,
capacity, status) separately from `capacityGiB` and `status`.

The generated `hardware.json` is versioned:

```json
{ "version": 1, "devices": [{ "id": "h100-sxm-80gb", "...": "..." }] }
```

Values keep authored aliases, normalized decimal `memory_gb`, optional
`usable_memory_gb` / `host_memory_gb`, and normalized partition capacities.

## Optional successful-run evidence

HTTP/OpenAPI schema compatibility is a separate contract; see
[serving documentation](serving_documentation_contract.md). A capacity estimate
and a completed generation do not establish HTTP client compatibility.

Add records inline under the manifest's `capacity.reference_runs`. One completed
generation is useful; benchmarking every configuration is unnecessary. Follow the
[serving guide](../cookbook/openai-api.md) with the FastWan baseline and set the
client's `FASTVIDEO_MODEL=fastwan21-1.3b` (or your actual served alias). FastWan
requires `FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN`; enable
`FASTVIDEO_STAGE_LOGGING=1` when collecting stage memory. A health check or model
load alone is insufficient. Record the effective request, including inherited defaults.

| Field | Requirement |
| --- | --- |
| `id`, `date` | Unique run ID within the model; date as `YYYY-MM-DD` |
| `hardware`, `gpu_count` | Hardware catalog ID and actual device count |
| `hardware_partition`, `hardware_allocation_gb` | Optional MIG/vGPU partition ID and provisioned allocation in decimal GB; a full-card run does not apply to a partition |
| `outcome`, `execution` | `completed`; `first_request` or `warm_request` |
| `evidence_url` | Accessible HTTP(S) logs or benchmark evidence |
| `context.runtime`, `context.deployment` | Serving runtime ID and the cookbook deployment ID used for the run; the deployment must exist and use that runtime |
| `context.revisions` | `fastvideo` and `model`; optional `components` mapping by role |
| `context.config` | Complete effective serving configuration, inline; matching model and GPU count |
| `context.request` | Normalized request with `sampling.height`, `sampling.width`, and `sampling.num_frames` as positive integers |
| `context.env` | Complete launch environment snapshot mapped to strings; use `{}` when none was set |
| `memory` | Optional measurements described below |

Prefer immutable code/checkpoint revisions; pin separately replaced components.
Preserve prompt, sampling, conditioning, runtime overrides, and relevant environment
without credentials. An authored baseline is insufficient as an effective config
snapshot. Validation checks structure and identity, not completeness of runtime capture.

### Memory measurements

If present, `memory` requires a `metric`, matching `scope`, `unit: GiB`, and a
`per_gpu` entry for each rank `0` through `gpu_count - 1`, without duplicates.
`torch_peak_allocated` and `torch_peak_reserved` cover the PyTorch caching
allocator (`scope: allocator`) and exclude the CUDA context and other processes.
`device_used` covers whole-device usage (`scope: device`) and can support a fit
comparison; allocator peaks cannot establish device fit by themselves. Every entry requires
`request_peak`; `startup_peak` and named `stages` are optional. All peaks must be
positive finite numbers; stage peaks use the same metric and cannot exceed `request_peak`.

This fragment uses **hypothetical values to illustrate the format**, not Wan measurements:

```yaml
memory:
  metric: torch_peak_allocated
  scope: allocator
  unit: GiB
  per_gpu:
    - rank: 0
      request_peak: 8.4
      stages:
        encoding: 5.5
        denoising: 8.4
        decoding: 7.0
```

Stage logs report `peak_allocated_mb` and `peak_reserved_mb` in binary MiB;
divide by 1024 for GiB. The human-readable `reserved=` is current reservation,
not its peak. Allocated tensors, allocator reservation, and device-wide usage
are different metrics; device-wide evidence must distinguish other processes.

The stage wrapper resets peak counters after each stage. A final counter alone
cannot describe the entire request. Preserve stage peaks and verify complete-request
coverage on every rank before using their maximum as `request_peak`. Establish
startup/load boundaries separately so they do not contaminate request measurements.
If logs cannot support a request peak, omit `memory` and keep successful-run evidence.

Successful-run evidence applies to its recorded model, hardware, and context;
it does not verify every edited configuration. Calibration additionally requires
matching metric, scope, phases, and execution context. One point can calibrate an
existing scaling assumption, not establish accuracy across modes. Missing startup
evidence leaves startup coverage unknown. Keep observations intact when deriving coefficients.

## FastH3 coverage

No FastH3 checkpoint has a full capacity model yet. The editor shows a
checkpoint-size breakdown and device budget, with `status: unknown` for runtime
fit. It does not subtract CPU-offloaded weights or divide file sizes by GPU
count. Those operations need phase residency and per-rank placement models.

Replacement components without matching data are omitted from the known-size
subtotal and named as missing coverage. Host-only system RAM cannot establish
GPU fit. A custom GiB budget keeps its unit; unified memory uses the shared
normalizer's usable-memory policy instead of treating nominal RAM as VRAM.

| Profile | First stop | Why |
| --- | --- | --- |
| OmniRef | Workload | Runs image-to-video; the estimator models text-to-video only |
| 8-Step CUDA | Topology | Defaults to four GPUs and Ulysses sequence parallelism |
| Trim Spark | Attention | `VIDEO_SPARSE_ATTN_H3` is not a dense backend, so VSA workspace is unmodeled |
| 4-Step Spark | Execution | `h3_sequential_load` loads components on demand |
| Omni NVFP4 decoder | Precision | The decoder swap changes decoder precision |

To make a FastH3 estimate possible, in order:

1. Add capacity entries for each checkpoint: transformer parameter counts and
   geometry, the text encoder, the VAE, and the NVFP4 decoder variants.
2. Model an image-to-video request: reference latents and audio tokens change
   the working set, and OmniRef passes ordered image, video, and audio inputs.
3. Model per-rank sharding for four-GPU Ulysses, including replicated weights
   and per-rank sequence slices, so the estimate compares against one device.
4. Add a VSA-H3 workspace term, or a dense upper bound that is labeled as one.
5. Support quantized decoder precision instead of rejecting the configuration.
6. Record successful runs on the RTX 5090, RTX PRO 6000, and DGX Spark profiles
   so the analytical estimate can be calibrated rather than trusted blindly.

Until those land, the cookbook can report checkpoint data but cannot answer
whether an edited FastH3 request fits a GPU.

## Static consumers

[`cookbook-capacity.mjs`](../js/cookbook-capacity.mjs) exposes these async loaders,
also re-exported by the shared cookbook API/bundle:

| Function | Returns |
| --- | --- |
| `loadCapacityIndex(url, options?)` | Index with model summaries, `hardware_url`, and resolved source `url` |
| `loadCapacityModel(index, id, options?)` | Model with components mapped by role, examples, and `reference_runs` |
| `loadCapacityHardware(index, options?)` | `{version: 1, devices, url}` with stable IDs and aliases |

Start with the generated index URL, then pass the returned index to the other
loaders. Each selected component is expanded with its `ref` and geometry;
examples contain authored `base_config` and `env`. `base_config` includes hidden
baseline settings, but is not a fully runtime-resolved configuration.

Each call accepts `{fetcher, signal}`; options are not retained between calls.
Model/hardware paths resolve relative to the index's source URL (`response.url`,
or the requested URL resolved against browser location). Use an absolute index URL
outside a browser. Calls reject on fetch/HTTP/JSON failures, cancellation, invalid
required structure/identity, or unknown model IDs. There is no module-level cache or retry.

These loaders fetch static facts; the calculator must resolve execution settings
and check rule coverage before estimating memory or recommending hardware.
`assessCapacity()` combines the analytical estimate with applicable measured
evidence; `matchEvidence()` reports applicable, unverified, and rejected runs;
`recommendCapacity()` returns measured or analytical fits and keeps unknown
results explicit. No successful generation is treated as a minimum-VRAM
measurement, and allocator-only peaks never establish device fit.
