# Inference VRAM estimation and hardware recommendations

Status: proposal for discussion. This document describes a possible design;
the catalogs, estimator, and recommendation API below are not implemented.

Feature tracking: [#1971](https://github.com/hao-ai-lab/FastVideo/issues/1971).
This develops the calculator work in Step 3 of the
[cookbook migration plan, #1949](https://github.com/hao-ai-lab/FastVideo/issues/1949).
It can proceed independently of the initial cookbook migration. A design PR
does not complete the implementation tasks or provide verified-run evidence.

## Problem and intended result

Users can choose a model, component precision, offloading, VAE behavior, output
shape, and parallel strategy, but still need to determine which hardware and
how many GPUs to provision. Existing recipes and recorded runs cover particular
configurations; they do not answer this question for every edited workload.

The feature should estimate memory per device and suggest suitable combinations
of hardware type, GPU count, and supported parallel strategy. Begin with the
largest memory costs and clearly identified assumptions. Approximate sizing is
useful before a complete profiling database exists.

The first recommendation means estimated memory fit with compatible execution
settings. It does not establish the cheapest or fastest deployment. Latency,
throughput, and cloud prices need separate evidence and are outside the initial
scope. Training and optimizer-state memory are also outside this feature.

## Proposed data ownership

Use contributor-maintained YAML metadata with three reusable catalogs:

| Catalog | Contents | Reuse |
| --- | --- | --- |
| Hardware | Name, device type/backend, memory per device, relevant capabilities | Shared across all models |
| Components | Parameter counts, loaded subset, architecture identity, metadata source/revision | Shared encoders, VAEs, and transformers |
| Models | Component references, geometry, supported estimation strategies, working-memory rules | One composition per model variant |

Possible locations are `docs/cookbook/capacity/hardware.yaml`,
`docs/cookbook/capacity/components.yaml`, and
`docs/cookbook/capacity/models/*.yaml`. These paths and schemas are illustrative.
Reuse existing model/configuration metadata where practical instead of creating
another source of runtime defaults.

### Hardware catalog

Start with a small set of common devices. Record memory units explicitly and
distinguish discrete VRAM from unified system memory. Hardware variants with
different capacities need separate identities. The entry below is a fictional
device showing the proposed shape, not a hardware recommendation:

```yaml
schema_version: 1
hardware:
  example-cuda-24gib:
    name: Example 24 GiB GPU
    type: discrete_gpu
    backend: cuda
    memory_per_device:
      value: 24
      unit: GiB
    supported_dtypes: [fp32, fp16, bf16]
    source: "<vendor specification or device-memory report>"
```

Real entries should identify their evidence and distinguish nominal capacity
from the runtime's available memory. Kernel support, device architecture, and
multi-GPU interconnect requirements must also be checked before recommending a
candidate. A dtype list alone does not establish kernel compatibility.

### Shared component catalog

Parameter counts belong to specific component architectures and loaded subsets.
For example, an encoder-only text model and a decoder-only VAE should not use
the parameter count of a larger encoder-plus-decoder checkpoint.

```yaml
schema_version: 1
components:
  wan21-t2v-1-3b-dit:
    role: transformer
    parameter_count: 1418996800
  umt5-xxl-encoder:
    role: text_encoder
    parameter_count: 5680910336
    loaded_subset: encoder
  wan21-vae-decoder:
    role: vae
    parameter_count: 73296659
    loaded_subset: decoder_only_module
```

For actual catalog contributions, every entry also needs its architecture
identity, source URL, immutable checkpoint revision or code revision, and count
method. Counts can come from tensor metadata, architecture arithmetic, or a
metadata-only module construction. This need not require downloading weights
or executing inference. Count shared/tied storage once where applicable.

Precision and offload remain configuration choices. Parameter count times dtype
size gives the basic weight-storage term; quantization metadata, unquantized
layers, persistent buffers, and conversion copies need additional accounting.

### Model composition

A model references shared components and supplies the geometry and estimation
rules needed to interpret the request. Alternate components must have a
supported, compatible runtime recipe; choosing an arbitrary VAE is not enough.

```yaml
schema_version: 1
model_id: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
checkpoint_revision: "<immutable checkpoint revision>"
components:
  transformer:
    ref: wan21-t2v-1-3b-dit
    count: 1
  text_encoder:
    ref: umt5-xxl-encoder
    count: 1
  vae:
    ref: wan21-vae-decoder
    count: 1
geometry:
  temporal_compression: 4
  spatial_compression: 8
  patch_size: [1, 2, 2]
  hidden_size: 1536
  ffn_size: 8960
memory_rules: wan21_t2v_v1
estimation_coverage:
  strategies: [single_gpu]
```

Here `estimation_coverage` describes what the estimator understands, not every
strategy the runtime supports. A first implementation can use a few named,
versioned memory rules instead of a general expression language in YAML.

## Configuration to memory estimate

```text
Effective model/request settings + target hardware + catalog metadata
                         |
        Resolve supported component placement and parallel layout
                         |
          Estimate weights and working memory for each phase/rank
                         |
              Peak per device + explicit provisioning headroom
                         |
          Compare compatible hardware/count/strategy candidates
```

Use effective values after model presets and explicit overrides. Include actual
component dtypes and quantization, offload behavior, attention implementation,
VAE mode, resolution, frames, batch, and conditioning inputs. Target-device
policy matters: some hardware changes offload behavior automatically.

For serving, state the worker/replica count and number of simultaneously
executing requests. Queued HTTP requests are different from overlapping GPU
execution. A recommendation covers a specific request or an enforced workload
envelope; request defaults alone do not constrain later client overrides.

At each phase and device, estimate:

```text
memory = resident component weights and buffers
       + active intermediate tensors and retained state
       + kernel/communication workspace and runtime overhead

required per device = maximum phase memory + provisioning headroom
```

Start with input/text encoding, denoising, and VAE decoding/output. Report the
limiting phase. Track loading and compilation/warmup separately: if these peaks
are not covered, expose that limitation and keep startup fit unknown rather
than implying the running estimate proves that the deployment can start.

Offload changes when weights are resident; it does not eliminate a component's
own GPU execution requirement. Layerwise offload includes active and prefetched
blocks, other resident parameters, and transfer buffers. It also increases host
RAM demand. Return a host-memory estimate or an explicit unknown when relevant.

Keep storage and compute precision separate. For example, BF16 VAE decode
autocast does not necessarily change FP32 VAE weight storage or every retained
feature tensor. Ordinary inference reuses intermediate memory across layers and
denoising steps; do not multiply a single-block activation estimate by both.

## Worked example: Wan2.1 T2V 1.3B

Consider one GPU, one active request, 480 x 832 output, 81 frames, and batch 1.
Use BF16 DiT/text weights, FP32 VAE weights with BF16 decode, text/VAE offload,
FlashAttention, and no DiT layerwise offload, VAE tiling, or compilation.

| Component | Parameters | Weight memory |
| --- | --- | --- |
| DiT, BF16 | 1,418,996,800 | 2.64 GiB |
| UMT5 text encoder, BF16 | 5,680,910,336 | 10.58 GiB |
| Loaded VAE decoder model, FP32 | 73,296,659 | 0.27 GiB |

The DiT and text counts follow the
[DiT tensor index](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/blob/main/transformer/diffusion_pytorch_model.safetensors.index.json)
and [text tensor index](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/blob/main/text_encoder/model.safetensors.index.json).
The decoder count comes from constructing the FastVideo decoder-only module on
the metadata device. These are example counts; production catalog entries must
pin the checkpoint and implementation revisions used to obtain them.

```text
weight GiB = parameter count x bytes per parameter / 1024^3

latent frames = (81 - 1) / 4 + 1 = 21
latent grid   = 21 x 60 x 104
DiT tokens    = 21 x (60 / 2) x (104 / 2) = 32,760

one BF16 hidden tensor = 32,760 x 1,536 x 2 bytes = 0.094 GiB
one BF16 FFN tensor    = 32,760 x 8,960 x 2 bytes = 0.547 GiB
```

For illustration, use provisional working-memory allowances of 0.5-1 GiB for
text encoding, 2-4 GiB for denoising, and 5-7 GiB for VAE decoding/output. The
last term includes retained VAE feature caches, active features, and accumulated
output; the decoder's small weight count does not describe those costs.
These allowances are uncalibrated design examples, not measured peaks or upper
bounds. The text allowance assumes one capped sequence of at most 512 tokens.
They explain the arithmetic rather than define initial estimator defaults.
Implemented rules must state their shape, precision, and offload coverage and
how their terms scale; settings outside that coverage return unknown.

Assuming the DiT stays resident and text/VAE weights are present only during
their stages:

| Phase | Resident weights | Working allowance | Phase estimate |
| --- | --- | --- | --- |
| Text encoding | 2.64 + 10.58 GiB | 0.5-1 GiB | 13.7-14.2 GiB |
| Denoising | 2.64 GiB | 2-4 GiB | 4.6-6.6 GiB |
| VAE decode/output | 2.64 + 0.27 GiB | 5-7 GiB | 7.9-9.9 GiB |

An illustrative additional 2 GiB runtime/headroom allowance gives a rough
16-17 GiB running budget. This is not a measured hardware recommendation;
startup fit remains unknown in this example. Doubling text weight storage to
FP32 adds about 10.58 GiB to its phase. Reducing frames/resolution lowers the
denoising and decoding terms but may leave text encoding as the peak.

## Hardware and GPU-count recommendations

Enumerate compatible `(hardware, GPU count, parallel strategy)` candidates and
run the same estimator for each layout. Keep requested workload and component
choices fixed. Present changes to offloading, precision, or strategy as explicit
alternatives that the user can choose.

For each candidate, report per-device peak/range, headroom, limiting phase,
assumptions, and estimate coverage. Compare against usable per-device memory,
not the sum of GPU capacities. Return possible fits and explain candidates that
exceed memory; unsupported or incomplete calculations remain unknown.

Strategy rules must describe actual tensor ownership. Sequence parallelism may
leave weights replicated; tensor parallelism shards only supported tensors;
FSDP needs temporary gathered weights and communication buffers. Multiple model
replicas duplicate weights, while concurrent requests can share weights and add
overlapping working state. VAE/encoder placement can make ranks asymmetric.
Consequently, adding GPUs must trigger a layout-specific calculation rather
than dividing a single-GPU estimate by the GPU count.

## Cookbook integration and contributor workflow

The existing [cookbook configuration API](../cookbook/design.md) and
[deployment manifests](../contributing/cookbook_configuration.md) remain the
authoring path for supported configurations. A future capacity catalog can join
by model identity or use an explicit reference after its schema is agreed.
The YAML examples above are not currently accepted recipe/runtime fields.

Keep the published calculator usable from static catalogs, with no model weight
downloads or live GPU required for website visitors. Share the estimation rules
across UIs and any future CLI/API adapter. How those rules are packaged remains
an implementation decision; maintain a single behavioral contract.

Contributors adding coverage should identify hardware evidence, register or
reuse component metadata, reference the model composition, and declare which
settings/ranges the memory rules support. Validate references, positive counts,
units, component compatibility, and provenance. A shared component update must
identify the affected model variants.

## Initial scope, validation, and open decisions

Start with a small hardware catalog and the single-GPU Wan example above, one
worker and one active request. Cover known precision/offload choices and output
geometry with weights plus major working-memory terms. Return GPU-count
suggestions only for implemented strategy rules; initially that means supported
single-GPU candidates, with multi-GPU results explicitly unknown.

Then add supported distributed layouts and more model families through reusable
rules. Improve the uncertain allowances with recorded loading, first-request,
warm-request, and mixed-shape measurements on every rank. Track both
underestimation and unnecessary overprovisioning on held-out configurations.

Label results as analytical, calibrated, measured for a matching configuration,
or unknown. Keep measured records separate from estimates and include exact
hardware, request, configuration, revisions, and test date. Neither estimates
nor this proposal qualify for the verified-run badge described in #1949.

Decisions for implementation review:

- Final catalog location and schemas, including reuse of existing model metadata.
- Named family rules and the initial supported settings/ranges.
- Default headroom policy and how unavailable startup/host-memory estimates appear.
- Placement in the cookbook UI and the shared result format.
- Priority of distributed strategies and requirements for calibrated coverage.
