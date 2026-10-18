# Temporary capacity API contract for calculator integration

**Temporary review handoff.** This document describes the current static loaders
for authors integrating the calculator proposal in
[PR #1978](https://github.com/hao-ai-lab/FastVideo/pull/1978/changes).
It adds no calculator API. **TODO: delete this exact file,
`docs/cookbook/capacity-api-contract.review.md`, and its discoverability link in
`docs/contributing/capacity_metadata.md` when integration lands and essential
stable API behavior has moved into that contributor guide.** Keep the API
modules, authored metadata, and catalog generation.

The [contributor guide](../contributing/capacity_metadata.md) owns authoring and
recorded-run requirements. The implementation is
[`cookbook-capacity.mjs`](../js/cookbook-capacity.mjs); generated JSON is build
output, not a second source to edit.

## Load index → model → hardware

The capacity module is pure ESM with no DOM, inference/runtime, or estimator
imports. From the repository root in an ESM consumer:

```javascript
import {
  loadCapacityIndex,
  loadCapacityModel,
  loadCapacityHardware,
} from "./docs/js/cookbook-capacity.mjs";

const options = { signal: new AbortController().signal };
const index = await loadCapacityIndex(
  "https://haoailab.com/FastVideo/assets/cookbook-config/capacity/index.json",
  options,
);
const model = await loadCapacityModel(index, "fastwan21", options);
const hardware = await loadCapacityHardware(index, options);
const h100 = hardware["h100-sxm-80gb"];
```

These three exports are also re-exported by
[`cookbook-config.mjs`](../js/cookbook-config.mjs). For a browser using the shared
bundle, load `/FastVideo/assets/cookbook-config/cookbook-config.js`, then use
`FastVideoConfigCookbook.loadCapacityIndex`, `.loadCapacityModel`, and
`.loadCapacityHardware`. Adjust the site prefix when hosted elsewhere.
Generate assets before serving them; the loaders do not build missing files.
Use your preview site's origin while these assets are not yet published.

| Function | Promise result |
| --- | --- |
| `loadCapacityIndex(url, {fetcher, signal} = {})` | Index JSON with a resolved source `url` added |
| `loadCapacityModel(index, id, {fetcher, signal} = {})` | One self-contained model record |
| `loadCapacityHardware(index, {fetcher, signal} = {})` | Hardware mapping keyed by hardware ID |

`fetcher` defaults to `globalThis.fetch`; it is called as
`fetcher(url, {signal})`. A custom fetcher must return a response with `ok`,
`status` for errors, `json()`, and preferably `url`. Each call receives its own
options; options passed to the index call are not saved for later calls.

The index's `url` uses `response.url` when present, including after redirects.
Otherwise it resolves the requested URL against `globalThis.location?.href`.
Use an absolute request URL outside a browser when a custom response lacks
`url`. Model and hardware paths resolve relative to `index.url` with `new URL`.
Pass the returned index to later loaders; raw index JSON lacks this source URL.

Failures reject the promise. A non-OK response throws
`Could not load capacity metadata: HTTP <status>`; fetch and JSON errors
propagate. The signal is forwarded to the fetcher and checked after fetching
and after parsing JSON, preserving cancellation even with a custom fetcher.
Invalid shapes/identities also reject. There is no module-level cache, retry,
or fallback; ordinary fetch cache behavior still applies.
Browser validation checks the loader's required structure; full authoring
validation belongs to the catalog build.

## Returned data

The generated [index](../assets/cookbook-config/capacity/index.json) currently
contains one participating model:

```javascript
{
  hardware_url: "hardware.json",
  models: [{
    id: "fastwan21",
    name: "FastWan2.1 1.3B",
    model_id: "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
    catalog_url: "models/fastwan21.json",
  }],
  url: "https://haoailab.com/FastVideo/assets/cookbook-config/capacity/index.json",
}
```

`id` is the cookbook manifest key; `model_id` is the checkpoint identity.
Models whose manifests omit optional `capacity` remain valid cookbook entries
but do not appear in this index. An unknown `id` rejects with
`Unknown capacity model <id>` before fetching a model.

The [FastWan model](../assets/cookbook-config/capacity/models/fastwan21.json)
returns `id`, `name`, `model_id`, `memory_rules`, `components`, `examples`,
`reference_runs`, and optional model `sources`. `components` is a **mapping by
execution role**, not an array. Each selected shared component has already
been expanded, retaining its `ref`:

| Role | `ref` | `kind` | `parameter_count` |
| --- | --- | --- | ---: |
| `transformer` | `fastwan2.1-t2v-1.3b-vsa-dit` | `transformer` | 1,489,821,760 |
| `text_encoder` | `umt5-xxl-encoder` | `encoder` | 5,680,910,336 |
| `vae` | `wan2.1-vae-decoder` | `vae` | 73,296,659 |

Components also have `name`, `sources`, and optional `block_count`,
`block_parameter_count`, `loaded_subset`, and `geometry`. Counts are numbers
for the loaded subset. The transformer has 30 blocks with 1,464,046,080 block
parameters; its geometry is `{hidden_size: 1536, ffn_size: 8960,
patch_size: [1, 2, 2]}`. The VAE geometry is `{temporal_compression: 4,
spatial_compression: 8, vae_decoder_channels: 96}`. The text encoder has no
geometry object in this record. Optional `alternatives`, when authored, are
expanded component objects; the loader does not select one.

Each manifest deployment becomes an example. Its `defaults` YAML is resolved
into `base_config`; its launch environment becomes `env`. FastWan currently has:

```javascript
model.examples[0] // selected fields; base_config remains the nested native config
// {
//   id: "cuda-rest",
//   deployment_id: "fastwan21/cuda-rest",
//   config: "examples/serving/openai_fastwan21_1_3b.yaml",
//   runtime: "fastvideo-cuda-rest",
//   env: { FASTVIDEO_ATTENTION_BACKEND: "VIDEO_SPARSE_ATTN" },
//   base_config: { generator: {...}, server: {...}, default_request: {...} }
// }
```

The exported baseline includes `generator.engine.compile.enabled: true`,
`generator.pipeline.experimental.VSA_sparsity: 0.8`, and
`generator.pipeline.experimental.dmd_denoising_steps: [1000, 757, 522]`.
`base_config` preserves the authored baseline; it is **not a fully
runtime-resolved configuration** or a completed-generation measurement.

`reference_runs` is optional when authoring; the builder normalizes omission to
`[]`, and model loader results always require an array. FastWan currently has
`reference_runs: []`. Future records can carry completed-generation evidence,
execution context, and optional per-rank memory measurements. Use the
contributor guide's schema and applicability rules; absence means no recorded
evidence, not zero memory usage.

The [hardware catalog](../assets/cookbook-config/capacity/hardware.json) is a
mapping, for example:

```javascript
hardware["h100-sxm-80gb"]
// {
//   name: "NVIDIA H100 SXM 80 GB", ui_id: "h100",
//   memory_type: "dedicated", memory_gb: 80, backend: "cuda",
//   supported_dtypes: ["fp32", "fp16", "bf16", "fp8"],
//   sources: [...], notes: "..."
// }
```

`memory_gb` is nominal **decimal GB per device** (`1 GB = 1e9 bytes`), not free
memory. Convert to GiB using `memory_gb * 1e9 / 1024**3`. Recorded-run memory
uses `unit: "GiB"`; its `torch_peak_allocated`, `torch_peak_reserved`, and
`device_used` metrics have different scopes. `memory_type` is `dedicated` or
`unified`; unified capacity is shared with the CPU. Optional
`supported_dtypes` describes hardware format capability, **not FastVideo kernel
support**. Missing capability metadata means unknown.

## Illustrative bridge to PR #1978

The following is **consumer-side illustration, not an implemented/exported
adapter**. It matches the proposal's flat geometry/component array and hardware
shape at [revision 349cc0b](https://github.com/Ishxn20/FastVideo/blob/349cc0b184ffcb10027234bd86b53316a93f580b/docs/js/cookbook-memory.mjs#L168).
Do not pass this output to an estimator until its coverage has been checked.
`runtimePrecision` must be supplied from actual execution resolution; never
infer weight or activation precision from parameter counts or geometry.

```javascript
function illustrateInputs(model, device, runtimePrecision) {
  if (device.memory_type !== "dedicated")
    throw new Error("Unified memory requires an explicit calculator policy");
  const components = Object.entries(model.components).map(([role, component]) => {
    const precision = runtimePrecision[role];
    if (!precision?.dtype || !precision?.activation_dtype)
      throw new Error(`Resolve actual storage and activation precision for ${role}`);
    return {
      role,
      parameter_count: component.parameter_count,
      block_count: component.block_count,
      block_parameter_count: component.block_parameter_count,
      dtype: precision.dtype,
      activation_dtype: precision.activation_dtype,
    };
  });
  return {
    model: {
      memory_rules: model.memory_rules,
      geometry: { ...model.components.transformer.geometry, ...model.components.vae.geometry },
      components,
    },
    hardware: {
      name: device.name,
      type: "discrete_gpu",
      memory_per_device: { value: device.memory_gb, unit: "GB" },
      supported_dtypes: device.supported_dtypes,
    },
  };
}
```

This illustration is specific to the shown Wan composition. Quantized parameter
subsets, retained original weights, streaming/offload behavior, batching,
topology, cache/tiling modes, backend workspaces, and actual tensor lifetimes
require execution-specific accounting. A selected dtype alone does not cover
these allocations. The later calculator also needs effective configuration,
request/workload, and launch environment; this bridge does not resolve them.

`memory_rules: "wan"` identifies a rule family; it does not validate kernels or
prove that a particular estimator covers this variant. FastWan VSA has extra
gate weights beyond ordinary Wan, already included in the transformer count.
Its VSA workspace, compile buffers, and DMD execution schedule also need explicit
coverage. The initial ordinary-Wan rule must return unsupported/unknown for
uncovered FastWan execution; matching geometry is insufficient. No working
calculator integration, hardware recommendation, or fit guarantee is added here.
