# Adding capacity metadata

Add reusable hardware and model metadata for inference memory estimates. This
guide covers YAML authoring, validation, static API access, and optional recorded
runs. The calculator, hardware recommendations, and UI consume this data in
later work; adding metadata does not establish that a configuration fits a GPU.

Run commands from the repository root with the Node version supported by the
[cookbook contributor guide](cookbook_configuration.md). Catalog validation and
export need no Python, GPU, model downloads, or network requests. Actual
generation is an optional, separate step in an installed FastVideo environment.

## Walkthrough: the existing FastWan deployment

FastWan already has a model manifest and a runnable serving baseline. Add its
optional `capacity` section to that manifest and reuse the deployment's
configuration and environment. The catalog derives the model key, title,
checkpoint identity, and example list from those existing sources.

### 1. Follow the source files

| Source | Responsibility |
| --- | --- |
| `docs/cookbook/capacity/hardware.yaml` | Shared, variant-specific hardware specifications |
| `docs/cookbook/capacity/components/transformers.yaml` | Shared transformer architectures and counts |
| `docs/cookbook/capacity/components/encoders.yaml` | Shared encoder architectures and counts |
| `docs/cookbook/capacity/components/vaes.yaml` | Shared VAE architectures and loaded subsets |
| `docs/cookbook/recipes/fastwan21.yaml` | Existing model manifest: deployments plus optional component references, memory-rule identity, and recorded runs |
| `examples/serving/openai_fastwan21_1_3b.yaml` | Existing runnable serving baseline, including VSA/DMD and compile settings |

A model composes entries from the shared component files. Reuse an existing
entry when the architecture and loaded subset match; add another entry to the
appropriate type file when they differ. Do not create one file per component
or copy shared counts into each model.

Append capacity facts to the existing manifest; keep its title and deployments:

```yaml
# docs/cookbook/recipes/fastwan21.yaml
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

Use the checked-in files for the complete component definitions and sources.
Models without `capacity` remain valid cookbook entries. Each participating
model inherits every deployment as an example: `defaults` supplies the baseline,
`env` supplies its launch environment, and `generator.model_path` supplies the
checkpoint identity. Do not repeat `id`, `name`, `model_id`, or `examples` in
`capacity`; unknown fields fail validation. The exported model ID is the
existing manifest key, `fastwan21`, and example deployment IDs remain unchanged.

The model has one `memory_rules` identity; component geometry stays with its
shared definition. FastWan's VSA transformer contains gate projections absent
from vanilla Wan: its pinned checkpoint has **1,489,821,760** parameters,
including **1,464,046,080** in 30 blocks. The encoder and VAE tensor names/shapes
match the pinned vanilla checkpoint, so those definitions are shared. Similar
model names alone are insufficient evidence for sharing a component.

### 2. Validate and inspect the resolved catalog

```bash
npm ci --prefix docs
npm run check:capacity --prefix docs
node docs/js/build-cookbook-capacity.mjs --preview fastwan21
```

The preview prints the model with component references expanded and each
deployment example's authored configuration under `base_config`. It does not write files
or start a server. Inspect the model identity, component counts and geometry,
baseline, and any recorded-run evidence. An authored `base_config` is not a
dump of all runtime-resolved defaults.

For validation alone, use `node docs/js/build-cookbook-capacity.mjs --check`.
The capacity builder first validates the existing deployment catalogs, then
resolves shared component references. Missing references, wrong field types,
invalid block/count relationships, and unused component entries fail validation
and CI. Leave `reference_runs` empty
when no successful run is recorded; invented measurements are not required to
make a contribution pass. Omitting `reference_runs` also resolves to an empty list.

### 3. Export static assets

```bash
node docs/js/build-cookbook-capacity.mjs
```

The exporter writes generated files under:

```text
docs/assets/cookbook-config/capacity/index.json
docs/assets/cookbook-config/capacity/hardware.json
docs/assets/cookbook-config/capacity/models/fastwan21.json
```

Commit the authored YAML and relevant source changes. Generated assets are
build output; do not hand-edit them. The static API below reads these files
without starting FastVideo or executing inference. The regular
`npm run build:catalog --prefix docs` also exports the capacity catalog.

## Metadata reference

### Hardware specifications

Hardware entries describe a particular capacity variant, such as H100 SXM
80 GB, rather than every device sharing a product-family name. `memory_gb`
means nominal **decimal GB per device**: `1 GB = 1e9 bytes`. It is not free
memory measured on a running machine. Convert to GiB with
`memory_gb * 1e9 / 1024**3` when comparing against GiB measurements.

Use `memory_type: dedicated` for discrete VRAM and `memory_type: unified` for
memory shared with the CPU. A unified-memory capacity is not a promise that
the entire amount is available to GPU inference. `backend` identifies the
platform; optional `ui_id` connects a specific variant to an existing UI choice.
Neither field creates a hardware recommendation or changes the current UI.

Keep vendor/specification links in `sources`. Optional `supported_dtypes`
describes cataloged hardware format capabilities, not support for every
FastVideo kernel. Missing capability metadata means unknown. There is no
hardware-wide verified flag or user-authored usable-memory fraction.

### Shared components and geometry

Counts describe the module actually loaded. An encoder-only text model or a
decoder-only VAE must not inherit the count of an entire encoder-decoder
checkpoint. Record total parameters, repeated-block count and parameter total
when relevant. Shared or tied storage needs deliberate accounting rather than
double counting names. Quantized storage needs additional subset/format
accounting in the later calculator; do not infer it from the total count alone.

Human-readable counts such as `1_418_996_800` are accepted in the designated
count fields and normalized to JavaScript numbers by the loader. This is not
a general conversion rule for arbitrary strings or geometry fields.

Store architecture geometry with its component: for example transformer patch,
hidden, and FFN dimensions, or VAE compression and decoder dimensions. Cache
structure belongs to the architecture/rule context. A count or geometry field
alone does not describe tensor lifetimes, attention workspace, or all cache
allocations. A future calculator must check that its family rule covers them.

Do not store the selected weight or activation dtype in the shared definition.
Storage precision, quantization, actual compute/retained-tensor precision, and
offload are resolved for each execution. A BF16 option need not make every
buffer BF16; backend casts or FP32 retained state can matter. Likewise, the
component inventory does not prescribe a streaming-offload implementation.

A component reference can list `alternatives` when other registered components
are relevant. Register and document only compatible choices. Resolving the
catalog does not select an alternative or rewrite a serving configuration.

### Finding the values

Start from the checkpoint's `model_index.json` and component `config.json`
files to identify composition and architecture dimensions. Tensor names,
shapes, and dtypes in checkpoint headers or an index can help count the loaded
subset. An index's total byte size is not generally a parameter count when
formats are mixed or weights are quantized.

Compare these sources with the FastVideo module/loader to determine which
parameters, buffers, and decoder/encoder pieces are loaded. Preserve source
links and revision information with the metadata. Review a new count or
geometry rule before sharing it across model variants.

An optional offline script, Python tool, or agent can prepare candidate
metadata. Its output still needs review. Such tools are contributor aids;
catalog builds and browser consumers do not run them, import model code, or
download checkpoints.

## Recording a successful run

One complete generation is a useful optional reference. There is no requirement
to benchmark every precision, offload choice, resolution, or GPU. A model with
no measurement can still supply metadata for a later family heuristic.

After following the [installation guide](../getting_started/installation.md),
start the existing FastWan baseline on your chosen hardware. Its deployment
requires the VSA attention backend:

```bash
FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN FASTVIDEO_STAGE_LOGGING=1 \
  fastvideo serve --config examples/serving/openai_fastwan21_1_3b.yaml \
  --server.host 127.0.0.1
```

In a second terminal, use the repository's local-server client:

```bash
python -m pip install openai==3.6.0
FASTVIDEO_MODEL=fastwan21-1.3b python examples/serving/clients/video.py
```

The client submits a prompt, waits for a completed job, and downloads its MP4.
It calls your FastVideo server, not an external generation service. The baseline
and client are runnable examples, not a measured hardware recommendation.
Use the actual served alias if you change it. A health check or successful model
load is not a completed-generation reference.

This client inherits sampling defaults from the server. Record the resulting
request dimensions and other effective values, not only the prompt payload.

Add the record inline under `capacity.reference_runs` in the existing model
manifest; a separate model or run file is not required. Required fields are:

| Field | Record |
| --- | --- |
| `id`, `date` | Unique run identity within the model and calendar date |
| `hardware`, `gpu_count` | Hardware catalog ID and actual device count |
| `outcome` | `completed`, backed by a successful generation |
| `execution` | `first_request` or `warm_request` |
| `evidence_url` | Accessible run logs or benchmark evidence |
| `context.revisions` | `fastvideo` and `model` revisions; optional component revision mapping |
| `context.config` | Complete actual effective serving configuration, inline |
| `context.request` | Actual normalized request, including its sampling dimensions |
| `context.env` | Optional environment settings that affected the run |

Prefer immutable commit/checkpoint revisions. Record component revisions when
the run replaces or separately pins a shared component. The effective config
must identify the same model and GPU count as the record. Preserve the prompt,
sampling, conditioning, and runtime overrides actually used. Include environment
backend selections; a copied baseline with omitted defaults is insufficient to
establish an exactly matching calibration context. Do not include credentials
or tokens in a published snapshot. The validator checks structure and identity;
it cannot prove that a contributor captured every runtime-resolved value.

### Optional memory measurements

`memory` is optional. If present, it declares one `metric`
(`torch_peak_allocated`, `torch_peak_reserved`, or `device_used`), `unit: GiB`,
and one `per_gpu` entry for every rank from zero through `gpu_count - 1`.
Each entry requires a positive `request_peak`; `startup_peak` and named
`stages` are optional. Stage values use the same metric and cannot exceed the
recorded request peak.

This fragment contains **hypothetical values solely to illustrate the format**;
it is not a Wan measurement or data to copy into an authored run:

```yaml
memory:
  metric: torch_peak_allocated
  unit: GiB
  per_gpu:
    - rank: 0
      request_peak: 8.4
      stages:
        encoding: 5.5
        denoising: 8.4
        decoding: 7.0
```

Stage logging records `peak_allocated_mb` and `peak_reserved_mb` in binary MiB;
divide those values by 1024 for this catalog's GiB fields. The human-readable
`reserved=` log is current reserved memory, not the reserved peak. Keep metrics
distinct: allocated tensors, allocator reservation, and device-wide usage do
not measure the same thing. Device-wide observations also need enough evidence
to distinguish this run from other processes.

The stage wrapper resets peak counters after each stage. Do not copy a final
request counter as the whole-request peak if earlier stages have already reset
it. Preserve the stage peaks and verify that measurement spans the complete
request on every rank; only then can their maximum represent that covered
request. Startup/load activity is separate and may contaminate a first stage
unless its measurement boundary is established. If the available logs cannot
support the required request peak, omit `memory` and keep the successful-run
evidence. No telemetry exporter is introduced by this catalog.

Successful-run evidence can support a future model-and-hardware tested badge.
It does not verify every edited configuration. Memory evidence is additionally
eligible for calibration only when its metric, scope, phases, and execution
context match the calculation. One point can calibrate an existing scaling
assumption; it cannot establish that assumption's accuracy across modes. Missing
startup evidence leaves startup coverage unknown. Keep observations intact
when deriving later rule coefficients.

## Static API and calculator boundary

During calculator integration, use the
[temporary API contract](../cookbook/capacity-api-contract.review.md) for exact
loader return shapes and the proposed input bridge. Remove that document and
this link when integration lands and essential stable behavior is recorded here.

The Node builder exports
`buildCapacityCatalog({root, catalogDir, recipesDir, catalogPath})`, with
repository defaults, returning `{hardware, models}`. `hardware` is keyed by
ID; `models` contains resolved model records. Each `components[role]` expands
the selected shared entry and retains its `ref`; geometry stays in that
component. `exportCapacityCatalog` writes the static files described above.

Browser callers use `docs/js/cookbook-capacity.mjs`, also exported through the
shared cookbook API/bundle:

```javascript
const api = FastVideoConfigCookbook;
const index = await api.loadCapacityIndex(
  "/FastVideo/assets/cookbook-config/capacity/index.json",
);
const model = await api.loadCapacityModel(index, "fastwan21");
const hardware = await api.loadCapacityHardware(index);
```

Each loader accepts an optional `{fetcher, signal}` argument for custom fetching
and cancellation. These methods fetch static metadata. Existing configuration
APIs remain unchanged, and no calculator or UI behavior is added here.

A later adapter can bridge the resolved catalog to the flat input used by the
calculator proposal in [PR #1978](https://github.com/hao-ai-lab/FastVideo/pull/1978).
For example, it can take transformer dimensions from
`model.components.transformer.geometry`, take compression/decoder dimensions
from `model.components.vae.geometry`, and attach **resolved execution dtypes**
to the component counts. It must also bind actual batching, backend, offload,
cache/tiling modes, topology, and evidence applicability. Flattening geometry
alone is not sufficient to establish coverage for every architecture. In
particular, the `wan` identity and shared dimensions do not by themselves cover
FastWan VSA workspaces, compile buffers, or its DMD execution schedule.

This Wan-only illustration shows the data bridge, not an exported helper.
`runtimePrecision` must come from a future runtime-aware resolver and contain
the actual storage `dtype` and `activation_dtype` for each role:

```javascript
const calculatorModel = {
  memory_rules: model.memory_rules,
  geometry: {
    ...model.components.transformer.geometry,
    ...model.components.vae.geometry,
  },
  components: Object.entries(model.components).map(([role, component]) => ({
    role,
    parameter_count: component.parameter_count,
    block_count: component.block_count,
    block_parameter_count: component.block_parameter_count,
    ...runtimePrecision[role],
  })),
};
```

Quantization and streaming-offload coverage need additional execution-specific
accounting; this structural example does not supply them.

The adapter must convert `memory_gb` to the calculator's declared units and
handle dedicated versus unified memory explicitly. A future UI can derive an
estimated used percentage from estimated bytes and hardware capacity; users
do not need to supply a usable-memory fraction. Headroom, compatibility checks,
GPU-count recommendations, and configuration changes belong to that later
calculator/integration layer.

This metadata work supports [feature #1971](https://github.com/hao-ai-lab/FastVideo/issues/1971)
and the calculator portion of [cookbook migration #1949](https://github.com/hao-ai-lab/FastVideo/issues/1949).
