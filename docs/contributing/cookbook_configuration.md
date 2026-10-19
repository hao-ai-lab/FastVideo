# Adding a model deployment

Add an already supported model/configuration to the cookbook so users can edit
settings and download a serving YAML. A **deployment** is one named combination
of a model, runtime and starting configuration, such as FastWan with CUDA REST.
Ordinary additions use YAML; they require no custom UI or JavaScript changes.

## Before you start

Choose a runnable serving configuration under `examples/serving/` and an existing
[runtime](#runtime-choices). This guide adds a cookbook entry; it assumes the
model and serving configuration already work in that runtime. If you need a new
baseline, follow the model's serving runbook first.

Run commands from the repository root. Local metadata checks need Node.js and
npm. Use Node 22.13+ in the 22.x line, or Node 24+; docs CI uses Node 22. Metadata checks do not load weights or require a GPU.

| File | What it controls | When to edit it |
| --- | --- | --- |
| `examples/serving/*.yaml` | Actual starting configuration, including hidden settings | Change recommended values or add a baseline |
| `docs/cookbook/recipes/<model-key>.yaml` | Deployment metadata and editable field selection; this file is the model manifest | Add or change a cookbook deployment |
| `docs/cookbook/options.yaml` | Shared field labels, types, constraints and runtime definitions | Add shared options or runtime definitions |

## Walkthrough: FastWan CUDA REST

Follow the existing `fastwan21/cuda-rest` deployment through the build and preview.
Use the same steps with your own model key and baseline when adding a new model.

### 1. Choose the baseline and model manifest

FastWan uses `examples/serving/openai_fastwan21_1_3b.yaml`. Its
`generator.model_path` identifies the model; its `server` and `default_request`
sections provide starting values. Here, `t2v` means text-to-video; `i2v` means
image-to-video.

If the model already has a manifest, add or edit a deployment in that file.
Otherwise, create `docs/cookbook/recipes/<model-key>.yaml`. Model keys and
deployment keys use lowercase letters/numbers separated by hyphens.
Each `generator.model_path` belongs to one manifest, and every deployment in
that manifest must use the same model path. Do not copy a model into a second
manifest just to add another runtime.

### 2. Describe the deployment and select fields

This example adapts the existing FastWan entry to hide one editor and constrain
the editable video height:

```yaml
# docs/cookbook/recipes/fastwan21.yaml
title: FastWan2.1 1.3B
deployments:
  cuda-rest:
    runtime: fastvideo-cuda-rest
    workload: t2v
    defaults: examples/serving/openai_fastwan21_1_3b.yaml
    env:
      FASTVIDEO_ATTENTION_BACKEND: VIDEO_SPARSE_ATTN
    options:
      - server.*
      - '!server.output_dir'
      - generator.engine.num_gpus
      - generator.engine.compile.enabled
      - default_request.sampling.height
    overrides:
      default_request.sampling.height:
        minimum: 16
        multipleOf: 16
```

The real manifest already exists and exposes additional sample fields. Keep
its existing entries when making an addition. `defaults` points to the full
baseline; `options` selects editors from the shared option catalog:

- An exact path, such as `generator.engine.num_gpus`, selects one field.
- `*` matches zero or more characters, including dots: `server.*` selects all
  declared server fields.
- A leading `!` removes matching fields: `'!server.output_dir'` removes that
  editor after `server.*` selected it. Quote leading `!` entries in YAML.

Entries run from top to bottom; see [option selectors](#option-selectors) for
the full rules. Excluding the output-directory editor preserves its baseline
value in downloaded YAML. Other hidden settings, including the DMD schedule,
also remain in the download.

The height `overrides` inherits the shared field's type and title, then requires
a value of at least 16 that is divisible by 16 (Wan VAE stride 8 times DiT patch size 2). It does not change the baseline
height of 480. Selecting the height in `options` makes it editable; an override
must refer to a field selected by the final `options` list.

| What you want to change | Where to change it |
| --- | --- |
| Starting value, such as the server port | Native baseline YAML |
| A field title or allowed range | Shared option schema or deployment `overrides` |
| Which fields users can edit | Deployment `options` |

**Values come from the baseline plus explicit user edits.** JSON Schema
`default` is an informational annotation; changing it does not change the
starting value. `overrides` changes field schemas, not baseline values.

### 3. Build, test and open the preview

```bash
npm ci --prefix docs
npm run build:catalog --prefix docs
npm run test:cookbook --prefix docs
npm run check:cookbook --prefix docs
pre-commit run --files docs/cookbook/recipes/fastwan21.yaml
```

Use your changed paths in the pre-commit command. The npm build/test commands
automatically generate the combined browser API/validator bundle first. A successful build writes
`docs/assets/cookbook-config/index.json` and, for this deployment,
`docs/assets/cookbook-config/recipes/fastwan21/cuda-rest.json`.

Inspect the merged deployment catalog as YAML in the terminal:

```bash
node docs/js/build-cookbook-config.mjs --preview fastwan21/cuda-rest
```

This prints metadata and field schemas as well as the serving configuration
under `base_config`. To save the catalog preview, append
`> /tmp/fastwan21-catalog-preview.yaml`. For another deployment, replace the ID
with `<model-key>/<deployment-key>`. Check that:

- Unchanged values match the baseline configuration; comments and formatting
  may differ.
- Hidden settings remain present in `base_config`.
- Selected fields, validation constraints, environment variables, and any
  setup notes or guide links appear as intended.

The preview does not start a model server. Metadata tests do not establish
server startup, native runtime validity or GPU compatibility.

Generated catalogs, the combined bundle and its license notices are ignored by
Git. Commit authored sources rather than generated output. For docs setup and
CI publication, see the [docs README](https://github.com/hao-ai-lab/FastVideo/blob/main/docs/README.md).

## Reference

### Manifest fields

| Level | Required | Optional |
| --- | --- | --- |
| Model | `title`, nonempty `deployments` | `summary`, `guide` |
| Deployment | `runtime`, `workload`, `defaults`, `options` | `label`, `summary`, `guide`, `env`, `requirements`, `overrides` |

Unknown keys fail validation. Paths are repository-relative: `defaults` must
reference a YAML file under `examples/serving/`; a non-null `guide` must reference
a Markdown file under `docs/`. `label` falls back to the runtime label. Only
`summary` and `guide` fall back from model to deployment; configuration, options
and environments are not shared between deployments.

`env` maps environment names to strings; quote numeric values. `requirements`
is a list of setup notes or documentation URLs. Copy necessary notes from the
baseline or existing runbook and identify the source in a YAML comment. Keep
hardware recommendations and support claims grounded in those sources.

A deployment `guide` replaces the model guide; omitting it inherits the model
guide, and `guide: null` suppresses it. For example:

```yaml
guide: docs/cookbook/openai-api.md
```

Reuse an existing runbook or omit the guide. Guide text describes the baseline;
generated instructions reflect user edits.

### Runtime choices

| Runtime ID | Example and baseline requirement |
| --- | --- |
| `fastvideo-cuda-rest` | FastWan T2V, FastH3 T2V, Wan2.1 I2V; `ServeConfig` with streaming omitted or null |
| `fastvideo-mlx-rest` | FastH3 T2V with prepared local checkpoints; `MLXServeConfig` |
| `fastvideo-cuda-streaming` | LTX2 Distilled; `ServeConfig` with a non-null `streaming` mapping |

Choose a `workload` allowed by the runtime's definition in `options.yaml`.
If the baseline declares `runtime`, it must match the backend; if it declares
`generator.pipeline.workload_type`, it must match the workload. An empty
`streaming: {}` counts as streaming for the builder.

To add another supported runtime for the same model, add a sibling deployment
with its own baseline and options. A new runtime needs metadata, schemas and
tests; a new launch protocol also needs JavaScript API support. See the
[design and API contract](../cookbook/design.md) for that boundary.

### Option selectors

`options` is an ordered list of declared paths or patterns; `[]` exposes no
fields. Selection starts empty and processes entries from top to bottom:

| Entry | Effect |
| --- | --- |
| `server.port` | Add one declared field |
| `server.*` | Add matching declared fields |
| `'!server.output_dir'` | Remove a field from the selection |
| `'!generator.engine.offload.*'` | Remove matching fields |

`*` matches zero or more characters, including dots. Each positive entry adds its matches in catalog declaration order. Fields
already selected retain their position; removing and later re-adding a field
moves it to the end. Selections never contain duplicates. Every entry must match a declared option,
even an exclusion that removes nothing. Bare `server`, `**` and `?` are unsupported.
Quote leading `!` entries because YAML otherwise treats them as tags.
Patterns match canonical option paths, not filesystem paths or schema properties.

Exclusions hide editors, not baseline values. Wildcards include future matching
options; use exact paths when you want a stable field selection. Selecting both
a parent field and its child fails validation.

Protected fields remain fixed in the baseline: `generator.model_path`,
`generator.pipeline.workload_type`, `generator.pipeline.preset`,
`generator.pipeline.preset_version`, `generator.pipeline.components.vae_weights`,
`default_request.output.save_video`, `default_request.output.return_frames`
and `default_request.output.output_path`.
Selecting a protected path, its parent or its child fails validation.
Streaming also protects `server.output_dir` and `server.served_model_name`;
when selecting `server.*`, exclude both, as in `ltx2-distilled.yaml`:

```yaml
options:
  - server.*
  - '!server.output_dir'
  - '!server.served_model_name'
```

### Shared options and model overrides

The root `options` map in `options.yaml` defines shared path-to-schema entries.
Each runtime's `options` map supplies overrides/additions, followed by the
deployment's `overrides`:

```text
common options -> runtime options -> deployment.overrides
```

Objects merge recursively; arrays and scalar values replace previous values.
Omitted schema keys inherit. Existing field types must remain unchanged, and
schemas must be self-contained: `$ref` is unsupported. Keep inherited bounds,
enum values and default annotations consistent with the resulting schema.
Tightening a bound may require changing an inherited annotation too. Present
editable baseline values must satisfy the merged schemas.

The walkthrough's height override demonstrates adding constraints while
retaining the shared type and title. For a model-local field, provide a complete
schema in `overrides` and select its path in `options`. Every deployment
override must survive the final selector list, including exclusions; otherwise
the build names the unused path and asks you to select it or remove the override.
Shared/runtime options may remain unselected. Invalid recipes fail before
export replaces catalogs or the index. Run the local build before pushing;
CI is a second check after the push.

### Choosing what to expose

Authors maintain schemas against native configuration definitions and supported
usage. The builder checks metadata consistency without importing Python schemas
or consulting the runtime registry. Schema acceptance does not prove runtime
support; native cross-field checks still apply at server startup.

Keep coupled settings fixed unless reviewed, such as FastWan's inference steps
and DMD schedule. Experimental or untyped fields need an authored schema before
they become controls. MLX sampling controls are declared only where their
conversion into `GenerationRequest` and the serving adapter have been checked.

Current samples keep HSDP fields hidden because FSDP is disabled. Wan and H3
retain TP for their supported text encoders; their DiTs do not thereby become
tensor-parallel. LTX2 uses SP and does not expose an ineffective TP editor.
The API checks GPU count and degree divisibility, preserving `-1` automatic
values instead of changing invalid edits. Common ceilings (8 GPUs, 512 frames,
4096 pixels per dimension, 120 FPS) are overridable editor policy, not capacity
guarantees. Put verified model geometry and seed constraints in `overrides`.

The current Wan sample dimensions use a 16-pixel grid. LTX2 uses 64 while its
baseline keeps refinement enabled. MLX exposes frames, dimensions and seed;
FPS, guidance and inference steps stay fixed by that serving route. MLX VSA
sparsity only has an effect when the baseline enables VSA. WebSocket recipes
can expose `streaming.session_timeout_seconds` and
`streaming.generation_segment_cap`; the timeout covers waiting for client
messages and pool acquisition, not a deadline for ongoing generation.

Combined geometry rules remain [future validation work](../cookbook/design.md#validation-boundaries-and-future-work).

### Check native declaration drift separately

In an existing FastVideo development environment, run:

```bash
python -m pytest --noconftest tests/local_tests/test_cookbook_native_contract.py -q
```

This check inspects the current authored runtime options and all discovered
recipes against native Python declarations. CUDA REST and streaming use
`ServeConfig`; MLX uses `MLXServeConfig`, with its opaque request-default map
checked against `GenerationRequest`. The known Wan experimental `flow_shift`
path maps explicitly to `PipelineConfig.flow_shift`.

The check detects missing paths and incompatible root types/nullability. It
allows narrower cookbook schemas and does not compare informational defaults,
numeric bounds, enum values or nested array-item schemas. It does not prove
that a model consumes every setting. This test is separate from Node docs CI
and needs the existing FastVideo environment; no model weights or GPU execution
are required. Ordinary recipe additions are discovered automatically.

### Inspect one merged deployment

Use the [build and preview walkthrough](#3-build-test-and-open-the-preview).
The direct `node docs/js/build-cookbook-config.mjs --preview MODEL/DEPLOYMENT`
command imports handwritten sources and needs only installed Node dependencies,
not generated browser assets. It validates every source recipe before selecting
one, so an invalid unrelated recipe can block the preview.

`--recipes-dir` and `--catalog` select custom inputs. `--root` supplies the
repository root for fixture/local work; baseline and guide paths resolve there.
`--output-dir` cannot be combined with `--preview`; redirect stdout to save a
catalog preview. Browser downloads contain the serving configuration, not the
catalog metadata.

For reusable UI methods and configuration guarantees, see the
[design and API contract](../cookbook/design.md). Streaming client instructions
follow the [WebSocket contract](../design/server_contracts/streaming.md); a health
response alone does not establish generation readiness.

## Memory estimates

`estimateMemory()` in `docs/js/cookbook-memory.mjs` estimates one request's peak
memory on one GPU from a resolved configuration. It is analytical: a result is
not a measurement and never marks a recipe as verified.

```javascript
const { config } = api.resolveConfig(catalog, edits);
const result = api.estimateMemory({ config, workload: catalog.workload, model, hardware });
```

Each phase (encoding, denoising, decoding) adds the weights resident on the GPU
to that phase's working memory. The largest phase plus headroom (10% and 1 GiB
by default) is compared with the device memory. `status` is `fits`, `exceeds`,
`unsupported` when the device lacks a weight type, or `unknown` with a `reason`.
The result also reports each phase's weights and working memory in GiB.

| Model field | Meaning |
| --- | --- |
| `memory_rules` | Working-memory rule; `wan` is the only rule |
| `geometry` | For `wan`: `temporal_compression`, `spatial_compression`, `patch_size`, `hidden_size`, `ffn_size` and `vae_decoder_channels` |
| `components[].role` | `text_encoder`, `transformer` or `vae` |
| `components[].parameter_count` | Parameters of the loaded subset |
| `components[].dtype` | Weight precision; optional `activation_dtype` sets the compute precision |
| `components[].block_count`, `block_parameter_count` | Repeated blocks and their total parameters, for streaming offload |
| `components[].streams_when_offloaded` | The text encoder streams its blocks through FSDP when offloaded |
| `components[].quantized_parameter_count` | Transformer parameters stored in the requested quantized format |

A hardware entry has `name`, `type: discrete_gpu`, `memory_per_device` with a
`value` and a `GiB` or `GB` unit, and `supported_dtypes`.

Offload follows the native runtime:

- Unset offload switches are on (`OffloadConfig` in `fastvideo/api/schema.py`).
  Layerwise DiT offload replaces whole-model DiT offload.
- An offloaded component leaves the GPU outside its own phase and is complete
  during it. Layerwise DiT offload and FSDP text-encoder offload instead keep
  the weights outside the blocks plus two blocks.
- Without FSDP, whole-model DiT offload leaves the transformer on the GPU.
- Packed NVFP4 weights are buffers, so layerwise offload keeps them resident.
- `lazy_module_load` defers only the components a pipeline lists in
  `_lazy_module_names`. Wan lists none, so the switch changes nothing.

Multiple GPUs, FSDP, unified memory, image-to-video, more than one video per
prompt, missing parameter counts, text-encoder quantization and loading peaks
return `unknown`. With VAE tiling enabled, a note says the
decode estimate assumes untiled decoding.

`WAN_RULE` holds the uncalibrated tensor counts. To calibrate them, run with
`FASTVIDEO_STAGE_LOGGING=1`: each stage then records `peak_allocated_mb` and
`peak_reserved_mb` and resets the peak counter. The request-level
`peak_memory_mb` is read after those resets, so it does not cover earlier
stages. Allocator figures exclude the CUDA context, which the headroom covers.

To add a model family, write a function that returns working-memory bytes per
phase, register it in `WORKING_MEMORY` with its required geometry fields, and
add cases to `docs/tests/cookbook-memory.test.mjs`. Run the tests with
`npm run test:cookbook --prefix docs`.

## Model pages in the cookbook

The Blume cookbook starts at `/cookbook` with model cards filtered by Video,
Image, Audio, or World. FastH3 appears first. Each model has its own page in a
sidebar family group.

`docs-site/lib/cookbook-pages.ts` derives the inventory from
`docs/assets/cookbook-recipes.json`. Recipes sharing a checkpoint become variants
on one model page. `transfer-docs.mjs` generates source-recipe pages and sidebar
entries. Preparation publishes checked-in YAML under `assets/cookbook-sources/`
and Python examples with source locations for editable literals under
`assets/cookbook-examples/`. Python examples are parsed with AST, never imported.
Explicit values in the maintained launch command take priority over argparse
defaults. The editor rewrites those literals and preserves required launch flags
and preparation commands. Controls inherited from a helper may use an explicitly
declared cookbook knob. Settings without a source binding stay hidden.

For a model with an editable serving catalog, add its display metadata to
`recipePages`, map the checkpoint in `backendModels`, and add an MDX overlay with
`<Cookbook modelId="your-model-key" />`. Use `mode: frame` so the island owns the
logo, heading, and YAML actions. Add the route to its sidebar family.
Presentation metadata must not define editable options or compatibility claims.

Editable pages organize catalog fields into a compact option matrix. The shared
API owns validation and YAML/command generation. Each model owns one page. Hardware backend buttons select declared
deployments, so unsupported NVIDIA, AMD, or MLX choices do not appear. Precision
and encoder controls appear only when the deployment declares them.

GPU targets are URL preferences for setup guidance. GPU count and parallelism
use backend fields. Automatic parallelism sets the supported degrees to `-1`;
it does not estimate memory. All configurations carry an unverified-recipe
label until hardware-fit and runtime validation are connected.

Old `/cookbook?deployment=<id>` links open the corresponding model page.

Provider labels follow the displayed logo. FastVideo-owned releases and converted
checkpoints using its mark display `FastVideo` as provider. FastH3 Trim replaces
the obsolete CompactH3 catalog entry. Its editable NVFP4 deployment uses the
released checkpoint and the maintained one-Spark inference configuration, with
a serving adaptation under `examples/serving/`. Do not restore the old local
assembly recipes or their four-forward sampling settings.

Every model uses the same header, run modes, command toolbar, hardware matrix,
settings, and reset controls. The top choices are Python SDK, CLI, and Serving,
limited to supported routes. Hardware backends display CUDA or MLX. Serving
appears only where a native server baseline exists.

The toolbar's Copy action writes the edited YAML or Python example with a quoted
heredoc, then runs the matching command. CUDA catalog pages use the same generator
settings for SDK, CLI, and serving. Inference uses `request`; serving uses `server`
and `default_request`. These commands assume the platform's installation guide
has already been followed. cURL and environment tools appear where the recipe
provides them. Do not invent CLI flags, serving support, or container commands.

Custom hardware names can be added through the matrix and persist in the page
URL. This is a device preference, not a memory lookup or compatibility result.
The component explorer is a design preview for weights, encoders, and decoders.
It does not alter serving YAML. Choosing an alternate component on an editable
recipe suppresses generated output until the baseline components are restored.
Connect those choices only after a matching backend recipe is defined.
