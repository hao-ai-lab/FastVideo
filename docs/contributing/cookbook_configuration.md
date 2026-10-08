# Adding a model deployment

Each model has one manifest under `docs/cookbook/recipes/`. Its `deployments`
mapping lists the maintained ways to serve that model. Users select Model, then
Deployment. Adding a deployment for an existing runtime uses native
configuration and YAML metadata; it does not require model-specific JavaScript.

The current adapters support CUDA video REST, native FastH3 MLX REST and CUDA
WebSocket streaming. Their existence does not make every model/backend/workload
combination valid. The builder UI remains a temporary demo of the shared
configuration and command generation.

## 1. Choose a native baseline

Reuse a maintained file under `examples/serving/`, or add one for a new
supported setup. It must identify `generator.model_path`; all deployments in
one model manifest must reference the same model identity. Keep every explicit
setting needed for the launch in this native YAML, not duplicated in the
manifest. Hidden values remain in the downloaded file; omitted values remain
inherited.

Choose the runtime matching the file:

| Runtime | Baseline and available controls |
| --- | --- |
| `fastvideo-cuda-rest` | Native `ServeConfig`; reviewed typed public fields, no active streaming block |
| `fastvideo-mlx-rest` | Native `MLXServeConfig`; typed server and generator fields |
| `fastvideo-cuda-streaming` | Native `ServeConfig` with an active `streaming` block; reviewed host, port, frames and resolution controls |

MLX currently supports the FastH3 models listed in its native schema. Its
`default_request` is untyped: required values stay in the baseline and are not
exposed as controls by borrowing HTTP-schema definitions. Check each baseline's
comments for conversion prerequisites, environment settings and local paths.

The maintained examples cover distinct workflows:

| Model manifest | Native baseline | Workflow |
| --- | --- | --- |
| `fastwan21.yaml` | `openai_fastwan21_1_3b.yaml` | One-GPU CUDA T2V REST |
| `fasth3-8step.yaml` | `openai_fasth3_8step.yaml` and `mlx_fasth3_8step.yaml` | CUDA and MLX T2V REST |
| `wan21-i2v.yaml` | `openai_wan21_i2v_14b.yaml` | Two-GPU CUDA I2V REST, requires an image reference |
| `ltx2-distilled.yaml` | `streaming_demo.yaml` | CUDA WebSocket streaming |

These baselines live under `examples/serving/`. H3's registry includes I2V, but
this cookbook uses the maintained Wan baseline for its I2V example; absence of
an H3 I2V recipe is not a declaration that H3 cannot do I2V.

## 2. Create or extend the model manifest

Use a stable lowercase hyphenated filename such as `fastwan21.yaml`. Its stem
is the model key. Add deployments as lowercase hyphenated mapping keys:

```yaml
# docs/cookbook/recipes/fastwan21.yaml
title: FastWan2.1 1.3B
summary: Distilled text-to-video serving with sparse attention.
guide: docs/cookbook/guides/fastwan21-serving.md

deployments:
  cuda-rest:
    runtime: fastvideo-cuda-rest
    workload: t2v
    config: examples/serving/openai_fastwan21_1_3b.yaml
    env:
      FASTVIDEO_ATTENTION_BACKEND: VIDEO_SPARSE_ATTN
    controls:
      - server
      - generator.engine.offload.dit_layerwise
      - generator.engine.offload.text_encoder
      - generator.engine.offload.vae
      - generator.engine.compile.enabled
      - default_request.sampling.num_frames
      - default_request.sampling.height
      - default_request.sampling.width
      - default_request.sampling.fps
      - default_request.sampling.seed
```

The model requires `title` and nonempty `deployments`. Model-level `summary` and
`guide` are optional. Each deployment requires `runtime`, `workload`, `config`
and `controls`; optional keys are `label`, `summary`, `guide`, `env`,
`requirements` and `hardware`. Unknown keys are errors. Only `summary` and `guide` fall back from
the model. Controls, environment and requirements have no inheritance.

- `config`: repository-relative path to native serving YAML.
- `label`: optional deployment-picker text; defaults to the runtime label.
  Use it to distinguish two setups with the same runtime.
- `workload`: one demonstrated workload (`t2v` or `i2v`) accepted by that runtime;
  it selects the client guidance, not the model's full capabilities.
- `env`: optional mapping of environment names to strings; quote numeric values.
- `requirements`: optional list of plain-text setup notes or documentation URLs.
- `guide`: optional repository-relative Markdown path under `docs/`.

There is no separate model list. Adding the manifest publishes a model;
adding another deployment publishes an additional choice under that model.
For example, FastH3 8-Step V2 can use two real native baselines:

```yaml
# docs/cookbook/recipes/fasth3-8step.yaml
title: FastH3 8-Step V2
deployments:
  cuda-rest:
    runtime: fastvideo-cuda-rest
    workload: t2v
    config: examples/serving/openai_fasth3_8step.yaml
    controls:
      - server
      - generator.engine.compile.enabled
  mlx-rest:
    runtime: fastvideo-mlx-rest
    workload: t2v
    config: examples/serving/mlx_fasth3_8step.yaml
    requirements:
      - Prepare the MLX checkpoint with VSA and set your local model paths.
    controls:
      - server
      - generator.model_root
      - generator.mlx_checkpoint
      - generator.prompt_cache_dir
      - generator.vae_dtype
      - generator.vsa_sparsity
```

This demonstrates a supported H3 runtime, not a promised FastWan MLX route.
The adapter supplies the different launch command, installation guide and field
metadata. A genuinely new runtime needs one shared adapter and its tests before
contributors can select it in manifests.

For LTX2 streaming, use `runtime: fastvideo-cuda-streaming`, `workload: t2v`
and the native `streaming_demo.yaml`. Its baseline requires NVFP4-compatible
compute capability >= 10.0, `flashinfer-python`, FFmpeg with `libx264` and the
FastVideo `streaming` extra. Record those prerequisites under `requirements`.
Expose `server.host`, `server.port`, `default_request.sampling.num_frames`,
`height` and `width` (each with the full sampling path). Keep the remaining
streaming and generation settings in the baseline.

Preserving the baseline's warmup, prompt, safety and pool fields does not mean
the bare serving entrypoint activates the corresponding DreamVerse integrations.
`CEREBRAS_API_KEY` is not required for this bare launch. Use
`docs/design/server_contracts/streaming.md` as the deployment's protocol guide.
The generated client section is a health/liveness check, the WebSocket endpoint
and that guide, rather than a REST generation command or a standalone client.
The health response alone does not establish generation readiness or output.

## 3. Select meaningful controls

`controls` is an explicit ordered list; `[]` is allowed. There are no implicit
controls, `add` or `hide`. Types and declared constraints come from the selected
runtime's public definitions; shared frontend code supplies labels and widgets.

Select an exact leaf or a non-nullable declared object namespace. `server`
expands to four server fields, giving the FastWan example thirteen controls.
Descendants follow declaration order; selectors follow manifest order. Arrays
are leaves. Every child of a namespace must be editable: protected, opaque or
unsupported descendants fail generation rather than disappearing silently.
Unknown paths, duplicates and overlaps such as `server` plus `server.port` are
errors. Nullable objects and free-form maps cannot be expanded. Namespace
selection opts into future public fields beneath it; use exact leaves for a
stable control surface.

Do not expose coupled settings independently. FastWan's three steps and DMD
timesteps belong together, so the example does not select the whole sampling
group. GPU count may require parallelism changes. Schema acceptance alone does
not establish that an option works for every model or combination. Experimental
settings remain preserved without automatically becoming editable controls.

## 4. Optional hardware evidence and guide

Hardware records belong to a deployment. Reference exact IDs from the shared
`docs/cookbook/hardware.yaml` inventory; add missing hardware facts there once.
Only explicitly listed devices appear for that deployment. Omitting the map
shows no hardware rows and supplies no evidence. Example deployment fragment:

```yaml
hardware:
  nvidia-h100-sxm-80gb:
    status: unverified
```

Use `verified` only with an HTTPS `evidence_url` recording a successful serving
run of this baseline. Use `unsupported` with a concrete `reason`. Other listed
records are unverified. Unknown inventory IDs or record keys are errors. Rated
memory is a device fact, not a claim that this configuration fits.

Evidence must identify GPU SKU/count, native config, environment, code/software
versions, workload and successful server output. Parser tests or direct Python
inference do not establish REST or WebSocket serving verification. User edits are custom and
unverified; baseline evidence keeps its original scope. Hardware rows do not
select or tune the configuration.

An optional guide explains prerequisites, launch, client workflow and relevant
troubleshooting. Put shared background at model level and override `guide` for
a deployment when needed. MkDocs renders it normally; the catalog contains its
compiled link, not a Markdown body. The [FastWan guide](../cookbook/guides/fastwan21-serving.md)
and [H3 server/client guide](../cookbook/openai-api.md) illustrate useful content.

## 5. Generate, preview and check

Use the CPU documentation environment in the
[documentation setup guide](https://github.com/hao-ai-lab/FastVideo/blob/main/docs/README.md).
From the repository root:

```bash
python docs/cookbook_config.py
mkdocs serve
```

Open `/cookbook/config-builder/`. Model manifests are ordered by filename;
deployments retain their mapping order. The exporter writes a model index and
complete catalogs at `docs/assets/cookbook-config/recipes/<model-key>/<deployment-key>.json`.
These are gitignored build output. Regenerate after changing manifests, native
baselines, hardware metadata or public configuration definitions.

Check both selectors and each deployment's instructions. With no edits,
downloaded YAML must preserve the complete baseline, including hidden fields.
Change each exposed control and confirm unrelated values survive. Reset must
restore the selected deployment. Absent values show Inherited; explicit `false`,
`0` and allowed `null` must not disappear. Switching deployment must load its own
baseline, controls and launch instructions. Verify REST sample port/alias and
the I2V image reference, or the streaming health-check and WebSocket URLs as
appropriate. Check required environment, guide URLs and hardware-evidence scope.

Run the focused checks and project hooks:

```bash
python -m pytest tests/local_tests/test_cookbook_config_metadata.py tests/local_tests/test_cookbook_config_roundtrip.py
node --test tests/local_tests/test_cookbook_config.mjs
mkdocs build
python docs/cookbook_config.py --check-site site
pre-commit run --files docs/cookbook/recipes/fastwan21.yaml docs/cookbook/recipes/fasth3-8step.yaml
```

Include other edited manifests and any new baseline, guide or inventory file
in pre-commit. Add focused
parser/adapter round trips for unusual requirements and reviewed controls.
Generation validates configuration without loading weights or starting CUDA or
MLX inference or importing streaming execution modules; actual serving evidence
remains separate. If field metadata is
missing, improve the existing public declaration and shared exporter rather
than duplicating its type/range in the manifest. Runtime default or validation
changes still need their own compatibility review.

See the [design](../design/serving-cookbook.md) for the generated-data contract.
