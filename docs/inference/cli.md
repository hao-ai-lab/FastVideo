# FastVideo CLI Inference

FastVideo uses a nested configuration with dotted-path command-line options.
`generate` requires a JSON or YAML config. `serve` accepts either a config file
with overrides or all settings directly on the command line.

## Basic Usage

```bash
fastvideo generate --config config.yaml
fastvideo serve --config serve.yaml
```

Serving without a config file uses the same typed settings and validation:

```bash
fastvideo serve --generator.model_path MODEL_ID --server.port 9000
```

<!-- TODO(cookbook-ui): Point the example link below to the final cookbook page
when the demo is removed. Keep the CLI-only serving documentation. -->

Supply the model's required engine, pipeline, and sampling options too. The
[serving cookbook example](../cookbook/serving-example.md) shows a complete
generated command for FastH3 V2. Only explicitly supplied request defaults are
pinned; omitted values continue to use the model's defaults.

## View All Arguments

```bash
fastvideo generate --help
```

The subcommands intentionally expose only `--config`. Any per-run CLI changes
must use dotted override paths such as:

- `--generator.engine.num_gpus 2`
- `--request.sampling.seed 42`
- `--server.port 9000`

## Using Config Files

```bash
fastvideo generate --config config.yaml
```

Config files can be JSON or YAML. Dotted CLI overrides take precedence over
config-file values.

Example `config.yaml`:

```yaml
generator:
  model_path: FastVideo/FastHunyuan-diffusers
  engine:
    num_gpus: 2
    parallelism:
      sp_size: 2
      tp_size: 1
request:
  prompt: A capybara lounging in a hammock
  sampling:
    num_frames: 45
    height: 720
    width: 1280
    num_inference_steps: 6
    seed: 1024
  output:
    output_path: outputs/
```

Notes:

- `generator` and `request` are the top-level keys for generation configs.
- `serve` configs use `generator`, `server`, and optional `default_request`.
- Prompt text files belong under `request.inputs.prompt_path`.

## Examples

Simple generation:

```bash
fastvideo generate --config config.yaml
```

Config + dotted override:

```bash
fastvideo generate --config config.yaml --request.prompt "A panda skiing at sunset"
```

Helper wrapper with positional config path:

```bash
bash scripts/inference/run.sh scripts/inference/inference_wan.yaml
```
