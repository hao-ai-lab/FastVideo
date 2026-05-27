# FastVideo CLI Inference

The FastVideo CLI is config-first. Inference runs are driven by a nested JSON or
YAML config, with optional dotted-path overrides on the command line. The
contract matches training: use an explicit subcommand plus `--config`, then add
any dotted overrides you need.

## Basic Usage

```bash
fastvideo generate --config config.yaml
fastvideo serve --config serve.yaml
```

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
