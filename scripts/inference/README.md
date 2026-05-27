# Inference Configs

These files are nested inference configs for the config-first CLI.

Run them with:

```bash
fastvideo generate --config scripts/inference/<config>.yaml
```

Or use the helper wrapper:

```bash
bash scripts/inference/run.sh scripts/inference/<config>.yaml
```

Override config values with dotted paths:

```bash
fastvideo generate --config scripts/inference/<config>.yaml \
    --request.sampling.seed 42 \
    --request.prompt "A panda skiing at sunset"
```

The same overrides work through the wrapper:

```bash
bash scripts/inference/run.sh scripts/inference/<config>.yaml \
    --generator.engine.num_gpus 2 \
    --request.output.output_path outputs/custom_run
```

Some configs require an attention backend environment variable. When needed,
the file header shows the exact command to use.
