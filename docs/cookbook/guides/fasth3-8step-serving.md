# FastH3 8-Step V2 serving notes

This guide adds model-specific preparation to the cookbook's generated YAML,
launch command and sample request. Both deployments use
`FastVideo/FastVideo-FastH3-8-Step-V2`.

## CUDA REST

Complete the [CUDA installation](../../getting_started/installation/gpu.md)
and the [FastH3 setup](../openai-api.md#cuda). The maintained CUDA baseline uses
four GPUs; it does not record an exact GPU model or a memory-fit guarantee.

Download the generated `config.yaml` and run the launch command shown by the
builder. Its hidden VSA and pipeline settings remain in the downloaded file,
even though only selected fields have controls. Keep the baseline's nine sigma
points: these correspond to eight transformer forwards for this checkpoint.

## MLX REST

Complete the [MLX installation](../../getting_started/installation/mlx.md),
including `ffmpeg`, on an Apple Silicon Mac. From the FastVideo checkout,
prepare the checkpoint once:

```bash
hf download FastVideo/FastVideo-FastH3-8-Step-V2 --local-dir ./FastH3-8-Step-V2
python scripts/checkpoint_conversion/convert_minimax_h3_mlx.py \
  --model-root ./FastH3-8-Step-V2/transformer \
  --out ./FastH3-8-Step-V2-MLX --formats int8 --include-vsa
```

Set `generator.model_root` and `generator.mlx_checkpoint` in the builder to
your prepared directories. Paths are relative to the directory where you
launch the server. Download the YAML and copy the generated MLX launch command.

## Send a request

Keep the server running, then use the generated sample request in another
terminal. That request uses your selected port and served model name. The YAML
configures the server; it is not sent as the HTTP request body. A client sends
JSON containing the prompt and any request-specific overrides.

For a remote server, use its reachable address or a forwarded local port.
`0.0.0.0` is a bind address; a same-machine client uses `127.0.0.1`. See the
[shared client guide](../openai-api.md#connect-your-app) for SDK examples.
