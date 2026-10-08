# Serve FastWan2.1 1.3B

This recipe serves `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` through FastVideo's
OpenAI-compatible video API on one CUDA GPU. Start from the
[native serving configuration](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/serving/openai_fastwan21_1_3b.yaml)
or choose **FastWan2.1 1.3B**, then its **CUDA REST** deployment in the
[serving configuration builder](../config-builder.md).

## Prepare and launch

Complete the [CUDA installation guide](../../getting_started/installation/gpu.md)
in a FastVideo clone and activate that environment. From the repository root,
start the native baseline:

```bash
FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN \
  fastvideo serve --config examples/serving/openai_fastwan21_1_3b.yaml
```

The baseline fixes the GPU count and parallelism to one, enables compilation,
and uses VSA sparse attention with its trained DMD schedule. Keep the VSA
environment variable on the launch command. The three inference steps and
the `[1000, 757, 522]` timestep schedule belong together; do not replace them
with a generic diffusion step count.

To adjust the recipe, use the builder, download `config.yaml`, and copy its
launch command. The builder preserves every explicit baseline setting,
including hidden VSA settings, DMD timesteps and the negative prompt. An
**Inherited** field stays omitted until edited; it does not display a resolved
runtime value. Reset restores this deployment's baseline. Frames, resolution,
offload and compilation can be adjusted without discarding its other settings.

## Send a request

Keep the server running and wait for model loading to finish. In another
terminal on the same machine, check readiness:

```bash
curl --fail-with-body http://127.0.0.1:8000/health
```

For the unchanged baseline, run the checked-in client with `curl` and `jq`
installed:

```bash
FASTVIDEO_BASE_URL=http://127.0.0.1:8000/v1 \
FASTVIDEO_MODEL=fastwan21-1.3b \
  bash examples/serving/clients/video.sh
```

The client submits a text prompt, polls the video job and downloads the MP4.
It leaves sampling values to the server defaults. Explicit sampling values in
a client request take precedence over those defaults.

After changing the server port or model alias, use the builder's current sample
request or update these client variables to match. `0.0.0.0` is a bind address;
the local client uses `127.0.0.1`. For a remote GPU, use the address reachable
from your client, such as a forwarded local port. See the
[server and client guide](../openai-api.md#connect-your-app) for port forwarding
and the shared Python and JavaScript clients; use this recipe's model alias.

If sparse attention is not selected, check that the launch includes
`FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN`. For memory or other Wan setup
issues, consult the [Wan guide](../wan.md#cookbook-troubleshooting) and
[inference configuration reference](../../inference/configuration.md).
