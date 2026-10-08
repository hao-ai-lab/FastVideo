# Serve Wan2.1 I2V 14B

This deployment turns a source image and a text prompt into a video using
`Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`. Its
[native configuration](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/serving/openai_wan21_i2v_14b.yaml)
uses two CUDA GPUs. It does not record an exact tested GPU or memory requirement.

## Prepare and launch

Complete the [CUDA installation guide](../../getting_started/installation/gpu.md).
From an activated FastVideo checkout, launch the maintained baseline:

```bash
fastvideo serve --config examples/serving/openai_wan21_i2v_14b.yaml
```

To use edits from the builder, download its `config.yaml` and copy the generated
launch command instead. Hidden baseline settings, including `flow_shift` and
the negative prompt, remain in the downloaded file.

## Supply an image with each request

Wait for the server to load, then call it from another terminal. The example
below assumes the unchanged port and model alias. Replace the image path with
a real file accessible on the **server machine**:

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/videos/sync \
  -H 'Content-Type: application/json' \
  --data '{"model":"wan21-i2v-14b","prompt":"The camera slowly moves across the scene","input_reference":"/absolute/path/to/first-frame.png"}' \
  --output output.mp4
```

The image is required for every generation; the server does not supply a
default image. Omitting sampling settings uses the server's explicit request
defaults. Client-supplied sampling values take precedence.

After changing the port or served model name, use the builder's updated sample
request. For a remote server, replace the loopback address with its reachable
address or a forwarded local port. See the [shared client guide](../openai-api.md#connect-your-app)
for SDK examples and the [Wan guide](../wan.md) for family setup and troubleshooting.
