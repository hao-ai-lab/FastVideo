# pyright: reportAttributeAccessIssue=false
"""Modal deployment entrypoint for the Dreamverse B200 backend."""

import os
import subprocess

import modal

IMAGE = os.environ.get("DREAMVERSE_IMAGE")
if not IMAGE:
    raise RuntimeError(
        "DREAMVERSE_IMAGE is required. Set it to a published SHA-specific Dreamverse image, "
        "for example a dreamverse-backend-cuda12.9.1-sha-* tag or a "
        "dreamverse-ui-cuda12.9.1-sha-* tag if serving the static UI."
    )

image = modal.Image.from_registry(IMAGE)
image = image.env({
    "HF_HOME": "/root/.cache/huggingface",
    "FASTVIDEO_DREAMVERSE_HOME": "/var/lib/dreamverse",
    "FASTVIDEO_ENABLE_STARTUP_WARMUP": "0",
    "FASTVIDEO_GPU_COUNT": "1",
    "ENABLE_TORCH_COMPILE": "0",
    "STREAM_MODE": "av_fmp4",
})

app = modal.App("dreamverse-b200")

hf_cache = modal.Volume.from_name("dreamverse-hf-cache", create_if_missing=True)
dreamverse_state = modal.Volume.from_name("dreamverse-state", create_if_missing=True)


@app.function(
    image=image,
    gpu="B200",
    cpu=16,
    memory=65536,
    timeout=7200,
    startup_timeout=4800,
    max_containers=1,
    secrets=[modal.Secret.from_name("dreamverse-api-keys")],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/var/lib/dreamverse": dreamverse_state,
    },
)
@modal.web_server(8009, startup_timeout=4800)
def serve():
    subprocess.Popen(["dreamverse-server", "--host", "0.0.0.0", "--port", "8009"])
