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

# ``@modal.web_server`` invokes ``serve()`` directly and bypasses the image
# ENTRYPOINT (``docker/docker_entrypoint.sh``).  That entrypoint normally
# ``:?``-validates these secret entries and ``source``-s ``ffmpeg-env.sh``;
# neither runs on Modal, so we replicate both here.
_REQUIRED_SECRET_KEYS: tuple[str, ...] = ("CEREBRAS_API_KEY", "GROQ_API_KEY")

image = modal.Image.from_registry(IMAGE)
image = image.env({
    "DREAMVERSE_IMAGE": IMAGE,
    "HF_HOME": "/root/.cache/huggingface",
    "FASTVIDEO_DREAMVERSE_HOME": "/var/lib/dreamverse",
    "FASTVIDEO_ENABLE_STARTUP_WARMUP": "1",
    "FASTVIDEO_GPU_COUNT": "1",
    "ENABLE_TORCH_COMPILE": "1",
    "DREAMVERSE_MAX_AUTOTUNE": os.environ.get("DREAMVERSE_MAX_AUTOTUNE", "1"),
    "STREAM_MODE": "av_fmp4",
    # Native ffmpeg is built into ``/opt/ffmpeg-native`` by the Dockerfile but
    # is not on the image's ``PATH``.  Without these, ``av_streaming``'s
    # ``shutil.which("ffmpeg")`` returns ``None`` and ``av_fmp4`` muxing has
    # no encoder.  ``ffmpeg-env.sh`` would normally set them.
    "FASTVIDEO_FFMPEG_BIN": "/opt/ffmpeg-native/bin/ffmpeg",
    "FASTVIDEO_VIDEO_CODEC": "libx264",
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
    min_containers=1,
    max_containers=1,
    secrets=[modal.Secret.from_name("dreamverse-api-keys")],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/var/lib/dreamverse": dreamverse_state,
    },
)
@modal.web_server(8009, startup_timeout=4800)
def serve():
    # ``or ""`` collapses ``None`` (unset) into an empty string, ``.strip()``
    # collapses whitespace-only values (e.g. ``"   "``) — both should be
    # treated as missing.
    missing = [
        k for k in _REQUIRED_SECRET_KEYS
        if not (os.environ.get(k) or "").strip()
    ]
    if missing:
        raise RuntimeError(
            "dreamverse-api-keys secret is missing required entries: "
            f"{', '.join(missing)}. Add them with `modal secret create "
            "dreamverse-api-keys ... --force` and redeploy "
            "(see apps/dreamverse/scripts/modal/README.md).")
    subprocess.Popen(["dreamverse-server", "--host", "0.0.0.0", "--port", "8009"])
