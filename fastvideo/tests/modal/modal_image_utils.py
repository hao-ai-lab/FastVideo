"""Shared helpers for the Modal launcher scripts in this directory.

Standalone module: the launchers import it by file path (sys.path trick), not
through the ``fastvideo`` package, so they keep working on thin CI hosts that
have ``modal`` but not torch.
"""
import json
import os
import urllib.request

_REGISTRY = "ghcr.io"
_ACCEPT_MANIFEST_TYPES = ", ".join([
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
])


def resolve_image_ref(image_ref: str) -> str:
    """Pin a mutable ghcr.io tag to its current digest (``tag@sha256:...``).

    Modal caches registry images by reference string and never re-pulls a
    moved tag, so a mutable tag like ``py3.12-latest`` silently serves stale
    bytes after a new image is published. Resolving the tag to its digest at
    launch time makes Modal's cache key track the actual image: a new publish
    changes the digest (fresh pull), an unchanged image keeps the cache hit.

    Falls back to the unresolved tag, with a warning, if the registry cannot
    be reached (e.g. offline) — that preserves today's behavior.
    """
    if "@" in image_ref or not image_ref.startswith(f"{_REGISTRY}/"):
        return image_ref
    repo_path, _, tag = image_ref[len(_REGISTRY) + 1:].partition(":")
    if not tag:
        return image_ref
    try:
        token_url = f"https://{_REGISTRY}/token?scope=repository:{repo_path}:pull"
        with urllib.request.urlopen(token_url, timeout=10) as response:
            token = json.load(response)["token"]
        manifest_request = urllib.request.Request(
            f"https://{_REGISTRY}/v2/{repo_path}/manifests/{tag}",
            method="HEAD",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": _ACCEPT_MANIFEST_TYPES,
            },
        )
        with urllib.request.urlopen(manifest_request, timeout=10) as response:
            digest = response.headers.get("Docker-Content-Digest", "")
        if digest.startswith("sha256:"):
            return f"{image_ref}@{digest}"
        print(f"WARNING: registry returned no digest for {image_ref}; "
              "Modal may reuse a stale cached image for this tag.")
    except Exception as error:  # noqa: BLE001 - never fail the launch over this
        print(f"WARNING: could not resolve {image_ref} to a digest ({error}); "
              "Modal may reuse a stale cached image for this tag.")
    return image_ref


def resolve_uv_torch_backend(image_tag: str) -> str | None:
    """UV_TORCH_BACKEND for a launcher image tag.

    A caller-set UV_TORCH_BACKEND always wins. Otherwise sniff the CUDA
    version from an explicit image tag (cuda13 -> cu130, cuda12.6 -> cu126)
    so uv resolves torch against the image's toolkit. Mutable tags (e.g.
    py3.12-latest) return None and inherit the registry image's baked
    backend, which keeps a latest-tag CUDA transition safe.
    """
    override = os.environ.get("UV_TORCH_BACKEND")
    if override:
        return override
    tag = image_tag.lower()
    if "cuda13" in tag:
        return "cu130"
    if "cuda12.6" in tag:
        return "cu126"
    return None
