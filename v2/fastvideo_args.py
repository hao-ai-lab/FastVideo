# STUB: re-exports fastvideo until vendored (see memory: v2-vendoring-approach).
"""``FastVideoArgs`` facade — the args object the component loaders consume. Re-exported so v2 code imports
``v2.fastvideo_args``; a vendored cutover slims it to inference-only args."""
from fastvideo.fastvideo_args import FastVideoArgs  # noqa: F401
