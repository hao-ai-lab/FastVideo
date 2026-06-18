# STUB: re-exports fastvideo until vendored (see memory: v2-vendoring-approach).
"""``forward_context`` facade. The fastvideo attention layer reads a thread-local ForwardContext via
``get_forward_context()`` (asserts it's set), so every torch forward in the backend runs inside
``set_forward_context(...)``. Re-exported here so v2 code imports ``v2.forward_context``; a vendored
cutover (which also requires forking the attention layer to drop the global context) replaces this body."""
from fastvideo.forward_context import get_forward_context, set_forward_context  # noqa: F401
