"""Backend registration modules. Imported for side effects by ``Platform.ensure_backends_loaded``.

Each module calls ``register_component`` / ``register_kernel`` at import time. They live behind the
lazy loader (not imported by ``platform/__init__`` or ``card/``) so the registry stays a pure leaf
and there are no import cycles. On disk these mirror the kernel-colocation answer: ``cpu`` is the
unified numpy reference; a real ``torch_cuda`` backend would register both unified primitives and
model-co-located fusion *variants* through this same API.
"""
