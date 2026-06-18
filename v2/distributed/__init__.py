# STUB: re-exports fastvideo until vendored (see memory: v2-vendoring-approach).
"""``distributed`` facade — single-GPU dist-init the loaders need (1x1 device mesh). Re-exported so v2
code imports ``v2.distributed``; a vendored cutover copies parallel_state getters + communication_op."""
from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel  # noqa: F401
