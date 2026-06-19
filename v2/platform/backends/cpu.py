"""CPU / numpy backend — the terminal rung and parity oracle.

This backend registers the numpy reference kernels. It is the bottom of every device fallback
chain: any op a richer backend hasn't implemented resolves here, and every other backend's output
is checked against this one (the consistency ladder's C0/C1 oracle).

Components are NOT registered here: the card's ``ComponentSpec.factory`` *is* the cpu/numpy
component rung (``Platform.build_component`` falls back to it when no ``(kind, device, …)`` cell
matches). That keeps every existing card working with zero registration.
"""
from __future__ import annotations

from v2.loop.sampler import flow_match_euler_step, flow_sde_step_with_logprob
from v2.platform.registry import FLOW_MATCH_STEP, FLOW_SDE_STEP, register_kernel

# The numpy solver primitives, registered as the terminal (cpu, numpy) kernels. These are the exact
# same functions the loops used to call directly — so dispatching through the registry on a CPU box
# is bit-for-bit identical to the pre-registry behavior.
register_kernel(FLOW_MATCH_STEP,
                flow_match_euler_step,
                device="cpu",
                arch="numpy",
                source="loop.sampler:flow_match_euler_step")
register_kernel(FLOW_SDE_STEP,
                flow_sde_step_with_logprob,
                device="cpu",
                arch="numpy",
                source="loop.sampler:flow_sde_step_with_logprob")
