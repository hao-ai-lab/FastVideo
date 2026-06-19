"""``accel`` — a pure-python stand-in accelerator backend.

There is no GPU in this environment, so to prove the dispatch substrate is genuinely
device-generic (not a single-backend abstraction with a CPU special case) this backend registers a
second device, ``accel``, entirely in CPU-resident Python. It exercises everything a real GPU
backend would:

  * a **component** override — ``AccelDiT`` is built via ``COMPONENTS[("dit", "accel", …)]`` instead
    of the card's numpy factory, while components it does NOT register (text_encoder, vae) fall back
    over the device chain ``accel → cpu`` to the numpy factory;
  * **kernels at different arch rungs** — ``flow_sde_step`` is registered at ``sm90`` directly, while
    ``flow_match_step`` is registered only at ``generic``, so resolving it on an ``sm90`` accel
    platform must walk the arch fallback ``sm90 → sm80 → generic`` to find it;
  * the **parity oracle** — the ODE solver op (``_accel_flow_match_step``) is an independently
    written kernel that matches the numpy reference bit-for-bit, so a denoise run on the accel
    platform equals the cpu run. That independent-impl-vs-reference equality is the oracle a real
    backend must pass. (The SDE op below deliberately reuses the reference impl — a backend may
    legitimately not specialize every op — so the SDE test exercises the dispatch path, not an
    independent implementation.)

On a real box this file is where a torch/cuda backend would register its adapters and fused kernels
(unified primitives + model-co-located fusion variants) through the same two functions.
"""
from __future__ import annotations

from typing import Any

from v2.platform.backends.toy import ToyDiT
from v2.platform.registry import FLOW_MATCH_STEP, FLOW_SDE_STEP, register_component, register_kernel


# --------------------------------------------------------------------------- #
# Component: an accel-resident DiT (stand-in for a torch module placed on the  #
# device). Wraps the same toy compute so the parity oracle is bit-identical.   #
# --------------------------------------------------------------------------- #
class AccelDiT:
    """A device-tagged DiT wrapper. On a real box this would hold a torch module on the accelerator;
    here it delegates to the same numpy toy so accel ≡ cpu numerically (the oracle)."""

    def __init__(self, inner: Any):
        self._inner = inner
        self.device = "accel"

    def __call__(self, *args, **kwargs):
        return self._inner(*args, **kwargs)

    def __getattr__(self, name: str):
        # delegate the trainable surface (clone/copy_from/blend_from/mse_grad_step) to the inner toy
        return getattr(self._inner, name)


def _build_accel_dit(spec: Any, instance: Any, platform: Any) -> AccelDiT:
    # Build the same component the cpu rung would (via the card's factory), then place it "on device".
    inner = spec.factory(instance) if getattr(spec, "factory", None) is not None else ToyDiT()
    return AccelDiT(inner)


register_component("dit", _build_accel_dit, device="accel", source="accel(stand-in):AccelDiT")


class AccelComponent:
    """Generic device-tagged wrapper for non-callable component kinds (vae, audio_vae, …). Delegates
    the full call surface (``decode``/``encode``/…) to the inner toy, so accel ≡ cpu numerically."""

    def __init__(self, inner: Any):
        self._inner = inner
        self.device = "accel"

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


def _build_accel_component(spec: Any, instance: Any, platform: Any) -> AccelComponent:
    if getattr(spec, "factory", None) is None:
        raise RuntimeError(f"accel: no factory to wrap for kind={spec.kind!r}")
    return AccelComponent(spec.factory(instance))


# Override more than one component kind on accel (proving the seam is kind-generic, not dit-special).
# text_encoder is deliberately left UNregistered so the device→cpu fallback path stays demonstrated.
for _kind in ("vae", "audio_vae"):
    register_component(_kind, _build_accel_component, device="accel", source="accel(stand-in):AccelComponent")


# --------------------------------------------------------------------------- #
# Kernels: independent accel implementations that match the numpy reference.   #
# --------------------------------------------------------------------------- #
def _accel_flow_match_step(x_t, velocity, sigma_t: float, sigma_next: float):
    """Stand-in 'fused' accel solver step. Identical math to the numpy reference (the oracle)."""
    return x_t + (sigma_next - sigma_t) * velocity


# flow_match_step lives only at the `generic` arch rung → resolving on sm90 must fall back through
# the arch chain (sm90 → sm80 → generic). This proves the arch-fallback walk.
register_kernel(FLOW_MATCH_STEP,
                _accel_flow_match_step,
                device="accel",
                arch="generic",
                source="accel(stand-in):fused_flow_match")

# flow_sde_step is registered at sm90 directly. This backend does NOT specialize the SDE op — it
# reuses the numpy reference, so the accel SDE path is the same function as cpu (the SDE parity test
# therefore checks dispatch-path equivalence, not an independent impl; the ODE op above is the real
# independent oracle). A device that DID specialize SDE would be checked the same way the ODE op is.
from v2.loop.sampler import flow_sde_step_with_logprob  # noqa: E402

register_kernel(FLOW_SDE_STEP,
                flow_sde_step_with_logprob,
                device="accel",
                arch="sm90",
                source="accel(stand-in):sde_logprob")
