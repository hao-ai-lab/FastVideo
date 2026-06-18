"""Multi-backend dispatch substrate: device/arch resolution, fallback, and the parity oracle (§17).

These tests prove the two tuple-keyed registries (COMPONENTS, KERNELS) + the Platform resolve
implementations generically across devices, with the numpy reference as the terminal rung — and
that a second backend (the pure-python ``accel`` stand-in) is bit-for-bit identical to it. The
``cuda`` cells are declared-but-unavailable here, so the matrix is enumerable without a GPU and
without importing torch.
"""
from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest

from v2._enums import ExecutionProfile
from v2.cache import CacheManager
from v2.card import load_card
from v2.loop.driver import LoopRunner
from v2.platform.backends.toy import ToyDiT, ToyTextEncoder, ToyVAE
from v2.recipes.common import text_encode_node_fn
from v2.recipes.wan21 import build_wan21_card
from v2.recipes.wan_causal import build_wan_causal_card
from v2.parity import bit_identical               # the typed C1 instrument (ladder.py)
from v2.platform import (
    FLOW_MATCH_STEP,
    FLOW_SDE_STEP,
    Platform,
    component_matrix,
    kernel_matrix,
)
from v2.platform.backends.accel import AccelComponent, AccelDiT
from v2.request import DiffusionParams, TaskType, make_request
from v2.request.streams import Stream
from v2.runtime import Engine
from v2.runtime.context import RuntimeLoopContext

_HUB = Engine()           # borrow its observer/interceptor hubs for the loop context


def _instance(platform):
    card = build_wan21_card()
    return load_card(card, cache_manager=CacheManager.from_card(card), platform=platform)


def _drive_denoise(inst, prompt="a fox", seed=1, *, steps=8, sde=False, loop_id="diffusion_denoise"):
    """Drive a loop step-by-step on ``inst``'s platform; return the final latent."""
    req = make_request(TaskType.T2V, inst.card.model_id, prompt,
                       diffusion=DiffusionParams(num_steps=steps, seed=seed, sde_rollout=sde))
    slots: dict = {}
    text_encode_node_fn(inst, slots, req, None)
    profile = ExecutionProfile.ROLLOUT if sde else ExecutionProfile.SERVE
    ctx = RuntimeLoopContext(inst, observers=_HUB.observers, interceptors=_HUB.interceptors,
                             slots=slots, stream=Stream(req.request_id), cancel_scope=None,
                             profile=profile, metrics={}, request_id=req.request_id)
    runner = LoopRunner(inst.loop(loop_id), ctx, req, inst)
    while not runner.done:
        runner.step()
    return np.asarray(runner.result.outputs["latents"])


# --------------------------------------------------------------------------- #
# Detection + the default (unchanged) CPU path                                #
# --------------------------------------------------------------------------- #
def test_detect_is_cpu_without_torch():
    p = Platform.detect()
    assert p.device == "cpu" and p.arch == "numpy"
    assert p.device_chain == ("cpu",)


def test_default_instance_is_cpu_and_factory_terminal_unchanged():
    inst = load_card(build_wan21_card(), cache_manager=None)   # no platform → detect() → cpu
    assert inst.platform.device == "cpu"
    # the card's factory is the cpu/numpy terminal rung: component build is the toy, untouched
    assert isinstance(inst.component("transformer"), ToyDiT)
    assert isinstance(inst.component("text_encoder"), ToyTextEncoder)
    assert isinstance(inst.component("vae"), ToyVAE)


# --------------------------------------------------------------------------- #
# Kernel resolution: terminal rung, arch fallback, variant fallback           #
# --------------------------------------------------------------------------- #
def test_cpu_kernels_resolve_to_numpy_reference():
    p = Platform.cpu()
    src_fm = p.kernels.resolved_source(FLOW_MATCH_STEP)
    src_sde = p.kernels.resolved_source(FLOW_SDE_STEP)
    assert "('flow_match_step', 'cpu', 'numpy', 'default')" in src_fm
    assert "loop.sampler:flow_match_euler_step" in src_fm
    assert "('flow_sde_step', 'cpu', 'numpy', 'default')" in src_sde


def test_accel_arch_fallback_walk():
    """flow_match is only at the `generic` arch rung → resolving on sm90 must walk sm90→sm80→generic;
    flow_sde is at sm90 directly."""
    p = Platform.accel("sm90")
    assert p.arch_chain("accel") == ("sm90", "sm80", "generic")
    assert "('flow_match_step', 'accel', 'generic', 'default')" in p.kernels.resolved_source(FLOW_MATCH_STEP)
    assert "('flow_sde_step', 'accel', 'sm90', 'default')" in p.kernels.resolved_source(FLOW_SDE_STEP)


def test_arch_fallback_is_monotonic_never_newer():
    """An OLDER device must never fall back to a newer-arch (binary-incompatible) kernel; a NEWER
    device may still run older-arch kernels (forward-compatible)."""
    old = Platform.accel("sm70")                       # older than the accel chain's sm90/sm80
    chain = old.arch_chain("accel")
    assert chain[0] == "sm70"
    assert "sm90" not in chain and "sm80" not in chain  # newer archs excluded
    assert "generic" in chain                           # portable terminal kept
    new = Platform.cuda("sm120")                        # newer than the cuda chain
    assert {"sm100", "sm90", "sm80"} <= set(new.arch_chain("cuda"))   # older archs still usable


def test_device_precedence_accel_before_cpu():
    """flow_sde exists on BOTH accel(sm90) and cpu(numpy) — the accel device wins (chain order)."""
    p = Platform.accel("sm90")
    assert "accel" in p.kernels.resolved_source(FLOW_SDE_STEP)


def test_unknown_variant_falls_back_to_default():
    p = Platform.cpu()
    # asking for a variant that was never registered still resolves to the "default" cell
    assert p.kernels.get(FLOW_MATCH_STEP, variant="does_not_exist") is p.kernels.get(FLOW_MATCH_STEP)


def test_cuda_platform_falls_back_to_cpu_when_unavailable():
    """A cuda platform on a torchless box: the cuda cells are unavailable, so resolution skips them
    and walks the device chain cuda→cpu to the numpy terminal."""
    p = Platform.cuda("sm90")
    assert "('flow_match_step', 'cpu', 'numpy', 'default')" in p.kernels.resolved_source(FLOW_MATCH_STEP)


# --------------------------------------------------------------------------- #
# Component resolution: device override + per-kind fallback to the factory    #
# --------------------------------------------------------------------------- #
def test_accel_component_override_and_device_fallback():
    inst = _instance(Platform.accel("sm90"))
    dit = inst.component("transformer")               # registered for accel → AccelDiT wrapper
    assert isinstance(dit, AccelDiT) and dit.device == "accel"
    vae = inst.component("vae")                        # vae is ALSO overridden on accel (kind-generic)
    assert isinstance(vae, AccelComponent) and vae.device == "accel"
    # text_encoder is NOT registered for accel → falls back over accel→cpu to the numpy factory toy
    assert isinstance(inst.component("text_encoder"), ToyTextEncoder)


def test_cuda_component_falls_back_to_factory_when_unavailable():
    inst = _instance(Platform.cuda("sm90"))           # cuda dit cell unavailable on this box
    assert isinstance(inst.component("transformer"), ToyDiT)   # → spec.factory terminal


# --------------------------------------------------------------------------- #
# THE PARITY ORACLE: a new backend must match the numpy reference bit-for-bit #
# (checked with the typed C1 instrument `parity.bit_identical`, not a bare ==) #
# --------------------------------------------------------------------------- #
def test_parity_oracle_accel_equals_cpu_ode():
    """accel's ODE solver kernel is an INDEPENDENT impl (`_accel_flow_match_step`) — this is the real
    oracle: a separately-written backend kernel must match the numpy reference, C1 bit-identical."""
    cpu_latent = _drive_denoise(_instance(Platform.cpu()), seed=7, steps=8)
    accel_latent = _drive_denoise(_instance(Platform.accel("sm90")), seed=7, steps=8)
    assert bit_identical(cpu_latent, accel_latent)


def test_accel_sde_dispatch_path_equivalent():
    """The SDE op is NOT specialized by accel (it reuses the numpy reference), so this checks the
    stochastic path still resolves + runs through the kernel table and matches — a dispatch-path
    test, not an independent-impl oracle (see accel.py)."""
    cpu_latent = _drive_denoise(_instance(Platform.cpu()), seed=11, steps=6, sde=True)
    accel_latent = _drive_denoise(_instance(Platform.accel("sm90")), seed=11, steps=6, sde=True)
    assert bit_identical(cpu_latent, accel_latent)


def test_accel_denoise_still_produces_valid_output():
    out = _drive_denoise(_instance(Platform.accel("sm90")), seed=3, steps=4)
    assert out.shape[0] == 4 and np.all(np.isfinite(out))


def test_parity_oracle_accel_equals_cpu_second_loop():
    """A SECOND loop family (wan-causal chunk rollout) also routes its solver through the kernel
    table — so the seam is finished across loops, not just wan21 denoise. accel ≡ cpu, bit-identical."""
    def drive(platform):
        card = build_wan_causal_card()
        inst = load_card(card, cache_manager=CacheManager.from_card(card), platform=platform)
        loop_id = next(iter(inst.card.loops))          # the chunk-rollout loop
        return _drive_denoise(inst, seed=5, loop_id=loop_id)
    assert bit_identical(drive(Platform.cpu()), drive(Platform.accel("sm90")))


# --------------------------------------------------------------------------- #
# Enumerable matrix — including unavailable backends, without importing torch  #
# --------------------------------------------------------------------------- #
def test_kernel_matrix_enumerates_all_backends_including_unavailable_cuda():
    rows = kernel_matrix()
    devices = {r["device"] for r in rows}
    assert {"cpu", "accel", "cuda"} <= devices
    cuda_rows = [r for r in rows if r["device"] == "cuda"]
    assert cuda_rows and all(r["available"] is False for r in cuda_rows)   # declared, not runnable here
    assert all(r["available"] is True for r in rows if r["device"] == "cpu")


def test_component_matrix_includes_declared_unavailable_torch_adapter():
    rows = component_matrix()
    cuda = [r for r in rows if r["device"] == "cuda" and r["kind"] == "dit"]
    assert cuda and cuda[0]["available"] is False
    accel = [r for r in rows if r["device"] == "accel" and r["kind"] == "dit"]
    assert accel and accel[0]["available"] is True


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is not None,
    reason="torch installed (GPU box): the cuda-availability probe imports torch by design; this "
           "no-torch-import invariant is only verifiable without torch.")
def test_matrix_enumeration_does_not_import_torch():
    """The manifest is built from declarations + a find_spec predicate — never an import of torch
    (when torch is absent). With torch installed the cuda-availability probe imports it by design, so
    this is skipped; the lazy adapter-module guard stays covered by test_torch_backend.py."""
    kernel_matrix(); component_matrix()
    assert "torch" not in sys.modules
