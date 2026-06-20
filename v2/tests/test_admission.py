"""Admission = reservation before stepping.

Two requests that fit individually but jointly OOM are rejected at admission, not at step 37.
And crucially: admission-forced serialization does NOT change outputs (still bit-identical).
"""
from __future__ import annotations

import numpy as np

from v2.memory import MemoryManager, OutOfMemory
from v2.recipes import build_default_engine
from v2.recipes.wan21.loop import latent_shape
from v2.parity import compare_outputs
from v2.request import DiffusionParams, TaskType, make_request
from v2.runtime import AdmissionController, Engine


def test_memory_manager_reserve_release_and_oom():
    mm = MemoryManager(total_bytes=100)
    r = mm.reserve("t", 60)
    assert mm.reserved == 60 and mm.available == 40
    try:
        mm.reserve("t", 60)
        assert False, "expected OutOfMemory"
    except OutOfMemory:
        pass
    mm.release(r)
    assert mm.reserved == 0


def test_sleep_wake_frees_by_tag():
    mm = MemoryManager(total_bytes=1000)
    mm.reserve("dit", 400)
    mm.reserve("vae", 200)
    freed = mm.sleep(["dit"])              # component-granular: drop DiT, keep VAE
    assert freed == 400 and mm.reserved == 200


def _req(prompt, seed):
    return make_request(TaskType.T2V, "wan2.1-1.3b", prompt,
                        diffusion=DiffusionParams(num_steps=3, seed=seed))


