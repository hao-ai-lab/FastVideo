"""Training methods on Wan2.1-1.3B (design_v3 §10) — all over the SHARED loops, train ≡ serve.

Verifies: finetune learns (loss ↓), DMD2 real/fake scores, NFT likelihood-free C2 + group reuse +
old-as-behavior-policy, self-forcing on the causal chunk loop + slab-KV.
"""
from __future__ import annotations

import numpy as np

from v2._enums import ConsistencyLevel
from v2.models.wan21 import build_wan21_card
from v2.models.wan_causal import build_wan_causal_card
from v2.training import (
    WeightRole,
    build_diffusion_nft,
    build_dmd2,
    build_finetune,
    build_self_forcing,
)

BATCH = {"prompts": ["a cat", "a dog"], "seeds": [1, 2]}


def test_finetune_learns_and_is_c1():
    ft = build_finetune(build_wan21_card())
    l0, _ = ft.train_step(BATCH, 0)
    l1, m1 = ft.train_step(BATCH, 1)
    assert np.isfinite(l1["loss"]) and l1["loss"] <= l0["loss"] + 1e-9   # real SGD ⇒ non-increasing
    assert m1["grad_norm/student"] > 0.0
    assert ft.consistency_level() is ConsistencyLevel.C1 and not ft.manages_optimization()


def test_dmd2_real_and_fake_scores_finite_with_two_optimizers():
    dm = build_dmd2(build_wan21_card(), generator_update_interval=1)
    lm, m = dm.train_step(BATCH, 0)
    assert np.isfinite(lm["dmd_loss"]) and np.isfinite(lm["critic_loss"])
    assert m["grad_norm/critic"] > 0.0 and m["grad_norm/student"] > 0.0      # student + critic


def test_diffusion_nft_is_likelihood_free_c2_with_group_reuse():
    nft = build_diffusion_nft(build_wan21_card(), num_video_per_prompt=4, num_inner_timesteps=2)
    assert nft.manages_optimization() and nft.consistency_level() is ConsistencyLevel.C2
    assert nft.old_sync.role is WeightRole.OLD_POLICY     # sample from OLD, not student (§8.4)
    lm, m = nft.managed_train_step(BATCH, 0)
    assert np.isfinite(lm["policy_loss"]) and np.isfinite(m["old_deviate"])
    assert m["advantage_std"] >= 0.0
    # the shared prompt encodes ONCE per K-sample group (the 24× text-encode reduction)
    assert nft.old.caches.stats()["feature"]["hits"] > 0


def test_nft_old_policy_decays_toward_student_over_iterations():
    nft = build_diffusion_nft(build_wan21_card(), num_video_per_prompt=2)
    _, m0 = nft.managed_train_step(BATCH, 0)
    _, m200 = nft.managed_train_step(BATCH, 200)
    assert m0["old_decay"] == 0.0                          # decay_type=1: min(step*0.001, 0.5)
    assert m200["old_decay"] == 0.2
    assert m0["old_weights_version"] != m200["old_weights_version"]   # version bumped each sync


def test_self_forcing_drives_causal_chunk_loop_and_slab_kv():
    sf = build_self_forcing(build_wan_causal_card(training_mode=True), generator_update_interval=1)
    assert sf.student_loop_id == "chunk_rollout"
    lm, m = sf.train_step(BATCH, 0)
    assert np.isfinite(lm["dmd_loss"]) and np.isfinite(lm["critic_loss"])
    assert sf.student.caches.stats()["slab_kv"]["used_bytes"] > 0     # KV slabs written during rollout


def test_methods_reference_the_same_loops_the_engine_serves():
    assert build_finetune(build_wan21_card()).student_loop_id == "diffusion_denoise"
    assert build_dmd2(build_wan21_card()).student_loop_id == "diffusion_denoise"
    assert build_self_forcing(build_wan_causal_card()).student_loop_id == "chunk_rollout"
