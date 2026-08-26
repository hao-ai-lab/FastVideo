# SPDX-License-Identifier: Apache-2.0
"""CPU-only coverage for the change-aware /merge test planner."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PLANNER_PATH = REPO_ROOT / ".github/scripts/plan_merge_ci.py"
SPEC = importlib.util.spec_from_file_location("plan_merge_ci", PLANNER_PATH)
assert SPEC is not None and SPEC.loader is not None
PLAN_MERGE_CI = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLAN_MERGE_CI
SPEC.loader.exec_module(PLAN_MERGE_CI)


def test_docs_only_merge_adds_no_gpu_lanes():
    plan = PLAN_MERGE_CI.classify_paths([
        "docs/contributing/testing.md",
        "README.md",
        "requirements-mkdocs.txt",
    ])

    assert plan.encoded_lanes() == ",none,"
    assert plan.encoded_golden_tests() == "none"
    assert plan.encoded_ssim_tests() == "none"


def test_model_family_change_selects_focused_golden_and_ssim():
    plan = PLAN_MERGE_CI.classify_paths(["fastvideo/models/dits/wanvideo.py"])

    assert plan.encoded_lanes() == ",golden-gate,ssim,"
    assert plan.encoded_golden_tests() == "test_wan_t2v.py"
    assert plan.encoded_ssim_tests() == (
        "test_causal_similarity.py,test_wan_i2v_similarity.py,test_wan_t2v_similarity.py")


def test_flux2_change_does_not_pull_unrelated_flux1_quality_tests():
    plan = PLAN_MERGE_CI.classify_paths(["fastvideo/pipelines/basic/flux_2/flux_2_pipeline.py"])

    assert plan.encoded_golden_tests() == "test_flux2_klein.py"
    assert plan.encoded_ssim_tests() == "test_flux2_similarity.py"


def test_changed_ssim_test_selects_only_that_file():
    plan = PLAN_MERGE_CI.classify_paths(["fastvideo/tests/ssim/test_flux_t2i_similarity.py"])

    assert plan.encoded_lanes() == ",ssim,"
    assert plan.encoded_ssim_tests() == "test_flux_t2i_similarity.py"


def test_shared_golden_harness_or_reference_requires_full_golden_lane():
    plan = PLAN_MERGE_CI.classify_paths(["fastvideo/tests/golden_gate/_harness.py"])

    assert plan.encoded_lanes() == ",golden-gate,"
    assert plan.encoded_golden_tests() == "all"


def test_shared_ssim_harness_requires_full_ssim_lane():
    plan = PLAN_MERGE_CI.classify_paths(["fastvideo/tests/ssim/inference_similarity_utils.py"])

    assert plan.encoded_lanes() == ",ssim,"
    assert plan.encoded_ssim_tests() == "all"


def test_training_domains_do_not_fan_out_to_unrelated_lanes():
    plan = PLAN_MERGE_CI.classify_paths([
        "fastvideo/training/wan_self_forcing_distillation_pipeline.py",
        "fastvideo/tests/training/lora/test_lora_training.py",
    ])

    assert plan.encoded_lanes() == ",self-forcing,lora-training,"


def test_shared_training_code_selects_all_legacy_training_consumers():
    plan = PLAN_MERGE_CI.classify_paths(["fastvideo/training/trackers.py"])

    assert plan.ordered_lanes() == PLAN_MERGE_CI.LEGACY_TRAINING_LANES


def test_dataset_change_selects_only_training_consumers():
    plan = PLAN_MERGE_CI.classify_paths(["fastvideo/dataset/dataloader/bucket.py"])

    assert plan.ordered_lanes() == PLAN_MERGE_CI.ALL_TRAINING_LANES


def test_performance_implementation_selects_only_performance_lane():
    plan = PLAN_MERGE_CI.classify_paths(["fastvideo/performance/metric_policy.py"])

    assert plan.encoded_lanes() == ",performance,"


def test_shared_runtime_change_gets_focused_quality_smoke_not_every_lane():
    plan = PLAN_MERGE_CI.classify_paths(["fastvideo/registry.py"])

    assert plan.encoded_lanes() == ",golden-gate,ssim,"
    assert plan.encoded_golden_tests() == "all"
    assert plan.encoded_ssim_tests() == "test_flux_t2i_similarity.py,test_wan_t2v_similarity.py"


def test_lane_script_selects_its_single_lane():
    plan = PLAN_MERGE_CI.classify_paths([".buildkite/scripts/lanes/api_server.sh"])

    assert plan.encoded_lanes() == ",api-server,"


def test_fastcheck_lane_script_is_not_repeated_by_merge():
    plan = PLAN_MERGE_CI.classify_paths([".buildkite/scripts/lanes/encoder.sh"])

    assert plan.encoded_lanes() == ",none,"


def test_cross_cutting_dependency_change_fails_out_to_every_merge_lane():
    plan = PLAN_MERGE_CI.classify_paths(["uv.lock"])

    assert plan.ordered_lanes() == PLAN_MERGE_CI.MERGE_LANES
    assert plan.encoded_golden_tests() == "all"
    assert plan.encoded_ssim_tests() == "all"


def test_unknown_path_fails_closed_to_every_merge_lane():
    plan = PLAN_MERGE_CI.classify_paths(["new_runtime/build_rules.toml"])

    assert plan.ordered_lanes() == PLAN_MERGE_CI.MERGE_LANES


def test_empty_or_failed_changed_file_query_fails_closed():
    assert PLAN_MERGE_CI.classify_paths([]).ordered_lanes() == PLAN_MERGE_CI.MERGE_LANES
    assert PLAN_MERGE_CI.classify_paths([
        "__FASTVIDEO_CI_PLAN_ALL__"
    ]).ordered_lanes() == PLAN_MERGE_CI.MERGE_LANES


def test_dreamverse_change_relies_on_its_existing_fastcheck_e2e_lane():
    plan = PLAN_MERGE_CI.classify_paths(["apps/dreamverse/web/src/App.tsx"])

    assert plan.encoded_lanes() == ",none,"


def test_local_only_parity_scaffold_does_not_trigger_unrelated_gpu_lanes():
    plan = PLAN_MERGE_CI.classify_paths(["tests/local_tests/flux/test_flux_dev_component_loaders.py"])

    assert plan.encoded_lanes() == ",none,"
