#!/usr/bin/env python3
"""Select the additive GPU integration lanes needed by a PR diff.

Fastcheck is the universal six-lane baseline and is intentionally not repeated
here.  This planner selects only the more expensive merge-gate lanes.  Unknown
source/build paths fail closed to the complete integration set, while explicit
documentation and repository-metadata paths require no additional GPU work.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

MERGE_LANES = (
    "golden-gate",
    "ssim",
    "lora-inference",
    "lora-extraction",
    "training",
    "distillation",
    "self-forcing",
    "lora-training",
    "training-vsa",
    "inference-vmoba",
    "performance",
    "api-server",
    "train-framework",
    "eval",
)

LANE_SCRIPT_TO_KEY = {
    "api_server.sh": "api-server",
    "distillation_dmd.sh": "distillation",
    "eval.sh": "eval",
    "golden_gate.sh": "golden-gate",
    "inference_lora.sh": "lora-inference",
    "inference_vmoba.sh": "inference-vmoba",
    "lora_extraction.sh": "lora-extraction",
    "performance.sh": "performance",
    "self_forcing.sh": "self-forcing",
    "ssim.sh": "ssim",
    "train_framework.sh": "train-framework",
    "training.sh": "training",
    "training_lora.sh": "lora-training",
    "training_vsa.sh": "training-vsa",
}

FASTCHECK_LANE_SCRIPTS = {
    "dreamverse.sh",
    "encoder.sh",
    "kernel_tests.sh",
    "transformer.sh",
    "vae.sh",
}

LEGACY_TRAINING_LANES = (
    "training",
    "distillation",
    "self-forcing",
    "lora-training",
    "training-vsa",
)

ALL_TRAINING_LANES = (*LEGACY_TRAINING_LANES, "train-framework")

SSIM_SMOKE_TESTS = (
    "test_flux_t2i_similarity.py",
    "test_wan_t2v_similarity.py",
)

SAFE_PATTERNS = (
    "*.md",
    "*.rst",
    ".agents/**",
    ".claude/**",
    ".codex/**",
    ".github/ISSUE_TEMPLATE/**",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/dependabot.yml",
    ".github/mergify.yml",
    ".github/scripts/**",
    ".github/workflows/**",
    ".buildkite/scripts/pre_commit.sh",
    ".git-blame-ignore-revs",
    ".gitattributes",
    ".gitignore",
    ".pre-commit-config.yaml",
    "AGENTS.md",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "NOTICE",
    "__init__.py",
    "collect_env.py",
    "SECURITY.md",
    "assets/**",
    "comfyui/**",
    "docs/**",
    "examples/**",
    "mkdocs.yml",
    "requirements-mkdocs.in",
    "requirements-mkdocs.txt",
    "scripts/**",
    "tests/__init__.py",
    "tests/local_tests/**",
)

ALL_IMPACT_PATTERNS = (
    ".buildkite/pipeline.yml",
    "docker/**",
    "pyproject.toml",
    "requirements*.txt",
    "setup.cfg",
    "setup.py",
    "uv.lock",
)


@dataclass(frozen=True)
class FamilyCoverage:
    pattern: re.Pattern[str]
    golden_tests: tuple[str, ...]
    ssim_tests: tuple[str, ...]


FAMILY_COVERAGE = (
    FamilyCoverage(
        re.compile(r"(^|[/_.-])dreamx(_world)?([/_.-]|$)"),
        ("test_dreamx.py", ),
        ("test_dreamx_world_similarity.py", ),
    ),
    FamilyCoverage(
        re.compile(r"(^|[/_.-])flux[_-]?2([/_.-]|$)"),
        ("test_flux2_klein.py", ),
        ("test_flux2_similarity.py", ),
    ),
    FamilyCoverage(
        re.compile(r"(^|[/_.-])flux(?![_-]?2)([/_.-]|$)"),
        ("test_flux.py", ),
        ("test_flux_t2i_similarity.py", ),
    ),
    FamilyCoverage(
        re.compile(r"(^|[/_.-])(hunyuan)?gamecraft([/_.-]|$)"),
        ("test_gamecraft.py", ),
        ("test_gamecraft_similarity.py", ),
    ),
    FamilyCoverage(
        re.compile(r"(^|[/_.-])gen3c([/_.-]|$)"),
        ("test_gen3c.py", ),
        ("test_gen3c_similarity.py", ),
    ),
    FamilyCoverage(
        re.compile(r"(^|[/_.-])glm[_-]?image([/_.-]|$)"),
        ("test_glm_image.py", ),
        ("test_glm_image_similarity.py", ),
    ),
    FamilyCoverage(
        re.compile(r"(^|[/_.-])kandinsky[_-]?5([/_.-]|$)"),
        ("test_kandinsky5.py", ),
        ("test_kandinsky5_similarity.py", ),
    ),
    FamilyCoverage(
        re.compile(r"(^|[/_.-])lingbot([a-z0-9_-]*)([/_.-]|$)"),
        ("test_lingbot.py", ),
        ("test_lingbot_similarity.py", ),
    ),
    FamilyCoverage(
        re.compile(r"(^|[/_.-])longcat([/_.-]|$)"),
        ("test_longcat.py", ),
        ("test_longcat_similarity.py", ),
    ),
    FamilyCoverage(
        re.compile(r"(^|[/_.-])ltx[_-]?2([/_.-]|$)"),
        ("test_ltx2.py", ),
        ("test_ltx2_similarity.py", ),
    ),
    FamilyCoverage(
        re.compile(r"(^|[/_.-])matrixgame[_-]?2([/_.-]|$)"),
        ("test_matrixgame.py", ),
        ("test_matrixgame2_similarity.py", ),
    ),
    FamilyCoverage(
        re.compile(r"(^|[/_.-])matrixgame[_-]?3([/_.-]|$)"),
        ("test_matrixgame.py", ),
        ("test_matrixgame3_similarity.py", ),
    ),
    FamilyCoverage(
        re.compile(r"(^|[/_.-])minimax[_-]?h3([/_.-]|$)"),
        ("test_minimax_h3_t2v.py", ),
        ("test_minimax_h3_similarity.py", ),
    ),
    FamilyCoverage(
        re.compile(r"(^|[/_.-])sd[_-]?3([._-]?5)?([/_.-]|$)"),
        ("test_sd35.py", ),
        ("test_sd35_similarity.py", ),
    ),
    FamilyCoverage(
        re.compile(r"(^|[/_.-])stable[_-]?audio([/_.-]|$)"),
        ("test_stable_audio.py", ),
        ("test_stable_audio_similarity.py", ),
    ),
    FamilyCoverage(
        re.compile(r"(^|[/_.-])turbo(diffusion)?([/_.-]|$)"),
        (),
        ("test_turbodiffusion_similarity.py", ),
    ),
    FamilyCoverage(
        re.compile(r"(^|[/_.-])wan(video)?([/_.-]|$)"),
        ("test_wan_t2v.py", ),
        (
            "test_causal_similarity.py",
            "test_wan_i2v_similarity.py",
            "test_wan_t2v_similarity.py",
        ),
    ),
    FamilyCoverage(
        re.compile(r"(^|[/_.-])z[_-]?image([/_.-]|$)"),
        ("test_zimage.py", ),
        ("test_zimage_similarity.py", ),
    ),
)


@dataclass
class MergePlan:
    lanes: set[str] = field(default_factory=set)
    golden_tests: set[str] = field(default_factory=set)
    ssim_tests: set[str] = field(default_factory=set)
    golden_all: bool = False
    ssim_all: bool = False
    reasons: list[str] = field(default_factory=list)

    def add_lanes(self, *lanes: str, reason: str) -> None:
        unknown = set(lanes) - set(MERGE_LANES)
        if unknown:
            raise ValueError(f"Unknown merge lanes: {sorted(unknown)}")
        self.lanes.update(lanes)
        self.reasons.append(reason)

    def add_golden(self, tests: tuple[str, ...], reason: str) -> None:
        self.add_lanes("golden-gate", reason=reason)
        self.golden_tests.update(tests)

    def add_ssim(self, tests: tuple[str, ...], reason: str) -> None:
        self.add_lanes("ssim", reason=reason)
        self.ssim_tests.update(tests)

    def require_all(self, reason: str) -> None:
        self.lanes.update(MERGE_LANES)
        self.golden_all = True
        self.ssim_all = True
        self.reasons.append(reason)

    def ordered_lanes(self) -> tuple[str, ...]:
        return tuple(lane for lane in MERGE_LANES if lane in self.lanes)

    def encoded_lanes(self) -> str:
        lanes = self.ordered_lanes()
        return "," + ",".join(lanes or ("none", )) + ","

    def encoded_golden_tests(self) -> str:
        if "golden-gate" not in self.lanes:
            return "none"
        if self.golden_all or not self.golden_tests:
            return "all"
        return ",".join(sorted(self.golden_tests))

    def encoded_ssim_tests(self) -> str:
        if "ssim" not in self.lanes:
            return "none"
        if self.ssim_all or not self.ssim_tests:
            return "all"
        return ",".join(sorted(self.ssim_tests))


def _matches_any(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def _family_coverage(path: str) -> tuple[set[str], set[str]]:
    normalized = path.lower()
    golden: set[str] = set()
    ssim: set[str] = set()
    for family in FAMILY_COVERAGE:
        if family.pattern.search(normalized):
            golden.update(family.golden_tests)
            ssim.update(family.ssim_tests)
    return golden, ssim


def _select_output_coverage(plan: MergePlan, path: str) -> None:
    golden, ssim = _family_coverage(path)
    if golden:
        plan.add_golden(tuple(sorted(golden)), reason=f"model-family golden coverage: {path}")
    else:
        plan.golden_all = True
        plan.add_lanes("golden-gate", reason=f"shared output golden coverage: {path}")
    if ssim:
        plan.add_ssim(tuple(sorted(ssim)), reason=f"model-family SSIM coverage: {path}")
    else:
        plan.add_ssim(SSIM_SMOKE_TESTS, reason=f"shared output SSIM smoke coverage: {path}")


def classify_paths(paths: list[str]) -> MergePlan:
    plan = MergePlan()
    normalized_paths: list[str] = []
    for raw_path in paths:
        path = raw_path.strip()
        while path.startswith("./"):
            path = path[2:]
        if path:
            normalized_paths.append(path)
    normalized_paths = sorted(set(normalized_paths))
    if not normalized_paths:
        plan.require_all("changed-file list was empty; failing closed")
        return plan

    for path in normalized_paths:
        if path == "__FASTVIDEO_CI_PLAN_ALL__":
            plan.require_all("changed-file API failed; failing closed")
            continue

        if path in {"requirements-mkdocs.in", "requirements-mkdocs.txt"}:
            plan.reasons.append(f"documentation dependencies need no GPU integration: {path}")
            continue

        if _matches_any(path, ALL_IMPACT_PATTERNS):
            plan.require_all(f"cross-cutting build/runtime surface: {path}")
            continue

        lane_script_prefix = ".buildkite/scripts/lanes/"
        if path.startswith(lane_script_prefix):
            script_name = Path(path).name
            lane = LANE_SCRIPT_TO_KEY.get(script_name)
            if lane is None:
                if script_name in FASTCHECK_LANE_SCRIPTS:
                    plan.reasons.append(f"covered by automatic Fastcheck lane: {path}")
                else:
                    plan.require_all(f"unknown lane script: {path}")
            elif lane == "golden-gate":
                plan.golden_all = True
                plan.add_lanes(lane, reason=f"golden lane implementation: {path}")
            elif lane == "ssim":
                plan.ssim_all = True
                plan.add_lanes(lane, reason=f"SSIM lane implementation: {path}")
            else:
                plan.add_lanes(lane, reason=f"lane implementation: {path}")
            continue

        if path.startswith("fastvideo/tests/golden_gate/"):
            name = Path(path).name
            if name.startswith("test_") and name.endswith(".py"):
                plan.add_golden((name, ), reason=f"changed golden test: {path}")
            elif name in {"AGENTS.md", "README.md"}:
                plan.reasons.append(f"golden documentation only: {path}")
            else:
                plan.golden_all = True
                plan.add_lanes("golden-gate", reason=f"shared golden harness/reference: {path}")
            continue

        if path.startswith("fastvideo/tests/ssim/"):
            name = Path(path).name
            if name.startswith("test_") and name.endswith(".py"):
                plan.add_ssim((name, ), reason=f"changed SSIM test: {path}")
            elif path.endswith((".py", ".json", ".pt", ".png", ".mp4")):
                plan.ssim_all = True
                plan.add_lanes("ssim", reason=f"shared SSIM harness/reference: {path}")
            continue

        if path.startswith("fastvideo/tests/performance/") or path.startswith(".buildkite/performance-benchmarks/"):
            plan.add_lanes("performance", reason=f"performance coverage: {path}")
            continue
        if path.startswith(("fastvideo/performance/", "fastvideo/performance_dashboard/",
                            "apps/performance_dashboard/")):
            plan.add_lanes("performance", reason=f"performance implementation: {path}")
            continue
        if path.startswith("fastvideo/benchmarks/"):
            if "/mlx_" in path or Path(path).name.startswith("mlx_"):
                plan.reasons.append(f"covered by the path-filtered macOS MLX workflow: {path}")
            else:
                plan.add_lanes("performance", reason=f"benchmark implementation: {path}")
            continue
        if path.startswith("fastvideo/tests/eval/") or path.startswith("fastvideo/eval/"):
            plan.add_lanes("eval", reason=f"evaluation coverage: {path}")
            continue
        if path.startswith("fastvideo/third_party/eval/"):
            plan.add_lanes("eval", reason=f"vendored evaluation implementation: {path}")
            continue
        if path.startswith("fastvideo/tests/lora_extraction/") or path.startswith("scripts/lora_extraction/"):
            plan.add_lanes("lora-extraction", reason=f"LoRA extraction coverage: {path}")
            continue
        if path.startswith("fastvideo/tests/inference/lora/"):
            plan.add_lanes("lora-inference", reason=f"LoRA inference coverage: {path}")
            continue
        if path.startswith("fastvideo/tests/inference/vmoba/"):
            plan.add_lanes("inference-vmoba", reason=f"VMoBA inference coverage: {path}")
            continue
        if path.startswith(("fastvideo/dataset/", "fastvideo/workflow/", "fastvideo/pipelines/preprocess/",
                            "fastvideo/pipelines/training/")):
            plan.add_lanes(*ALL_TRAINING_LANES, reason=f"shared data/training input surface: {path}")
            continue
        if path.startswith("fastvideo/tests/train/") or path.startswith("fastvideo/train/"):
            plan.add_lanes("train-framework", reason=f"modular training coverage: {path}")
            continue

        if path.startswith("fastvideo/tests/training/"):
            lowered = path.lower()
            if "/vanilla/" in lowered:
                plan.add_lanes("training", reason=f"vanilla training coverage: {path}")
            elif "/distill/" in lowered:
                plan.add_lanes("distillation", reason=f"distillation coverage: {path}")
            elif "/self-forcing/" in lowered:
                plan.add_lanes("self-forcing", reason=f"self-forcing coverage: {path}")
            elif "/lora/" in lowered:
                plan.add_lanes("lora-training", reason=f"LoRA training coverage: {path}")
            elif "/vsa/" in lowered:
                plan.add_lanes("training-vsa", reason=f"VSA training coverage: {path}")
            else:
                plan.add_lanes(*LEGACY_TRAINING_LANES, reason=f"shared legacy training coverage: {path}")
            continue

        if path.startswith("fastvideo/training/"):
            lowered = path.lower()
            if "self_forcing" in lowered:
                plan.add_lanes("self-forcing", reason=f"self-forcing implementation: {path}")
            elif "distill" in lowered:
                plan.add_lanes("distillation", reason=f"distillation implementation: {path}")
            elif "lora" in lowered:
                plan.add_lanes("lora-training", reason=f"LoRA training implementation: {path}")
            else:
                plan.add_lanes(*LEGACY_TRAINING_LANES, reason=f"shared legacy training implementation: {path}")
            continue

        lowered = path.lower()
        if "vmoba" in lowered and path.startswith(("fastvideo/", ".buildkite/")):
            plan.add_lanes("inference-vmoba", reason=f"VMoBA implementation: {path}")
            plan.add_golden(("test_wan_t2v.py", ), reason=f"VMoBA end-to-end coverage: {path}")
            continue
        if "lora" in lowered and path.startswith("fastvideo/"):
            plan.add_lanes(
                "lora-inference",
                "lora-extraction",
                "lora-training",
                reason=f"shared LoRA implementation: {path}",
            )
            _select_output_coverage(plan, path)
            continue

        if path.startswith("fastvideo/entrypoints/") or path.startswith("fastvideo/api/"):
            plan.add_lanes("api-server", reason=f"API/entrypoint integration: {path}")
            if "openai" not in lowered and "/cli/" not in lowered:
                _select_output_coverage(plan, path)
            continue
        if path.startswith("fastvideo/worker/"):
            plan.add_lanes("api-server", reason=f"worker/API integration: {path}")
            _select_output_coverage(plan, path)
            continue
        if path.startswith("fastvideo/distributed/"):
            plan.add_lanes(
                "training",
                "train-framework",
                reason=f"distributed runtime integration: {path}",
            )
            _select_output_coverage(plan, path)
            continue
        if path.startswith(("fastvideo/hooks/", "fastvideo/platforms/", "fastvideo/third_party/")):
            _select_output_coverage(plan, path)
            continue
        if path.startswith(("fastvideo/models/", "fastvideo/pipelines/", "fastvideo/configs/",
                            "fastvideo/layers/", "fastvideo/attention/")):
            _select_output_coverage(plan, path)
            continue
        if path in {
                "fastvideo/fastvideo_args.py",
                "fastvideo/forward_context.py",
                "fastvideo/image_processor.py",
                "fastvideo/registry.py",
                "fastvideo/utils.py",
        }:
            _select_output_coverage(plan, path)
            continue
        if path.startswith("fastvideo/mlx_runtime/"):
            plan.reasons.append(f"covered by the path-filtered macOS MLX workflow: {path}")
            continue
        if path.startswith("fastvideo/logging_utils/") or path in {
                "fastvideo/__init__.py",
                "fastvideo/envs.py",
                "fastvideo/logger.py",
                "fastvideo/profiler.py",
                "fastvideo/version.py",
        }:
            plan.reasons.append(f"covered by automatic Fastcheck: {path}")
            continue
        if path.startswith(("fastvideo-kernel/", "csrc/")):
            plan.add_golden(("test_wan_t2v.py", ), reason=f"kernel integration smoke: {path}")
            plan.add_ssim(("test_wan_t2v_similarity.py", ), reason=f"kernel numerical smoke: {path}")
            continue

        if path.startswith("apps/dreamverse/"):
            # DreamVerse is already one of the six automatic Fastcheck lanes.
            plan.reasons.append(f"covered by automatic DreamVerse Fastcheck: {path}")
            continue
        if path.startswith("fastvideo/tests/"):
            # The automatic unit/component Fastcheck lanes own the remaining
            # package tests. Domain-specific expensive test roots were handled
            # above.
            plan.reasons.append(f"covered by automatic Fastcheck: {path}")
            continue
        if path in {".buildkite/scripts/unit_test.sh", ".buildkite/scripts/pr_test.sh"}:
            plan.reasons.append(f"covered by automatic unit Fastcheck: {path}")
            continue
        if _matches_any(path, SAFE_PATTERNS):
            plan.reasons.append(f"no additional GPU integration needed: {path}")
            continue

        plan.require_all(f"unclassified path; failing closed: {path}")

    return plan


def _write_github_output(output: TextIO, plan: MergePlan) -> None:
    output.write(f"merge_test_plan={plan.encoded_lanes()}\n")
    output.write(f"merge_golden_tests={plan.encoded_golden_tests()}\n")
    output.write(f"merge_ssim_tests={plan.encoded_ssim_tests()}\n")
    output.write(f"merge_plan_label={','.join(plan.ordered_lanes()) or 'none'}\n")


def _write_summary(output: TextIO, plan: MergePlan) -> None:
    output.write("## Change-aware merge test plan\n\n")
    output.write("| Selection | Value |\n|---|---|\n")
    output.write(f"| Additional Slurm lanes | `{','.join(plan.ordered_lanes()) or 'none'}` |\n")
    output.write(f"| Golden tests | `{plan.encoded_golden_tests()}` |\n")
    output.write(f"| SSIM tests | `{plan.encoded_ssim_tests()}` |\n\n")
    output.write("Fastcheck remains the universal six-lane baseline.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-file", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--summary-file", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = args.paths_file.read_text(encoding="utf-8").splitlines()
    plan = classify_paths(paths)
    print(f"MERGE_TEST_PLAN={plan.encoded_lanes()}")
    print(f"MERGE_GOLDEN_TESTS={plan.encoded_golden_tests()}")
    print(f"MERGE_SSIM_TESTS={plan.encoded_ssim_tests()}")
    for reason in plan.reasons:
        print(f"- {reason}")
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as output:
            _write_github_output(output, plan)
    if args.summary_file:
        with args.summary_file.open("a", encoding="utf-8") as output:
            _write_summary(output, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
