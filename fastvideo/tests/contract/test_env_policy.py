# SPDX-License-Identifier: Apache-2.0
"""Guard: FastVideo code follows the environment-variable policy in
docs/contributing/env_vars.md.

The test parses every Python file under fastvideo/ (except fastvideo/third_party/
and the registry fastvideo/envs.py) with ``ast`` and reports code that:

- reads the environment directly for a name outside EXTERNAL_ALLOWLIST (test
  code under fastvideo/tests/ may also read CI_ONLY_VARIABLES);
- writes the environment directly (os.environ, os.putenv, monkeypatch.setenv),
  or through the envs.*_external helpers for a name outside
  EXTERNAL_WRITE_ALLOWLIST;
- uses the whole environment (os.environ.copy(), dict(os.environ), patch.dict);
- uses a registry field without calling one of its methods (``envs.X == "a"``,
  ``getter = envs.X.get``);
- calls ``envs.X.get()`` outside a function, so the value is read at import.

It also checks every entry in fastvideo/envs.py (FASTVIDEO_ prefix, category,
description, at least one reader) and checks that the table in
docs/contributing/env_vars.md matches the registry.

Violations that existed when the policy was introduced are listed in
KNOWN_VIOLATIONS. The list only shrinks: a violation that is not listed fails,
and a listed violation that no longer exists also fails, so the entry gets
deleted.

Run ``python fastvideo/tests/contract/test_env_policy.py`` to regenerate the
table in docs/contributing/env_vars.md after editing fastvideo/envs.py.

Static analysis only: fastvideo/envs.py is loaded as a standalone file, so the
test imports neither fastvideo nor torch.
"""
import ast
import importlib.util
import re
from collections import Counter
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "fastvideo"
REGISTRY_PATH = PACKAGE_ROOT / "envs.py"
DOC_PATH = REPO_ROOT / "docs" / "contributing" / "env_vars.md"
POLICY_DOC = "docs/contributing/env_vars.md"
EXCLUDED_DIRS = (PACKAGE_ROOT / "third_party", )
TESTS_ROOT = PACKAGE_ROOT / "tests"

DOC_TABLE_BEGIN = "<!-- BEGIN GENERATED ENV TABLE: python fastvideo/tests/contract/test_env_policy.py -->"
DOC_TABLE_END = "<!-- END GENERATED ENV TABLE -->"

# Variables that other tools own. Code may read them directly with a literal
# name. A trailing "*" allows every name with that prefix.
EXTERNAL_ALLOWLIST = {
    "CUDA_*": "CUDA runtime and device selection.",
    "NCCL_*": "NCCL communication library.",
    "TORCH_*": "PyTorch runtime settings.",
    "OMP_*": "OpenMP threading.",
    "HF_HUB_*": "huggingface_hub settings.",
    "RANK": "Set by torchrun and other launchers.",
    "LOCAL_RANK": "Set by torchrun and other launchers.",
    "WORLD_SIZE": "Set by torchrun and other launchers.",
    "LOCAL_WORLD_SIZE": "Set by torchrun and other launchers.",
    "SLURM_*": "Set by Slurm for each srun task; the external-launcher executor reads the task identity.",
    "MASTER_ADDR": "Set by torchrun and other launchers.",
    "MASTER_PORT": "Set by torchrun and other launchers.",
    "HOME": "User home directory.",
    "PATH": "Executable search path.",
    "HF_TOKEN": "Token variable that huggingface_hub reads.",
    "HUGGING_FACE_HUB_TOKEN": "Token variable that huggingface_hub reads.",
    "GEMINI_API_KEY": "Gemini API key for the judge metrics.",
    "GOOGLE_API_KEY": "Google API key, which the Gemini SDK also accepts.",
    "CEREBRAS_API_KEY": "Cerebras API key for the streaming prompt provider.",
    "GROQ_API_KEY": "Groq API key for the streaming prompt provider.",
    "RAY_USAGE_STATS_ENABLED": "Ray usage-statistics switch.",
    "BUILDKITE_*": "Set by the Buildkite agent in each CI job.",
    "GITHUB_STEP_SUMMARY": "Set by GitHub Actions.",
    "WANDB_MODE": "wandb setting.",
    "WANDB_API_KEY": "wandb API key.",
    "UV_TORCH_BACKEND": "uv setting that selects the PyTorch build.",
    "CFLAGS": "C compiler flags.",
    "CXXFLAGS": "C++ compiler flags.",
    "LDFLAGS": "Linker flags.",
    "CMAKE_ARGS": "Extra CMake arguments.",
    "CC": "C compiler.",
    "CXX": "C++ compiler.",
    "LD": "Linker.",
    "CUDACXX": "CUDA compiler.",
    "CUBLAS_WORKSPACE_CONFIG": "cuBLAS workspace setting.",
    "TORCHDYNAMO_DISABLE": "PyTorch Dynamo switch.",
    "PYTORCH_MPS_*": "PyTorch MPS memory settings.",
    "USER": "Login name of the current user.",
    "PYTHONPATH": "Python module search path.",
    "FASTVIDEO_VSA_CUTEDSL": "Read by fastvideo-kernel, which is outside this policy.",
    "FASTVIDEO_VSA_TRITON": "Read by fastvideo-kernel, which is outside this policy.",
    "FASTVIDEO_KERNEL_VSA_FORCE_TRITON": "Read by fastvideo-kernel, which is outside this policy.",
}

# Variables that FastVideo's own CI and CI tooling define. They keep their names
# and test code under fastvideo/tests/ reads them directly; the value names what sets them.
CI_ONLY_VARIABLES = {
    "TEST_SCOPE": ".github/workflows/ci-slash-commands.yml, ci-trigger-full-suite.yml, ci-scheduled-ssim.yml",
    "PERF_RUN_SOURCE": ".buildkite/scripts/lanes/performance.sh",
    "PERF_UPLOAD_POLICY": ".buildkite/scripts/lanes/performance.sh",
    "PERF_REPORTS_DIR": ".buildkite/scripts/lanes/performance.sh",
    "PERF_PYTEST_RC": ".buildkite/scripts/lanes/performance.sh",
    "IMAGE_VERSION": ".buildkite/pipeline.yml",
    "PERFORMANCE_TRACKING_ROOT": ".buildkite/scripts/lanes/performance.sh",
    "HF_REPO_ID": "Performance tracking tooling (fastvideo/tests/modal/pr_test.py).",
    "PERFORMANCE_RESEED_STAGING_ROOT": "Performance reseed tooling (.agents/skills/reseed-performance-baseline).",
    "DASHBOARD_DAYS": "Performance dashboard tooling (fastvideo/tests/performance/dashboard.py).",
    "GPU_BACKEND": "fastvideo-kernel/CMakeLists.txt (kernel build).",
    "FASTVIDEO_PERFORMANCE_PROFILE_VERSION": "fastvideo/tests/modal/launch_l40s_job.py; recorded in performance identity.",
    "FASTVIDEO_CONTAINER_IMAGE_REF": "Modal CI launchers; recorded in performance identity.",
    "FASTVIDEO_SSIM_MODEL_ID": "fastvideo/tests/ssim/ci_runner.py (SSIM CI scheduler).",
    "FASTVIDEO_MODAL_IMAGE": "Modal launch tooling in fastvideo/tests/modal/.",
    "FASTVIDEO_MODAL_VOLUME": "Modal launch tooling in fastvideo/tests/modal/.",
    "FASTVIDEO_KERNEL_CACHE_ROOT": "Kernel build-cache tooling (fastvideo/tests/modal/kernel_build_cache.py).",
    "FASTVIDEO_KERNEL_PREBUILT_INFO": "Kernel build-cache tooling (fastvideo/tests/modal/kernel_build_cache.py).",
    "FASTVIDEO_SSIM_BOOTSTRAP_MODE": ".buildkite/scripts/lanes/ssim.sh; --ssim-bootstrap-mode in fastvideo/tests/ssim/conftest.py.",
}

# Variables that FastVideo sets for other tools. Code writes them only through
# envs.set_external, envs.setdefault_external, or envs.unset_external, with a
# literal name.
EXTERNAL_WRITE_ALLOWLIST = {
    "RANK": "torch.distributed rendezvous for single-process runs and workers.",
    "LOCAL_RANK": "torch.distributed rendezvous for single-process runs and workers.",
    "WORLD_SIZE": "torch.distributed rendezvous for single-process runs and workers.",
    "LOCAL_WORLD_SIZE": "The external-launcher executor maps Slurm's task layout to torchrun's name.",
    "MASTER_ADDR": "torch.distributed rendezvous for single-process runs.",
    "MASTER_PORT": "torch.distributed rendezvous for single-process runs.",
    "CUDA_VISIBLE_DEVICES": "Pins a streaming worker process to its GPU.",
    "TORCH_NCCL_AVOID_RECORD_STREAMS": "Avoids NCCL record-stream memory growth in workers.",
    "NCCL_ASYNC_ERROR_HANDLING": "Unset in workers because the value that Ray sets breaks graph building.",
    "TORCH_HOME": "Points torch.hub at the evaluation cache.",
    "HF_TOKEN": "Passes the resolved Hugging Face token to huggingface_hub.",
    "VIDEO_MAX_PIXELS": "Pixel budget that qwen_vl_utils reads for Qwen2.5-Omni.",
    "RAY_USAGE_STATS_ENABLED": "Turns off Ray usage statistics unless the user turned them on.",
    "CUTE_DSL_ENABLE_TVM_FFI": "CUTLASS DSL switch that the NVFP4 FlashAttention-4 path needs.",
    "CUBLAS_WORKSPACE_CONFIG": "cuBLAS workspace size; golden-gate tests set it before cuBLAS initializes.",
}
EXTERNAL_WRITE_HELPERS = {"set_external", "setdefault_external", "unset_external", "override_external"}

# Methods that registry fields expose; ``get`` and ``is_set`` count as reads.
REGISTRY_METHODS = {"get", "set", "override", "is_set", "clear"}
REGISTRY_READ_METHODS = {"get", "is_set"}

# Violations that existed when the policy was introduced, as
# "<path>: <kind> <name>" -> number of occurrences. Lower or delete an entry when
# its violations are fixed; never add one. Kinds are described in
# docs/contributing/env_vars.md.
KNOWN_VIOLATIONS: dict[str, int] = {
    'fastvideo/attention/utils/flash_attn_default.py: import-time-read FASTVIDEO_FA4': 1,
    'fastvideo/benchmarks/eval_metalfx_rife.py: write FASTVIDEO_MLX_COMPILE': 1,
    'fastvideo/benchmarks/mlx_fastwan_bench.py: write FASTVIDEO_MLX_COMPILE': 1,
    'fastvideo/distributed/device_communicators/cpu_communicator.py: read VLLM_DIST_IDENT': 1,
    'fastvideo/entrypoints/cli/utils.py: whole-environ': 1,
    'fastvideo/entrypoints/openai/api_server.py: write FASTVIDEO_STAGE_LOGGING': 1,
    'fastvideo/entrypoints/video_generator.py: write FASTVIDEO_NVFP4_FA4': 1,
    'fastvideo/mlx_runtime/memory.py: write <dynamic>': 1,
    'fastvideo/performance/hf_store.py: read HF_REPO_ID': 1,
    'fastvideo/performance/hf_store.py: read PERFORMANCE_TRACKING_SYNC_REUSE_TTL_SECONDS': 1,
    'fastvideo/performance_dashboard/api.py: read PERFORMANCE_TRACKING_ROOT': 1,
    'fastvideo/tests/contract/test_profiler_regions.py: whole-environ': 2,
    'fastvideo/tests/contract/test_wan_validation_order.py: whole-environ': 1,
    'fastvideo/tests/distributed/test_sp_hunyuanvideo.py: whole-environ': 1,
    'fastvideo/tests/distributed/test_sp_lingbot_video.py: whole-environ': 1,
    'fastvideo/tests/distributed/test_sp_ltx2.py: whole-environ': 1,
    'fastvideo/tests/distributed/test_sp_wan.py: whole-environ': 1,
    'fastvideo/tests/distributed/test_ulysses_a2a_parity.py: whole-environ': 1,
    'fastvideo/tests/distributed/test_ulysses_fault_injection.py: whole-environ': 1,
    'fastvideo/tests/entrypoints/test_openai_api_integration.py: whole-environ': 1,
    'fastvideo/tests/entrypoints/test_openai_video_client.py: whole-environ': 1,
    'fastvideo/tests/inference/test_basic_fasth3_profile.py: write <dynamic>': 2,
    'fastvideo/tests/layers/test_rmsnorm_forward_dispatch.py: whole-environ': 1,
    'fastvideo/tests/modal/kernel_build_cache.py: whole-environ': 1,
    'fastvideo/tests/modal/launch_l40s_job.py: read <dynamic>': 2,
    'fastvideo/tests/modal/launch_l40s_job.py: read FASTVIDEO_ATTENTION_BACKEND': 1,
    'fastvideo/tests/modal/launch_l40s_job.py: read FASTVIDEO_FA4': 1,
    'fastvideo/tests/modal/launch_l40s_job.py: whole-environ': 1,
    'fastvideo/tests/modal/pr_test.py: read <dynamic>': 2,
    'fastvideo/tests/modal/pr_test.py: read FASTVIDEO_FA4': 1,
    'fastvideo/tests/modal/pr_test.py: read HF_API_KEY': 1,
    'fastvideo/tests/modal/ssim_test.py: read <dynamic>': 1,
    'fastvideo/tests/modal/ssim_test.py: read FASTVIDEO_FA4': 1,
    'fastvideo/tests/modal/ssim_test.py: whole-environ': 2,
    'fastvideo/tests/nightly/test_e2e_kandinsky5_dmd_t2v_overfit.py: whole-environ': 3,
    'fastvideo/tests/nightly/test_e2e_ltx2_overfit_new_stack.py: whole-environ': 2,
    'fastvideo/tests/ssim/ci_runner.py: whole-environ': 1,
    'fastvideo/tests/ssim/reference_videos_cli.py: read <dynamic>': 1,
    'fastvideo/tests/train/methods/test_minimax_h3_finetune.py: whole-environ': 2,
    'fastvideo/tests/train/models/test_load_kandinsky5.py: write FASTVIDEO_ATTENTION_BACKEND': 1,
    'fastvideo/tests/training/distill/test_anyflow_smoke.py: whole-environ': 2,
    'fastvideo/train/entrypoint/train.py: write FASTVIDEO_ATTENTION_BACKEND': 2,
    'fastvideo/utils.py: read <dynamic>': 5,
    'fastvideo/utils.py: write <dynamic>': 1,
    'fastvideo/worker/ray_distributed_executor.py: read <dynamic>': 2,
    'fastvideo/worker/ray_distributed_executor.py: whole-environ': 1,
    'fastvideo/worker/ray_env.py: read <dynamic>': 1,
    'fastvideo/worker/ray_utils.py: whole-environ': 1,
    'fastvideo/worker/worker_base.py: read <dynamic>': 1,
    'fastvideo/worker/worker_base.py: write <dynamic>': 1,
}


def load_registry() -> ModuleType:
    """Load fastvideo/envs.py as a standalone module, without importing the fastvideo package."""
    spec = importlib.util.spec_from_file_location("_fastvideo_envs_registry", REGISTRY_PATH)
    assert spec is not None and spec.loader is not None
    registry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(registry)
    return registry


def is_external_write_allowed(helper: str, name: str, in_tests: bool) -> bool:
    """Whether an envs.*_external call may write ``name``.

    override_external restores the previous value, so test code under fastvideo/tests/
    may use it for any variable that code reads directly; library code and the other
    helpers need the write allowlist.
    """
    if name in EXTERNAL_WRITE_ALLOWLIST:
        return True
    return in_tests and helper == "override_external" and (is_allowlisted(name) or name in CI_ONLY_VARIABLES)


def is_allowlisted(name: str) -> bool:
    for pattern in EXTERNAL_ALLOWLIST:
        if pattern.endswith("*") and name.startswith(pattern[:-1]):
            return True
        if name == pattern:
            return True
    return False


class EnvAccessScanner:
    """Find environment accesses and registry-field uses in one module.

    ``violations`` holds (kind, name, line) tuples; ``registry_reads`` counts
    ``envs.X.get()`` and ``envs.X.is_set()`` calls by variable name.
    """

    def __init__(self, tree: ast.Module, registry_names: set[str], in_tests: bool) -> None:
        self.registry_names = registry_names
        self.in_tests = in_tests
        self.violations: list[tuple[str, str, int]] = []
        self.registry_reads: Counter[str] = Counter()
        self.parents: dict[ast.AST, ast.AST] = {}
        self._collect_aliases(tree)
        self._visit(tree, in_function=False)

    def _collect_aliases(self, tree: ast.Module) -> None:
        """Record the local names bound to os, os.environ, os.getenv-like functions, and fastvideo.envs."""
        self.os_names: set[str] = set()
        self.environ_names: set[str] = set()
        self.getenv_names: set[str] = set()
        self.setenv_names: set[str] = set()
        self.envs_module_names: set[str] = set()
        # Module-level string constants, so that os.environ.get(NAME_ENV) resolves to the name.
        self.constants: dict[str, str] = {}
        for statement in tree.body:
            if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name) and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)):
                self.constants[statement.targets[0].id] = statement.value.value
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "os" or alias.name.startswith("os."):
                        self.os_names.add(alias.asname or "os")
                    if alias.name == "fastvideo.envs" and alias.asname:
                        self.envs_module_names.add(alias.asname)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    local = alias.asname or alias.name
                    if node.module == "os":
                        if alias.name in ("environ", "environb"):
                            self.environ_names.add(local)
                        elif alias.name in ("getenv", "getenvb"):
                            self.getenv_names.add(local)
                        elif alias.name in ("putenv", "unsetenv"):
                            self.setenv_names.add(local)
                    elif node.module == "fastvideo" and alias.name == "envs":
                        self.envs_module_names.add(local)

    def _literal(self, node: ast.AST | None) -> str:
        """Return the variable name that a call argument spells, or "<dynamic>"."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name) and node.id in self.constants:
            return self.constants[node.id]
        return "<dynamic>"

    def _is_os_attr(self, node: ast.AST, attrs: tuple[str, ...]) -> bool:
        return (isinstance(node, ast.Attribute) and node.attr in attrs and isinstance(node.value, ast.Name)
                and node.value.id in self.os_names)

    def _is_environ(self, node: ast.AST) -> bool:
        return self._is_os_attr(node, ("environ", "environb")) or (isinstance(node, ast.Name)
                                                                    and node.id in self.environ_names)

    def _is_envs_module(self, node: ast.AST) -> bool:
        return ((isinstance(node, ast.Name) and node.id in self.envs_module_names)
                or (isinstance(node, ast.Attribute) and node.attr == "envs"))

    def _visit(self, node: ast.AST, in_function: bool) -> None:
        """Walk the tree, tracking whether each node runs inside a function body."""
        self._check(node, in_function)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            # Decorators, defaults, and annotations run when the function is defined.
            outside = [*getattr(node, "decorator_list", []), *node.args.defaults, *node.args.kw_defaults]
            for child in outside:
                if child is not None:
                    self.parents[child] = node
                    self._visit(child, in_function)
            body = node.body if isinstance(node.body, list) else [node.body]
            for child in body:
                self.parents[child] = node
                self._visit(child, True)
            return
        for child in ast.iter_child_nodes(node):
            self.parents[child] = node
            self._visit(child, in_function)

    def _check(self, node: ast.AST, in_function: bool) -> None:
        parent = self.parents.get(node)
        if self._is_environ(node):
            self._check_environ_use(node, parent)
        elif isinstance(node, ast.Call):
            self._check_call(node)
        elif isinstance(node, ast.Attribute) and node.attr in self.registry_names and self._is_envs_module(node.value):
            self._check_registry_use(node, parent, in_function)

    def _check_environ_use(self, node: ast.AST, parent: ast.AST | None) -> None:
        line = node.lineno  # type: ignore[attr-defined]
        if isinstance(parent, ast.Attribute) and isinstance(self.parents.get(parent), ast.Call):
            call = self.parents[parent]
            assert isinstance(call, ast.Call)
            if call.func is parent and parent.attr == "get":
                self.violations.append(("read", self._literal(call.args[0] if call.args else None), line))
                return
            if call.func is parent and parent.attr in ("setdefault", "pop"):
                self.violations.append(("write", self._literal(call.args[0] if call.args else None), line))
                return
        if isinstance(parent, ast.Subscript) and parent.value is node:
            kind = "read" if isinstance(parent.ctx, ast.Load) else "write"
            self.violations.append((kind, self._literal(parent.slice), line))
            return
        if (isinstance(parent, ast.Compare) and len(parent.ops) == 1 and isinstance(parent.ops[0], (ast.In, ast.NotIn))
                and parent.comparators[0] is node):
            self.violations.append(("read", self._literal(parent.left), line))
            return
        self.violations.append(("whole-environ", "", line))

    def _check_call(self, node: ast.Call) -> None:
        func = node.func
        name = self._literal(node.args[0] if node.args else None)
        if self._is_os_attr(func, ("getenv", "getenvb")) or (isinstance(func, ast.Name)
                                                               and func.id in self.getenv_names):
            self.violations.append(("read", name, node.lineno))
        elif (self._is_os_attr(func, ("putenv", "unsetenv"))
              or (isinstance(func, ast.Name) and func.id in self.setenv_names)
              or (isinstance(func, ast.Attribute) and func.attr in ("setenv", "delenv"))):
            self.violations.append(("write", name, node.lineno))
        elif (isinstance(func, ast.Attribute) and func.attr in EXTERNAL_WRITE_HELPERS
              and self._is_envs_module(func.value) and not is_external_write_allowed(func.attr, name, self.in_tests)):
            self.violations.append(("write", name, node.lineno))

    def _check_registry_use(self, node: ast.Attribute, parent: ast.AST | None, in_function: bool) -> None:
        # Only a call counts: ``getter = envs.X.get`` neither reads the variable
        # nor uses the field through a method.
        grandparent = self.parents.get(parent) if parent is not None else None
        if not (isinstance(parent, ast.Attribute) and parent.attr in REGISTRY_METHODS
                and isinstance(grandparent, ast.Call) and grandparent.func is parent):
            self.violations.append(("bare-field", node.attr, node.lineno))
            return
        if parent.attr in REGISTRY_READ_METHODS:
            self.registry_reads[node.attr] += 1
            if not in_function:
                self.violations.append(("import-time-read", node.attr, node.lineno))


def scanned_files() -> list[Path]:
    return sorted(path for path in PACKAGE_ROOT.rglob("*.py")
                  if path != REGISTRY_PATH and not any(excluded in path.parents for excluded in EXCLUDED_DIRS))


def registry_internal_reads(registry_names: set[str]) -> Counter[str]:
    """Count registry reads inside fastvideo/envs.py itself, such as a computed default that reads another field."""
    reads: Counter[str] = Counter()
    for node in ast.walk(ast.parse(REGISTRY_PATH.read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Attribute) and node.attr in REGISTRY_READ_METHODS and isinstance(node.value, ast.Name)
                and node.value.id in registry_names):
            reads[node.value.id] += 1
    return reads


def collect_violations(registry: ModuleType) -> tuple[dict[str, list[int]], Counter[str]]:
    """Return every violation key with its line numbers, plus registry read counts."""
    registry_names = set(registry.environment_variables)
    found: dict[str, list[int]] = {}
    reads: Counter[str] = registry_internal_reads(registry_names)
    for path in scanned_files():
        relative = path.relative_to(REPO_ROOT).as_posix()
        in_tests = TESTS_ROOT in path.parents
        scanner = EnvAccessScanner(ast.parse(path.read_text(encoding="utf-8"), filename=relative), registry_names,
                                   in_tests)
        reads.update(scanner.registry_reads)
        for kind, name, line in scanner.violations:
            if kind == "read" and (is_allowlisted(name) or (in_tests and name in CI_ONLY_VARIABLES)):
                continue
            key = f"{relative}: {kind} {name}".rstrip()
            found.setdefault(key, []).append(line)

    registry_path = REGISTRY_PATH.relative_to(REPO_ROOT).as_posix()
    for name in registry.environment_variables:
        if not re.fullmatch(r"FASTVIDEO_[A-Z0-9_]+", name):
            found.setdefault(f"{registry_path}: prefix {name}", []).append(0)
        if reads[name] == 0:
            found.setdefault(f"{registry_path}: unread {name}", []).append(0)
    return found, reads


def test_env_access_follows_policy():
    found, _ = collect_violations(load_registry())
    counts = Counter({key: len(lines) for key, lines in found.items()})
    known = Counter(KNOWN_VIOLATIONS)

    new = counts - known
    fixed = known - counts
    messages = []
    if new:
        lines = [f"  {key}  (lines {found[key]})" for key in sorted(new)]
        messages.append(f"New environment-variable policy violations. See {POLICY_DOC} for the rule and the fix:\n" +
                        "\n".join(lines))
    if fixed:
        lines = [f"  {key}" for key in sorted(fixed)]
        messages.append("These KNOWN_VIOLATIONS entries are fixed; delete them from "
                        "fastvideo/tests/contract/test_env_policy.py:\n" + "\n".join(lines))
    assert not messages, "\n\n".join(messages)


def test_registry_method_counts_only_when_called():
    """An uncalled ``envs.NAME.get`` is a bare field, not a read."""
    source = ("import fastvideo.envs as envs\n"
              "def f():\n"
              "    getter = envs.FASTVIDEO_FA4.get\n"
              "    return envs.FASTVIDEO_FA4.get()\n")
    scanner = EnvAccessScanner(ast.parse(source), {"FASTVIDEO_FA4"}, in_tests=False)
    assert scanner.violations == [("bare-field", "FASTVIDEO_FA4", 3)]
    assert scanner.registry_reads["FASTVIDEO_FA4"] == 1


def test_registry_entries_have_category_and_description():
    registry = load_registry()
    problems = []
    for name, field in registry.environment_variables.items():
        if field.category not in registry.CATEGORIES:
            problems.append(f"{name}: category {field.category!r} is not in envs.CATEGORIES")
        if not field.doc.strip():
            problems.append(f"{name}: empty description")
    assert not problems, f"Registry entries break {POLICY_DOC}:\n" + "\n".join(problems)


def _render_default(field) -> str:
    if callable(field.default):
        return "computed"
    if field.default is None:
        return "unset"
    return f"`{field.format(field.default)}`" if field.format(field.default) else '`""`'


def _escape(text: str) -> str:
    return text.replace("|", "\\|").replace("*", "\\*").replace("<", "&lt;").replace(">", "&gt;")


def render_env_table(registry: ModuleType) -> str:
    """Render the registry and the deprecation table as aligned Markdown tables, in declaration order."""
    rows = [["Variable", "Type", "Default", "Category", "Description"]]
    for name, field in registry.environment_variables.items():
        doc = _escape(field.doc)
        if field.deprecated_names:
            doc += " Deprecated names: " + ", ".join(f"`{old}`" for old in field.deprecated_names) + "."
        rows.append([f"`{name}`", field.type_name, _render_default(field), field.category, doc])
    deprecated_rows = [["Deprecated variable", "Reason"]]
    deprecated_rows.extend([f"`{name}`", reason] for name, reason in registry.DEPRECATED_VARIABLES.items())
    return (_render_table(rows) + "\n\nVariables that FastVideo no longer reads; setting one logs a warning:\n\n" +
            _render_table(deprecated_rows))


def _render_table(rows: list[list[str]]) -> str:
    """Render rows as a Markdown table whose pipes line up."""
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]

    def render_row(cells: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(width) for cell, width in zip(cells, widths)) + " |"

    lines = [render_row(rows[0]), render_row(["-" * width for width in widths])]
    lines.extend(render_row(row) for row in rows[1:])
    return "\n".join(lines)


def _split_doc(text: str) -> tuple[str, str, str]:
    """Split the policy doc into the text before the generated table, the table, and the text after it."""
    before, found_begin, rest = text.partition(DOC_TABLE_BEGIN)
    table, found_end, after = rest.partition(DOC_TABLE_END)
    assert found_begin and found_end, f"{POLICY_DOC} must contain the generated-table markers"
    return before + found_begin + "\n", table.strip("\n"), "\n" + found_end + after


def test_env_doc_table_matches_registry():
    _, table, _ = _split_doc(DOC_PATH.read_text(encoding="utf-8"))
    assert table == render_env_table(load_registry()), (
        f"The table in {POLICY_DOC} is out of date. Run `python fastvideo/tests/contract/test_env_policy.py`.")


if __name__ == "__main__":
    head, _, tail = _split_doc(DOC_PATH.read_text(encoding="utf-8"))
    DOC_PATH.write_text(head + render_env_table(load_registry()) + tail, encoding="utf-8")
    print(f"Updated {POLICY_DOC}")
