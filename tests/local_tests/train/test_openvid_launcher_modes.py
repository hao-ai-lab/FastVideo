from __future__ import annotations

import ast
import importlib.util
import os
import py_compile
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
if (HERE / "prepare_openvid_a12_a15.py").is_file():
    LAUNCHER_ROOT = HERE
    PREPARE = HERE / "prepare_openvid_a12_a15.py"
else:
    REPO_ROOT = next(
        parent
        for parent in HERE.parents
        if (parent / "scripts/train/prepare_openvid_a12_a15.py").is_file()
    )
    LAUNCHER_ROOT = REPO_ROOT / "scripts/train/openvid_causal_a12_a15"
    PREPARE = REPO_ROOT / "scripts/train/prepare_openvid_a12_a15.py"
TRAIN_STAGE = LAUNCHER_ROOT / "train_stage.sh"
RUN_CONDITION = LAUNCHER_ROOT / "run_condition.sh"
RUN_ALL = LAUNCHER_ROOT / "run_all_sequential.sh"
BASH = shutil.which("bash")


pytestmark = pytest.mark.skipif(BASH is None, reason="bash is required")


def test_all_stages_are_locked_to_21_latents_and_81_frames() -> None:
    spec = importlib.util.spec_from_file_location("openvid_prepare", PREPARE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for condition, condition_spec in module.CONDITIONS.items():
        generated = module.configs(
            f"/tmp/{condition}", condition, condition_spec, "/repo"
        )
        assert set(generated) == {"tf", "cd", "sf"}
        for stage, config in generated.items():
            data = config["training"]["data"]
            assert data["num_latent_t"] == 21, (condition, stage)
            assert data["num_frames"] == 81, (condition, stage)
            assert config["callbacks"]["validation"]["num_frames"] == 81, (
                condition,
                stage,
            )


def _bash_path(path: Path) -> str:
    """Return a path understood by bash.exe/WSL as well as native Linux bash."""
    path = path.resolve()
    if os.name != "nt":
        return str(path)
    posix = path.as_posix()
    drive, suffix = posix.split(":", 1)
    return f"/mnt/{drive.lower()}{suffix}"


def _run_script(
    script: Path,
    *args: str | Path,
    env: dict[str, str | Path] | None = None,
    unset: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    command = ["env"]
    for name in unset:
        command.extend(["-u", name])
    for name, value in (env or {}).items():
        if isinstance(value, Path):
            value = _bash_path(value)
        command.append(f"{name}={value}")
    command.extend(["bash", _bash_path(script)])
    command.extend(_bash_path(arg) if isinstance(arg, Path) else str(arg) for arg in args)
    return subprocess.run(
        [BASH, "-lc", shlex.join(command)],
        capture_output=True,
        text=True,
        check=False,
    )


def _write_shell(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    path.chmod(0o755)


def _stage_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, dict[str, str | Path]]:
    run_root = tmp_path / "run"
    stage_dir = run_root / "tf"
    for relative in ("config", "state", "logs", "checkpoints", "validation", "tracker"):
        (stage_dir / relative).mkdir(parents=True, exist_ok=True)
    (stage_dir / "config" / "run.yaml").write_text("training: {}\n", encoding="utf-8")

    repo = tmp_path / "repo"
    repo.mkdir()
    env_dir = tmp_path / "env"
    call_log = tmp_path / "python-calls.log"
    fake_bin = tmp_path / "fake-bin"

    _write_shell(fake_bin / "git", "#!/usr/bin/env bash\nexit 0\n")
    # train_stage has fixed Lustre cache roots.  A no-op mkdir keeps this unit
    # test hermetic; all stage directories needed by the test are pre-created.
    _write_shell(fake_bin / "mkdir", "#!/usr/bin/env bash\nexit 0\n")
    _write_shell(
        env_dir / "bin" / "python",
        r'''#!/usr/bin/env bash
set -euo pipefail
printf 'argv=' >> "$CALL_LOG"
printf '%q ' "$@" >> "$CALL_LOG"
printf '\n' >> "$CALL_LOG"
if [[ "${1:-}" == - ]]; then
  cat >/dev/null
  echo "OpenVid streaming config preflight passed: ${2:-missing}"
  exit 0
fi
if [[ "$*" == *wandb.util.generate_id* ]]; then
  echo generate_id >> "$CALL_LOG"
  echo generated-id
  exit 0
fi
if [[ "$*" == *torch.distributed.run* ]]; then
  printf 'torch mode=%s api=%s run_id=%s resume=%s\n' \
    "${WANDB_MODE-<unset>}" "${WANDB_API_KEY-<unset>}" \
    "${WANDB_RUN_ID-<unset>}" "${WANDB_RESUME-<unset>}" >> "$CALL_LOG"
  exit 0
fi
echo "unexpected fake-python invocation: $*" >&2
exit 91
''',
    )
    common_env: dict[str, str | Path] = {
        "RUN_ROOT": run_root,
        "STAGE": "tf",
        "MASTER_PORT": "29991",
        "CONDITION": "A12",
        "REPO": repo,
        "ENV_DIR": env_dir,
        "CALL_LOG": call_log,
        "PATH": f"{_bash_path(fake_bin)}:/usr/bin:/bin",
    }
    return run_root, stage_dir, call_log, fake_bin, common_env


@pytest.mark.parametrize("entrypoint", ["stage", "condition", "all"])
def test_online_without_key_fails_before_state_or_children(tmp_path: Path, entrypoint: str) -> None:
    root = tmp_path / entrypoint
    if entrypoint == "stage":
        result = _run_script(
            TRAIN_STAGE,
            env={
                "RUN_ROOT": root,
                "STAGE": "tf",
                "MASTER_PORT": "29991",
                "WANDB_MODE": "online",
                "PREFLIGHT_ONLY": "0",
            },
            unset=("WANDB_API_KEY",),
        )
    elif entrypoint == "condition":
        result = _run_script(
            RUN_CONDITION,
            "A12",
            root,
            "29990",
            env={"WANDB_MODE": "online", "PREFLIGHT_ONLY": "0"},
            unset=("WANDB_API_KEY",),
        )
    else:
        result = _run_script(
            RUN_ALL,
            root,
            env={"WANDB_MODE": "online", "PREFLIGHT_ONLY": "0"},
            unset=("WANDB_API_KEY",),
        )

    assert result.returncode == 2
    assert "WANDB_MODE=online requires WANDB_API_KEY at runtime." in result.stderr
    assert not root.exists(), "missing-key validation must precede filesystem state writes"


def test_offline_needs_no_key_and_never_reuses_a_run_id(tmp_path: Path) -> None:
    run_root, stage_dir, call_log, _, common_env = _stage_fixture(tmp_path)
    marker = stage_dir / "state" / "wandb_run_id"
    marker.write_text("file-stale\n", encoding="utf-8")
    env = {
        **common_env,
        "WANDB_MODE": "offline",
        "PREFLIGHT_ONLY": "0",
        "WANDB_API_KEY": "must-be-ignored-offline",
        "WANDB_RUN_ID": "env-stale",
        "WANDB_RESUME": "must-not-leak",
    }

    first = _run_script(TRAIN_STAGE, env=env)
    second = _run_script(TRAIN_STAGE, env=env)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    calls = call_log.read_text(encoding="utf-8")
    torch_calls = [line for line in calls.splitlines() if line.startswith("torch ")]
    assert len(torch_calls) == 2
    assert all("mode=offline" in line for line in torch_calls)
    assert all("api=<unset>" in line for line in torch_calls)
    assert all("run_id=<unset>" in line for line in torch_calls)
    assert all("resume=<unset>" in line for line in torch_calls)
    assert "generate_id" not in calls
    assert marker.read_text(encoding="utf-8") == "file-stale\n"
    assert (stage_dir / "state" / "status").read_text(encoding="utf-8") == "completed\n"
    assert run_root.exists()


def test_direct_preflight_is_keyless_and_has_no_lifecycle_side_effects(tmp_path: Path) -> None:
    _, stage_dir, call_log, _, common_env = _stage_fixture(tmp_path)
    state = stage_dir / "state"
    sentinels = {
        "status": "status-sentinel\n",
        "started_at": "started-sentinel\n",
        "wandb_run_id": "id-sentinel\n",
    }
    for name, value in sentinels.items():
        (state / name).write_text(value, encoding="utf-8")

    result = _run_script(
        TRAIN_STAGE,
        env={**common_env, "WANDB_MODE": "online", "PREFLIGHT_ONLY": "1"},
        unset=("WANDB_API_KEY",),
    )

    assert result.returncode == 0, result.stderr
    assert "Launcher preflight passed without starting training" in result.stdout
    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 1
    assert calls[0].startswith("argv=- ")
    assert not any("torch.distributed.run" in line for line in calls)
    assert not any("generate_id" in line for line in calls)
    for name, value in sentinels.items():
        assert (state / name).read_text(encoding="utf-8") == value
    assert not (state / "finished_at").exists()
    assert not (state / "exit_code").exists()
    assert not (stage_dir / "logs" / "train.log").exists()


def _make_completed_condition(root: Path, condition: str, call_log: Path) -> Path:
    condition_root = root / condition
    _write_shell(
        condition_root / "scripts" / "train_stage.sh",
        r'''#!/usr/bin/env bash
set -euo pipefail
printf '%s|%s|%s|%s\n' "$CONDITION" "$STAGE" "$PREFLIGHT_ONLY" "$MASTER_PORT" >> "$CALL_LOG"
''',
    )
    for stage, final in (("tf", 3000), ("cd", 2000), ("sf", 1000)):
        (condition_root / stage / "checkpoints" / f"checkpoint-{final}" / "dcp").mkdir(
            parents=True
        )
        stage_state = condition_root / stage / "state"
        stage_state.mkdir(parents=True)
        (stage_state / "exit_code").write_text("0\n", encoding="utf-8")
        (stage_state / "status").write_text(f"{stage}-sentinel\n", encoding="utf-8")
    root_state = condition_root / "state"
    root_state.mkdir()
    (root_state / "status").write_text("root-sentinel\n", encoding="utf-8")
    return condition_root


def _assert_condition_sentinels(condition_root: Path) -> None:
    assert (condition_root / "state" / "status").read_text(encoding="utf-8") == "root-sentinel\n"
    assert not (condition_root / "state" / "started_at").exists()
    assert not (condition_root / "state" / "finished_at").exists()
    for stage in ("tf", "cd", "sf"):
        state = condition_root / stage / "state"
        assert (state / "status").read_text(encoding="utf-8") == f"{stage}-sentinel\n"
        assert (state / "exit_code").read_text(encoding="utf-8") == "0\n"
        assert not (state / "started_at").exists()
        assert not (state / "finished_at").exists()
        assert not (state / "wandb_run_id").exists()


def test_condition_preflight_checks_all_stages_even_if_checkpoints_are_complete(
    tmp_path: Path,
) -> None:
    call_log = tmp_path / "calls.log"
    condition_root = _make_completed_condition(tmp_path, "A12", call_log)

    result = _run_script(
        RUN_CONDITION,
        "A12",
        condition_root,
        "29820",
        env={
            "WANDB_MODE": "online",
            "PREFLIGHT_ONLY": "1",
            "CALL_LOG": call_log,
        },
        unset=("WANDB_API_KEY",),
    )

    assert result.returncode == 0, result.stderr
    assert call_log.read_text(encoding="utf-8").splitlines() == [
        "A12|tf|1|29821",
        "A12|cd|1|29822",
        "A12|sf|1|29823",
    ]
    assert "no training was started" in result.stdout
    _assert_condition_sentinels(condition_root)


def test_run_all_preflight_checks_twelve_configs_without_changing_state(tmp_path: Path) -> None:
    call_log = tmp_path / "calls.log"
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2(RUN_CONDITION, scripts / "run_condition.sh")
    conditions = ["A12", "A13", "A14", "A15"]
    condition_roots = [
        _make_completed_condition(tmp_path, condition, call_log) for condition in conditions
    ]

    result = _run_script(
        RUN_ALL,
        tmp_path,
        env={
            "WANDB_MODE": "online",
            "PREFLIGHT_ONLY": "1",
            "CALL_LOG": call_log,
        },
        unset=("WANDB_API_KEY",),
    )

    assert result.returncode == 0, result.stderr
    expected: list[str] = []
    for number, condition in enumerate(conditions, start=12):
        base = 29800 + number * 10
        expected.extend(
            [
                f"{condition}|tf|1|{base + 1}",
                f"{condition}|cd|1|{base + 2}",
                f"{condition}|sf|1|{base + 3}",
            ]
        )
    assert call_log.read_text(encoding="utf-8").splitlines() == expected
    for condition_root in condition_roots:
        _assert_condition_sentinels(condition_root)


@pytest.mark.parametrize(
    ("mode", "preflight", "message"),
    [
        ("airplane", "0", "Unsupported WANDB_MODE=airplane"),
        ("offline", "maybe", "PREFLIGHT_ONLY must be 0 or 1"),
    ],
)
def test_invalid_launcher_modes_fail_before_state(
    tmp_path: Path, mode: str, preflight: str, message: str
) -> None:
    root = tmp_path / "run"
    result = _run_script(
        TRAIN_STAGE,
        env={
            "RUN_ROOT": root,
            "STAGE": "tf",
            "MASTER_PORT": "29991",
            "WANDB_MODE": mode,
            "PREFLIGHT_ONLY": preflight,
        },
    )
    assert result.returncode == 2
    assert message in result.stderr
    assert not root.exists()


def test_generated_readmes_document_online_offline_and_preflight(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    experiment = tmp_path / "experiment"
    result = subprocess.run(
        [
            sys.executable,
            str(PREPARE),
            "--repo",
            str(repo),
            "--experiment-root",
            str(experiment),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    repo_readme = (
        repo / "examples" / "train" / "configs" / "ablation" / "openvid_causal_a12_a15" / "README.md"
    ).read_text(encoding="utf-8")
    experiment_readme = (experiment / "README.md").read_text(encoding="utf-8")
    assert repo_readme == experiment_readme

    for required in (
        "Online W&B is the default",
        "WANDB_API_KEY=...",
        "WANDB_MODE=offline",
        "does not merge separate offline process restarts into one run",
        "PREFLIGHT_ONLY=1",
        "run_condition.sh",
        "run_all_sequential.sh",
        "all twelve configs",
        "without starting training",
    ):
        assert required in repo_readme
    assert not re.search(r"WANDB_API_KEY=(?!\.\.\.)(\S+)", repo_readme)


def _literal_constants(path: Path, names: set[str]) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in names:
            value = ast.literal_eval(node.value)
            assert isinstance(value, str)
            found[target.id] = value
    assert found.keys() == names
    return found


def test_prepare_templates_match_standalone_scripts_and_all_parse(tmp_path: Path) -> None:
    names = {"TRAIN_STAGE", "QUEUE", "SEQUENCE", "REQUIRED_ANCESTOR"}
    constants = _literal_constants(PREPARE, names)
    train_text = TRAIN_STAGE.read_text(encoding="utf-8")
    match = re.search(r'^REPO="\$\{REPO:-([^}]+)\}"$', train_text, re.MULTILINE)
    assert match is not None
    default_repo = match.group(1)
    rendered = {
        TRAIN_STAGE: constants["TRAIN_STAGE"]
        .replace("__REPO__", default_repo)
        .replace("__COMMIT__", constants["REQUIRED_ANCESTOR"]),
        RUN_CONDITION: constants["QUEUE"].replace("__REPO__", default_repo),
        RUN_ALL: constants["SEQUENCE"],
    }
    for path, expected in rendered.items():
        assert path.read_text(encoding="utf-8") == expected
        syntax = subprocess.run(
            [BASH, "-n", _bash_path(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert syntax.returncode == 0, syntax.stderr

    condition_script = RUN_CONDITION.read_text(encoding="utf-8")
    assert 'export_stage tf 3000 student' in condition_script
    assert 'export_stage cd 2000 ema' in condition_script
    assert 'export_stage sf 1000 student_ema' in condition_script

    py_compile.compile(str(PREPARE), cfile=str(tmp_path / "prepare.pyc"), doraise=True)
