"""CPU-only lifecycle and isolation contracts for GPU CI adapters."""

import json
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.gpu_ci.backends import KubernetesBackend, ModalBackend, REPOSITORY

IMAGE = "ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev@sha256:" + "a" * 64
REQUEST = {"build_id": "build-one", "repository": REPOSITORY, "commit": "b" * 40,
           "pr_number": "12", "scope": "direct", "env": {"LD_PRELOAD": "bad.so"}}
LANE = {"key": "unit", "script": ".buildkite/scripts/unit_test.sh", "gpus": 1,
        "extras": ["test"], "fa4": "0", "timeout_seconds": 10}


def kubernetes(tmp_path, **config):
    return KubernetesBackend({"image": IMAGE, **config}, ROOT / "scripts/gpu_ci", tmp_path)


def test_manifest_pins_gpu_count_and_does_not_expose_cluster_or_personal_storage(tmp_path):
    backend = kubernetes(tmp_path, hf_secret="ci-hf-readonly")
    handle = backend.handle("build-one", "ssim")
    manifest = backend.manifest(handle, REQUEST, {**LANE, "gpus": 4})
    pod = manifest["spec"]["template"]["spec"]
    worker = pod["containers"][0]
    assert pod["automountServiceAccountToken"] is False
    assert not any(pod[key] for key in ("hostNetwork", "hostPID", "hostIPC"))
    assert worker["securityContext"]["privileged"] is False
    assert worker["resources"]["limits"]["nvidia.com/gpu"] == "4"
    assert manifest["spec"]["backoffLimit"] == 0
    assert worker["image"] == IMAGE
    assert not any("hostPath" in volume or "persistentVolumeClaim" in volume for volume in pod["volumes"])
    env = {entry["name"]: entry for entry in worker["env"]}
    assert "LD_PRELOAD" not in env
    assert env["HF_TOKEN"]["valueFrom"]["secretKeyRef"]["name"] == "ci-hf-readonly"
    assert env["FASTVIDEO_CI_LOCAL_ONLY"]["value"] == "1"
    assert handle == backend.handle("build-one", "ssim")


@pytest.mark.parametrize("config", [
    {"image": "image:latest"}, {"namespace": "default"},
    {"cache_pvc": "lustre-pvc-vllm", "cache_subpath": "satyam"},
    {"cache_pvc": "ci-cache", "cache_subpath": "../personal"},
    {"artifacts_pvc": "ci-output", "artifacts_subpath": "/"},
])
def test_unsafe_backend_configuration_is_rejected(tmp_path, config):
    with pytest.raises(ValueError):
        kubernetes(tmp_path, **config)


def test_pvc_mounts_are_narrow_and_cache_is_readonly(tmp_path):
    backend = kubernetes(tmp_path, cache_pvc="ci-cache", cache_subpath="hf-hub",
                         artifacts_pvc="ci-output", artifacts_subpath="runs")
    handle = backend.handle("build-one", "unit")
    pod = backend.manifest(handle, REQUEST, LANE)["spec"]["template"]["spec"]
    mounts = {mount["name"]: mount for mount in pod["containers"][0]["volumeMounts"]}
    assert mounts["cache"]["readOnly"]
    assert mounts["artifacts"]["subPath"] == "runs/" + handle["name"]
    init = pod["initContainers"][0]
    assert init["command"] == ["mkdir", "-p", "/ci-artifacts/" + handle["name"]]
    assert init["volumeMounts"][0]["subPath"] == "runs"


def test_kubectl_create_uses_json_stdin_and_no_shell(tmp_path, monkeypatch):
    backend = kubernetes(tmp_path)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="created", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    backend._kubectl(backend.handle("b", "unit"), "create", "-f", "-", body={"safe": True})
    command, kwargs = calls[0]
    assert command[-3:] == ["create", "-f", "-"]
    assert json.loads(kwargs["input"]) == {"safe": True}
    assert not kwargs.get("shell", False)


def fake_cluster(backend, handle, monkeypatch, exit_code=7):
    state = {"created": False, "deleted": False}
    job = {"metadata": {"uid": "job-uid", "labels": {"fastvideo-ci/run-id": handle["run_id"]}}}
    pod = {"metadata": {"name": "worker-pod", "ownerReferences": [{"uid": "job-uid"}]},
           "status": {"containerStatuses": [{"name": "worker", "state": {"terminated": {"exitCode": exit_code}}}]}}
    monkeypatch.setattr(backend, "_job", lambda unused: job if state["created"] else None)
    monkeypatch.setattr(backend, "_pods", lambda unused: [pod] if state["created"] else [])

    def kubectl(unused, *args, **kwargs):
        if args[0] == "create":
            state["created"] = True
        elif args[0] == "delete":
            assert "--cascade=foreground" in args
            assert "--wait=true" in args
            state.update(created=False, deleted=True)
        elif args[0] == "logs":
            return "captured worker output\n"
        return ""

    monkeypatch.setattr(backend, "_kubectl", kubectl)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace(wait=lambda **kw: 0))
    return state, job, pod


@pytest.mark.parametrize("exit_code", [0, 7, 137])
def test_kubernetes_propagates_authoritative_worker_exit(tmp_path, monkeypatch, exit_code):
    backend = kubernetes(tmp_path)
    handle = backend.handle("b", "unit")
    state, _, _ = fake_cluster(backend, handle, monkeypatch, exit_code)
    assert backend.run(handle, REQUEST, LANE, threading.Event()) == exit_code
    assert backend.stop(handle)
    assert state["deleted"]


def test_worker_metadata_uses_validated_request_and_pinned_image(tmp_path):
    backend = kubernetes(tmp_path)
    request = {**REQUEST, "buildkite_build_id": "real-build", "buildkite_job_id": "real-job",
               "env": {"BUILDKITE_REPO": "https://bad.example/repo", "BUILDKITE_COMMIT": "bad",
                       "BUILDKITE_BUILD_ID": "fake", "BUILDKITE_JOB_ID": "fake",
                       "BUILDKITE_PULL_REQUEST": "999", "FASTVIDEO_CONTAINER_IMAGE_REF": "image:latest"}}
    env = backend._environment(request, LANE)
    assert env["BUILDKITE_COMMIT"] == REQUEST["commit"]
    assert env["BUILDKITE_REPO"] == REPOSITORY
    assert env["BUILDKITE_BUILD_ID"] == "real-build"
    assert env["BUILDKITE_JOB_ID"] == "real-job"
    assert env["BUILDKITE_PULL_REQUEST"] == "12"
    assert env["FASTVIDEO_CONTAINER_IMAGE_REF"] == IMAGE
    with pytest.raises(ValueError, match="build/job identity"):
        backend._environment({**request, "buildkite_job_id": "bad\nidentifier"}, LANE)


@pytest.mark.parametrize("logs", ["", "error"])
def test_worker_success_without_retrievable_logs_is_infrastructure_failure(tmp_path, monkeypatch, logs):
    backend = kubernetes(tmp_path)
    handle = backend.handle("b", "unit")
    fake_cluster(backend, handle, monkeypatch, 0)
    original = backend._kubectl

    def kubectl(*args, **kwargs):
        if args[1] == "logs":
            if logs == "error":
                raise RuntimeError("kubectl logs failed")
            return logs
        return original(*args, **kwargs)

    monkeypatch.setattr(backend, "_kubectl", kubectl)
    with pytest.raises(RuntimeError, match="logs"):
        backend.run(handle, REQUEST, LANE, threading.Event())
    assert backend.stop(handle)


def test_final_log_fetch_recovers_failed_stream(tmp_path, monkeypatch):
    backend = kubernetes(tmp_path)
    handle = backend.handle("b", "unit")
    fake_cluster(backend, handle, monkeypatch, 0)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace(wait=lambda **kw: 1))
    assert backend.run(handle, REQUEST, LANE, threading.Event()) == 0
    assert "captured worker output" in (tmp_path / f"{handle['name']}.final.log").read_text()


def fake_modal(monkeypatch, backend, handle, exit_code=0):
    class NotFoundError(Exception):
        pass

    state = {"created": None, "terminated": False, "existing": None}
    sandbox = SimpleNamespace(
        object_id="sb-test", stdout=iter(["test output\n"]), stderr=iter([]),
        poll=lambda: exit_code, get_tags=lambda: {"fastvideo-ci-run-id": handle["run_id"]}, detach=lambda: None)

    def lookup(*args):
        if state["existing"] is None:
            raise NotFoundError
        return state["existing"]

    def create(*args, **kwargs):
        state["created"] = (args, kwargs)
        state["existing"] = sandbox
        return sandbox

    def terminate(*, wait):
        assert wait is True
        state["terminated"] = True
        return 137

    sandbox.terminate = terminate
    image = SimpleNamespace(add_local_file=lambda *args: "worker-image")
    module = SimpleNamespace(
        exception=SimpleNamespace(NotFoundError=NotFoundError),
        Sandbox=SimpleNamespace(from_name=lookup, create=create),
        Image=SimpleNamespace(from_registry=lambda image_ref: image),
        App=SimpleNamespace(lookup=lambda *args, **kwargs: "app"),
        Secret=SimpleNamespace(from_name=lambda name: name))
    monkeypatch.setattr(backend, "_modal", lambda: module)
    return state


@pytest.mark.parametrize("gpus", [1, 2, 4])
def test_modal_uses_single_bounded_sandbox_and_same_worker(tmp_path, monkeypatch, gpus):
    backend = ModalBackend({"image": IMAGE}, ROOT / "scripts/gpu_ci", tmp_path)
    handle = backend.handle("b", "unit")
    state = fake_modal(monkeypatch, backend, handle, 7)
    assert backend.run(handle, REQUEST, {**LANE, "gpus": gpus}, threading.Event()) == 7
    args, kwargs = state["created"]
    assert args == ("/bin/bash", "/opt/fastvideo-ci/worker.sh")
    assert kwargs["gpu"] == f"L40S:{gpus}"
    assert kwargs["name"] == handle["name"]
    assert kwargs["env"]["FASTVIDEO_CI_COMMIT"] == REQUEST["commit"]
    assert backend.stop(handle)
    assert state["terminated"]


def test_modal_unknown_existing_sandbox_is_not_reused_or_stopped(tmp_path, monkeypatch):
    backend = ModalBackend({"image": IMAGE}, ROOT / "scripts/gpu_ci", tmp_path)
    handle = backend.handle("b", "unit")
    state = fake_modal(monkeypatch, backend, handle)
    state["existing"] = SimpleNamespace(get_tags=lambda: {"fastvideo-ci-run-id": "other"})
    with pytest.raises(RuntimeError, match="already exists"):
        backend.run(handle, REQUEST, LANE, threading.Event())
    assert backend.stop(handle) is False


@pytest.mark.parametrize("key,gpus", [("training-vsa", 2), ("encoder", 1), ("kernel-tests", 1)])
def test_modal_hopper_lanes_keep_reviewed_hardware(tmp_path, monkeypatch, key, gpus):
    backend = ModalBackend({"image": IMAGE}, ROOT / "scripts/gpu_ci", tmp_path)
    handle = backend.handle("b", key)
    state = fake_modal(monkeypatch, backend, handle)
    lane = {**LANE, "key": key, "gpus": gpus, "modal_gpu": "H100", "modal_fa4": "0"}
    assert backend.run(handle, REQUEST, lane, threading.Event()) == 0
    assert state["created"][1]["gpu"] == f"H100:{gpus}"
    assert backend.stop(handle)


def test_modal_attention_policy_does_not_mutate_shared_gb200_lane(tmp_path):
    modal = ModalBackend({"image": IMAGE}, ROOT / "scripts/gpu_ci", tmp_path / "modal")
    kube = kubernetes(tmp_path / "kubernetes")
    lane = {**LANE, "modal_gpu": "L40S", "modal_fa4": "1"}
    assert modal._environment(REQUEST, lane)["FASTVIDEO_FA4"] == "1"
    assert kube._environment(REQUEST, lane)["FASTVIDEO_FA4"] == "0"
    assert lane["fa4"] == "0"
    assert modal._environment(REQUEST, lane)["FASTVIDEO_CONTAINER_IMAGE_REF"] == IMAGE
    assert modal.image == IMAGE


@pytest.mark.parametrize("override", [{"modal_gpu": "A100"}, {"modal_gpu": "H100:8"}, {"modal_fa4": "2"}])
def test_modal_rejects_unreviewed_hardware_or_attention_policy(tmp_path, override):
    backend = ModalBackend({"image": IMAGE}, ROOT / "scripts/gpu_ci", tmp_path)
    with pytest.raises(ValueError):
        backend._environment(REQUEST, {**LANE, **override})


def test_worker_rejects_repository_override_before_checkout(tmp_path):
    result = subprocess.run(["bash", str(ROOT / "scripts/gpu_ci/worker.sh")],
                            env={"PATH": "/usr/bin:/bin", "FASTVIDEO_CI_REPOSITORY": "https://example.invalid/repo"},
                            cwd=tmp_path, capture_output=True)
    assert result.returncode != 0
    assert not list(tmp_path.iterdir())


def test_missing_exit_status_is_not_a_pass(tmp_path, monkeypatch):
    backend = kubernetes(tmp_path)
    handle = backend.handle("b", "unit")
    fake_cluster(backend, handle, monkeypatch, None)
    with pytest.raises(RuntimeError, match="numeric exit"):
        backend.run(handle, REQUEST, LANE, threading.Event())


def test_unknown_job_and_orphaned_pods_do_not_release_lease(tmp_path, monkeypatch):
    backend = kubernetes(tmp_path)
    handle = backend.handle("b", "unit")
    state, job, pod = fake_cluster(backend, handle, monkeypatch)
    state["created"] = True
    job["metadata"]["labels"]["fastvideo-ci/run-id"] = "someone-else"
    assert backend.stop(handle) is False
    assert not state["deleted"]
    monkeypatch.setattr(backend, "_job", lambda unused: None)
    monkeypatch.setattr(backend, "_pods", lambda unused: [pod])
    assert backend.stop(handle) is False


def test_uncertain_create_response_remains_reserved(tmp_path, monkeypatch):
    backend = kubernetes(tmp_path)
    handle = backend.handle("b", "unit")
    monkeypatch.setattr(backend, "_job", lambda unused: None)
    monkeypatch.setattr(backend, "_pods", lambda unused: [])
    (tmp_path / f"{handle['name']}.creating").touch()
    assert backend.stop(handle) is False


def test_kubernetes_cancellation_confirms_deletion(tmp_path, monkeypatch):
    backend = kubernetes(tmp_path)
    handle = backend.handle("b", "unit")
    state, _, _ = fake_cluster(backend, handle, monkeypatch)
    answers = iter([False, True, True])
    event = SimpleNamespace(is_set=lambda: next(answers))
    assert backend.run(handle, REQUEST, LANE, event) == 130
    assert state["deleted"]
