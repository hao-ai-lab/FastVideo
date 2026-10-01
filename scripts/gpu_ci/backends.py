"""Trusted execution adapters. No PR code executes in the controller process.

Modal API: https://modal.com/docs/sdk/py/latest/Sandbox
The controller persists deterministic handles before calling ``run``.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import subprocess
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Any

REPOSITORY = "https://github.com/hao-ai-lab/FastVideo.git"
ENV_ALLOWLIST = {
    "FASTVIDEO_GOLDEN_TEST_FILES", "FASTVIDEO_SSIM_TEST_FILES", "BUILDKITE_BRANCH",
    "BUILDKITE_SOURCE", "BUILDKITE_PULL_REQUEST", "BUILDKITE_BUILD_NUMBER", "BUILDKITE_BUILD_URL",
    "BUILDKITE_COMMIT", "BUILDKITE_REPO", "BUILDKITE_BUILD_ID", "BUILDKITE_JOB_ID",
}


def _image(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._/:\-]+@sha256:[0-9a-f]{64}", value):
        raise ValueError("Backend image must be pinned by sha256 digest")
    return value


def _subpath(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or value in {".", "/"}:
        raise ValueError("PVC mounts require a dedicated relative CI subpath")
    return value


class Backend:
    name = "abstract"

    def __init__(self, config: dict, trusted_dir: Path, run_dir: Path):
        self.config = config
        self.trusted_dir = trusted_dir
        self.run_dir = run_dir
        self.image = _image(config["image"])
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def handle(self, build_id: str, lane_id: str) -> dict:
        identity = hashlib.sha256(f"{build_id}\0{lane_id}".encode()).hexdigest()[:32]
        return {"backend": self.name, "name": f"fv-ci-{identity}", "run_id": identity}

    def _environment(self, request: dict, lane: dict) -> dict[str, str]:
        if request["repository"] != REPOSITORY or not re.fullmatch(r"[0-9a-f]{40}", request["commit"]):
            raise ValueError("Worker requires the fixed repository and an immutable commit")
        if type(lane["gpus"]) is not int or lane["gpus"] not in range(1, 5):
            raise ValueError("A lane must request between one and four GPUs")
        script = lane["script"]
        if script != ".buildkite/scripts/unit_test.sh" and not re.fullmatch(
                r"\.buildkite/scripts/lanes/[a-z_]+\.sh", script):
            raise ValueError("Invalid lane script")
        extras = lane.get("extras", ["test"])
        if not extras or any(not re.fullmatch(r"[a-z][a-z0-9-]*", extra) for extra in extras):
            raise ValueError("Invalid dependency extras")
        if str(lane["fa4"]) not in {"0", "1"}:
            raise ValueError("Invalid attention policy")
        build_id = str(request.get("buildkite_build_id") or request["build_id"])
        job_id = str(request.get("buildkite_job_id") or request.get("env", {}).get("BUILDKITE_JOB_ID")
                     or request["build_id"])
        if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}", value) for value in (build_id, job_id)):
            raise ValueError("Invalid Buildkite build/job identity")
        pr_number = str(request.get("pr_number") or "false")
        if pr_number != "false" and not re.fullmatch(r"[1-9][0-9]*", pr_number):
            raise ValueError("Invalid PR number")
        env = {key: str(value) for key, value in request.get("env", {}).items() if key in ENV_ALLOWLIST}
        env.update({
            "BUILDKITE_COMMIT": request["commit"], "BUILDKITE_REPO": REPOSITORY,
            "BUILDKITE_BUILD_ID": build_id, "BUILDKITE_JOB_ID": job_id,
            "BUILDKITE_PULL_REQUEST": pr_number, "FASTVIDEO_CONTAINER_IMAGE_REF": self.image,
            "FASTVIDEO_CI_REPOSITORY": REPOSITORY, "FASTVIDEO_CI_COMMIT": request["commit"],
            "FASTVIDEO_CI_PR_NUMBER": str(request.get("pr_number") or ""),
            "FASTVIDEO_CI_GPUS": str(lane["gpus"]), "FASTVIDEO_CI_SCRIPT": script,
            "FASTVIDEO_CI_EXTRAS": ",".join(extras), "FASTVIDEO_FA4": str(lane["fa4"]),
            "FASTVIDEO_CI_KERNEL": "1" if lane.get("kernel", True) else "0",
            "FASTVIDEO_CI_LOCAL_ONLY": "1", "FASTVIDEO_SSIM_BOOTSTRAP_MODE": "0",
            "TEST_SCOPE": request.get("scope", "direct"),
        })
        return env

    def run(self, handle: dict, request: dict, lane: dict, cancel: threading.Event) -> int:
        raise NotImplementedError

    def stop(self, handle: dict) -> bool:
        raise NotImplementedError


class KubernetesBackend(Backend):
    name = "kubernetes"

    def __init__(self, config: dict, trusted_dir: Path, run_dir: Path):
        super().__init__(config, trusted_dir, run_dir)
        if config.get("namespace", "vllm") != "vllm":
            raise ValueError("Kubernetes CI is restricted to namespace vllm")
        for kind in ("cache", "artifacts"):
            claim = config.get(f"{kind}_pvc")
            if claim:
                if claim in {"lustre-pvc-vllm", "nfs-pvc-vllm"}:
                    raise ValueError("Personal development PVCs cannot be mounted in CI")
                _subpath(config.get(f"{kind}_subpath", ""))

    def handle(self, build_id: str, lane_id: str) -> dict:
        result = super().handle(build_id, lane_id)
        result.update({"namespace": "vllm", "context": self.config.get("context")})
        return result

    def _command(self, handle: dict, *args: str) -> list[str]:
        if handle.get("namespace") != "vllm" or handle.get("context") != self.config.get("context"):
            raise ValueError("Handle does not match configured Kubernetes target")
        command = ["kubectl", "--request-timeout=20s", "--namespace=vllm"]
        if handle.get("context"):
            command += ["--context", handle["context"]]
        return [*command, *args]

    def _kubectl(self, handle: dict, *args: str, body: dict | None = None) -> str:
        result = subprocess.run(self._command(handle, *args), input=json.dumps(body) if body else None,
                                text=True, capture_output=True, timeout=35, check=False)
        if result.returncode:
            raise RuntimeError(f"kubectl {args[0]} failed: {result.stderr.strip()}")
        return result.stdout

    def _job(self, handle: dict) -> dict | None:
        raw = self._kubectl(handle, "get", "job", handle["name"], "--ignore-not-found", "-o", "json")
        return json.loads(raw) if raw.strip() else None

    def _pods(self, handle: dict) -> list[dict]:
        raw = self._kubectl(handle, "get", "pods", "-l", f"job-name={handle['name']}", "-o", "json")
        return json.loads(raw)["items"]

    def manifest(self, handle: dict, request: dict, lane: dict) -> dict:
        env = [{"name": key, "value": value} for key, value in self._environment(request, lane).items()]
        if self.config.get("hf_secret"):
            env.append({"name": "HF_TOKEN", "valueFrom": {"secretKeyRef": {
                "name": self.config["hf_secret"], "key": self.config.get("hf_secret_key", "HF_TOKEN")}}})
        volumes = [{"name": "workspace", "emptyDir": {}}, {"name": "shm", "emptyDir": {
            "medium": "Memory", "sizeLimit": self.config.get("shared_memory", "8Gi")}}]
        mounts = [{"name": "workspace", "mountPath": "/workspace"}, {"name": "shm", "mountPath": "/dev/shm"}]
        for kind, path in (("cache", "/ci-cache"), ("artifacts", "/workspace/artifacts")):
            if self.config.get(f"{kind}_pvc"):
                subpath = _subpath(self.config[f"{kind}_subpath"])
                if kind == "artifacts":
                    subpath += "/" + handle["name"]
                volumes.append({"name": kind, "persistentVolumeClaim": {"claimName": self.config[f"{kind}_pvc"]}})
                mounts.append({"name": kind, "mountPath": path, "subPath": subpath, "readOnly": kind == "cache"})
        resources = {"cpu": str(self.config.get("cpu", "8")), "memory": self.config.get("memory", "64Gi"),
                     "nvidia.com/gpu": str(lane["gpus"])}
        labels = {"app.kubernetes.io/managed-by": "fastvideo-gpu-ci", "fastvideo-ci/run-id": handle["run_id"]}
        manifest = {"apiVersion": "batch/v1", "kind": "Job", "metadata": {
            "name": handle["name"], "namespace": "vllm", "labels": labels}, "spec": {
                "backoffLimit": 0, "activeDeadlineSeconds": int(lane["timeout_seconds"]),
                "ttlSecondsAfterFinished": 86400, "template": {"metadata": {"labels": labels}, "spec": {
                    "restartPolicy": "Never", "automountServiceAccountToken": False,
                    "hostNetwork": False, "hostPID": False, "hostIPC": False,
                    "nodeSelector": {"kubernetes.io/arch": "arm64", "nvidia.com/gpu.product": "NVIDIA-GB200"},
                    "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
                    "containers": [{"name": "worker", "image": self.image,
                                    "command": ["/bin/bash", "-c", (self.trusted_dir / "worker.sh").read_text()],
                                    "env": env, "resources": {"requests": resources, "limits": resources},
                                    "securityContext": {"privileged": False, "allowPrivilegeEscalation": False,
                                                        "capabilities": {"drop": ["ALL"], "add": [
                                                            "CHOWN", "DAC_OVERRIDE", "FOWNER", "SETGID", "SETUID"]}},
                                    "volumeMounts": mounts}], "volumes": volumes}}}}
        if self.config.get("artifacts_pvc"):
            # Only this fixed init command sees the CI artifact parent. PR code
            # mounts its own run directory and cannot alter adjacent runs.
            manifest["spec"]["template"]["spec"]["initContainers"] = [{
                "name": "prepare-artifacts", "image": self.image,
                "command": ["mkdir", "-p", "/ci-artifacts/" + handle["name"]],
                "securityContext": {"allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]}},
                "volumeMounts": [{"name": "artifacts", "mountPath": "/ci-artifacts",
                                  "subPath": _subpath(self.config["artifacts_subpath"])}]}]
        return manifest

    @staticmethod
    def _owned(job: dict, handle: dict) -> bool:
        return job.get("metadata", {}).get("labels", {}).get("fastvideo-ci/run-id") == handle["run_id"]

    def run(self, handle: dict, request: dict, lane: dict, cancel: threading.Event) -> int:
        if self._job(handle) is not None or self._pods(handle):
            raise RuntimeError("Resource already exists; explicit reconciliation is required")
        manifest = self.manifest(handle, request, lane)
        (self.run_dir / f"{handle['name']}.job.json").write_text(json.dumps(manifest, indent=2))
        if not self.config.get("artifacts_pvc"):
            (self.run_dir / f"{handle['name']}.artifacts.txt").write_text(
                "No dedicated artifacts PVC configured. Controller logs/status survive; worker artifacts are ephemeral.\n")
        if cancel.is_set():
            return 130
        marker = self.run_dir / f"{handle['name']}.creating"
        marker.touch()
        self._kubectl(handle, "create", "-f", "-", body=manifest)
        marker.unlink()
        deadline = time.monotonic() + int(lane["timeout_seconds"])
        stream = None
        log_file = (self.run_dir / f"{handle['name']}.log").open("w")
        try:
            while True:
                if cancel.is_set() or time.monotonic() >= deadline:
                    if not self.stop(handle):
                        raise RuntimeError("Cannot confirm Kubernetes job cancellation")
                    return 130 if cancel.is_set() else 124
                job = self._job(handle)
                if job is None or not self._owned(job, handle):
                    raise RuntimeError("Kubernetes job disappeared or its ownership changed")
                pods = self._pods(handle)
                if len(pods) > 1:
                    raise RuntimeError("Unexpected replacement or duplicate CI pods")
                for pod in pods:
                    owners = pod.get("metadata", {}).get("ownerReferences", [])
                    if not any(owner.get("uid") == job["metadata"]["uid"] for owner in owners):
                        raise RuntimeError("Pod does not belong to the expected job")
                    statuses = pod.get("status", {}).get("containerStatuses", [])
                    worker = next((status for status in statuses if status["name"] == "worker"), {})
                    state = worker.get("state", {})
                    if stream is None and ("running" in state or "terminated" in state):
                        stream = subprocess.Popen(self._command(handle, "logs", "-f", pod["metadata"]["name"],
                                                                "-c", "worker", "--timestamps"),
                                                  stdout=log_file, stderr=subprocess.STDOUT, text=True)
                    terminated = state.get("terminated")
                    if terminated is not None:
                        exit_code = terminated.get("exitCode")
                        if type(exit_code) is not int:
                            raise RuntimeError("Worker termination has no numeric exit status")
                        (self.run_dir / f"{handle['name']}.status.json").write_text(json.dumps(pod["status"], indent=2))
                        # Streaming may fail independently of the pod. Fetch a
                        # final complete log before accepting even a zero exit.
                        final_log = self._kubectl(handle, "logs", pod["metadata"]["name"], "-c", "worker", "--timestamps")
                        if not final_log.strip():
                            raise RuntimeError("Worker completed but Kubernetes logs are empty")
                        (self.run_dir / f"{handle['name']}.final.log").write_text(final_log)
                        return exit_code
                    if pod.get("status", {}).get("phase") == "Failed":
                        raise RuntimeError("Pod failed without a worker exit status")
                if any(condition.get("type") == "Failed" and condition.get("status") == "True"
                       for condition in job.get("status", {}).get("conditions", [])):
                    raise RuntimeError("Kubernetes job failed before worker completion")
                cancel.wait(2)
        finally:
            if stream is not None:
                try:
                    stream.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    stream.terminate()
                    try:
                        stream.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        stream.kill()
                        stream.wait()
            log_file.close()

    def stop(self, handle: dict) -> bool:
        job = self._job(handle)
        if job is not None:
            if not self._owned(job, handle):
                return False
            self._kubectl(handle, "delete", "job", handle["name"], "--cascade=foreground",
                          "--wait=true", "--timeout=20s")
            (self.run_dir / f"{handle['name']}.creating").unlink(missing_ok=True)
        elif (self.run_dir / f"{handle['name']}.creating").exists():
            # A timed-out create may still commit remotely. Keep its GPU lease.
            return False
        # A missing Job alone is insufficient: orphaned Pods may still use GPUs.
        return self._job(handle) is None and not self._pods(handle)


class ModalBackend(Backend):
    name = "modal"

    def __init__(self, config: dict, trusted_dir: Path, run_dir: Path):
        super().__init__(config, trusted_dir, run_dir)
        if config.get("gpu", "L40S") != "L40S":
            raise ValueError("Modal default must remain L40S; Hopper lanes are selected by reviewed lane policy")
        self._sandboxes: dict[str, Any] = {}

    @staticmethod
    def _modal() -> Any:
        return importlib.import_module("modal")

    @staticmethod
    def _gpu_type(lane: dict) -> str:
        gpu_type = lane.get("modal_gpu", "L40S")
        if gpu_type not in ("L40S", "H100"):
            raise ValueError("Reviewed Modal hardware must be L40S or H100")
        return gpu_type

    def _environment(self, request: dict, lane: dict) -> dict[str, str]:
        self._gpu_type(lane)
        # Attention/reference policy differs from the GB200 lanes. Copy the
        # lane record so concurrent tasks never mutate shared policy or image.
        modal_lane = {**lane, "fa4": lane.get("modal_fa4", lane["fa4"])}
        return super()._environment(request, modal_lane)

    def handle(self, build_id: str, lane_id: str) -> dict:
        result = super().handle(build_id, lane_id)
        result["app"] = self.config.get("app", "fastvideo-gpu-ci")
        return result

    def _lookup(self, handle: dict) -> Any:
        if handle["app"] != self.config.get("app", "fastvideo-gpu-ci"):
            raise ValueError("Handle does not match configured Modal app")
        modal = self._modal()
        try:
            return modal.Sandbox.from_name(handle["app"], handle["name"])
        except modal.exception.NotFoundError:
            return None

    def run(self, handle: dict, request: dict, lane: dict, cancel: threading.Event) -> int:
        env = self._environment(request, lane)
        if self._lookup(handle) is not None:
            raise RuntimeError("Sandbox already exists; explicit reconciliation is required")
        modal = self._modal()
        image = modal.Image.from_registry(self.image).add_local_file(
            str(self.trusted_dir / "worker.sh"), "/opt/fastvideo-ci/worker.sh")
        app = modal.App.lookup(handle["app"], create_if_missing=True)
        secrets = [modal.Secret.from_name(self.config["hf_secret"])] if self.config.get("hf_secret") else []
        (self.run_dir / f"{handle['name']}.artifacts.txt").write_text(
            "Modal adapter retains controller logs/status. Worker artifacts are ephemeral.\n")
        if cancel.is_set():
            return 130
        marker = self.run_dir / f"{handle['name']}.creating"
        marker.touch()
        sandbox = modal.Sandbox.create(
            "/bin/bash", "/opt/fastvideo-ci/worker.sh", app=app, name=handle["name"], image=image,
            env=env, secrets=secrets, gpu=f"{self._gpu_type(lane)}:{lane['gpus']}",
            timeout=int(lane["timeout_seconds"]),
            cpu=float(self.config.get("cpu", 8)), memory=int(self.config.get("memory_mib", 65536)),
            tags={"fastvideo-ci-run-id": handle["run_id"]})
        self._sandboxes[handle["name"]] = sandbox
        marker.unlink()
        (self.run_dir / f"{handle['name']}.sandbox.json").write_text(json.dumps({"id": sandbox.object_id}))
        return self._watch(sandbox, handle, lane, cancel)

    def _watch(self, sandbox: Any, handle: dict, lane: dict, cancel: threading.Event) -> int:
        errors: list[Exception] = []

        def capture(stream: Any, suffix: str) -> None:
            try:
                with (self.run_dir / f"{handle['name']}.{suffix}.log").open("w") as output:
                    for line in stream:
                        output.write(line)
                        output.flush()
            except Exception as error:
                errors.append(error)

        threads = [threading.Thread(target=capture, args=(sandbox.stdout, "stdout"), daemon=True),
                   threading.Thread(target=capture, args=(sandbox.stderr, "stderr"), daemon=True)]
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + int(lane["timeout_seconds"])
        try:
            while True:
                if cancel.is_set() or time.monotonic() >= deadline:
                    if not self.stop(handle):
                        raise RuntimeError("Cannot confirm Modal sandbox cancellation")
                    return 130 if cancel.is_set() else 124
                code = sandbox.poll()
                if code is not None:
                    if type(code) is not int:
                        raise RuntimeError("Sandbox returned a nonnumeric exit status")
                    (self.run_dir / f"{handle['name']}.status.json").write_text(json.dumps({"exit_code": code}))
                    for thread in threads:
                        thread.join(timeout=5)
                    if errors or any(thread.is_alive() for thread in threads):
                        raise RuntimeError("Modal log collection did not complete")
                    return code
                cancel.wait(2)
        finally:
            # Do not detach a running sandbox: root must confirm stop before
            # freeing its shared GPU lease, including after SDK/network errors.
            for thread in threads:
                thread.join(timeout=1)

    def stop(self, handle: dict) -> bool:
        sandbox = self._sandboxes.get(handle["name"]) or self._lookup(handle)
        marker = self.run_dir / f"{handle['name']}.creating"
        if sandbox is None:
            return not marker.exists()
        if sandbox.get_tags().get("fastvideo-ci-run-id") != handle["run_id"]:
            return False
        code = sandbox.terminate(wait=True)
        confirmed = type(code) is int and sandbox.poll() is not None
        if confirmed:
            marker.unlink(missing_ok=True)
            sandbox.detach()
            self._sandboxes.pop(handle["name"], None)
        return confirmed
