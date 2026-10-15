# SPDX-License-Identifier: Apache-2.0
"""Single-GPU, same-user CUDA IPC weights for restartable inference consumers.

The donor owns registered tensors; each consumer owns new Python/module state.
The donor must outlive every mapped tensor. This prototype serves one active
consumer at a time and never falls back to checkpoint loading.
"""
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
import contextlib
import fcntl
import hashlib
import os
from pathlib import Path
import pickle
import socket
import stat
import struct
import threading
from typing import Any

import torch
from torch import nn
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor

_VERSION = 1
_MAX_MESSAGE = 64 * 1024 * 1024


@dataclass(frozen=True)
class WeightCacheKey:
    """Caller supplies immutable checkpoint/config and weight-processing identities.

    Method-body edits may reuse weights; edits affecting names/layout/preparation
    need a different processing_version. Tensor compatibility is also checked.
    """
    checkpoint: str
    component: str
    config_hash: str
    dtype: str
    processing_version: str

    def __post_init__(self):
        if any(not isinstance(getattr(self, item.name), str) or not getattr(self, item.name)
               for item in fields(self)):
            raise ValueError("Every weight-cache identity field must be a nonempty string")


@dataclass
class WeightAttachment:
    model: nn.Module
    donor_pid: int
    handles: dict[str, str]


def _send(conn: socket.socket, payload: Any) -> None:
    data = pickle.dumps(payload, protocol=5)
    if len(data) > _MAX_MESSAGE:
        raise ValueError("Weight-cache descriptor exceeds message limit")
    conn.sendall(struct.pack("!I", len(data)) + data)


def _read(conn: socket.socket, length: int) -> bytes:
    data = bytearray()
    while len(data) < length:
        chunk = conn.recv(length - len(data))
        if not chunk:
            raise ConnectionError("Weight-cache peer disconnected")
        data.extend(chunk)
    return bytes(data)


def _recv(conn: socket.socket) -> Any:
    length, = struct.unpack("!I", _read(conn, 4))
    if length > _MAX_MESSAGE:
        raise ValueError("Weight-cache descriptor exceeds message limit")
    return pickle.loads(_read(conn, length))


def _same_user(conn: socket.socket) -> None:
    # Descriptors contain pickle objects: authenticate the local peer before
    # reading any payload. CUDA IPC is currently a Linux-only contract here.
    _, uid, _ = struct.unpack("3i", conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
    if uid != os.getuid():
        raise PermissionError("Weight-cache peer is not the socket owner")


def _inventory(model: nn.Module) -> tuple[dict[str, torch.Tensor], dict[str, dict], dict[str, str]]:
    tensors, specs, aliases, canonical = {}, {}, {}, {}
    params = list(model.named_parameters(remove_duplicate=False))
    buffers = list(model.named_buffers(remove_duplicate=False))
    modules = dict(model.named_modules(remove_duplicate=False))
    for kind, values in (("parameter", params), ("buffer", buffers)):
        for name, tensor in values:
            if type(tensor) not in (torch.Tensor, nn.Parameter) or tensor.layout != torch.strided:
                raise ValueError(f"Only native strided tensors are supported: {name}")
            if id(tensor) in canonical:
                aliases[name] = canonical[id(tensor)]
            else:
                canonical[id(tensor)] = name
                tensors[name] = tensor
            parent, _, leaf = name.rpartition(".")
            specs[name] = {"kind": kind, "shape": tuple(tensor.shape), "dtype": str(tensor.dtype),
                           "stride": tuple(tensor.stride()), "offset": tensor.storage_offset(),
                           "persistent": kind == "parameter" or leaf not in modules[parent]._non_persistent_buffers_set}
    return tensors, specs, aliases


def _manifest(model: nn.Module) -> dict:
    _, specs, aliases = _inventory(model)
    return {"model_class": type(model).__module__ + "." + type(model).__qualname__,
            "specs": specs, "aliases": aliases}


def _validate_manifest(model: nn.Module, manifest: dict) -> None:
    if _manifest(model) != manifest:
        raise ValueError("Cached model names, shapes, dtypes, strides, aliases or class do not match")


def _bind(model: nn.Module, tensors: dict[str, torch.Tensor], manifest: dict) -> None:
    """Validate everything before replacing any meta registration."""
    _validate_manifest(model, manifest)
    expected = set(manifest["specs"]) - set(manifest["aliases"])
    if set(tensors) != expected:
        raise ValueError("Incomplete cached tensor set")
    for name, tensor in tensors.items():
        spec = manifest["specs"][name]
        if (tuple(tensor.shape), str(tensor.dtype), tuple(tensor.stride()), tensor.storage_offset()) != (
                spec["shape"], spec["dtype"], spec["stride"], spec["offset"]):
            raise ValueError(f"Cached tensor layout mismatch: {name}")
    modules = dict(model.named_modules(remove_duplicate=False))
    registered = {}
    for name in [*tensors, *manifest["aliases"]]:
        source = manifest["aliases"].get(name)
        spec = manifest["specs"][name]
        obj = registered[source] if source is not None else tensors[name]
        if source is None and spec["kind"] == "parameter":
            obj = nn.Parameter(obj, requires_grad=False)
        parent, _, leaf = name.rpartition(".")
        if spec["kind"] == "parameter":
            modules[parent].register_parameter(leaf, obj)
        else:
            modules[parent].register_buffer(leaf, obj, persistent=spec["persistent"])
        registered[name] = obj


def _gpu_uuid(device: torch.device) -> str:
    return str(torch.cuda.get_device_properties(device).uuid)


def _private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise PermissionError("Weight-cache socket directory must be owned by this user with mode 0700")


def serve_weight_cache(model: nn.Module, key: WeightCacheKey, socket_path: str | Path,
                       *, stop: threading.Event | None = None) -> None:
    """Serve an already-loaded, idle inference model until stop is set.

    One active consumer is supported. Graceful stop waits for its connection to
    close before releasing the model. SIGKILL/donor crashes invalidate consumers.
    Callers own signal handling and must not mutate or offload donor tensors.
    """
    if model.training:
        raise ValueError("The donor model must be in evaluation mode")
    tensors, _, _ = _inventory(model)
    devices = {tensor.device for tensor in tensors.values()}
    if not tensors or len(devices) != 1 or next(iter(devices)).type != "cuda":
        raise ValueError("The donor requires native tensors on one CUDA GPU")
    device = next(iter(devices))
    torch.cuda.synchronize(device)
    manifest = _manifest(model)
    path = Path(socket_path)
    _private_directory(path.parent)
    stop = stop if stop is not None else threading.Event()
    lock_fd = os.open(str(path) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(lock_fd)
        raise
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        path.unlink(missing_ok=True)  # Only the lock holder removes a stale socket.
        server.bind(str(path))
        os.chmod(path, 0o600)
        server.listen()
        server.settimeout(0.5)
        while not stop.is_set():
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            with conn:
                try:
                    _same_user(conn)
                    conn.settimeout(30)
                    request = _recv(conn)
                    if request != {"version": _VERSION, "key": asdict(key), "torch": torch.__version__,
                                   "gpu_uuid": _gpu_uuid(device)}:
                        _send(conn, {"error": "Weight-cache identity, runtime or GPU mismatch"})
                        continue
                    _send(conn, {"manifest": manifest, "pid": os.getpid()})
                    if _recv(conn) != "attach":
                        continue
                    # Generate reductions PER CONSUMER: each export acquires its
                    # own PyTorch CUDA IPC refcounts. Never reuse reduction args.
                    entries = {name: reduce_tensor(tensor.detach())[1] for name, tensor in tensors.items()}
                    _send(conn, entries)
                    conn.settimeout(None)
                    conn.recv(1)  # EOF/release: keep donor storage alive throughout the lease.
                except (ConnectionError, OSError, EOFError):
                    continue
                except (ValueError, TypeError, pickle.UnpicklingError) as error:
                    with contextlib.suppress(OSError):
                        _send(conn, {"error": str(error)})
                finally:
                    torch.cuda.ipc_collect()
    finally:
        server.close()
        path.unlink(missing_ok=True)
        os.close(lock_fd)


class WeightCacheClient:
    """Attach fresh module objects to donor-owned GPU storage; no disk fallback."""
    def __init__(self, socket_path: str | Path, key: WeightCacheKey, device: torch.device | None = None):
        self.socket_path = Path(socket_path)
        self.key = key
        self.device = device if device is not None else torch.device("cuda", torch.cuda.current_device())

    @contextmanager
    def load_model(self, factory: Callable[[], nn.Module]) -> Iterator[WeightAttachment]:
        """Construct on meta, validate the full layout, and bind IPC tensors.

        Hold this context until all GPU work finishes. Shared tensors are for
        read-only inference. Mutable request state must be constructed locally;
        training, in-place weight edits, CPU buffers and DTensor are unsupported.
        """
        with torch.device("meta"):
            model = factory()
        meta_tensors, _, _ = _inventory(model)
        template = _manifest(model)
        if not meta_tensors or any(tensor.device.type != "meta" for tensor in meta_tensors.values()):
            raise ValueError("Consumer factory must construct all registered tensors on meta")
        _private_directory(self.socket_path.parent)
        info = self.socket_path.stat()
        if info.st_uid != os.getuid() or not stat.S_ISSOCK(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077:
            raise PermissionError("Weight-cache socket must be private and owned by this user")
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.settimeout(30)
        mapped = {}
        bound = False
        try:
            conn.connect(str(self.socket_path))
            _same_user(conn)
            _send(conn, {"version": _VERSION, "key": asdict(self.key), "torch": torch.__version__,
                         "gpu_uuid": _gpu_uuid(self.device)})
            header = _recv(conn)
            if "error" in header:
                raise ValueError(header["error"])
            _validate_manifest(model, header["manifest"])
            _send(conn, "attach")
            entries = _recv(conn)
            index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            mapped, handles = {}, {}
            for name, exported in entries.items():
                args = list(exported)
                args[6] = index  # Exporter and consumer can have different visible-device ordinals.
                mapped[name] = rebuild_cuda_tensor(*args)
                handles[name] = hashlib.sha256(args[7] or b"").hexdigest()
            _bind(model, mapped, header["manifest"])
            bound = True
            yield WeightAttachment(model.eval(), header["pid"], handles)
        finally:
            # Synchronize even when inference raises before releasing the lease.
            try:
                torch.cuda.synchronize(self.device)
                if bound:
                    _bind(model, meta_tensors, template)
                mapped.clear()
            finally:
                conn.close()
