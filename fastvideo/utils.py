# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/utils.py

import argparse
import ctypes
import hashlib
import importlib
import importlib.util
import inspect
import ipaddress
import json
import math
import multiprocessing
from multiprocessing.context import BaseContext
import os
import signal
import socket
import sys
import tempfile
import warnings
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass
from functools import lru_cache, partial, wraps
from pathlib import Path
from typing import Any, TextIO, TypeVar, cast

import cloudpickle
import filelock
import imageio
import numpy as np
import torch
import torchvision.utils as make_grid
import yaml
from diffusers.loaders.lora_base import (
    _best_guess_weight_name)  # watch out for potetential removal from diffusers
from einops import rearrange
from huggingface_hub import snapshot_download
from remote_pdb import RemotePdb
from torch.distributed.fsdp import MixedPrecisionPolicy

import fastvideo.envs as envs
from fastvideo.logger import init_logger

logger = init_logger(__name__)

T = TypeVar("T")

# TODO(will): used to convert fastvideo_args.precision to torch.dtype. Find a
# cleaner way to do this.
PRECISION_TO_TYPE = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

STR_BACKEND_ENV_VAR: str = "FASTVIDEO_ATTENTION_BACKEND"
STR_ATTN_CONFIG_ENV_VAR: str = "FASTVIDEO_ATTENTION_CONFIG"


def find_nccl_library() -> str:
    """
    We either use the library file specified by the `FASTVIDEO_NCCL_SO_PATH`
    environment variable, or we find the library file brought by PyTorch.
    After importing `torch`, `libnccl.so.2` or `librccl.so.1` can be
    found by `ctypes` automatically.
    """
    so_file = envs.FASTVIDEO_NCCL_SO_PATH

    # manually load the nccl library
    if so_file:
        logger.info(
            "Found nccl from environment variable FASTVIDEO_NCCL_SO_PATH=%s",
            so_file)
    else:
        if torch.version.cuda is not None:
            so_file = "libnccl.so.2"
        elif torch.version.hip is not None:
            so_file = "librccl.so.1"
        else:
            raise ValueError("NCCL only supports CUDA and ROCm backends.")
        logger.info("Found nccl from library %s", so_file)
    return str(so_file)


def find_hccl_library() -> str:
    """
    We either use the library file specified by the `HCCL_SO_PATH`
    environment variable, or we find the library file brought by PyTorch.
    After importing `torch`, `libhccl.so` can be
    found by `ctypes` automatically.
    """
    so_file = envs.HCCL_SO_PATH

    # manually load the nccl library
    if so_file:
        logger.info("Found hccl from environment variable HCCL_SO_PATH=%s",
                    so_file)
    else:
        if torch.version.cann is not None:  # codespell:ignore cann
            so_file = "libhccl.so"
        else:
            raise ValueError("HCCL only supports Ascend NPU backends.")
        logger.info("Found hccl from library %s", so_file)
    return so_file


prev_set_stream = torch.cuda.set_stream

_current_stream = None


def _patched_set_stream(stream: torch.cuda.Stream | None) -> None:
    global _current_stream
    _current_stream = stream
    if stream is not None:
        prev_set_stream(stream)


torch.cuda.set_stream = _patched_set_stream


def current_stream() -> torch.cuda.Stream | None:
    """
    replace `torch.cuda.current_stream()` with `fastvideo.utils.current_stream()`.
    it turns out that `torch.cuda.current_stream()` is quite expensive,
    as it will construct a new stream object at each call.
    here we patch `torch.cuda.set_stream` to keep track of the current stream
    directly, so that we can avoid calling `torch.cuda.current_stream()`.

    the underlying hypothesis is that we do not call `torch._C._cuda_setStream`
    from C/C++ code.
    """
    from fastvideo.platforms import current_platform

    # For non-CUDA platforms, return None
    if not current_platform.is_cuda_alike():
        return None

    global _current_stream
    if _current_stream is None:
        # when this function is called before any stream is set,
        # we return the default stream.
        # On ROCm using the default 0 stream in combination with RCCL
        # is hurting performance. Therefore creating a dedicated stream
        # per process
        _current_stream = torch.cuda.Stream() if current_platform.is_rocm(
        ) else torch.cuda.current_stream()
    return _current_stream


class StoreBoolean(argparse.Action):

    def __init__(self,
                 option_strings,
                 dest,
                 default=False,
                 required=False,
                 help=None):
        super().__init__(option_strings=option_strings,
                         dest=dest,
                         nargs='?',
                         const=True,
                         default=default,
                         required=required,
                         help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        if values is None:
            setattr(namespace, self.dest, True)
        elif isinstance(values, str):
            if values.lower() == "true":
                setattr(namespace, self.dest, True)
            elif values.lower() == "false":
                setattr(namespace, self.dest, False)
            else:
                raise ValueError(f"Invalid boolean value: {values}. "
                                 "Expected 'true' or 'false'.")
        else:
            setattr(namespace, self.dest, bool(values))


class SortedHelpFormatter(argparse.HelpFormatter):
    """SortedHelpFormatter that sorts arguments by their option strings."""

    def add_arguments(self, actions):
        actions = sorted(actions, key=lambda x: x.option_strings)
        super().add_arguments(actions)


class FlexibleArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that allows both underscore and dash in names."""

    def __init__(self, *args, **kwargs) -> None:
        # Set the default 'formatter_class' to SortedHelpFormatter
        if 'formatter_class' not in kwargs:
            kwargs['formatter_class'] = SortedHelpFormatter
        super().__init__(*args, **kwargs)

    def parse_args(  # type: ignore[override]
            self, args=None, namespace=None) -> argparse.Namespace:
        if args is None:
            args = sys.argv[1:]

        if '--config' in args:
            args = self._pull_args_from_config(args)

        # Convert underscores to dashes and vice versa in argument names
        processed_args = []
        for arg in args:
            if arg.startswith('--'):
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    key = '--' + key[len('--'):].replace('_', '-')
                    processed_args.append(f'{key}={value}')
                else:
                    processed_args.append('--' +
                                          arg[len('--'):].replace('_', '-'))
            elif arg.startswith('-O') and arg != '-O' and len(arg) == 2:
                # allow -O flag to be used without space, e.g. -O3
                processed_args.append('-O')
                processed_args.append(arg[2:])
            else:
                processed_args.append(arg)

        namespace = super().parse_args(processed_args, namespace)

        # Track which arguments were explicitly provided
        namespace._provided = set()

        i = 0
        while i < len(args):
            arg = args[i]
            if arg.startswith('--'):
                # Handle --key=value format
                if '=' in arg:
                    key = arg.split('=')[0][2:].replace('-', '_')
                    namespace._provided.add(key)
                    i += 1
                # Handle --key value format
                else:
                    key = arg[2:].replace('-', '_')
                    namespace._provided.add(key)
                    # Skip the value if there is one
                    if i + 1 < len(args) and not args[i + 1].startswith('-'):
                        i += 2
                    else:
                        i += 1
            else:
                i += 1

        return namespace  # type: ignore[no-any-return]

    def _pull_args_from_config(self, args: list[str]) -> list[str]:
        """Method to pull arguments specified in the config file
        into the command-line args variable.

        The arguments in config file will be inserted between
        the argument list.

        example:
        ```yaml
            port: 12323
            tensor-parallel-size: 4
        ```
        ```python
        $: vllm {serve,chat,complete} "facebook/opt-12B" \
            --config config.yaml -tp 2
        $: args = [
            "serve,chat,complete",
            "facebook/opt-12B",
            '--config', 'config.yaml',
            '-tp', '2'
        ]
        $: args = [
            "serve,chat,complete",
            "facebook/opt-12B",
            '--port', '12323',
            '--tp-size', '4',
            '-tp', '2'
            ]
        ```

        Please note how the config args are inserted after the sub command.
        this way the order of priorities is maintained when these are args
        parsed by super().
        """
        assert args.count(
            '--config') <= 1, "More than one config file specified!"

        index = args.index('--config')
        if index == len(args) - 1:
            raise ValueError("No config file specified! \
                             Please check your command-line arguments.")

        file_path = args[index + 1]

        config_args = self._load_config_file(file_path)

        # 0th index is for {serve,chat,complete}
        # followed by model_tag (only for serve)
        # followed by config args
        # followed by rest of cli args.
        # maintaining this order will enforce the precedence
        # of cli > config > defaults
        if args[0] == "serve":
            if index == 1:
                raise ValueError(
                    "No model_tag specified! Please check your command-line"
                    " arguments.")
            args = [args[0]] + [
                args[1]
            ] + config_args + args[2:index] + args[index + 2:]
        else:
            args = [args[0]] + config_args + args[1:index] + args[index + 2:]

        return args

    def _load_config_file(self, file_path: str) -> list[str]:
        """Loads a yaml file and returns the key value pairs as a
        flattened list with argparse like pattern
        ```yaml
            port: 12323
            tensor-parallel-size: 4
            vae_config:
                load_encoder: false
                load_decoder: true
        ```
        returns:
            processed_args: list[str] = [
                '--port': '12323',
                '--tp-size': '4',
                '--vae-config.load-encoder': 'false',
                '--vae-config.load-decoder': 'true'
            ]
        """

        extension: str = file_path.split('.')[-1]
        if extension not in ('yaml', 'yml', 'json'):
            raise ValueError(
                "Config file must be of a yaml/yml/json type.\
                              %s supplied", extension)

        processed_args: list[str] = []

        config: dict[str, Any] = {}
        try:
            with open(file_path) as config_file:
                config = yaml.safe_load(config_file)
        except Exception as ex:
            logger.error(
                "Unable to read the config file at %s. \
                Make sure path is correct", file_path)
            raise ex

        store_boolean_arguments = [
            action.dest for action in self._actions
            if isinstance(action, StoreBoolean)
        ]

        def process_dict(prefix: str, d: dict[str, Any]):
            for key, value in d.items():
                full_key = f"{prefix}.{key}" if prefix else key

                if isinstance(value,
                              bool) and full_key not in store_boolean_arguments:
                    if value:
                        processed_args.append('--' + full_key)
                    else:
                        processed_args.append('--' + full_key)
                        processed_args.append('false')
                elif isinstance(value, list):
                    processed_args.append('--' + full_key)
                    for item in value:
                        processed_args.append(str(item))
                elif isinstance(value, dict):
                    process_dict(full_key, value)
                else:
                    processed_args.append('--' + full_key)
                    processed_args.append(str(value))

        process_dict("", config)

        return processed_args


def get_lock(model_name_or_path: str):
    lock_dir = tempfile.gettempdir()
    os.makedirs(os.path.dirname(lock_dir), exist_ok=True)
    model_name = model_name_or_path.replace("/", "-")
    hash_name = hashlib.sha256(model_name.encode()).hexdigest()
    # add hash to avoid conflict with old users' lock files
    lock_file_name = hash_name + model_name + ".lock"
    # mode 0o666 is required for the filelock to be shared across users
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name), mode=0o666)
    return lock


def warn_for_unimplemented_methods(cls: type[T]) -> type[T]:
    """
    A replacement for `abc.ABC`.
    When we use `abc.ABC`, subclasses will fail to instantiate
    if they do not implement all abstract methods.
    Here, we only require `raise NotImplementedError` in the
    base class, and log a warning if the method is not implemented
    in the subclass.
    """

    original_init = cls.__init__

    def find_unimplemented_methods(self: object):
        unimplemented_methods = []
        for attr_name in dir(self):
            # bypass inner method
            if attr_name.startswith('_'):
                continue

            try:
                attr = getattr(self, attr_name)
                # get the func of callable method
                if callable(attr):
                    attr_func = attr.__func__
            except AttributeError:
                continue
            src = inspect.getsource(attr_func)
            if "NotImplementedError" in src:
                unimplemented_methods.append(attr_name)
        if unimplemented_methods:
            method_names = ','.join(unimplemented_methods)
            msg = (f"Methods {method_names} not implemented in {self}")
            logger.warning(msg)

    @wraps(original_init)
    def wrapped_init(self, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)
        find_unimplemented_methods(self)

    type.__setattr__(cls, '__init__', wrapped_init)
    return cls


def align_to(value: int, alignment: int) -> int:
    """align height, width according to alignment

    Args:
        value (int): height or width
        alignment (int): target alignment factor

    Returns:
        int: the aligned value
    """
    return int(math.ceil(value / alignment) * alignment)


def resolve_obj_by_qualname(qualname: str) -> Any:
    """
    Resolve an object by its fully qualified name.
    """
    module_name, obj_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


# From vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/utils.py
def import_pynvml():
    """
    Historical comments:

    libnvml.so is the library behind nvidia-smi, and
    pynvml is a Python wrapper around it. We use it to get GPU
    status without initializing CUDA context in the current process.
    Historically, there are two packages that provide pynvml:
    - `nvidia-ml-py` (https://pypi.org/project/nvidia-ml-py/): The official
        wrapper. It is a dependency of FastVideo, and is installed when users
        install FastVideo. It provides a Python module named `pynvml`.
    - `pynvml` (https://pypi.org/project/pynvml/): An unofficial wrapper.
        Prior to version 12.0, it also provides a Python module `pynvml`,
        and therefore conflicts with the official one which is a standalone Python file.
        This causes errors when both of them are installed.
        Starting from version 12.0, it migrates to a new module
        named `pynvml_utils` to avoid the conflict.
    It is so confusing that many packages in the community use the
    unofficial one by mistake, and we have to handle this case.
    For example, `nvcr.io/nvidia/pytorch:24.12-py3` uses the unofficial
    one, and it will cause errors, see the issue
    https://github.com/vllm-project/vllm/issues/12847 for example.
    After all the troubles, we decide to copy the official `pynvml`
    module to our codebase, and use it directly.
    """
    import fastvideo.third_party.pynvml as pynvml
    return pynvml


def maybe_download_model(model_name_or_path: str,
                         local_dir: str | None = None,
                         download: bool = True) -> str:
    """
    Check if the model path is a Hugging Face Hub model ID and download it if needed.
    
    Args:
        model_name_or_path: Local path or Hugging Face Hub model ID
        local_dir: Local directory to save the model
        download: Whether to download the model from Hugging Face Hub
        
    Returns:
        Local path to the model
    """

    # If the path exists locally, return it
    if os.path.exists(model_name_or_path):
        logger.info("Model already exists locally at %s", model_name_or_path)
        return model_name_or_path

    # Otherwise, assume it's a HF Hub model ID and try to download it
    try:
        logger.info("Downloading model snapshot from HF Hub for %s...",
                    model_name_or_path)
        with get_lock(model_name_or_path):
            local_path = snapshot_download(
                repo_id=model_name_or_path,
                ignore_patterns=["*.onnx", "*.msgpack"],
                local_dir=local_dir)
        logger.info("Downloaded model to %s", local_path)
        return str(local_path)
    except Exception as e:
        raise ValueError(
            f"Could not find model at {model_name_or_path} and failed to download from HF Hub: {e}"
        ) from e


def maybe_download_lora(model_name_or_path: str,
                        local_dir: str | None = None,
                        download: bool = True) -> str:
    """
    Check if the model path is a Hugging Face Hub model ID and download it if needed.
    Args:
        model_name_or_path: Local path or Hugging Face Hub model ID
        local_dir: Local directory to save the model
        download: Whether to download the model from Hugging Face Hub
        
    Returns:
        Local path to the model
    """

    local_path = maybe_download_model(model_name_or_path, local_dir, download)
    weight_name = _best_guess_weight_name(model_name_or_path,
                                          file_extension=".safetensors")
    return os.path.join(local_path, weight_name)


def verify_model_config_and_directory(model_path: str) -> dict[str, Any]:
    """
    Verify that the model directory contains a valid diffusers configuration.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        The loaded model configuration as a dictionary
    """

    # Check for model_index.json which is required for diffusers models
    config_path = os.path.join(model_path, "model_index.json")
    if not os.path.exists(config_path):
        raise ValueError(
            f"Model directory {model_path} does not contain model_index.json. "
            "Only Hugging Face diffusers format is supported.")

    # Check for transformer and vae directories
    transformer_dir = os.path.join(model_path, "transformer")
    vae_dir = os.path.join(model_path, "vae")

    if not os.path.exists(transformer_dir):
        raise ValueError(
            f"Model directory {model_path} does not contain a transformer/ directory."
        )

    if not os.path.exists(vae_dir):
        raise ValueError(
            f"Model directory {model_path} does not contain a vae/ directory.")

    # Load the config
    with open(config_path) as f:
        config = json.load(f)

    # Verify diffusers version exists
    if "_diffusers_version" not in config:
        raise ValueError("model_index.json does not contain _diffusers_version")

    logger.info("Diffusers version: %s", config["_diffusers_version"])
    return cast(dict[str, Any], config)


def maybe_download_model_index(model_name_or_path: str) -> dict[str, Any]:
    """
    Download and extract just the model_index.json for a Hugging Face model.
    
    Args:
        model_name_or_path: Path or HF Hub model ID
        
    Returns:
        The parsed model_index.json as a dictionary
    """
    import tempfile

    from huggingface_hub import hf_hub_download

    # If it's a local path, verify it directly
    if os.path.exists(model_name_or_path):
        return verify_model_config_and_directory(model_name_or_path)

    # For remote models, download just the model_index.json
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Download just the model_index.json file
            model_index_path = hf_hub_download(repo_id=model_name_or_path,
                                               filename="model_index.json",
                                               local_dir=tmp_dir)

            # Load the model_index.json
            with open(model_index_path) as f:
                config: dict[str, Any] = json.load(f)

            # Verify it has the required fields
            if "_class_name" not in config:
                raise ValueError(
                    f"model_index.json for {model_name_or_path} does not contain _class_name field"
                )

            if "_diffusers_version" not in config:
                raise ValueError(
                    f"model_index.json for {model_name_or_path} does not contain _diffusers_version field"
                )

            # Add the pipeline name for downstream use
            config["pipeline_name"] = config["_class_name"]

            logger.info("Downloaded model_index.json for %s, pipeline: %s",
                        model_name_or_path, config["_class_name"])
            return config

    except Exception as e:
        raise ValueError(
            f"Failed to download or parse model_index.json for {model_name_or_path}: {e}"
        ) from e


def update_environment_variables(envs: dict[str, str]):
    for k, v in envs.items():
        if k in os.environ and os.environ[k] != v:
            logger.warning(
                "Overwriting environment variable %s "
                "from '%s' to '%s'", k, os.environ[k], v)
        os.environ[k] = v


def run_method(obj: Any, method: str | bytes | Callable, args: tuple[Any],
               kwargs: dict[str, Any]) -> Any:
    """
    Run a method of an object with the given arguments and keyword arguments.
    If the method is string, it will be converted to a method using getattr.
    If the method is serialized bytes and will be deserialized using
    cloudpickle.
    If the method is a callable, it will be called directly.
    """
    if isinstance(method, bytes):
        func = partial(cloudpickle.loads(method), obj)
    elif isinstance(method, str):
        try:
            func = getattr(obj, method)
        except AttributeError:
            raise NotImplementedError(f"Method {method!r} is not"
                                      " implemented.") from None
    else:
        func = partial(method, obj)  # type: ignore
    return func(*args, **kwargs)


def shallow_asdict(obj) -> dict[str, Any]:
    if not is_dataclass(obj):
        raise TypeError("Expected dataclass instance")
    return {f.name: getattr(obj, f.name) for f in fields(obj)}


# TODO: validate that this is fine
def kill_itself_when_parent_died() -> None:
    # if sys.platform == "linux":
    # sigkill this process when parent worker manager dies
    PR_SET_PDEATHSIG = 1
    import platform
    if platform.system() == "Linux":
        libc = ctypes.CDLL("libc.so.6")
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    # elif platform.system() == "Darwin":
    #     libc = ctypes.CDLL("libc.dylib")
    #     logger.warning("kill_itself_when_parent_died is only supported in linux.")
    else:
        logger.warning(
            "kill_itself_when_parent_died is only supported in linux.")


def get_exception_traceback() -> str:
    etype, value, tb = sys.exc_info()
    err_str = "".join(traceback.format_exception(etype, value, tb))
    return err_str


class TypeBasedDispatcher:

    def __init__(self, mapping: list[tuple[type, Callable]]):
        self._mapping = mapping

    def __call__(self, obj: Any):
        for ty, fn in self._mapping:
            if isinstance(obj, ty):
                return fn(obj)
        raise ValueError(f"Invalid object: {obj}")


# For non-torch.distributed debugging
def remote_breakpoint() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("localhost", 0))  # Let the OS pick an ephemeral port.
        port = s.getsockname()[1]
        RemotePdb(host="localhost", port=port).set_trace()


@dataclass
class MixedPrecisionState:
    param_dtype: torch.dtype | None = None
    reduce_dtype: torch.dtype | None = None
    output_dtype: torch.dtype | None = None
    compute_dtype: torch.dtype | None = None
    mp_policy: MixedPrecisionPolicy | None = None


# Thread-local storage for mixed precision state
_mixed_precision_state = threading.local()


def get_mixed_precision_state() -> MixedPrecisionState:
    """Get the current mixed precision state."""
    if not hasattr(_mixed_precision_state, 'state'):
        raise ValueError("Mixed precision state not set")
    return cast(MixedPrecisionState, _mixed_precision_state.state)


def set_mixed_precision_policy(
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    output_dtype: torch.dtype | None = None,
    mp_policy: MixedPrecisionPolicy | None = None,
):
    """Set mixed precision policy globally.
    
    Args:
        param_dtype: Parameter dtype used for training
        reduce_dtype: Reduction dtype used for gradients
        output_dtype: Optional output dtype
    """
    state = MixedPrecisionState(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=output_dtype,
        mp_policy=mp_policy,
    )
    _mixed_precision_state.state = state


def get_compute_dtype() -> torch.dtype:
    """Get the current compute dtype from mixed precision policy.
    
    Returns:
        torch.dtype: The compute dtype to use, defaults to get_default_dtype() if no policy set
    """
    if not hasattr(_mixed_precision_state, 'state'):
        return torch.get_default_dtype()
    else:
        state = get_mixed_precision_state()
        return state.param_dtype


def dict_to_3d_list(
    mask_strategy: dict[str, Any] | None = None,
    t_max: int | None = None,
    l_max: int | None = None,
    h_max: int | None = None,
) -> list[list[list[torch.Tensor | None]]]:
    """
    Convert a dictionary of mask indices to a 3D list of tensors.
    Args:
        mask_strategy: keys are "t_l_h", values are torch.Tensor masks.
        t_max, l_max, h_max: if provided (all three), force the output shape to (t_max, l_max, h_max).
                            If all three are None, infer shape from the data.
    """
    # Case 1: no data, but fixed shape requested
    if mask_strategy is None:
        assert t_max is not None and l_max is not None and h_max is not None, (
            "If mask_strategy is None, you must provide t_max, l_max, and h_max"
        )
        return [[[None for _ in range(h_max)] for _ in range(l_max)]
                for _ in range(t_max)]

    # Parse all keys into integer tuples
    indices = [tuple(map(int, key.split("_"))) for key in mask_strategy]

    # Decide on dimensions
    if t_max is None and l_max is None and h_max is None:
        # fully dynamic: infer from data
        max_timesteps_idx = max(t for t, _, _ in indices) + 1
        max_layer_idx = max(l for _, l, _ in indices) + 1  # noqa: E741
        max_head_idx = max(h for _, _, h in indices) + 1
    else:
        # require all three to be provided
        assert t_max is not None and l_max is not None and h_max is not None, (
            "Either supply none of (t_max, l_max, h_max) to infer dimensions, "
            "or supply all three to fix the shape.")
        max_timesteps_idx = t_max
        max_layer_idx = l_max
        max_head_idx = h_max

    # Preallocate
    result = [[[None for _ in range(max_head_idx)]
               for _ in range(max_layer_idx)] for _ in range(max_timesteps_idx)]

    # Fill in, skipping any out-of-bounds entries
    for key, value in mask_strategy.items():
        t, l, h = map(int, key.split("_"))  # noqa: E741
        if 0 <= t < max_timesteps_idx and 0 <= l < max_layer_idx and 0 <= h < max_head_idx:
            result[t][l][h] = value
        # else: silently ignore any key that doesn't fit

    return result


def set_random_seed(seed: int) -> None:
    from fastvideo.platforms import current_platform
    current_platform.seed_everything(seed)


@lru_cache(maxsize=1)
def is_vsa_available() -> bool:
    return importlib.util.find_spec("vsa") is not None


@lru_cache(maxsize=1)
def is_vmoba_available() -> bool:
    if importlib.util.find_spec("csrc.attn.vmoba_attn.vmoba") is None:
        return False
    try:
        import flash_attn
        return flash_attn.__version__ >= "2.7.4"
    except Exception:
        return False


# adapted from: https://github.com/Wan-Video/Wan2.2/blob/main/wan/utils/utils.py
def masks_like(tensor,
               zero=False,
               generator=None,
               p=0.2) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    assert isinstance(tensor, list)
    out1 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]

    out2 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]

    if zero:
        if generator is not None:
            for u, v in zip(out1, out2, strict=False):
                random_num = torch.rand(1,
                                        generator=generator,
                                        device=generator.device).item()
                if random_num < p:
                    u[:, 0] = torch.normal(mean=-3.5,
                                           std=0.5,
                                           size=(1, ),
                                           device=u.device,
                                           generator=generator).expand_as(
                                               u[:, 0]).exp()
                    v[:, 0] = torch.zeros_like(v[:, 0])
                else:
                    u[:, 0] = u[:, 0]
                    v[:, 0] = v[:, 0]

        else:
            for u, v in zip(out1, out2, strict=False):
                u[:, 0] = torch.zeros_like(u[:, 0])
                v[:, 0] = torch.zeros_like(v[:, 0])

    return out1, out2


# adapted from: https://github.com/Wan-Video/Wan2.2/blob/main/wan/utils/utils.py
def best_output_size(w, h, dw, dh, expected_area):
    # float output size
    ratio = w / h
    ow = (expected_area * ratio)**0.5
    oh = expected_area / ow

    # process width first
    ow1 = int(ow // dw * dw)
    oh1 = int(expected_area / ow1 // dh * dh)
    assert ow1 % dw == 0 and oh1 % dh == 0 and ow1 * oh1 <= expected_area
    ratio1 = ow1 / oh1

    # process height first
    oh2 = int(oh // dh * dh)
    ow2 = int(expected_area / oh2 // dw * dw)
    assert oh2 % dh == 0 and ow2 % dw == 0 and ow2 * oh2 <= expected_area
    ratio2 = ow2 / oh2

    # compare ratios
    if max(ratio / ratio1, ratio1 / ratio) < max(ratio / ratio2,
                                                 ratio2 / ratio):
        return ow1, oh1
    else:
        return ow2, oh2


def save_decoded_latents_as_video(decoded_latents: list[torch.Tensor],
                                  output_path: str, fps: int):
    # Process outputs
    videos = rearrange(decoded_latents, "b c t h w -> t b c h w")
    frames = []
    for x in videos:
        x = make_grid(x, nrow=6)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        frames.append((x * 255).numpy().astype(np.uint8))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps, format="mp4")


def _format_bytes(num_bytes: int | float | None) -> str:
    if num_bytes is None:
        return "N/A"
    return f"{num_bytes / (1024 ** 3):.2f} GB"


def log_torch_cuda_memory(
        tag: str | None = None,
        *,
        log_fn: Callable[[str], None] | None = None,
        log_file_path: str | os.PathLike[str] | None = "memory_trace.txt"
) -> None:
    """Log CUDA memory statistics via logger and append to a trace file."""

    log_fn = log_fn or logger.info
    prefix = f"[{tag}] " if tag else ""

    if not torch.cuda.is_available():
        message = f"{prefix}CUDA not available on this host."
        log_fn(message)
        _append_to_memory_trace(message, log_file_path)
        return

    try:
        device_index = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_index)
        allocated = torch.cuda.memory_allocated(device_index)
        reserved = torch.cuda.memory_reserved(device_index)
        max_allocated = torch.cuda.max_memory_allocated(device_index)
        max_reserved = torch.cuda.max_memory_reserved(device_index)
        free_mem, total_mem = torch.cuda.mem_get_info(device_index)
    except Exception as exc:  # noqa: BLE001
        message = f"{prefix}Unable to query CUDA memory stats: {exc}"
        log_fn(message)
        _append_to_memory_trace(message, log_file_path)
        return

    used_mem = total_mem - free_mem

    stats = [
        f"device={device_name} (index={device_index})",
        f"allocated={_format_bytes(allocated)}",
        f"reserved={_format_bytes(reserved)}",
        f"max_allocated={_format_bytes(max_allocated)}",
        f"max_reserved={_format_bytes(max_reserved)}",
        f"used={_format_bytes(used_mem)}",
        f"free={_format_bytes(free_mem)}",
        f"total={_format_bytes(total_mem)}",
    ]

    message = f"{prefix}CUDA memory stats: {' | '.join(stats)}"
    log_fn(message)
    _append_to_memory_trace(message, log_file_path)


def _append_to_memory_trace(
        message: str, log_file_path: str | os.PathLike[str] | None) -> None:
    if not log_file_path:
        return

    try:
        path = Path(log_file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as outfile:
            outfile.write(message + "\n")
    except Exception:  # noqa: BLE001
        # Avoid raising/logging recursively if writing to the file fails.
        pass


# TODO(xingyu): add adopted message for this
def get_ip() -> str:
    host_ip = envs.FASTVIDEO_HOST_IP
    if host_ip:
        return host_ip

    # IP is not set, try to get it from the network interface

    # try ipv4
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        pass

    # try ipv6
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        # Google's public DNS server, see
        # https://developers.google.com/speed/public-dns/docs/using#addresses
        s.connect(("2001:4860:4860::8888", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        pass

    warnings.warn(
        "Failed to get the IP address, using 0.0.0.0 by default."
        "The value can be set by the environment variable FASTVIDEO_HOST_IP.",
        stacklevel=2)
    return "0.0.0.0"


def test_loopback_bind(address: str, family: socket.AddressFamily) -> bool:
    try:
        s = socket.socket(family, socket.SOCK_DGRAM)
        s.bind((address, 0))  # Port 0 = auto assign
        s.close()
        return True
    except OSError:
        return False


def get_loopback_ip() -> str:
    loopback_ip = envs.FASTVIDEO_LOOPBACK_IP
    if loopback_ip:
        return loopback_ip

    # FASTVIDEO_LOOPBACK_IP is not set, try to get it based on network interface

    if test_loopback_bind("127.0.0.1", socket.AF_INET):
        return "127.0.0.1"
    elif test_loopback_bind("::1", socket.AF_INET6):
        return "::1"
    else:
        raise RuntimeError(
            "Neither 127.0.0.1 nor ::1 are bound to a local interface. "
            "Set the FASTVIDEO_LOOPBACK_IP environment variable explicitly.")


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def get_distributed_init_method(ip: str, port: int) -> str:
    return get_tcp_uri(ip, port)


def get_tcp_uri(ip: str, port: int) -> str:
    if is_valid_ipv6_address(ip):
        return f"tcp://[{ip}]:{port}"
    else:
        return f"tcp://{ip}:{port}"


def get_open_port(port: int | None = None) -> int:
    if port is not None:
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", port))
                    return port
            except OSError:
                port += 1  # Increment port number if already in use
                logger.info("Port %d is already in use, trying port %d",
                            port - 1, port)
    # try ipv4
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        # try ipv6
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]


def cuda_is_initialized() -> bool:
    """Check if CUDA is initialized."""
    if not torch.cuda._is_compiled():
        return False
    return torch.cuda.is_initialized()


def xpu_is_initialized() -> bool:
    """Check if XPU is initialized."""
    if not torch.xpu._is_compiled():
        return False
    return torch.xpu.is_initialized()


def force_spawn() -> None:
    if os.environ.get("FASTVIDEO_WORKER_MULTIPROC_METHOD") == "fork":
        logger.warning("We must use the `spawn` multiprocessing start method.")
        os.environ["FASTVIDEO_WORKER_MULTIPROC_METHOD"] = "spawn"


def get_mp_context() -> BaseContext:
    """Get a multiprocessing context with a particular method (spawn or fork).
    By default we follow the value of the FASTVIDEO_WORKER_MULTIPROC_METHOD to
    determine the multiprocessing method (default is fork). However, under
    certain conditions, we may enforce spawn and override the value of
    FASTVIDEO_WORKER_MULTIPROC_METHOD.
    """
    force_spawn()
    mp_method = envs.FASTVIDEO_WORKER_MULTIPROC_METHOD
    return multiprocessing.get_context(mp_method)


# ANSI color codes
CYAN = '\033[1;36m'
RESET = '\033[0;0m'


def _add_prefix(file: TextIO, worker_name: str, pid: int) -> None:
    """Prepend each output line with process-specific prefix"""

    prefix = f"{CYAN}({worker_name} pid={pid}){RESET} "
    file_write = file.write

    def write_with_prefix(s: str):
        if not s:
            return
        if file.start_new_line:  # type: ignore[attr-defined]
            file_write(prefix)
        idx = 0
        while (next_idx := s.find('\n', idx)) != -1:
            next_idx += 1
            file_write(s[idx:next_idx])
            if next_idx == len(s):
                file.start_new_line = True  # type: ignore[attr-defined]
                return
            file_write(prefix)
            idx = next_idx
        file_write(s[idx:])
        file.start_new_line = False  # type: ignore[attr-defined]

    file.start_new_line = True  # type: ignore[attr-defined]
    file.write = write_with_prefix  # type: ignore[method-assign]


def decorate_logs(process_name: str | None = None) -> None:
    """
    Adds a process-specific prefix to each line of output written to stdout and
    stderr.

    Args:
        process_name: Optional; the name of the process to use in the prefix.
            If not provided, the current process name from the multiprocessing
            context is used.
    """
    if process_name is None:
        process_name = get_mp_context().current_process().name
    pid = os.getpid()
    _add_prefix(sys.stdout, process_name, pid)
    _add_prefix(sys.stderr, process_name, pid)
