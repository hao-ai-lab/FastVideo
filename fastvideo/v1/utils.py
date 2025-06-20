# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/utils.py

import argparse
import ctypes
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
import signal
import socket
import sys
import tempfile
import threading
import traceback
from dataclasses import dataclass, fields, is_dataclass
from functools import lru_cache, partial, wraps
from typing import (Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar,
                    Union, cast)

import cloudpickle
import filelock
import torch
import yaml
from diffusers.loaders.lora_base import (
    _best_guess_weight_name)  # watch out for potetential removal from diffusers
from huggingface_hub import snapshot_download
from remote_pdb import RemotePdb

import fastvideo.v1.envs as envs
from fastvideo.v1.logger import init_logger

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
    We either use the library file specified by the `VLLM_NCCL_SO_PATH`
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


prev_set_stream = torch.cuda.set_stream

_current_stream = None


def _patched_set_stream(stream: torch.cuda.Stream) -> None:
    global _current_stream
    _current_stream = stream
    prev_set_stream(stream)


torch.cuda.set_stream = _patched_set_stream


def current_stream() -> torch.cuda.Stream:
    """
    replace `torch.cuda.current_stream()` with `fastvideo.v1.utils.current_stream()`.
    it turns out that `torch.cuda.current_stream()` is quite expensive,
    as it will construct a new stream object at each call.
    here we patch `torch.cuda.set_stream` to keep track of the current stream
    directly, so that we can avoid calling `torch.cuda.current_stream()`.

    the underlying hypothesis is that we do not call `torch._C._cuda_setStream`
    from C/C++ code.
    """
    from fastvideo.v1.platforms import current_platform
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

    def _pull_args_from_config(self, args: List[str]) -> List[str]:
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

    def _load_config_file(self, file_path: str) -> List[str]:
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

        processed_args: List[str] = []

        config: Dict[str, Any] = {}
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

        def process_dict(prefix: str, d: Dict[str, Any]):
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


def warn_for_unimplemented_methods(cls: Type[T]) -> Type[T]:
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
    import fastvideo.v1.third_party.pynvml as pynvml
    return pynvml


def maybe_download_model(model_name_or_path: str,
                         local_dir: Optional[str] = None,
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
                        local_dir: Optional[str] = None,
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


def verify_model_config_and_directory(model_path: str) -> Dict[str, Any]:
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
    return cast(Dict[str, Any], config)


def maybe_download_model_index(model_name_or_path: str) -> Dict[str, Any]:
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
                config: Dict[str, Any] = json.load(f)

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


def update_environment_variables(envs: Dict[str, str]):
    for k, v in envs.items():
        if k in os.environ and os.environ[k] != v:
            logger.warning(
                "Overwriting environment variable %s "
                "from '%s' to '%s'", k, os.environ[k], v)
        os.environ[k] = v


def run_method(obj: Any, method: Union[str, bytes, Callable], args: tuple[Any],
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


def shallow_asdict(obj) -> Dict[str, Any]:
    if not is_dataclass(obj):
        raise TypeError("Expected dataclass instance")
    return {f.name: getattr(obj, f.name) for f in fields(obj)}


def kill_itself_when_parent_died() -> None:
    # if sys.platform == "linux":
    # sigkill this process when parent worker manager dies
    PR_SET_PDEATHSIG = 1
    libc = ctypes.CDLL("libc.so.6")
    libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    # else:
    #     logger.warning("kill_itself_when_parent_died is only supported in linux.")


def get_exception_traceback() -> str:
    etype, value, tb = sys.exc_info()
    err_str = "".join(traceback.format_exception(etype, value, tb))
    return err_str


class TypeBasedDispatcher:

    def __init__(self, mapping: List[Tuple[Type, Callable]]):
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
    master_dtype: Optional[torch.dtype] = None
    param_dtype: Optional[torch.dtype] = None
    reduce_dtype: Optional[torch.dtype] = None
    output_dtype: Optional[torch.dtype] = None
    compute_dtype: Optional[torch.dtype] = None


# Thread-local storage for mixed precision state
_mixed_precision_state = threading.local()


def get_mixed_precision_state() -> MixedPrecisionState:
    """Get the current mixed precision state."""
    if not hasattr(_mixed_precision_state, 'state'):
        raise ValueError("Mixed precision state not set")
    return cast(MixedPrecisionState, _mixed_precision_state.state)


def set_mixed_precision_policy(master_dtype: torch.dtype,
                               param_dtype: torch.dtype,
                               reduce_dtype: torch.dtype,
                               output_dtype: Optional[torch.dtype] = None):
    """Set mixed precision policy globally.
    
    Args:
        param_dtype: Parameter dtype used for training
        reduce_dtype: Reduction dtype used for gradients
        output_dtype: Optional output dtype
    """
    state = MixedPrecisionState(
        master_dtype=master_dtype,
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=output_dtype,
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
    mask_strategy: Optional[Dict[str, Any]] = None,
    t_max: Optional[int] = None,
    l_max: Optional[int] = None,
    h_max: Optional[int] = None,
) -> List[List[List[Optional[torch.Tensor]]]]:
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
        # else: silently ignore any key that doesn’t fit

    return result


def set_random_seed(seed: int) -> None:
    from fastvideo.v1.platforms import current_platform
    current_platform.seed_everything(seed)


@lru_cache(maxsize=1)
def is_vsa_available() -> bool:
    return importlib.util.find_spec("vsa") is not None
