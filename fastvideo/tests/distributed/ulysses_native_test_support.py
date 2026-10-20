# SPDX-License-Identifier: Apache-2.0
"""Load a separately built communication extension for development validation."""
import importlib.util
import runpy
import sys
from pathlib import Path

import torch  # noqa: F401 - load the extension's libtorch dependencies first


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def install_native_ops(source: Path, extension: Path):
    import fastvideo_kernel

    native = load_module('ulysses_native', extension)
    wrapper = load_module('fastvideo_kernel.comm_ops',
                          source / 'fastvideo-kernel/python/fastvideo_kernel/comm_ops.py')
    wrapper._ops = native
    fastvideo_kernel.comm_ops = wrapper
    return wrapper


if __name__ == '__main__':
    source, extension, target = map(Path, sys.argv[1:4])
    install_native_ops(source, extension)
    sys.argv = [str(target), *sys.argv[4:]]
    runpy.run_path(str(target), run_name='__main__')
