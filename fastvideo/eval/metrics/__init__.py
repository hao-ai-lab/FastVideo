"""Auto-discover and register all built-in metrics (recursive).

Manually walks the metrics package tree and imports any leaf-package's
`metric` module so @register decorators fire. Skips any path component
starting with `_` (e.g. _vendor_*, _third_party) to avoid pulling in
vendored model code that may have side-effecting imports.
"""

import importlib
import os


def _walk(path: str, prefix: str):
    """Recursively yield (module_name, is_pkg) for non-underscore packages."""
    for entry in os.listdir(path):
        if entry.startswith("_") or entry.startswith("."):
            continue
        full = os.path.join(path, entry)
        if os.path.isdir(full) and os.path.exists(os.path.join(full, "__init__.py")):
            sub_prefix = f"{prefix}.{entry}"
            yield (sub_prefix, True)
            yield from _walk(full, sub_prefix)


for _pkg_path in __path__:
    for _modname, _ispkg in _walk(_pkg_path, __name__):
        try:
            importlib.import_module(f"{_modname}.metric")
        except ModuleNotFoundError:
            pass
