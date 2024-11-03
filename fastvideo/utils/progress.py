import contextlib
from typing import Any, Iterable, Iterator, Optional

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from ray.experimental.tqdm_ray import tqdm as ray_tqdm
except:
    ray_tqdm = None

# Global state
_current_progress_type = "none"
_is_progress_bar_active = False


class DummyProgressBar:
    """A no-op progress bar that mimics tqdm interface"""

    def __init__(self, iterable=None, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def update(self, n=1):
        pass

    def close(self):
        pass

    def set_description(self, desc):
        pass


def get_new_progress_bar(iterable: Optional[Iterable] = None, **kwargs) -> Any:
    if not _is_progress_bar_active:
        return DummyProgressBar(iterable=iterable, **kwargs)

    if _current_progress_type == "tqdm":
        if tqdm is None:
            raise ImportError("tqdm is required but not installed. Please install tqdm to use the tqdm progress bar.")
        return tqdm(iterable=iterable, **kwargs)
    elif _current_progress_type == "ray_tqdm":
        if ray_tqdm is None:
            raise ImportError("ray is required but not installed. Please install ray to use the ray_tqdm progress bar.")
        return ray_tqdm(iterable=iterable, **kwargs)
    return DummyProgressBar(iterable=iterable, **kwargs)


@contextlib.contextmanager
def progress_bar(type: str = "none", enabled=True):
    """
    Context manager for setting progress bar type and options.

    Args:
        type: Type of progress bar ("none" or "tqdm")
        **options: Options to pass to the progress bar (e.g., total, desc)

    Raises:
        ValueError: If progress bar type is invalid
        RuntimeError: If progress bars are nested

    Example:
        with progress_bar(type="tqdm", total=100):
            for i in get_new_progress_bar(range(100)):
                process(i)
    """
    if type not in ("none", "tqdm", "ray_tqdm"):
        raise ValueError("Progress bar type must be 'none' or 'tqdm' or 'ray_tqdm'")
    if not enabled:
        type = "none"
    global _current_progress_type, _is_progress_bar_active

    if _is_progress_bar_active:
        raise RuntimeError("Nested progress bars are not supported")

    _is_progress_bar_active = True
    _current_progress_type = type

    try:
        yield
    finally:
        _is_progress_bar_active = False
        _current_progress_type = "none"
