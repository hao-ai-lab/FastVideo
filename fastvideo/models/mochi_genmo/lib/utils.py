import os
import subprocess
import tempfile
import time

import numpy as np
from moviepy.editor import ImageSequenceClip
from PIL import Image

from genmo.lib.progress import get_new_progress_bar


class Timer:
    def __init__(self):
        self.times = {}  # Dictionary to store times per stage

    def __call__(self, name):
        print(f"Timing {name}")
        return self.TimerContextManager(self, name)

    def print_stats(self):
        total_time = sum(self.times.values())
        # Print table header
        print("{:<20} {:>10} {:>10}".format("Stage", "Time(s)", "Percent"))
        for name, t in self.times.items():
            percent = (t / total_time) * 100 if total_time > 0 else 0
            print("{:<20} {:>10.2f} {:>9.2f}%".format(name, t, percent))

    class TimerContextManager:
        def __init__(self, outer, name):
            self.outer = outer  # Reference to the Timer instance
            self.name = name
            self.start_time = None

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            end_time = time.perf_counter()
            elapsed = end_time - self.start_time
            self.outer.times[self.name] = self.outer.times.get(self.name, 0) + elapsed


def save_video(final_frames, output_path, fps=30):
    assert final_frames.ndim == 4 and final_frames.shape[3] == 3, f"invalid shape: {final_frames} (need t h w c)"
    if final_frames.dtype != np.uint8:
        final_frames = (final_frames * 255).astype(np.uint8)
    ImageSequenceClip(list(final_frames), fps=fps).write_videofile(output_path)


def create_memory_tracker():
    import torch

    previous = [None]  # Use list for mutable closure state

    def track(label="all2all"):
        current = torch.cuda.memory_allocated() / 1e9
        if previous[0] is not None:
            diff = current - previous[0]
            sign = "+" if diff >= 0 else ""
            print(f"GPU memory ({label}): {current:.2f} GB ({sign}{diff:.2f} GB)")
        else:
            print(f"GPU memory ({label}): {current:.2f} GB")
        previous[0] = current  # type: ignore

    return track
