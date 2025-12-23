import math
import os
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from typing import Any

import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange

from fastvideo.configs.sample import SamplingParam
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines import ForwardBatch
from fastvideo.utils import align_to, shallow_asdict
from fastvideo.worker.executor import Executor

logger = init_logger(__name__)


class StreamingVideoGenerator:
    def __init__(self, fastvideo_args: FastVideoArgs,
                 executor_class: type[Executor], log_stats: bool):
        self.fastvideo_args = fastvideo_args
        self.executor = executor_class(fastvideo_args)
        self.accumulated_frames: list[np.ndarray] = []
        self.sampling_param: SamplingParam | None = None
        self.batch: ForwardBatch | None = None
        self._save_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video_save_")
        self._save_future: Future | None = None

    @classmethod
    def from_pretrained(cls,
                        model_path: str,
                        device: str | None = None,
                        torch_dtype: torch.dtype | None = None,
                        **kwargs) -> "StreamingVideoGenerator":
        kwargs['model_path'] = model_path
        fastvideo_args = FastVideoArgs.from_kwargs(**kwargs)
        return cls.from_fastvideo_args(fastvideo_args)

    @classmethod
    def from_fastvideo_args(cls,
                            fastvideo_args: FastVideoArgs) -> "StreamingVideoGenerator":
        executor_class = Executor.get_class(fastvideo_args)
        return cls(
            fastvideo_args=fastvideo_args,
            executor_class=executor_class,
            log_stats=False,
        )

    def reset(self, 
              prompt: str = "A gameplay video of a cyberpunk city",
              image_path: str | None = None,
              num_frames: int = 120, # Default max frames
              **kwargs) -> list[np.ndarray]:
        self.accumulated_frames = []
        self.executor.execute_streaming_clear()

        # Handle batch processing from text file
        if self.sampling_param is None:
            self.sampling_param = SamplingParam.from_pretrained(
                self.fastvideo_args.model_path)
            
        self.sampling_param.update(kwargs)
        self.sampling_param.prompt = prompt
        if image_path:
            self.sampling_param.image_path = image_path
        self.sampling_param.num_frames = num_frames

        fastvideo_args = self.fastvideo_args
        
        self.sampling_param.height = align_to(self.sampling_param.height, 16)
        self.sampling_param.width = align_to(self.sampling_param.width, 16)
        
        latents_size = [(self.sampling_param.num_frames - 1) // 4 + 1,
                        self.sampling_param.height // 8, self.sampling_param.width // 8]
        n_tokens = latents_size[0] * latents_size[1] * latents_size[2]

        self.sampling_param.return_frames = True 
        self.sampling_param.save_video = False
        
        self.batch = ForwardBatch(
            **shallow_asdict(self.sampling_param),
            eta=0.0,
            n_tokens=n_tokens,
            VSA_sparsity=fastvideo_args.VSA_sparsity,
        )
        
        logger.info("Initializing streaming generation...")
        start_time = time.perf_counter()
        output_batch = self.executor.execute_streaming_reset(self.batch, fastvideo_args)
        
        frames = self._process_output_batch(output_batch)
        self.accumulated_frames.extend(frames)
        logger.info(f"Initialized. Generated {len(frames)} frames in {time.perf_counter() - start_time:.2f}s")
        return frames

    def step(self, keyboard_cond: torch.Tensor, mouse_cond: torch.Tensor, 
             save_path: str | None = None) -> list[np.ndarray]:
        if self.batch is None:
            raise RuntimeError("Call reset() before step()")
        
        # Wait for previous save to complete before starting new inference
        self._wait_for_save()
        
        start_time = time.perf_counter()
        output_batch = self.executor.execute_streaming_step(
            keyboard_action=keyboard_cond, 
            mouse_action=mouse_cond
        )
        
        frames = self._process_output_batch(output_batch)
        if len(frames) > 0:
            self.accumulated_frames.extend(frames)
        
        step_time = time.perf_counter() - start_time
        logger.debug(f"Step completed in {step_time:.2f}s, got {len(frames)} frames")

        if save_path:
            self._start_background_save(save_path)
            
        return frames

    async def step_async(self, keyboard_cond: torch.Tensor, mouse_cond: torch.Tensor, save_path: str | None = None) -> list[np.ndarray]:
        if self.batch is None:
            raise RuntimeError("Call reset() before step_async()")

        self._wait_for_save()

        start_time = time.perf_counter()
        output_batch = await self.executor.execute_streaming_step_async(
            keyboard_action=keyboard_cond,
            mouse_action=mouse_cond,
        )

        frames = self._process_output_batch(output_batch)
        if len(frames) > 0:
            self.accumulated_frames.extend(frames)

        step_time = time.perf_counter() - start_time
        logger.debug(f"Async step completed in {step_time:.2f}s, got {len(frames)} frames")

        if save_path:
            self._start_background_save(save_path)

        return frames

    def _wait_for_save(self) -> None:
        if self._save_future is not None and not self._save_future.done():
            self._save_future.result()  # Block until done
            
    def _start_background_save(self, output_path: str, fps: int = 24) -> None:
        # Make a copy of frames list to avoid race conditions
        frames_copy = list(self.accumulated_frames)
        
        def save_task():
            logger.info(f"Saving {len(frames_copy)} frames to {output_path}...")
            imageio.mimsave(output_path, frames_copy, fps=fps, format="mp4")
            logger.info(f"Saved video to {output_path}")
            
        self._save_future = self._save_executor.submit(save_task)

    def finalize(self, output_path: str = "streaming_output.mp4", fps: int = 24) -> str:
        self._wait_for_save()
        
        if not self.accumulated_frames:
            logger.warning("No frames to save.")
            return ""
            
        logger.info(f"Saving {len(self.accumulated_frames)} frames to {output_path}...")
        imageio.mimsave(output_path, self.accumulated_frames, fps=fps, format="mp4")
        
        self.executor.execute_streaming_clear()
        self.accumulated_frames = []
        return output_path

    def _process_output_batch(self, output_batch: ForwardBatch) -> list[np.ndarray]:
        if output_batch.output is None:
            return []
            
        samples = output_batch.output
        # [B, C, T, H, W] or [1, C, T, H, W]
        if len(samples.shape) == 5:
             # Rearrange to [T, B, C, H, W] for processing loop
             videos = rearrange(samples, "b c t h w -> t b c h w")
        else:
             logger.warning(f"Unexpected output shape: {samples.shape}")
             return []

        frames = []
        for x in videos:
            x = torchvision.utils.make_grid(x, nrow=1) 
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
            frames.append((x * 255).cpu().numpy().astype(np.uint8))
            
        return frames

    def shutdown(self):
        self._wait_for_save()
        self._save_executor.shutdown(wait=True)
        self.executor.shutdown()
