import asyncio
import os
from concurrent.futures import Future, ThreadPoolExecutor

import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange

from fastvideo.configs.sample import SamplingParam
from fastvideo.entrypoints.video_generator import VideoGenerator
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines import ForwardBatch
from fastvideo.utils import align_to, shallow_asdict
from fastvideo.worker.executor import Executor
from fastvideo.worker.multiproc_executor import MultiprocExecutor

logger = init_logger(__name__)


class IncrementalVideoWriter:

    def __init__(
        self,
        path: str,
        fps: int = 24,
        block_dir: str | None = None,
        quality: int = 8,
    ):
        self._executor = ThreadPoolExecutor(max_workers=2,
                                            thread_name_prefix="video_write_")
        self._path = path
        # quality 0-10: higher = sharper MP4 (default 5 is often blurry)
        self._quality = min(10, max(0, quality))
        self._writer = imageio.get_writer(path,
                                          fps=fps,
                                          format="mp4",
                                          quality=self._quality)
        self._pending_main: Future | None = None
        self._block_dir = block_dir
        self._block_idx = 0
        self._fps = fps

    @property
    def path(self) -> str:
        return self._path

    def add_frames(self, frames: list[np.ndarray]) -> Future | None:
        # Wait for previous main video write to complete
        if self._pending_main is not None:
            self._pending_main.result()

        # Copy frames to avoid race conditions
        frames_copy = [f.copy() for f in frames]
        self._pending_main = self._executor.submit(self._write_frames,
                                                   frames_copy)

        # Write block file if block_dir is set
        block_future = None
        if self._block_dir:
            self._block_idx += 1
            block_path = os.path.join(self._block_dir,
                                      f"b{self._block_idx}.mp4")
            block_future = self._executor.submit(self._write_block, frames_copy,
                                                 block_path)
        return block_future

    def _write_frames(self, frames: list[np.ndarray]) -> None:
        for frame in frames:
            self._writer.append_data(frame)

    def _write_block(self, frames: list[np.ndarray], path: str) -> str:
        imageio.mimsave(path,
                        frames,
                        fps=self._fps,
                        format="mp4",
                        quality=self._quality)
        return path

    def close(self) -> None:
        if self._pending_main is not None:
            self._pending_main.result()
            self._pending_main = None
        if self._writer:
            self._writer.close()
            self._writer = None
        self._executor.shutdown(wait=True)


class StreamingVideoGenerator(VideoGenerator):
    """
    This class extends VideoGenerator with streaming capabilities,
    allowing incremental video generation with step-by-step control.
    """

    def __init__(self,
                 fastvideo_args: FastVideoArgs,
                 executor_class: type[Executor],
                 log_stats: bool,
                 use_queue_mode: bool = True):
        super().__init__(fastvideo_args, executor_class, log_stats)
        self.accumulated_frames: list[np.ndarray] = []
        self.sampling_param: SamplingParam | None = None
        self.batch: ForwardBatch | None = None
        self._use_queue_mode = use_queue_mode and isinstance(
            self.executor, MultiprocExecutor)
        self.writer: IncrementalVideoWriter | None = None
        self.block_dir: str | None = None
        self.block_idx: int = 0

    @classmethod
    def from_fastvideo_args(
            cls, fastvideo_args: FastVideoArgs) -> "StreamingVideoGenerator":
        executor_class = Executor.get_class(fastvideo_args)
        return cls(
            fastvideo_args=fastvideo_args,
            executor_class=executor_class,
            log_stats=False,
        )

    def reset(
            self,
            prompt: str = "A gameplay video of a cyberpunk city",
            image_path: str | None = None,
            num_frames: int = 120,  # Default max frames
            **kwargs):
        self.accumulated_frames = []
        self.block_idx = 0
        self.block_dir = None
        if self.writer:
            self.writer.close()
            self.writer = None
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

        if "output_path" in kwargs:
            output_path = self._prepare_output_path(kwargs["output_path"],
                                                    prompt)
            # Create block directory for individual block files
            block_dir = output_path.replace(".mp4", "")
            os.makedirs(block_dir, exist_ok=True)
            self.block_dir = block_dir
            # Use model fps (e.g. 60 for Waypoint) and higher quality for less blur
            fps = getattr(self.sampling_param, "fps", 24)
            quality = getattr(self.sampling_param, "video_quality", 8)
            self.writer = IncrementalVideoWriter(output_path,
                                                 fps=fps,
                                                 block_dir=block_dir,
                                                 quality=quality)

        fastvideo_args = self.fastvideo_args

        self.sampling_param.height = align_to(self.sampling_param.height, 16)
        self.sampling_param.width = align_to(self.sampling_param.width, 16)

        latents_size = [(self.sampling_param.num_frames - 1) // 4 + 1,
                        self.sampling_param.height // 8,
                        self.sampling_param.width // 8]
        n_tokens = latents_size[0] * latents_size[1] * latents_size[2]

        self.sampling_param.return_frames = True
        self.sampling_param.save_video = False

        batch_kw = shallow_asdict(self.sampling_param)
        batch_kw.pop("video_quality",
                     None)  # writer-only, not a ForwardBatch field
        self.batch = ForwardBatch(
            **batch_kw,
            eta=0.0,
            n_tokens=n_tokens,
            VSA_sparsity=fastvideo_args.VSA_sparsity,
        )

        if self._use_queue_mode:
            self.executor.submit_reset(self.batch, fastvideo_args)
            result = self.executor.wait_result()
            if result.error:
                raise result.error
        else:
            self.executor.execute_streaming_reset(self.batch, fastvideo_args)

    def step(
            self, keyboard_cond: torch.Tensor,
            mouse_cond: torch.Tensor) -> tuple[list[np.ndarray], Future | None]:
        if self.batch is None:
            raise RuntimeError("Call reset() before step()")

        if self._use_queue_mode and self.executor._streaming_enabled:
            self.executor.submit_step(keyboard_cond, mouse_cond)
            result = self.executor.wait_result()
            if result.error:
                raise result.error
            output_batch = result.output_batch
        else:
            # Fallback to RPC-based
            output_batch = self.executor.execute_streaming_step(
                keyboard_action=keyboard_cond, mouse_action=mouse_cond)

        frames = self._process_output_batch(output_batch)
        block_future = None
        if len(frames) > 0:
            self.accumulated_frames.extend(frames)
            self.block_idx += 1
            if self.writer:
                # Returns Future for block file, or None if no block_dir
                block_future = self.writer.add_frames(frames)

        return frames, block_future

    async def step_async(
            self, keyboard_cond: torch.Tensor,
            mouse_cond: torch.Tensor) -> tuple[list[np.ndarray], Future | None]:
        if self.batch is None:
            raise RuntimeError("Call reset() before step_async()")

        if self._use_queue_mode and self.executor._streaming_enabled:
            self.executor.submit_step(keyboard_cond, mouse_cond)

            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, self.executor.wait_result)

            if result.error:
                raise result.error
            output_batch = result.output_batch
        else:
            # Fallback to RPC-based
            output_batch = await self.executor.execute_streaming_step_async(
                keyboard_action=keyboard_cond,
                mouse_action=mouse_cond,
            )

        frames = self._process_output_batch(output_batch)
        block_future = None
        if len(frames) > 0:
            self.accumulated_frames.extend(frames)
            self.block_idx += 1
            if self.writer:
                block_future = self.writer.add_frames(frames)

        return frames, block_future

    def finalize(self,
                 output_path: str = "streaming_output.mp4",
                 fps: int = 24) -> str:
        if not self.accumulated_frames:
            logger.warning("No frames to save.")
            return ""

        if self.writer:
            output_path = self.writer.path
            self.writer.close()
            self.writer = None
            logger.info("Saved video to %s", output_path)
        else:
            imageio.mimsave(
                output_path,
                self.accumulated_frames,
                fps=fps,
                format="mp4",
                quality=8,
            )
            logger.info("Saved video to %s", output_path)

        if self._use_queue_mode and self.executor._streaming_enabled:
            self.executor.submit_clear()
        else:
            self.executor.execute_streaming_clear()
        self.accumulated_frames = []
        return output_path

    def _process_output_batch(self,
                              output_batch: ForwardBatch) -> list[np.ndarray]:
        if output_batch.output is None:
            return []

        samples = output_batch.output
        # [B, C, T, H, W] or [1, C, T, H, W]
        if len(samples.shape) == 5:
            # Rearrange to [T, B, C, H, W] for processing loop
            videos = rearrange(samples, "b c t h w -> t b c h w")
        else:
            logger.warning("Unexpected output shape: %s", samples.shape)
            return []

        frames = []
        target_h = getattr(self.sampling_param, "height", None)
        target_w = getattr(self.sampling_param, "width", None)
        pad_for_writer = (self.writer is not None and target_h is not None
                          and target_w is not None)
        for x in videos:
            x = torchvision.utils.make_grid(x, nrow=1)
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
            # Support both float [0, 1] and uint8 [0, 255] from pipelines
            if x.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32,
                           torch.int64):
                arr = x.cpu().numpy()
            else:
                arr = (x.float().clamp(0.0, 1.0).cpu().numpy() * 255.0)
            arr = arr.astype(np.uint8)
            # Pad to target size so imageio does not resize (avoids macro_block_size warning)
            if pad_for_writer and (arr.shape[0] != target_h
                                   or arr.shape[1] != target_w):
                padded = np.zeros((target_h, target_w, arr.shape[2]),
                                  dtype=arr.dtype)
                padded[:arr.shape[0], :arr.shape[1], :] = arr
                arr = padded
            frames.append(arr)

        return frames

    def shutdown(self):
        if self.writer:
            self.writer.close()
            self.writer = None

        if self._use_queue_mode and self.executor._streaming_enabled:
            self.executor.disable_streaming()

        super().shutdown()
