import asyncio
import multiprocessing as mp
import os
import subprocess
import traceback
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Process, Queue
from typing import Optional

import numpy as np
import torch

from config import (
    MODEL_CONFIG, MODEL_REGISTRY, DEFAULT_MODEL_ID,
    NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_INFERENCE_STEPS,
    MAX_USERS_PER_GPU,
)


class CommandType(Enum):
    """Commands sent from main process to GPU worker."""
    INIT = "init"
    STEP = "step"
    RESET = "reset"
    SHUTDOWN = "shutdown"
    # Multi-user commands
    USER_JOIN = "user_join"
    USER_STEP = "user_step"
    USER_LEAVE = "user_leave"
    RELOAD_MODEL = "reload_model"


@dataclass
class Command:
    """Command sent to GPU worker subprocess."""
    type: CommandType
    data: dict = None
    user_id: str = None

    def __post_init__(self):
        if self.data is None:
            self.data = {}


@dataclass
class Response:
    """Response from GPU worker subprocess."""
    success: bool
    frames: Optional[list[np.ndarray]] = None
    error: Optional[str] = None
    timings: Optional[dict] = None
    user_id: Optional[str] = None


def gpu_worker_process(
    gpu_id: int,
    cuda_device: str,
    command_queue: Queue,
    response_queue: Queue,
):
    """
    Worker process that runs on a single GPU.

    This function runs in a subprocess with CUDA_VISIBLE_DEVICES set to a single GPU.
    Supports both single-user mode (legacy) and multi-user mode (ORCA batching).
    """
    # Set CUDA_VISIBLE_DEVICES BEFORE importing torch or any CUDA code
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "FLASH_ATTN"

    # Now import the generator (this will initialize CUDA with the single visible GPU)
    from fastvideo.entrypoints.streaming_generator import StreamingVideoGenerator
    from fastvideo.models.dits.matrixgame.utils import expand_action_to_frames
    from fastvideo.configs.sample.base import SamplingParam
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    from fastvideo.utils import align_to, shallow_asdict
    from fastvideo.worker.multiproc_executor import (
        StreamingTask, StreamingTaskType, StreamingResult,
    )

    import gc
    import time

    generator = None
    current_model_config = dict(MODEL_CONFIG)

    def _gpu_mem():
        a = torch.cuda.memory_allocated() / 1024**3
        r = torch.cuda.memory_reserved() / 1024**3
        return f"alloc={a:.2f}GiB, reserved={r:.2f}GiB"

    def initialize_generator(model_config=None):
        nonlocal generator, current_model_config
        if model_config is not None:
            current_model_config = model_config

        # Free old generator if reloading
        if generator is not None:
            print(f"[GPU {gpu_id}] Freeing old model...")
            try:
                generator.shutdown()
            except Exception:
                pass
            del generator
            generator = None
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[GPU {gpu_id}] After cleanup: {_gpu_mem()}")

        print(f"[GPU {gpu_id}] Loading model: {current_model_config['model_path']}")
        if current_model_config.get("init_weights_from_safetensors"):
            print(f"[GPU {gpu_id}] With custom weights: {current_model_config['init_weights_from_safetensors']}")
        print(f"[GPU {gpu_id}] Before model load: {_gpu_mem()}")

        load_kwargs = dict(
            num_gpus=1,
            dit_layerwise_offload=False,
            use_fsdp_inference=False,
            dit_cpu_offload=False,
            vae_cpu_offload=False,
            text_encoder_cpu_offload=True,
            pin_cpu_memory=True,
        )
        if current_model_config.get("init_weights_from_safetensors"):
            load_kwargs["init_weights_from_safetensors"] = current_model_config["init_weights_from_safetensors"]
        if current_model_config.get("override_transformer_cls_name"):
            load_kwargs["override_transformer_cls_name"] = current_model_config["override_transformer_cls_name"]
        if current_model_config.get("override_pipeline_cls_name"):
            load_kwargs["override_pipeline_cls_name"] = current_model_config["override_pipeline_cls_name"]

        generator = StreamingVideoGenerator.from_pretrained(
            current_model_config["model_path"],
            **load_kwargs,
        )

        print(f"[GPU {gpu_id}] After model load: {_gpu_mem()}")

        # Warmup: run initial reset to exercise the full pipeline once.
        # This validates the model can encode 597 frames, warms up CUDA,
        # and allocates KV caches. The caches will be freed when the first
        # multi-user join arrives (add_user cleans up existing state).
        actions = {
            "keyboard": torch.zeros((NUM_FRAMES, current_model_config["keyboard_dim"])),
            "mouse": torch.zeros((NUM_FRAMES, 2))
        }
        generator.reset(
            prompt="",
            image_path=current_model_config.get("image_path", current_model_config["image_url"]),
            mouse_cond=actions["mouse"].unsqueeze(0),
            keyboard_cond=actions["keyboard"].unsqueeze(0),
            grid_sizes=torch.tensor([150, 44, 80]),
            num_frames=NUM_FRAMES,
            height=FRAME_HEIGHT,
            width=FRAME_WIDTH,
            num_inference_steps=NUM_INFERENCE_STEPS,
        )

        print(f"[GPU {gpu_id}] After warmup: {_gpu_mem()}")
        print(f"[GPU {gpu_id}] Model loaded and ready")

    def do_step(keyboard_vector: list, mouse_vector: list) -> tuple[list[np.ndarray], dict]:
        """Execute a generation step with timing."""
        import time
        timings = {}

        # Prepare action tensors
        t0 = time.perf_counter()
        action = {
            "keyboard": torch.tensor(keyboard_vector).cuda(),
            "mouse": torch.tensor(mouse_vector).cuda()
        }
        keyboard_cond, mouse_cond = expand_action_to_frames(action, 12)
        torch.cuda.synchronize()
        timings["action_prep_ms"] = (time.perf_counter() - t0) * 1000

        # Model step (DiT + VAE)
        t0 = time.perf_counter()
        frames, _, stage_timings = generator.step(keyboard_cond, mouse_cond)
        torch.cuda.synchronize()
        timings["model_step_ms"] = (time.perf_counter() - t0) * 1000

        # Add stage timings if available
        if stage_timings:
            timings.update(stage_timings)

        return (frames if frames is not None else [], timings)

    def do_reset() -> list[np.ndarray]:
        """Reset the generator and return initial frames."""
        # Clear local state
        generator.accumulated_frames = []
        generator.block_idx = 0
        generator.block_dir = None
        if generator.writer:
            generator.writer.close()
            generator.writer = None

        # Use queue mode to clear
        if generator._use_queue_mode and generator.executor._streaming_enabled:
            generator.executor.submit_clear()
            clear_result = generator.executor.wait_result()
            if clear_result.error:
                raise clear_result.error

        # Set up sampling parameters
        actions = {
            "keyboard": torch.zeros((NUM_FRAMES, current_model_config["keyboard_dim"])),
            "mouse": torch.zeros((NUM_FRAMES, 2))
        }

        if generator.sampling_param is None:
            generator.sampling_param = SamplingParam.from_pretrained(
                generator.fastvideo_args.model_path)

        generator.sampling_param.update({
            "prompt": "",
            "image_path": current_model_config.get("image_path", current_model_config["image_url"]),
            "mouse_cond": actions["mouse"].unsqueeze(0),
            "keyboard_cond": actions["keyboard"].unsqueeze(0),
            "grid_sizes": torch.tensor([150, 44, 80]),
            "num_frames": NUM_FRAMES,
            "height": FRAME_HEIGHT,
            "width": FRAME_WIDTH,
            "num_inference_steps": NUM_INFERENCE_STEPS,
        })

        generator.sampling_param.height = align_to(generator.sampling_param.height, 16)
        generator.sampling_param.width = align_to(generator.sampling_param.width, 16)

        latents_size = [
            (generator.sampling_param.num_frames - 1) // 4 + 1,
            generator.sampling_param.height // 8,
            generator.sampling_param.width // 8
        ]
        n_tokens = latents_size[0] * latents_size[1] * latents_size[2]

        generator.sampling_param.return_frames = True
        generator.sampling_param.save_video = False

        # Create new batch
        generator.batch = ForwardBatch(
            **shallow_asdict(generator.sampling_param),
            eta=0.0,
            n_tokens=n_tokens,
            VSA_sparsity=generator.fastvideo_args.VSA_sparsity,
        )

        # Use queue mode to reset
        if generator._use_queue_mode:
            generator.executor.submit_reset(generator.batch, generator.fastvideo_args)
            result = generator.executor.wait_result()
            if result.error:
                raise result.error
        else:
            generator.executor.execute_streaming_reset(generator.batch, generator.fastvideo_args)

        # Generate initial frame
        return do_step([0, 0, 0, 0], [0, 0])

    def prepare_user_batch() -> ForwardBatch:
        """Create a fresh ForwardBatch for a new user joining."""
        actions = {
            "keyboard": torch.zeros((NUM_FRAMES, current_model_config["keyboard_dim"])),
            "mouse": torch.zeros((NUM_FRAMES, 2))
        }

        if generator.sampling_param is None:
            generator.sampling_param = SamplingParam.from_pretrained(
                generator.fastvideo_args.model_path)

        generator.sampling_param.update({
            "prompt": "",
            "image_path": current_model_config.get("image_path", current_model_config["image_url"]),
            "mouse_cond": actions["mouse"].unsqueeze(0),
            "keyboard_cond": actions["keyboard"].unsqueeze(0),
            "grid_sizes": torch.tensor([150, 44, 80]),
            "num_frames": NUM_FRAMES,
            "height": FRAME_HEIGHT,
            "width": FRAME_WIDTH,
            "num_inference_steps": NUM_INFERENCE_STEPS,
        })

        generator.sampling_param.height = align_to(generator.sampling_param.height, 16)
        generator.sampling_param.width = align_to(generator.sampling_param.width, 16)

        latents_size = [
            (generator.sampling_param.num_frames - 1) // 4 + 1,
            generator.sampling_param.height // 8,
            generator.sampling_param.width // 8
        ]
        n_tokens = latents_size[0] * latents_size[1] * latents_size[2]

        generator.sampling_param.return_frames = True
        generator.sampling_param.save_video = False

        batch = ForwardBatch(
            **shallow_asdict(generator.sampling_param),
            eta=0.0,
            n_tokens=n_tokens,
            VSA_sparsity=generator.fastvideo_args.VSA_sparsity,
        )
        return batch

    def multi_user_event_loop(first_cmd: Command = None):
        """Non-blocking event loop for multi-user mode.

        Drains commands from command_queue, submits them to the executor
        without blocking, and collects results asynchronously. This enables
        actual multi-user batching by allowing multiple USER_STEP commands
        to be submitted before waiting for results.
        """
        import queue as queue_module

        pending_results = 0

        def reload_model(cmd: Command):
            """Reload the model. Drains pending results first."""
            nonlocal pending_results
            # Wait for all pending results to complete
            while pending_results > 0:
                result = generator.executor.get_result(timeout=5.0)
                if result is not None:
                    pending_results -= 1
                    # Route result to response queue
                    if result.error:
                        response_queue.put(Response(
                            success=False, error=str(result.error),
                            user_id=result.user_id))
                    else:
                        response_queue.put(Response(
                            success=True, user_id=result.user_id))

            model_config = cmd.data.get("model_config")
            initialize_generator(model_config)

            # Re-enable streaming for the new executor
            if not generator.executor._streaming_enabled:
                generator.executor.enable_streaming()

            response_queue.put(Response(success=True, user_id=cmd.user_id))

        def submit_command(cmd: Command):
            """Submit a command to the executor. Returns True if a result is expected."""
            nonlocal pending_results
            if cmd.type == CommandType.USER_JOIN:
                batch = prepare_user_batch()
                generator.executor.submit_user_join(
                    cmd.user_id, batch, generator.fastvideo_args)
                pending_results += 1
            elif cmd.type == CommandType.USER_STEP:
                action = {
                    "keyboard": torch.tensor(
                        cmd.data.get("keyboard", [0, 0, 0, 0])).cuda(),
                    "mouse": torch.tensor(
                        cmd.data.get("mouse", [0, 0])).cuda()
                }
                keyboard_cond, mouse_cond = expand_action_to_frames(action, 12)
                torch.cuda.synchronize()
                generator.executor.submit_user_step(
                    cmd.user_id, keyboard_cond, mouse_cond)
                pending_results += 1
            elif cmd.type == CommandType.USER_LEAVE:
                generator.executor.submit_user_leave(cmd.user_id)
                pending_results += 1

        def collect_results():
            """Collect available results from executor (non-blocking)."""
            nonlocal pending_results
            collected = False
            while pending_results > 0:
                result = generator.executor.get_result(timeout=0)
                if result is None:
                    break
                collected = True
                pending_results -= 1

                if result.error:
                    response_queue.put(Response(
                        success=False, error=str(result.error),
                        user_id=result.user_id))
                elif result.task_type == StreamingTaskType.USER_STEP:
                    frames = []
                    timings = {}
                    if (result.output_batch is not None
                            and result.output_batch.output is not None):
                        frames = _process_output_to_frames(result.output_batch)
                        stage_timings = getattr(
                            result.output_batch, 'stage_timings', None)
                        if stage_timings:
                            timings.update(stage_timings)
                    response_queue.put(Response(
                        success=True, frames=frames, timings=timings,
                        user_id=result.user_id))
                elif result.task_type == StreamingTaskType.USER_JOIN:
                    print(f"[GPU {gpu_id}] User {result.user_id[:8]} joined")
                    response_queue.put(Response(
                        success=True, user_id=result.user_id))
                elif result.task_type == StreamingTaskType.USER_LEAVE:
                    print(f"[GPU {gpu_id}] User {result.user_id[:8]} left")
                    response_queue.put(Response(
                        success=True, user_id=result.user_id))
            return collected

        # Handle the first command that triggered the switch
        if first_cmd is not None:
            if first_cmd.type == CommandType.SHUTDOWN:
                generator.shutdown()
                response_queue.put(Response(success=True))
                return
            try:
                submit_command(first_cmd)
            except Exception as e:
                response_queue.put(Response(
                    success=False, error=str(e),
                    user_id=first_cmd.user_id))

        print(f"[GPU {gpu_id}] Entered multi-user event loop")

        while True:
            did_work = False

            # 1. Drain all available commands (non-blocking)
            #    If nothing pending, block for the first command
            if pending_results == 0:
                try:
                    cmd = command_queue.get(timeout=0.1)
                    if cmd.type == CommandType.SHUTDOWN:
                        generator.shutdown()
                        response_queue.put(Response(success=True))
                        return
                    if cmd.type == CommandType.RELOAD_MODEL:
                        try:
                            reload_model(cmd)
                        except Exception as e:
                            response_queue.put(Response(
                                success=False, error=str(e),
                                user_id=cmd.user_id))
                        did_work = True
                        continue
                    try:
                        submit_command(cmd)
                        did_work = True
                    except Exception as e:
                        response_queue.put(Response(
                            success=False, error=str(e),
                            user_id=cmd.user_id))
                except queue_module.Empty:
                    continue

            # Drain remaining commands (non-blocking)
            while True:
                try:
                    cmd = command_queue.get_nowait()
                    if cmd.type == CommandType.SHUTDOWN:
                        generator.shutdown()
                        response_queue.put(Response(success=True))
                        return
                    if cmd.type == CommandType.RELOAD_MODEL:
                        try:
                            reload_model(cmd)
                        except Exception as e:
                            response_queue.put(Response(
                                success=False, error=str(e),
                                user_id=cmd.user_id))
                        continue
                    try:
                        submit_command(cmd)
                        did_work = True
                    except Exception as e:
                        response_queue.put(Response(
                            success=False, error=str(e),
                            user_id=cmd.user_id))
                except queue_module.Empty:
                    break

            # 2. Collect results from executor (non-blocking)
            if collect_results():
                did_work = True

            # 3. Brief sleep to avoid busy-waiting when results are pending
            if not did_work and pending_results > 0:
                time.sleep(0.001)

    def _process_output_to_frames(output_batch) -> list[np.ndarray]:
        """Convert output batch tensor to list of numpy frames."""
        import torchvision
        from einops import rearrange

        samples = output_batch.output
        if samples is None:
            return []
        if len(samples.shape) == 5:
            videos = rearrange(samples, "b c t h w -> t b c h w")
        else:
            return []
        frames = []
        for x in videos:
            x = torchvision.utils.make_grid(x, nrow=1)
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
            frames.append((x * 255).cpu().numpy().astype(np.uint8))
        return frames

    # Main worker loop
    print(f"[GPU {gpu_id}] Worker process starting...")

    try:
        while True:
            cmd: Command = command_queue.get()

            if cmd.type == CommandType.SHUTDOWN:
                print(f"[GPU {gpu_id}] Shutting down...")
                if generator:
                    generator.shutdown()
                response_queue.put(Response(success=True))
                break

            elif cmd.type == CommandType.INIT:
                try:
                    initialize_generator()
                    response_queue.put(Response(success=True))
                except Exception as e:
                    print(f"[GPU {gpu_id}] Init error: {e}")
                    traceback.print_exc()
                    response_queue.put(Response(success=False, error=str(e)))

            elif cmd.type == CommandType.RELOAD_MODEL:
                try:
                    model_config = cmd.data.get("model_config")
                    initialize_generator(model_config)
                    response_queue.put(Response(success=True, user_id=cmd.user_id))
                except Exception as e:
                    print(f"[GPU {gpu_id}] Reload error: {e}")
                    traceback.print_exc()
                    response_queue.put(Response(
                        success=False, error=str(e), user_id=cmd.user_id))

            elif cmd.type == CommandType.STEP:
                try:
                    keyboard = cmd.data.get("keyboard", [0, 0, 0, 0])
                    mouse = cmd.data.get("mouse", [0, 0])
                    frames, timings = do_step(keyboard, mouse)
                    response_queue.put(Response(success=True, frames=frames, timings=timings))
                except Exception as e:
                    print(f"[GPU {gpu_id}] Step error: {e}")
                    traceback.print_exc()
                    response_queue.put(Response(success=False, error=str(e)))

            elif cmd.type == CommandType.RESET:
                try:
                    frames = do_reset()
                    response_queue.put(Response(success=True, frames=frames))
                except Exception as e:
                    print(f"[GPU {gpu_id}] Reset error: {e}")
                    traceback.print_exc()
                    response_queue.put(Response(success=False, error=str(e)))

            elif cmd.type in (CommandType.USER_JOIN,
                              CommandType.USER_STEP,
                              CommandType.USER_LEAVE):
                # Enable streaming if needed (synchronous RPC)
                if not generator.executor._streaming_enabled:
                    generator.executor.enable_streaming()
                # Switch to non-blocking event loop for multi-user mode
                multi_user_event_loop(first_cmd=cmd)
                break  # event loop handles everything until shutdown

    except Exception as e:
        print(f"[GPU {gpu_id}] Worker crashed: {e}")
        traceback.print_exc()

    print(f"[GPU {gpu_id}] Worker process exiting")


class GPUSlot:
    """Manages a single GPU worker subprocess with multi-client support."""

    def __init__(self, gpu_id: int, cuda_device: str):
        self.gpu_id = gpu_id
        self.cuda_device = cuda_device
        self.process: Optional[Process] = None
        self.command_queue: Optional[Queue] = None
        self.response_queue: Optional[Queue] = None
        self.ready: bool = False
        self._lock = asyncio.Lock()

        # Multi-client state
        self.connected_users: set[str] = set()  # all connected user IDs
        self._pending_futures: dict[str, asyncio.Future] = {}  # user_id -> pending Future
        self._response_reader_task: Optional[asyncio.Task] = None
        self._multi_user_mode: bool = False
        self._reader_lock: Optional[asyncio.Lock] = None
        self.current_model_id: str = DEFAULT_MODEL_ID

    @property
    def client_count(self) -> int:
        return len(self.connected_users)

    @property
    def is_available(self) -> bool:
        """A GPU is available if it has capacity for more users."""
        alive = self.ready and self.process is not None and self.process.is_alive()
        if not alive:
            return False
        if self._multi_user_mode:
            return len(self.connected_users) < MAX_USERS_PER_GPU
        else:
            return len(self.connected_users) == 0

    @property
    def is_empty(self) -> bool:
        return len(self.connected_users) == 0

    async def start(self):
        """Start the GPU worker subprocess."""
        ctx = mp.get_context("spawn")
        self.command_queue = ctx.Queue()
        self.response_queue = ctx.Queue()

        self.process = ctx.Process(
            target=gpu_worker_process,
            args=(self.gpu_id, self.cuda_device, self.command_queue, self.response_queue),
            daemon=False,
        )
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.process.start)

        # Send init command and wait for response
        response = await self._send_command(
            Command(CommandType.INIT), timeout=600.0)
        if not response.success:
            raise RuntimeError(
                f"GPU {self.gpu_id} failed to initialize: {response.error}")

        self.ready = True

    async def _send_command(
        self, cmd: Command, timeout: float = 300.0
    ) -> Response:
        """Send a command and wait for response (single-user mode)."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.command_queue.put, cmd)

        def get_response():
            return self.response_queue.get(timeout=timeout)

        response = await loop.run_in_executor(None, get_response)
        return response

    async def _send_command_tagged(
        self, cmd: Command, timeout: float = 300.0
    ) -> Response:
        """Send a tagged command and wait for the matching response (multi-user mode).

        Uses the response reader background task to route responses.
        """
        loop = asyncio.get_event_loop()
        future = loop.create_future()

        # Register this request's pending future
        self._pending_futures[cmd.user_id] = future

        # Send command
        await loop.run_in_executor(None, self.command_queue.put, cmd)

        # Ensure response reader is running
        await self._ensure_response_reader()

        # Wait for the response
        try:
            response = await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            # Clean up only our own future
            if (cmd.user_id in self._pending_futures
                    and self._pending_futures[cmd.user_id] is future):
                self._pending_futures.pop(cmd.user_id, None)
            raise

        return response

    async def _ensure_response_reader(self):
        """Start the response reader background task if not running."""
        if self._reader_lock is None:
            self._reader_lock = asyncio.Lock()

        async with self._reader_lock:
            if (self._response_reader_task is None
                    or self._response_reader_task.done()):
                self._response_reader_task = asyncio.create_task(
                    self._response_reader())

    async def _response_reader(self):
        """Background task that reads response queue and routes to per-user futures."""
        loop = asyncio.get_event_loop()

        while self._multi_user_mode:
            try:
                def get_response_nonblocking():
                    try:
                        return self.response_queue.get(timeout=0.5)
                    except Exception:
                        return None

                response = await loop.run_in_executor(
                    None, get_response_nonblocking)

                if response is None:
                    continue

                user_id = response.user_id
                if user_id and user_id in self._pending_futures:
                    future = self._pending_futures.pop(user_id)
                    if not future.done():
                        future.set_result(response)
                else:
                    # No matching pending request - log and discard
                    print(f"[GPU {self.gpu_id}] Unmatched response for user "
                          f"{user_id[:8] if user_id else 'None'}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[GPU {self.gpu_id}] Response reader error: {e}")
                await asyncio.sleep(0.01)

    # --- Single-user API (legacy) ---

    async def step(self, keyboard: list, mouse: list) -> tuple[list[np.ndarray], dict]:
        """Execute a generation step (single-user mode)."""
        async with self._lock:
            response = await self._send_command(
                Command(CommandType.STEP, {"keyboard": keyboard, "mouse": mouse})
            )
            if not response.success:
                raise RuntimeError(f"Step failed: {response.error}")
            return response.frames or [], response.timings or {}

    async def reset(self) -> list[np.ndarray]:
        """Reset the generator (single-user mode)."""
        async with self._lock:
            response = await self._send_command(Command(CommandType.RESET))
            if not response.success:
                raise RuntimeError(f"Reset failed: {response.error}")
            return response.frames or []

    # --- Multi-user API ---

    async def join_user(self, user_id: str, model_id: str = None) -> Response:
        """Add a user to this GPU's multi-user engine.

        If model_id differs from the currently loaded model, triggers a full
        model reload. This invalidates all existing users on this GPU — their
        pending futures are cancelled with an error.
        """
        if model_id is None:
            model_id = DEFAULT_MODEL_ID

        # Reload model if a different one is requested
        if model_id != self.current_model_id and model_id in MODEL_REGISTRY:
            print(f"[GPU {self.gpu_id}] Model switch: "
                  f"{self.current_model_id} -> {model_id}")

            # Cancel pending futures for existing users (their steps will error)
            for uid, future in list(self._pending_futures.items()):
                if not future.done():
                    future.set_exception(
                        RuntimeError("Model changed, session reset"))
            self._pending_futures.clear()
            self.connected_users.clear()

            # Send reload command to GPU worker.
            # Use _send_command (not _send_command_tagged) because
            # _multi_user_mode is still False, so the response reader
            # would exit immediately and never route the response.
            model_config = MODEL_REGISTRY[model_id]
            response = await self._send_command(
                Command(CommandType.RELOAD_MODEL,
                        data={"model_config": model_config},
                        user_id="__reload__"),
                timeout=600.0)
            if not response.success:
                raise RuntimeError(
                    f"Model reload failed: {response.error}")

            self.current_model_id = model_id
            print(f"[GPU {self.gpu_id}] Model reloaded: {model_id}")

        self._multi_user_mode = True
        self.connected_users.add(user_id)
        try:
            response = await self._send_command_tagged(
                Command(CommandType.USER_JOIN, user_id=user_id),
                timeout=600.0)
            if not response.success:
                self.connected_users.discard(user_id)
                raise RuntimeError(
                    f"User join failed for {user_id[:8]}: {response.error}")
            return response
        except Exception:
            self.connected_users.discard(user_id)
            raise

    async def user_step(
        self, user_id: str, keyboard: list, mouse: list
    ) -> tuple[list[np.ndarray], dict]:
        """Execute a generation step for a specific user."""
        response = await self._send_command_tagged(
            Command(CommandType.USER_STEP,
                    {"keyboard": keyboard, "mouse": mouse},
                    user_id=user_id),
            timeout=300.0)
        if not response.success:
            raise RuntimeError(
                f"User step failed for {user_id[:8]}: {response.error}")
        return response.frames or [], response.timings or {}

    async def leave_user(self, user_id: str) -> None:
        """Remove a user from this GPU's multi-user engine."""
        try:
            response = await self._send_command_tagged(
                Command(CommandType.USER_LEAVE, user_id=user_id),
                timeout=30.0)
        except Exception as e:
            print(f"[GPU {self.gpu_id}] Leave user error: {e}")
        finally:
            self.connected_users.discard(user_id)
            self._pending_futures.pop(user_id, None)

    async def shutdown(self):
        """Shutdown the worker subprocess."""
        self._multi_user_mode = False  # Stop the response reader loop
        if self._response_reader_task and not self._response_reader_task.done():
            self._response_reader_task.cancel()
            try:
                await self._response_reader_task
            except asyncio.CancelledError:
                pass

        if self.process and self.process.is_alive():
            try:
                await self._send_command(
                    Command(CommandType.SHUTDOWN), timeout=30.0)
            except Exception:
                pass
            self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()


class GPUPool:
    """Manages multiple GPU worker subprocesses with multi-user support."""

    def __init__(self, gpu_ids: list[int]):
        self.gpu_ids = gpu_ids
        self.slots: dict[int, GPUSlot] = {
            gpu_id: GPUSlot(gpu_id, str(gpu_id)) for gpu_id in gpu_ids
        }
        self.waiting_list: list[tuple[str, asyncio.Event, "WebSocket"]] = []
        self.client_gpu_map: dict[str, int] = {}
        self._pool_lock = asyncio.Lock()

    async def initialize(self):
        """Start initializing all GPU workers in the background."""
        print(f"Initializing GPU pool with {len(self.gpu_ids)} GPUs: {self.gpu_ids}")
        for gpu_id in self.gpu_ids:
            asyncio.create_task(self._init_gpu(gpu_id))

    async def _init_gpu(self, gpu_id: int):
        """Initialize a single GPU and assign any waiting clients."""
        try:
            await self.slots[gpu_id].start()
        except Exception as e:
            print(f"GPU {gpu_id} failed to initialize: {e}")
            return

        print(f"GPU pool: {gpu_id} ready "
              f"({sum(1 for s in self.slots.values() if s.ready)}/{len(self.gpu_ids)})")

        # Check if anyone is waiting for a GPU
        async with self._pool_lock:
            slot = self.slots[gpu_id]
            if slot.is_available and self.waiting_list:
                waiting_client_id, ready_event, _ = self.waiting_list.pop(0)
                self.client_gpu_map[waiting_client_id] = gpu_id
                print(f"Client {waiting_client_id[:8]} assigned GPU {gpu_id} from queue")
                ready_event.set()

                await self._send_queue_updates()

    async def acquire(self, client_id: str, websocket=None) -> tuple[int, GPUSlot]:
        """Acquire a GPU slot for a client.

        In multi-user mode, returns a GPU that has capacity for more users.
        """
        async with self._pool_lock:
            # First, try to find a GPU with existing users (better batching)
            for gpu_id, slot in self.slots.items():
                if (slot.is_available and slot._multi_user_mode
                        and slot.client_count > 0):
                    self.client_gpu_map[client_id] = gpu_id
                    print(f"Client {client_id[:8]} acquired GPU {gpu_id} "
                          f"(co-located, {slot.client_count} users)")
                    return gpu_id, slot

            # Then, try any available GPU
            for gpu_id, slot in self.slots.items():
                if slot.is_available:
                    self.client_gpu_map[client_id] = gpu_id
                    print(f"Client {client_id[:8]} acquired GPU {gpu_id}")
                    return gpu_id, slot

        # No slot available, wait in queue
        print(f"Client {client_id[:8]} waiting in queue "
              f"(all {len(self.gpu_ids)} GPUs at capacity)")
        ready_event = asyncio.Event()
        async with self._pool_lock:
            self.waiting_list.append((client_id, ready_event, websocket))

        await ready_event.wait()

        gpu_id = self.client_gpu_map.get(client_id)
        if gpu_id is None:
            raise RuntimeError(
                f"Client {client_id} was signaled but has no GPU assigned")

        return gpu_id, self.slots[gpu_id]

    async def release(self, client_id: str):
        """Release a client from its GPU slot."""
        async with self._pool_lock:
            gpu_id = self.client_gpu_map.pop(client_id, None)
            if gpu_id is None:
                return

            slot = self.slots[gpu_id]
            print(f"Client {client_id[:8]} released GPU {gpu_id}")

            if slot._multi_user_mode:
                # Multi-user: just remove this user, GPU stays active
                try:
                    await slot.leave_user(client_id)
                except Exception as e:
                    print(f"[GPU {gpu_id}] Leave user failed: {e}")
            else:
                # Single-user: reset the generator
                try:
                    await slot.reset()
                    print(f"[GPU {gpu_id}] Reset complete, ready for next client")
                except Exception as e:
                    print(f"[GPU {gpu_id}] Reset on release failed: {e}")

            # Assign to next waiting client if GPU has capacity
            if slot.is_available and self.waiting_list:
                waiting_client_id, ready_event, _ = self.waiting_list.pop(0)
                self.client_gpu_map[waiting_client_id] = gpu_id
                print(f"Client {waiting_client_id[:8]} assigned GPU {gpu_id} from queue")
                ready_event.set()

                await self._send_queue_updates()

    async def _send_queue_updates(self):
        """Send updated queue positions to all waiting clients."""
        for i, (cid, _, ws) in enumerate(self.waiting_list):
            if ws is not None:
                try:
                    await ws.send_json({
                        "type": "queue_status",
                        "position": i + 1,
                        "total_gpus": len(self.gpu_ids),
                        "available_gpus": 0,
                    })
                except Exception:
                    pass

    async def shutdown(self):
        """Shutdown all GPU workers."""
        print("Shutting down GPU pool...")
        tasks = [slot.shutdown() for slot in self.slots.values()]
        await asyncio.gather(*tasks, return_exceptions=True)
        print("GPU pool shutdown complete")

    def get_status(self) -> dict:
        """Get the current status of the GPU pool."""
        return {
            "total_gpus": len(self.gpu_ids),
            "available_gpus": sum(
                1 for slot in self.slots.values() if slot.is_available),
            "queue_size": len(self.waiting_list),
            "gpu_status": {
                gpu_id: {
                    "available": slot.is_available,
                    "client_count": slot.client_count,
                    "multi_user": slot._multi_user_mode,
                    "current_model_id": slot.current_model_id,
                    "process_alive": (
                        slot.process.is_alive() if slot.process else False),
                }
                for gpu_id, slot in self.slots.items()
            }
        }


def get_available_gpus() -> list[int]:
    """Get list of available GPU IDs from environment or auto-detect."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        return [int(x.strip()) for x in cuda_visible.split(",") if x.strip()]

    # Auto-detect available GPUs
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return [int(x.strip()) for x in result.stdout.strip().split("\n") if x.strip()]
    except Exception:
        pass

    return [0]
