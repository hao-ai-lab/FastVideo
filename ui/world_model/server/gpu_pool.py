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
    MODEL_CONFIG, NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_INFERENCE_STEPS
)


class CommandType(Enum):
    """Commands sent from main process to GPU worker."""
    INIT = "init"
    STEP = "step"
    RESET = "reset"
    SHUTDOWN = "shutdown"


@dataclass
class Command:
    """Command sent to GPU worker subprocess."""
    type: CommandType
    data: dict = None

    def __post_init__(self):
        if self.data is None:
            self.data = {}


@dataclass
class Response:
    """Response from GPU worker subprocess."""
    success: bool
    frames: Optional[list[np.ndarray]] = None
    error: Optional[str] = None


def gpu_worker_process(
    gpu_id: int,
    command_queue: Queue,
    response_queue: Queue,
):
    """
    Worker process that runs on a single GPU.

    This function runs in a subprocess with CUDA_VISIBLE_DEVICES set to a single GPU.
    """
    # Set CUDA_VISIBLE_DEVICES BEFORE importing torch or any CUDA code
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "FLASH_ATTN"

    # Now import the generator (this will initialize CUDA with the single visible GPU)
    from fastvideo.entrypoints.streaming_generator import StreamingVideoGenerator
    from fastvideo.models.dits.matrixgame.utils import expand_action_to_frames
    from fastvideo.configs.sample.base import SamplingParam
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    from fastvideo.utils import align_to, shallow_asdict

    generator = None

    def initialize_generator():
        nonlocal generator
        print(f"[GPU {gpu_id}] Loading model...")

        generator = StreamingVideoGenerator.from_pretrained(
            MODEL_CONFIG["model_path"],
            num_gpus=1,
            use_fsdp_inference=True,
            dit_cpu_offload=True,
            vae_cpu_offload=False,
            text_encoder_cpu_offload=True,
            pin_cpu_memory=True,
        )

        # Initial reset
        actions = {
            "keyboard": torch.zeros((NUM_FRAMES, MODEL_CONFIG["keyboard_dim"])),
            "mouse": torch.zeros((NUM_FRAMES, 2))
        }

        generator.reset(
            prompt="",
            image_path=MODEL_CONFIG["image_url"],
            mouse_cond=actions["mouse"].unsqueeze(0),
            keyboard_cond=actions["keyboard"].unsqueeze(0),
            grid_sizes=torch.tensor([150, 44, 80]),
            num_frames=NUM_FRAMES,
            height=FRAME_HEIGHT,
            width=FRAME_WIDTH,
            num_inference_steps=NUM_INFERENCE_STEPS,
        )

        print(f"[GPU {gpu_id}] Model loaded and ready")

    def do_step(keyboard_vector: list, mouse_vector: list) -> list[np.ndarray]:
        """Execute a generation step."""
        action = {
            "keyboard": torch.tensor(keyboard_vector).cuda(),
            "mouse": torch.tensor(mouse_vector).cuda()
        }
        keyboard_cond, mouse_cond = expand_action_to_frames(action, 12)
        frames, _ = generator.step(keyboard_cond, mouse_cond)
        return frames if frames is not None else []

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
            "keyboard": torch.zeros((NUM_FRAMES, MODEL_CONFIG["keyboard_dim"])),
            "mouse": torch.zeros((NUM_FRAMES, 2))
        }

        if generator.sampling_param is None:
            generator.sampling_param = SamplingParam.from_pretrained(
                generator.fastvideo_args.model_path)

        generator.sampling_param.update({
            "prompt": "",
            "image_path": MODEL_CONFIG["image_url"],
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

            elif cmd.type == CommandType.STEP:
                try:
                    keyboard = cmd.data.get("keyboard", [0, 0, 0, 0])
                    mouse = cmd.data.get("mouse", [0, 0])
                    frames = do_step(keyboard, mouse)
                    response_queue.put(Response(success=True, frames=frames))
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

    except Exception as e:
        print(f"[GPU {gpu_id}] Worker crashed: {e}")
        traceback.print_exc()

    print(f"[GPU {gpu_id}] Worker process exiting")


class GPUSlot:
    """Manages a single GPU worker subprocess."""

    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.client_id: Optional[str] = None
        self.process: Optional[Process] = None
        self.command_queue: Optional[Queue] = None
        self.response_queue: Optional[Queue] = None
        self.ready: bool = False
        self._lock = asyncio.Lock()

    @property
    def is_available(self) -> bool:
        return self.client_id is None and self.ready and self.process is not None and self.process.is_alive()

    async def start(self):
        """Start the GPU worker subprocess."""
        ctx = mp.get_context("spawn")
        self.command_queue = ctx.Queue()
        self.response_queue = ctx.Queue()

        self.process = ctx.Process(
            target=gpu_worker_process,
            args=(self.gpu_id, self.command_queue, self.response_queue),
            daemon=False,  # Must be False so executor can spawn child workers
        )
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.process.start)

        # Send init command and wait for response
        response = await self._send_command(Command(CommandType.INIT), timeout=600.0)
        if not response.success:
            raise RuntimeError(f"GPU {self.gpu_id} failed to initialize: {response.error}")

        self.ready = True

    async def _send_command(self, cmd: Command, timeout: float = 300.0) -> Response:
        """Send a command to the worker and wait for response."""
        loop = asyncio.get_event_loop()

        # Send command in thread pool to not block
        await loop.run_in_executor(None, self.command_queue.put, cmd)

        # Wait for response with timeout
        def get_response():
            return self.response_queue.get(timeout=timeout)

        response = await loop.run_in_executor(None, get_response)
        return response

    async def step(self, keyboard: list, mouse: list) -> list[np.ndarray]:
        """Execute a generation step."""
        async with self._lock:
            response = await self._send_command(
                Command(CommandType.STEP, {"keyboard": keyboard, "mouse": mouse})
            )
            if not response.success:
                raise RuntimeError(f"Step failed: {response.error}")
            return response.frames or []

    async def reset(self) -> list[np.ndarray]:
        """Reset the generator."""
        async with self._lock:
            response = await self._send_command(Command(CommandType.RESET))
            if not response.success:
                raise RuntimeError(f"Reset failed: {response.error}")
            return response.frames or []

    async def shutdown(self):
        """Shutdown the worker subprocess."""
        if self.process and self.process.is_alive():
            try:
                await self._send_command(Command(CommandType.SHUTDOWN), timeout=30.0)
            except Exception:
                pass
            self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()


class GPUPool:
    """Manages multiple GPU worker subprocesses."""

    def __init__(self, gpu_ids: list[int]):
        self.gpu_ids = gpu_ids
        self.slots: dict[int, GPUSlot] = {gpu_id: GPUSlot(gpu_id) for gpu_id in gpu_ids}
        self.waiting_list: list[tuple[str, asyncio.Event, "WebSocket"]] = []
        self.client_gpu_map: dict[str, int] = {}
        self._pool_lock = asyncio.Lock()

    async def initialize(self):
        """Start initializing all GPU workers in the background.

        Returns immediately so the server can start accepting connections.
        GPUs become available as each one finishes loading.
        """
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

        print(f"GPU pool: {gpu_id} ready ({sum(1 for s in self.slots.values() if s.ready)}/{len(self.gpu_ids)})")

        # Check if anyone is waiting for a GPU
        async with self._pool_lock:
            slot = self.slots[gpu_id]
            if slot.is_available and self.waiting_list:
                waiting_client_id, ready_event, _ = self.waiting_list.pop(0)
                slot.client_id = waiting_client_id
                self.client_gpu_map[waiting_client_id] = gpu_id
                print(f"Client {waiting_client_id[:8]} assigned GPU {gpu_id} from queue")
                ready_event.set()

                # Notify remaining clients of updated positions
                await self._send_queue_updates()

    async def acquire(self, client_id: str, websocket=None) -> tuple[int, GPUSlot]:
        """
        Acquire a GPU slot for a client.
        Returns (gpu_id, slot) when available.
        """
        async with self._pool_lock:
            for gpu_id, slot in self.slots.items():
                if slot.is_available:
                    slot.client_id = client_id
                    self.client_gpu_map[client_id] = gpu_id
                    print(f"Client {client_id[:8]} acquired GPU {gpu_id}")
                    return gpu_id, slot

        # No slot available, wait in queue
        print(f"Client {client_id[:8]} waiting in queue (all {len(self.gpu_ids)} GPUs busy)")
        ready_event = asyncio.Event()
        async with self._pool_lock:
            self.waiting_list.append((client_id, ready_event, websocket))

        await ready_event.wait()

        gpu_id = self.client_gpu_map.get(client_id)
        if gpu_id is None:
            raise RuntimeError(f"Client {client_id} was signaled but has no GPU assigned")

        return gpu_id, self.slots[gpu_id]

    async def release(self, client_id: str):
        """Release a GPU slot when a client disconnects."""
        async with self._pool_lock:
            gpu_id = self.client_gpu_map.pop(client_id, None)
            if gpu_id is None:
                return

            slot = self.slots[gpu_id]
            print(f"Client {client_id[:8]} released GPU {gpu_id}")

            # Reset the generator so it's fresh for the next client
            try:
                await slot.reset()
                print(f"[GPU {gpu_id}] Reset complete, ready for next client")
            except Exception as e:
                print(f"[GPU {gpu_id}] Reset on release failed: {e}")

            slot.client_id = None

            # Assign to next waiting client
            if self.waiting_list:
                waiting_client_id, ready_event, _ = self.waiting_list.pop(0)
                slot.client_id = waiting_client_id
                self.client_gpu_map[waiting_client_id] = gpu_id
                print(f"Client {waiting_client_id[:8]} assigned GPU {gpu_id} from queue")
                ready_event.set()

                # Notify remaining clients of updated positions
                await self._send_queue_updates()

    async def _send_queue_updates(self):
        """Send updated queue positions to all waiting clients. Must be called with _pool_lock held."""
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
                    pass  # Client may have disconnected

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
            "available_gpus": sum(1 for slot in self.slots.values() if slot.is_available),
            "queue_size": len(self.waiting_list),
            "gpu_status": {
                gpu_id: {
                    "available": slot.is_available,
                    "client_id": slot.client_id[:8] if slot.client_id else None,
                    "process_alive": slot.process.is_alive() if slot.process else False,
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
