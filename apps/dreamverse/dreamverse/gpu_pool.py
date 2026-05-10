# pyright: reportMissingTypeArgument=false, reportArgumentType=false, reportOptionalSubscript=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportUndefinedVariable=false
# ruff: noqa: UP038, SIM105, F821
# mypy: ignore-errors
import asyncio
import multiprocessing as mp
import os
import subprocess
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Process, Queue

from dreamverse.config import (
    DEFAULT_MODEL_ID,
    MODEL_REGISTRY,
    STARTUP_WARMUP_ENABLED,
    STARTUP_WARMUP_PROMPT,
    STARTUP_WARMUP_TIMEOUT_SECONDS,
)
from dreamverse.av_streaming import (
    SHARED_STREAM_BUFFER_BYTES,
    USE_SHARED_STREAM_BUFFER,
    StreamChunk,
    StreamComplete,
    StreamEvent,
    StreamInit,
    generate_stream_id,
    stream_fmp4,
)
from dreamverse.worker_ipc import (
    CommandPayload,
    InitAck,
    JoinAck,
    LeaveAck,
    MediaChunk,
    MediaComplete,
    MediaInit,
    ReloadAck,
    ReloadModelPayload,
    ShutdownAck,
    StepComplete,
    UserStepPayload,
    WarmupComplete,
    WarmupPayload,
    WorkerError,
    WorkerEvent,
)


def _parse_requested_gpu_limit() -> int | None:
    raw_value = os.getenv("FASTVIDEO_GPU_COUNT", "").strip().lower()
    if not raw_value:
        return 1
    if raw_value == "all":
        return None
    try:
        requested = int(raw_value)
    except ValueError as exc:
        raise RuntimeError("Invalid FASTVIDEO_GPU_COUNT. Use a positive integer or 'all'.") from exc
    if requested <= 0:
        raise RuntimeError("Invalid FASTVIDEO_GPU_COUNT. Use a positive integer or 'all'.")
    return requested


def _limit_gpu_ids(gpu_ids: list[int]) -> list[int]:
    requested_limit = _parse_requested_gpu_limit()
    print(f"[INFO] Using GPU limit={requested_limit}")
    if requested_limit is None:
        return gpu_ids
    return gpu_ids[:requested_limit] or gpu_ids


class CommandType(Enum):
    """Commands sent from main process to GPU worker."""
    INIT = "init"
    WARMUP = "warmup"
    SHUTDOWN = "shutdown"
    USER_JOIN = "user_join"
    USER_STEP = "user_step"
    USER_LEAVE = "user_leave"
    RELOAD_MODEL = "reload_model"


@dataclass
class Command:
    """Command sent to GPU worker subprocess.

    Commands that carry data (USER_STEP, WARMUP, RELOAD_MODEL)
    populate ``payload`` with a typed payload from ``worker_ipc``.
    Commands that don't (INIT, SHUTDOWN, USER_JOIN, USER_LEAVE)
    leave ``payload`` as ``None``.
    """
    type: CommandType
    payload: CommandPayload | None = None
    user_id: str | None = None


def _stream_event_to_worker_event(
    event: StreamEvent,
    user_id: str,
    segment_idx: int,
) -> WorkerEvent:
    """Translate an av_streaming event into a typed worker event.

    ``StreamEvent`` (ffmpeg output layer) carries no routing info;
    this adds ``user_id`` + ``segment_idx`` so the pool's
    ``_response_reader`` can dispatch to the right per-user queue.
    """
    match event:
        case StreamInit(stream_id=sid, mime=m, uses_shared_buffer=u):
            return MediaInit(
                user_id=user_id,
                segment_idx=segment_idx,
                stream_id=sid,
                mime=m,
                uses_shared_buffer=u,
            )
        case StreamChunk(
            stream_id=sid,
            chunk=c,
            chunk_offset=co,
            chunk_length=cl,
            uses_shared_buffer=u,
        ):
            return MediaChunk(
                user_id=user_id,
                segment_idx=segment_idx,
                stream_id=sid,
                chunk=c,
                chunk_offset=co,
                chunk_length=cl,
                uses_shared_buffer=u,
            )
        case StreamComplete(stream_id=sid, chunks=n):
            return MediaComplete(
                user_id=user_id,
                segment_idx=segment_idx,
                stream_id=sid,
                chunks=n,
            )
        case _:
            raise ValueError(f"unknown stream event: {type(event).__name__}")


def gpu_worker_process(
    gpu_id: int,
    cuda_device: str,
    command_queue: Queue,
    response_queue: Queue,
    shared_stream_buffer=None,
    shared_stream_buffer_bytes: int = 0,
):
    """Worker process that runs on a single GPU.

    CUDA_VISIBLE_DEVICES must be set BEFORE importing VideoGenerationWorker
    (which transitively touches CUDA).  Delegates model lifecycle and
    generation to VideoGenerationWorker; AV muxing to av_streaming.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "FLASH_ATTN"

    from dreamverse.video_generation import VideoGenerationWorker

    worker = VideoGenerationWorker(gpu_id)

    def event_loop(first_cmd: Command = None):
        """Blocking event loop for LTX2; dispatches user commands."""
        print(f"[GPU {gpu_id}] Entering event loop")

        def handle_command(cmd: Command):
            if cmd.type == CommandType.USER_JOIN:
                print(f"[GPU {gpu_id}] User {cmd.user_id[:8]} joined")
                worker.clear_conditioning()
                response_queue.put(JoinAck(user_id=cmd.user_id))

            elif cmd.type == CommandType.USER_STEP:
                try:
                    assert isinstance(cmd.payload, UserStepPayload), (f"USER_STEP requires UserStepPayload, "
                                                                      f"got {type(cmd.payload).__name__}")
                    payload = cmd.payload
                    segment_idx = payload.segment_idx
                    step_result = worker.generate_step(
                        payload.prompt,
                        segment_idx,
                        image_path=payload.image_path,
                        reset_conditioning=payload.reset_conditioning,
                    )
                    head_trim_frames = step_result.head_trim_frames
                    head_trim_audio_frames = step_result.head_trim_audio_frames
                    if head_trim_frames > 0 or head_trim_audio_frames > 0:
                        print(f"[GPU {gpu_id}] Segment {segment_idx}: "
                              f"trimming video={head_trim_frames} "
                              f"audio={head_trim_audio_frames} "
                              f"overlap frames from AV output")
                    audio_shape = getattr(step_result.audio, "shape", None)
                    print(f"[GPU {gpu_id}] AV attempt segment "
                          f"{segment_idx}: "
                          f"audio_present={step_result.audio is not None}, "
                          f"audio_shape={audio_shape}, "
                          f"audio_sample_rate={step_result.audio_sample_rate}")
                    stream_id = generate_stream_id(segment_idx)

                    def _publish(event: StreamEvent) -> None:
                        response_queue.put(_stream_event_to_worker_event(event, cmd.user_id, segment_idx))

                    av_ok, av_error = stream_fmp4(
                        frames=step_result.frames,
                        audio=step_result.audio,
                        audio_sample_rate=step_result.audio_sample_rate,
                        stream_id=stream_id,
                        timings=step_result.timings,
                        head_trim_frames=head_trim_frames,
                        head_trim_audio_frames=head_trim_audio_frames,
                        shared_buffer=shared_stream_buffer,
                        shared_buffer_bytes=shared_stream_buffer_bytes,
                        publish=_publish,
                        log_prefix=f"[GPU {gpu_id}]",
                    )
                    if not av_ok:
                        raise RuntimeError(av_error or "worker av_fmp4 stream failed")
                    print(f"[GPU {gpu_id}] AV streamed segment {segment_idx}: "
                          f"encode_total={step_result.timings.get('av_encode_stream_ms', 0):.0f}ms "
                          f"wav_write={step_result.timings.get('av_wav_write_ms', 0):.1f}ms "
                          f"spawn={step_result.timings.get('av_ffmpeg_spawn_ms', 0):.1f}ms "
                          f"first_chunk={step_result.timings.get('av_first_chunk_ms', 0):.0f}ms "
                          f"chunk_interval_med={step_result.timings.get('av_chunk_interval_ms_median', 0):.1f}ms "
                          f"chunk_interval_p95={step_result.timings.get('av_chunk_interval_ms_p95', 0):.1f}ms "
                          f"publish_med={step_result.timings.get('av_chunk_publish_ms_median', 0):.2f}ms "
                          f"read_med={step_result.timings.get('av_chunk_read_ms_median', 0):.1f}ms")
                    step_result.timings["ipc_put_start_ns"] = time.time_ns()
                    response_queue.put(
                        StepComplete(
                            user_id=cmd.user_id,
                            segment_idx=segment_idx,
                            timings=step_result.timings,
                        ))
                except Exception as e:
                    print(f"[GPU {gpu_id}] Step error: {e}")
                    traceback.print_exc()
                    response_queue.put(WorkerError(
                        user_id=cmd.user_id,
                        message=str(e),
                    ))

            elif cmd.type == CommandType.USER_LEAVE:
                print(f"[GPU {gpu_id}] User {cmd.user_id[:8]} left")
                worker.clear_conditioning()
                response_queue.put(LeaveAck(user_id=cmd.user_id))

            elif cmd.type == CommandType.SHUTDOWN:
                return False  # Signal to exit

            elif cmd.type == CommandType.WARMUP:
                try:
                    assert isinstance(cmd.payload, WarmupPayload), (f"WARMUP requires WarmupPayload, "
                                                                    f"got {type(cmd.payload).__name__}")
                    timings = worker.warmup(cmd.payload.prompt)
                    response_queue.put(WarmupComplete(
                        user_id=cmd.user_id,
                        timings=timings,
                    ))
                except Exception as e:
                    print(f"[GPU {gpu_id}] Warmup error: {e}")
                    traceback.print_exc()
                    response_queue.put(WorkerError(
                        user_id=cmd.user_id,
                        message=str(e),
                    ))

            elif cmd.type == CommandType.RELOAD_MODEL:
                try:
                    assert isinstance(cmd.payload, ReloadModelPayload), (f"RELOAD_MODEL requires ReloadModelPayload, "
                                                                         f"got {type(cmd.payload).__name__}")
                    worker.initialize(cmd.payload.model_config)
                    response_queue.put(ReloadAck(user_id=cmd.user_id))
                except Exception as e:
                    print(f"[GPU {gpu_id}] Reload error: {e}")
                    traceback.print_exc()
                    response_queue.put(WorkerError(
                        user_id=cmd.user_id,
                        message=str(e),
                    ))

            return True  # Continue loop

        if first_cmd is not None:
            if first_cmd.type == CommandType.SHUTDOWN:
                worker.shutdown()
                response_queue.put(ShutdownAck())
                return
            handle_command(first_cmd)

        import queue as queue_module
        while True:
            try:
                cmd = command_queue.get(timeout=1.0)
            except queue_module.Empty:
                continue

            if cmd.type == CommandType.SHUTDOWN:
                print(f"[GPU {gpu_id}] Event loop shutting down")
                worker.shutdown()
                response_queue.put(ShutdownAck())
                return

            if not handle_command(cmd):
                return

    print(f"[GPU {gpu_id}] Worker process starting...")

    try:
        while True:
            cmd: Command = command_queue.get()

            if cmd.type == CommandType.SHUTDOWN:
                print(f"[GPU {gpu_id}] Shutting down...")
                worker.shutdown()
                response_queue.put(ShutdownAck())
                break

            elif cmd.type == CommandType.INIT:
                try:
                    worker.initialize()
                    response_queue.put(InitAck(success=True))
                except Exception as e:
                    print(f"[GPU {gpu_id}] Init error: {e}")
                    traceback.print_exc()
                    response_queue.put(InitAck(success=False, error=str(e)))

            elif cmd.type == CommandType.WARMUP:
                try:
                    assert isinstance(cmd.payload, WarmupPayload), (f"WARMUP requires WarmupPayload, "
                                                                    f"got {type(cmd.payload).__name__}")
                    timings = worker.warmup(cmd.payload.prompt)
                    response_queue.put(WarmupComplete(
                        user_id=cmd.user_id,
                        timings=timings,
                    ))
                except Exception as e:
                    print(f"[GPU {gpu_id}] Warmup error: {e}")
                    traceback.print_exc()
                    response_queue.put(WorkerError(
                        user_id=cmd.user_id,
                        message=str(e),
                    ))

            elif cmd.type == CommandType.RELOAD_MODEL:
                try:
                    assert isinstance(cmd.payload, ReloadModelPayload), (f"RELOAD_MODEL requires ReloadModelPayload, "
                                                                         f"got {type(cmd.payload).__name__}")
                    worker.initialize(cmd.payload.model_config)
                    response_queue.put(ReloadAck(user_id=cmd.user_id))
                except Exception as e:
                    print(f"[GPU {gpu_id}] Reload error: {e}")
                    traceback.print_exc()
                    response_queue.put(WorkerError(
                        user_id=cmd.user_id,
                        message=str(e),
                    ))

            elif cmd.type in (CommandType.USER_JOIN, CommandType.USER_STEP, CommandType.USER_LEAVE):
                event_loop(first_cmd=cmd)
                break

    except Exception as e:
        print(f"[GPU {gpu_id}] Worker crashed: {e}")
        traceback.print_exc()

    print(f"[GPU {gpu_id}] Worker process exiting")


class GPUSlot:
    """Manages a single GPU worker subprocess."""

    def __init__(self, gpu_id: int, cuda_device: str):
        self.gpu_id = gpu_id
        self.cuda_device = cuda_device
        self.process: Process | None = None
        self.command_queue: Queue | None = None
        self.response_queue: Queue | None = None
        self.ready: bool = False
        self.warmup_enabled: bool = STARTUP_WARMUP_ENABLED
        self.warmup_success: bool = False
        self.warmup_error: str | None = None
        self.warmup_timings: dict[str, float] = {}
        self._lock = asyncio.Lock()

        # Client state
        self.connected_users: set[str] = set()
        self._pending_futures: dict[str, asyncio.Future] = {}
        self._stream_queues: dict[str, asyncio.Queue] = {}
        self._response_reader_task: asyncio.Task | None = None
        self._active: bool = False
        self._reader_lock: asyncio.Lock | None = None
        self.current_model_id: str = DEFAULT_MODEL_ID
        self.shared_stream_buffer = None
        self.shared_stream_buffer_size = SHARED_STREAM_BUFFER_BYTES

    @property
    def client_count(self) -> int:
        return len(self.connected_users)

    @property
    def is_available(self) -> bool:
        """A GPU is available if it has no active users."""
        alive = self.ready and self.process is not None and self.process.is_alive()
        if not alive:
            return False
        return len(self.connected_users) == 0

    @property
    def is_empty(self) -> bool:
        return len(self.connected_users) == 0

    async def start(self):
        """Start the GPU worker subprocess."""
        self.ready = False
        self.warmup_success = False
        self.warmup_error = None
        self.warmup_timings = {}

        ctx = mp.get_context("spawn")
        self.command_queue = ctx.Queue()
        self.response_queue = ctx.Queue()
        if USE_SHARED_STREAM_BUFFER and self.shared_stream_buffer_size > 0:
            # Fixed shared byte buffer to avoid per-chunk IPC payload copies.
            self.shared_stream_buffer = mp.RawArray("B", self.shared_stream_buffer_size)

        self.process = ctx.Process(
            target=gpu_worker_process,
            args=(
                self.gpu_id,
                self.cuda_device,
                self.command_queue,
                self.response_queue,
                self.shared_stream_buffer,
                self.shared_stream_buffer_size,
            ),
            daemon=False,
        )
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.process.start)

        # Send init command and wait for response
        init_response = await self._send_command(Command(CommandType.INIT), timeout=600.0)
        if not isinstance(init_response, InitAck) or not init_response.success:
            error_msg = (init_response.error if isinstance(init_response, InitAck) else
                         f"unexpected init response: {type(init_response).__name__}")
            raise RuntimeError(f"GPU {self.gpu_id} failed to initialize: {error_msg}")

        if self.warmup_enabled:
            try:
                warmup_response = await self._send_command(
                    Command(
                        CommandType.WARMUP,
                        payload=WarmupPayload(prompt=STARTUP_WARMUP_PROMPT),
                        user_id=f"__warmup_gpu_{self.gpu_id}__",
                    ),
                    timeout=float(STARTUP_WARMUP_TIMEOUT_SECONDS),
                )
            except Exception as exc:
                self.warmup_error = str(exc)
                raise RuntimeError(f"GPU {self.gpu_id} warmup failed: {self.warmup_error}") from exc
            match warmup_response:
                case WarmupComplete(timings=timings):
                    self.warmup_timings = {
                        key: float(value)
                        for key, value in timings.items() if isinstance(value, (int, float))
                    }
                    self.warmup_success = True
                case WorkerError(message=msg):
                    self.warmup_error = msg or "Warmup failed."
                    raise RuntimeError(f"GPU {self.gpu_id} warmup failed: {self.warmup_error}")
                case _:
                    self.warmup_error = (f"unexpected warmup response: "
                                         f"{type(warmup_response).__name__}")
                    raise RuntimeError(f"GPU {self.gpu_id} warmup failed: {self.warmup_error}")
        else:
            print(f"[GPU {self.gpu_id}] Startup warmup disabled by "
                  "FASTVIDEO_ENABLE_STARTUP_WARMUP")

        self.ready = True

    async def _send_command(self, cmd: Command, timeout: float = 300.0) -> WorkerEvent:
        """Send a command and wait for the response or worker death.

        Whichever arrives first wins.  A worker dying (exit, signal, OOM,
        segfault) flips the kernel-level sentinel fd readable, which
        asyncio notices via add_reader — typically within ~10 ms.  The
        queue timeout still bounds hangs where the worker stays alive but
        never replies.
        """
        loop = asyncio.get_running_loop()
        process = self.process
        await loop.run_in_executor(None, self.command_queue.put, cmd)

        response_fut = loop.run_in_executor(None, lambda: self.response_queue.get(timeout=timeout))

        death_fut: asyncio.Future | None = None
        sentinel_fd = process.sentinel if process is not None else None

        if sentinel_fd is not None:
            death_fut = loop.create_future()

            def _on_death() -> None:
                try:
                    loop.remove_reader(sentinel_fd)
                except (ValueError, OSError):
                    pass
                if death_fut is not None and not death_fut.done():
                    death_fut.set_result(None)

            loop.add_reader(sentinel_fd, _on_death)

        waiters = [response_fut] + ([death_fut] if death_fut is not None else [])
        try:
            done, _ = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        finally:
            if (sentinel_fd is not None and death_fut is not None and not death_fut.done()):
                try:
                    loop.remove_reader(sentinel_fd)
                except (ValueError, OSError):
                    pass
                death_fut.cancel()

        if (death_fut is not None and death_fut in done and response_fut not in done):
            try:
                return self.response_queue.get_nowait()
            except Exception:
                pass
            self.ready = False
            pid = process.pid if process is not None else "?"
            # Reap the process so .exitcode is populated.  Sentinel
            # readability means the kernel has already exited the
            # process; join is non-blocking in practice.
            if process is not None:
                try:
                    process.join(timeout=1)
                except Exception:
                    pass
            exitcode = process.exitcode if process is not None else None
            raise RuntimeError(f"GPU {self.gpu_id} worker died during command "
                               f"(pid={pid}, exitcode={exitcode})")

        return response_fut.result()

    async def _send_command_tagged(self, cmd: Command, timeout: float = 300.0) -> WorkerEvent:
        """Send a tagged command and wait for the matching response.

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
            if (cmd.user_id in self._pending_futures and self._pending_futures[cmd.user_id] is future):
                self._pending_futures.pop(cmd.user_id, None)
            raise

        return response

    async def _ensure_response_reader(self):
        """Start the response reader background task if not running."""
        if self._reader_lock is None:
            self._reader_lock = asyncio.Lock()

        async with self._reader_lock:
            if (self._response_reader_task is None or self._response_reader_task.done()):
                self._response_reader_task = asyncio.create_task(self._response_reader())

    async def _response_reader(self):
        """Background task that reads response queue and routes to per-user futures."""
        loop = asyncio.get_event_loop()

        while self._active:
            try:

                def get_response_nonblocking():
                    try:
                        return self.response_queue.get(timeout=0.5)
                    except Exception:
                        return None

                event = await loop.run_in_executor(None, get_response_nonblocking)

                if event is None:
                    continue

                # AV streaming events → route to the user's stream queue.
                if isinstance(event, (MediaInit, MediaChunk, MediaComplete)):
                    stream_queue = self._stream_queues.get(event.user_id)
                    if stream_queue is not None:
                        await stream_queue.put(event)
                    else:
                        print(f"[GPU {self.gpu_id}] Unmatched stream event for user "
                              f"{event.user_id[:8]}")
                    continue

                # System-level acks shouldn't reach the tagged router; they
                # belong to the untagged `_send_command` path.
                if isinstance(event, (InitAck, ShutdownAck)):
                    print(f"[GPU {self.gpu_id}] System event leaked into "
                          f"tagged reader: {type(event).__name__}")
                    continue

                # Late-mutation of timings for observability.  Only events
                # that actually carry timings get this annotation.
                if isinstance(event, (StepComplete, WarmupComplete)):
                    event.timings["ipc_get_done_ns"] = time.time_ns()

                user_id = event.user_id
                if user_id and user_id in self._pending_futures:
                    future = self._pending_futures.pop(user_id)
                    if not future.done():
                        future.set_result(event)
                else:
                    print(f"[GPU {self.gpu_id}] Unmatched response for user "
                          f"{user_id[:8] if user_id else 'None'}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[GPU {self.gpu_id}] Response reader error: {e}")
                await asyncio.sleep(0.01)

    def register_stream_queue(self, user_id: str) -> asyncio.Queue:
        """Register stream-event queue for a specific user."""
        queue = asyncio.Queue()
        self._stream_queues[user_id] = queue
        return queue

    def unregister_stream_queue(self, user_id: str) -> None:
        """Remove stream-event queue for a specific user."""
        self._stream_queues.pop(user_id, None)

    async def join_user(self, user_id: str, model_id: str = None) -> JoinAck:
        """Add a user to this GPU."""
        if model_id is None:
            model_id = DEFAULT_MODEL_ID

        # Reload model if a different one is requested
        if model_id != self.current_model_id and model_id in MODEL_REGISTRY:
            print(f"[GPU {self.gpu_id}] Model switch: "
                  f"{self.current_model_id} -> {model_id}")

            for uid, future in list(self._pending_futures.items()):
                if not future.done():
                    future.set_exception(RuntimeError("Model changed, session reset"))
            self._pending_futures.clear()
            self._stream_queues.clear()
            self.connected_users.clear()

            model_config = MODEL_REGISTRY[model_id]
            reload_response = await self._send_command(Command(CommandType.RELOAD_MODEL,
                                                               payload=ReloadModelPayload(model_config=model_config),
                                                               user_id="__reload__"),
                                                       timeout=600.0)
            match reload_response:
                case ReloadAck():
                    pass
                case WorkerError(message=msg):
                    raise RuntimeError(f"Model reload failed: {msg}")
                case _:
                    raise RuntimeError(f"Unexpected reload response: "
                                       f"{type(reload_response).__name__}")

            self.current_model_id = model_id
            print(f"[GPU {self.gpu_id}] Model reloaded: {model_id}")

        self._active = True
        self.connected_users.add(user_id)
        try:
            response = await self._send_command_tagged(Command(CommandType.USER_JOIN, user_id=user_id), timeout=600.0)
            match response:
                case JoinAck() as ack:
                    return ack
                case WorkerError(message=msg):
                    self.connected_users.discard(user_id)
                    raise RuntimeError(f"User join failed for {user_id[:8]}: {msg}")
                case _:
                    self.connected_users.discard(user_id)
                    raise RuntimeError(f"Unexpected join response for {user_id[:8]}: "
                                       f"{type(response).__name__}")
        except Exception:
            self.connected_users.discard(user_id)
            raise

    async def user_step(
        self,
        user_id: str,
        prompt: str,
        segment_idx: int = 1,
        image_path: str | None = None,
        reset_conditioning: bool = False,
    ) -> dict[str, float]:
        """Execute a generation step for a specific user.

        Returns the timings dict.  Frames/audio are no longer part of
        this return — they stream asynchronously via the AV media
        events (MediaInit/MediaChunk/MediaComplete) which the caller
        consumes through ``register_stream_queue``.
        """
        payload = UserStepPayload(
            prompt=prompt,
            segment_idx=segment_idx,
            image_path=image_path,
            reset_conditioning=bool(reset_conditioning),
        )
        response = await self._send_command_tagged(Command(CommandType.USER_STEP, payload=payload, user_id=user_id),
                                                   timeout=1800.0)
        match response:
            case StepComplete(timings=timings):
                return timings
            case WorkerError(message=msg):
                raise RuntimeError(f"User step failed for {user_id[:8]}: {msg}")
            case _:
                raise RuntimeError(f"Unexpected step response for {user_id[:8]}: "
                                   f"{type(response).__name__}")

    async def leave_user(self, user_id: str) -> None:
        """Remove a user from this GPU."""
        try:
            await self._send_command_tagged(Command(CommandType.USER_LEAVE, user_id=user_id), timeout=30.0)
        except Exception as e:
            print(f"[GPU {self.gpu_id}] Leave user error: {e}")
        finally:
            self.connected_users.discard(user_id)
            self._pending_futures.pop(user_id, None)
            self._stream_queues.pop(user_id, None)

    async def shutdown(self):
        """Shutdown the worker subprocess."""
        self._active = False
        if self._response_reader_task and not self._response_reader_task.done():
            self._response_reader_task.cancel()
            try:
                await self._response_reader_task
            except asyncio.CancelledError:
                pass

        if self.process is not None:
            if self.process.is_alive():
                try:
                    await self._send_command(Command(CommandType.SHUTDOWN), timeout=30.0)
                except Exception:
                    pass
                self.process.terminate()
                self.process.join(timeout=5)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(timeout=5)
            else:
                # Process already died (sentinel path).  Reap it so the
                # OS releases the PID.
                try:
                    self.process.join(timeout=1)
                except Exception:
                    pass

        for q in (self.command_queue, self.response_queue):
            if q is not None:
                try:
                    q.close()
                except Exception:
                    pass
                try:
                    q.join_thread()
                except Exception:
                    pass


class GPUPool:
    """Manages multiple GPU worker subprocesses."""

    def __init__(self, gpu_ids: list[int]):
        self.gpu_ids = gpu_ids
        self.slots: dict[int, GPUSlot] = {gpu_id: GPUSlot(gpu_id, str(gpu_id)) for gpu_id in gpu_ids}
        self.waiting_list: list[tuple[str, asyncio.Event, WebSocket]] = []
        self.client_gpu_map: dict[str, int] = {}
        self._pool_lock = asyncio.Lock()

    async def initialize(self):
        """Initialize all GPU workers and wait for them to be ready.

        Any per-GPU failure aborts startup so uvicorn refuses to serve
        traffic with no functional GPUs.
        """
        print(f"Initializing GPU pool with {len(self.gpu_ids)} GPUs: {self.gpu_ids}")
        await asyncio.gather(*(self._init_gpu(gpu_id) for gpu_id in self.gpu_ids))

    async def _init_gpu(self, gpu_id: int):
        """Initialize a single GPU and assign any waiting clients."""
        try:
            await self.slots[gpu_id].start()
        except Exception as e:
            print(f"GPU {gpu_id} failed to initialize: {e}")
            await self.slots[gpu_id].shutdown()
            raise

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
        """Acquire a GPU slot for a client."""
        async with self._pool_lock:
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
            await self._send_queue_updates()

        try:
            await ready_event.wait()
        except asyncio.CancelledError:
            # Client disconnected while queued; remove stale queue entry.
            async with self._pool_lock:
                self.waiting_list = [
                    item for item in self.waiting_list if not (item[0] == client_id and item[1] is ready_event)
                ]
                self.client_gpu_map.pop(client_id, None)
                await self._send_queue_updates()
            raise

        gpu_id = self.client_gpu_map.get(client_id)
        if gpu_id is None:
            raise RuntimeError(f"Client {client_id} was signaled but has no GPU assigned")

        return gpu_id, self.slots[gpu_id]

    async def release(self, client_id: str):
        """Release a client from its GPU slot."""
        async with self._pool_lock:
            # Remove stale queue entries if the client disconnected while waiting.
            prev_wait_len = len(self.waiting_list)
            self.waiting_list = [item for item in self.waiting_list if item[0] != client_id]
            removed_from_queue = len(self.waiting_list) != prev_wait_len

            gpu_id = self.client_gpu_map.pop(client_id, None)
            if gpu_id is None:
                if removed_from_queue:
                    await self._send_queue_updates()
                return

            slot = self.slots[gpu_id]
            print(f"Client {client_id[:8]} released GPU {gpu_id}")

            try:
                await slot.leave_user(client_id)
            except Exception as e:
                print(f"[GPU {gpu_id}] Leave user failed: {e}")

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
        warmed_count = sum(1 for slot in self.slots.values() if slot.warmup_success)
        warmup_failures = sum(1 for slot in self.slots.values()
                              if slot.warmup_enabled and slot.warmup_error is not None)
        return {
            "total_gpus": len(self.gpu_ids),
            "available_gpus": sum(1 for slot in self.slots.values() if slot.is_available),
            "queue_size": len(self.waiting_list),
            "warmup_enabled": STARTUP_WARMUP_ENABLED,
            "warmup_successful_gpus": warmed_count,
            "warmup_failed_gpus": warmup_failures,
            "gpu_status": {
                gpu_id: {
                    "ready": slot.ready,
                    "available": slot.is_available,
                    "client_count": slot.client_count,
                    "current_model_id": slot.current_model_id,
                    "process_alive": (slot.process.is_alive() if slot.process else False),
                    "warmup_enabled": slot.warmup_enabled,
                    "warmup_success": slot.warmup_success,
                    "warmup_error": slot.warmup_error,
                    "warmup_timings": slot.warmup_timings,
                }
                for gpu_id, slot in self.slots.items()
            }
        }


def get_available_gpus() -> list[int]:
    """Get list of available GPU IDs from environment or auto-detect."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        visible_gpu_ids = [int(x.strip()) for x in cuda_visible.split(",") if x.strip()]
        return _limit_gpu_ids(visible_gpu_ids)

    # Auto-detect available GPUs
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                                capture_output=True,
                                text=True)
        if result.returncode == 0:
            detected_gpu_ids = [int(x.strip()) for x in result.stdout.strip().split("\n") if x.strip()]
            print(f"Auto-detected GPU IDs: {detected_gpu_ids}")
            return _limit_gpu_ids(detected_gpu_ids)
    except Exception:
        pass

    return _limit_gpu_ids([0])
