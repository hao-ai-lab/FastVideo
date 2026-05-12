"""Per-WebSocket-connection session controller.

Lift-and-wrap extraction of the original ``main.websocket_endpoint``. The
function body is preserved verbatim inside ``SessionController.run()``;
the only substitutions are references to ``runtime.<service>`` rewritten
to ``self.<service>``, and a local alias ``websocket = self.ws`` at the
top of ``run()`` so the body can keep using the ``websocket`` name it
originally bound as a function parameter.

Captured locals, nested closures, and ``nonlocal`` declarations are
unchanged — this is a module-boundary move, not a state-semantics
rewrite. Decomposing the 1880-line ``run`` body into smaller methods
and extracting a ``SessionState`` dataclass is explicitly deferred until
a real trigger appears (new WS action, reproducible concurrency bug,
etc.).
"""
from __future__ import annotations
# pyright: reportArgumentType=false, reportMissingImports=false, reportMissingTypeArgument=false, reportOptionalMemberAccess=false
# ruff: noqa: SIM105
# mypy: ignore-errors

import asyncio
import time
import uuid
from typing import TYPE_CHECKING

from fastapi import WebSocket, WebSocketDisconnect
from dreamverse.gpu_pool import GPUSlot
from dreamverse.session_init_image import cleanup_session_init_image, persist_session_init_image
from dreamverse.worker_ipc import MediaChunk, MediaComplete, MediaInit

from dreamverse.config import (
    DEFAULT_MODEL_ID,
    GENERATION_SEGMENT_CAP,
    PROMPT_AUTO_SLEEP_MS,
    PROMPT_AUTO_TIMEOUT_MS,
    PROMPT_TIMEOUT_MS,
    SESSION_TIMEOUT_SECONDS,
)
from dreamverse.rewrite_prompt_payload import normalize_prompt_window_prompts

from dreamverse.session.messages import PromptSubmission, ReadyPrompt
from dreamverse.utils import (
    _main_print,
    _resolve_generation_segment_cap,
    PROMPT_EXTENSION_FAILURE_USER_MESSAGE,
)

if TYPE_CHECKING:
    from dreamverse.gpu_pool import GPUPool
    from dreamverse.session_logger import SessionEventLogger
    from dreamverse.prompt_enhancer import PromptEnhancer
    from dreamverse.prompt_safety import PromptSafetyFilter


class SessionController:
    """Runs one WebSocket session from accept() through disconnect."""

    def __init__(
        self,
        ws: WebSocket,
        gpu_pool: GPUPool,
        prompt_enhancer: PromptEnhancer,
        prompt_safety_filter: PromptSafetyFilter | None,
        session_event_logger: SessionEventLogger | None,
    ) -> None:
        self.ws = ws
        self.gpu_pool = gpu_pool
        self.prompt_enhancer = prompt_enhancer
        self.prompt_safety_filter = prompt_safety_filter
        self.session_event_logger = session_event_logger

    async def run(self) -> None:
        websocket = self.ws
        # --- lift-and-wrap body begins ---
        await websocket.accept()

        if self.gpu_pool is None:
            await websocket.send_json({
                "type": "error",
                "message": "GPU pool not initialized",
            })
            await websocket.close(code=1011)
            return
        if self.prompt_enhancer is None:
            await websocket.send_json({
                "type": "error",
                "message": "Prompt enhancer not initialized",
            })
            await websocket.close(code=1011)
            return

        client_id = str(uuid.uuid4())
        print(f"Client {client_id[:8]} connected")

        async def log_event(
            event: str,
            payload: dict[str, object] | None = None,
        ) -> None:
            if self.session_event_logger is None:
                return
            try:
                await self.session_event_logger.write_event(
                    event=event,
                    client_id=client_id,
                    payload=payload,
                )
            except Exception as exc:
                _main_print(
                    "WARN",
                    f"Failed to write session log ({event}): {exc}",
                )

        await log_event("ws_session_start")

        send_lock = asyncio.Lock()
        stop_event = asyncio.Event()

        async def ws_send_json(payload: dict):
            async with send_lock:
                await websocket.send_json(payload)

        async def ws_send_bytes(payload: bytes):
            async with send_lock:
                await websocket.send_bytes(payload)

        def preview_text(text: str, limit: int = 180) -> str:
            normalized = (text or "").replace("\n", "\\n")
            if len(normalized) <= limit:
                return normalized
            return normalized[:limit] + "..."

        def get_prompt_safety_error(prompt: str) -> str | None:
            if self.prompt_safety_filter is None:
                return None
            return self.prompt_safety_filter.get_prompt_safety_error(prompt)

        def get_first_blocked_prompt(prompts: list[str]):
            if self.prompt_safety_filter is None:
                return None
            return self.prompt_safety_filter.get_first_blocked_prompt(prompts)

        # Send initial queue status
        status = self.gpu_pool.get_status()
        await ws_send_json({
            "type": "queue_status",
            "position": status["queue_size"] + 1 if status["available_gpus"] == 0 else 0,
            "total_gpus": status["total_gpus"],
            "available_gpus": status["available_gpus"],
        })

        gpu_id: int | None = None
        slot: GPUSlot | None = None
        timeout_task: asyncio.Task | None = None
        websocket_reader_task: asyncio.Task | None = None
        prompt_worker_task: asyncio.Task | None = None
        rewrite_seed_prompts_task: asyncio.Task | None = None
        session_init_image = None

        async def session_timeout():
            """Close the session after timeout."""
            await asyncio.sleep(SESSION_TIMEOUT_SECONDS)
            if stop_event.is_set():
                return
            stop_event.set()
            print(f"[GPU {gpu_id}] Session timeout for client {client_id[:8]}")
            try:
                await ws_send_json({
                    "type": "session_timeout",
                    "message": f"Session expired after {SESSION_TIMEOUT_SECONDS} seconds",
                })
                await websocket.close(code=1000, reason="Session timeout")
            except Exception:
                pass

        async def cancel_task(task: asyncio.Task | None):
            if task is None or task.done():
                return
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        try:
            # Wait for client init message.
            init_data = {}
            try:
                init_data = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
            except asyncio.TimeoutError:
                init_data = {}

            init_type = init_data.get("type")
            preset_id = init_data.get("preset_id")
            preset_label = str(init_data.get("preset_label") or "").strip()
            initial_rollout_prompt = str(init_data.get("initial_rollout_prompt") or "").strip()
            enhancement_enabled = bool(init_data.get("enhancement_enabled", True))
            auto_extension_enabled = bool(init_data.get("auto_extension_enabled", False))
            loop_generation_enabled = bool(init_data.get("loop_generation_enabled", False))
            single_clip_mode = bool(init_data.get("single_clip_mode", False))
            rewrite_model = self.prompt_enhancer.resolve_rewrite_model(init_data.get("rewrite_model"))
            rewrite_system_prompt_override = str(init_data.get("rewrite_window_system_prompt") or "").strip()
            rewrite_user_system_prompt_override = str(init_data.get("rewrite_user_system_prompt") or "").strip()
            rewrite_system_prompt = self.prompt_enhancer.resolve_rewrite_system_prompt(rewrite_system_prompt_override)
            rewrite_timeout_ms = PROMPT_TIMEOUT_MS
            rewrite_temperature = self.prompt_enhancer.resolve_rewrite_temperature(init_data.get("rewrite_temperature"))

            if init_type == "session_init_v2":
                incoming_prompts = init_data.get("curated_prompts", [])
            elif init_type == "select_model":
                incoming_prompts = init_data.get("segment_prompts", [])
            else:
                incoming_prompts = init_data.get("segment_prompts", [])

            curated_prompts = []
            if isinstance(incoming_prompts, list):
                for prompt in incoming_prompts:
                    if isinstance(prompt, str) and prompt.strip():
                        curated_prompts.append(prompt.strip())

            blocked_init_prompt = get_first_blocked_prompt(curated_prompts)
            if blocked_init_prompt is not None:
                message = ("Prompt safety filter blocked initial prompt "
                           f"{blocked_init_prompt.index + 1}. {blocked_init_prompt.error}")
                _main_print(
                    "WARN",
                    f"Initial prompt blocked for client {client_id[:8]} "
                    f"index={blocked_init_prompt.index + 1} "
                    f"preview={preview_text(blocked_init_prompt.prompt)}",
                )
                await ws_send_json({
                    "type": "error",
                    "message": message,
                })
                await websocket.close(code=1008, reason="Prompt blocked")
                return

            try:
                session_init_image = persist_session_init_image(init_data.get("initial_image"))
            except ValueError as exc:
                await ws_send_json({
                    "type": "error",
                    "message": str(exc),
                })
                await websocket.close(code=1003, reason="Invalid initial image")
                return

            if preset_id:
                print(f"Client {client_id[:8]} selected preset: {preset_id} "
                      f"label={preset_label or '(unset)'} "
                      f"({len(curated_prompts)} curated prompts, "
                      f"enhancement_enabled={enhancement_enabled}, "
                      f"auto_extension_enabled={auto_extension_enabled}, "
                      f"loop_generation_enabled={loop_generation_enabled}, "
                      f"single_clip_mode={single_clip_mode})")
            if session_init_image is not None:
                print(f"Client {client_id[:8]} uploaded initial image: "
                      f"{session_init_image.display_name}")

            # Acquire a GPU slot.
            gpu_id, slot = await self.gpu_pool.acquire(client_id, websocket)

            # Start session timeout.
            timeout_task = asyncio.create_task(session_timeout())

            # Join the engine on this GPU.
            await slot.join_user(client_id, model_id=DEFAULT_MODEL_ID)

            # Notify client they're connected to a GPU.
            await ws_send_json({
                "type": "gpu_assigned",
                "gpu_id": gpu_id,
                "session_timeout": SESSION_TIMEOUT_SECONDS,
            })
            await log_event(
                "gpu_assigned",
                {
                    "gpu_id": gpu_id,
                },
            )

            # Session queues and mutable state.
            raw_prompt_queue: asyncio.Queue[PromptSubmission] = asyncio.Queue()
            ready_prompt_queue: asyncio.Queue[ReadyPrompt] = asyncio.Queue()

            curated_idx = 0
            segment_idx = 0
            locked_segment_prompts: list[str] = []
            seed_prompt_memory: list[str] = list(curated_prompts)
            generated_segment_count = 0
            generation_cap_blocked = False
            auto_extension_blocked_segment_idx: int | None = None
            prompt_sources_drained_logged = False
            generation_paused = bool(initial_rollout_prompt and not single_clip_mode and len(curated_prompts) == 0)
            pending_seed_reset = False
            pending_seed_reset_reason = ""
            pending_reset_conditioning = False
            loop_iteration = 0 if generation_paused else 1
            force_curated_restart_segment = False
            pending_simple_prompt_submission: PromptSubmission | None = None
            single_clip_waiting_for_request = False
            rollout_waiting_for_rewrite = False
            initial_rollout_waiting_for_rewrite = generation_paused
            segment_generation_active = False
            rewrite_restart_pending = False
            project_active = True
            project_stream_started = False
            pending_project_end = False

            def replace_session_init_image(initial_image_payload: object) -> None:
                nonlocal session_init_image
                next_session_init_image = persist_session_init_image(initial_image_payload)
                previous_session_init_image = session_init_image
                session_init_image = next_session_init_image
                if previous_session_init_image is not None:
                    cleanup_session_init_image(previous_session_init_image)

            async def schedule_simple_generate_request(payload: dict[str, object]) -> None:
                nonlocal preset_id
                nonlocal preset_label
                nonlocal enhancement_enabled
                nonlocal auto_extension_enabled
                nonlocal loop_generation_enabled
                nonlocal single_clip_mode
                nonlocal generation_paused
                nonlocal pending_seed_reset
                nonlocal pending_seed_reset_reason
                nonlocal seed_prompt_memory
                nonlocal curated_prompts
                nonlocal pending_simple_prompt_submission

                raw_prompt = str(payload.get("prompt", "")).strip()
                prompt_id = str(payload.get("prompt_id") or uuid.uuid4())
                if not raw_prompt:
                    await ws_send_json({
                        "type": "error",
                        "message": "simple_generate requires a non-empty prompt",
                    })
                    return

                blocked_error = get_prompt_safety_error(raw_prompt)
                if blocked_error is not None:
                    _main_print(
                        "WARN",
                        f"Simple prompt blocked for client {client_id[:8]} "
                        f"preview={preview_text(raw_prompt)}",
                    )
                    await log_event(
                        "prompt_blocked",
                        {
                            "kind": "user_raw",
                            "prompt_id": prompt_id,
                            "raw_prompt": raw_prompt,
                            "error": blocked_error,
                        },
                    )
                    await ws_send_json({
                        "type": "error",
                        "message": blocked_error,
                    })
                    return

                try:
                    replace_session_init_image(payload.get("initial_image"))
                except ValueError as exc:
                    await ws_send_json({
                        "type": "error",
                        "message": str(exc),
                    })
                    return

                next_preset_id = str(payload.get("preset_id") or "").strip()
                if next_preset_id:
                    preset_id = next_preset_id
                next_preset_label = str(payload.get("preset_label") or "").strip()
                if next_preset_label:
                    preset_label = next_preset_label

                next_enhancement_enabled = bool(payload.get("enhancement_enabled", True))

                enhancement_enabled = next_enhancement_enabled
                auto_extension_enabled = False
                loop_generation_enabled = False
                single_clip_mode = True
                generation_paused = False
                pending_seed_reset = True
                pending_seed_reset_reason = "simple_generate"
                seed_prompt_memory = ([] if next_enhancement_enabled else [raw_prompt])
                curated_prompts = list(seed_prompt_memory)
                pending_simple_prompt_submission = (PromptSubmission(
                    prompt_id=prompt_id,
                    raw_prompt=raw_prompt,
                    created_at_s=time.time(),
                ) if next_enhancement_enabled else None)

                await log_event(
                    "simple_generate",
                    {
                        "prompt_id": prompt_id,
                        "prompt": raw_prompt,
                        "preset_id": preset_id,
                        "enhancement_enabled": next_enhancement_enabled,
                        "has_initial_image": session_init_image is not None,
                    },
                )

                if session_init_image is not None:
                    _main_print(
                        "INFO", f"Client {client_id[:8]} set simple initial image: "
                        f"{session_init_image.display_name}")

                if next_enhancement_enabled:
                    await ws_send_json({
                        "type": "prompt_received",
                        "prompt_id": prompt_id,
                        "queue_depth": 1,
                    })

            async def apply_project_init_payload(payload: dict[str, object]) -> bool:
                nonlocal preset_id
                nonlocal preset_label
                nonlocal initial_rollout_prompt
                nonlocal enhancement_enabled
                nonlocal auto_extension_enabled
                nonlocal loop_generation_enabled
                nonlocal single_clip_mode
                nonlocal generation_paused
                nonlocal curated_prompts
                nonlocal seed_prompt_memory
                nonlocal curated_idx
                nonlocal segment_idx
                nonlocal locked_segment_prompts
                nonlocal pending_seed_reset
                nonlocal pending_seed_reset_reason
                nonlocal pending_reset_conditioning
                nonlocal force_curated_restart_segment
                nonlocal generated_segment_count
                nonlocal generation_cap_blocked
                nonlocal auto_extension_blocked_segment_idx
                nonlocal prompt_sources_drained_logged
                nonlocal pending_simple_prompt_submission
                nonlocal single_clip_waiting_for_request
                nonlocal rollout_waiting_for_rewrite
                nonlocal initial_rollout_waiting_for_rewrite
                nonlocal rewrite_restart_pending
                nonlocal loop_iteration
                nonlocal rewrite_model
                nonlocal rewrite_system_prompt
                nonlocal rewrite_system_prompt_override
                nonlocal rewrite_user_system_prompt_override
                nonlocal rewrite_temperature
                nonlocal project_active
                nonlocal project_stream_started
                nonlocal pending_project_end

                next_initial_rollout_prompt = str(payload.get("initial_rollout_prompt") or "").strip()
                next_enhancement_enabled = bool(payload.get("enhancement_enabled", True))
                next_auto_extension_enabled = bool(payload.get("auto_extension_enabled", False))
                next_loop_generation_enabled = bool(payload.get("loop_generation_enabled", False))
                next_single_clip_mode = bool(payload.get("single_clip_mode", False))

                next_preset_id = str(payload.get("preset_id") or "").strip()
                if next_preset_id:
                    preset_id = next_preset_id
                next_preset_label = str(payload.get("preset_label") or "").strip()
                if next_preset_label:
                    preset_label = next_preset_label

                next_rewrite_model = self.prompt_enhancer.resolve_rewrite_model(payload.get("rewrite_model"))
                next_rewrite_system_prompt_override = str(payload.get("rewrite_window_system_prompt") or "").strip()
                next_rewrite_user_system_prompt_override = str(payload.get("rewrite_user_system_prompt") or "").strip()
                next_rewrite_temperature = (self.prompt_enhancer.resolve_rewrite_temperature(
                    payload.get("rewrite_temperature")))

                next_curated_prompts = []
                incoming_prompts = payload.get("curated_prompts", [])
                if isinstance(incoming_prompts, list):
                    for prompt in incoming_prompts:
                        if isinstance(prompt, str) and prompt.strip():
                            next_curated_prompts.append(prompt.strip())

                blocked_init_prompt = get_first_blocked_prompt(next_curated_prompts)
                if blocked_init_prompt is not None:
                    message = ("Prompt safety filter blocked initial prompt "
                               f"{blocked_init_prompt.index + 1}. "
                               f"{blocked_init_prompt.error}")
                    _main_print(
                        "WARN",
                        f"Project init prompt blocked for client {client_id[:8]} "
                        f"index={blocked_init_prompt.index + 1} "
                        f"preview={preview_text(blocked_init_prompt.prompt)}",
                    )
                    await ws_send_json({
                        "type": "error",
                        "message": message,
                    })
                    return False

                try:
                    replace_session_init_image(payload.get("initial_image"))
                except ValueError as exc:
                    await ws_send_json({
                        "type": "error",
                        "message": str(exc),
                    })
                    return False

                initial_rollout_prompt = next_initial_rollout_prompt
                enhancement_enabled = next_enhancement_enabled
                auto_extension_enabled = next_auto_extension_enabled
                loop_generation_enabled = next_loop_generation_enabled
                single_clip_mode = next_single_clip_mode
                rewrite_model = next_rewrite_model
                rewrite_system_prompt_override = (next_rewrite_system_prompt_override)
                rewrite_user_system_prompt_override = (next_rewrite_user_system_prompt_override)
                rewrite_system_prompt = (
                    self.prompt_enhancer.resolve_rewrite_system_prompt(rewrite_system_prompt_override))
                rewrite_temperature = next_rewrite_temperature

                curated_prompts = list(next_curated_prompts)
                seed_prompt_memory = list(next_curated_prompts)
                curated_idx = 0
                segment_idx = 0
                locked_segment_prompts = []
                pending_seed_reset = False
                pending_seed_reset_reason = ""
                pending_reset_conditioning = True
                force_curated_restart_segment = False
                generated_segment_count = 0
                generation_cap_blocked = False
                auto_extension_blocked_segment_idx = None
                prompt_sources_drained_logged = False
                pending_simple_prompt_submission = None
                single_clip_waiting_for_request = False
                rollout_waiting_for_rewrite = False
                generation_paused = bool(initial_rollout_prompt and not single_clip_mode and len(curated_prompts) == 0)
                initial_rollout_waiting_for_rewrite = generation_paused
                rewrite_restart_pending = False
                loop_iteration = 0
                project_active = True
                project_stream_started = False
                pending_project_end = False

                return True

            async def run_rewrite_seed_prompts(
                rewrite_instruction: str,
                rewrite_model_name: str,
                rewrite_timeout_ms_value: int,
                rewrite_temperature_value: float,
                prompt_window_prompts: list[str] | None = None,
            ):
                nonlocal seed_prompt_memory
                nonlocal curated_prompts
                nonlocal preset_id
                nonlocal preset_label
                nonlocal rewrite_seed_prompts_task
                nonlocal pending_seed_reset
                nonlocal pending_seed_reset_reason
                nonlocal generated_segment_count
                nonlocal generation_cap_blocked
                nonlocal generation_paused
                nonlocal rollout_waiting_for_rewrite
                nonlocal initial_rollout_waiting_for_rewrite
                nonlocal rewrite_restart_pending
                nonlocal rewrite_system_prompt
                nonlocal rewrite_system_prompt_override
                nonlocal rewrite_user_system_prompt_override
                try:
                    snapshot_prompts = (list(prompt_window_prompts) if isinstance(prompt_window_prompts, list)
                                        and len(prompt_window_prompts) > 0 else list(seed_prompt_memory))
                    effective_system_prompt = (self.prompt_enhancer.resolve_rewrite_new_rollout_system_prompt(
                        rewrite_user_system_prompt_override or rewrite_system_prompt_override, )
                                               if len(snapshot_prompts) == 0 else rewrite_system_prompt)
                    rewrite_result = await self.prompt_enhancer.rewrite_prompt_sequence(
                        snapshot_prompts,
                        preset_id=preset_id,
                        preset_label=preset_label,
                        rewrite_instruction=rewrite_instruction,
                        rewrite_model=rewrite_model_name,
                        rewrite_temperature=rewrite_temperature_value,
                        timeout_ms=rewrite_timeout_ms_value,
                        system_prompt_override=effective_system_prompt,
                    )

                    if stop_event.is_set():
                        return

                    blocked_rewrite_prompt = get_first_blocked_prompt(rewrite_result.prompts)
                    if blocked_rewrite_prompt is not None:
                        error_message = ("Prompt safety filter blocked rewritten prompt "
                                         f"{blocked_rewrite_prompt.index + 1}. "
                                         f"{blocked_rewrite_prompt.error}")
                        await log_event(
                            "rewrite_blocked",
                            {
                                "kind": "seed_rewrite",
                                "rewrite_instruction": rewrite_instruction,
                                "model": rewrite_result.model,
                                "prompt_window_prompts": snapshot_prompts,
                                "rewritten_prompts": list(rewrite_result.prompts),
                                "blocked_prompt_index": blocked_rewrite_prompt.index,
                                "blocked_prompt": blocked_rewrite_prompt.prompt,
                                "error": error_message,
                            },
                        )
                        await ws_send_json({
                            "type": "rewrite_seed_prompts_complete",
                            "fallback_used": True,
                            "error": error_message,
                            "model": rewrite_result.model,
                            "latency_ms": round(rewrite_result.latency_ms, 2),
                        })
                        rewrite_restart_pending = False
                        return

                    seed_prompt_memory = list(rewrite_result.prompts)
                    curated_prompts = list(seed_prompt_memory)
                    if not rewrite_result.fallback_used:
                        preset_id = rewrite_result.rollout_id
                        preset_label = rewrite_result.rollout_label
                    if initial_rollout_waiting_for_rewrite and not rewrite_result.fallback_used:
                        pending_seed_reset = True
                        pending_seed_reset_reason = "initial_rewrite"
                        generation_paused = False
                        initial_rollout_waiting_for_rewrite = False
                        await ws_send_json({
                            "type": "generation_paused_updated",
                            "paused": generation_paused,
                        })
                    if rollout_waiting_for_rewrite and not rewrite_result.fallback_used:
                        pending_seed_reset = True
                        pending_seed_reset_reason = "rewrite_rollout"
                        generated_segment_count = 0
                        generation_cap_blocked = False
                        generation_paused = False
                    elif rewrite_restart_pending and not rewrite_result.fallback_used:
                        pending_seed_reset = True
                        pending_seed_reset_reason = "rewrite_during_generation"
                        generated_segment_count = 0
                        generation_cap_blocked = False
                        generation_paused = False
                        rewrite_restart_pending = False
                    elif rewrite_restart_pending:
                        rewrite_restart_pending = False
                    await log_event(
                        "rewrite_done",
                        {
                            "kind": "seed_rewrite",
                            "rewrite_instruction": rewrite_instruction,
                            "latency_ms": round(rewrite_result.latency_ms, 2),
                            "response": rewrite_result.raw_response_text or "",
                        },
                    )
                    await ws_send_json({
                        "type": "seed_prompts_updated",
                        "prompts": seed_prompt_memory,
                        "preset_id": preset_id,
                        "preset_label": preset_label,
                        "reason": "rewrite",
                        "fallback_used": rewrite_result.fallback_used,
                        "error": rewrite_result.error,
                        "model": rewrite_result.model,
                        "latency_ms": round(rewrite_result.latency_ms, 2),
                        "raw_llm_output": rewrite_result.raw_response_text,
                        "rewrite_instruction": rewrite_instruction,
                    })
                    await ws_send_json({
                        "type": "rewrite_seed_prompts_complete",
                        "fallback_used": rewrite_result.fallback_used,
                        "error": rewrite_result.error,
                        "preset_id": preset_id,
                        "preset_label": preset_label,
                        "model": rewrite_result.model,
                        "latency_ms": round(rewrite_result.latency_ms, 2),
                    })
                except asyncio.CancelledError:
                    rewrite_restart_pending = False
                    await log_event(
                        "rewrite_cancelled",
                        {
                            "kind": "seed_rewrite",
                            "rewrite_instruction": rewrite_instruction,
                            "rewrite_model": rewrite_model_name,
                        },
                    )
                    raise
                except Exception as exc:
                    if stop_event.is_set():
                        return
                    await log_event(
                        "rewrite_exception",
                        {
                            "kind": "seed_rewrite",
                            "rewrite_instruction": rewrite_instruction,
                            "rewrite_model": rewrite_model_name,
                            "error": str(exc),
                        },
                    )
                    await ws_send_json({
                        "type": "rewrite_seed_prompts_complete",
                        "fallback_used": True,
                        "error": str(exc),
                        "model": rewrite_model_name,
                        "latency_ms": 0.0,
                    })
                    rewrite_restart_pending = False
                finally:
                    rewrite_seed_prompts_task = None

            async def websocket_reader_loop():
                nonlocal enhancement_enabled
                nonlocal auto_extension_enabled
                nonlocal loop_generation_enabled
                nonlocal generation_paused
                nonlocal pending_seed_reset
                nonlocal pending_seed_reset_reason
                nonlocal generated_segment_count
                nonlocal generation_cap_blocked
                nonlocal auto_extension_blocked_segment_idx
                nonlocal seed_prompt_memory
                nonlocal curated_prompts
                nonlocal rewrite_seed_prompts_task
                nonlocal rewrite_model
                nonlocal rewrite_system_prompt
                nonlocal rewrite_system_prompt_override
                nonlocal rewrite_user_system_prompt_override
                nonlocal rewrite_timeout_ms
                nonlocal rewrite_temperature
                nonlocal rollout_waiting_for_rewrite
                nonlocal initial_rollout_waiting_for_rewrite
                nonlocal rewrite_restart_pending
                nonlocal project_active
                nonlocal project_stream_started
                nonlocal pending_project_end
                while not stop_event.is_set():
                    try:
                        data = await websocket.receive_json()
                    except WebSocketDisconnect:
                        loop_generation_enabled = False
                        stop_event.set()
                        return
                    except Exception as exc:
                        _main_print("INFO", f"Client {client_id[:8]} reader stopped: {exc}")
                        loop_generation_enabled = False
                        stop_event.set()
                        return

                    msg_type = data.get("type")
                    if msg_type == "leave":
                        loop_generation_enabled = False
                        stop_event.set()
                        return

                    if msg_type == "end_project_keep_session":
                        if (rewrite_seed_prompts_task is not None and not rewrite_seed_prompts_task.done()):
                            await cancel_task(rewrite_seed_prompts_task)
                        loop_generation_enabled = False
                        generation_paused = False
                        rollout_waiting_for_rewrite = False
                        initial_rollout_waiting_for_rewrite = False
                        rewrite_restart_pending = False
                        pending_project_end = True
                        continue

                    if msg_type == "project_init_v1":
                        if project_active or pending_project_end:
                            await ws_send_json({
                                "type":
                                "error",
                                "message": ("Current project is still ending. "
                                            "Wait for the reset to finish before starting another project."),
                            })
                            continue
                        init_applied = await apply_project_init_payload(data)
                        if not init_applied:
                            continue
                        if initial_rollout_waiting_for_rewrite:
                            await ws_send_json({
                                "type": "rewrite_seed_prompts_started",
                                "model": rewrite_model,
                            })
                            rewrite_seed_prompts_task = asyncio.create_task(
                                run_rewrite_seed_prompts(
                                    initial_rollout_prompt,
                                    rewrite_model,
                                    rewrite_timeout_ms,
                                    rewrite_temperature,
                                    [],
                                ))
                        else:
                            pending_seed_reset = True
                            pending_seed_reset_reason = "project_init"
                        continue

                    if not project_active:
                        if msg_type in {
                                "append_prompt",
                                "rewrite_seed_prompts",
                                "reset_to_seed_prompts",
                                "restart_generation",
                        }:
                            await ws_send_json({
                                "type": "error",
                                "message": ("Start a new project before sending prompts."),
                            })
                        continue

                    if msg_type == "append_prompt":
                        if ((rollout_waiting_for_rewrite or initial_rollout_waiting_for_rewrite)
                                and not single_clip_mode):
                            await ws_send_json({
                                "type": "error",
                                "message": ("Use Rewrite to create or restart the rollout."),
                            })
                            continue
                        raw_prompt = str(data.get("prompt", "")).strip()
                        prompt_id = str(data.get("prompt_id") or uuid.uuid4())
                        if not raw_prompt:
                            await ws_send_json({
                                "type": "error",
                                "message": "append_prompt requires a non-empty prompt",
                            })
                            continue

                        blocked_error = get_prompt_safety_error(raw_prompt)
                        if blocked_error is not None:
                            _main_print(
                                "WARN",
                                f"Live prompt blocked for client {client_id[:8]} "
                                f"preview={preview_text(raw_prompt)}",
                            )
                            await log_event(
                                "prompt_blocked",
                                {
                                    "kind": "user_raw",
                                    "prompt_id": prompt_id,
                                    "raw_prompt": raw_prompt,
                                    "error": blocked_error,
                                },
                            )
                            await ws_send_json({
                                "type": "error",
                                "message": blocked_error,
                            })
                            continue

                        submission = PromptSubmission(
                            prompt_id=prompt_id,
                            raw_prompt=raw_prompt,
                            created_at_s=time.time(),
                        )
                        await raw_prompt_queue.put(submission)
                        await log_event(
                            "append_prompt",
                            {
                                "prompt": raw_prompt,
                            },
                        )
                        await ws_send_json({
                            "type": "prompt_received",
                            "prompt_id": prompt_id,
                            "queue_depth": raw_prompt_queue.qsize(),
                        })
                        continue

                    if msg_type == "simple_generate":
                        await schedule_simple_generate_request(data)
                        continue

                    if msg_type == "set_enhancement":
                        enhancement_enabled = bool(data.get("enabled", enhancement_enabled))
                        await ws_send_json({
                            "type": "enhancement_updated",
                            "enabled": enhancement_enabled,
                        })
                        continue

                    if msg_type == "set_auto_extension":
                        was_enabled = auto_extension_enabled
                        auto_extension_enabled = bool(data.get("enabled", auto_extension_enabled))
                        if not auto_extension_enabled:
                            auto_extension_blocked_segment_idx = None
                            _main_print(
                                "INFO",
                                f"Auto extension disabled for client {client_id[:8]} "
                                "(cleared_blocked_state=true)",
                            )
                        else:
                            cleared = auto_extension_blocked_segment_idx is not None
                            auto_extension_blocked_segment_idx = None
                            _main_print(
                                "INFO",
                                f"Auto extension enabled for client {client_id[:8]} "
                                f"(was_enabled={was_enabled} "
                                f"cleared_blocked_state={cleared})",
                            )
                        await ws_send_json({
                            "type": "auto_extension_updated",
                            "enabled": auto_extension_enabled,
                        })
                        continue

                    if msg_type == "set_loop_generation":
                        loop_generation_enabled = bool(data.get("enabled", loop_generation_enabled))
                        await ws_send_json({
                            "type": "loop_generation_updated",
                            "enabled": loop_generation_enabled,
                        })
                        continue

                    if msg_type == "set_generation_paused":
                        generation_paused = bool(data.get("paused", generation_paused))
                        await ws_send_json({
                            "type": "generation_paused_updated",
                            "paused": generation_paused,
                        })
                        continue

                    if msg_type == "reset_to_seed_prompts":
                        pending_seed_reset = True
                        pending_seed_reset_reason = "manual_reset"
                        await ws_send_json({
                            "type": "seed_prompts_reset_pending",
                            "seed_prompt_count": len(seed_prompt_memory),
                        })
                        continue

                    if msg_type == "restart_generation":
                        generated_segment_count = 0
                        generation_cap_blocked = False
                        rollout_waiting_for_rewrite = False
                        pending_seed_reset = True
                        pending_seed_reset_reason = "cap_restart"
                        generation_paused = False
                        await ws_send_json({
                            "type":
                            "generation_restarted",
                            "reason":
                            "cap_restart",
                            "segment_cap":
                            _resolve_generation_segment_cap(
                                single_clip_mode=single_clip_mode,
                                cap=GENERATION_SEGMENT_CAP,
                            ),
                        })
                        continue

                    if msg_type == "rewrite_seed_prompts":
                        rewrite_instruction = str(data.get("rewrite_instruction", "")).strip()
                        rewrite_model = self.prompt_enhancer.resolve_rewrite_model(data.get("rewrite_model"))
                        rewrite_system_prompt_override = str(data.get("rewrite_window_system_prompt") or "").strip()
                        rewrite_user_system_prompt_override = str(data.get("rewrite_user_system_prompt") or "").strip()
                        rewrite_system_prompt = self.prompt_enhancer.resolve_rewrite_system_prompt(
                            rewrite_system_prompt_override)
                        rewrite_temperature = self.prompt_enhancer.resolve_rewrite_temperature(
                            data.get("rewrite_temperature"))
                        if len(seed_prompt_memory) == 0 and not (initial_rollout_waiting_for_rewrite
                                                                 and rewrite_instruction):
                            await ws_send_json({
                                "type": "error",
                                "message": "No seed prompts available to rewrite.",
                            })
                            continue
                        if rewrite_seed_prompts_task is not None and not rewrite_seed_prompts_task.done():
                            await ws_send_json({
                                "type": "error",
                                "message": "rewrite_seed_prompts already in progress.",
                            })
                            continue
                        if (segment_generation_active and not single_clip_mode and not rollout_waiting_for_rewrite
                                and not initial_rollout_waiting_for_rewrite):
                            rewrite_restart_pending = True

                        await ws_send_json({
                            "type": "rewrite_seed_prompts_started",
                            "model": rewrite_model,
                        })
                        requested_prompt_window = normalize_prompt_window_prompts(data.get("prompt_window_prompts"))
                        rewrite_seed_prompts_task = asyncio.create_task(
                            run_rewrite_seed_prompts(
                                rewrite_instruction,
                                rewrite_model,
                                rewrite_timeout_ms,
                                rewrite_temperature,
                                requested_prompt_window,
                            ))
                        continue

                    if msg_type == "set_rewrite_model":
                        rewrite_model = self.prompt_enhancer.resolve_rewrite_model(data.get("rewrite_model"))
                        await ws_send_json({
                            "type": "rewrite_model_updated",
                            "rewrite_model": rewrite_model,
                        })
                        continue

                    if msg_type == "set_rewrite_temperature":
                        rewrite_temperature = self.prompt_enhancer.resolve_rewrite_temperature(
                            data.get("rewrite_temperature"))
                        await ws_send_json({
                            "type": "rewrite_temperature_updated",
                            "rewrite_temperature": rewrite_temperature,
                        })
                        continue

            async def prompt_worker_loop():
                while not stop_event.is_set():
                    try:
                        submission = await asyncio.wait_for(raw_prompt_queue.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue
                    _main_print("INFO", f"Received user prompt for enhancement: {submission.raw_prompt}")
                    prompt_id = submission.prompt_id
                    raw_prompt = submission.raw_prompt
                    blocked_raw_prompt_error = get_prompt_safety_error(raw_prompt)
                    if blocked_raw_prompt_error is not None:
                        _main_print(
                            "WARN",
                            f"User prompt blocked for client {client_id[:8]} "
                            f"prompt_id={prompt_id} preview={preview_text(raw_prompt)}",
                        )
                        await log_event(
                            "prompt_blocked",
                            payload={
                                "kind": "user_raw",
                                "prompt_id": prompt_id,
                                "raw_prompt": raw_prompt,
                                "error": blocked_raw_prompt_error,
                            },
                        )
                        await ws_send_json({
                            "type": "error",
                            "message": blocked_raw_prompt_error,
                        })
                        continue
                    await log_event(
                        "enhance_request",
                        {
                            "prompt_id": prompt_id,
                            "raw_prompt": raw_prompt,
                            "enhancement_enabled": enhancement_enabled,
                            "rewrite_model": rewrite_model,
                        },
                    )
                    if enhancement_enabled:
                        locked_snapshot = list(locked_segment_prompts)
                        next_segment_idx = len(locked_snapshot) + 1
                        _main_print(
                            "INFO",
                            f"Enhancement context: next_segment={next_segment_idx} "
                            f"locked_count={len(locked_snapshot)}",
                        )
                        await ws_send_json({
                            "type": "prompt_enhancing",
                            "prompt_id": prompt_id,
                        })
                        result = await self.prompt_enhancer.enhance_prompt(
                            raw_prompt,
                            locked_segments=locked_snapshot,
                            next_segment_idx=next_segment_idx,
                            preset_id=preset_id,
                            mode="single_clip" if single_clip_mode else "user_live",
                            model=rewrite_model,
                            timeout_ms=PROMPT_TIMEOUT_MS,
                        )
                        final_prompt = result.prompt.strip() if result.prompt else ""
                        blocked_final_prompt_error = get_prompt_safety_error(final_prompt)
                        _main_print("INFO", f"Enhanced prompt: {final_prompt}")
                        await log_event(
                            "rewrite_done",
                            {
                                "kind": "enhance_prompt",
                                "latency_ms": round(result.latency_ms, 2),
                                "response": final_prompt,
                            },
                        )
                        if blocked_final_prompt_error is not None:
                            await log_event(
                                "prompt_blocked",
                                {
                                    "kind": "user_enhanced",
                                    "prompt_id": prompt_id,
                                    "raw_prompt": raw_prompt,
                                    "output_prompt": final_prompt,
                                    "fallback_used": result.fallback_used,
                                    "latency_ms": round(result.latency_ms, 2),
                                    "provider": result.provider,
                                    "model": result.model,
                                    "error": blocked_final_prompt_error,
                                },
                            )
                            await ws_send_json({
                                "type": "error",
                                "message": blocked_final_prompt_error,
                            })
                            continue
                        if result.fallback_used or not final_prompt:
                            source = "user_enhancement_failed"
                            _main_print(
                                "WARN",
                                f"User prompt enhancement fallback: client={client_id[:8]} "
                                f"prompt_id={prompt_id} latency_ms={result.latency_ms:.2f} "
                                f"error={result.error}",
                            )
                            await ws_send_json({
                                "type": "prompt_fallback_used",
                                "prompt_id": prompt_id,
                                "prompt": final_prompt,
                                "source": source,
                                "latency_ms": round(result.latency_ms, 2),
                                "error": PROMPT_EXTENSION_FAILURE_USER_MESSAGE,
                            })
                            # Enhancement is strict JSON-only; do not enqueue raw
                            # prompt when enhancement fails.
                            continue
                        else:
                            source = "user_enhanced"
                            await ws_send_json({
                                "type": "prompt_ready",
                                "prompt_id": prompt_id,
                                "prompt": final_prompt,
                                "source": source,
                                "latency_ms": round(result.latency_ms, 2),
                            })
                        await ready_prompt_queue.put(
                            ReadyPrompt(
                                prompt_id=prompt_id,
                                prompt=final_prompt,
                                source=source,
                                fallback_used=result.fallback_used,
                                loop_iteration=loop_iteration,
                            ))
                    else:
                        await ready_prompt_queue.put(
                            ReadyPrompt(
                                prompt_id=prompt_id,
                                prompt=raw_prompt,
                                source="user_raw",
                                fallback_used=False,
                                loop_iteration=loop_iteration,
                            ))
                        await ws_send_json({
                            "type": "prompt_ready",
                            "prompt_id": prompt_id,
                            "prompt": raw_prompt,
                            "source": "user_raw",
                            "latency_ms": 0.0,
                        })

            def queue_snapshot() -> dict[str, object]:
                return {
                    "user_ready": ready_prompt_queue.qsize(),
                    "curated_remaining": max(len(curated_prompts) - curated_idx, 0),
                    "auto_enabled": auto_extension_enabled,
                    "auto_blocked_segment": auto_extension_blocked_segment_idx,
                    "loop_enabled": loop_generation_enabled,
                    "paused": generation_paused,
                    "seed_prompt_count": len(seed_prompt_memory),
                    "loop_iteration": loop_iteration,
                }

            def drain_queue_nowait(queue: asyncio.Queue) -> int:
                drained = 0
                while True:
                    try:
                        queue.get_nowait()
                        drained += 1
                    except asyncio.QueueEmpty:
                        break
                return drained

            async def enter_project_idle() -> None:
                nonlocal curated_prompts
                nonlocal seed_prompt_memory
                nonlocal curated_idx
                nonlocal segment_idx
                nonlocal locked_segment_prompts
                nonlocal generated_segment_count
                nonlocal generation_cap_blocked
                nonlocal auto_extension_blocked_segment_idx
                nonlocal prompt_sources_drained_logged
                nonlocal generation_paused
                nonlocal pending_seed_reset
                nonlocal pending_seed_reset_reason
                nonlocal pending_reset_conditioning
                nonlocal force_curated_restart_segment
                nonlocal pending_simple_prompt_submission
                nonlocal single_clip_waiting_for_request
                nonlocal rollout_waiting_for_rewrite
                nonlocal initial_rollout_waiting_for_rewrite
                nonlocal rewrite_restart_pending
                nonlocal initial_rollout_prompt
                nonlocal project_active
                nonlocal project_stream_started
                nonlocal pending_project_end

                dropped_raw = drain_queue_nowait(raw_prompt_queue)
                dropped_ready = drain_queue_nowait(ready_prompt_queue)

                curated_prompts = []
                seed_prompt_memory = []
                curated_idx = 0
                segment_idx = 0
                locked_segment_prompts = []
                generated_segment_count = 0
                generation_cap_blocked = False
                auto_extension_blocked_segment_idx = None
                prompt_sources_drained_logged = False
                generation_paused = False
                pending_seed_reset = False
                pending_seed_reset_reason = ""
                pending_reset_conditioning = True
                force_curated_restart_segment = False
                pending_simple_prompt_submission = None
                single_clip_waiting_for_request = False
                rollout_waiting_for_rewrite = False
                initial_rollout_waiting_for_rewrite = False
                rewrite_restart_pending = False
                initial_rollout_prompt = ""
                project_active = False
                pending_project_end = False

                if project_stream_started:
                    project_stream_started = False
                    await ws_send_json({"type": "ltx2_stream_complete"})
                    await log_event("ws_stream_complete")

                await ws_send_json({
                    "type": "project_idle",
                    "dropped_user_raw_queue": dropped_raw,
                    "dropped_user_ready_queue": dropped_ready,
                })

            def pick_next_prompt_nowait() -> ReadyPrompt | None:
                nonlocal curated_idx
                try:
                    return ready_prompt_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass

                if curated_idx < len(curated_prompts):
                    prompt_idx = curated_idx
                    prompt = curated_prompts[prompt_idx]
                    curated_idx += 1
                    return ReadyPrompt(
                        prompt=prompt,
                        source="curated",
                        seed_prompt_index=prompt_idx,
                        loop_iteration=loop_iteration,
                    )

                return None

            websocket_reader_task = asyncio.create_task(websocket_reader_loop())
            prompt_worker_task = asyncio.create_task(prompt_worker_loop())
            await asyncio.sleep(0)

            await ws_send_json({
                "type": "loop_generation_updated",
                "enabled": loop_generation_enabled,
            })
            await ws_send_json({
                "type": "generation_paused_updated",
                "paused": generation_paused,
            })
            if pending_project_end:
                await enter_project_idle()
            elif initial_rollout_waiting_for_rewrite:
                await ws_send_json({
                    "type": "rewrite_seed_prompts_started",
                    "model": rewrite_model,
                })
                rewrite_seed_prompts_task = asyncio.create_task(
                    run_rewrite_seed_prompts(
                        initial_rollout_prompt,
                        rewrite_model,
                        rewrite_timeout_ms,
                        rewrite_temperature,
                        [],
                    ))
            else:
                project_stream_started = True
                await ws_send_json({
                    "type":
                    "ltx2_stream_start",
                    "total_segments":
                    len(curated_prompts),
                    "preset_id":
                    preset_id,
                    "stream_mode":
                    "av_fmp4",
                    "live_mode":
                    True,
                    "loop_generation_enabled":
                    loop_generation_enabled,
                    "loop_iteration":
                    loop_iteration,
                    "generation_segment_cap":
                    _resolve_generation_segment_cap(
                        single_clip_mode=single_clip_mode,
                        cap=GENERATION_SEGMENT_CAP,
                    ),
                })
                await ws_send_json({
                    "type": "seed_prompts_updated",
                    "prompts": seed_prompt_memory,
                    "preset_id": preset_id,
                    "preset_label": preset_label,
                    "reason": "init",
                    "seed_prompt_count": len(seed_prompt_memory),
                })

            av_event_queue = slot.register_stream_queue(client_id)

            while not stop_event.is_set():
                if pending_project_end:
                    await enter_project_idle()
                    await asyncio.sleep(0)
                    continue

                if not project_active:
                    await asyncio.sleep(0.05)
                    continue

                if pending_seed_reset:
                    nonlocal_reason = pending_seed_reset_reason
                    curated_prompts = list(seed_prompt_memory)
                    curated_idx = 0
                    segment_idx = 0
                    locked_segment_prompts = []
                    auto_extension_blocked_segment_idx = None
                    prompt_sources_drained_logged = False
                    pending_seed_reset = False
                    pending_reset_conditioning = True
                    force_curated_restart_segment = len(curated_prompts) > 0
                    single_clip_waiting_for_request = False
                    dropped_raw = drain_queue_nowait(raw_prompt_queue)
                    dropped_ready = drain_queue_nowait(ready_prompt_queue)

                    loop_iteration += 1
                    project_stream_started = True
                    await ws_send_json({
                        "type":
                        "ltx2_stream_start",
                        "total_segments":
                        len(curated_prompts),
                        "preset_id":
                        preset_id,
                        "stream_mode":
                        "av_fmp4",
                        "live_mode":
                        True,
                        "loop_generation_enabled":
                        loop_generation_enabled,
                        "loop_iteration":
                        loop_iteration,
                        "generation_segment_cap":
                        _resolve_generation_segment_cap(
                            single_clip_mode=single_clip_mode,
                            cap=GENERATION_SEGMENT_CAP,
                        ),
                    })
                    if nonlocal_reason == "loop_restart":
                        await ws_send_json({
                            "type": "loop_restarted",
                            "loop_iteration": loop_iteration,
                            "seed_prompt_count": len(seed_prompt_memory),
                            "dropped_user_raw_queue": dropped_raw,
                            "dropped_user_ready_queue": dropped_ready,
                        })
                    else:
                        await ws_send_json({
                            "type": "seed_prompts_reset_applied",
                            "reason": nonlocal_reason or "manual_reset",
                            "loop_iteration": loop_iteration,
                            "seed_prompt_count": len(seed_prompt_memory),
                            "dropped_user_raw_queue": dropped_raw,
                            "dropped_user_ready_queue": dropped_ready,
                        })
                    if nonlocal_reason == "cap_restart":
                        generation_cap_blocked = False
                        generated_segment_count = 0
                    rollout_waiting_for_rewrite = False
                    pending_seed_reset_reason = ""

                if pending_simple_prompt_submission is not None:
                    await raw_prompt_queue.put(pending_simple_prompt_submission)
                    pending_simple_prompt_submission = None

                if (not single_clip_mode and not generation_cap_blocked and not rollout_waiting_for_rewrite
                        and GENERATION_SEGMENT_CAP > 0 and generated_segment_count >= GENERATION_SEGMENT_CAP):
                    loop_generation_enabled = False
                    rollout_waiting_for_rewrite = True
                    _main_print(
                        "INFO",
                        f"Segment cap reached for client {client_id[:8]} "
                        f"(cap_segments={GENERATION_SEGMENT_CAP}, "
                        f"generated_segments={generated_segment_count}); "
                        "waiting for rollout rewrite",
                    )
                    project_stream_started = False
                    await ws_send_json({"type": "ltx2_stream_complete"})
                    await log_event("ws_stream_complete")

                if generation_cap_blocked:
                    await asyncio.sleep(0.05)
                    continue

                if rollout_waiting_for_rewrite:
                    await asyncio.sleep(0.05)
                    continue

                if initial_rollout_waiting_for_rewrite:
                    await asyncio.sleep(0.05)
                    continue

                if generation_paused:
                    await asyncio.sleep(0.05)
                    continue

                if rewrite_restart_pending:
                    if (rewrite_seed_prompts_task is not None and not rewrite_seed_prompts_task.done()):
                        await asyncio.sleep(0.05)
                        continue
                    rewrite_restart_pending = False

                if force_curated_restart_segment and curated_idx < len(curated_prompts):
                    prompt_idx = curated_idx
                    prompt = curated_prompts[prompt_idx]
                    curated_idx += 1
                    selected = ReadyPrompt(
                        prompt=prompt,
                        source="curated",
                        seed_prompt_index=prompt_idx,
                        loop_iteration=loop_iteration,
                    )
                    force_curated_restart_segment = False
                else:
                    force_curated_restart_segment = False
                    selected = pick_next_prompt_nowait()
                if selected is None and loop_generation_enabled and len(seed_prompt_memory) > 0:
                    pending_seed_reset = True
                    pending_seed_reset_reason = "loop_restart"
                    await asyncio.sleep(0)
                    continue

                if selected is None and auto_extension_enabled:
                    next_segment = len(locked_segment_prompts) + 1
                    if auto_extension_blocked_segment_idx == next_segment:
                        if not prompt_sources_drained_logged:
                            snapshot = queue_snapshot()
                            _main_print(
                                "WARN",
                                f"Prompt sources drained for client {client_id[:8]} "
                                f"at segment boundary {segment_idx + 1} "
                                f"(user_ready={snapshot['user_ready']} "
                                f"curated_remaining={snapshot['curated_remaining']} "
                                f"auto_enabled={snapshot['auto_enabled']} "
                                "auto_state=blocked_after_failed_attempt)",
                            )
                            prompt_sources_drained_logged = True
                        await asyncio.sleep(PROMPT_AUTO_SLEEP_MS / 1000.0)
                        continue

                    locked_snapshot = list(locked_segment_prompts)
                    last_locked_preview = (preview_text(locked_snapshot[-1]) if locked_snapshot else "(none)")
                    _main_print(
                        "INFO",
                        f"Auto prompt generation request: client={client_id[:8]} "
                        f"next_segment={next_segment} "
                        f"locked_count={len(locked_snapshot)} "
                        f"last_locked_preview={last_locked_preview}",
                    )
                    result = await self.prompt_enhancer.generate_auto_prompt(
                        locked_segments=locked_snapshot,
                        next_segment_idx=next_segment,
                        model=rewrite_model,
                        timeout_ms=PROMPT_AUTO_TIMEOUT_MS,
                    )
                    _main_print(
                        "INFO",
                        f"Auto prompt generation response: client={client_id[:8]} "
                        f"next_segment={next_segment} "
                        f"fallback_used={result.fallback_used} "
                        f"returned_len={len(result.prompt)} "
                        f"latency_ms={result.latency_ms:.2f} error={result.error} "
                        f"returned_preview={preview_text(result.prompt)}",
                    )
                    auto_prompt = result.prompt.strip() if result.prompt else ""
                    blocked_auto_prompt_error = get_prompt_safety_error(auto_prompt)
                    if result.fallback_used or not auto_prompt:
                        auto_extension_blocked_segment_idx = next_segment
                        _main_print(
                            "WARN",
                            f"Auto prompt generation failed: client={client_id[:8]} "
                            f"next_segment={next_segment} "
                            f"latency_ms={result.latency_ms:.2f} "
                            f"error={result.error or 'Auto prompt generation returned no prompt.'} "
                            "policy=blocked_no_retry_until_external_action",
                        )
                        await ws_send_json({
                            "type": "auto_prompt_failed",
                            "segment_idx": next_segment,
                            "latency_ms": round(result.latency_ms, 2),
                            "error": result.error or "Auto prompt generation returned no prompt.",
                        })
                        await asyncio.sleep(PROMPT_AUTO_SLEEP_MS / 1000.0)
                        continue
                    if blocked_auto_prompt_error is not None:
                        auto_extension_blocked_segment_idx = next_segment
                        await log_event(
                            "auto_prompt_blocked",
                            {
                                "next_segment_idx": next_segment,
                                "output_prompt": auto_prompt,
                                "error": blocked_auto_prompt_error,
                                "latency_ms": round(result.latency_ms, 2),
                                "provider": result.provider,
                                "model": result.model,
                            },
                        )
                        await ws_send_json({
                            "type": "auto_prompt_failed",
                            "segment_idx": next_segment,
                            "latency_ms": round(result.latency_ms, 2),
                            "error": blocked_auto_prompt_error,
                        })
                        await asyncio.sleep(PROMPT_AUTO_SLEEP_MS / 1000.0)
                        continue

                    auto_extension_blocked_segment_idx = None
                    selected = ReadyPrompt(
                        prompt=auto_prompt,
                        source="auto_llm",
                        prompt_id=None,
                        fallback_used=False,
                        loop_iteration=loop_iteration,
                    )

                if selected is None:
                    if single_clip_mode:
                        await asyncio.sleep(PROMPT_AUTO_SLEEP_MS / 1000.0)
                        continue
                    if not prompt_sources_drained_logged:
                        snapshot = queue_snapshot()
                        _main_print(
                            "WARN",
                            f"Prompt sources drained for client {client_id[:8]} "
                            f"at segment boundary {segment_idx + 1} "
                            f"(user_ready={snapshot['user_ready']} "
                            f"curated_remaining={snapshot['curated_remaining']} "
                            f"auto_enabled={snapshot['auto_enabled']} "
                            f"auto_blocked_segment={snapshot['auto_blocked_segment']})",
                        )
                        prompt_sources_drained_logged = True
                        await ws_send_json({
                            "type": "prompt_sources_blocked",
                            "segment_idx": segment_idx + 1,
                        })
                    await asyncio.sleep(PROMPT_AUTO_SLEEP_MS / 1000.0)
                    continue

                if prompt_sources_drained_logged:
                    prompt_sources_drained_logged = False
                    _main_print(
                        "INFO",
                        f"Prompt source available again for client {client_id[:8]} "
                        f"at segment boundary {segment_idx + 1} "
                        f"(source={selected.source} prompt_id={selected.prompt_id})",
                    )
                    await ws_send_json({
                        "type": "prompt_sources_resumed",
                        "segment_idx": segment_idx + 1,
                        "source": selected.source,
                        "prompt_id": selected.prompt_id,
                    })

                segment_idx += 1
                single_clip_waiting_for_request = False
                total_segments_hint = max(segment_idx, len(curated_prompts))
                prompt = selected.prompt
                locked_segment_prompts.append(prompt)
                if (auto_extension_blocked_segment_idx is not None
                        and auto_extension_blocked_segment_idx <= segment_idx):
                    auto_extension_blocked_segment_idx = None

                await ws_send_json({
                    "type": "segment_prompt_source",
                    "segment_idx": segment_idx,
                    "source": selected.source,
                    "prompt_id": selected.prompt_id,
                    "fallback_used": selected.fallback_used,
                    "seed_prompt_index": selected.seed_prompt_index,
                    "loop_iteration": selected.loop_iteration or loop_iteration,
                })
                await ws_send_json({
                    "type": "ltx2_segment_start",
                    "segment_idx": segment_idx,
                    "total_segments": total_segments_hint,
                    "prompt": prompt,
                    "source": selected.source,
                    "seed_prompt_index": selected.seed_prompt_index,
                    "loop_iteration": selected.loop_iteration or loop_iteration,
                })
                await log_event(
                    "segment_start",
                    {
                        "segment_idx": segment_idx,
                    },
                )

                t_start = time.perf_counter()
                timings: dict = {}
                av_chunks_relayed = 0
                av_bytes_relayed = 0
                av_streamed = False

                step_reset_conditioning = pending_reset_conditioning
                pending_reset_conditioning = False
                step_image_path = (str(session_init_image.file_path)
                                   if segment_idx == 1 and session_init_image is not None else None)
                step_task = asyncio.create_task(
                    slot.user_step(
                        client_id,
                        prompt=prompt,
                        segment_idx=segment_idx,
                        image_path=step_image_path,
                        reset_conditioning=step_reset_conditioning,
                    ))
                segment_generation_active = True
                try:
                    while not stop_event.is_set() and (not step_task.done() or not av_event_queue.empty()):
                        try:
                            event = await asyncio.wait_for(av_event_queue.get(), timeout=0.05)
                        except asyncio.TimeoutError:
                            continue

                        if event.segment_idx != segment_idx:
                            print(f"[GPU {gpu_id}] Ignoring out-of-order AV event: "
                                  f"event_seg={event.segment_idx}, current_seg={segment_idx}, "
                                  f"kind={type(event).__name__}")
                            continue

                        match event:
                            case MediaInit(mime=mime, stream_id=stream_id):
                                av_streamed = True
                                await ws_send_json({
                                    "type": "media_init",
                                    "segment_idx": segment_idx,
                                    "mime": mime,
                                    "stream_id": stream_id,
                                    "mode": "av_fmp4",
                                })
                            case MediaChunk(
                                chunk=chunk_bytes,
                                chunk_offset=chunk_offset,
                                chunk_length=chunk_length,
                                uses_shared_buffer=uses_shared,
                            ):
                                if (uses_shared and chunk_offset is not None and chunk_length is not None
                                        and slot.shared_stream_buffer is not None):
                                    start = chunk_offset
                                    end = start + chunk_length
                                    # Copy out of shared buffer for websocket send.
                                    chunk = bytes(slot.shared_stream_buffer[start:end])
                                else:
                                    chunk = chunk_bytes or b""
                                if chunk:
                                    av_chunks_relayed += 1
                                    av_bytes_relayed += len(chunk)
                                    await ws_send_bytes(chunk)
                            case MediaComplete(stream_id=stream_id, chunks=n):
                                await ws_send_json({
                                    "type": "media_segment_complete",
                                    "segment_idx": segment_idx,
                                    "stream_id": stream_id,
                                    "chunks": n if n is not None else av_chunks_relayed,
                                })
                            case _:
                                print(f"[GPU {gpu_id}] Unknown AV event: "
                                      f"{type(event).__name__}")

                    if not step_task.done():
                        step_task.cancel()
                    else:
                        timings = await step_task
                finally:
                    segment_generation_active = False
                    if not step_task.done():
                        try:
                            await step_task
                        except asyncio.CancelledError:
                            pass

                if stop_event.is_set():
                    break

                t_generation = time.perf_counter() - t_start
                worker_e2e_ms = float(timings.get("e2e_latency_ms", 0.0) or 0.0)
                main_user_step_ms = t_generation * 1000.0
                overhead_vs_worker_ms = main_user_step_ms - worker_e2e_ms

                ipc_put_start_ns = timings.get("ipc_put_start_ns")
                ipc_get_done_ns = timings.get("ipc_get_done_ns")
                ipc_queue_transfer_ms = None
                if isinstance(ipc_put_start_ns, int) and isinstance(ipc_get_done_ns, int):
                    ipc_queue_transfer_ms = (ipc_get_done_ns - ipc_put_start_ns) / 1_000_000.0

                ipc_queue_str = (f"{ipc_queue_transfer_ms:.0f}ms" if ipc_queue_transfer_ms is not None else "n/a")
                print(f"[GPU {gpu_id}] Segment {segment_idx}: "
                      f"source={selected.source}, "
                      f"worker_e2e={worker_e2e_ms:.0f}ms, "
                      f"main_user_step={main_user_step_ms:.0f}ms, "
                      f"overhead={overhead_vs_worker_ms:.0f}ms, "
                      f"ipc_queue={ipc_queue_str}")

                if not av_streamed:
                    raise RuntimeError(f"Segment {segment_idx} AV stream did not initialize "
                                       "(no media_init event)")
                generated_segment_count += 1

                print(f"[GPU {gpu_id}] Segment {segment_idx}: "
                      f"relayed av chunks={av_chunks_relayed}, "
                      f"bytes={av_bytes_relayed / (1024 * 1024):.1f}MB")
                _main_print(
                    "INFO",
                    f"Segments generated for client {client_id[:8]}: "
                    f"{generated_segment_count}",
                )
                await log_event(
                    "segment_complete",
                    {
                        "segment_idx": segment_idx,
                        "latency_ms": {
                            "total": round(main_user_step_ms, 2),
                            "worker_e2e": round(worker_e2e_ms, 2),
                            "main_user_step": round(main_user_step_ms, 2),
                            "overhead": round(overhead_vs_worker_ms, 2),
                        },
                        "data_size_bytes": av_bytes_relayed,
                    },
                )
                await ws_send_json({
                    "type": "step_complete",
                    "latency_ms": {
                        "total": round(main_user_step_ms, 2),
                        "worker_e2e": round(worker_e2e_ms, 2),
                        "main_user_step": round(main_user_step_ms, 2),
                        "overhead": round(overhead_vs_worker_ms, 2),
                    },
                })
                await ws_send_json({
                    "type": "ltx2_segment_complete",
                    "segment_idx": segment_idx,
                    "total_segments": total_segments_hint,
                })
                if single_clip_mode and not single_clip_waiting_for_request:
                    single_clip_waiting_for_request = True
                    project_stream_started = False
                    await ws_send_json({"type": "ltx2_stream_complete"})
                    await log_event("ws_stream_complete")
                    _main_print(
                        "INFO",
                        f"Single-clip session idle for client {client_id[:8]} "
                        f"after segment {segment_idx}; waiting for next create",
                    )

            if not stop_event.is_set() and project_stream_started:
                project_stream_started = False
                await ws_send_json({"type": "ltx2_stream_complete"})
                await log_event("ws_stream_complete")
                print(f"[GPU {gpu_id}] LTX2 streaming complete for {client_id[:8]}")

        except WebSocketDisconnect:
            stop_event.set()
            _main_print("INFO", f"Client {client_id[:8]} disconnected")
        except Exception as exc:
            stop_event.set()
            _main_print("ERROR", f"Client {client_id[:8]} error: {exc}")
            try:
                await ws_send_json({
                    "type": "error",
                    "message": f"AV streaming failed: {exc}",
                })
            except Exception:
                pass
            import traceback
            _main_print("ERROR", f"Traceback: {traceback.format_exc()}")
        finally:
            stop_event.set()
            await cancel_task(websocket_reader_task)
            await cancel_task(prompt_worker_task)
            await cancel_task(rewrite_seed_prompts_task)

            if slot is not None and client_id:
                slot.unregister_stream_queue(client_id)

            if timeout_task and not timeout_task.done():
                timeout_task.cancel()
                try:
                    await timeout_task
                except asyncio.CancelledError:
                    pass

            try:
                if client_id and self.gpu_pool is not None:
                    await self.gpu_pool.release(client_id)
            finally:
                cleanup_session_init_image(session_init_image)
