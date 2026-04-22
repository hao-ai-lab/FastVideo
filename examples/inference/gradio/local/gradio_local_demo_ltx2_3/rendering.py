import html
from pathlib import Path

from .config import DEFAULT_FPS, GENERATED_CLIP_ROOT, MAX_SESSION_CLIPS

def create_timing_display(inference_time, total_time, stage_execution_times, num_frames):
    timing_html = f"""
    <div class="timing-shell">
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-bottom: 10px;">
            <div class="timing-card">
                <div style="font-size: 20px;">🎬</div>
                <div style="font-weight: bold; margin: 3px 0; font-size: 14px;">Video Generation Time</div>
                <div style="font-size: 18px; color: #2563eb;">{inference_time:.1f}s</div>
            </div>
            <div class="timing-card timing-card-highlight">
                <div style="font-size: 20px;">📊</div>
                <div style="font-weight: bold; margin: 3px 0; font-size: 14px;">E2E Latency</div>
                <div style="font-size: 18px; color: #0277bd;">{total_time:.1f}s</div>
            </div>
        </div>"""
    
    if inference_time > 0:
        fps = num_frames / inference_time
        timing_html += f"""
        <div class="performance-card" style="margin-top: 15px;">
            <span style="font-weight: bold;">Generation Speed: </span>
            <span style="font-size: 18px; color: #6366f1; font-weight: bold;">{fps:.1f} frames/second</span>
        </div>"""
    
    return timing_html + "</div>"

def create_timing_placeholder() -> str:
    return """
    <div class="timing-shell">
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-bottom: 10px;">
            <div class="timing-card">
                <div style="font-weight: bold; margin: 3px 0; font-size: 14px;">Video Generation Time</div>
                <div style="font-size: 18px; color: #4f8cff;">--</div>
            </div>
            <div class="timing-card timing-card-highlight">
                <div style="font-weight: bold; margin: 3px 0; font-size: 14px;">E2E Latency</div>
                <div style="font-size: 18px; color: #4f8cff;">--</div>
            </div>
        </div>
        <div class="performance-card">
            <span style="font-weight: bold;">Generation Speed: </span>
            <span style="font-size: 18px; color: #4f8cff; font-weight: bold;">--</span>
        </div>
    </div>
    """

def _truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "..."

def _clip_duration_seconds(num_frames: int, fps: int) -> int:
    return max(1, round(num_frames / max(fps, 1)))

def _make_clip_public_path(output_path: str) -> str:
    resolved_path = Path(output_path).resolve()
    relative_path = resolved_path.relative_to(GENERATED_CLIP_ROOT.resolve())
    return f"/generated-clips/{relative_path.as_posix()}"

def _record_session_clip(
    session_clips: list[dict[str, str | int | float]],
    *,
    output_path: str,
    prompt: str,
    model_name: str,
    num_frames: int,
    generation_time: float,
) -> list[dict[str, str | int | float]]:
    clip_entry = {
        "video_url": _make_clip_public_path(output_path),
        "prompt": prompt.strip(),
        "prompt_preview": _truncate_text(prompt.strip(), 120),
        "model_name": model_name,
        "duration_label": f"{_clip_duration_seconds(num_frames, DEFAULT_FPS)} Sec",
        "num_frames": num_frames,
        "generation_time": generation_time,
    }
    updated_clips = list(session_clips)
    updated_clips.append(clip_entry)
    if len(updated_clips) > MAX_SESSION_CLIPS:
        updated_clips = updated_clips[-MAX_SESSION_CLIPS:]
    return updated_clips

def render_completed_clips(clips: list[dict[str, str | int | float]]) -> str:
    if not clips:
        return """
        <div class="completed-clips-empty">
            <div class="completed-clips-empty-title">Nothing in gallery yet</div>
            <div class="completed-clips-empty-copy">
                Build your personal gallery for this browser session by creating videos.
            </div>
        </div>
        """

    cards: list[str] = []
    for clip in reversed(clips):
        video_url = html.escape(str(clip["video_url"]), quote=True)
        prompt = str(clip["prompt"])
        prompt_preview = html.escape(str(clip["prompt_preview"]))
        model_name = html.escape(str(clip["model_name"]))
        duration_label = html.escape(str(clip["duration_label"]))
        cards.append(
            f"""
            <article class="completed-clip-card">
                <div class="completed-clip-video-shell">
                    <video class="completed-clip-video" src="{video_url}" controls preload="metadata" playsinline></video>
                </div>
                <div class="completed-clip-body">
                    <div class="completed-clip-title" title="{html.escape(prompt, quote=True)}">{prompt_preview}</div>
                    <div class="completed-clip-meta">
                        <span class="completed-clip-badge">{model_name}</span>
                        <span class="completed-clip-badge completed-clip-duration">{duration_label}</span>
                    </div>
                    <details class="completed-clip-prompt">
                        <summary>Prompt</summary>
                        <div>{html.escape(prompt)}</div>
                    </details>
                </div>
            </article>
            """
        )

    return f"""
    <div class="completed-clips-grid">
        {''.join(cards)}
    </div>
    """

def render_error_message(message: str) -> str:
    return f"""
    <div class="stage-error-card">
        <div class="stage-error-title">Error</div>
        <div class="stage-error-copy">{html.escape(message)}</div>
    </div>
    """

def render_prompt_blocked_message(
    message: str,
    category: str | None = None,
) -> str:
    details = ""
    if category:
        details = (
            '<div class="stage-error-copy">'
            f"Policy: {html.escape(category)}"
            "</div>"
        )
    return f"""
    <div class="stage-error-card">
        <div class="stage-error-title">Prompt Blocked</div>
        {details}
        <div class="stage-error-copy">{html.escape(message)}</div>
    </div>
    """

def render_input_image_status(input_image: str | None) -> str:
    if not input_image:
        return ""

    image_name = html.escape(Path(str(input_image)).name)
    return (
        "<div class='image-upload-status'>"
        f"Image ready: {image_name}"
        "</div>"
    )
