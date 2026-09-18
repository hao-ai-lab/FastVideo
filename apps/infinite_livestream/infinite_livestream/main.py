"""Infinite Livestream: one process, from prompt to playlist.

  chat ──▶ Director ──▶ PromptUpsampler (any OpenAI-compatible LLM)
              │
              ▼ enqueue / move / pop
            Engine ──▶ FastH3Backend ──▶ FastVideo (4 GPUs)
              │ frames + audio
              ▼
            Pacer ──▶ HlsSink ──▶ the page's <video>
              ▲
            webapp: serves the page, the playlist, and the chat box

The pacer and the sink start before the model, so a viewer arriving during the
~3 minute load sees the page and a live black stream rather than a refused
connection.

Usage:
    export OPENAI_API_KEY=... LIVESTREAM_WEIGHTS_PATH=/path/to/fasth3
    infinite-livestream-server                       # configs/infinite_livestream.yaml
    infinite-livestream-server --config my.yaml      # or a copy of it
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import warnings

from .chat import WebChat
from .config import Config, load_model_config
from .director import Director
from .engine import MODEL_FPS, MODEL_SAMPLE_RATE, Engine
from .moderator import Moderator
from .pacer import Pacer
from .sink import AudioFormat, HlsSink, VideoFormat
from .upsampler import PromptUpsampler
from .webapp import DemoWeb

logger = logging.getLogger("infinite_livestream")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    warnings.filterwarnings("ignore", category=DeprecationWarning)


async def serve(config: Config) -> None:
    """Build every component, wire them together, and run until one dies."""
    model_config = load_model_config(config.config_path)

    # Constructing the engine checks the weights without loading the model, so
    # a missing component fails in milliseconds rather than after minutes of
    # GPU work.
    engine = Engine(config, model_config)

    upsampler = PromptUpsampler(
        api_key=config.openai_api_key,
        model=config.openai_model,
        style=config.style,
        free_viewer_style=config.viewer_free_style,
        max_chunks=config.max_chunks,
        base_url=config.openai_base_url,
    )
    moderator = Moderator(
        api_key=config.moderation_api_key,
        model=config.moderation_model,
        enabled=config.moderation_enabled,
        base_url=config.moderation_base_url,
    )
    if not moderator.enabled:
        logger.warning("moderation is DISABLED — every chat prompt reaches the upsampler unchecked")
    # Viewers type into the same page they watch on, so the chat source and the
    # web app are two halves of one thing -- and the web app is built before the
    # director, because the director has to be able to tell a viewer that their
    # prompt was dropped.
    chat = WebChat(config.chat_command)
    sink = HlsSink(config.hls_dir, video_bitrate_k=config.video_bitrate_k)
    # The page positions the now-playing title by comparing the playlist's
    # PROGRAM-DATE-TIME against the timeline the web app keeps, so the two have
    # to be stamped by the same clock: the sink's.
    web = DemoWeb(chat,
                  config.hls_dir,
                  host=config.web_host,
                  port=config.web_port,
                  stream_clock=sink.stream_time,
                  live_edge_clock=sink.published_until)
    engine.add_listener(web.listener)

    def announce_reject(author: str, reason: str) -> None:
        """Put a dropped prompt back in front of the viewer who sent it."""
        web.state.note("error", f"not queued -- {reason}", author=author)
        web.broadcast()

    director = Director(
        engine,
        upsampler,
        moderator,
        cooldown_s=config.chat_cooldown_s,
        idle_prompts=config.idle_prompts,
        idle_queue_target=config.idle_queue_target,
        on_reject=announce_reject,
    )
    web.cooldown_remaining = director.cooldown_remaining
    # The canvas is this deployment's own config rather than something
    # negotiated with a remote, so the pacer can start immediately.
    width, height = engine.canvas
    pacer = Pacer(sink, VideoFormat(width=width, height=height, fps=MODEL_FPS),
                  AudioFormat(sample_rate=MODEL_SAMPLE_RATE, channels=1))
    engine.attach_pacer(pacer)

    tasks = [
        asyncio.create_task(pacer.run(), name="pacer"),
        asyncio.create_task(engine.run(), name="engine"),
        asyncio.create_task(director.run(), name="director"),
        asyncio.create_task(director.run_playout(), name="playout"),
        asyncio.create_task(chat.run(director.submit), name="chat"),
        asyncio.create_task(web.run(), name="webapp"),
    ]
    # Gated because any finished task is a shutdown signal and run_idle returns
    # immediately at target 0. A preset with no idle prompts still gets the
    # task: the filler idles until a `!switch` brings prompts.
    if config.idle_queue_target > 0:
        tasks.append(asyncio.create_task(director.run_idle(), name="idle-filler"))
    else:
        logger.info("idle filler off (IDLE_QUEUE_TARGET=0)")

    logger.info("streaming %dx%d@%dfps, %d idle prompts, chat command %r, page on http://%s:%d", width, height,
                MODEL_FPS, len(config.idle_prompts), config.chat_command, config.web_host, config.web_port)
    try:
        done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if task.cancelled():
                continue
            error = task.exception()
            if error is not None:
                logger.error("task %s died: %s", task.get_name(), error)
                raise error
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await sink.stop()
        logger.info("services stopped")


def cli() -> None:
    setup_logging()
    config = Config.load()
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(serve(config))


if __name__ == "__main__":
    cli()
