# SPDX-License-Identifier: Apache-2.0
"""Typed omni request plane (design.md §6.1).

This module introduces the request-plane vocabulary the next-generation
runtime is built on:

- :class:`OmniRequest` — a typed multimodal request whose ``task`` is
  *declared, never inferred*, and whose inputs are typed
  :class:`ModalPart` s instead of per-model fields on a god-object.
- :class:`OmniOutput` — a typed multimodal output whose modalities are
  named :class:`Artifact` slots carrying provenance, replacing the
  ``extra["audio"]`` escape hatch (design.md P3).
- :data:`OmniEvent` — one streaming-event union (progress / chunk /
  final), the single channel that ``LoopStage.step`` emits through.

It is deliberately additive and engine-agnostic: nothing here imports
``torch`` or the pipeline machinery, and adapters bridge to/from today's
:class:`~fastvideo.api.schema.GenerationRequest` /
:class:`~fastvideo.api.results.GenerationResult` so the new types are
usable through the existing ``VideoGenerator`` before the engine itself
is rebuilt (plan.md M1). Later milestones evolve ``GenerationRequest``
into ``OmniRequest`` in place and drop the adapters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar
from uuid import uuid4

from fastvideo.api.results import (
    GenerationResult,
    VideoEvent,
    VideoFinalEvent,
    VideoPartialEvent,
    VideoProgressEvent,
)
from fastvideo.api.schema import (
    GenerationRequest,
    InputConfig,
    OutputConfig,
    RequestRuntimeConfig,
    SamplingConfig,
)


class Modality(str, Enum):
    """A media modality. ``str`` mixin keeps it JSON/serialization friendly."""

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    ACTION = "action"
    LATENT = "latent"


class TaskType(str, Enum):
    """The declared task (design.md §6.1 / kills P7).

    The pipeline graph branches on ``request.task``. Heuristics may only
    *suggest* a default at the API boundary (see :func:`infer_task`); they
    never decide control flow inside the runtime.
    """

    T2V = "t2v"  # text -> video
    I2V = "i2v"  # image (+text) -> video
    TI2V = "ti2v"  # text + init image -> video
    V2V = "v2v"  # video -> video (edit / restyle)
    V2W = "v2w"  # video -> world (continue a world-model rollout)
    T2I = "t2i"  # text -> image
    I2I = "i2i"  # image -> image (edit)
    T2A = "t2a"  # text -> audio
    T2VS = "t2vs"  # text -> video + sound (joint A/V)
    A2W = "a2w"  # action -> world (interactive world model)
    REASON = "reason"  # text -> text (AR reasoner / prompt upsampling)


# ---------------------------------------------------------------------------
# Inputs: typed modality parts (replaces InputConfig's per-model fields, P3).
# ---------------------------------------------------------------------------


@dataclass
class ModalPart:
    """Base for a typed input part.

    ``modality`` is intrinsic to the concrete subclass (a ``ClassVar``, not
    an instance field). ``role`` disambiguates several parts of one
    modality — e.g. ``"prompt"`` vs ``"negative"`` text, ``"init"`` vs
    ``"conditioning"`` image — and is keyword-only so subclass payloads stay
    positional.
    """

    modality: ClassVar[Modality]
    role: str | None = field(default=None, kw_only=True)


@dataclass
class TextPart(ModalPart):
    modality: ClassVar[Modality] = Modality.TEXT
    text: str = ""


@dataclass
class ImagePart(ModalPart):
    modality: ClassVar[Modality] = Modality.IMAGE
    image: Any | None = None  # PIL.Image / ndarray / tensor
    path: str | None = None


@dataclass
class VideoPart(ModalPart):
    modality: ClassVar[Modality] = Modality.VIDEO
    video: Any | None = None
    path: str | None = None
    fps: float | None = None


@dataclass
class AudioPart(ModalPart):
    modality: ClassVar[Modality] = Modality.AUDIO
    audio: Any | None = None
    path: str | None = None
    sample_rate: int | None = None


@dataclass
class ActionPart(ModalPart):
    modality: ClassVar[Modality] = Modality.ACTION
    action: Any | None = None  # mouse / keyboard / camera tensors
    kind: str | None = None  # "mouse" | "keyboard" | "camera" | ...


@dataclass
class LatentPart(ModalPart):
    modality: ClassVar[Modality] = Modality.LATENT
    latents: Any | None = None
    of_modality: Modality = Modality.VIDEO  # modality these latents decode to


# ---------------------------------------------------------------------------
# Per-call parameters: AR sampling vs diffusion, separated (design.md §6.1).
# ---------------------------------------------------------------------------


@dataclass
class SamplingParams:
    """AR decode knobs.

    Distinct from the legacy diffusion ``fastvideo.api.SamplingParam``: these
    drive ``ARDecodeLoop`` (Cosmos3 reasoner, omni thinkers/talkers, codec
    decode), not denoising.
    """

    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    stop: list[str] = field(default_factory=list)
    seed: int | None = None


@dataclass
class DiffusionParams:
    """Denoise knobs. ``guidance_per_modality`` carries per-modality CFG
    scales for joint A/V denoise (LTX-2, Cosmos3 t2vs); ``guidance_scale`` is
    the scalar default."""

    steps: int = 50
    guidance_scale: float = 1.0
    guidance_per_modality: dict[Modality, float] = field(default_factory=dict)
    sigmas: list[float] | None = None
    flow_shift: float | None = None
    height: int | None = None
    width: int | None = None
    num_frames: int | None = None
    fps: int | None = None
    seed: int | None = None


# ---------------------------------------------------------------------------
# Outputs spec: requested modalities + streaming + capture flags.
# ---------------------------------------------------------------------------


@dataclass
class StreamSpec:
    """Per-modality streaming policy. ``chunk_ms`` applies to audio chunks,
    ``per_chunk`` to chunked-causal video; ``enabled`` alone covers token
    text."""

    enabled: bool = True
    chunk_ms: int | None = None
    per_chunk: bool = False


@dataclass
class OutputSpec:
    """Requested output modalities, streaming, and capture flags."""

    modalities: list[Modality] = field(default_factory=lambda: [Modality.VIDEO])
    stream: dict[Modality, StreamSpec] = field(default_factory=dict)
    return_latents: bool = False
    return_trajectory: bool = False

    @property
    def streaming(self) -> bool:
        """True if any modality is requested as a stream."""
        return any(spec.enabled for spec in self.stream.values())


@dataclass
class NodeOverrides:
    """Per-graph-node parameter overrides (design.md §6.1).

    For a multi-loop graph, ``node_params["refine"].steps`` overrides the
    refine loop's step count without leaking ``refine_*`` onto the universal
    schema. Validation against each node's declared schema arrives with
    ``PipelineSpec`` (a later milestone); for now this is a typed bag with
    ``get`` / item / attribute access.
    """

    params: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.params[key]

    def __getattr__(self, key: str) -> Any:
        # Only invoked when normal attribute lookup fails, so ``params``
        # itself resolves through ``__dict__`` and never recurses.
        try:
            return self.__dict__["params"][key]
        except KeyError as exc:
            raise AttributeError(key) from exc


@dataclass
class OmniRequest:
    """A typed multimodal request (design.md §6.1)."""

    task: TaskType
    inputs: list[ModalPart] = field(default_factory=list)
    sampling: SamplingParams = field(default_factory=SamplingParams)
    diffusion: DiffusionParams = field(default_factory=DiffusionParams)
    outputs: OutputSpec = field(default_factory=OutputSpec)
    node_params: dict[str, NodeOverrides] = field(default_factory=dict)
    priority: int = 0
    request_id: str = field(default_factory=lambda: uuid4().hex)

    # -- accessors ----------------------------------------------------------

    def parts(self, modality: Modality) -> list[ModalPart]:
        return [p for p in self.inputs if p.modality is modality]

    @property
    def prompt(self) -> str | None:
        for part in self.inputs:
            if isinstance(part, TextPart) and part.role in (None, "prompt"):
                return part.text
        return None

    @property
    def negative_prompt(self) -> str | None:
        for part in self.inputs:
            if isinstance(part, TextPart) and part.role in ("negative", "negative_prompt"):
                return part.text
        return None

    # -- constructors / adapters -------------------------------------------

    @classmethod
    def from_prompt(
        cls,
        prompt: str | None,
        task: TaskType,
        *,
        negative_prompt: str | None = None,
        **kwargs: Any,
    ) -> OmniRequest:
        """Build a request from a bare prompt — what ``generate_video`` calls
        internally so the offline shim constructs an ``OmniRequest`` (G5)."""
        inputs: list[ModalPart] = []
        if prompt is not None:
            inputs.append(TextPart(prompt))
        if negative_prompt is not None:
            inputs.append(TextPart(negative_prompt, role="negative"))
        return cls(task=task, inputs=inputs, **kwargs)

    @classmethod
    def from_generation_request(
        cls,
        request: GenerationRequest,
        task: TaskType | None = None,
    ) -> OmniRequest:
        """Lift a legacy typed :class:`GenerationRequest` into an
        ``OmniRequest``; ``task`` defaults to the boundary heuristic."""
        resolved = task if task is not None else infer_task(request)
        inputs: list[ModalPart] = []
        prompt = request.prompt
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else None
        if prompt is not None:
            inputs.append(TextPart(prompt))
        if request.negative_prompt is not None:
            inputs.append(TextPart(request.negative_prompt, role="negative"))

        inp = request.inputs
        image_path = inp.image_path if isinstance(inp.image_path, str) else None
        if image_path is not None or inp.pil_image is not None:
            inputs.append(ImagePart(image=inp.pil_image, path=image_path))
        video_path = inp.video_path if isinstance(inp.video_path, str) else None
        if video_path is not None:
            inputs.append(VideoPart(path=video_path))
        if inp.mouse_cond is not None or inp.keyboard_cond is not None:
            inputs.append(ActionPart(action=inp.mouse_cond, kind="mouse"))

        sampling = request.sampling
        diffusion = DiffusionParams(
            steps=sampling.num_inference_steps,
            guidance_scale=sampling.guidance_scale,
            sigmas=sampling.sigmas,
            height=sampling.height,
            width=sampling.width,
            num_frames=sampling.num_frames,
            fps=sampling.fps,
            seed=sampling.seed,
        )
        outputs = OutputSpec(
            return_trajectory=request.runtime.return_trajectory_latents,
            return_latents=request.runtime.return_trajectory_decoded,
        )
        node_params = {
            node: NodeOverrides(params=dict(overrides))
            for node, overrides in request.stage_overrides.items() if isinstance(overrides, dict)
        }
        return cls(
            task=resolved,
            inputs=inputs,
            diffusion=diffusion,
            outputs=outputs,
            node_params=node_params,
        )

    def to_generation_request(self) -> GenerationRequest:
        """Lower to a legacy :class:`GenerationRequest` so today's
        ``VideoGenerator`` can execute an ``OmniRequest`` unchanged (M1)."""
        diff = self.diffusion
        sampling = SamplingConfig()
        sampling.num_inference_steps = diff.steps
        sampling.guidance_scale = diff.guidance_scale
        sampling.sigmas = diff.sigmas
        if diff.seed is not None:
            sampling.seed = diff.seed
        if diff.num_frames is not None:
            sampling.num_frames = diff.num_frames
        if diff.height is not None:
            sampling.height = diff.height
        if diff.width is not None:
            sampling.width = diff.width
        if diff.fps is not None:
            sampling.fps = diff.fps

        image_path: str | None = None
        pil_image: Any | None = None
        video_path: str | None = None
        for part in self.inputs:
            if isinstance(part, ImagePart):
                image_path = image_path or part.path
                pil_image = pil_image if pil_image is not None else part.image
            elif isinstance(part, VideoPart):
                video_path = video_path or part.path
        inputs = InputConfig(image_path=image_path, pil_image=pil_image, video_path=video_path)

        runtime = RequestRuntimeConfig(
            return_trajectory_latents=self.outputs.return_trajectory,
            return_trajectory_decoded=self.outputs.return_latents,
        )
        output = OutputConfig(return_frames=Modality.VIDEO in self.outputs.modalities)
        stage_overrides = {node: dict(ov.params) for node, ov in self.node_params.items()}
        return GenerationRequest(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            inputs=inputs,
            sampling=sampling,
            runtime=runtime,
            output=output,
            stage_overrides=stage_overrides,
        )


# ---------------------------------------------------------------------------
# Outputs: named artifacts with provenance (kills extra["audio"], P3).
# ---------------------------------------------------------------------------


@dataclass
class Artifact:
    """Base output artifact. ``source_node`` records which graph node
    produced it (provenance, design.md §6.1)."""

    modality: ClassVar[Modality]
    source_node: str | None = field(default=None, kw_only=True)


@dataclass
class VideoArtifact(Artifact):
    modality: ClassVar[Modality] = Modality.VIDEO
    frames: Any | None = None  # numpy (N, H, W, 3) uint8
    tensor: Any | None = None  # raw sample tensor
    path: str | None = None
    fps: float | None = None


@dataclass
class AudioArtifact(Artifact):
    modality: ClassVar[Modality] = Modality.AUDIO
    audio: Any | None = None
    sample_rate: int | None = None


@dataclass
class TextArtifact(Artifact):
    modality: ClassVar[Modality] = Modality.TEXT
    text: str = ""
    token_ids: list[int] | None = None


@dataclass
class TensorArtifact(Artifact):
    """Action tensors and other raw tensor outputs."""

    modality: ClassVar[Modality] = Modality.ACTION
    tensor: Any | None = None


@dataclass
class LatentArtifact(Artifact):
    modality: ClassVar[Modality] = Modality.LATENT
    latents: Any | None = None
    timesteps: Any | None = None
    of_modality: Modality = Modality.VIDEO


@dataclass
class RequestMetrics:
    generation_time: float | None = None
    peak_memory_mb: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class OmniOutput:
    """Typed multimodal output (design.md §6.1)."""

    request_id: str
    artifacts: dict[str, Artifact] = field(default_factory=dict)
    metrics: RequestMetrics = field(default_factory=RequestMetrics)

    def get(self, name: str) -> Artifact | None:
        return self.artifacts.get(name)

    @property
    def video(self) -> VideoArtifact | None:
        art = self.artifacts.get("video")
        return art if isinstance(art, VideoArtifact) else None

    @property
    def audio(self) -> AudioArtifact | None:
        art = self.artifacts.get("audio")
        return art if isinstance(art, AudioArtifact) else None

    @classmethod
    def from_generation_result(
        cls,
        result: GenerationResult,
        request_id: str = "",
    ) -> OmniOutput:
        """Map a legacy :class:`GenerationResult` into named artifacts.

        Audio becomes a first-class :class:`AudioArtifact` carrying its
        sample rate, instead of riding in ``extra["audio"]`` (P3).
        """
        artifacts: dict[str, Artifact] = {}
        if result.frames is not None or result.samples is not None or result.video_path is not None:
            artifacts["video"] = VideoArtifact(
                frames=result.frames,
                tensor=result.samples,
                path=result.video_path,
                source_node="decode",
            )
        if result.audio is not None:
            artifacts["audio"] = AudioArtifact(
                audio=result.audio,
                sample_rate=result.audio_sample_rate,
                source_node="audio_decode",
            )
        if result.trajectory is not None or result.trajectory_decoded is not None:
            artifacts["latents"] = LatentArtifact(
                latents=result.trajectory,
                timesteps=result.trajectory_timesteps,
                source_node="denoise",
            )
        return cls(
            request_id=request_id,
            artifacts=artifacts,
            metrics=RequestMetrics(
                generation_time=result.generation_time,
                peak_memory_mb=result.peak_memory_mb,
            ),
        )


# ---------------------------------------------------------------------------
# Streaming: one event union (evolves api/results.py's Video*Event).
# ---------------------------------------------------------------------------


@dataclass
class OmniProgressEvent:
    """Per-step progress telemetry."""

    step: int
    total_steps: int
    node: str = "denoise"


@dataclass
class OmniChunkEvent:
    """A streamed artifact chunk — the universal ``StepResult.emit`` channel
    (design.md §6.2.2): text tokens, audio chunks, or decoded frame chunks.

    ``pts`` optionally carries a presentation timestamp for raw-frame
    streaming over WebRTC (design.md §9.1)."""

    modality: Modality
    index: int
    payload: Any = None
    pts: float | None = None
    node: str | None = None


@dataclass
class OmniFinalEvent:
    """Terminal event carrying the full :class:`OmniOutput`."""

    output: OmniOutput


OmniEvent = OmniProgressEvent | OmniChunkEvent | OmniFinalEvent
"""Union of every event the engine streams; consumers match by ``isinstance``."""


def infer_task(request: GenerationRequest) -> TaskType:
    """Best-effort boundary heuristic for a legacy request (design.md §6.1).

    A *suggestion* only — the runtime always branches on the explicit
    ``OmniRequest.task``, never on this. Single-frame requests are images;
    a video input implies edit; an image input implies image-to-video.
    """
    inp = request.inputs
    has_image = bool(inp.image_path or inp.pil_image)
    has_video = bool(inp.video_path or inp.stage1_video)
    if request.sampling.num_frames == 1:
        return TaskType.I2I if has_image else TaskType.T2I
    if has_video:
        return TaskType.V2V
    if has_image:
        return TaskType.I2V
    return TaskType.T2V


def omni_event_from_video_event(event: VideoEvent, request_id: str = "") -> OmniEvent:
    """Adapt a legacy :data:`~fastvideo.api.results.VideoEvent` to an
    :data:`OmniEvent` so the streaming surface can migrate incrementally."""
    if isinstance(event, VideoProgressEvent):
        return OmniProgressEvent(step=event.step, total_steps=event.total_steps, node=event.stage)
    if isinstance(event, VideoPartialEvent):
        return OmniChunkEvent(modality=Modality.VIDEO, index=event.index, payload=event.frames, node="decode")
    if isinstance(event, VideoFinalEvent):
        if event.result is not None:
            output = OmniOutput.from_generation_result(event.result, request_id)
        else:
            output = OmniOutput(request_id=request_id)
            if event.frames is not None:
                output.artifacts["video"] = VideoArtifact(frames=event.frames, source_node="decode")
        return OmniFinalEvent(output=output)
    raise TypeError(f"unknown VideoEvent type: {type(event).__name__}")


__all__ = [
    "ActionPart",
    "Artifact",
    "AudioArtifact",
    "AudioPart",
    "DiffusionParams",
    "ImagePart",
    "LatentArtifact",
    "LatentPart",
    "Modality",
    "ModalPart",
    "NodeOverrides",
    "OmniChunkEvent",
    "OmniEvent",
    "OmniFinalEvent",
    "OmniOutput",
    "OmniProgressEvent",
    "OmniRequest",
    "OutputSpec",
    "RequestMetrics",
    "SamplingParams",
    "StreamSpec",
    "TaskType",
    "TensorArtifact",
    "TextArtifact",
    "TextPart",
    "VideoArtifact",
    "VideoPart",
    "infer_task",
    "omni_event_from_video_event",
]
