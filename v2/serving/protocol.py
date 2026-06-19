"""OpenAI-shaped request schemas (vllm-omni protocol, framework-free).

Plain dataclasses parsed from JSON (no pydantic dep). Each translates to a typed ``OmniRequest`` at
the server boundary — the request is the only currency crossing the product boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from v2.request import DiffusionParams, OutputSpec, SamplingParams, TaskType, make_request


def _wh(size: str, default=(832, 480)) -> tuple[int, int]:
    try:
        w, h = size.lower().split("x")
        return int(w), int(h)
    except Exception:
        return default


def _task_for(model_id: str, kind: str) -> TaskType:
    m = model_id.lower()
    if kind == "image":
        return TaskType.T2I if ("bagel" in m or "image" in m) else TaskType.T2V
    if kind == "video":
        return TaskType.T2V
    if kind == "chat":
        if "bagel" in m:
            return TaskType.T2I
        if "cosmos" in m or "omni" in m:
            return TaskType.T2V
        return TaskType.T2V
    return TaskType.T2V


@dataclass
class ImageGenerationRequest:
    prompt: str
    model: str
    n: int = 1
    size: str = "832x480"
    num_inference_steps: int = 4
    guidance_scale: float = 5.0
    seed: int = 0

    @classmethod
    def from_json(cls, d: dict) -> ImageGenerationRequest:
        return cls(prompt=d.get("prompt", ""),
                   model=d.get("model", ""),
                   n=int(d.get("n", 1)),
                   size=d.get("size", "832x480"),
                   num_inference_steps=int(d.get("num_inference_steps", 4)),
                   guidance_scale=float(d.get("guidance_scale", 5.0)),
                   seed=int(d.get("seed", 0)))

    def to_omni(self):
        w, h = _wh(self.size)
        return make_request(_task_for(self.model, "image"),
                            self.model,
                            self.prompt,
                            diffusion=DiffusionParams(num_steps=self.num_inference_steps,
                                                      guidance_scale=self.guidance_scale,
                                                      height=h,
                                                      width=w,
                                                      seed=self.seed))


@dataclass
class VideoGenerationRequest:
    prompt: str
    model: str
    seconds: float = 2.0
    size: str = "832x480"
    num_frames: int = 81
    num_inference_steps: int = 4
    guidance_scale: float = 5.0
    seed: int = 0
    stream: dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, d: dict) -> VideoGenerationRequest:
        return cls(prompt=d.get("prompt", ""),
                   model=d.get("model", ""),
                   seconds=float(d.get("seconds", 2.0)),
                   size=d.get("size", "832x480"),
                   num_frames=int(d.get("num_frames", 81)),
                   num_inference_steps=int(d.get("num_inference_steps", 4)),
                   guidance_scale=float(d.get("guidance_scale", 5.0)),
                   seed=int(d.get("seed", 0)),
                   stream={"video": True} if d.get("stream") else {})

    def to_omni(self):
        w, h = _wh(self.size)
        return make_request(TaskType.T2V,
                            self.model,
                            self.prompt,
                            diffusion=DiffusionParams(num_steps=self.num_inference_steps,
                                                      guidance_scale=self.guidance_scale,
                                                      height=h,
                                                      width=w,
                                                      num_frames=self.num_frames,
                                                      seed=self.seed),
                            outputs=OutputSpec(modalities=frozenset({"video"}), stream=self.stream))


@dataclass
class ChatCompletionRequest:
    model: str
    messages: list[dict]
    stream: bool = False
    max_tokens: int = 6
    num_inference_steps: int = 4
    seed: int = 0

    @classmethod
    def from_json(cls, d: dict) -> ChatCompletionRequest:
        return cls(model=d.get("model", ""),
                   messages=d.get("messages", []),
                   stream=bool(d.get("stream", False)),
                   max_tokens=int(d.get("max_tokens", 6)),
                   num_inference_steps=int(d.get("num_inference_steps", 4)),
                   seed=int(d.get("seed", 0)))

    def prompt(self) -> str:
        for m in reversed(self.messages):
            if m.get("role") == "user":
                c = m.get("content", "")
                return c if isinstance(c, str) else " ".join(p.get("text", "") for p in c if isinstance(p, dict))
        return ""

    def to_omni(self):
        return make_request(_task_for(self.model, "chat"),
                            self.model,
                            self.prompt(),
                            sampling=SamplingParams(max_tokens=self.max_tokens, seed=self.seed),
                            diffusion=DiffusionParams(num_steps=self.num_inference_steps, seed=self.seed),
                            outputs=OutputSpec(stream={"video": True} if self.stream else {}))
