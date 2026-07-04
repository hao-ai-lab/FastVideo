"""OpenAI-shaped request schemas (vllm-omni protocol, framework-free).

Plain dataclasses parsed from JSON (no pydantic dep). Each translates to a typed ``OmniRequest`` at
the server boundary — the request is the only currency crossing the product boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from v2.core.request import DiffusionParams, ImagePart, OutputSpec, SamplingParams, TaskType, TextPart, make_request


def _wh(size: str, default=(832, 480)) -> tuple[int, int]:
    try:
        w, h = size.lower().split("x")
        return int(w), int(h)
    except Exception:
        return default


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

    def to_omni(self, task: TaskType = TaskType.T2I):
        w, h = _wh(self.size)
        return make_request(task,
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

    def to_omni(self, task: TaskType = TaskType.T2V):
        w, h = _wh(self.size)
        return make_request(task,
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
                return c if isinstance(c, str) else " ".join(
                    p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
        return ""

    def image_parts(self) -> list[ImagePart]:
        parts: list[ImagePart] = []
        for m in self.messages:
            content = m.get("content", "")
            if not isinstance(content, list):
                continue
            for p in content:
                if not isinstance(p, dict):
                    continue
                typ = p.get("type")
                if typ not in ("image", "image_url", "input_image"):
                    continue
                url = p.get("url") or p.get("image_url") or p.get("image")
                if isinstance(url, dict):
                    url = url.get("url")
                parts.append(ImagePart(path=url if isinstance(url, str) else None))
        return parts

    def has_image(self) -> bool:
        return bool(self.image_parts())

    def to_omni(self, task: TaskType = TaskType.REASON):
        stream = {}
        if self.stream:
            stream = {"audio": True} if task == TaskType.T2A else {"video": True}
        inputs = (TextPart(self.prompt()), *self.image_parts())
        return make_request(task,
                            self.model,
                            inputs=inputs,
                            sampling=SamplingParams(max_tokens=self.max_tokens, seed=self.seed),
                            diffusion=DiffusionParams(num_steps=self.num_inference_steps, seed=self.seed),
                            outputs=OutputSpec(stream=stream))
