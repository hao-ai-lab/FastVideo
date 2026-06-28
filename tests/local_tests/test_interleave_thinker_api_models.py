import base64
import io
import sys
import types

from PIL import Image
import pytest

from fastvideo.workflow.interleave_thinker.generator import (
    NanoBananaImageGeneratorBackend,
)
from fastvideo.workflow.interleave_thinker.schema import InterleaveEditRequest
from fastvideo.train.methods.rl.rewards.interleave_api import (
    GeminiInterleaveImageScorer,
    GeminiNanoBananaEditScorer,
)
from fastvideo.train.methods.rl.rewards.interleave_thinker import (
    InterleaveThinkerEditRequest,
)
from fastvideo.train.models.interleave_thinker.data import IMAGE_EXTENSIONS


def _png_base64(color="red"):
    image = Image.new("RGB", (8, 8), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _install_fake_google_genai(monkeypatch):
    captured = {
        "calls": [],
        "configs": [],
        "parts": [],
    }

    class FakeImagePart:

        def __init__(self, image):
            self._image = image

        def as_image(self):
            return self._image

    class FakePartFactory:

        @staticmethod
        def from_bytes(data, mime_type):
            captured["parts"].append((data, mime_type))
            return {
                "data": data,
                "mime_type": mime_type,
            }

    class FakeImageConfig:

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeGenerateContentConfig:

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            captured["configs"].append(kwargs)

    class FakeTypes:
        GenerateContentConfig = FakeGenerateContentConfig
        ImageConfig = FakeImageConfig
        Part = FakePartFactory

    class FakeModels:

        def generate_content(self, **kwargs):
            captured["calls"].append(kwargs)
            model = kwargs["model"]
            if "image" in model:
                return types.SimpleNamespace(parts=[FakeImagePart(Image.new("RGB", (8, 8), "blue"))])
            return types.SimpleNamespace(text='{"semantic_score": 8.0, "quality_score": 7.0}')

    class FakeClient:

        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.models = FakeModels()

    google_mod = types.ModuleType("google")
    genai_mod = types.ModuleType("google.genai")
    genai_mod.Client = FakeClient
    genai_mod.types = FakeTypes
    google_mod.genai = genai_mod

    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)
    return captured


def test_nano_banana_backend_wraps_google_genai_image_api(monkeypatch, tmp_path):
    captured = _install_fake_google_genai(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    backend = NanoBananaImageGeneratorBackend(
        model="nano-banana-2",
        output_dir=str(tmp_path),
        aspect_ratio="1:1",
        image_size="1K",
    )

    result = backend.generate(
        InterleaveEditRequest(
            prompt="make a blue square",
            image=_png_base64("red"),
        ),
        request_id="abc",
    )

    assert result.file_path is not None
    assert Image.open(result.file_path).getpixel((0, 0)) == (0, 0, 255)
    assert captured["client_kwargs"]["api_key"] == "test-key"
    assert captured["calls"][0]["model"] == "gemini-3.1-flash-image"
    assert captured["configs"][0]["image_config"].kwargs == {
        "aspect_ratio": "1:1",
        "image_size": "1K",
    }


def test_gemini_nano_banana_edit_scorer_generates_and_scores(monkeypatch, tmp_path):
    captured = _install_fake_google_genai(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    source = tmp_path / "source.png"
    Image.new("RGB", (8, 8), "white").save(source)
    scorer = GeminiNanoBananaEditScorer(
        image_model="nano-banana",
        judge_model="gemini-2.5-pro",
        output_dir=str(tmp_path / "reward"),
        max_attempts=1,
    )

    score = scorer(
        InterleaveThinkerEditRequest(
            index=0,
            origin_prompt="draw a blue square",
            previous_prompt="blue square",
            refine_prompt="make it a clean blue square",
            origin_image_path=str(source),
            previous_image_path=str(source),
            previous_step_success=False,
            previous_semantic_score=4.0,
            previous_quality_score=5.0,
        ))

    assert score is not None
    assert score.semantic_score == pytest.approx(8.0)
    assert score.quality_score == pytest.approx(7.0)
    assert captured["calls"][0]["model"] == "gemini-2.5-flash-image"
    assert captured["calls"][1]["model"] == "gemini-2.5-pro"
    assert len(captured["parts"]) == 2


def test_gemini_image_scorer_uses_dataset_supported_mime_types(monkeypatch, tmp_path):
    captured = _install_fake_google_genai(monkeypatch)
    expected_mime_types = {
        ".bmp": "image/bmp",
        ".gif": "image/gif",
        ".jpeg": "image/jpeg",
        ".jpg": "image/jpeg",
        ".png": "image/png",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".webp": "image/webp",
    }
    assert set(expected_mime_types) == IMAGE_EXTENSIONS
    scorer = GeminiInterleaveImageScorer(max_attempts=1)

    for suffix, expected_mime_type in expected_mime_types.items():
        image_path = tmp_path / f"image{suffix}"
        image_path.write_bytes(b"image-bytes")
        part = scorer._image_part(str(image_path))
        assert part["mime_type"] == expected_mime_type

    assert [mime_type for _, mime_type in captured["parts"]] == list(expected_mime_types.values())
