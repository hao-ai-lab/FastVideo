# SPDX-License-Identifier: Apache-2.0
"""Draft MiniMax-H3 jobs for a scene with a DeepSeek Harness agent.

One draft is one DSH SDK session in its own workspace:

    <runs_root>/<run_id>/
        workspace/script.md                     the scene, as given
        workspace/references.md                 the reference files and what they are
        workspace/.dsh/skills/h3-job-writer/    the job-writing skill (copied)
        jobs/                                   the jobs the agent wrote and the validator passed
        scene-writer.patch.yml                  points the h3-scene-writer plugin at this run

The plugin is a DSH bundle (`dsh/`) installed into the harness home's `sdk`
profile, so its harness imports resolve; it is (re)installed automatically
whenever its files change.

The agent writes jobs only through the plugin's `h3_write_job` tool, which
validates each one, so everything in `jobs/` already passed the structural
checks. Nothing is posted to the studio here; the caller shows the drafts.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKILLS_DIR = HERE / "skills"
SKILL_DIR = SKILLS_DIR / "h3-job-writer"
REVISER_SKILL_DIR = SKILLS_DIR / "h3-job-reviser"
BUNDLE_DIR = HERE / "dsh"
BUNDLE_SOURCES = ("h3-scene-writer.ts", "validate-job.ts", "reference-images.ts")
BUNDLE_FILES = ("package.json", "cordis.patch.yml", *BUNDLE_SOURCES)

_install_lock = threading.Lock()

# The harness's own DeepSeek adapter; any other provider goes through its pi-ai
# adapter (Cerebras, Groq, OpenAI, ...), which needs a route in the run's patch.
DEEPSEEK_PROVIDER = "deepseek-official"
DEFAULT_MODELS = {DEEPSEEK_PROVIDER: "deepseek-v4-flash", "cerebras": "qwen-3.8-27b"}
# Models a provider serves that the harness's installed catalog does not list yet, as llm-pi-ai model profiles
# (from the provider's own model list). `input` must name `image` for the agent's `read_image` tool to work.
EXTRA_MODELS = {
    "cerebras": {
        "qwen-3.8-27b": {
            "name": "Qwen 3.8 27B",
            "contextWindow": 65536,
            "maxTokens": 32768,
            "input": ["text", "image"]
        },
    },
}
IMAGE_PIXEL_BUDGET = 1024 * 1024
# The child env var the pi-ai route reads its key from.
PI_AI_KEY_ENV = "SCENE_WRITER_API_KEY"

DEFAULT_SETTINGS_FILE = Path.home() / ".config" / "fastvideo-studio" / "scene-writer.env"

# Drafts hold the reference images their jobs point at, so they belong where
# the server that renders those jobs can read them: beside its uploads.
_server_runs_root: Path | None = None


def use_server_data_dir(data_dir: Path) -> None:
    """Keep drafts under the studio server's data dir unless SCENE_WRITER_RUNS_ROOT says otherwise."""
    global _server_runs_root
    _server_runs_root = Path(data_dir) / "scene_writer"


def read_settings_file(path: Path) -> dict[str, str]:
    """`KEY=value` lines from the settings file (an optional `export ` prefix and quotes allowed)."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def _setting(name: str, default=None):
    """An environment variable, else the settings file, else the default. Read per draft, so file edits apply without a restart."""
    if name in os.environ:
        return os.environ[name]
    path = Path(os.environ.get("SCENE_WRITER_SETTINGS_FILE", DEFAULT_SETTINGS_FILE)).expanduser()
    return read_settings_file(path).get(name, default)


@dataclass
class SceneWriterSettings:
    """Where the harness lives and which model drives it (from the environment or the settings file by default)."""

    harness_root: Path = field(
        default_factory=lambda: Path(_setting("SCENE_WRITER_HARNESS_ROOT", "/mnt/lustre/vlm-kevin/deepseek-harness")))
    runs_root: Path = field(default_factory=lambda: Path(
        _setting("SCENE_WRITER_RUNS_ROOT") or _server_runs_root or Path.home() / "h3_scene_writer").expanduser())
    node_bin_dir: str | None = field(default_factory=lambda: _setting("SCENE_WRITER_NODE_BIN_DIR"))
    provider: str = field(default_factory=lambda: _setting("SCENE_WRITER_PROVIDER", DEEPSEEK_PROVIDER))
    model: str | None = field(default_factory=lambda: _setting("SCENE_WRITER_MODEL"))
    base_url: str | None = field(default_factory=lambda: _setting("SCENE_WRITER_BASE_URL"))
    api_key: str | None = field(default_factory=lambda: _setting("SCENE_WRITER_API_KEY"))

    @property
    def dsh_bin(self) -> Path:
        return self.harness_root / "apps" / "cli" / "lib" / "bin.js"

    @property
    def sdk_src(self) -> Path:
        return self.harness_root / "python" / "sdk" / "src"

    @property
    def dsh_home(self) -> Path:
        return self.runs_root / "dsh-home"

    @property
    def uses_pi_ai(self) -> bool:
        return self.provider != DEEPSEEK_PROVIDER

    def resolved_model(self) -> str:
        model = self.model or DEFAULT_MODELS.get(self.provider)
        if not model:
            raise RuntimeError(f"Set SCENE_WRITER_MODEL: there is no default model for provider {self.provider!r}.")
        return model

    def child_env(self) -> dict[str, str]:
        env: dict[str, str] = {}
        if self.node_bin_dir:
            env["PATH"] = f"{self.node_bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
        if self.uses_pi_ai and self.api_key:
            env[PI_AI_KEY_ENV] = self.api_key
        return env


@dataclass
class ReferenceFile:
    path: str
    media_type: str  # "image" | "audio" | "video"
    description: str = ""


def video_has_sound(path: str) -> bool:
    """Whether a reference video carries a soundtrack; H3 then gives it an <Audio N> label too."""
    try:
        import av  # noqa: PLC0415
        with av.open(path) as container:
            return bool(container.streams.audio)
    except Exception:  # noqa: BLE001 - unreadable here means the validator's own existence check reports it
        return False


def extract_frames(video: str,
                   out_dir: Path,
                   seconds_apart: float = 2.0,
                   max_frames: int = 8,
                   width: int = 768) -> list[Path]:
    """Stills from a rendered take for the agent to look at: evenly spaced, about `seconds_apart` apart, each
    named by its time (`t03.5s.jpg`). Returns nothing for a missing or unreadable video."""
    try:
        import av  # noqa: PLC0415
        with av.open(video) as container:
            stream = container.streams.video[0]
            frames = [(f.time or 0.0, f.to_image()) for f in container.decode(stream)]
    except Exception:  # noqa: BLE001 - a take without a readable video is revised from its job alone
        return []
    if not frames:
        return []
    duration = frames[-1][0] + 1 / float(stream.average_rate or 24)
    count = max(1, min(max_frames, round(duration / seconds_apart)))
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(count):
        # The middle of each equal slice, so the first and last stills are not the clip's edge frames.
        t, image = frames[min(len(frames) - 1, int((i + 0.5) * len(frames) / count))]
        image = image.resize((width, round(image.height * width / image.width)))
        path = out_dir / f"t{t:04.1f}s.jpg"
        image.save(path, quality=90)
        paths.append(path)
    return paths


@dataclass
class DraftResult:
    run_id: str
    jobs: list[dict]
    job_files: list[str]
    final_response: str | None
    finish_reason: str | None
    fetched_references: list[dict] = field(default_factory=list)


def _import_sdk(settings: SceneWriterSettings):
    if str(settings.sdk_src) not in sys.path:
        sys.path.insert(0, str(settings.sdk_src))
    from deepseek_harness import DeepSeekHarness  # noqa: PLC0415

    return DeepSeekHarness


def _yaml_str(value: str) -> str:
    return json.dumps(value)  # JSON strings are valid YAML scalars


def _bundle_digest() -> str:
    digest = hashlib.sha256()
    for name in BUNDLE_FILES:
        digest.update(name.encode())
        digest.update((BUNDLE_DIR / name).read_bytes())
    return digest.hexdigest()


def _build_bundle(settings: SceneWriterSettings, env: dict[str, str]) -> None:
    """Compile the plugin into one `dist/` file: DSH loads installed bundles as plain JavaScript."""
    esbuilds = sorted(
        (settings.harness_root / "node_modules" / ".pnpm").glob("esbuild@*/node_modules/esbuild/bin/esbuild"))
    if not esbuilds:
        raise RuntimeError(f"esbuild not found under {settings.harness_root}/node_modules; run `pnpm install` there.")
    subprocess.run([
        str(esbuilds[-1]), BUNDLE_SOURCES[0], "--bundle", "--external:@deepseek-ai/*", "--format=esm",
        "--platform=node", "--target=node22", "--outfile=dist/h3-scene-writer.js", "--log-level=warning"
    ],
                   cwd=BUNDLE_DIR,
                   env=env,
                   check=True,
                   capture_output=True,
                   text=True,
                   timeout=120)


def ensure_bundle_installed(settings: SceneWriterSettings) -> None:
    """Install the h3-scene-writer bundle into the sdk profile, or refresh a stale copy."""
    marker = settings.dsh_home / "h3-scene-writer.installed"
    digest = _bundle_digest()
    with _install_lock:
        if marker.is_file() and marker.read_text() == digest:
            return
        settings.dsh_home.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, **settings.child_env(), "DSH_HOME": str(settings.dsh_home)}
        _build_bundle(settings, env)
        proc = subprocess.run(
            ["node",
             str(settings.dsh_bin), "plugin", "--profile", "sdk", "add", "--force", f"file:{BUNDLE_DIR}"],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
            check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Installing the scene-writer bundle failed:\n{proc.stdout[-2000:]}{proc.stderr[-2000:]}")
        marker.write_text(digest)


def _yaml_list(values: list[str]) -> str:
    return "".join(f"\n      - {_yaml_str(v)}" for v in values) or " []"


def _write_patch(path: Path,
                 out_dir: Path,
                 references: list[ReferenceFile],
                 sound_videos: list[str],
                 refs_dir: Path | None,
                 settings: SceneWriterSettings,
                 expected_name: str = "") -> None:
    # An id-targeted row replaces the installed row's whole config.
    text = ("- id: h3-scene-writer\n"
            "  config:\n"
            f"    outDir: {_yaml_str(str(out_dir))}\n"
            f"    references:{_yaml_list([r.path for r in references])}\n"
            f"    soundVideos:{_yaml_list(sound_videos)}\n"
            f"    refsDir: {_yaml_str(str(refs_dir) if refs_dir else '')}\n"
            f"    expectedName: {_yaml_str(expected_name)}\n")
    if settings.uses_pi_ai:
        # The base profile mounts llm-pi-ai with no routes; this gives it one for the provider.
        text += ("- id: llm-pi-ai\n"
                 "  config:\n"
                 "    providers:\n"
                 f"      {_yaml_str(settings.provider)}:\n"
                 f"        apiKeyEnv: {PI_AI_KEY_ENV}\n" +
                 (f"        baseURL: {_yaml_str(settings.base_url)}\n" if settings.base_url else ""))
        model = settings.resolved_model()
        extra = EXTRA_MODELS.get(settings.provider, {}).get(model)
        if extra:
            # A `models` list replaces the route's catalog; the run only needs its own model.
            text += f"        models: [{json.dumps({'id': model, **extra})}]\n"
        # Images the agent reads are sent at most this many pixels (the harness default is 2048x2048): enough to judge
        # a reference or a take, at a quarter of the tokens, which matters in a 65k context.
        text += f"        requestImagePixelBudget: {IMAGE_PIXEL_BUDGET}\n"
    path.write_text(text)


def _references_md(references: list[ReferenceFile], sound_videos: list[str], find_references: bool) -> str:

    def row(r: ReferenceFile) -> str:
        kind = "video with sound: it also takes the next <Audio N> label" if r.path in sound_videos else r.media_type
        return f"- `{r.path}` ({kind}){': ' + r.description if r.description else ''}"

    rows = "\n".join(row(r) for r in references) if references else "(none given)"
    text = f"# Reference files\n\nUse these exact absolute paths in `references`.\n\n{rows}\n"
    if find_references:
        text += ("\nYou may also use `find_reference_image` to download a photo for a person, place or object the "
                 "script needs that none of these files shows. Never replace a file listed here.\n")
    return text


def fetched_references(run_dir: Path) -> list[dict]:
    """The images `find_reference_image` downloaded in a run, with their source pages and licenses."""
    sources = run_dir / "refs" / "sources.json"
    return json.loads(sources.read_text()) if sources.is_file() else []


def build_prompt(scene_name: str, instructions: str) -> str:
    stem = scene_name.replace("-", "_")
    extra = f"\n\nAdditional direction from the user:\n{instructions.strip()}" if instructions.strip() else ""
    return ("/h3-job-writer\n\n"
            "Write MiniMax-H3 jobs for the scene in `script.md`, using only the reference files listed in "
            "`references.md` (and any it lets you download). "
            f"Name the jobs `{scene_name}-clip-01`, `{scene_name}-clip-02`, ... and pass `file_name` values "
            f"`{stem}_clip_01`, `{stem}_clip_02`, ... to `h3_write_job`. "
            "Write every clip through `h3_write_job` and fix anything it reports. Do not post or start jobs."
            f"{extra}")


def _check_ready(settings: SceneWriterSettings) -> str:
    """The model to use; raises when the harness or the model settings are not usable."""
    model = settings.resolved_model()
    if settings.uses_pi_ai and not settings.api_key:
        raise RuntimeError(f"Set SCENE_WRITER_API_KEY to your {settings.provider} API key.")
    if not settings.dsh_bin.is_file():
        raise RuntimeError(f"DeepSeek Harness is not built: {settings.dsh_bin} is missing "
                           "(run `pnpm install && pnpm run build` in the harness checkout).")
    ensure_bundle_installed(settings)
    return model


def _run_session(settings: SceneWriterSettings, model: str, workspace: Path, patch: Path, prompt: str, session_id: str,
                 on_event: Callable[[dict], None] | None):
    """Run one agent turn in `workspace` with the run's patch; returns the SDK's RunResult."""
    DeepSeekHarness = _import_sdk(settings)
    with DeepSeekHarness(
            dsh_home=str(settings.dsh_home),
            cwd=str(workspace),
            dsh_bin=str(settings.dsh_bin),
            profile="sdk",
            patches=(str(patch), ),
            provider=settings.provider,
            model=model,
            # These two reach only the DeepSeek adapter; a pi-ai route gets them from the patch.
            base_url=None if settings.uses_pi_ai else settings.base_url,
            api_key=None if settings.uses_pi_ai else settings.api_key,
            env=settings.child_env(),
    ) as harness:
        result = harness.run(
            prompt,
            session_id=session_id,
            on_notification=(lambda n: on_event(n.payload)) if on_event else None,
        )
        # A plugin that fails to load only logs to stderr and the session runs without its tools.
        load_errors = [
            line for line in list(harness.client._stderr_lines)
            if line.startswith("h3-scene-writer (fastvideo-h3-scene-writer):")
        ]
    if load_errors:
        raise RuntimeError(f"The scene-writer plugin failed to load: {load_errors[0]}")
    if result.finish_reason == "error":
        raise RuntimeError(_turn_error_message(result.events, settings))
    return result


def _turn_error_message(events: list[dict], settings: SceneWriterSettings) -> str:
    """Explain why the agent's turn ended in an error (usually the model provider refusing the request)."""
    message = "unknown error"
    for event in reversed(events):
        if event.get("type") == "turn/end":
            error = ((event.get("data") or {}).get("reason") or {}).get("error") or {}
            message = error.get("message") or message
            break
    if message.startswith("402"):
        return (f"{settings.provider} refused the request with 402 Payment Required: the account "
                "behind SCENE_WRITER_API_KEY has no credits or billing set up.")
    if message.startswith("401") or message.startswith("403"):
        return f"{settings.provider} rejected SCENE_WRITER_API_KEY ({message})."
    return f"The model call failed: {message}"


def draft_scene(
    script: str,
    references: list[ReferenceFile],
    scene_name: str,
    instructions: str = "",
    settings: SceneWriterSettings | None = None,
    on_event: Callable[[dict], None] | None = None,
    run_id: str | None = None,
    find_references: bool = False,
) -> DraftResult:
    """Run one drafting session and return the jobs it wrote, in clip order.

    With `find_references`, the agent may download reference photos from Wikimedia Commons into the run's
    `refs/` directory; `fetched_references` lists them.
    """
    if not references and not find_references:
        raise ValueError("Give at least one reference file, or let the writer find references online.")
    settings = settings or SceneWriterSettings()
    model = _check_ready(settings)
    run_id = run_id or uuid.uuid4().hex[:12]
    run_dir = settings.runs_root / run_id
    workspace = run_dir / "workspace"
    out_dir = run_dir / "jobs"
    workspace.mkdir(parents=True, exist_ok=False)
    out_dir.mkdir()
    (workspace / "script.md").write_text(script)
    sound_videos = [r.path for r in references if r.media_type == "video" and video_has_sound(r.path)]
    (workspace / "references.md").write_text(_references_md(references, sound_videos, find_references))
    shutil.copytree(SKILL_DIR, workspace / ".dsh" / "skills" / SKILL_DIR.name)
    patch = run_dir / "scene-writer.patch.yml"
    _write_patch(patch, out_dir, references, sound_videos, run_dir / "refs" if find_references else None, settings)

    result = _run_session(settings, model, workspace, patch, build_prompt(scene_name, instructions), f"scene-{run_id}",
                          on_event)
    files = sorted(out_dir.glob("*.json"))
    return DraftResult(
        run_id=run_id,
        jobs=[json.loads(f.read_text()) for f in files],
        job_files=[str(f) for f in files],
        final_response=result.final_response,
        finish_reason=result.finish_reason,
        fetched_references=fetched_references(run_dir),
    )


@dataclass
class RevisionResult:
    run_id: str
    original: dict
    #: the revised job, or None when the agent wrote nothing
    job: dict | None
    #: the agent's explanation of the cause and its changes
    final_response: str | None
    finish_reason: str | None


# The job-file fields; a Studio job also carries runtime state (status, outputs) the agent does not need.
JOB_FILE_FIELDS = ("model_id", "name", "workload_type", "prompt", "references", "negative_prompt", "guidance_scale",
                   "num_frames", "fps", "height", "width", "num_inference_steps", "seed", "num_gpus",
                   "use_fsdp_inference")


def job_file_form(job: dict) -> dict:
    """A job in the job-file form the skills use: only job-file fields, `prompt` as a list of lines."""
    form = {k: job[k] for k in JOB_FILE_FIELDS if job.get(k) is not None}
    prompt = form.get("prompt", "")
    form["prompt"] = prompt.split("\n") if isinstance(prompt, str) else list(prompt)
    form["references"] = [{"source": r["source"], "media_type": r["media_type"]} for r in form.get("references") or []]
    return form


def build_revision_prompt(job_name: str, file_name: str, has_frames: bool = False) -> str:
    frames = ("Stills from the rendered take are in `take_frames/`, named by their time in the clip; look at them "
              "with `read_image` and check the feedback against what was rendered. " if has_frames else "")
    return ("/h3-job-reviser\n\n"
            f"Revise the clip `{job_name}` in `current_job.json` to fix the problem described in `feedback.md`. "
            "The clips around it in the scene are in `neighbors/`. "
            f"{frames}"
            f"Write the revised job with `h3_write_job`, keeping the name `{job_name}` and passing `file_name` "
            f"`{file_name}`. Do not post or start jobs. Finish with the cause you found and each change you made.")


def revise_clip(
    job: dict,
    feedback: str,
    neighbors: list[dict] | None = None,
    settings: SceneWriterSettings | None = None,
    on_event: Callable[[dict], None] | None = None,
    run_id: str | None = None,
) -> RevisionResult:
    """Run one revision session for a clip job (job-file or Studio form) and return the revised job.

    The revision may use the clip's own references and its neighbours' (for example to add a voice that the
    clip before it already uses); it must keep the clip's name.
    """
    name = job.get("name") or ""
    if not name:
        raise ValueError("The job to revise has no name.")
    if not feedback.strip():
        raise ValueError("Describe what is wrong with the clip.")
    settings = settings or SceneWriterSettings()
    model = _check_ready(settings)
    run_id = run_id or uuid.uuid4().hex[:12]
    run_dir = settings.runs_root / run_id
    workspace = run_dir / "workspace"
    out_dir = run_dir / "jobs"
    (workspace / "neighbors").mkdir(parents=True, exist_ok=False)
    out_dir.mkdir()

    original = job_file_form(job)
    neighbors = [job_file_form(n) for n in neighbors or []]
    (workspace / "current_job.json").write_text(json.dumps(original, indent=2) + "\n")
    (workspace / "feedback.md").write_text(f"# What went wrong in {name}\n\n{feedback.strip()}\n")
    for n in neighbors:
        stem = (n.get("name") or "clip").replace("-", "_")
        (workspace / "neighbors" / f"{stem}.json").write_text(json.dumps(n, indent=2) + "\n")

    references: list[ReferenceFile] = []
    for j in [original, *neighbors]:
        for r in j.get("references") or []:
            if r["source"] not in {ref.path for ref in references}:
                references.append(ReferenceFile(r["source"], r["media_type"]))
    sound_videos = [r.path for r in references if r.media_type == "video" and video_has_sound(r.path)]
    (workspace / "references.md").write_text(_references_md(references, sound_videos, False))
    for skill in (SKILL_DIR, REVISER_SKILL_DIR):
        shutil.copytree(skill, workspace / ".dsh" / "skills" / skill.name)
    patch = run_dir / "scene-writer.patch.yml"
    _write_patch(patch, out_dir, references, sound_videos, None, settings, expected_name=name)

    # A Studio take carries its rendered video; a job file has none.
    frames = extract_frames(job["output_path"], workspace / "take_frames") if job.get("output_path") else []

    file_name = name.replace("-", "_")
    result = _run_session(settings, model, workspace, patch, build_revision_prompt(name, file_name, bool(frames)),
                          f"revise-{run_id}", on_event)
    # The newest file the agent wrote; the plugin already checked it keeps the clip's name.
    written = sorted(out_dir.glob("*.json"), key=lambda f: f.stat().st_mtime)
    return RevisionResult(
        run_id=run_id,
        original=original,
        job=json.loads(written[-1].read_text()) if written else None,
        final_response=result.final_response,
        finish_reason=result.finish_reason,
    )
