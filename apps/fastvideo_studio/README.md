# FastVideo Studio

A lightweight web-based UI for interacting with FastVideo.

## Features

The UI currently supports:

- Inference
- Finetuning & Distillation
- Datasets
- Gallery View

## Project Structure

```
apps/fastvideo_studio/
├── server.py / job_runner.py / database.py   # FastAPI backend + job lifecycle
├── job_queue.py                              # Queue scheduling rules + last-frame references (GPU-free)
├── merge.py                                  # Join a scene's clips into one video (ffmpeg stream copy, GPU-free)
├── trim.py                                   # Cut a clip to an in/out range, non-destructively (GPU-free)
├── mock_server.py                            # In-memory API mock for e2e tests
├── models/                                   # Pydantic request models (shared with the mock)
├── training_config.py                        # Studio workloads → fastvideo/train YAML configs
├── tests/                                    # Backend unit tests (pytest)
├── e2e/                                      # Playwright specs (run against the mock)
└── src/
    ├── app/                                  # Next.js App Router pages (thin routes)
    ├── components/
    │   ├── shell/                            # App chrome: header, sidebars, layout
    │   ├── jobs/                             # Job queue, cards, create-job modal, log sidebar
    │   ├── datasets/                         # Dataset cards, upload, captions
    │   └── ui/                               # shadcn-style primitives (shared theme)
    ├── stores/                               # Framework-agnostic state + React bridge (hooks/)
    ├── lib/                                  # API client, types, option persistence
    └── test/                                 # Vitest setup + factories
```

The visual theme (slate light/dark palettes, IBM Plex type, `#356cff` accent)
is shared with `apps/dreamverse`; the toggle in the header persists the choice
per browser.

## Testing

```bash
npm run typecheck        # tsc
npm test                 # vitest unit tests
npm run e2e              # Playwright against the in-memory mock backend
python -m pytest tests/  # backend unit tests (from apps/fastvideo_studio)
```

## Quick Start

For local development, install dependencies and start the Next.js dev server:

```bash
cd apps/fastvideo_studio
npm install
npm run dev
```

You can then access the app at [http://localhost:3000](http://localhost:3000).
Start the Python API server (default port 8189) in a separate terminal — see below.

For a production build of the web app together with the API server:

```bash
cd apps/fastvideo_studio
npm install
npm run build
npm run start:all
```

### Running API and Web Separately

The UI is composed of two separate components:

- API Server
- Web Server

To run each component separately, you can use the commands `npm run start:web` (after `npm run build`) and `npm run start:api`.

You can also run the API server with this command from the `apps/` directory:

```bash
python -m fastvideo_studio.server --output-dir /path/to/videos --log-dir /path/to/logs
```

Running it this way allows you to pass command line parameters.

The API server defaults to port 8189. To configure this, you can edit `.env.local` file.
Refer to `.env.example` for reference.

### Job queue

`POST /api/jobs/queue` puts jobs in a queue; the server runs them oldest-first,
`--max-concurrent-jobs` at a time (default 1). Start on a job while that many
are already running is refused; queue it instead. Queued jobs survive a server
restart and carry on.

Inference jobs run on that many long-lived worker threads. The model's GPU
processes are killed when the thread that started them exits, so a worker that
never exits is what keeps the model loaded from one job to the next (it loads
once, not per job). A worker holds one model at a time: a job with a different
model or settings replaces it, and a failed or stopped job unloads it so the next
one starts clean. Each extra worker loads its own copy of the model.

A job can start from how another job's video ended, two ways:

* **Last frame** -- a reference whose source is `job-last-frame:<job id>`. A
  hard, exact opening frame: the prompt is rewritten to say the shot begins
  from it.
* **Last clip** -- a reference whose source is `job-last-clip:<job id>`, and
  `media_type: "video"`. The trailing ~1s of the source job's video instead of
  a single still, so the model has real motion to continue rather than a pose
  to guess it from. Costs one of H3's 3 video-reference slots instead of one
  of its 9 image slots.

Either is extracted when the job runs, so the reference can be added before
that job has finished, and the queue holds the job until it has. If that job
fails, is stopped, or is deleted, the waiting job fails with a message saying
so (and so does anything waiting on *it*); fix the earlier job and queue them
again. Starting such a job directly while its source is unfinished is refused
with a 409.

### Merging a scene

**Merge scene** on the Scenes page joins a scene's clips, in the order shown (the
take chosen for each), into one video under `<output-dir>/merged/`, then plays it
and offers a download. It is enabled once every clip has finished, and refuses
otherwise, so a scene missing a clip can't pass for the whole thing.

The clips are copied, not re-encoded (about a second for 15 clips, no quality
loss). That needs them to share size, frame rate and audio, which clips from one
scene do; if one differs, the merge says which. Each clip is cut at the end of its
last video frame, because H3's audio runs a few milliseconds longer and would
otherwise leave a small gap at every join. It uses `ffmpeg` from `PATH`,
`$FASTVIDEO_FFMPEG_BIN`, or the copy bundled with `imageio-ffmpeg`.

### Editing a clip's video

**Edit video** on a finished clip in Scenes cuts it to a chosen in/out range and
adjusts its brightness, contrast and saturation, in place -- other jobs'
last-frame/last-clip references, and a later merge, all just read the job's
`output_path`, so nothing else needs to know a clip was edited. Frame-accurate
(re-encoded, not stream-copied), so the cut isn't limited to the source's
keyframes. Audio is cut to the same range and carried over too (re-encoded to
AAC regardless of the source's codec), unless the source has none.

Color grading is per **section**, not the whole clip at once, like a razor/blade
tool: **Split at current time** cuts the kept range wherever the preview is
paused, and each resulting section gets its own brightness, contrast and
saturation sliders, next to a before/after preview of a still frame
(**Preview** on a section seeks to its start and captures that frame; **Use
current frame** captures wherever the preview is paused instead). That preview
is computed entirely in the browser (`src/lib/grading.ts` mirrors `trim.py`'s
`_apply_grade` pixel math) as the sliders move, so it never waits on the
server -- only **Apply** does that, to actually re-render the file. **Previous
frame**/**Next frame** step the preview by exactly one frame (at the clip's
own fps), for lining up a cut precisely -- the native scrub bar alone isn't
frame-accurate. **Merge with next** undoes a cut, keeping the earlier of the
two sections' grade.
Sections are always contiguous and cover the whole kept range -- there's no
reordering or gaps, only where to cut and what to grade each side; a plain
range trim with no cuts is just the one-section case. Audio is one continuous
track regardless of how many sections there are.

The untouched original is always kept alongside it the first time a clip is
edited, and **every edit always re-renders from that original** with the full
set of values shown in the dialog -- range and sections together, in one pass.
That's what lets adjusting either one leave the other in place instead of each
one silently discarding the other. The job itself records the range/sections
of its last edit (`edit_start_seconds`/`edit_end_seconds`/`edit_segments`,
alongside `output_path`), so reopening the dialog -- even after closing it or
reloading the page -- shows exactly what produced the current video, not a
guess: a never-edited clip shows one neutral section covering its full length;
an edited one shows every section that made it, cuts and start offset
included. **Restore original** undoes every edit in one step and resets those
recorded values back to one neutral section too.

### API Endpoints

| Method   | Path                          | Description                                |
| -------- | ----------------------------- | ------------------------------------------ |
| `GET`    | `/api/models`                 | List available models                      |
| `GET`    | `/api/jobs`                   | List all jobs (newest first)               |
| `GET`    | `/api/jobs/{id}`              | Get a single job's details                 |
| `POST`   | `/api/jobs`                   | Create a new job                           |
| `POST`   | `/api/jobs/{id}/start`        | Start a pending/stopped/failed job         |
| `POST`   | `/api/jobs/{id}/stop`         | Request a running job to stop              |
| `POST`   | `/api/jobs/queue`             | Queue jobs (`{"job_ids": [...]}`), in order |
| `POST`   | `/api/jobs/{id}/queue`        | Queue one job                              |
| `POST`   | `/api/jobs/{id}/dequeue`      | Take a queued job back out of the queue    |
| `POST`   | `/api/jobs/{id}/last-frame`   | Save the job's last frame, for another job to start from |
| `POST`   | `/api/jobs/{id}/last-clip`    | Save the job's trailing ~1s, for another job to continue from |
| `POST`   | `/api/scenes/merge`           | Join finished jobs' videos (`{"job_ids": [...], "name": ""}`) in order |
| `POST`   | `/api/jobs/{id}/trim`         | Cut a job's video to a range and/or grade it section by section (`{"start_seconds": 0, "end_seconds": null, "segments": [{"end_seconds": null, "brightness": 0, "contrast": 1, "saturation": 1}]}`), in place |
| `POST`   | `/api/jobs/{id}/restore-video`| Undo every edit, back to the original video |
| `GET`    | `/api/merged/{filename}`      | Stream a merged scene video                |
| `DELETE` | `/api/jobs/{id}`              | Delete a job                               |
| `GET`    | `/api/jobs/{id}/video`        | Stream the generated video/image           |
| `GET`    | `/api/jobs/{id}/logs`         | Get job logs (polling, supports `?after=`) |
| `GET`    | `/api/jobs/{id}/download_log` | Download the job's log file                |
