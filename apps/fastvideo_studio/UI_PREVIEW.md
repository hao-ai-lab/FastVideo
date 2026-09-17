# Studio inference UI preview

This is a small reviewable iteration on the existing Studio UI. The preview
can run against a copied local database and saved outputs without generating
new media or starting a remote GPU.

## What to try

1. Open Inference and choose a status or model from the dropdowns, or search
   words in a prompt. Filters combine; Clear resets them together. The model
   list shares Create Job's catalog and retains models in recorded jobs.
2. Click a completed job's poster to play its video or inspect its image,
   review the model/settings, and download the result.
   Click the job card to view its configuration; use Details & logs for the
   sidebar with progress and logs. Edit remains a separate action.
   The small download icon on the media works in both the job list and Gallery.
3. Open Create Job or edit a pending job. Output and Generation options are
   shown first; advanced settings are grouped into collapsible sections.
4. Type a precise slider value, or use its number field's up/down controls.
5. Open API example at the top right of the form, copy the cURL command,
   change a form field, and Refresh.
   The export uses the same JSON request descriptor as the UI submission.

The highlighted API panel currently covers job creation and editing. Refresh
does not send the request. Creating a job queues it; starting it is a separate
API operation. The configured API base URL includes `/api` and becomes
`BACKEND_URL`. Existing input paths appear as editable exports such as
`IMAGE_PATH` and `DATA_PATH`; dataset fields can also contain a saved dataset
ID. These variables reference backend inputs and do not upload local files.
Path-bearing examples require `jq` to encode variable values safely as JSON;
the cURL request runs only if that encoding succeeds. Examples without input
path fields remain plain cURL.

Gallery is a completed-results view over the same jobs. It shares the media
preview component, download handler, original output files, and cached posters
with the job list; opening Gallery creates no separate media library.

Posters contain one frame at a maximum of 320 pixels per dimension. They are
generated on demand using Pillow and FFmpeg (or imageio-ffmpeg), with one
decoder at a time and a disk cache capped at 50 files. Lists use lazy images;
the full player mounts only when a result is opened. Gallery pages contain
at most 50 results. A missing poster still allows opening the original media.

## Custom fine-tuned and distilled models

A universal checkpoint selector is deferred because it crosses the UI,
backend validation, and model-loading boundary. The current implementation
has these constraints:

- `server.py:create_job` restricts inference to registered model IDs.
- `job_runner.py:_get_or_create_generator` loads that ID with
  `VideoGenerator.from_pretrained` and caches the resulting generator.
- Training records the latest `checkpoint-*` directory as its output.
  The modular trainer writes distributed training state under `dcp/`, with
  training metadata. This is not a complete inference model directory.
- A converter already exists:
  `fastvideo/train/entrypoint/dcp_to_diffusers.py`. It can export a selected
  role (default `student`) and supports `--verify` to reload its transformer.
  Conversion/verification has not been run for the saved snapshot here.
- LoRA needs a base model plus adapter identity/strength. Distilled models
  also need compatible architecture, attention, scheduler, and sampling
  settings. A filesystem path alone does not express those requirements.

The smallest useful follow-up is **already exported, compatible full model
directories**, limited initially to one known model family:

1. Convert one completed checkpoint with the existing exporter and strictly
   verify it on the backend machine. Keep the training checkpoint intact.
2. Add a small saved-model record containing a display name, backend path,
   base family/preset, workload, and source job. Validate the directory and
   model metadata before making it selectable.
3. Resolve that record in inference validation/loading; preserve its identity
   in job history and include it in generator caching. Show a friendly model
   name in the picker, cURL payload, and result details.
4. Smoke-test loading and generation with that one model on a GPU. Handle
   missing paths and incompatible checkpoints visibly.

Raw training-checkpoint browsing, automatic conversion, LoRA selection, and
arbitrary distilled architectures should be separate changes. Their main
difficulties are export compatibility, backend storage, and selecting the
right inference preset, rather than rendering the dropdown.
