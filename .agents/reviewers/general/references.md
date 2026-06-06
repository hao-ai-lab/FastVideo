# General reviewer — references

## Public API / serving

- `fastvideo/api/` — public Python API, sampling params, serving config.
- `fastvideo/entrypoints/` — CLI entry points and dispatch.
- `fastvideo/worker/` — worker subprocess.

Recent API work (the `[feat] [N/n] Improve API` series):
- #1218 (1/n) initial files for new public API.
- #1220 (2/n) `video_generator` support.
- #1226 (3/n) CLI extension.
- #1234 (4/n) sampling param refactor + presets.
- #1235 misc API cleanup.
- #1237 (5/n) `ServeConfig.default_request` into OpenAI serving.
- #1238 (5.5/n) streaming server config surface + serve dispatch.
- #1239 (6/n) LTX-2 public preset + gpu_pool translation.

## Pipeline stages

- `fastvideo/pipelines/stages/` — cross-model stages (text_encoding,
  denoising, decode, etc).
- `fastvideo/pipelines/samplers/` — sampler implementations.

## CI

- `.github/workflows/ci-precommit.yml`
- `.github/workflows/ci-aggregate-status.yml`
- `.github/workflows/ci-slash-commands.yml`
- `.github/workflows/ci-trigger-full-suite.yml`
- `.github/workflows/community-*.yml` (issue labeling, stale, welcome).
- `.github/workflows/publish-*.yml` (release publishing).
- `.github/mergify.yml` — auto-labeling + merge protections.
- `.buildkite/` — Buildkite pipeline.

Recent CI work (Eigensystem / Jinzhe Pan, heavy churn Mar–Apr 2026):
- #1186, #1187 — CI infrastructure cleanup + workflow reorganization (1/2, 2/2).
- #1200 — Replace Merge Queue with auto-merge.
- #1202 — Fork PR checkout fix.
- #1210, #1211, #1212 — test retry + reference-video handling.
- #1213 — `pull_request_target` for Full Suite.
- #1214 — direct test retry.
- #1215 — update instead of rebase for auto sync.
- #1216 — mergify upgrade.

## Docs

- `docs/` — MkDocs source.
- `docs/contributing/pull_requests.md` — PR flow + `/test <scope>` commands.
- `docs/contributing/ci_architecture.md` — the CI model.
- `docs/contributing/overview.md`.
- `docs/contributing/coding_agents.md` — how agents should contribute.
- `docs/design/overview.md` — architecture.
- `mkdocs.yml` — nav.

## UI

- `ui/` — Job Runner UI (recently re-landed in #1189 after revert in #1188).

## Misc

- `pyproject.toml` — deps + build config.
- `mkdocs.yml` — docs build.
- `AGENTS.md` / `CLAUDE.md` — agent / style guide.
- `.github/PULL_REQUEST_TEMPLATE.md` — PR template.
- `.github/mergify.yml` — label auto-application + merge protections.

## Related

- `.agents/skills/` — agent skills (scanning one can quickly show whether a
  task should be a skill or a reviewer).
- `.agents/workflows/` — SOPs.
