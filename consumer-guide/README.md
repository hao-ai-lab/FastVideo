# FastVideo Config Generator

A standalone Next.js app that turns maintained FastVideo inference recipes into
executable YAML or Python. The docs build exports it beneath the GitHub Pages
base path at `config-generator/`.

Two pages are available:

- **Quick Start** (`/config-generator/`) — choose a task, model profile, and GPU
  count.
- **Advanced Tuning** (`/config-generator/tuning/`) — start from an exact model
  recipe and adjust supported configuration fields.

## Run locally

```bash
pnpm install
pnpm dev
```

Open <http://localhost:3003/config-generator/>.

## Build

```bash
pnpm build
```

The build first runs `scripts/validate-data.mjs`, then writes a static export to
`out/`.

## Recipe data

`data/tuning.json` is the app's recipe source. Its `source` block pins the exact
FastVideo revision and repository paths used to transcribe model defaults,
pipeline settings, examples, and verified attention compatibility.

`data/quickstart.json` contains only the curated task/profile choices and engine
offload settings. Each profile references a model from `tuning.json`; the build
fails on missing models, task mismatches, duplicate profiles, unknown attention
backends, or unsupported performance fields.

The guide intentionally publishes no latency, VRAM, RAM, or speedup claims.
Add those only when a reproducible benchmark record includes the FastVideo and
model revisions, GPU type/count, complete resolved recipe, software environment,
prompt/seed, warmup/sample protocol, latency statistic, and peak memory.

## Updating a recipe

1. Check the current registry, model-family preset, pipeline config, maintained
   example, and support matrix.
2. Update the model entry in `data/tuning.json` and advance
   `source.fastvideoCommit` to the inspected revision.
3. Update `data/quickstart.json` only if the curated profile selection changes.
4. Run `pnpm build`.

The UI reads the JSON directly; model defaults do not belong in the `.tsx`
components.
