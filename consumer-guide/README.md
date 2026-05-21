# FastVideo Config Generator

A standalone Next.js app that helps users generate FastVideo inference commands.
Deployed under `/config-generator` on the FastVideo docs site.

Two pages:

- **Quick Start** (`/config-generator/`) — pick a task + hardware + output mode, get a ready-to-run command.
- **Advanced Tuning** (`/config-generator/tuning/`) — tune generation parameters and see the real benchmarked time.

## Run locally

```bash
pnpm install
pnpm dev
```

Open **http://localhost:3003/config-generator/** (the app is served under the `/config-generator` base path).

## Build

```bash
pnpm build   # static export to ./out
```

## Data files

All data lives in `data/`. The `.tsx` pages contain no hardcoded model or benchmark
data — they read these JSON files:

- **`data/quickstart.json`** — drives the Quick Start page: curated product `profiles`,
  plus task / tier / hardware option lists and defaults.
- **`data/tuning.json`** — drives the Advanced Tuning page: every benchmark `run` is one
  real measured configuration, plus model / workload / gpu / preset option lists and defaults.

Both are the machine-readable record of the RTX 5090 / 4090 inference benchmarks
(`5090_inference.xlsx`, `4090_inference.xlsx`). A configuration is included only when the
benchmark actually recorded a generation time — **nothing is estimated**.

## Adding a configuration

The pages are data-driven — to add or change a configuration, edit the JSON, not the `.tsx`:

- **Quick Start** — add an entry to `profiles[]` in `data/quickstart.json`. If it
  introduces a new task / hardware / tier, also add it to the matching option list.
- **Advanced Tuning** — add an entry to `runs[]` in `data/tuning.json`. If it
  introduces a new model / workload / gpu / preset, also add it to the matching list.

The only thing not in JSON: a task's `icon` must be one of the lucide-react icons
imported in `basic_page.tsx` (`Film`, `Image`, `Sparkles`, `Gamepad2`). A new icon
needs a small `.tsx` edit.

## Quick Start page logic

- Matches the user's selection (**task + tier + gpu + vram**) against `quickstart.json`
  `profiles[]`. One profile per combination.
- Tier rule: **Quick Generate** = fastest small/distilled model; **High Quality** = largest
  model. Prefer FastVideo-owned models; fall back to other models. Both must have a passing
  benchmark.
- If no profile matches, the combination shows "Not suitable" — no command is generated.
- All displayed numbers (time, VRAM) come straight from the benchmark.

## Advanced Tuning page logic

- The **Model** dropdown is filtered by **Workload Type + GPU**: it only lists models with a
  passing benchmark for that combination. Models that ran out of memory on a GPU do not
  appear for it.
- **Est. Time** is a real measured generation time, looked up by model + GPU + attention
  backend — there is no formula. Selecting a model snaps resolution / frames / steps /
  attention to its benchmarked values; tuning them away flags the time with `⚠`.
- Top metrics — Video Length (`frames/fps`), Total Pixels (`h*w*frames`), Latent Tokens
  (`((frames-1)/4+1)*(h/8)*(w/8)`) — are deterministic arithmetic on the chosen config.
- The generated Python command maps each control to a real FastVideo API field
  (`VideoGenerator.from_pretrained` kwargs, `SamplingParam` attributes).

## Notes

This app is independent from the main FastVideo Python package. Run all commands from
this `consumer-guide/` directory.
