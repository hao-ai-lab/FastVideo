# Current State — 2026-05-03

Point-in-time snapshot of branches, commits, and live infrastructure.
Update whenever commits land or services restart.

## Branch tips

| Repo | Branch | Tip | Distance |
|---|---|---|---|
| FastVideo | `will/ltx2_sr_port` | `156103b9` | 14 commits ahead of i2v port base `cfccd292` |
| Dreamverse | `will/integrate-public-fastvideo` | `ec8ef92` | 10 commits ahead of `737f3c1` (the dep switch) |
| FastVideo-internal | their `main` | (read-only ref) | — |

Both worktrees clean. FastVideo worktree currently checked out on
`will/ltx2_sr_port` (was on `will/uv-pip-install-everywhere` per the
May 2 handoff — that's been corrected).

## FastVideo: commit chain `cfccd292..156103b9`

Three layers since LTX-2 i2v port:

### Layer 1 — LTX-2 SR port + alignment harness (5 commits)

```
365a66c7  feat(quantization): upstream LTX-2 FP4Config with lazy flashinfer
433d26b2  feat(ltx2): port LTX-2 SR runtime — upsampler, refine stages, refine args
751d05de  feat(ltx2): wire SR pipeline graph + port denoising/latent-prep stages
af6bbfea  test(ltx2-sr): add numerical alignment harness — public vs internal
974cd430  fix(ltx2-sr): close port gaps surfaced by alignment harness retries
b6ac7630  test(ltx2-sr): pin ltx2 sampling knobs in harness for parity diff
b043d550  fix(api): align public SamplingParam ltx2 defaults with distilled
663dda80  fix(registry): order LTX-2 detectors so distilled wins for distilled paths
cfccd292  feat(ltx2): full i2v conditioning + continuation latent port  (BASE)
```

(Predates the May 2 handoff.)

### Layer 2 — NVFP4 wire-up + per-component compile (6 commits, May 2 handoff)

```
a4760bae  fix(api): propagate generic refine_* args + match internal randn
221cb20a  feat(api): typed per-component CompileConfig + FastVideoArgs carriers
6da342ba  feat(compile): per-component compile + transformer_refine + prepare hook
42b30bf9  feat(ltx2): wire FP4 inference through fastvideo.layers.quantization
94c983a2  refactor(quant): rename FP4 → NVFP4 to disambiguate from other FP4 variants
c6c14c55  test(nvfp4): lock LTX-2 wiring + typed transformer_quant flow
```

See [quantization.md](quantization.md) for what each commit locks in.

### Layer 3 — Post-handoff parity/perf fixes (3 commits, since May 2)

```
a5fcd19c  [fix]: lazy-import flash_attn 2 fallback in attention backend
d4ee5be2  [fix]: avoid model.to() round-trip in Gemma encoder forward
156103b9  [fix]: unwrap list-of-generator before torch.randn in LTX-2 latent prep  (HEAD)
```

Three small fixes — no new features. Continued parity tightening with internal.

## Dreamverse: commit chain `737f3c1..ec8ef92`

```
737f3c1   chore: switch fastvideo dep from FastVideo-internal to public FastVideo
4cc6b30   chore: gitignore Playwright + Next.js build artifacts under apps/web
33caa92   test(e2e): align Playwright specs with the actual production composer
6fd137c   test(e2e): tighten frontend-shell + preset specs to match actual UI
248060b   test(e2e): add Playwright tier with backend-health smoke + preset run
d80c2a8   refactor(server): drive FP4 + per-component compile via typed GeneratorConfig
3d7fd89   feat(skill): launch-demo orchestrator + fastvideo serve YAML
72f69b9   Update ffmpeg installation instructions.
1ba5635   fix(server): block startup on GPU warmup readiness, propagate failures
ec8ef92   fix(server): detect worker death in _send_command via proc.sentinel  (HEAD)
```

The post-handoff trio (`72f69b9`, `1ba5635`, `ec8ef92`) hardens server
startup robustness — ffmpeg install docs, GPU warmup readiness gate, and
worker-death detection.

## Live services (do not duplicate)

| Port | Service | PID | Status |
|---|---|---|---|
| 8009 | `dreamverse-server` | 2453227 | `/readyz` returns 200, 1 warmed GPU worker, queue 0 |
| 5274 | `next-server` (dev) | 2399103 | 200, ~13.6 KB shell |
| 8000 | unknown FastAPI | — | **Not in handoff.** Probably stray `fastvideo serve`. Verify with `lsof -i :8000` before launching a new BE on the default port. |

## Stashes — DO NOT POP

| Repo | Stash | Reason |
|---|---|---|
| FastVideo | `stash@{0}: WIP on main: 71bfc13d HunyuanVideo plugin` | Pre-existing, unrelated to integration work |
| Dreamverse | `stash@{0}: wip: server modular refactor (split config/prompting/runtime/session)` | 3867-line orphan modular split, **not part of `will/integrate-public-fastvideo`**. Recover on a separate branch if needed. |

## Test status (from May 2 handoff, not re-verified post-Layer-3)

| Suite | Status |
|---|---|
| FastVideo `fastvideo/tests/api/` + `contract/` + `nvfp4_*` + `ltx2_pipeline_smoke` | 222 passed, 1 skipped |
| Playwright e2e against live BE+FE | 8 passed (5 backend-health + 2 frontend-shell + 1 preset-prompt-generation) |
| `fastvideo serve --config streaming_demo.yaml` validation | parses cleanly; dotted overrides work |
| `bash -n` on launch-demo skill scripts | clean across all 4 |

The 3 post-handoff commits are small parity fixes; full re-verification is
recommended but not required to read this state.

## Pre-existing failures (NOT caused by this work)

| Test | Failure | Notes |
|---|---|---|
| `fastvideo/tests/ops/quantization/test_absmax_fp8.py::test_create_weights_rejects_invalid_dtype` | `AssertionError not raised` | Pre-existing on `main`. Verified via `git stash` that NVFP4 work doesn't introduce it. See [open-threads.md](open-threads.md) item #2. |

## Source docs (archived 2026-05-03)

The 7 source docs that this memory dir consolidates have been moved into
[`source-archive/`](source-archive/) — see the
[archive README](source-archive/README.md) for the archive policy and
synthesis mapping.

Other untracked items at the FastVideo repo root:
- Nested clones: `dynamo/`, `ray/`, `vllm-omni/`
- Lock files: `uv.lock`, `fastvideo/tests/ssim/.reference_videos_download.lock`
- Skill dirs: `.agents/skills/diagnose-ssim-failure/`, `.agents/skills/review-pr-link/`
- `.agents/exploration/pr-link-review.md` (kept; already promoted to a skill)

## Quick orientation commands

```bash
# FastVideo state
cd /home/william5lin/FastVideo
git log --oneline cfccd292..HEAD     # 14 commits this round

# Dreamverse state
cd /home/william5lin/Dreamverse
git log --oneline 737f3c1..HEAD      # 10 commits this round

# Live stack health (already running)
curl -s http://localhost:8009/readyz | head -c 300
curl -s http://localhost:5274/ -o /dev/null -w "%{http_code}\n"

# Re-verify test suite
.venv/bin/python -m pytest fastvideo/tests/api/ \
    fastvideo/tests/contract/ \
    fastvideo/tests/ops/quantization/test_nvfp4_*.py \
    tests/local_tests/pipelines/test_ltx2_pipeline_smoke.py \
    -q --no-header
```
