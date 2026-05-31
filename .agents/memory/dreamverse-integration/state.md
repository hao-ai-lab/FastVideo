# Current State — 2026-05-06 (D-26 rebase onto origin/main; PR base flipped)

Point-in-time snapshot of branches, commits, and live infrastructure.
Update whenever commits land or services restart.

For HOW to commit / push / verify see [runbook.md](runbook.md). For
roster of co-authors to credit on every commit see
[authors.md](authors.md).

## Branch tips

| Repo | Branch | Tip | Distance |
|---|---|---|---|
| FastVideo | `will/ltx2_sr_port` (**PR #1288 head**) | `fbd823df` | merged-into-main pending; latest tip post-D-16 + integration-review + integration-plan + GPU4 smoke validation |
| FastVideo | **`will/dreamverse-monorepo`** (**REBASED ONTO `origin/main`**, was forked from `will/ltx2_sr_port`) | `83829c5e` | 66 commits ahead of `origin/main` (`c17d33bf`). Contains the full LTX-2 SR port + NVFP4 + Dreamverse monorepo migration + audio kwarg fix + warmup + NVENC build + benchmarks + integration memory dir, all rebased to be openable as a single PR against main. PR #1288 on `will/ltx2_sr_port` is untouched. End-to-end verified on GPU4 pre-rebase with audio continuation across segments 1→2 (`Cached audio latents shape=(1, 8, 126, 16) for segment 2`, `Segment 2: relayed av chunks=22, bytes=3.8MB`, no BrokenPipeError). NVENC build supported but not usable on this dev host (B200 has no NVENC silicon — verified by direct ffmpeg probe). See [decisions-log.md D-19](decisions-log.md#d-19) + [D-20](decisions-log.md#d-20) + [D-21](decisions-log.md#d-21) + [D-26](decisions-log.md#d-26). |
| FastVideo | `will/dreamverse-monorepo-pre-main-rebase-backup-20260506` | `2ee839a3` | **local-only safety backup** of the pre-rebase chain (the same 66 commits stacked on `2aaeee2a`). Keep until the new chain is fully verified by the next round of e2e on a non-stuck dev box or until the PR merges. |
| FastVideo | `will/api_7.10` | `6ae7a99f` | **deprecated** — PR #1287 closed in favor of #1288. Branch can be deleted on origin and locally; kept for now as historical reference. |
| FastVideo | `will/api_8`, `will/ltx2_sr_runtime`, `will/ltx2_nvfp4`, `will/ltx2_post_fixes`, `will/agents_cleanup` | (various) | **deprecated** split bookmarks. Strategy reversed to single mega-PR (D-17). Safe to delete locally; not pushed to origin. |
| FastVideo | `will/ltx2_sr_port-pre-1286-rebase` | `1baa60bb` | **local-only safety backup** of pre-rebase chain (37 commits); keep until next slice merges |
| Dreamverse | `will/integrate-public-fastvideo` | `ec8ef92` | 10 commits ahead of `737f3c1` (the dep switch) |
| FastVideo-internal | their `main` | (read-only ref) | — |

FastVideo worktree default branch is `will/ltx2_sr_port`. Other agents
share this worktree — if `git branch --show-current` shows something
else, switch back cleanly with `git checkout will/ltx2_sr_port` (don't
disturb their uncommitted work). I observed this happen repeatedly in
the 2026-05-05 session — confirmed harmless; switching back was always
safe with a clean working tree.

## Post-#1286 rebase summary

PR #1286 merged at `2aaeee2a` (squash). `will/ltx2_sr_port` was rebased
onto new `origin/main`, dropping 4 commits whose content is now in main:

- `cd76cf51` `[feat] streaming: router (multi-replica load balancer)`
- `1ac1e732` `[feat] streaming: fastvideo router-serve CLI`
- `b0b7f59c` `[test] streaming: router registry + health loop ...`
- `40e265b8` `[fix] streaming: router polish — bridge cancel + state
  machine + deps` (squashed into `2aaeee2a` via cherry-pick `a152cb77`)

Rebase was clean — no conflicts. All 33 surviving commits got new SHAs
(rebase rewrites). The pre-rebase tip `1baa60bb` is preserved on the
local backup branch `will/ltx2_sr_port-pre-1286-rebase`.

## New linearized chain (33 commits, slice indices for STACK.md)

| Slice | PR | Commits | Tip SHA | Subject |
|---|---|---|---|---|
| 1-3 | 7.10 (PR #1287) | 3 | `6ae7a99f` | `[test] streaming: generate_async coverage + refreshed streaming test` |
| 4-6 | 8 | 3 | `f32e31ec` | `[test] streaming: contract tests for Dreamverse + Dynamo shapes` |
| 7-15 | LTX-2 SR | 9 | `e7297519` | `feat(ltx2): full i2v conditioning + continuation latent port` |
| 16-21 | NVFP4 | 6 | `6793166b` | `test(nvfp4): lock LTX-2 wiring + typed transformer_quant flow` |
| 22-23 | LTX-2 post-fixes | 2 | `25897b67` | `[fix]: unwrap list-of-generator before torch.randn in LTX-2 latent prep` |
| 24-33 | agents_cleanup | 10 | `b34d9704` | `[docs] dreamverse-integration: add runbook + fresh-context onboarding` |

5 PRs landed (7.5, 7.6, 7.7, 7.8, 7.9), 1 in flight (7.10), 5 remaining
(8 / LTX-2 SR / NVFP4 / post-fixes / agents_cleanup).

## Historical commit chain analysis (pre-#1286 rebase)

The layered chain analysis below documented the pre-rebase SHAs (LTX-2
SR layer, NVFP4 layer, post-handoff fixes layer). Those SHAs no longer
exist on `will/ltx2_sr_port` — they live only on
`will/ltx2_sr_port-pre-1286-rebase`. Content semantics are unchanged;
SHAs were rewritten by the rebase. Kept here for narrative continuity.

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
| 8009 | `dreamverse-server` | 705513 | `/readyz` returns 200, 1 GPU worker on GPU 4, NVFP4 (50.9 GiB), `ENABLE_TORCH_COMPILE=0`, `FASTVIDEO_FFMPEG_BIN=$HOME/opt/ffmpeg-native/bin/ffmpeg`, `FASTVIDEO_VIDEO_CODEC=libx264` (post-D-20 deploy at 2026-05-05 14:??) |
| 5274 | `next-server` (dev) | 707746 | 200 |
| 8000 | unknown FastAPI | — | **Not in handoff.** Probably stray `fastvideo serve`. Verify with `lsof -i :8000` before launching a new BE on the default port. |

## Stashes — DO NOT POP

| Repo | Stash | Reason |
|---|---|---|
| FastVideo | `stash@{0}: WIP on main: 71bfc13d HunyuanVideo plugin` | Pre-existing, unrelated to integration work |
| Dreamverse | `stash@{0}: wip: server modular refactor (split config/prompting/runtime/session)` | 3867-line orphan modular split, **not part of `will/integrate-public-fastvideo`**. Recover on a separate branch if needed. |

## Test status

| Suite | Status |
|---|---|
| FastVideo `fastvideo/tests/api/` (post-D-20) | 185 passed (was 222 before some tests moved; new `test_extra_overrides_routing.py` adds 7) |
| FastVideo `contract/` + `nvfp4_*` + `ltx2_pipeline_smoke` (May 2 handoff) | 222 passed, 1 skipped |
| Playwright e2e against live BE+FE (D-19) | 8 passed (5 backend-health + 2 frontend-shell + 1 preset-prompt-generation) |
| Live segment-1→segment-2 audio continuation (D-20) | passes — `Cached audio latents shape=(1, 8, 126, 16) for segment 2` + `Segment 2: relayed av chunks=22, bytes=3.8MB`, no BrokenPipeError |
| `dreamverse-deploy.sh` flag parser standalone test (D-20) | 13/13 permutations pass + bad-flag rejection (defaults / single flags / both flags / `--no-*` overrides / env-only / flag-overrides-env / both-env+both-no-flags / flags interleaved with positional args) |
| `fastvideo serve --config streaming_demo.yaml` validation (May 2 handoff) | parses cleanly; dotted overrides work |
| `bash -n` on `apps/dreamverse/scripts/install_native_ffmpeg.sh` + `dreamverse-deploy.sh` (D-20) | clean |

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
