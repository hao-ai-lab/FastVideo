# Decisions Log — D + Q Resolutions

Cross-doc consolidated decision log. Each entry: ID, source doc,
question/decision, rationale, current status.

For implementation status see [pr-roadmap.md](pr-roadmap.md). For
follow-up actions see [open-threads.md](open-threads.md).

**Last updated:** 2026-05-06 (added D-21 — chunk-stutter root cause is software libx264 encoding consuming ~22% of segment wall-time, NOT a migration regression; verified by 3 parallel explore agents that NVFP4 + torch.compile coverage matches FastVideo-internal exactly; landed opt-in NVENC build path in install_native_ffmpeg.sh + `--nvenc`/`--no-nvenc` flag in dreamverse-deploy.sh + `apps/dreamverse/server/benchmarks/benchmark_av_streaming.py` regression test + memory dir update; default codec stays `libx264` for backward compat, opt-in via `--nvenc`. Earlier: added D-12 — GpuPool layer separation, Oracle review post-#1257-merge; added D-13 — prompt enhancer / LLMProvider abstraction shape, Oracle review pre-#1258-merge; added D-14 — streaming auxiliaries cohesion, Oracle review during #1284 review cycle; added D-15 — streaming router placement + sticky/active-active deferral, Oracle review during #1286 review cycle; added D-16 — streaming router polish round 2, second-pass review on top of D-15 covering bridge cancellation hygiene, registry state machine, httpx hard-fail, replica YAML parsing, and `websockets` dep; added D-17 — strategy reversal: abandon 6-PR split in favor of single mega-PR #1288 on `will/ltx2_sr_port`; added D-18 — Option B+ chosen: Dreamverse FE+product-server move into FastVideo as `apps/dreamverse/` subfolder while generic backend stays at `fastvideo.entrypoints.streaming.*`; integration-review.md deprecated, integration-plan.md is the executable migration plan; added D-19 — D-18 executed: 5 commits land on `will/dreamverse-monorepo`, fix-up commits corrected the integration-plan's invalid "delete generic-merged, import public substitutes" assumption — generic-merged files carried product-local instead, e2e passes against migrated code with /proc-verified evidence; added D-20 — segment-2 BrokenPipe root cause was a TWO-direction silent drop of LTX-2 audio kwargs in public `VideoGenerator`).

## Status legend

- ✅ **Resolved** — decision made and implementation complete (or no implementation needed)
- 🟡 **Deferred** — decision made, implementation deferred to a known PR
- 🔴 **Open** — needs decision

## Post-merge architecture decisions

### D-26: Rebase `will/dreamverse-monorepo` directly onto `origin/main` (PR base flip)

**Status:** ✅ Resolved 2026-05-06 (UTC; 2026-05-07 local). 66 commits cleanly replayed via `git rebase origin/main`; force-pushed via `--force-with-lease`. Local backup branch `will/dreamverse-monorepo-pre-main-rebase-backup-20260506` preserved at the pre-rebase tip `2ee839a3`. PR #1288 on `will/ltx2_sr_port` is untouched.
**Source:** User directive: "update this branch will/dreamverse-monorepo to be directly against origin/main for PR purposes instead of ltx_sr_port (don't open new PRs)".

**Question:** `will/dreamverse-monorepo` was forked from `will/ltx2_sr_port` (PR #1288's head). Should it stay stacked on top of `will/ltx2_sr_port`, or be rebased so its PR base becomes `origin/main` directly?

**Decision:** Rebase `will/dreamverse-monorepo` onto `origin/main` directly. This makes the branch openable as a single PR against main containing the entire 66-commit chain (40 from the legacy `will/ltx2_sr_port` work + 26 dreamverse-monorepo-specific commits including D-19/D-20/D-21/D-22).

**Pre-rebase state:**

```
origin/main (c17d33bf)  ← 1 new commit since 2aaeee2a (PR #1253 SSIM cosine)
              │
              └─ 2aaeee2a (PR #1286 squash merge; old shared base)
                          ├─ 40 commits → will/ltx2_sr_port @ fbd823df  (PR #1288)
                          └─ 40 + 26 = 66 commits → will/dreamverse-monorepo @ 2ee839a3
```

**Post-rebase state:**

```
origin/main (c17d33bf)
              │
              └─ 66 commits → will/dreamverse-monorepo @ 83829c5e   (NEW SHAs, same content)

origin/main (c17d33bf)  -- still in main lineage
              │
              └─ 2aaeee2a
                          └─ 40 commits → will/ltx2_sr_port @ fbd823df  (PR #1288, UNCHANGED)
```

**Operation:**

1. **Backup**: `git branch will/dreamverse-monorepo-pre-main-rebase-backup-20260506 will/dreamverse-monorepo` (local-only safety net pinning `2ee839a3`).
2. **Rebase**: `git rebase origin/main` while on `will/dreamverse-monorepo`. All 66 commits replayed conflict-free in ~30s. The single new origin/main commit `c17d33bf` (#1253 SSIM cosine regression) touches only `fastvideo/tests/ssim/*`, `fastvideo/tests/modal/ssim_test.py`, `.agents/skills/seed-ssim-references/SKILL.md`, `fastvideo/pipelines/basic/stable_audio/stages/decoding.py`, and a 56-line block in `fastvideo/entrypoints/video_generator.py`. None of our 66 commits touch the SSIM files; the `video_generator.py` overlap auto-merged because our D-20 changes (`_BATCH_EXTRA_PASSTHROUGH_KEYS`, result-dict surface) and #1253's changes are in different parts of the file.
3. **Verification (file-level, before push)**:
   - Commit count: 66 (PASS)
   - Co-author trailer count: 231 across 66 commits — within historical norm (some legacy `will/ltx2_sr_port` commits had partial trailers per [`authors.md`](authors.md) "Known gaps").
   - Tree-diff vs backup: only the 845-line forward delta from `c17d33bf` (expected). All D-20/D-21/D-22-introduced content (`_BATCH_EXTRA_PASSTHROUGH_KEYS` x2, "unknown field" x1, `av_chunk_interval_ms` x4, `av_chunk_publish_ms` x2, `enable-nvenc` x2, `NVENC_OVERRIDE` x4) survives intact.
   - `pre-commit run --files` on the 7 most-edited files: PASS (yapf/ruff/codespell/mypy/spaces).
   - Runtime pytest hung mid-import on this shared dev box (pre-existing CUDA/torch init issue affecting other commands too) — NOT a rebase regression. The same `fastvideo/tests/api/` suite passed 185/185 pre-rebase.
4. **Force-push**: `git push --force-with-lease=will/dreamverse-monorepo:2ee839a3... origin will/dreamverse-monorepo`. Updated `2ee839a3...83829c5e (forced update)`.

**Implications:**

- The branch can now be opened as a PR against `origin/main` containing all the LTX-2 SR port + NVFP4 + Dreamverse monorepo migration + audio kwarg fix + warmup + NVENC build + benchmarks + integration memory dir, in a single review unit.
- PR #1288 on `will/ltx2_sr_port` is untouched and continues to track its own subset of the work. If PR #1288 lands first, the duplicated commits on `will/dreamverse-monorepo` will be reconciled at the next rebase by `git rebase origin/main` dropping commits whose content is now in main (same mechanism as the post-#1286 rebase recorded in [state.md](state.md) "Post-#1286 rebase summary").
- Local backup `will/dreamverse-monorepo-pre-main-rebase-backup-20260506 @ 2ee839a3` keeps the old chain available; recommend deleting after the new branch state is verified by running on a non-stuck dev box (or after the PR merges).

**Watch-outs:**

- Anyone with a checkout of the OLD `will/dreamverse-monorepo` (pre-rebase) needs to `git fetch origin will/dreamverse-monorepo --force` and discard local commits, OR rebase their local commits onto the new `83829c5e`. None observed in the runbook's worktree-sharing model.
- The trailer-count 231 (vs expected 264 for full coverage) is NOT introduced by this rebase — it's the historical gap from [`authors.md`](authors.md). Verifiable by counting trailers on the backup branch (same 231).

### D-21: AV chunk stutter root cause — software libx264 encoding consumes ~22% of segment wall time; opt-in NVENC fix

**Status:** 🟡 Deferred — fix landed (NVENC build + `--nvenc` flag), default unchanged so no behavior regression for operators who haven't rebuilt ffmpeg yet. Switch default to `h264_nvenc` once benchmark numbers are captured + a release notes entry is published.
**Source:** Live debug session 2026-05-06 prompted by user observation: "stuttering still between chunks" after D-20 + warmup r3 fix. 3 explore agents fanned out (NVFP4 config, torch.compile config, AV chunk pacing).

**Question:** With D-20 (audio kwarg routing fix) + r3 warmup pass landed and warmup_success=true, why is the live deploy still stuttering between chunks? Is the migration missing some quantization or compile coverage that the FastVideo-internal reference has?

**Investigation (3 parallel explore agents):**

1. **NVFP4 config diff** (`bg_0629273d`): No divergence. apps/dreamverse, original Dreamverse, and FastVideo-internal ALL use `layer_profile="refine"` with the same 48-block `fp4_layers` superset, e2m1 mantissa, fp32 alpha, layout_128x4 scale, group size 16. Stage gating (`base` vs `refine`) and `transformer_refine_quant` carrier match. The only naming difference is the public surface rename `FP4Config` → `NVFP4Config`.
2. **torch.compile config diff** (`bg_97081cc5`): No divergence. ALL THREE repos use `inductor / fullgraph=True / max-autotune-no-cudagraphs / dynamic=False` and compile only the transformer + text_encoder. **VAE / audio_vae / vocoder are eager bf16 in all three** — so the migrated repo is not missing a compile pass that the original had. Only behavioral difference: FastVideo-internal calls `target.eval()` on its audio encoder; the migrated worker does not (separate D-22 follow-up).
3. **AV chunk pacing** (`bg_574e831e`): `stream_fmp4` has only one timer (`av_encode_stream_ms`) — no per-chunk instrumentation. Between `worker.generate_step()` returning and `stream_fmp4()` returning there is no significant work other than ffmpeg's own encode + chunk emission. The `main_user_step − worker_e2e ≈ 1300ms` gap is therefore mostly ffmpeg + controller relay loop, NOT IPC. Per-chunk emission cadence is driven by stdout read size (1 MiB), not fragment boundary, so non-uniform pacing is plausible.

**Root cause synthesis:**

| Phase | Wall-time | Compute | Compiled? | NVFP4? |
|---|---|---|---|---|
| transformer denoise (gen) | ~4500 ms | GPU | YES | YES |
| save_conditioning | ~100 ms | GPU/CPU | n/a | n/a |
| **stream_fmp4 (libx264 software encode)** | **~1300 ms** | **CPU** | **NO** | **NO** |
| TOTAL wall | ~6000 ms | producing 4.67 s playable | | |

Realtime ratio = 4.67 s playable / 6.0 s wall = **0.78x**. FE buffer drains 1.3 s per segment until empty → inter-segment stutter. **The cause is not a migration regression** (FastVideo-internal exhibits the same architectural limit); it is the choice to use `libx264 ultrafast` software encoding on the segment-streaming hot path on machines that have idle hardware encoders.

**Build-time finding:** `apps/dreamverse/scripts/install_native_ffmpeg.sh` was building ffmpeg with only `--enable-libx264 --enable-lto`. No `--enable-nvenc`, no nv-codec-headers prereq. So even setting `FASTVIDEO_VIDEO_CODEC=h264_nvenc` would have failed at runtime with "encoder not found".

**Resolution (this round, default-preserving):**

1. **`apps/dreamverse/scripts/install_native_ffmpeg.sh`** now:
   - Clones `nv-codec-headers` (NVENC/NVDEC API headers; no CUDA libs) into `$INSTALL_PREFIX` so `ffnvcodec.pc` is on `pkg-config`'s search path.
   - Configures ffmpeg with `--enable-cuda --enable-nvenc --enable-cuvid --enable-nvdec` plus `--extra-cflags=-I$CUDA_PREFIX/include` and `--extra-ldflags=-L$CUDA_PREFIX/lib64`.
   - Drops `--enable-cuda-nvcc` and `--enable-libnpp` so the build does NOT require `--enable-nonfree` (we only need hardware encode/decode, not GPU-side filters).
   - New env knobs: `ENABLE_NVENC=0|1` (default 1), `CUDA_PREFIX` (default `/usr/local/cuda`), `NV_CODEC_REF`.
   - Sanity check verifies the resulting binary exposes `h264_nvenc` / `hevc_nvenc` encoders before exiting 0.
   - **Emits `FASTVIDEO_VIDEO_CODEC=libx264` in the env file by default** so existing deploys are runtime-unchanged.

2. **`.agents/skills/dreamverse-deploy/scripts/dreamverse-deploy.sh`** now:
   - Accepts `--nvenc` / `--no-nvenc` flag (defaults to off, matches existing behavior). When `--nvenc`, exports `FASTVIDEO_VIDEO_CODEC=h264_nvenc` into the backend setsid block.
   - Validates the binary actually has `h264_nvenc` encoder via `ffmpeg -encoders | grep h264_nvenc`. Fails fast if `--nvenc` is requested against a libx264-only ffmpeg with a clear hint to rerun the install script.
   - `DREAMVERSE_NVENC` env var as the env-var counterpart (flag overrides).
   - Banner now prints `nvenc=true|false` alongside `warmup` and `torch_compile`.

3. **`apps/dreamverse/server/benchmarks/benchmark_av_streaming.py`** is the regression test:
   - Drives `stream_fmp4` directly with synthetic 121-frame 1920x1088 video + 5-second 24kHz stereo audio (production shape).
   - Sweeps `libx264` and `h264_nvenc` (auto-skips encoders not present in the active ffmpeg).
   - Reports per-codec: wall_ms_min/median/p95/max, bytes_median, chunks_median, and **realtime_ratio_median** (playable / wall, must be ≥ 1.0 to avoid buffer drain).
   - Exit code 1 if any codec produces realtime_ratio < 1.0; useful as a CI gate.

**Empirical findings (this dev host, B200):**

- `libx264 ultrafast` benchmark on production-shape input (121 frames, 1920x1088, 5s 24kHz stereo audio): wall_median=611ms, wall_p95=644ms, 8.25x realtime ratio in isolation. Captured via `apps/dreamverse/server/benchmarks/benchmark_av_streaming.py --runs 5 --codecs libx264`.
- **B200 has NO NVENC silicon.** Direct ffmpeg probe (`-c:v h264_nvenc` against a 64x64 0.2s color frame) fails with `OpenEncodeSessionEx failed: unsupported device (2): No capable devices found`. This is a hardware omission — datacenter Blackwell (B200) and some H100 SKUs prioritize compute density and ship without NVENC. Only consumer Blackwell (RTX 50-series) and select datacenter SKUs (T4, A10, A100 PCIe) have NVENC.
- The deploy script's `--nvenc` path now does TWO checks: (a) `h264_nvenc` is in the encoder list (build-time); (b) a 64x64 ffmpeg probe actually succeeds (runtime-time). On a B200, the runtime probe fails fast with a clear "no NVENC silicon on this host" error pointing at SKU-level alternatives.
- **Stutter on B200 is NOT primarily ffmpeg encoding.** The benchmark shows libx264 ultrafast at 611ms, but production logs show a 1300ms gap between `worker_e2e` and `main_user_step`. The other ~700ms is IPC + controller WebSocket relay (not measured per-chunk in current `stream_fmp4`).

**Open / deferred:**

- ~~D-22~~ ✅ **Resolved 2026-05-06** in `bade2c0a` — `stream_fmp4` is now fully instrumented with per-phase + per-chunk timings; gpu_pool prints a per-segment summary line. The controller-side WS-send slice is tracked separately as D-22-CTL.
- D-22-CTL: per-chunk timing in the controller's AV relay loop in `session/controller.py` (between media event arrival and `ws_send_bytes`). Worker side is captured by D-22; the controller side is the still-unmeasured remainder of the 700ms IPC+relay gap.
- D-23: B200 + RTX 5090 split — for B200-class deploys without NVENC, the stutter fix needs a different approach (pipeline gen N+1 with encode N, or larger initial FE buffer pre-fill). For RTX 5090 / T4 / A10 deploys, `--nvenc` is the answer once tested. Make the deploy default conditional: probe NVENC at boot, default `--nvenc` if available.
- D-24: re-evaluate flipping the deploy default from `libx264` to `h264_nvenc` after a real-NVENC host (RTX 5090 or H100 PCIe) benchmark is run. Tradeoff: NVENC has ~5-10% lower compression at same quality but is hardware-accelerated. For real-time streaming the latency win dominates IF the host has NVENC.
- D-25: pipeline benchmark via Python SDK (`apps/dreamverse/server/benchmarks/benchmark_pipeline.py`, landed in `f98811e0`) — captures per-stage timings via `FASTVIDEO_STAGE_LOGGING=1`. Cold-run baseline on B200 NVFP4 (no compile, no warmup): 6.99s for 5.04s playable = 0.72x realtime. Per-stage profile dominated by LTX2RefineLoRAStage (40% on cold, includes one-time LoRA conversion + adapter load), LTX2DenoisingStage (20% — base DiT 5-step denoise), LTX2TextEncodingStage (11%), DecodingStage (4%), audio_decoding/upsample/init each <1%. Future work: rerun under `compile_warm` scenario for steady-state numbers.
- The `target.eval()` call on the audio encoder that FastVideo-internal does and the migrated Dreamverse does not — surface as a separate decision once observed empirically (probably a no-op in inference path but worth the symmetry).

**Cross-references:**

- [D-20](decisions-log.md#d-20) (audio kwarg routing fix) made segment 2 generate cleanly. D-21 makes the segment-to-segment chain stutter-free in real-time.
- [`apps/dreamverse/server/av_streaming.py::stream_fmp4`](file:///home/william5lin/FastVideo/apps/dreamverse/server/av_streaming.py) is the hot path; the cmd builder there respects `FASTVIDEO_VIDEO_CODEC` and branches on `*_nvenc` codecs to use NVENC presets (`p1`/`p2`/...) and `-rc constqp -qp 28` instead of libx264 presets (`ultrafast`/...).
- The benchmark cross-references D-21 in its module docstring so a future contributor reading the script alone has the context.

### D-20: Segment-2 BrokenPipe root cause — two-direction silent drop of LTX-2 audio kwargs in public `VideoGenerator`

**Status:** ✅ Resolved 2026-05-05. 4 commits on `will/dreamverse-monorepo` (`1b686f4e`..`5eaf0a13`); pushed to origin. End-to-end verified on GPU4 (`/proc/$BE_PID/environ` + `Cached audio latents shape=(1, 8, 126, 16) for segment 2` log line + `Segment 2: relayed av chunks=22, bytes=3.8MB`).
**Source:** Live debugging session triggered by recurring `RuntimeError: ffmpeg frame writer failed: [Errno 32] Broken pipe` on segment 2 in `/tmp/opencode/dreamverse-deploy/backend-gpu4.log`.

**Question:** Why does segment 2 of every Dreamverse session crash with a writer-side EPIPE, while segment 1 streams fine?

**Symptom chain decoded:**

1. ffmpeg writer thread in `apps/dreamverse/server/av_streaming.py:307-317` raises `BrokenPipeError` mid-frame.
2. ffmpeg's exit code is `0` — the `if rc != 0` branch of `stream_fmp4` never fires; only `if writer_error[0] is not None` does.
3. `rc=0 + BrokenPipeError` means ffmpeg exited cleanly *before* the writer finished pushing all frames → ffmpeg closed stdin early due to `-shortest` + audio shorter than video.
4. Manual repro confirmed: with audio 71240 samples (~2.97s) and video 112 frames (~4.67s), ffmpeg `-shortest` + 1MB pipe + 1920x1088x3 frames produces exactly this signature (`rc=0  out_bytes=751028  writer_error=BrokenPipeError(32)`).
5. The 1.7s audio undershoot is suspicious because Dreamverse's `apply_audio()` in `apps/dreamverse/server/video_generation.py:132-186` explicitly extends `audio_num_frames = NUM_FRAMES + audio_extra` (=`121 + 40 = 161`) for continuation segments. Audio should be 6.71s, not 5.01s. The kwarg was being silently dropped.

**Root cause (TWO directions):**

| Direction | Where | What was missing | What it broke |
|---|---|---|---|
| **Inbound** (kwargs → `batch.extra`) | `fastvideo/entrypoints/video_generator.py::_generate_video_impl` | The 5-key extraction block (`ltx2_audio_latents`, `ltx2_audio_clean_latent`, `ltx2_audio_denoise_mask`, `audio_num_frames`, `video_position_offset_sec`) that FastVideo-internal has at lines 168-183. Without it, `apply_audio`'s kwargs landed in `sampling_param.update(kwargs)` which `logger.error`'d "%s has no field %s" and dropped them. | `audio_num_frames` never reached `batch.extra` → `ltx2_denoising.py:325` fell back to `batch.num_frames=121` → audio generated for 5.01s instead of 6.71s. After `head_trim_audio_frames=49` removed the leading 2.04s, only 2.97s of audio remained for 4.67s of video. |
| **Outbound** (`batch.extra` → result dict) | `fastvideo/entrypoints/video_generator.py::_generate_single_video` | `"ltx2_audio_latents": output_batch.extra.get("ltx2_audio_latents")` in the result dict. Internal exposes it at line 556; public didn't expose it at all. | `Dreamverse._derive_next_audio_latents()` always saw `None` → `self.continuation.audio_latents` was never set → segment 2's `apply_audio` short-circuited (`if not (… and self.audio_latents is not None): return`) → no audio continuation, no `audio_num_frames` extension, no `ltx2_audio_clean_latent` carry-over. |

The two directions hid each other: even after the inbound block was ported, segment 2 still broke until the outbound surface was added. Both must be present for audio continuation to round-trip.

**Resolution — 4 commits on `will/dreamverse-monorepo`:**

| SHA | Subject |
|---|---|
| `1b686f4e` | `[fix] dreamverse: pin ffmpeg native build toolchain by uname -m` |
| `265ce1a6` | `[fix] api: route LTX-2 audio kwargs through batch.extra; strict update` |
| `dab9499c` | `[feat] dreamverse-deploy: native ffmpeg + compile-off defaults` |
| `5eaf0a13` | `[feat] dreamverse-deploy: --warmup / --torch-compile CLI flags` |

`265ce1a6` is the substantive public-API fix: ports `_BATCH_EXTRA_PASSTHROUGH_KEYS` extraction in `_generate_video_impl`, ports `_extra_overrides` consumption in `_generate_single_video` (writes into `batch.extra`), surfaces `ltx2_audio_latents` in the result dict, and converts `SamplingParam.update()` from `logger.error`-and-drop to `raise ValueError` on unknown keys so the next contributor who adds an unrecognized kwarg hits a loud failure pointing at `_BATCH_EXTRA_PASSTHROUGH_KEYS` instead of debugging a broken-pipe-shaped symptom hours later. Adds `fastvideo/tests/api/test_extra_overrides_routing.py` (7 tests pinning the contract).

**Why the strict-update is non-negotiable:** the `logger.error`-and-continue pattern is the exact mechanism that hid this bug for the entire Dreamverse-monorepo migration window. It silently traded loud-failure-now for silent-corruption-later. Strict raise + a clear "route via `_BATCH_EXTRA_PASSTHROUGH_KEYS`" hint converts the next regression of this shape from "broken pipe in production" to "ValueError at first call".

**Cross-references:**

- [D-11](decisions-log.md#d-11) (Apr 26) noted "ffmpeg fragment write Broken pipe" but framed it as cosmetic (client-disconnect race). Today's was a different code path: server-side EPIPE caused by a server-internal A/V duration mismatch. Both are now resolved; D-11's fix domain (swallow on intentional disconnect) remains unchanged but lower priority.

- The `[fix] api: route LTX-2 audio kwargs ...` commit (`265ce1a6`) is on `will/dreamverse-monorepo` only. Cherry-picking onto `will/ltx2_sr_port` so it lands as part of PR #1288's mega-PR is a deferred follow-up — see open-threads.md "Cherry-pick API audio routing fix to PR #1288".

**Side effects of the fix:**

- `[feat] dreamverse-deploy: native ffmpeg + compile-off defaults` (`dab9499c`) wires the native LTO+libx264+native-arch ffmpeg the team playbook mandates into every deploy via `FASTVIDEO_FFMPEG_BIN=$HOME/opt/ffmpeg-native/bin/ffmpeg` and disables `torch.compile` by default (`ENABLE_TORCH_COMPILE=0`) so segment-1 cold start drops from ~3-4min (max-autotune) to ~45s (pure inference) — required for any iterative debugging cycle to fit inside the 300s session timeout.
- `[fix] dreamverse: pin ffmpeg native build toolchain by uname -m` (`1b686f4e`) makes `apps/dreamverse/scripts/install_native_ffmpeg.sh` immune to conda envs that activate both `gcc_linux-64` and `gcc_linux-aarch64` (the aarch64 activation script sorts later and wins, so the inherited `CC=aarch64-conda-linux-gnu-cc` defeated the script's `: "${CC:=...}"` deferred default and tripped x264's compiler probe with "unknown value 'native' for '-march'").
- `[feat] dreamverse-deploy: --warmup / --torch-compile CLI flags` (`5eaf0a13`) adds `--warmup`/`--no-warmup` and `--torch-compile`/`--no-torch-compile` flags that override the env-var defaults and accept any position relative to the positional GPU/port args (verified across 13 parser permutations).

**Watch-outs for downstream contributors:**

- Adding a new pipeline-specific kwarg now requires either making it a `SamplingParam` field OR adding it to `_BATCH_EXTRA_PASSTHROUGH_KEYS` in `fastvideo/entrypoints/video_generator.py`. Strict `update()` will raise `ValueError` on unknown kwargs that aren't routed through one of those paths.
- The `[fix] api:` commit (`265ce1a6`) is BACKWARD-INCOMPATIBLE for any caller that was relying on `SamplingParam.update()` to silently swallow unknown keys. If pre-existing CI breaks elsewhere on this surface, the fix is to add the legitimate kwarg to `SamplingParam` or `_BATCH_EXTRA_PASSTHROUGH_KEYS` (not to revert the strict mode).

### D-19: D-18 executed — Dreamverse migration on `will/dreamverse-monorepo`, generic-merged files carried product-local

**Status:** ✅ Resolved 2026-05-05. 5 commits on top of `will/ltx2_sr_port` HEAD `fbd823df`. Branch pushed to origin at `c1fe5d4c`. e2e definitively passes.
**Source:** Execution of [integration-plan.md](integration-plan.md) (D-18 plan), with one significant deviation surfaced by Oracle review.

**Commits:**

| SHA | Message |
|---|---|
| `08828d96` | `[feat] dreamverse-monorepo: Phase 1 — skeleton + tooling` |
| `f3a863ba` | `[feat] dreamverse-monorepo: Phase 2 — backend move + import rewires` |
| `876f7eb3` | `[feat] dreamverse-monorepo: Phase 3 — frontend + public assets` |
| `1d47ede6` | `[fix] dreamverse-monorepo: carry generic-merged files product-local` |
| `c1fe5d4c` | `[fix] dreamverse-monorepo: entrypoint + audio re-encode + verification` |

Final stats: 164 files changed, +53294/-2 LOC, 31725 files under `apps/dreamverse/`.

**Question:** [integration-plan.md](integration-plan.md) Phase 2 instructed "DELETE generic-merged files (`gpu_pool.py`, `av_streaming.py`, `worker_ipc.py`, `mock_server.py`, `session_init_image.py`, `session_logger.py`) from the Dreamverse copy and rewire imports to public `fastvideo.entrypoints.streaming.*` substitutes". The first execution attempt followed this instruction. Was that the right call?

**Decision (forced by Oracle finding):** No. The "delete + import substitute" strategy assumed the public modules were API-compatible drop-ins for the Dreamverse-product modules. They are NOT. Public `fastvideo.entrypoints.streaming.GpuPool` is an abstract base class with `acquire/run/release/shutdown/health` semantics designed for a future Phase 4 streaming-runtime API. Dreamverse's product `GPUPool(gpu_ids).initialize()` has totally different shape (per-GPU subprocess workers with `slot.user_step / register_stream_queue / acquire(client_id, websocket) -> (gpu_id, slot)` semantics). Substituting one for the other made `apps/dreamverse/server/main.py:63` fail at boot with `TypeError: GpuPool() takes no arguments`.

**Fix:** Carry all 7 generic-merged files (gpu_pool, av_streaming, worker_ipc, mock_server, session_init_image, session_logger, server_entry) into `apps/dreamverse/server/` as PRODUCT-LOCAL — same status as the rest of the Dreamverse server tree. Imports stay flat (`from gpu_pool import GPUPool`, etc.), matching Dreamverse's existing sys-path-injection convention. Public `fastvideo/entrypoints/streaming/*` reverts cleanly to its `fbd823df` state — `git diff fbd823df..c1fe5d4c -- fastvideo/entrypoints/streaming/` is empty.

The "import public substitutes" promise becomes a future Phase 4 task: actually harmonize the APIs so Dreamverse can drop its product-local copies. Not in scope for this migration.

**Why the e2e initially gave a false positive:**

The first run of all 8 Playwright tests passed (5.1s) — Oracle caught that this was misleading. The `dreamverse-server` console script in `/home/william5lin/miniconda3/envs/fv-main/bin/dreamverse-server` was installed by a prior `pip install -e /home/william5lin/Dreamverse`, so `from server_entry import cli` resolved to `/home/william5lin/Dreamverse/server/...` (the canonical install), not `apps/dreamverse/server/...` (the migrated tree). The migrated code was never actually exercised.

**Fix to entrypoint resolution** (in `c1fe5d4c`): wrapper scripts at `apps/dreamverse/scripts/dreamverse-server` (and `dreamverse-mock-server`) explicitly do:

```bash
#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO_ROOT}/apps/dreamverse/server"
exec "${REPO_ROOT}/.venv/bin/python" main.py "$@"
```

This guarantees the migrated tree is what runs. Verified end-to-end via `/proc/$PID/cwd = /home/william5lin/FastVideo/apps/dreamverse/server` during the second e2e run. All 8 Playwright tests pass against this confirmed-migrated backend.

**Phase 0 environment prereqs (validated 2026-05-05):**

- `flashinfer-python` in FastVideo `.venv` — required for NVFP4 path. Without it, model load fails with `ImportError: NVFP4 quantization requires flashinfer`.
- `cerebras-cloud-sdk` and `openai` in `.venv` — required by the migrated prompt enhancer.
- For B200 / sm_100a + gcc-15 conda toolchain: nvcc rejects host compiler. Workaround:
  ```bash
  CUDAHOSTCXX=/usr/bin/g++-13
  NVCC_PREPEND_FLAGS="-ccbin /usr/bin/gcc-13 -allow-unsupported-compiler"
  ```
  Without these, flashinfer JIT compilation fails with `error: #error -- unsupported GNU version! gcc versions later than 14 are not supported!`.

These are **operator-side prerequisites**, not migration code defects. Documented in `apps/dreamverse/README.md` and `docs/contributing/dreamverse-development.md`.

**E2E evidence (live, post-fix-up):**

```
PID 179112 cwd: /home/william5lin/FastVideo/apps/dreamverse/server
/healthz → 200 {"status":"ok","service":"ltx2-streaming-backend",...}
/readyz  → 200 {"status":"ready","ready_gpu_workers":1,"total_gpus":1,...}
GPU4 mem: 50.9 GiB (NVFP4 model loaded)

Playwright (8/8 PASS in 5.1s):
  ✓ backend-health/healthz returns ok via the next.js rewrite (32ms)
  ✓ backend-health/readyz reports gpu pool state (11ms)
  ✓ backend-health/status endpoint exposes gpu pool snapshot (12ms)
  ✓ backend-health/prompt-system-config exposes operator-tunable prompts (15ms)
  ✓ backend-health/curated presets endpoint serves a non-empty list (devtools only) (19ms)
  ✓ frontend-shell/main page loads and exposes the FastVideo brand chip (1.4s)
  ✓ frontend-shell/composer hydrates with curated preset cards (1.3s)
  ✓ preset-prompt-generation/generates the first segment from a curated preset prompt (1.8s)
```

**Implications:**

- The integration-plan.md is **partially superseded** by D-19 outcome:
  - Phase 2's "DELETE generic-merged" prescription is invalid; replace with "carry product-local".
  - Phase 0 prereqs need the gcc-13 / `NVCC_PREPEND_FLAGS` workaround documented for B200 hosts.
  - Phase 4 (in a future PR) is now responsible for actually harmonizing public-vs-Dreamverse pool APIs so the carried product-local modules can be deleted.
- `dreamverse-mock-server` script under `apps/dreamverse/scripts/` is the only canonical launcher. The conda env's legacy `/home/william5lin/miniconda3/envs/fv-main/bin/dreamverse-server` should NOT be used (it points at canonical Dreamverse repo).
- The audio re-encode handling in `apps/dreamverse/server/video_generation.py` was carried per `c1fe5d4c`; verify the exact shape (restored from source vs deferred) when reading the commit.
- Once this branch lands as a PR + merges, the Dreamverse repo can be archived per [integration-plan.md](integration-plan.md) Phase 7.

**Open follow-ups:**

- Open a PR for `will/dreamverse-monorepo` (target main, base on `will/ltx2_sr_port` until #1288 merges).
- Re-Oracle the post-fix-up state to confirm the prior FAIL is now PASS.
- Eventually move audio re-encode into a public module (Phase 4) so the carried product file can be slimmed.
- Eventually do real Phase 4 API harmonization between Dreamverse pool and public `fastvideo.entrypoints.streaming.GpuPool` so the 7 carried product files can be deleted.

### D-18: Option B+ — Dreamverse becomes `apps/dreamverse/` subfolder under FastVideo

**Status:** ✅ Resolved 2026-05-05. [integration-plan.md](integration-plan.md) is the executable migration plan; [integration-review.md](integration-review.md) is deprecated but kept for drift audit + OSS precedents.
**Source:** User decision after reviewing [integration-review.md](integration-review.md)'s Option D recommendation.

**Question:** [integration-review.md](integration-review.md) recommended **Option D** — Dreamverse stays a separate repo, generic backend (streaming runtime, GPU pool, prompt enhancer, router) merges into `fastvideo.entrypoints.streaming.*`. The user reviewed this and chose a different shape: keep the generic-backend principle from Option D but ALSO move the Dreamverse FE + product server into FastVideo as a subfolder (`apps/dreamverse/`). Combination is "Option B+" (Option B layout with Option D's backend principle).

**Decision:** Option B+. Concrete shape:

- **One repo**: `hao-ai-lab/FastVideo`. Dreamverse repo gets archived after migration completes.
- **Python ML library** stays at root: `fastvideo/`, `fastvideo-kernel/`.
- **Generic backend** stays at `fastvideo.entrypoints.streaming.*` (already there per #1257/#1258/#1284/#1286/#1288).
- **Dreamverse product** moves into `apps/dreamverse/{server,web,prompts,serve_configs,scripts}/`.
- **Tooling**: uv workspace for Python (`[tool.uv.workspace] members = ["apps/dreamverse/server"]`), standalone pnpm for the FE (no root `package.json`), split CI workflows with path-filter triggers.

**Rationale:**

- Drops the cross-repo coordination overhead identified in the post-#1286 rebase cycle (D-17 handled by consolidating into mega-PR; D-18 prevents the next round of cross-repo coordination from happening).
- Keeps the architectural separation Option D recommended (FastVideo owns reusable runtime; product owns product). The boundary is now `apps/dreamverse/` directory rather than two repos.
- Single repo means atomic cross-cutting refactors (e.g. GpuPool API change + Dreamverse adoption) ship as one PR.
- OSS precedents support the shape (chainlit uv-workspace + pnpm; open-webui Python + Svelte with paths-ignore CI). The librarian explicitly noted no precedent for "Python ML library + Next.js product merged into library namespace" — but this isn't that pattern. Dreamverse goes into a sibling directory, NOT into `fastvideo.entrypoints.dreamverse.*`. Library namespace stays clean.

**Why not Option D (separate repos):**

- Each upstream merge into FastVideo invalidates Dreamverse's lockfile/imports; the post-#1286 rebase showed this requires coordination overhead that scales with feature velocity.
- Cross-repo contract tests catch shape drift but not behavior drift.
- Two repos means two `AGENTS.md`, two CI configs, two release stories, two Dependabot dashboards.

**Why not Option C (full merge into `fastvideo.entrypoints.dreamverse.*`):**

- Forces FastVideo to ship Tailwind config + curated preset JSON + Next.js build artifacts.
- Locks Dreamverse product cadence to FastVideo PyPI releases.
- Librarian: "no 1:1 precedent for Python ML library + Next.js product merged into library namespace" — argues against this.

**Why not Option B (subfolder, but generic backend folded into `apps/dreamverse/server/`):**

- Other consumers (Dynamo, future streaming clients) need the backend without the Dreamverse product. Folding the backend under `apps/dreamverse/server/` would force Dynamo to either depend on `apps/` paths (ugly) or carry a fork.

**Implications:**

- [integration-review.md](integration-review.md) is **deprecated** (banner header + reading-guide demotion). Kept in tree for drift audit + OSS precedent reference.
- [integration-plan.md](integration-plan.md) is the **canonical executable plan** with 7 phases (Phase 0: land #1288; Phase 1: skeleton + tooling; Phase 2: backend move; Phase 3: FE move; Phase 4: promote generic-pending; Phase 5: prompt enhancer fork retirement; Phase 6: CI/release cutover; Phase 7: archive Dreamverse repo).
- Dreamverse repo will be **archived** at end of Phase 7 — not before.
- Dreamverse history does NOT migrate cross-repo via `git mv` (technical limitation); original history stays in archived Dreamverse repo, and Phase 2 PR body records the source SHA(s).
- New top-level `apps/` directory created — must be excluded from FastVideo PyPI wheel via `[tool.setuptools.packages.find] exclude = ["apps*", ...]`.
- Drift items from [integration-review.md](integration-review.md) get folded into specific phases of [integration-plan.md](integration-plan.md) (e.g. health routes → Phase 4, DR-1 → Phase 5).

**Open questions deferred to phase planning:**

- DR-2 (`cerebras_ifm`): public Literal vs Dreamverse-side custom provider — decide before Phase 5.
- VPO (`video_position_offset_sec` semantics): persistent vs per-segment — decide in Phase 4.
- Cross-repo history: fresh import vs `git subtree` import — decide before Phase 2.
- CORS / write-endpoint security policy: dev-only vs auth vs firewall — decide before Phase 6.

### D-17: Abandon 6-PR split — land everything as single mega-PR #1288

**Status:** ✅ Resolved 2026-05-05. PR #1287 closed; PR #1288 opened on `will/ltx2_sr_port` covering the full chain.
**Source:** User decision after observing the post-#1286 rebase + re-slice cycle.

**Question:** The original plan ([STACK.md](../../../STACK.md), [pr-roadmap.md](pr-roadmap.md)) called for the remaining `will/ltx2_sr_port` content (after PRs 7.5/7.6/7.7/7.8/7.9 landed) to ship as 6 stacked PRs: 7.10 (#1287, generate_async), 8 (server contract docs), LTX-2 SR runtime, NVFP4, post-fixes, agents-cleanup. PR #1287 was opened on 2026-05-05 as the first slice. Should the remaining 5 slices be opened sequentially as planned, or should everything be consolidated into one PR?

**Decision:** Consolidate. Close #1287; open one mega-PR (#1288) on `will/ltx2_sr_port` covering all 34 commits / 71 files / +13,074 LOC at once.

**Rationale:**

- The post-#1286 rebase + re-slice cycle exposed real overhead: backup branch, interactive rebase with manual `drop` directives, force-push, re-slice 6 bookmarks, push next slice as new remote, open new PR, update memory dir. Repeating that 6 more times for the remaining slices accumulates substantial review-coordination overhead with diminishing structural benefit.
- The 6 layers are not independent in the way that landed PRs 7.5-7.9 were. PR 7.10 (`generate_async`) is the only API-shape change; PR 8 is docs+tests on top; LTX-2 SR / NVFP4 / post-fixes / agents-cleanup are feature/fix/docs work that doesn't shape the public API. Reviewing them as one ordered diff is at least as easy as reviewing 6 stacked PRs whose dependencies must be tracked manually.
- Single PR keeps CI / merge queue simpler and avoids the 6-PR cascade where every upstream merge invalidates the chain below it.

**Implications:**

- [STACK.md](../../../STACK.md) (top-level, 10-PR split tracker) is **deprecated**. Kept in tree as a historical artifact with the merged half (PRs 1-4 of the 10) accurate. Safe to delete in a follow-up.
- [authors.md](authors.md), [co-authors.md](co-authors.md) — co-author roster is unchanged; trailers still apply per-commit on every commit in the consolidated PR.
- [runbook.md](runbook.md) — "After a PR merges (re-slice protocol)" section replaced by a simpler "After PR #1288 merges" section.
- Local split bookmarks (`will/api_7.10`, `will/api_8`, `will/ltx2_sr_runtime`, `will/ltx2_nvfp4`, `will/ltx2_post_fixes`, `will/agents_cleanup`) are no longer maintained; safe to delete locally.
- `origin/will/api_7.10` — pushed during the #1287 cycle; can be deleted on origin once #1287 close-cleanup completes.

**Watch outs:**

- The PR is large (71 files, +13,074 LOC). Reviewers will need commit-by-commit review; the PR body structures the layers in commit order to make this tractable.
- If #1288 becomes too large to merge cleanly later (e.g. main moves significantly underneath it), the fallback is to re-split — but the current expectation is to land it as-is.

### D-12: `GpuPool` layer separation — keep distinct from `VideoGenerator`

**Status:** ✅ Resolved (interim) + 🟡 Deferred long-term shape to PR 7.10.
**Source:** Oracle review on 2026-05-04, post-PR-#1257 merge.

**Question:** Should `fastvideo.entrypoints.streaming.GpuPool` (PR #1257) be
folded into `fastvideo.entrypoints.video_generator.VideoGenerator`, or kept
separate? Three alternatives were evaluated:

| Alt | Approach | Verdict |
|---|---|---|
| A | Status quo — `VideoGenerator` (single inference call) and `GpuPool` (multi-session orchestration) stay separate | ✅ Correct as **interim** |
| B | `VideoGenerator` absorbs the pool's role (`from_pretrained_pool`, `acquire/release/run`) | ❌ **Wrong layer.** Conflates execution with serving scheduler. |
| C | `GpuPool` becomes a thin **session-aware async executor** over PR 7.10's `generate_async` | ✅ Correct **long-term destination** |

**Decision:** Alt A as interim; evolve toward Alt C once PR 7.10 lands
`generate_async`. Do NOT pursue Alt B.

**Rationale:**

- `VideoGenerator` is a library handle — "execute one request, possibly
  across ranks via `MultiprocExecutor`/`RayDistributedExecutor`."
- `GpuPool` is serving infrastructure — "schedule N concurrent sessions
  across N independent replicas, with sticky session-to-GPU affinity for
  cache locality."
- These are different layers driven by different consumers (a Python
  script doing `gen.generate(req)` vs. a WebSocket server with sticky
  sessions). Folding them muddies both surfaces.

**Key finding — `MultiprocExecutor` and `SubprocessGpuPool` are orthogonal,
not redundant:**

| Layer | Job | Granularity |
|---|---|---|
| `MultiprocExecutor` (`fastvideo/worker/`) | TP/SP shard ONE inference call across N GPU ranks | per-call |
| `streaming_generator.py` (existing real-time path) | Per-frame streaming via `MultiprocExecutor.submit_step`/`get_result` | per-step within one generator |
| `SubprocessGpuPool` (`entrypoints/streaming/`, PR #1257) | Serve N concurrent sessions on N replicas, sticky-bound | per-session |

Both spawn subprocesses because **CUDA contexts demand process boundaries**,
not because they solve the same problem. Sharing low-level lifecycle
utilities (process spawn, queue plumbing, shutdown) is a future refactor;
unifying the abstractions is wrong.

**Sticky binding stays in the pool, NOT in `VideoGenerator`:** sticky
session-to-GPU affinity is a serving policy driven by LTX-2's per-GPU
continuation cache (last-9-decoded-frames + audio-latents). Different
consumers want different policies — stateless OpenAI HTTP wants
per-request leasing; LTX-2 streaming wants sticky affinity; per-frame
real-time streaming wants a continuous queue. Keeping policy in the pool
keeps `VideoGenerator` policy-free.

**Specific risks flagged in PR #1257 (already merged):**

| Risk | Mitigation (when relevant) |
|---|---|
| `GpuPool.run() -> Any` is sync — fine for whole-segment dispatch, blocks on cancellation | Replace with `run_async() -> AsyncIterator[VideoEvent]` in PR 7.10 cycle (`generate_async` makes this trivial) |
| `PoolAssignment.gpu_id: int` assumes one-GPU-per-worker | Don't lock as public API. Future may need `device_ids: list[int]` for topology-aware pooling (one worker = group of GPUs running internal `MultiprocExecutor`) |
| `GpuPool` could be documented as the canonical FastVideo serving API | Mark as **experimental / server-internal** in docstring until PR 7.10 lands. Don't include in user-facing API docs yet |
| Memory: N processes = N model replicas (~10-50 GB each) | Expected for concurrent serving with crash isolation. CUDA IPC weight sharing loses isolation; CPU-shared-memory loading helps host RAM not device. Real scalable path is topology-aware pooling later. |

**Action items (carried into post-7.10 cycle):**

- [ ] Update `GpuPool` ABC docstring to note "API may change post-PR-7.10"
- [ ] Plan to replace `run()` with `run_async() -> AsyncIterator[VideoEvent]` in PR 7.10 cycle
- [ ] Don't promote `gpu_id: int` to public API; revisit shape post-7.10
- [ ] Consider clarifying field naming (e.g. `worker_id` is the stable identifier; `gpu_id` is current-impl detail)
- [ ] When opening 7.10's PR, have it consume `generate_async` from `GpuPool.run_async` end-to-end

**Open thread it touches:** PR 7.10 (`open-threads.md` item D — generate_async)
unblocks Alt C and is the natural place to land the API shape change.

### D-15: Streaming router (PR #1286) — keep in-repo, defer sticky / active-active

**Status:** ✅ Resolved (interim). Pre-merge polishes applied. Three follow-up
items tracked.
**Source:** Oracle review on 2026-05-05, during PR #1286 review cycle.

**Question:** Where should the multi-replica WebSocket router live? Should it
ship at all (vs. delegating to nginx/envoy)? Should sticky session routing
or weighted/round-robin balancing be in the initial PR?

| Alt | Approach | Verdict |
|---|---|---|
| A | Status quo — `fastvideo/entrypoints/streaming/router/`, FastAPI-based, single-primary failover, lazy `httpx`/`websockets` imports | ✅ **Keep** |
| B | Move to separate package `fastvideo-router/` | ❌ **Premature** — adds packaging/release/compat overhead before evidence of independent adoption |
| C | Fold router into the streaming server itself (one app, mode flag) | ❌ Conflates router/generator lifecycles, mode-dependent config, drags inference deps into routing deployments |
| D | Replace with reverse proxy (nginx/envoy/HAProxy) recipes | ❌ Not as the SOLE answer — mature proxies don't naturally emit FastVideo typed `gpu_unavailable` frames or evolve with FastVideo session semantics. Recommend external proxies as a complement at high scale. |
| E | Add sticky session routing now | ❌ **Defer** — implementing correctly depends on where `session_id` is available (URL/header is easy, first JSON frame is invasive). Reconnects are rare today. |
| F | Add weighted / round-robin now | ❌ **Defer** — active-active without sticky routing is worse for LTX-2 continuation locality than active-passive failover |

**Decision:** Alt A — keep current shape. Apply pre-merge polishes; preserve
forward-compat for sticky routing.

**Rationale:**

- Python router is justified as a FastVideo-aware control-plane component,
  not a replacement for Envoy/HAProxy. It can emit typed
  `gpu_unavailable` frames, evolve with FastVideo session semantics,
  and ship local/dev deployment without ceremony.
- The current abstraction is small + testable: `RouterConfig`,
  `ReplicaRegistry`, `ReplicaStatus`, `HttpProbe` (Protocol/structural alias).
  Adding strategy registries / telemetry interfaces / active-active policies
  now would be over-engineering.
- Active-passive (single primary) is the right MVP for LTX-2 streaming —
  preserves continuation cache locality (D-12 sticky binding rationale)
  better than naive active-active.
- The biggest architectural risk isn't placement; it's accidentally baking
  in unstated semantics. Define single-primary behavior + config validation
  now so future active-active or sticky routing becomes additive.

**Pre-merge polishes applied (per gemini + Oracle review):**

| # | What | Why |
|---|---|---|
| 1 | `ReplicaRegistry.select()` docstring rewrite | gemini flagged "round-robin via insertion order" claim was misleading — implementation always returns `[0]`. Replaced with explicit "first healthy primary, else first healthy non-primary; this MVP picks first match within tier; round-robin/weighted deferred". |
| 2 | Refactored `run_health_check_loop` to share single `httpx.AsyncClient` across the loop's lifetime via `_build_default_probe()` async context manager | gemini flagged per-probe client instantiation as inefficient. With ~1 probe/second default polling, TCP/TLS handshake overhead is non-trivial; now reuses connection. Tests inject probes directly so the path stays bypassable. |
| 3 | Probe all replicas concurrently per cycle via `asyncio.gather(..., return_exceptions=True)` | gemini flagged sequential probes risk falling behind `health_check_interval_seconds` if replicas time out. Now per-cycle wall time = max(probe latencies), not sum. |
| 4 | `RouterConfig.__post_init__` validation | Oracle recommended: empty replicas, non-positive intervals/timeouts, thresholds < 1, non-`http(s)://` URLs, and >1 primary all `raise ValueError`. Surfaces misconfiguration at config-load instead of confusing runtime failures. |
| 5 | Migrated `@app.on_event("startup"/"shutdown")` to `@contextlib.asynccontextmanager`-based `_lifespan()` | Pre-merge — FastAPI deprecated the old API. Was tracked as the 7.9 caveat in pr-roadmap.md. |

**One review comment intentionally not implemented:**

| Comment | Decision |
|---|---|
| gemini medium: `_load_router_config` duplicates `fastvideo.api.parser.parse_config` logic | Kept manual flat-from-nested mapping. The YAML schema has nested `health_check:` block but `RouterConfig` is flat; using `parse_config` directly would require either restructuring `RouterConfig` to have a nested `HealthCheckConfig` (schema change beyond this PR's scope) or accepting incomplete parsing. Manual mapping is intentional and well-typed. |

All 4 review threads marked resolved on the GitHub PR.

**Action items (deferred):**

- [ ] Track sticky session routing extensibility — when needed, add
  `ReplicaRegistry.select(routing_key: str | None = None)` so registry
  evolution is additive; document upfront where `session_id` should
  appear (URL/header preferred over first JSON frame to avoid
  buffering/peeking)
- [ ] Track `_bridge_session()` backpressure note — fine for MVP because
  `websockets` library provides basic transport backpressure, but at
  high scale add max_size/timeouts or recommend Envoy/HAProxy in front
- [ ] If active-active multi-primary becomes a requirement, define
  behavior (round-robin within healthy primaries, weighted, sticky-by-key)
  rather than letting `select()` silently pick `[0]`

**Watch outs:**

- `session_id` in WebSocket URL/headers is the cleanest sticky-routing
  hook. If it ends up only in the first JSON message, sticky routing
  later will require buffering/peeking before backend selection.
- Multi-primary configs are now explicitly rejected by validation;
  documented + enforced.
- `_bridge_session()` is fine for MVP (the libraries provide basic
  backpressure), but not production-grade for edge load. Document the
  limit.

**Open thread it touches:** open-threads.md items #13 (sticky routing),
#14 (bridge backpressure), #15 (multi-primary semantics).

### D-16: Streaming router polish round 2 — second-pass fixes on top of D-15

**Status:** ✅ Resolved. Applied as `[fix] streaming: router polish — bridge
cancel + state machine + deps` (`a152cb77` on `will/api_7.9`, `40e265b8` on
`will/ltx2_sr_port`).
**Source:** Second-pass review on PR #1286, 2026-05-05, after D-15's pre-merge
polishes landed.

**Question:** D-15 closed the structural review (placement, sticky/active-active
deferral, basic `__post_init__` validation). On a second pass through the same
files, five latent issues surfaced that weren't covered by gemini's first pass
or Oracle's structural review. Apply them on top of the merged D-15 polishes,
or queue for a follow-up PR?

**Decision:** Apply on top of `will/api_7.9` directly. All five are bug-class
or DX-class — none are scope-expanding architecture changes — so folding them
into PR #1286 keeps the router landing in one reviewable unit instead of
shipping a router PR plus an immediate follow-up fix PR.

**Fixes applied:**

| # | File | What | Why |
|---|---|---|---|
| 1 | `router/main.py::_bridge_session` | Replaced `asyncio.gather()` with `wait(FIRST_COMPLETED)` + explicit `cancel()`/drain + `_is_normal_disconnect()` classifier | `gather` waited for both directions; on client disconnect, the backend-reader task leaked and stayed pending. Backend `ConnectionClosed` also surfaced as an unhandled exception in server logs. New shape: first task to finish triggers explicit cancel of the other, both are drained, and only non-routine exceptions re-raise. |
| 2 | `router/registry.py::record_success` | Split state transitions: `UNKNOWN -> HEALTHY` is now immediate on first successful probe; only `UNHEALTHY -> HEALTHY` remains gated by `recovery_threshold` | Previously a fresh registry needed `recovery_threshold` consecutive successes before any replica was selectable. With default `recovery_threshold=2` and `health_check_interval=1s`, that meant 2-3s of `gpu_unavailable` rejections at startup. Now the first probe promotes immediately; recovery gating still protects against flapping replicas. |
| 3 | `router/registry.py::_build_default_probe` | Missing `httpx` now raises `RuntimeError` with install hint instead of yielding a "disabled" probe stub | Previous behavior: silently returned `(0.0, "httpx not installed; ...")` for every probe, which `record_failure` then folded into `UNHEALTHY` after `failure_threshold` cycles. Operators saw replicas drop UNHEALTHY with a confusing reason and no clear remediation. Hard-fail at startup is the right surface. |
| 4 | `router/config.py::__post_init__` | Extended D-15 polish #4 with: rejects `urlparse(url).path not in ("", "/")`, rejects `query`/`fragment`, rejects duplicate URLs across replicas | D-15's validation rejected non-`http(s)://` URLs and >1 primary; it didn't catch `http://host/api` (the router appends `/health` and `/v1/stream` itself, so a base-URL with path yields malformed routes) or `[{url: x}, {url: x}]` (replica registry keys by URL — duplicates would silently collapse to one entry, masking the misconfiguration). |
| 5 | `cli/router_serve.py::_load_router_config` | Replaced silent list-comprehension filter (`for r in replicas_raw if isinstance(r, dict) and r.get("url")`) with per-index `raise ValueError` | Original parser silently dropped malformed YAML entries. A single typo in `replicas[2].url` would yield 2 replicas instead of 3 with no log line. New shape: explicit per-index error message ("missing required key 'url'", "must be a mapping"). |
| 6 | `pyproject.toml::[streaming]` extra | Added `websockets` as explicit dep | `router/main.py::_bridge_session` does `import websockets` lazily and raises `RuntimeError` if missing. The `[streaming]` extra was an implicit transitive — anyone installing only `[streaming]` (and not the broader requirements) hit the runtime error. Now explicit. |

**Tests added (7 cases in `fastvideo/tests/entrypoints/streaming/test_router.py`):**

- `TestUnknownToHealthyImmediate.test_first_success_promotes_unknown` — first probe success transitions `UNKNOWN -> HEALTHY` regardless of `recovery_threshold`
- `TestUnknownToHealthyImmediate.test_unhealthy_recovery_still_gated_by_threshold` — `UNHEALTHY -> HEALTHY` still requires `recovery_threshold` successes
- `TestConfigValidation.test_rejects_path_in_url` / `test_rejects_query_in_url` / `test_rejects_fragment_in_url` / `test_rejects_duplicate_urls` / `test_accepts_trailing_slash` — `__post_init__` URL validation matrix

**Verification:** 17/17 router tests pass on both branches. `pre-commit run`
clean (yapf / ruff / codespell / mypy). `lsp_diagnostics` clean on changed
regions; the one pre-existing `Task` generic-type warning at `main.py:37` is
unrelated and predates this commit.

**In-flight pre-commit corrections (not part of the 6 fixes themselves):**

- yapf auto-reformatted 4 files (kept verbatim).
- ruff `UP038`: rewrote `isinstance(exc, (CancelledError, WebSocketDisconnect))`
  to `isinstance(exc, CancelledError | WebSocketDisconnect)`.
- mypy `[misc]`: renamed loop var `exc` (inside `for task in done`) to
  `task_exc` to avoid name collision with the outer
  `except ImportError as exc` binding.

**Open thread it touches:** None new. Item #14 (bridge backpressure) and
item #13 (sticky routing) from D-15 remain deferred — this round addressed
**cancellation/disconnect** semantics on the bridge, which is distinct from
**throughput backpressure**. Item #14 still applies: at higher load, add
`_bridge_session()` max-size + timeout limits or recommend Envoy/HAProxy
in front.

### D-14: Streaming auxiliaries (PR #1284) — cohesion + concrete-vs-Protocol scoping

**Status:** ✅ Resolved (interim). Two polish items applied during review; one
operational caveat tracked.
**Source:** Oracle review on 2026-05-04, during PR #1284 review cycle.

**Question:** Is PR #1284's bundle of 4 streaming-server auxiliary modules
(`prompt/safety.py`, `prompt/rewrite.py`, `session_logger.py`,
`mock_server.py`) correctly scoped? Should `mock_server` live in production
module path? Should `PromptSafetyFilter` be a Protocol? Should the bundle
have been split into 4 PRs?

| Alt | Approach | Verdict |
|---|---|---|
| A | Status quo — single PR, 4 modules under `streaming/`, mock_server in production path, concrete safety filter | ✅ **Keep** |
| B | Split into 4 separate PRs | ❌ Process overhead, not architectural improvement |
| C | Move `mock_server.py` into `tests/` | ❌ Would reduce discoverability + install-time usability of `python -m fastvideo.entrypoints.streaming.mock_server` |
| D | Move `session_logger.py` to `streaming/observability/` (or top-level `fastvideo/observability/`) | ❌ Premature — currently session-shaped + streaming-specific; promote when a non-streaming consumer appears |
| E | Convert `PromptSafetyFilter` to Protocol (like `LLMProvider`) | ❌ Premature abstraction — only one classifier exists; small duck-typed surface preserves future Protocol introduction without breaking the concrete |
| F | Convert `MockGenerator` to Protocol | ❌ Same — small duck-typed surface; no second mock generator exists |

**Decision:** Alt A — keep current shape. Apply two polish items from
Oracle's review before merge.

**Rationale:**

- "Streaming-server auxiliaries" is cohesive enough at 730 LOC with
  isolated modules + tests. Each module has independent code path but
  shared deployment context (the streaming server boots them all).
- `mock_server.py` in production path is a strength: reuses
  `build_app()` for protocol parity. Hiding it under `tests/` would lose
  `python -m fastvideo.entrypoints.streaming.mock_server` CLI access for
  FE devs.
- Concrete `PromptSafetyFilter` matches "ship what we have, abstract
  later" pattern. Internal had multi-classifier composition; public
  ships single + leaves chaining as a Dreamverse-side concern (per D-2).
- Same pattern for `MockGenerator`: small duck-typed `_GeneratorLike`
  surface lets a second mock implementation drop in without inheritance.
- `threading.Lock` (not `asyncio.Lock`) in `session_logger.py` is
  correct — writes come from real encoder/control threads via
  `run_in_executor`, not from coroutines directly. `asyncio.Lock` would
  be the wrong primitive for cross-thread concurrency.

**Pre-merge polishes applied (per Oracle):**

| Polish | What | Why |
|---|---|---|
| 1 | Removed `RewriteOptions.user_system_prompt_override` | Inert public field — was declared but never threaded through to `enhancer.rewrite()`. Shipping unused public options is more likely to bite than any structural choice. Re-add when actually wired through. |
| 2 | Sanitized `session_id` filename in `session_logger.SessionLogger._get_file()` | Defense-in-depth: today session_id is server-generated UUID, but a future code path that accepts client-supplied ids would otherwise allow path traversal via `../`. Added `_FILENAME_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]")` + sub before `os.path.join`. |

**Operational caveat tracked (not a code change):**

- `SafetyDecision.UNAVAILABLE` is treated as `ALLOW` by callers — a
  policy choice that's correct for an opt-in safety filter, but
  callers should log loudly so operators know the filter is degraded.
  Tracked as open-threads.md item #12.

**Pre-merge review feedback (4 of 4 resolved on the GitHub PR):**

| # | File:Line | Severity | Issue | Fix applied |
|---|---|---|---|---|
| 1 | `session_logger.py:57` | High | `log()` race vs `close()` — `KeyError` on `_locks[session_id]` | Atomic capture in `_get_file()`; master `_registry_lock`; `with lock, contextlib.suppress(ValueError):` |
| 2 | `rewrite.py:71` | Medium | `re.compile()` in hot path | Module-level `_LEADING_MARKER_RE`, top-level `import re` |
| 3 | `safety.py:105` | Medium | `_ensure_loaded()` race on concurrent fastText load | `_load_lock = threading.Lock()` + double-check pattern |
| 4 | `pyproject.toml:145` | Medium | `streaming` extra missing `prompt-safety` | Added to aggregator |

All 4 review threads marked resolved via GraphQL `resolveReviewThread`.

**Action items (deferred):**

- [ ] Track `SafetyDecision.UNAVAILABLE` log loudness in
  open-threads.md item #12 — when streaming server starts using the
  safety filter, ensure operator-visible logging on `UNAVAILABLE`
  results
- [ ] If a second safety classifier appears (Perspective API, Detoxify,
  custom rules), promote `PromptSafetyFilter` to a Protocol — same
  pattern as `LLMProvider` per D-13
- [ ] If a second mock generator appears (different frame patterns,
  different latency models), promote `MockGenerator` to a Protocol

**Open thread it touches:** PR #1284 itself; future safety-classifier
Protocol promotion; future observability module extraction.

### D-13: Prompt enhancer / `LLMProvider` abstraction shape — keep streaming-scoped

**Status:** ✅ Resolved (interim) + 🟡 Three deferred polishes after metrics or 2nd consumer.
**Source:** Oracle review on 2026-05-04, pre-PR-#1258-merge.

**Question:** Is PR #1258's `fastvideo.entrypoints.streaming.prompt.*` module
correctly designed? Should it be (a) Protocol-based vs ABC, (b) under
`streaming/` vs top-level `fastvideo.prompt.*`, (c) closed 3-op enum vs
open `complete()` API?

| Alt | Approach | Verdict |
|---|---|---|
| A | Status quo — `streaming/prompt/*`, Protocol provider, fixed 3 ops, lazy `httpx`, per-call `AsyncClient` | ✅ **Keep** |
| B | Move to top-level `fastvideo.prompt.*` (decouple from streaming) | ❌ **Premature.** No second consumer exists yet. |
| C | Convert `LLMProvider` Protocol → ABC with default impls + retry classification | ❌ **Wrong direction.** Biases extension toward OpenAI shape; `_openai_compat.py` already factors that as helper not inheritance. |

**Decision:** Alt A as interim. Promote to Alt B only when a second
non-streaming consumer (OpenAI server, batch generation, tooling) actually
needs the prompt enhancer. Don't pursue Alt C.

**Rationale:**

- Public contract is tiny — `name: str` + `async complete(LLMRequest) -> LLMResponse`. ABC adds zero value.
- `_openai_compat.py` is the right place for shared logic — helper, not base class. Anthropic / local / custom providers stay first-class.
- The 3 ops (enhance / auto_extend / rewrite) are LTX-2 streaming concepts. `auto_extend` (continue prompt sequence) and `rewrite` (multi-line alternatives) come directly from session UX. Calling this "the FastVideo prompt API" misrepresents that.

**Specific risks flagged in PR #1258 (already merged-pending review):**

| Risk | Mitigation (when relevant) |
|---|---|
| API publicity — calling this "the FastVideo prompt API" before a second consumer exists | Document module as "streaming-server prompt enhancement" in user-facing docs; keep it nested under `entrypoints/streaming/` |
| `httpx.AsyncClient` per-call (no connection pooling) | Acceptable for ~6-10 calls per LTX-2 session; LLM latency dominates. Add optional `client_factory` parameter LATER if metrics show connect overhead is meaningful. |
| 3 fixed operations could constrain future generic use | Closed enum is right for application-level orchestration. Future generic consumers should either call `provider.complete()` directly, or get a thin separate enhancer that shares the provider/fallback machinery. |
| `register_provider(priority=-1)` semantics rely on Python's negative-index `list.insert` | Cosmetic concern; docstring is clear. Could be tightened to explicit branch later. |
| `runtime_checkable` Protocol with `name: str` instance attribute — static type checkers may miss missing `name` | Acceptable; runtime check via `isinstance(p, LLMProvider)` works for plugin discovery. |

**Action items (deferred):**

- [ ] Document `fastvideo.entrypoints.streaming.prompt.*` as streaming-scoped in user-facing docs (PR 12 docs migration); avoid promoting as framework-level
- [ ] Add optional `client_factory` parameter to providers when metrics justify pooling
- [ ] Plan future move to `fastvideo.prompt.*` (with import shim) when second non-streaming consumer materializes
- [ ] Track Q-2 reactivation: promote LTX-2 prompt orchestration (locked segments, segment-prompts JSON parsing) to public `fastvideo.entrypoints.streaming.prompt.ltx2_orchestration` when a second LTX-2-style consumer appears

**Open thread it touches:** Dreamverse migration (open-threads.md DR-1)
will be the first real test of the public surface. Lessons learned there
inform whether Alt B becomes feasible.

## D-decisions (from `dreamverse_review.md`, Apr 26)

### D-1: Realtime runtime → streaming GpuPool migration shape

**Status:** ✅ Resolved.

Internal `RealtimeRuntimeConfig` had a multi-model registry +
flattened sampling defaults. Public `SubprocessGpuPool` is single-model
+ uses per-request `SamplingConfig`.

**Decision:** Drop multi-model registry on integration branch (not used
in production). Construct `GeneratorConfig` for chosen model and pass to
`SubprocessGpuPool`. Move sampling defaults to a server-side
`default_request: GenerationRequest` template.

**Risk:** Migration branch surfaces missing-model errors if a flow
silently relied on registry to swap models per-session. Integration
tests exercise at least one segment per supported model id before
merging.

### D-2: PR 7.7 prompt enhancer API surface narrower than internal

**Status:** ✅ Resolved.

Public `PromptEnhancer.enhance/auto_extend/rewrite` returns
`LLMResponse(content, provider, model, latency_ms, fallback_used)`.
Internal returns `EnhanceResult(prompt, fallback_used, error, ...)` /
`RewriteResult(prompts, ..., rollout_id, rollout_label, ...)`.

**Decision:** Adapt at the call site via
`Dreamverse/server/prompting/_internal_compat.py` shim. Locked-segment /
next-segment-index plumbing stays Dreamverse-side. Public stays minimal
and provider-agnostic.

**Open question (Q-2):** Promote LTX-2-specific orchestration into
`fastvideo.entrypoints.streaming.prompt.ltx2_orchestration` once a
second consumer appears. Logged for future review.

### D-3: Multi-stage provider race vs. sequential fallback

**Status:** ✅ Resolved (public stays sequential).

Internal enhancer runs all providers in a stage in parallel
(`_run_provider_race`). Public enhancer runs sequentially with
retryable-error fallback.

**Decision:** Public stays sequential for PR 7.7. Race is a
Dreamverse-specific tail-latency optimization that depends on parallel
API budgets.

**Risk / Q-3:** First-segment latency on Dreamverse may regress
slightly when Cerebras has a bad minute (sequential waits 20s before
trying Groq). If real production concern, add public
`concurrency: int = 1` knob behind a race path — but only after measuring.

### D-4: Skip PR 7.9 router for the integration branch

**Status:** ✅ Resolved.

Internal stack ships `router/main.py` for multi-replica load balancing.
Dreamverse deployment uses single replica per region.

**Decision:** Land PR 7.9 publicly (upstream the surface). Skip wiring
into Dreamverse integration branch. Dreamverse's `server/main.py` does
not import from `router/`.

### D-5: Audio re-encode (PR 7.10) needed for streaming, deferred

**Status:** 🟡 Deferred to PR 7.10.

Internal streaming server's per-step path runs `_re_encode_audio` inside
`_stream_av_fmp4_events` so each fMP4 segment ships with
continuation-conditioning audio. Whole-segment `pool.run()` path doesn't
need this.

**Decision:** Land PR 7.10's `generate_async` publicly. Dreamverse
integration branch initially keeps using `pool.run()` (whole segment, no
re-encode). Follow-up branch swaps to `generate_async` + audio re-encode.

**Open question (Q-5):** Acceptable for first switch, or does
Dreamverse audio quality regress vs. internal until 7.10 wires in?

### D-6: `realtime/local_runtime.py` is NOT upstreamed

**Status:** ✅ Resolved.

It was the FastVideo-internal precursor to `streaming.gpu_pool`.
Upstreaming both would create two GPU pool implementations in public.

**Decision:** Don't upstream `realtime/local_runtime.py`. Dreamverse
switches to `streaming.gpu_pool.SubprocessGpuPool` on integration
branch. Internal module can be deleted at follow-up.

### D-7 / Q-6: `FP4Config` is private-only

**Status:** ✅ **Resolved May 2.**

April 26: `Dreamverse/server/video_generation.py:271` imported
`fastvideo.layers.quantization.fp4_config.FP4Config` from
FastVideo-internal only. The 411-line module hard-imported `flashinfer`.

**Two options at the time:**

1. Colocate publicly with `flashinfer` as optional extra
   `pip install fastvideo[fp4]`; refactor `FP4QuantizeMethod` to take
   layer-prefix list from a pipeline-config field instead of hardcoding
   ltx2 paths.
2. Keep private — Dreamverse imports from internal via thin shim.

**Recommendation at the time:** option 1 once API refactor settles.

**Resolution:** May 2 work chose option 1.
- `365a66c7` upstreamed FP4Config with lazy `flashinfer` import in
  loader helper (no public hard-dep)
- `94c983a2` renamed FP4 → NVFP4 to disambiguate from MX-FP4 / OCP-FP4
- `42b30bf9` wired through `fastvideo.layers.quantization`

See [quantization.md](quantization.md) for full details.

### D-8: `ltx2_image_crf` silently dropped by public schema

**Status:** 🔴 **Unverified post-`d80c2a8`.**

April 26: Dreamverse's `server/video_generation.py:406` passed
`ltx2_image_crf=0.0` to `SamplingParam(...)`. Public
`fastvideo.api.sampling_param.SamplingParam` did NOT have this field;
the BE logged ERROR and silently dropped the kwarg.

**Migration target** (per [design.md](design.md) compatibility map):
`request.stage_overrides.refine.image_crf`.

**Resolution status:** `d80c2a8` (May 2) refactored
`server/video_generation.py` to use typed `GeneratorConfig` +
`preset_overrides["refine"]`. Whether this PR routed `image_crf`
through the typed `stage_overrides` path or left it silently dropped is
unverified. See [open-threads.md](open-threads.md).

### D-9: `aarch64-conda-linux-gnu-cc` triton compile failure

**Status:** ✅ Resolved (operational).

Conda env injected an ARM cross-compiler ahead of `gcc` on `$PATH`, so
`torch._inductor`'s triton launcher failed compilation. Setting
`ENABLE_TORCH_COMPILE=0` bypasses it.

**Long-term fix:** clean conda env's compiler shadowing or add
`CC=gcc` override in Dreamverse's worker bootstrap.

### D-10: Warmup OOM on shared GPU

**Status:** ✅ Resolved (operational).

When `CUDA_VISIBLE_DEVICES` lands on a GPU another tenant uses, LTX-2
warmup fails with OOM. Picking an idle GPU (4-7 in test setup) is a
manual step.

**Improvement:** pre-warm probe that checks free memory before booting
the pool would prevent this.

### D-11: ffmpeg fragment write `Broken pipe`

**Status:** ✅ Resolved (cosmetic).

When WS client closes before backend finishes streaming first segment,
ffmpeg hits `[Errno 32] Broken pipe`. Currently propagates to
"User step failed". Cosmetic — swallowing pipe-broken on intentional
disconnect would clean up logs.

## Q-questions (from `streaming-server-upstream-plan.md`, Apr 17)

### Q-1: Router placement (in-repo or separate package)

**Status:** ✅ Resolved (in-tree).

**Recommendation at the time:** separate package `fastvideo-router/` or
`fastvideo/contrib/router/`; defer final call to PR 7.9.

**Resolution:** PR 7.9 implementation places router in-tree at
`fastvideo/entrypoints/streaming/router/`.

### Q-2: Session ID authority

**Status:** ✅ Resolved (server-generated).

**Recommendation:** server-generated UUID; accept externally provided
session ID only for resume flows.

### Q-3: Torch compile kwargs typing (opaque vs full vs hybrid)

**Status:** ✅ Resolved (hybrid).

**Recommendation:** hybrid — type the common four (`backend`,
`fullgraph`, `mode`, `dynamic`) + allow `extras: dict[str, Any]`.

**Resolution:** PR 6 + NVFP4 `221cb20a` shipped exactly this hybrid.

### Q-4: Prompt safety / fasttext dependency

**Status:** ✅ Resolved (optional extra).

**Recommendation:** ship as optional extra `pip install fastvideo[prompt-safety]`.

**Resolution:** PR 7.8 implements as optional extra.

### Q-5: Audio-specific tensor payloads in continuation

**Status:** ✅ Resolved (typed `LTX2ContinuationState`).

`ltx2_audio_clean_latent`, `ltx2_audio_denoise_mask`,
`ltx2_audio_latents` not in pre-refactor public schema.

**Recommendation:** classify as opaque fields inside
`LTX2ContinuationState.payload`, not top-level sampling fields.

**Resolution:** PR 7's typed `LTX2ContinuationState` lifts these into
typed fields (see [cross-repo-surfaces.md](cross-repo-surfaces.md)
field mapping table).

### Q-6: Dynamo subpackage home

**Status:** ✅ Resolved (lives in Dynamo repo).

**Resolution:** No Dynamo code in FastVideo. Full backend package
(handler, adapter, registration, health check) owned by Dynamo repo at
`components/src/dynamo/fastvideo/`, same pattern as vllm/sglang.
FastVideo only guarantees the public API contract.

### Q-7 (was Q-6 in dreamverse_review): How to land FP4Config publicly

**Status:** ✅ Resolved May 2 — option 1 (colocate publicly).

See D-7 above.

### Q-8: Disaggregation readiness contract test

**Status:** 🟡 Recommended; not yet shipped.

PR ai-dynamo/dynamo#7544 is aggregated-only. `ContinuationState` hybrid
already supports future prefill/decode split.

**Recommendation:** PR 7.10 explicitly validate `ContinuationState`
survives round-trip through Dynamo-style RPC (pickle or JSON), even
though Dynamo isn't using it today. Cheap regression guard.

### Q-9: Dynamo progress/status passthrough

**Status:** 🟡 Deferred until Dynamo clarifies.

`NvVideosResponse` has `status` and `progress` fields.

**Recommendation:** PR 7.10 stays aggregated-final-only to match PR
#7544 shape; revisit after Dynamo clarifies their streaming/progress
semantics.

## Cross-doc questions still 🔴 OPEN

These need decisions; tracked also in [open-threads.md](open-threads.md):

| ID | Question | Source | Why it matters |
|---|---|---|---|
| **D-8** | Did `d80c2a8` route `ltx2_image_crf` correctly, or is it still silently dropped? | dreamverse_review | Latent silent-drop bug; FP4-disabled paths may degrade |
| **VPO** | `video_position_offset_sec` — persistent accumulation (a) vs per-segment hint (b) | dreamverse_integration | Needs decision before PR 7.6 emits state |
| **SBS** | `SessionStore` / `BlobStore` lifecycle (TTL/eviction/blob-drop on state replacement) | dreamverse_integration | Needs decision in PR 7.5 design pass |
| **#1** | Migrate `/healthz`+`/readyz`+`/status` into FastVideo `build_app` | streaming-upstream-plan + handoff | Closes BE_FLAVOR=fastvideo FE-compatibility |
| **#3** | Add `cerebras_ifm` to public `PromptEnhancerConfig.provider` Literal | handoff | Internal supports it; public schema doesn't |
| **#4** | Expose `layer_profile` on typed `engine.quantization` | handoff | Removes Dreamverse's `experimental["pipeline_config"]` dodge |
| **#5** | Typed `dit_config.quant_config` carrier (design TBD) | handoff | Eliminates the `experimental["pipeline_config"]` escape hatch entirely |
