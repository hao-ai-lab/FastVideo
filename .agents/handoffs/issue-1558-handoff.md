# Issue 1558 Handoff

## Current State

- Issue: #1558
- Title: [Bug] FA4 (flash_attn.cute) fmax() TypeError crashes full-suite training/model-load tests - cutlass-dsl API mismatch, forced by FASTVIDEO_FA4=1 default
- URL: https://github.com/hao-ai-lab/FastVideo/issues/1558
- State: OPEN
- Labels: installation, platform, scope: training, scope: inference, scope: attention, scope: kernel, scope: model
- Assignees: none
- Created: 2026-07-05T22:21:59Z
- Updated: 2026-07-05T22:28:48Z
- Repo: hao-ai-lab/FastVideo
- Branch: issue/1558-fa4-fmax-typeerror
- Worktree: /tmp/fastvideo-worktrees/issue-1558-fa4-fmax-typeerror
- Handoff path: .agents/handoffs/issue-1558-handoff.md
- Current stage: Stage 2 - Implement The User-Directed Fix
- Implementation begun: yes

## Authentication And Git Notes

- gh identity verified outside sandbox on 2026-07-06T05:18:10Z: macthecadillac.
- Local remotes include origin=macthecadillac/FastVideo and upstream=hao-ai-lab/FastVideo.
- SolitaryThinker remote exists in the base checkout but was not used.
- Fetched origin and upstream before branch creation. `git fetch origin` completed with known_hosts cross-device-link warnings but no nonzero exit.
- No local/fetched branch or handoff containing issue 1558 existed before this branch was created.
- Worktree was created from upstream/main at 9d909f5f0 ([test]: remove dead and duplicate tests (-489 lines) (#1556)).
- 2026-07-06T05:39Z: Recreated the issue worktree from existing branch `issue/1558-fa4-fmax-typeerror` after the `/tmp` worktree disappeared from `git worktree list`; resumed from pushed handoff commit 8be27f48625f3a0bc1b2515eed7da5b15050d16a.
- 2026-07-06T05:52Z: After user interruption, `/tmp` issue and validation worktrees were again missing/prunable. Ran `git worktree prune`, recreated `/tmp/fastvideo-worktrees/issue-1558-fa4-fmax-typeerror` and `/tmp/fastvideo-worktrees/interleavethinker-modal`, and reapplied the Stage 2 patch.

## Stage 1 Log

- 2026-07-06T05:18:10Z: Initialized Stage 1 handoff. No implementation changes have been made.
- 2026-07-06T05:20Z: Read full issue body and all comments with `gh issue view 1558`.
- 2026-07-06T05:21Z: Checked open PRs with `gh pr list` and searched PRs/issues for `1558`, `fmax`, `FASTVIDEO_FA4`, `FA4_CUTE_REF`, `nvidia-cutlass-dsl`, `flash_attn.cute`, `cutlass-dsl`, and `flash-attn-4`.
- 2026-07-06T05:22Z: Read related issue #1524 and draft PR #1557 because #1558 references #1524 as adjacent cutlass-dsl runtime-compile work.
- 2026-07-06T05:27Z: Read root, `fastvideo/`, `fastvideo/attention/`, and `fastvideo/tests/` AGENTS guidance before code inspection.
- 2026-07-06T05:31Z: Inspected FA4 env declarations, Modal harness defaults, Docker FA4 overlay, pyproject dependency surface, and FA4 attention routing.
- 2026-07-06T05:34Z: Read merged PRs #1539, #1540, and #1541 for the intended FA4 opt-in and kernel-pin behavior.

## Stage 2 Log

- 2026-07-06T05:39Z: User approved the recommended targeted CI harness scope fix.
- 2026-07-06T05:39Z: Re-checked issue #1558 and comments with gh. State remained OPEN, no new comments since 2026-07-05T22:28:48Z.
- 2026-07-06T05:39Z: Re-checked open PRs for `1558`, `fmax`, `FASTVIDEO_FA4`, `FA4_CUTE_REF`, and `nvidia-cutlass-dsl`. No open PR mentions or closes #1558.
- 2026-07-06T05:39Z: Searched `.agents/lessons`, Modal harness files, attention tests, and contract tests for related pitfalls/coverage terms. No relevant lesson was found; existing attention tests cover FA4 explicit opt-in behavior.
- Selected approach: Option A - keep SSIM FA4 default enabled, disable FA4 for model-load/training-style Modal lanes that do not intend to exercise FA4, and make the generic L40S launcher default follow product behavior (`FASTVIDEO_FA4=0`) unless the caller opts in.
- Implementation scope: edit `fastvideo/tests/modal/pr_test.py` and `fastvideo/tests/modal/launch_l40s_job.py`; do not edit FA4 runtime fallback behavior, Docker pins, `pyproject.toml`, or `fastvideo/tests/modal/ssim_test.py`.
- 2026-07-06T05:53Z: Patched `fastvideo/tests/modal/pr_test.py` so transformer/model-load/training-style lanes set `FASTVIDEO_FA4=0` in their pytest command strings while leaving image-level FA4 default `1` for inference/perf parity.
- 2026-07-06T05:53Z: Patched `fastvideo/tests/modal/launch_l40s_job.py` so generic ad hoc jobs default `FASTVIDEO_FA4` to `0`, preserving explicit opt-in through local env or `--env-vars`.
- 2026-07-06T05:53Z: Added `fastvideo/tests/contract/test_modal_fa4_policy.py`, a source-only contract test that guards the generic launcher default, the SSIM FA4 default, and the pr_test training/model-load command policy.
- 2026-07-06T06:00Z: Modal L40S validation attempt `ap-E3CvsFRa3g2mEb5UH9Vyu4` failed before tests ran. Command used interleavethinker launcher with `--install-extra test`, `--env-vars FASTVIDEO_FA4=0`, and local code/test patch. Failure was dependency install: `uv pip install -e '.[test]'` tried to build `fastvideo-kernel==0.3.2` from sdist and failed with `fatal error: cutlass/cutlass.h: No such file or directory`. This did not reach pytest and is unrelated to the FA4 policy patch.
- 2026-07-06T06:01Z: Modal L40S validation attempt `ap-AWGczZL2hKCLnzOLMnvIvD` used `--install-extra none` and reached the command, but failed before pytest because `hf auth login --token $HF_API_KEY` received an empty token. This did not reach pytest and is unrelated to the FA4 policy patch.
- 2026-07-06T06:57Z: Modal L40S validation `ap-dBp3HNyPscUyqd2TamekW3` passed with interleavethinker launcher, `--install-extra none`, `--env-vars FASTVIDEO_FA4=0`, and local code/test patch:
  `pytest fastvideo/tests/contract/test_modal_fa4_policy.py fastvideo/tests/train/models -vs`
  Result: `9 passed in 92.34s`. The suite included the new contract test plus Cosmos, Hunyuan, LongCat, MatrixGame2, Wan, and Wan causal model-load/forward tests. Logs showed `flash_attn.cute (FA4) is installed but not enabled` and `Using FlashAttention-2 backend` for the affected path, so the reported FA4 `fmax()` compile path was avoided.
- 2026-07-06T07:03Z: Modal L40S validation `ap-hrzrFjSWJ2w8h8qcjaDmqW` passed with interleavethinker launcher, `--install-extra none`, `--env-vars FASTVIDEO_FA4=0`, and local code/test patch:
  `pytest fastvideo/tests/transformers -vs`
  Result: `5 passed, 1 skipped in 265.13s`. Logs again showed FA4 installed but not enabled and FlashAttention-2 selected for the affected FLASH_ATTN path.
- 2026-07-06T07:04Z: Created signed commit `81baaa34048484add1aacb720ebe9d29ace869fe` (`[ci] Stop forcing FA4 in model-load lanes`) and pushed it to `origin/issue/1558-fa4-fmax-typeerror`. `git log -1 --show-signature` reported a good signature from `Mac Lee <macthecadillac@gmail.com>`.
- 2026-07-06T07:08Z: Modal L40S pre-commit validation `ap-4iiXhNnoFvM881XZRV0iY4` passed on pushed commit `81baaa34048484add1aacb720ebe9d29ace869fe` using interleavethinker launcher:
  `pre-commit run --all-files`
  Result: yapf, ruff, codespell, PyMarkdown, actionlint, mypy, filename-space check, and suggestion all passed.

## Stage 3 Log

- 2026-07-06T07:08Z: Spawned fresh review-code sub-agent `019f363f-94fc-7873-9b13-5ca79f824585` / nickname `Lorentz` to review committed branch `macthecadillac/FastVideo issue/1558-fa4-fmax-typeerror` for issue #1558. The sub-agent was instructed to use `$review-code` and not modify code or GitHub state.
- 2026-07-06T07:10Z: Review-code sub-agent completed with no actionable findings. Summary:
  - Reviewed commit `81baaa34048484add1aacb720ebe9d29ace869fe` in a temp worktree.
  - Found the branch matches issue #1558's CI-unblock path: model-load/training lanes force `FASTVIDEO_FA4=0`, generic launcher defaults FA4 off, and the new contract test covers the intended Modal policy.
  - Issue fit: addresses the "do not force FA4 for model-load/training lanes" path; explicit FA4 image compatibility remains a separate follow-up.
  - Related branches: only `issue/1558-fa4-fmax-typeerror`.
  - Validation gaps noted by reviewer: full `pr_test.py` suite and explicit FA4/image-pin validation remain unverified.
- No adjudicator/fixer sub-agent was spawned because the review-code sub-agent reported no actionable findings.

## Draft PR Message

Title: `[ci] Stop forcing FA4 in model-load lanes`

Body:

```markdown
## Summary

Fixes #1558 by keeping FA4 enabled only where CI intentionally exercises FA4, while returning model-load and training-style Modal lanes to the product default (`FASTVIDEO_FA4=0` unless explicitly opted in).

- Set `FASTVIDEO_FA4=0` for transformer/model-load and training-style `pr_test.py` lanes.
- Default generic `launch_l40s_job.py` runs to FA4 off, preserving opt-in through local env or `--env-vars`.
- Add a source-only contract test covering the Modal FA4 policy and preserving SSIM's FA4 default.

## Validation

- Modal L40S `pytest fastvideo/tests/contract/test_modal_fa4_policy.py fastvideo/tests/train/models -vs`
  - `9 passed in 92.34s`
  - App: `ap-dBp3HNyPscUyqd2TamekW3`
- Modal L40S `pytest fastvideo/tests/transformers -vs`
  - `5 passed, 1 skipped in 265.13s`
  - App: `ap-hrzrFjSWJ2w8h8qcjaDmqW`
- Modal L40S `pre-commit run --all-files`
  - passed
  - App: `ap-4iiXhNnoFvM881XZRV0iY4`

## Notes

This unblocks CI lanes that do not need FA4. It does not resolve the underlying explicit-FA4 image compatibility mismatch between the pinned FA4 cute overlay and the currently resolved `nvidia-cutlass-dsl`; that remains a separate image/dependency follow-up.

# Checklist
- [ ] I ran pre-commit run --all-files and fixed all issues
- [ ] I added or updated tests for my changes
- [ ] I updated documentation if needed
- [ ] I considered GPU memory impact of my changes
For model/pipeline changes, also check:
- [ ] I verified targeted Wan T2V SSIM regression tests pass on L40S
- [ ] I updated the support matrix if adding a new model
```

## Files Changed

- `fastvideo/tests/modal/pr_test.py`
  - Kept image-level `FASTVIDEO_FA4` default as `1` for existing inference/perf parity.
  - Added `FASTVIDEO_FA4=0` to `run_transformer_tests`, legacy training, LoRA training, VSA training, distill DMD, self-forcing, modular train model/methods, and grad-norm seeding commands.
- `fastvideo/tests/modal/launch_l40s_job.py`
  - Changed generic launcher default from `os.environ.get("FASTVIDEO_FA4", "1")` to `os.environ.get("FASTVIDEO_FA4", "0")`.
- `fastvideo/tests/contract/test_modal_fa4_policy.py`
  - New source/AST contract coverage for the intended Modal FA4 policy.

## GitHub Context

- Issue body: reporter says full-suite training/model-load tests fail in FA4 `flash_attn.cute` compile with `TypeError: fmax() takes 2 positional arguments but 3 positional arguments (and 3 keyword-only arguments) were given`.
- Reporter-proposed root cause: pinned/overlaid `flash-attn` cute code expects an older `nvidia_cutlass_dsl` `nvvm.fmax` signature under CUDA 12.9, while the installed cutlass-dsl binding exposes a two-positional-argument form.
- Reporter-proposed CI trigger: Modal harnesses default `FASTVIDEO_FA4` to `1` in `fastvideo/tests/modal/pr_test.py`, `fastvideo/tests/modal/launch_l40s_job.py`, and `fastvideo/tests/modal/ssim_test.py`, forcing FA4 after #1540 made FA4 explicit opt-in. This sends model-load/training lanes through `flash_attn.cute`, even when the PR under test is unrelated.
- Reporter reproduction: full suite failures in model-load tests for Cosmos, Hunyuan, LongCat, MatrixGame2, and Wan. Abbreviated path: `fastvideo/attention/backends/flash_attn.py:125` -> `fastvideo/attention/utils/flash_attn_cute.py:301` -> `flash_attn/cute/interface.py` -> `flash_attn/cute/softmax.py` -> `flash_attn/cute/utils.py:fmax`.
- Reporter suggested fixes: either align image pins so `flash-attn` cute and `nvidia_cutlass_dsl` agree, or stop forcing FA4 for model-load/training lanes while keeping FA4 where explicitly intended.
- Single issue comment by Mister-Raggs: narrows the image-level mismatch to Dockerfile FA4 cute overlay handling. Notes `FA4_CUTE_REF=940cd9680f3315f2f06b43ab5bea2c2cf2d96806`, says the overlaid cute rev still calls the old three-positional `nvvm.fmax`, and says `nvidia-cutlass-dsl` is not pinned in `pyproject.toml` and floats transitively via flashinfer/quack. Proposed owner-level fixes: pin `nvidia-cutlass-dsl` to the compatible version or bump `FA4_CUTE_REF` to a cute revision compatible with current cutlass-dsl. Comment explicitly says this needs an image rebuild and GPU full-suite validation.
- Open PR scan: no open PR closes, mentions, or appears to address #1558 or the specific terms `fmax`, `FASTVIDEO_FA4`, `FA4_CUTE_REF`, or `nvidia-cutlass-dsl`.
- Related issue #1524: same broad cutlass-dsl/runtime-compile family, but it is about FastWan-QAD FP4 on RTX 5080/SM120 and FlashInfer/CUTLASS runtime compilation. It is not the same `flash_attn.cute` `fmax()` failure in CI L40S full-suite lanes.
- Related draft PR #1557: draft, open, closes #1524, branch `fix-latent-handoff-fastwan-qad-path`. It moves latent worker outputs to CPU and fixes the FastWan-QAD TAEHV example path; it does not address #1558 and its draft status was not changed.

## Code Findings

- `fastvideo/envs.py:216-221` defines `FASTVIDEO_FA4` as opt-in and defaults it to `0`. The core product default is therefore FA4 off.
- `fastvideo/attention/backends/flash_attn.py:21-35` honors `FASTVIDEO_FA4=1` by importing `fastvideo.attention.utils.flash_attn_cute` and selecting `fa_version = "4"`. If the import fails it raises a loud RuntimeError naming the env var.
- `fastvideo/attention/backends/flash_attn.py:119-125` routes FA4 selected calls through `flash_attn_func` from `flash_attn_cute`.
- `fastvideo/attention/utils/flash_attn_cute.py:57-67` routes pre-sm90 grad-enabled or GQA calls to FA2, but `flash_attn_cute.py:280-302` sends no-grad MHA calls to `torch.ops.fastvideo._flash_attn_cute_forward`, which calls `_flash_attn_fwd` and can trigger the reported cutlass-dsl `fmax()` compile failure.
- `fastvideo/tests/train/models/test_load_wan.py:65-75` and sibling model-load tests run transformer forwards under `torch.no_grad()`, matching the no-grad FA4 path above. Reporter listed failures in Cosmos, Hunyuan, LongCat, MatrixGame2, and Wan model-load tests.
- `fastvideo/tests/modal/pr_test.py:63-66` globally injects `FASTVIDEO_FA4 = os.environ.get("FASTVIDEO_FA4", "1")` into the Modal image. That forces FA4 for every pr_test lane unless the caller explicitly overrides it.
- `fastvideo/tests/modal/pr_test.py:197-199`, `211-213`, `224-226`, `257-260`, `270-272`, and `328-330` launch legacy training, LoRA training, VSA training, distill, self-forcing, and modular train model/method tests under the same global FA4 image env.
- `fastvideo/tests/modal/launch_l40s_job.py:94-103` also injects `FASTVIDEO_FA4` default `1` into the generic Modal image. `_run_gpu_job` later inherits that image env and does not reset it, so arbitrary validation jobs default to FA4 unless `--env-vars FASTVIDEO_FA4=0` or local env override is supplied.
- `fastvideo/tests/modal/ssim_test.py:66-68` intentionally keeps FA4 default `1` because seeded SSIM references were generated with FA4 inference. This is the strongest reason not to globally remove FA4 from all Modal harnesses.
- `docker/Dockerfile:67-71` and `164-184` install a prebuilt `flash-attn` wheel, remove stale cute files, then overlay `flash-attn-4` from `FA4_CUTE_REF=940cd9680f3315f2f06b43ab5bea2c2cf2d96806`. The comments say this is intended to be cutlass-4.5-safe.
- `pyproject.toml:115-117` pins the `flash-attn-4` source to the same ref, but `pyproject.toml` does not pin `nvidia-cutlass-dsl`; only `flashinfer-python` is a direct core dependency at line 36 and `flash-attn-4`/`flashinfer-python` are in the dreamverse extra at lines 176-183. This supports the reporter's claim that cutlass-dsl can float transitively.
- PR #1540 deliberately made FA4 explicit opt-in and kept the Modal launchers setting `FASTVIDEO_FA4=1` for CI/image parity. Its body also says L40S training grad-enabled attention routes to FA2, but no-grad model-load forwards with FA4 enabled still use FA4.
- PRs #1539 and #1541 were about `fastvideo-kernel` pinning and FA4 tile API compatibility, not this `nvvm.fmax` signature mismatch.

## Hypothesis And Scope

- The bug report is valid. Core FastVideo correctly defaults FA4 off, but the Modal CI harnesses force FA4 on by default. That exposes an image/dependency mismatch in FA4 cute during no-grad model-load forwards, causing unrelated PRs to fail in full-suite training/model-load lanes.
- The root cause has two layers:
  1. Environment/dependency skew: pinned FA4 cute ref and floating `nvidia-cutlass-dsl` disagree on `nvvm.fmax`.
  2. Blast-radius issue: CI utilities force the optional FA4 path for lanes that do not need to validate FA4.
- A dependency pin or FA4 ref bump is the more direct image-quality fix, but it needs a rebuilt image and GPU full-suite validation. The fastest unblock with least behavioral risk is to stop defaulting FA4 on for model-load/training lanes and generic Modal jobs, while preserving explicit FA4 in SSIM/inference/performance lanes that intentionally validate that path.

## Alternatives Considered

- Option A - targeted CI harness scope fix (recommended): keep `ssim_test.py` defaulting FA4 to `1`, keep or explicitly set FA4 only for intended FA4/performance lanes, and set `FASTVIDEO_FA4=0` for model-load/training commands in `pr_test.py`. Also change generic `launch_l40s_job.py` to default to `0` so ad hoc validation follows product defaults unless the caller opts in.
- Option B - image/dependency fix only: pin `nvidia-cutlass-dsl` or bump `FA4_CUTE_REF` to a compatible flash-attn cute revision. This addresses FA4 itself but needs dependency research, image rebuild, and full GPU validation. It may be the long-term fix but does not reduce CI blast radius if FA4 breaks again.
- Option C - runtime fallback from FA4 to FA2/FA3 on this TypeError. This conflicts with #1540's design: `FASTVIDEO_FA4=1` should be a loud opt-in failure, not a silent fallback. It also risks masking unrelated runtime errors and recreating removed fallback behavior.
- Option D - globally set Modal `FASTVIDEO_FA4` defaults to `0`. This unblocks model-load/training but may change SSIM/performance reference parity where FA4 is intentional.

## Recommended Plan

- Implement Option A unless the user specifically wants the image/dependency pin work first.
- In `fastvideo/tests/modal/pr_test.py`, keep the image-level caller override behavior, but force `FASTVIDEO_FA4=0` in the shell commands for model-load/training lanes that do not intend to exercise FA4:
  - legacy training (`run_training_tests`)
  - LoRA training
  - VSA training
  - distill DMD
  - self-forcing
  - modular train model/method tests (`run_train_framework_tests`)
  - grad-norm seeding if kept as part of the same training surface
- Decide whether transformer tests should also run with FA4 disabled. They are not listed in the issue reproduction, but they are no-grad model forwards and may hit the same FA4 path. Conservative implementation should include them unless preserving FA4 coverage there is intentional.
- Change `fastvideo/tests/modal/launch_l40s_job.py` default `FASTVIDEO_FA4` from `1` to `0`, preserving caller override through the local environment or `--env-vars`.
- Leave `fastvideo/tests/modal/ssim_test.py` at default `1` because comments and #1540 say SSIM references were seeded with FA4.
- Add/update a focused test or contract check if feasible without GPU. A practical low-risk test would assert the Modal launcher defaults/commands encode the intended FA4 policy, but avoid over-building parser abstractions just for this issue.
- Avoid touching FA4 attention runtime fallback behavior.
- Future follow-up: separately align the Docker/pyproject FA4 cute + cutlass-dsl pin after confirming the compatible pair in a rebuilt image.

## Validation Status

- No local pytest run, per FastVideo validation rules.
- Modal model-load and transformer-lane validation passed on L40S as above.
- Modal `pre-commit run --all-files` passed on L40S as above.
- Latest pushed fix commit: `81baaa34048484add1aacb720ebe9d29ace869fe`.
- Stage 3 review-code loop completed with no actionable findings. No PR has been opened.
- Handoff remains active and must be removed before any future Stage 4 draft PR creation.

## Planned Validation

- Stage 2 should use Modal L40S through `fastvideo/tests/modal/launch_l40s_job.py` from branch `interleavethinker`, not local pytest.
- Focused validation should include the exact failing model-load test group with `FASTVIDEO_FA4=0` default behavior, for example:
  `pytest fastvideo/tests/train/models -vs`
- If transformer lanes are changed, validate `pytest fastvideo/tests/transformers -vs` or the smallest affected subset.
- If training command envs are changed, validate at least the modular train model/method lane command used by `run_train_framework_tests`.
- Before any draft PR in Stage 4, run `pre-commit run --all-files` and fix failures.
- No draft PR should be opened until the user explicitly asks for Stage 4; if opened, it must be draft and must not alter any existing PR draft status.

## Next Steps

- Present Stage 1 report and ask the user to choose the approach.
- Stage 2 must not start until user guidance is received.
