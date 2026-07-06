# Issue 864 Handoff

## Metadata

- Issue: #864, "[Bug] size mismatch for FastWan2.2-TI2V-5B-FullAttn-Diffusers"
- URL: https://github.com/hao-ai-lab/FastVideo/issues/864
- Repo: hao-ai-lab/FastVideo
- Worktree: /tmp/fastvideo-worktrees/issue-864-fastwan22-ti2v-size-mismatch
- Branch: issue/864-fastwan22-ti2v-size-mismatch
- Current stage: Stage 1 complete; awaiting user guidance before implementation
- Implementation begun: no
- Handoff path: .agents/handoffs/issue-864-handoff.md
- Created: 2026-07-06T05:21:35Z
- Recreated after interrupted /tmp worktree loss: 2026-07-06
- Last updated: 2026-07-06T05:39:09Z

## Stage 0 Resume Or Start

- Used `fix-issue` skill for issue #864.
- Read required skill files completely:
  - /home/toolbox/.codex/skills/fix-issue/SKILL.md
  - /home/toolbox/.codex/skills/fix-issue/references/handoff.md
  - /home/toolbox/.codex/skills/fix-issue/references/stages.md
- Verified `gh` identity as `macthecadillac` using `gh api user --jq .login`.
- Initial `git fetch origin` completed with a known_hosts cross-device-link warning but no fetch failure.
- No local or remote branch containing `864` was found by `git branch --all --list '*864*'` at initial start.
- Main checkout had no `.agents/handoffs` directory at initial start.
- Created issue branch/worktree from `origin/main`:
  - branch: `issue/864-fastwan22-ti2v-size-mismatch`
  - worktree: `/tmp/fastvideo-worktrees/issue-864-fastwan22-ti2v-size-mismatch`
  - original base commit: `6a32cf3a5 [ci]: expose LoRA extraction slash command (#1542)`
- The user interrupted while this handoff was about to be finalized. On resume, `/tmp/fastvideo-worktrees` had been removed and git listed the issue worktree as prunable. Ran `git worktree prune`, recreated `/tmp/fastvideo-worktrees`, and reattached the same branch with:
  - `git worktree add /tmp/fastvideo-worktrees/issue-864-fastwan22-ti2v-size-mismatch issue/864-fastwan22-ti2v-size-mismatch`
- Rechecked GitHub issue and PR state after resume.
- A second `git fetch origin` on resume hung silently for about a minute and was interrupted with Ctrl-C. The branch was then fast-forwarded to the locally fetched `origin/main` using `git merge --ff-only origin/main`.
  - Current base after fast-forward: `9d909f5f0`

## Repository Guidance

- Read root `AGENTS.md` in the issue worktree.
- Read relevant subsystem guidance:
  - `fastvideo/AGENTS.md`
  - `fastvideo/configs/AGENTS.md`
  - `fastvideo/models/AGENTS.md`
  - `fastvideo/pipelines/AGENTS.md`
  - `fastvideo/attention/AGENTS.md`
  - `fastvideo/tests/AGENTS.md`
- Searched `.agents/lessons` for FastWan/TI2V/FullAttn/VSA/pred_noise_to_pred_video terms. No relevant lesson files matched.

## GitHub Context Reviewed

- Issue read with:
  - `gh issue view 864 -R hao-ai-lab/FastVideo --json number,title,state,body,labels,assignees,author,comments,createdAt,updatedAt,url,milestone`
- Issue snapshot:
  - State: OPEN
  - Labels: bug
  - Assignees: none
  - Author: Lifedecoder
  - Created: 2025-11-04T10:10:35Z
  - Updated: 2026-02-06T19:45:44Z
- Body reports a runtime shape mismatch in installed `fastvideo/models/utils.py`, line 159, inside `pred_noise_to_pred_video`:
  - `pred_video = noise_input_latent - sigma_t * pred_noise`
  - RuntimeError: tensor a size 53 differs from tensor b size 52 at non-singleton dimension 3.
- Reproduction uses:
  - `FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN`
  - `VideoGenerator.from_pretrained("FastWan2.2-TI2V-5B-FullAttn-Diffusers", num_gpus=1)`
  - `generator.generate_video(..., height=480, width=848)`
  - no image input
  - Environment: `fastvideo 1.6.0`
- Issue comments:
  - Lifedecoder asked whether `FastWan2.2-TI2V-5B-FullAttn-Diffusers` does not support IT2V and only supports T2V.
  - zhisbug replied on 2026-02-06: "yes it only support T2V".
- Open PR list read with:
  - `gh pr list -R hao-ai-lab/FastVideo --state open --limit 200 --json number,title,isDraft,body,url,closingIssuesReferences,headRefName,updatedAt`
  - Raw output was large/truncated in terminal.
- Targeted PR search on resume:
  - `gh pr list -R hao-ai-lab/FastVideo --state open --search '864 OR FastWan2.2-TI2V-5B-FullAttn-Diffusers OR "size of tensor a"' --json ...`
  - Only related open PR returned: #1494, `[feat]: hard-fail on missing/mismatched attention backends; require VSA for FastWan`, head `attn-loud-fail`, ready-for-review (`isDraft=false`), not closing #864.

## Related Issues And PRs

- Search command:
  - `gh api -X GET search/issues -f q='repo:hao-ai-lab/FastVideo FastWan2.2-TI2V-5B-FullAttn-Diffusers' --jq ...`
- Related results:
  - #1494 open PR: attention backend hard-fail work, includes a new FullAttn config route in that branch but is not merged and has merge/pre-commit history.
  - #1153 open RFC: FullAttn output does not follow input image. Collaborator said this is expected because FastWan2.2 FullAttn was only distilled with T2V.
  - #762 closed bug: same tensor mismatch family for FullAttn with VSA. Collaborator noted the checkpoint cannot use `FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN`; PR #763 fixed only a wrong HF model string in the old sampling registry. Later comment suggested checking the pipeline config registry.
  - #711 closed bug: asks whether FullAttn supports I2V; collaborator said FastWan-5B was trained only on T2V data and might only support T2V.
- Related PR #763:
  - Merged; changed only `fastvideo/configs/sample/registry.py` in older code to use `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers`.
  - Does not address current central registry/config behavior.
- Related PR #1494:
  - Ready-for-review, open, not draft.
  - Body says it introduces hard failures for attention backend mismatch and a `FastWan2_2_TI2V_5B_FullAttn_Config` without a VSA requirement, routing `FastWan2.2-TI2V-5B-FullAttn-Diffusers` to it.
  - Changed-file list includes `fastvideo/configs/pipelines/wan.py`, `fastvideo/registry.py`, attention selector/backend files, Wan model, tests, and a renamed `scripts/inference/inference_wan_FullAttn_DMD_5B_720P.yaml`.
  - This branch cannot be assumed available because it is not merged.

## Code Searches And Files Inspected

- Searched for:
  - `FastWan2.2`
  - `TI2V`
  - `FastWan`
  - `FullAttn`
  - `pred_noise_to_pred_video`
  - `VIDEO_SPARSE_ATTN`
  - `image_path`
- Files inspected:
  - `fastvideo/registry.py`
  - `fastvideo/configs/pipelines/wan.py`
  - `fastvideo/pipelines/basic/wan/presets.py`
  - `fastvideo/pipelines/stages/denoising.py`
  - `fastvideo/pipelines/stages/input_validation.py`
  - `fastvideo/pipelines/stages/latent_preparation.py`
  - `fastvideo/models/utils.py`
  - `fastvideo/models/dits/wanvideo.py`
  - `fastvideo/attention/selector.py`
  - `fastvideo/attention/layer.py`
  - `fastvideo/platforms/cuda.py`
  - `fastvideo/configs/models/dits/base.py`
  - `fastvideo/configs/models/dits/wanvideo.py`
  - `fastvideo/entrypoints/video_generator.py`
  - `fastvideo/api/compat.py`
  - `fastvideo/api/sampling_param.py`
  - `fastvideo/fastvideo_args.py`
  - `fastvideo/configs/pipelines/base.py`
  - `docs/inference/support_matrix.md`
  - `scripts/inference/inference_wan_VSA_DMD_5B_720P.yaml`
  - tests under `fastvideo/tests/api`, `fastvideo/tests/entrypoints`, `fastvideo/tests/stages`

## Code Findings

- `fastvideo/registry.py` maps both `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers` and `FastVideo/FastWan2.2-TI2V-5B-Diffusers` to `FastWan2_2_TI2V_5B_Config` with `workload_types=(WorkloadType.T2V, WorkloadType.I2V)`.
- `fastvideo/configs/pipelines/wan.py` defines:
  - `Wan2_2_TI2V_5B_Config` with `ti2v_task=True`, `expand_timesteps=True`, and VAE encoder enabled.
  - `FastWan2_2_TI2V_5B_Config(Wan2_2_TI2V_5B_Config)` inheriting `ti2v_task=True`.
- `docs/inference/support_matrix.md` marks `FastWan2.2 TI2V 5B Full Attn*` under VSA as supported, but the model is known from issue comments to be T2V-only. The table does not expose T2V/I2V capability columns in this section.
- `scripts/inference/inference_wan_VSA_DMD_5B_720P.yaml` points the FullAttn checkpoint at a VSA-named example, although comments in #762 and the "FullAttn" name indicate this checkpoint should not be run with `FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN`.
- `fastvideo/models/dits/wanvideo.py` chooses `WanTransformerBlock_VSA` whenever `envs.FASTVIDEO_ATTENTION_BACKEND == "VIDEO_SPARSE_ATTN"`, independent of whether the checkpoint is FullAttn. This happens before the per-layer attention selector can provide a model-specific guard.
- `fastvideo/pipelines/stages/denoising.py` has a TI2V branch that only activates when `pipeline_config.ti2v_task` and `batch.pil_image is not None`; the issue #864 reproducer has no image input, so the immediate mismatch is more directly about the VSA backend selection and latent geometry than image conditioning.
- `fastvideo/models/utils.py::pred_noise_to_pred_video` simply subtracts two same-layout latent tensors after expanding sigma. It is the failure site, not the likely root cause. Cropping or padding here would hide upstream shape disagreement and risk corrupting inference.
- `fastvideo/pipelines/stages/input_validation.py` loads `image_path` into `batch.pil_image` and then does TI2V-specific preprocessing if `pipeline_config.ti2v_task` is true. Because FullAttn is registered as TI2V/I2V-capable, image inputs can currently flow into code paths that issue/maintainer comments say are unsupported.

## Merits And Scope

- The issue appears valid as a product/code defect: a supported public model ID can be used with documented-looking `VIDEO_SPARSE_ATTN` and/or exposed I2V/TI2V behavior and fail with an opaque internal tensor mismatch.
- The model itself is T2V-only according to two maintainer/collaborator comments (#711/#1153/#864), despite the TI2V name. The issue should not be fixed by trying to make this checkpoint do I2V.
- The shape mismatch should not be patched in `pred_noise_to_pred_video`; that helper expects matching tensor layouts and is used by several training/inference paths.
- Scope should be narrow: make the unsupported combination fail early and accurately, and correct registry/config surfaces so UIs and callers stop advertising I2V for this checkpoint. A broader attention-backend requirement system overlaps open PR #1494 and should either be coordinated with it or kept minimal.

## Possible Approaches

### Approach A: Minimal Model-Specific Guard And Registry Fix

- Add a FullAttn-specific pipeline config for `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers` that inherits the FastWan DMD defaults but sets `ti2v_task=False`.
- Register the FullAttn model path as `workload_types=(WorkloadType.T2V,)` only.
- Add a clear early validation error if this FullAttn config is used with `FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN`, directing users to use a dense backend such as FlashAttention/SDPA.
- Optionally leave `FastVideo/FastWan2.2-TI2V-5B-Diffusers` on the existing config if that checkpoint is actually VSA/TI2V, or split it only after verifying intended behavior.
- Tests:
  - registry resolves FullAttn to the FullAttn config;
  - registered workloads for FullAttn include only T2V;
  - FullAttn config has `ti2v_task=False`;
  - VSA env with FullAttn raises the new clear error before model construction/inference.
- Pros: small blast radius, directly addresses issue #864 and I2V confusion.
- Cons: does not solve the general backend footgun for every model; may conflict with #1494 if both edit the same files.

### Approach B: Pull Forward The Relevant Pieces Of #1494

- Port the model-level attention-backend requirement/check pattern from #1494 after adapting to current `origin/main`.
- Include `required_attention_backend` support, strict env validation, and FullAttn route with no VSA requirement.
- Pros: systemic fix for the backend mismatch class.
- Cons: larger, overlaps an active ready-for-review PR by another contributor, touches more files, and increases review/conflict risk. Not necessary to solve #864 narrowly unless maintainers want to replace or unblock #1494.

### Approach C: Documentation/Example Only

- Rename or update `scripts/inference/inference_wan_VSA_DMD_5B_720P.yaml` and docs to avoid telling users to run FullAttn with VSA.
- Clarify FullAttn is T2V-only.
- Pros: very low code risk.
- Cons: does not prevent the runtime crash for existing API users or UI workload enumeration. Insufficient for the reported bug.

### Approach D: Shape Tolerance In `pred_noise_to_pred_video`

- Crop/pad one tensor to match the other.
- Pros: might avoid the exact crash.
- Cons: hides a real upstream shape/config mismatch and risks corrupted output across many shared paths. Not recommended.

## Recommended Plan

Recommend Approach A unless the user explicitly wants to coordinate with or supersede #1494.

Implementation steps for Stage 2:

1. Re-check issue #864, related comments, and open PR #1494 before editing.
2. Add `FastWan2_2_TI2V_5B_FullAttn_Config` in `fastvideo/configs/pipelines/wan.py`.
   - Inherit from `FastWan2_2_TI2V_5B_Config` if that preserves DMD defaults.
   - Override `ti2v_task=False`.
   - Ensure VAE encoder loading is disabled unless another code path requires it for T2V.
   - Keep DMD denoising steps unchanged.
3. Update `fastvideo/registry.py`.
   - Import the new config.
   - Register `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers` separately as T2V-only.
   - Leave or explicitly decide what to do with `FastVideo/FastWan2.2-TI2V-5B-Diffusers`.
4. Add an early error for `FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN` with the FullAttn config/model path.
   - Best location should be before `WanTransformer3DModel` chooses `WanTransformerBlock_VSA`; candidate surfaces are the config validation path or Wan model constructor.
   - Message should name the model/checkpoint and say FullAttn is dense/T2V-only and does not support VSA.
5. Update docs/examples only if needed:
   - Rename or edit `scripts/inference/inference_wan_VSA_DMD_5B_720P.yaml` to avoid a VSA example for the FullAttn checkpoint.
   - Clarify support matrix wording for FullAttn T2V-only if a durable doc line is appropriate.
6. Add focused tests in `fastvideo/tests/api` or `fastvideo/tests/attention`.
   - Use config/registry tests where possible to avoid model weight downloads.
   - Do not run local tests; validate via Modal in Stage 2 if needed.
7. Commit with GPG signing and push immediately.
8. Run Stage 3 review/adjudication loop before presenting a draft PR message.

## Validation Plan

- No tests or Modal jobs were run in Stage 1.
- Stage 2 should not run local tests under FastVideo rules.
- Suggested targeted validation on Modal L40S through `fastvideo/tests/modal/launch_l40s_job.py` from branch `interleavethinker`:
  - focused registry/config test file(s) added for this issue;
  - any attention/config validation test added for the VSA rejection path.
- If source files outside tests/docs change, run relevant targeted Modal pytest first.
- Before any future draft PR creation, Stage 4 must run `pre-commit run --all-files`, fix all issues, retire the handoff with `git rm`, commit and push the deletion, and verify the handoff is absent from the branch.
- No SSIM regeneration is expected for the recommended fix because it should prevent invalid usage rather than change valid generation output. If the implementation touches actual Wan denoising behavior beyond validation/registration, run targeted Wan T2V/SSIM validation on L40S.

## Open Questions

- Should `FastVideo/FastWan2.2-TI2V-5B-Diffusers` remain mapped to the TI2V config, or should it also be treated as T2V-only? The issue and comments primarily discuss the FullAttn checkpoint.
- Should we coordinate with #1494 instead of creating a narrow fix? #1494 covers similar terrain but is broader and currently open.

## Next Decision

- Await user guidance before implementation.
- Recommended choice: proceed with Approach A, the minimal FullAttn-specific T2V-only config/registry fix plus early VSA rejection and focused tests.
