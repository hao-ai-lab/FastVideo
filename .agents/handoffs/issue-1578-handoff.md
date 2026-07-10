# Issue 1578 Handoff

## Metadata
- Issue: #1578
- Title: `[Bug] Qwen2.5-VL vision tower always resolves to float32: torch_dtype read from vision_config, where it is never set`
- Repo: `hao-ai-lab/FastVideo`
- Branch: `issue/1578-qwen-vl-vision-dtype`
- Worktree: `/tmp/fastvideo-worktrees/issue-1578-qwen-vl-vision-dtype`
- Handoff path: `.agents/handoffs/issue-1578-handoff.md`
- Current stage: Stage 1 complete - awaiting user guidance for Stage 2
- Implementation begun: no
- Created: 2026-07-10T08:36:53Z
- Last updated: 2026-07-10T09:02:00Z

## Workflow State
- Stage 0 completed: no existing local or fetched `origin`/`upstream` branch containing `1578` was found.
- No existing `.agents/handoffs/issue-1578-handoff.md` was found in the main checkout or known FastVideo worktrees.
- Created dedicated worktree from `origin/main`.
- Verified `gh` identity as `macthecadillac`.
- `git fetch origin` completed with a known_hosts warning: `hostfile_replace_entries failed ... Invalid cross-device link`, exit code 0.
- `git fetch upstream` completed successfully.
- After the issue branch was created from stale `origin/main` (`9d909f5f0457ac91f489d5fc8000931f042b72ce`), it was fast-forwarded to current `upstream/main` (`4c08ffce497bb7fe732454135d1fb7bf6733fbb2`) with `git merge --ff-only upstream/main`.
- Branch status after fast-forward: ahead of `origin/main` by 15 commits, plus staged/unstaged handoff updates only.

## Stage 1 Notes
- Stage 1 is implementation-free. Only this handoff has been created/updated.
- Root `AGENTS.md` read in the issue worktree.
- `fastvideo/AGENTS.md`, `fastvideo/models/AGENTS.md`, and `fastvideo/tests/AGENTS.md` read.
- Relevant lessons read:
  - `.agents/lessons/2026-05-07_conversion-cast-bf16-suffix-allowlist.md`
  - `.agents/lessons/2026-05-07_dit-dtype-boundary-with-flash-attn.md`
- Local config-only Python reproduction was attempted, but local environment lacks `torch`:
  `ModuleNotFoundError: No module named 'torch'`. Do not rely on local tests; validate on Modal in Stage 2.

## GitHub Context
- `gh issue view 1578 -R hao-ai-lab/FastVideo --json ...` reviewed at 2026-07-10T08:40Z.
- Issue state: OPEN.
- URL: https://github.com/hao-ai-lab/FastVideo/issues/1578
- Author: `Mister-Raggs`.
- Assignees: none.
- Labels: `installation`, `platform`, `scope: inference`, `scope: attention`, `scope: model`.
- Created: 2026-07-10T03:27:36Z.
- Updated: 2026-07-10T03:27:43Z.
- Comments: none.
- Reported bug:
  - `fastvideo/models/encoders/qwen2_5_vl_custom.py` vision tower derives `self.dtype` from `config.torch_dtype`.
  - The config passed to the vision tower is `config.vision_config`.
  - Real `Qwen/Qwen2.5-VL-7B-Instruct` config stores `"torch_dtype": "bfloat16"` at the composite/top level, not in `vision_config`.
  - Therefore `config.vision_config.torch_dtype` is `None`, and `self.dtype` resolves to `torch.float32`.
  - `pixel_values` and `pixel_values_videos` are later cast to `self.visual.dtype`, so the image/video vision path casts inputs to fp32 while weights may be bf16.
  - Reporter also notes a latent bug: comparing only to string `"bfloat16"` misses transformers versions that materialize `torch_dtype` as `torch.bfloat16`.
  - Reporter believes image/video path is dormant in current Cosmos 2.5 Reason1 pipeline and has not run real-weight reproduction.
- Commenter-proposed fixes: none beyond reporter's suggestions:
  - read dtype from parent/composite config;
  - compare against both string and `torch.dtype`;
  - or derive from actual module parameter dtype.
- Open PR sweep:
  - `gh pr list -R hao-ai-lab/FastVideo --state open --limit 200 --json ...` reviewed. Large output included many unrelated open PRs.
  - Targeted `gh search prs --repo hao-ai-lab/FastVideo "Qwen2.5-VL"` found #1577 as the only open Qwen2.5-VL PR.
  - `gh search prs --repo hao-ai-lab/FastVideo "qwen2_5_vl_custom"` returned none.
  - `gh search prs --repo hao-ai-lab/FastVideo "vision_config torch_dtype"` returned none.
  - `gh search issues --repo hao-ai-lab/FastVideo "qwen2_5_vl_custom torch_dtype"` returned none.
- Related PR #1577:
  - URL: https://github.com/hao-ai-lab/FastVideo/pull/1577
  - Title: `[ci]: cover transformers 5.x config compat for Qwen2.5-VL (#1576)`
  - State: OPEN.
  - Draft status: ready-for-review (`isDraft=false`); draft status was not changed.
  - Branch: `test/qwen25vl-transformers5-compat`.
  - Files: only `fastvideo/tests/encoders/test_qwen2_5_vl_config_compat.py` added.
  - Covers transformers 5.x config API compatibility for #1576, not the production vision dtype resolution path in #1578.

## Code Findings
- Current upstream code still has the reported bug.
- `fastvideo/models/encoders/qwen2_5_vl_custom.py:307-310`:
  `Qwen2_5_VisionTransformerPretrainedModel.__init__` accepts only the vision sub-config and sets
  `self.dtype = torch.bfloat16 if config.torch_dtype == "bfloat16" else torch.float32`.
- `fastvideo/models/encoders/qwen2_5_vl_custom.py:1457-1461`:
  `Qwen2_5_VLForConditionalGenerationSimple.__init__` flattens text config fields, stores the parent config, then constructs
  `self.visual = Qwen2_5_VisionTransformerPretrainedModel(config.vision_config)`.
  It does not pass parent/composite `config.torch_dtype` to the visual tower.
- `_flatten_text_config()` only copies missing attributes from `text_config` to the parent config. It does not touch `vision_config` or dtype propagation.
- `fastvideo/models/encoders/qwen2_5_vl_custom.py:1744-1764`:
  image and video inputs are explicitly cast with `pixel_values.type(self.visual.dtype)` and
  `pixel_values_videos.type(self.visual.dtype)`.
- `fastvideo/models/encoders/qwen2_5_vl_custom.py:115-116`:
  `Qwen2_5_VisionPatchEmbed.forward` then casts patch input to `self.proj.weight.dtype`.
  This means the current code is unlikely to fail at the patch-embed conv solely because `self.visual.dtype` is fp32; it can still create an unnecessary fp32 temporary before casting back to the parameter dtype.
- `fastvideo/models/encoders/reason1.py:141-164`:
  Reason1's loader currently skips `visual` weights, `lm_head`, and `decoder`, supporting the reporter's note that the image/video vision path is dormant for the current Cosmos 2.5 Reason1 usage.
- Searches run:
  - `rg -n "Qwen2_5|qwen2_5|vision_config|torch_dtype|self\\.dtype|visual\\.dtype|pixel_values" fastvideo tests docs examples scripts`
  - `rg -n "flatten_text_config|torch_dtype|Qwen2_5_VisionTransformerPretrainedModel|self\\.visual|visual\\.dtype" fastvideo/models/encoders/qwen2_5_vl_custom.py fastvideo/models/encoders/reason1.py fastvideo/tests/encoders`
  - `rg -n "\\.type\\(self\\.visual\\.dtype\\)|visual\\.dtype|self\\.dtype =|target_dtype = .*weight\\.dtype" fastvideo/models/encoders/qwen2_5_vl_custom.py fastvideo/models/encoders`

## Current Hypothesis
- The issue is valid: top-level `torch_dtype` is not visible to the vision-only sub-config, and the current string-only comparison also misses `torch.dtype` values.
- Severity is limited by current usage:
  - Reason1 appears to use the text path and skip visual weights.
  - Patch embed casts to parameter dtype internally, so the most likely practical effect is avoidable fp32 materialization and a misleading `self.visual.dtype`, not a guaranteed runtime conv dtype mismatch.
- The fix should be narrow and local to the custom Qwen2.5-VL model, with lightweight CPU/config tests.

## Alternatives Considered
- Approach A - preferred: add a small dtype normalization helper and pass parent dtype into the visual constructor.
  - Change `Qwen2_5_VisionTransformerPretrainedModel.__init__` to accept an optional parent/composite dtype.
  - Prefer explicit `vision_config.torch_dtype` when present, otherwise fall back to parent `config.torch_dtype`.
  - Normalize both strings such as `"bfloat16"` / `"torch.bfloat16"` and `torch.dtype` objects such as `torch.bfloat16`.
  - Update `Qwen2_5_VLForConditionalGenerationSimple.__init__` to pass `getattr(config, "torch_dtype", None)` into the visual constructor.
  - Tradeoff: small API change to an internal class, but no mutation of HF config objects and no runtime weight dependency.
- Approach B: mutate `config.vision_config.torch_dtype` from the parent config during `_flatten_text_config()`.
  - Simpler constructor signature, but it mutates an external config object and mixes text-flattening compatibility with vision dtype propagation.
  - Less preferred.
- Approach C: stop using `self.visual.dtype` and cast image/video tensors directly from the actual first visual parameter dtype.
  - Most aligned with loaded weights, but this custom model can have meta/unloaded visual parameters in Reason1 and the public `self.visual.dtype` remains wrong unless also updated.
  - More behavioral surface than needed for this report.
- Approach D: leave code unchanged because the active pipeline does not use the image/video path.
  - Not preferred; the bug is real and easy to guard with low-cost tests.

## Recommended Plan
- Use Approach A.
- Implementation steps for Stage 2 after user approval:
  1. Add a local helper near the Qwen2.5-VL visual classes, e.g. `_resolve_torch_dtype(dtype, default=torch.float32)`, that accepts `None`, `str`, and `torch.dtype`.
  2. Support at least `bfloat16`/`bf16`, `float16`/`fp16`/`half`, and `float32`/`fp32`/`float`, including optional `"torch."` prefixes. Unknown/`None` values should fall back to `torch.float32` to preserve current behavior.
  3. Change `Qwen2_5_VisionTransformerPretrainedModel.__init__(self, config, parent_torch_dtype=None)` so `vision_config.torch_dtype` wins when non-`None`, otherwise it uses the parent dtype.
  4. Change `Qwen2_5_VLForConditionalGenerationSimple.__init__` to pass parent `config.torch_dtype` when constructing `self.visual`.
  5. Add focused tests under `fastvideo/tests/encoders/`, likely `test_qwen2_5_vl_vision_dtype.py`, using tiny `types.SimpleNamespace` configs so no HF download or real weights are required.
  6. Cover parent string `"bfloat16"`, parent `torch.bfloat16`, explicit vision override, and fallback/default fp32 behavior.
  7. Keep the fix scoped to dtype resolution; do not refactor Reason1 loading or enable the dormant visual path.
- Compatibility concerns:
  - Preserves current fp32 fallback for absent or unknown dtype.
  - Should not alter text-only Reason1 behavior.
  - No expected GPU memory increase; expected to reduce avoidable fp32 image/video input temporary allocation when the visual path is used with bf16.
- Documentation impact: none expected; this is internal model dtype plumbing.

## Validation Plan
- Do not run local tests; local environment is missing `torch` and FastVideo rules require Modal validation.
- Stage 2 targeted validation on Modal L40S from the issue worktree after implementation:
  `python -m modal run fastvideo/tests/modal/launch_l40s_job.py --install-extra none --command "pytest fastvideo/tests/encoders/test_qwen2_5_vl_vision_dtype.py -q"`
- If the implementation touches only this helper/model path and CPU-only tests, full SSIM is not warranted; no generated-pixel path should change for existing text-only pipelines.
- Before any future draft PR creation, run mandatory pre-PR gate:
  `pre-commit run --all-files`
- Future PR creation, if requested, must create only a draft PR if no PR exists, and must not change any existing PR draft status.

## Next Steps
1. Stage and commit/push this handoff-only Stage 1 state with GPG signing.
2. Present Stage 1 report to the user.
3. Await user guidance before any implementation.
