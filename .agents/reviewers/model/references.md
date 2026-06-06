# Model reviewer — references

Files and docs to consult before flagging. Review these rather than guessing.

## Base classes / patterns

- `fastvideo/models/dits/base.py` — DiT base class.
- `fastvideo/models/vaes/` — VAE implementations.
- `fastvideo/models/encoders/` — text/image encoders.
- `fastvideo/models/loader/` — HF safetensors loading + shape checks.
- `fastvideo/registry.py` — unified registration surface.
- `fastvideo/configs/models/dits/base.py` (if present) or look at
  `fastvideo/configs/models/dits/wanvideo.py` for arch config pattern.

## Good reference implementations

When reviewing a new model, compare against one of these that's already
landed:

- **Wan T2V** (most mature): `fastvideo/models/dits/wanvideo.py` +
  `fastvideo/configs/models/dits/wanvideo.py` +
  `fastvideo/pipelines/basic/wan/`.
- **LTX-2**: `fastvideo/models/dits/ltx2.py` + `fastvideo/pipelines/basic/ltx2/`.
- **HunyuanVideo** (recently ported, PR #1175 for training plugin): a good
  example of config + registry + training wrapper.
- **Flux / SD3**: `fastvideo/models/dits/sd3.py` — T2I-style.
- **Cosmos 2.5** (recently added, PR #1224/#1227): modern port with training.

## SSIM test pattern

- `fastvideo/tests/ssim/conftest.py` — fixtures.
- `fastvideo/tests/ssim/inference_similarity_utils.py` — test harness.
- `fastvideo/tests/ssim/reference_utils.py` — HF reference download.
- `fastvideo/tests/ssim/test_wan_t2v_similarity.py` — canonical pattern.
- `fastvideo/tests/ssim/test_ltx2_similarity.py` — another known-good test.

When a PR adds a new model without an SSIM test, quote this path and ask for
one: `fastvideo/tests/ssim/test_<model>_similarity.py`.

## Documentation

- `docs/contributing/coding_agents.md` — the repo's own "how to add a model"
  guide. Quote from this when a PR skips steps.
- `docs/design/overview.md` — pipeline / config architecture.

## Cross-reference (when in doubt)

- Mergify label rules: `.github/mergify.yml` (paths that trigger `scope: model`).
- AGENTS.md / CLAUDE.md: `/Users/willlin/src/FastVideo/AGENTS.md` — style and
  testing expectations.
- Contributor PR template: `.github/PULL_REQUEST_TEMPLATE.md`.

## Seed references

- Skill to seed new SSIM references: `.agents/skills/seed-ssim-references/SKILL.md`.
- HF reference videos repo: `FastVideo/ssim-reference-videos`.
