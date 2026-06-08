---
name: add-model-01-prep
description: Use at the start of a FastVideo model port to gather required inputs, inspect/download HF weights, clone and install the official reference repo in the current environment, create a local_tests README skeleton, and produce a handoff before conversion or implementation.
---

# Add Model Prep

## Goal

Prepare external assets and the shared parity-test environment for a FastVideo
model port. Stop before writing conversion scripts, model components, pipeline
code, registry entries, or executable parity tests.

## Ask First

Ask once, then proceed if the HF token is already exported:

```text
Before prep: (1) official reference repo or Diffusers pipeline URL, (2) HF repo
id or local weights path and whether it has a root model_index.json, (3) target
model_family, (4) workload types, (5) which token env var is exported:
HF_TOKEN, HUGGINGFACE_HUB_TOKEN, or HF_API_KEY, (6) may I stage clone and
weights under the FastVideo repo root, and (7) may I install official reference
dependencies into the current FastVideo conda/env for parity tests?
```

Useful optional inputs: `pipeline_class`, `reference_dir`, `hf_revision`,
`official_revision`, `reuse_hints`, `download_scope`.

## Rules

- Follow `../add-model/shared/common_rules.md` for token/auth safety, state files,
  escape hatches, and skip/pass semantics.
- Run from the FastVideo repo root.
- Use repo-relative defaults: `<ReferenceDir>/`,
  `official_weights/<model_family>/`, `converted_weights/<model_family>/`.
- Install official reference deps into the current FastVideo environment, not a
  new venv/conda env, so parity tests run both implementations with one shared
  numeric stack.
- If the reference is a Diffusers class/package instead of a cloneable repo,
  record import path and version instead of cloning.
- Prep may create only the local-test README and `PORT_STATUS.md` skeletons;
  executable `.py` parity tests belong to `../add-model-02-parity/SKILL.md`.

## Escape Hatches

Follow `../add-model/shared/common_rules.md`. Prep-specific ask cases include
overwriting an existing clone or weight directory, installing untrusted/private
deps, choosing between incompatible official references, large downloads outside
the agreed scope, or missing gated-repo auth setup by env var name.

## Workflow

1. Verify the repo:

```bash
git rev-parse --show-toplevel
```

Expected markers: `fastvideo/`, `scripts/checkpoint_conversion/`,
`scripts/huggingface/download_hf.py`, `fastvideo/registry.py`.

2. Inspect HF or local weight layout:

```bash
python ".agents/skills/add-model-01-prep/scripts/inspect_hf_layout.py" \
    "Org/Model" \
    --revision "<revision>" \
    --json
```

For a local path, replace `Org/Model` with `/path/to/weights`. Record
`source_layout`, `needs_conversion`, `model_index_class`, and
`components_seen`.

3. Download HF weights if needed:

```bash
python ".agents/skills/add-model-01-prep/scripts/download_hf_weights.py" \
    "Org/Model" \
    "official_weights/<model_family>" \
    --revision "<revision>"
```

For selected files, repeat `--file-name`. For partial snapshots, repeat
`--allow-pattern` or `--ignore-pattern`. If the user provided a local path,
record it instead of copying large weights by default.

4. Clone the official reference repo if applicable:

```bash
python ".agents/skills/add-model-01-prep/scripts/clone_reference_repo.py" \
    "<official_repo_url>" \
    "<ReferenceDir>" \
    --branch "<tag-or-branch>" \
    --commit "<commit-sha>" \
    --update-gitignore
```

Omit `--branch`, `--commit`, or `--update-gitignore` when not needed. The
helper refuses to overwrite existing paths and prints remote/HEAD instead.

5. Keep prep assets ignored. Ensure `.gitignore` includes relevant entries:

```gitignore
/<ReferenceDir>/
/official_weights/
/converted_weights/
```

6. Follow the official repo's setup instructions in the current environment.
Inspect dependency files and README install docs before installing anything:

- `README*`, install docs, or model-card instructions.
- `requirements*.txt`, `pyproject.toml`, `setup.py`, `environment.yml`.

Use the current FastVideo conda/env. Do not create a new env even if upstream
docs recommend one; translate the needed install commands into the active env.
Prefer editable/no-deps first so the official source is importable without
changing shared pins:

```bash
uv pip install --no-deps -e ./<ReferenceDir>
```

Then install only missing official deps needed for parity imports. Stop before
installing requirements that would change FastVideo's core stack. If upstream
requires private/non-PyPI deps, record that parity needs a local stub helper
rather than pretending setup is complete.

7. Create the model-family local test skeleton and top-level port state file:

```bash
mkdir -p tests/local_tests/<model_family>
cp ".agents/skills/add-model-01-prep/templates/local_tests_readme.md" \
    tests/local_tests/<model_family>/README.md
cp ".agents/skills/add-model-01-prep/templates/port_status.md" \
    tests/local_tests/<model_family>/PORT_STATUS.md
```

Edit every placeholder in the README and `PORT_STATUS.md`. The README gives
later review agents enough information to reproduce the shared environment and
run/review parity work:

- official code URL or import path, local clone path, and commit/version;
- HF URL or local weight path, revision, access notes, and token env var name
  only;
- commands already run and any blocked official dependency installs;
- shared-env install commands to re-run without changing core pins;
- expected local parity test paths and pytest commands;
- private-dependency stubs or known setup gaps;
- PR/review notes explaining which parity tests are required before handoff.

Do not include raw tokens, absolute cache paths that are not repo-reproducible,
or large generated outputs. If prep is blocked before imports work, still create
the README with `official_env_status=blocked` and the exact blocker.

`PORT_STATUS.md` must follow `../add-model/contracts/port_state.md`. Record open
questions and prep issues immediately, using stable IDs such as `Q001` and
`I001`. Keep resolved questions/issues in the table with a resolution instead of
deleting them.

## Handoff

End with the canonical prep handoff contract from
`../add-model/contracts/prep_handoff.md` and update the shared state files before
handoff.

## Helper Scripts

- `scripts/inspect_hf_layout.py`: classify HF/local layout.
- `scripts/download_hf_weights.py`: download HF snapshot or selected files.
- `scripts/clone_reference_repo.py`: clone reference repo safely.
