# `outside/` (Phase 2 config root)

This directory is a Phase 2 workaround for iterating on new YAML/JSON configs
without modifying the main repository's `fastvideo/configs/` tree.

Phase 2 does **not** rewrite config paths automatically. Put configs under this
tree and pass the real path to the entrypoint (e.g. `--config
fastvideo/distillation/outside/fastvideo/configs/distillation/foo.yaml`).

Recommended layout:

- `fastvideo/distillation/outside/fastvideo/configs/distillation/*.yaml`

Keep `outside/` for **data/config files only** (do not place importable Python
code here).
