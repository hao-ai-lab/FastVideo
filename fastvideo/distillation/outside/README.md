# `outside/` overlay

This directory is a Phase 2 workaround for iterating on new YAML/JSON configs
without modifying the main repository's `fastvideo/configs/` tree.

The distillation config loader resolves file paths via an overlay lookup:

- if a run config references `fastvideo/configs/foo.json`, it will first check
  `fastvideo/distillation/outside/fastvideo/configs/foo.json`;
- otherwise it falls back to the original path.

Keep `outside/` for **data/config files only** (do not place importable Python
code here).

