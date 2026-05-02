# Local Tests

Local-only parity, smoke, and component tests for FastVideo model ports. They
compare FastVideo against the official reference implementations and are
**skipped in CI**; run them locally on a single GPU (or CPU where noted).

For the CI-backed test suite, see [`fastvideo/tests/`](../../fastvideo/tests/).

## Layout

Tests are organized by **model family**, one directory per port. Each family
directory follows the layout produced by the
[`add-model-prep`](https://github.com/anthropic-skills/add-model-prep) skill:

```
tests/local_tests/<family>/
├── README.md         # reference assets, setup, test inventory
├── __init__.py
└── test_<family>_<component>_*.py
```

## Families

| Family | Workload | Dir |
|---|---|---|
| Hunyuan GameCraft | T2V / I2V | [`gamecraft/`](./gamecraft/) |
| GEN3C | T2V | [`gen3c/`](./gen3c/) |
| Kandinsky-5 | T2V | [`kandinsky5/`](./kandinsky5/) |
| LTX-2 | T2V (+ audio) | [`ltx2/`](./ltx2/) |
| Stable Diffusion 3.5 | T2I | [`sd35/`](./sd35/) |
| Stable Audio Open 1.0 | T2A | [`stable_audio/`](./stable_audio/) |
| Wan2.2 | I2V | [`wan22/`](./wan22/) |

## Running a family

```bash
pytest tests/local_tests/<family>/ -v -s
```

See each family's `README.md` for setup (clones, weights, env vars, gated
tokens) and the per-component parity coverage.

## Adding a new family

Run the [`add-model-prep`](https://github.com/anthropic-skills/add-model-prep)
skill from the FastVideo repo root. It scaffolds the family directory and
`README.md` and stages reference clones + weights. (In-progress ports may also
add a `PORT_STATUS.md` from the same skill template; remove it once the port
lands.)
