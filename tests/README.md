# Tests

- `tests/local_tests/` are local-only tests that require a checked-out `LTX-2/`
  directory under the repo root (`FastVideo/LTX-2`). Without that repo, they
  will skip or fail.
- The CI-backed test suite still lives in `fastvideo/tests/`.
- Eventually, all tests will move under `tests/`.
