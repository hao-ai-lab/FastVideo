# Hardware catalog picker evidence

Captured 2026-10-19 against `blume build` + `blume preview` on Chromium 1248.
Third-party requests were aborted. The test fixture mirrors the generated
`capacity/hardware.json` shape (`{version, devices:[...]}`) with representative
NVIDIA and Apple entries; production data comes from Lane A's catalog.

| State | Capture |
| --- | --- |
| Directory filtered to Apple MLX with a selected M4 Max target | `after/directory-m4-max.png` |
| Model page with a selected unified-memory device and its lookup open | `after/model-unified.png` |
| Lookup with no matching devices | `after/no-results.png` |
| Narrow lookup | `after/narrow-lookup.png` |

The captures show one selected target, separate dedicated/unified wording, and
one plain empty-state line. They are not memory-fit or performance evidence.
