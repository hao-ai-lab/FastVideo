# Cookbook UI trust pass evidence

Captured 2026-10-19 against the local production build (`blume build`) served by
`blume preview --host 127.0.0.1 --port 4327`, Playwright Chromium headless shell
1248, viewport 1440×1000 or 390×844, light/dark as noted. Third-party requests
were aborted; no product behavior depends on them.

| Before                             | After                             | What changed                                                                                                                                         |
| ---------------------------------- | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `before/directory-desktop.png`     | `after/directory-desktop.png`     | Command labels and source/model entry points; no preselected hardware.                                                                               |
| `before/fasth3-desktop.png`        | `after/fasth3-desktop.png`        | Hardware is explicitly unset, the component preview is gone, and fit/evidence facts sit at the hardware decision point instead of above the command. |
| `before/fasth3-narrow.png`         | `after/fasth3-narrow.png`         | Narrow layout keeps the same controls without overflow.                                                                                              |
| `before/directory-dark-narrow.png` | `after/directory-dark-narrow.png` | Dark narrow directory.                                                                                                                               |
| —                                  | `after/fasth3-mlx.png`            | MLX target language describes unified memory, not VRAM.                                                                                              |
| —                                  | `after/source-recipe-desktop.png` | Source recipe uses the same factual status language and command labels.                                                                              |

Interaction evidence is in `docs-site/tests/cookbook.spec.ts`: copy buttons are
compared against the rendered YAML/client command, a target does not alter YAML,
custom devices survive reload, and the hardware radio group starts unchecked.
