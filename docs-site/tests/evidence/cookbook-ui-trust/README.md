# Cookbook UI evidence

Captured 2026-10-19 against the local production build (`blume build`) served by
`blume preview --host 127.0.0.1 --port 4327`, Playwright Chromium headless shell
1248, viewport 1440×1000. Third-party requests were aborted; no product behavior
depends on them.

| Before                         | After                         | What changed                                                                                                                                         |
| ------------------------------ | ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `before/directory-desktop.png` | `after/directory-desktop.png` | Command labels and source/model entry points; no preselected hardware.                                                                               |
| `before/fasth3-desktop.png`    | `after/fasth3-desktop.png`    | Hardware is explicitly unset, the component preview is gone, and fit/evidence facts sit at the hardware decision point instead of above the command. |

Narrow, dark, and MLX states were checked locally but are not committed. The
interaction tests cover those states, copied YAML/commands, reload persistence,
and the unselected/default behavior.
