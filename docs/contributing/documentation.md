# Documenting FastVideo

FastVideo documentation is built with Blume and published at
[haoailab.com/FastVideo](https://haoailab.com/FastVideo/).

## Source ownership

- Edit guides under `docs/`. Keep runnable examples under `examples/` and `scripts/`.
- Edit the welcome page and Fast Models catalog under `docs-site/content/`,
  `docs-site/islands/`, and `docs-site/lib/fast-models.ts`.
- Edit sidebar labels and grouping in `docs-site/navigation.json`. Group by reader
  task. Hardware, integrations, and model-specific guides belong under Run inference.
- Python reference pages are extracted from source with Griffe. They require no
  FastVideo installation, GPU packages, or model imports.
- Do not edit `.generated/`, `public/`, or `dist/`. Preparation recreates them.

## Write a task guide

Give each page one task and one audience. Use this order where it applies:

1. Explain what the reader will do and when to use the guide.
2. List prerequisites, supported runtimes, and hardware assumptions.
3. Provide the shortest runnable example and its expected output.
4. Explain configuration choices after the working example.
5. Put troubleshooting beside the relevant command or in a short closing section.
6. Link to the next task and the relevant reference.

Keep measured results, hardware and limitations together. Put implementation
history and rejected experiments in an architecture or research page, rather
than between setup commands. Link to installation instead of repeating it.
The main quick start owns the first generation example; specialized guides
build on it. Keep model IDs in the catalog and runnable sources consistent.

## Build and check

From the repository root, use Node 24 and Python 3.12:

```bash
uv venv docs-site/.venv --python 3.12
uv pip install --python docs-site/.venv/bin/python -r requirements-docs.txt
npm ci --prefix docs
npm ci --prefix docs-site
npm run dev --prefix docs-site
```

Open the printed URL under `/FastVideo`. After editing canonical guides, rerun
`npm run prepare:content --prefix docs-site` and restart the dev server.
For a static production build:

```bash
npm run check --prefix docs-site
npm run format:check --prefix docs-site
npm run build --prefix docs-site
npm run validate --prefix docs-site
npm run test:content --prefix docs-site
npm test --prefix docs-site
```

Use `DOCS_PYTHON=/path/to/python` to select another documentation environment.
Install Chromium with `npm exec --prefix docs-site -- playwright install chromium`
if the browser suite requests it. Python renderer tests run with
`docs-site/.venv/bin/python -m unittest discover -s docs-site/tests -p 'test_*.py'`.

## Publish

The existing `.github/workflows/infra-docs.yml` builds Blume, checks content and
browser interactions, and uploads `docs-site/dist/` to GitHub Pages. Only pushes
to `main` deploy. The custom domain, GitHub Pages environment, and `/FastVideo`
base path remain the same. Pull requests validate the build without deploying.

When changing dependencies, edit `requirements-docs.in` and regenerate its lock:

```bash
uv pip compile requirements-docs.in -o requirements-docs.txt \
  --python-platform x86_64-manylinux_2_28 --python-version 3.12
```

The committed migration anchor manifest records the previous site's URLs and
fragments. It is a compatibility contract, not generated content to refresh on
each build. A deliberate URL change needs a redirect and an updated regression
check before removing an old route.
