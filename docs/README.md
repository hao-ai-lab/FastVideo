# FastVideo Documentation

This directory contains the FastVideo documentation built with MkDocs.

## Build the docs locally

```bash
# Install docs dependencies
uv pip install -r requirements-mkdocs.txt
npm ci --prefix docs

# Build the browser API/validator bundle and generate cookbook catalogs
npm run build:catalog --prefix docs

# Serve docs with live reload (recommended for development)
mkdocs serve

# Or build static site
mkdocs build
```

Run these commands from the repository root with Node 22.13+ (22.x) or Node 24+. Catalog generation uses authored
YAML and Node dependencies; it does not import FastVideo or install model runtime
dependencies. `mkdocs build` and `mkdocs serve` use the generated files without
regenerating them. A small MkDocs hook checks required assets before and after
the build and explains how to prepare them if missing. Rerun `build:catalog`
after changing cookbook sources. The npm `prebuild:catalog` and `pretest:cookbook`
hooks automatically run `build:assets`, including in CI. The combined
`docs/assets/cookbook-config/cookbook-config.js`, its
`cookbook-config.LICENSE.txt`, index and catalogs are Git-ignored and published
together; do not hand-edit or commit them. Handwritten modules live in `docs/js/`
and tests in `docs/tests/`.

## View the docs

### Development server (with live reload)

```bash
mkdocs serve
```

Then open your browser to: http://127.0.0.1:8000

### Static build

```bash
mkdocs build
python -m http.server -d site/
```

Then open your browser to: http://localhost:8000

## Automatic Deployment

Documentation is automatically built and deployed to GitHub Pages when relevant
changes are pushed to the `main` branch via `.github/workflows/infra-docs.yml`.
Pull requests run the build without deploying. The normal docs job installs
`requirements-mkdocs.txt` and the Node dependencies with `npm ci --prefix docs`,
runs scoped lint/format checks, cookbook tests and asset-check tests, generates catalogs with `npm run build:catalog --prefix docs`,
and runs `mkdocs build`. Invalid authored
metadata stops the build. The generated JSON remains Git-ignored but is copied
to `site/assets/cookbook-config/` for publication. The source-only
`docs/cookbook/options.yaml` and recipe manifests are excluded from the site.
No separate metadata job or runtime server is required.

## Update documentation dependencies

Edit `requirements-mkdocs.in`, then regenerate the pinned Linux/Python 3.12 lock file:

```bash
uv pip compile requirements-mkdocs.in \
  -o requirements-mkdocs.txt \
  --python-platform x86_64-manylinux_2_28 \
  --python-version 3.12
```

## Local cookbook assets and design demo

The temporary UI lives in `examples/cookbook/`; it is not part of the published
MkDocs site. The reusable JSON catalogs and JavaScript API are published independently
of this demo.

Generate catalogs from the repository root:

```bash
npm ci --prefix docs
npm run build:catalog --prefix docs
```

This command builds the combined API/validator bundle first. Rerun it whenever
shared options, recipe manifests or baseline YAML change. Then `mkdocs build`
or `mkdocs serve` includes the generated assets. Generation loads no model
weights and performs no inference.

To preview the standalone demo without building the docs site:

```bash
python -m http.server 8195 --bind 127.0.0.1
```

Open `http://127.0.0.1:8195/examples/cookbook/`. This serves a static local page;
it does not start a model server. The page reads generated JSON from
`docs/assets/cookbook-config/`, which is ignored by Git but included in the
documentation build. Existing guide links open the published runbooks.

See [the contributor guide](contributing/cookbook_configuration.md) for recipe
authoring and local test commands, and [the API contract](cookbook/design.md)
for using the same catalogs in another UI. The temporary demo will be removed
when the final UI replaces it.

## Check cookbook changes

```bash
npm run test:cookbook --prefix docs
npm run check:cookbook --prefix docs
python -m unittest discover -s docs/tests -p 'test_*.py'
```

`npm run format:cookbook --prefix docs` formats handwritten cookbook modules,
build scripts, tooling configuration and Node tests under `docs/`. The temporary
demo adapter follows the existing `examples/` exclusion. The check commands do not change files.
The Node tests use synthetic fixtures for reusable behavior and discover every
production recipe for integration checks. They do not launch serving processes.
