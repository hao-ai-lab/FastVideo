# FastVideo Documentation

This directory contains the FastVideo documentation built with MkDocs.

## Build the docs locally

```bash
# Install docs dependencies
uv pip install -r requirements-mkdocs.txt

# Serve docs with live reload (recommended for development)
mkdocs serve

# Or build static site
mkdocs build
```

Run these commands from the repository root. Ordinary docs builds need only the
MkDocs dependencies and do not regenerate cookbook metadata. For local cookbook
assets, run the explicit export described below before building or serving.

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
Pull requests run the build without deploying. The workflow watches docs and
examples plus the configuration, registry and serving definitions used by the exporter.

The workflow's `cookbook_metadata` job installs the CPU dependencies from
`requirements-cookbook.txt`, runs `python docs/cookbook_config.py`, and uploads
the `cookbook-metadata` artifact. The dependent `build` job installs only
`requirements-mkdocs.txt`, downloads that artifact into
`docs/assets/cookbook-config/`, and runs `mkdocs build`. Both jobs use the same
source revision in the same workflow run; export failures block deployment.
The JSON remains Git-ignored but is copied to `site/assets/cookbook-config/`
for publication. No runtime server is involved.

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

For a separate CPU configuration environment, install the metadata dependencies
and export from the repository root:

```bash
uv pip install -r requirements-cookbook.txt --torch-backend=cpu
python docs/cookbook_config.py
```

These pins target Linux/Python 3.12. On macOS, use `requirements-cookbook.in`
instead so the installer selects compatible wheels. Export inspects FastVideo
configuration without loading weights or running inference. Rerun it whenever
its source definitions, manifests or baseline YAML change. Then `mkdocs build`
or `mkdocs serve` in the docs environment includes the generated assets.

To preview the standalone demo without building the docs site:

```bash
python -m http.server 8195 --bind 127.0.0.1
```

Open `http://127.0.0.1:8195/examples/cookbook/`. This serves a static local page;
it does not start a model server. The page reads generated JSON from
`docs/assets/cookbook-config/`, which is ignored by Git but included in the
documentation build. Existing guide links open the published runbooks.

See [the contributor guide](contributing/cookbook_configuration.md) for recipe
authoring and local test commands, and [the API contract](design/serving-cookbook.md)
for using the same catalogs in another UI. The final UI is a separate follow-up.

To update the metadata dependency lock:

```bash
uv pip compile requirements-cookbook.in \
  -o requirements-cookbook.txt \
  --python-platform x86_64-manylinux_2_28 \
  --python-version 3.12 \
  --torch-backend=cpu --prerelease=disallow
```
