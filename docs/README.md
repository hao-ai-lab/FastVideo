# FastVideo Documentation

This directory contains the FastVideo documentation built with MkDocs.

## Build the docs locally

```bash
# Install dependencies
uv pip install -r requirements-mkdocs.txt

# Serve docs with live reload (recommended for development)
mkdocs serve

# Or build static site
mkdocs build
```

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

Documentation is automatically built and deployed to GitHub Pages when changes are pushed to the `main` branch via the `.github/workflows/infra-docs.yml` workflow.

## Update documentation dependencies

Edit `requirements-mkdocs.in`, then regenerate the pinned Linux/Python 3.12 lock file:

```bash
uv pip compile requirements-mkdocs.in \
  -o requirements-mkdocs.txt \
  --python-platform x86_64-manylinux_2_28 \
  --python-version 3.12
```

## Local cookbook design demo

The cookbook exporter and JavaScript API are a separate development prototype.
The temporary UI lives in `examples/cookbook/`; it is not part of the published
MkDocs site or documentation deployment workflow.

From the repository root, in an environment with the FastVideo configuration
dependencies installed:

```bash
python docs/cookbook_config.py
python -m http.server 8195 --bind 127.0.0.1
```

Open `http://127.0.0.1:8195/examples/cookbook/`. This serves a static local page;
it does not start a model server. The page reads generated JSON from
`docs/assets/cookbook-config/`, which is ignored by Git and excluded from the
normal documentation build. Existing guide links open the published runbooks.

For a separate CPU-only development environment, install the metadata dependencies
from `requirements-cookbook.in` with `--torch-backend=cpu`. The Linux/Python 3.12
pins are in `requirements-cookbook.txt`; macOS should use the input file so the
installer selects compatible wheels. No model weights are loaded by export.

See [the contributor guide](contributing/cookbook_configuration.md) for recipe
authoring and local test commands, and [the API contract](design/serving-cookbook.md)
for using the same catalogs in another UI. Production integration belongs to
the final UI follow-up.

To update the metadata dependency lock:

```bash
uv pip compile requirements-cookbook.in \
  -o requirements-cookbook.txt \
  --python-platform x86_64-manylinux_2_28 \
  --python-version 3.12 \
  --torch-backend=cpu --prerelease=disallow
```
