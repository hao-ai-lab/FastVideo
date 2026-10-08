# FastVideo Documentation

This directory contains the FastVideo documentation built with MkDocs.

## Build the docs locally

```bash
# Run from the repository root in an activated Linux/Python 3.12 environment.
# Install the docs tools and CPU dependencies for configuration metadata.
uv pip install -r requirements-mkdocs.txt -r requirements-cookbook.txt --torch-backend=cpu

# Serve docs with live reload (recommended for development)
python docs/cookbook_config.py && mkdocs serve

# Or build static site
python docs/cookbook_config.py && mkdocs build
```

The configuration builder reads model IDs from
`docs/cookbook/config-builder-models.yaml`. The exporter generates its index and
per-model catalogs under `docs/assets/cookbook-config/`; these files are ignored
by Git. Regenerate them when the selection YAML or Python configuration
definitions change. MkDocs does not run the exporter automatically.

`requirements-cookbook.txt` supplies the configuration import dependencies.
CPU PyTorch is sufficient; generating the catalogs does not load model weights
or require a GPU, custom kernels, or a full FastVideo installation. On Linux,
keep `--torch-backend=cpu` to avoid installing CUDA packages.

The pinned files target Linux/Python 3.12. For local macOS development, resolve
the cookbook dependencies from their input file so uv selects compatible
PyTorch wheels:

```bash
uv pip install -r requirements-mkdocs.txt -r requirements-cookbook.in --torch-backend=cpu
```

## View the docs

### Development server (with live reload)

```bash
python docs/cookbook_config.py && mkdocs serve
```

Then open your browser to: http://127.0.0.1:8000

### Static build

```bash
python docs/cookbook_config.py && mkdocs build
python docs/cookbook_config.py --check-site site
python -m http.server -d site/
```

Then open your browser to: http://localhost:8000

## Automatic Deployment

Documentation is automatically built and deployed to GitHub Pages when changes are pushed to the `main` branch via the `.github/workflows/infra-docs.yml` workflow.
The workflow generates the selected catalogs before building and verifies that
their index and files are included in the built site before deployment.

## Update documentation dependencies

Edit `requirements-mkdocs.in`, then regenerate the pinned Linux/Python 3.12 lock file:

```bash
uv pip compile requirements-mkdocs.in \
  -o requirements-mkdocs.txt \
  --python-platform x86_64-manylinux_2_28 \
  --python-version 3.12
```

For configuration export dependencies, edit `requirements-cookbook.in` and
regenerate its separate CPU lock after updating the MkDocs lock:

```bash
uv pip compile requirements-cookbook.in \
  -o requirements-cookbook.txt \
  --python-platform x86_64-manylinux_2_28 \
  --python-version 3.12 \
  --torch-backend=cpu --prerelease=disallow
```
