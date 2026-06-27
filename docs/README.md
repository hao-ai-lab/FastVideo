# FastVideo Documentation

This directory contains the FastVideo documentation built with MkDocs.

## Build the docs locally

```bash
# Install dependencies
uv pip install -r requirements-mkdocs.txt

# Serve docs with live reload (recommended for development)
mkdocs serve

# Or build static site
mkdocs build --strict
```

## View the docs

### Development server (with live reload)

```bash
mkdocs serve
```

Then open your browser to: http://127.0.0.1:8000

### Static build

```bash
mkdocs build --strict
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
