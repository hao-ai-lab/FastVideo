# FastVideo Documentation

This directory contains the FastVideo documentation built with MkDocs.

## Build the docs locally

```bash
# Install dependencies
pip install -r requirements-mkdocs.txt

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

Documentation is automatically built and deployed to GitHub Pages when changes are pushed to the `main` branch via the `.github/workflows/docs.yml` workflow.
