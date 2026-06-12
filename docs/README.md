# FastVideo Documentation

This directory contains the FastVideo documentation built with MkDocs.

The Quick Start and Advanced Tuning pages embed an interactive **config
generator** — a separate Next.js app in [`consumer-guide/`](../consumer-guide/)
that is built independently and copied into the docs site under
`/config-generator/`. To work on that app itself, see
[`consumer-guide/README.md`](../consumer-guide/README.md).

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

> **Note:** `mkdocs serve` only builds the docs. The embedded config generator
> is **not** built, so its iframe on the Quick Start / Advanced Tuning pages
> shows a 404. This is expected — use "Full local preview" below to see it.

### Full local preview (docs + config generator)

To preview exactly what gets deployed, replicate the CI build:

```bash
# 1. Build the docs
mkdocs build

# 2. Build the config generator
cd consumer-guide
pnpm install
NEXT_BASE_PATH=/config-generator pnpm build
cd ..

# 3. Copy it into the docs site
mkdir -p site/config-generator
cp -R consumer-guide/out/. site/config-generator/

# 4. Serve the merged site
python -m http.server -d site/
```

Then open your browser to: http://localhost:8000

## Automatic Deployment

Documentation is automatically built and deployed to GitHub Pages when changes
are pushed to the `main` branch via the `.github/workflows/infra-docs.yml`
workflow. The same workflow also builds the config generator and copies it into
the deployed site.
