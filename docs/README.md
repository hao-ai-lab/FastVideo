# FastVideo documentation

Canonical guides live in this directory. The Blume application in
[docs-site](../docs-site/README.md) builds and publishes them at
[haoailab.com/FastVideo](https://haoailab.com/FastVideo/).

See [Documenting FastVideo](contributing/documentation.md) for local setup,
source ownership, page structure, validation, and deployment. The existing
GitHub Pages workflow publishes Blume from `main` with the same domain and base
path. MkDocs is no longer part of the build.

## Cookbook sources

Recipe metadata, shared JavaScript, and tests remain under `docs/`. From the
repository root:

```bash
npm ci --prefix docs
npm run build:catalog --prefix docs
npm run check:cookbook --prefix docs
npm run test:cookbook --prefix docs
python -m unittest discover -s docs/tests -p 'test_*.py'
```

The combined browser bundle, notices, index and catalogs are generated together
under `assets/cookbook-config/`. Do not hand-edit or commit those outputs.
The site preparation builds and copies them into Blume's public assets.
See the [recipe authoring guide](contributing/cookbook_configuration.md) and
[configuration API contract](cookbook/design.md) before changing the schema.
The [capacity metadata guide](contributing/capacity_metadata.md) covers reusable
hardware/component catalogs and optional recorded inference runs in existing
model manifests.
