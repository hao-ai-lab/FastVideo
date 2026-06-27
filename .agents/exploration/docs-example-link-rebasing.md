# Exploration Log: Generated Example Link Rebasing

## Status: under_review

## Context

`docs/generate_examples.py` copies Markdown from `examples/` into generated
documentation directories. Relative links remain relative to their source file,
so they can break after the Markdown is relocated.

## Progress

- [x] Reproduce the docs link-check failure.
- [x] Define link-rebasing behavior for copied Markdown.
- [x] Add regression coverage and implement the fix.
- [x] Run focused docs validation.

## Findings

The link in `examples/training/finetune/wan_t2v_1.3B/mixkit/README.md`
correctly resolves from the example directory, but the unchanged target does not
resolve from the generated `docs/training/examples/wan_t2v_1.3B_mixkit.md`.
The generator now rebases relative inline links against each generated page,
while preserving optional Markdown titles, external links, page fragments, and
examples inside fenced code blocks. Existing image assets outside `docs/` are
rewritten to raw GitHub URLs so they remain available on the published site;
missing assets remain local so link validation can still report them. The docs
link checker uses the same titled-link grammar. The generated MixKit link is
now `../data_preprocess.md`.

## Mistakes / Dead Ends

An initial edit changed the similarly named `Index.generate` method instead of
`Example.generate`; the focused unit test caught this before final validation.
The focused regression tests, complete example generation, and docs link check
all pass. The complete MkDocs page build could not run locally because MkDocs
is not installed in this environment.

## Proposed Standardization

Keep link relocation behavior covered by focused unit tests. This isolated
generator fix does not currently warrant a reusable skill or workflow.
