# Exploration Log: Documentation Build Review Fixes

## Status: under_review

## Context

The documentation modernization branch had no existing SOP for validating
generated MkDocs pages, API-autonav paths, and Material edit actions together.

## Progress

- [x] Reproduced the API output-path collision with the pinned dependencies.
- [x] Moved the API overview to a non-colliding route.
- [x] Hid edit actions only for build-generated example pages.
- [x] Disabled strict mode consistently in CI and contributor documentation.

## Findings

- `mkdocs-api-autonav` replaces a physical source file when its virtual source
  URI matches. Moving the overview preserves established generated API URLs.
- Material renders the edit action from `page.edit_url`; clearing that value in
  `on_page_context` preserves edit links for repository-backed pages.

## Mistakes / Dead Ends

The reviewed branch was checked out in a separate worktree from the initial
working directory. Checking worktree ownership before editing avoided changing
an unrelated branch.

## Proposed Standardization

No new SOP is proposed for this small repair. Future docs-build changes should
verify both generated HTML routes and page actions with the pinned environment.
