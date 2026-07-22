"""Gates and gate tooling for the Wan2.1 family — the measurement side.

Everything in here observes or judges the logic code above it; nothing in
here is imported by it (card/model/loop/pipeline never import gates — the
one-way rule). Contents: golden probe definitions and IO (``goldens``),
per-implementation anchor adapters and comparisons (``anchor``), the official
capture shim (``capture_official``), the cross-implementation comparison CLI
(``compare_upstream``), and the delta-decomposition diagnostic (``probe_dit``).
Core orchestration (tiers, ledger, CLI) stays in ``fastvideo2/verify.py``;
``reference.py`` stays beside the model because it is the executable *spec*
of the family, not a measurement of it.
"""
