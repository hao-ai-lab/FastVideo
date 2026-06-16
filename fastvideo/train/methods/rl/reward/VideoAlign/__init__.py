"""Vendored runtime subset of VideoAlign.

Source: https://github.com/KlingAIResearch/VideoAlign
Commit: aba26b658fec7d9fd30c295187b548ea673c8769
Purpose: Runtime reward inference integration for FastVideo GenRL.

This is temporary minimal vendoring for PR integration. It is expected to be
cleaned up and normalized later.

Porting rules:
- Include only files required by the runtime import closure used by FastVideo.
- When an upstream file is required, copy the entire file faithfully.
- Only adjust imports as needed to make the vendored code import through package
  paths instead of sys.path mutation or ambiguous top-level imports.
- Do not perform style, typing, or behavioral cleanup as part of this vendoring
  step.
"""
