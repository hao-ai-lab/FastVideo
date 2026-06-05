"""Vendored runtime subset of VideoAlign.

Source: https://github.com/KlingAIResearch/VideoAlign
Commit: 219ab9db64c045e5181a2202d11f686439351292
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
