# SPDX-License-Identifier: Apache-2.0
"""Contract tests guarding FastVideo's public API against drift.

These tests run against the public surface only (`fastvideo.VideoGenerator`,
`fastvideo.api.*`) — never via private helpers. They fail at FastVideo CI
if a change breaks the shape the Dynamo backend package and the private
Dreamverse adapter depend on, so drift is caught here before it reaches
downstream integrators.
"""
