# SPDX-License-Identifier: Apache-2.0
# Importing continuation registers the "ltx2.v1" continuation kind with
# the public compat layer so GenerationRequest.state(kind="ltx2.v1") is
# recognized on the public API boundary.
from fastvideo.pipelines.basic.ltx2 import continuation  # noqa: F401
