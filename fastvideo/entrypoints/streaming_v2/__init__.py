# SPDX-License-Identifier: Apache-2.0
"""Stateless HTTP + SSE server that generates one video segment per request.

``fastvideo serve --config <yaml>`` starts this server when the config carries
a ``streaming_v2: {}`` block. The server holds one ``VideoGenerator`` and keeps
no state between requests; every request carries all of its inputs.

Routes:

* ``POST /v1/streamv2/generate`` takes a JSON body (see
  :class:`~fastvideo.entrypoints.streaming_v2.messages.GenerateRequest`) and
  answers with an SSE stream: ``last_frame`` (only when requested),
  ``video_start``, one or more ``video_chunk``, then ``done``. A failure after
  the stream starts ends it with ``error``. An invalid request is rejected
  before streaming with HTTP 400 and ``{"code": "invalid_request", ...}``.
* ``GET /v1/streamv2/capabilities`` returns the served model's facts.
* ``GET /v1/streamv2/health`` reports that the server is serving.

Generation runs one request at a time; streaming a finished segment overlaps
the next request's generation.
"""
