# Server Contracts

FastVideo's typed public API (`fastvideo.api`) is consumed by three
server-class integrations that must share one execution substrate so we
don't grow three near-duplicate progress loops:

| Consumer | Transport | Request shape | State model |
| --- | --- | --- | --- |
| [Stateless OpenAI](openai.md) | HTTP POST `/v1/videos` | `VideoGenerationsRequest` → `GenerationRequest` merged onto `ServeConfig.default_request` | Stateless; optional `ContinuationState` round-trip |
| [Streaming WebSocket](streaming.md) | WebSocket JSON + binary fMP4 | `GenerationRequest` per segment | Server-held `SessionStore`, snapshot-on-demand |
| [Dynamo native backend](dynamo.md) | Dynamo RPC | `NvCreateVideoRequest` → adapter → `GenerationRequest` | Aggregated today; disaggregated via `ContinuationState` later |

All three consume the same underlying surface:

```python
from fastvideo import VideoGenerator
from fastvideo.api import (
    ContinuationState,
    GenerationRequest,
    InputConfig,
    OutputConfig,
    SamplingConfig,
    ServeConfig,
)

# Sync today, async after PR 7.10 lands VideoGenerator.generate_async.
result = generator.generate(request)
```

These docs lock down the request/response shapes so drift between
FastVideo, the internal UI, and Dynamo can be caught at review time.
PR 8 does not ship runtime code; it ships the contract reference and
the contract tests that guard it.

## Related

- [API refactor design](../overview.md)
- Parity inventory: [`inference_schema_parity_inventory.yaml`](../inference_schema_parity_inventory.yaml)
- [Streaming server upstream plan](../../../.agents/exploration/streaming-server-upstream-plan.md)
- Draft PR (closed) that establishes the Dynamo shape:
  https://github.com/ai-dynamo/dynamo/pull/7544
