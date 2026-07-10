# FastVideo API Reference

FastVideo exposes a small public API for loading pipelines and generating media.
The detailed package reference is generated automatically from the source code.

## Core API

- [`fastvideo.VideoGenerator`][] loads a pipeline and provides the high-level
  generation interface.
- [`fastvideo.PipelineConfig`][] describes the components required by a pipeline.
- [`fastvideo.SamplingParam`][] contains generation-time sampling parameters.

## Package Reference

- [API contracts and sampling][fastvideo.api]
- [Configuration][fastvideo.configs]
- [Models][fastvideo.models]
- [Pipelines][fastvideo.pipelines]
- [Training][fastvideo.train]
- [Distributed runtime][fastvideo.distributed]
- [Attention backends][fastvideo.attention]
- [Layers][fastvideo.layers]
- [Entrypoints][fastvideo.entrypoints]

## Guides

- [V1 API](../getting_started/v1_api.md)
- [Inference quick start](../inference/inference_quick_start.md)
- [Training overview](../training/overview.md)
