# V1 API

FastVideo's V1 API provides a streamlined interface for video generation tasks with powerful customization options. This page documents the primary components of the API.

## Video Generator

This class will be the primary Python API for generating videos and images.

<api-reference path="fastvideo.entrypoints.video_generator.VideoGenerator" heading="3" members="from_pretrained" />
`VideoGenerator.from_pretrained()` should be the primary way of creating a new video generator.

## Configuring FastVideo

The following two classes `PipelineConfig` and `SamplingParam` are used to configure initialization and sampling parameters, respectively.

### PipelineConfig

<api-reference path="fastvideo.configs.pipelines.base.PipelineConfig" heading="4" members="from_pretrained,dump_to_json" />
### SamplingParam

<api-reference path="fastvideo.api.sampling_param.SamplingParam" heading="4" members="from_pretrained" />
### Returning samples without previews

Set `OutputConfig(return_samples=True, return_frames=False, save_video=False)`
to receive the decoded CPU tensor in `GenerationResult.samples` without building
preview grids or writing media. Pixel samples retain the existing float `[0, 1]`
contract. `return_frames=True` continues to return both samples and preview frames.
The legacy sampling/keyword interface accepts the same `return_samples` option.
