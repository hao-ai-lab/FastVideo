# V1 API

FastVideo's V1 API provides a streamlined interface for video generation tasks with powerful customization options. This page documents the primary components of the API.

## Video Generator

This class will be the primary Python API for generating videos and images.

::: fastvideo.entrypoints.video_generator.VideoGenerator
    options:
      show_root_heading: true
      show_source: false
      members:
        - from_pretrained
      heading_level: 3

`VideoGenerator.from_pretrained()` should be the primary way of creating a new video generator.

## Configuring FastVideo

The following two classes `PipelineConfig` and `SamplingParam` are used to configure initialization and sampling parameters, respectively.

### PipelineConfig

::: fastvideo.configs.pipelines.base.PipelineConfig
    options:
      show_root_heading: true
      show_source: false
      members:
        - from_pretrained
        - dump_to_json
      heading_level: 4

### SamplingParam

::: fastvideo.configs.sample.base.SamplingParam
    options:
      show_root_heading: true
      show_source: false
      members:
        - from_pretrained
      heading_level: 4
