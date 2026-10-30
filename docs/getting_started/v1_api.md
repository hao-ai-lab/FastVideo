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

### Choosing an output filename

Set `request.output.output_video_name` to choose a filename when
`request.output.output_path` is a directory:

```python
result = generator.generate({
    "prompt": "A waterfall in the mountains",
    "output": {"output_path": "outputs/", "output_video_name": "my_clip"},
})
# Saves outputs/my_clip.mp4 when that file does not already exist.
```

An explicit file path such as `outputs/custom.mp4` takes precedence over
`output_video_name`. Without a name, the filename is derived from the prompt
(or the input video for prompt-less super-resolution pipelines). Names are
sanitized, and existing files are preserved by adding numeric suffixes such as
`my_clip_1.mp4`, including when processing multiple prompts. A matching extension
in the name is not duplicated. Image workloads accept PNG, JPEG, or WebP names
and default to PNG; audio workloads use WAV.

The legacy `SamplingParam` and `generate_video` keyword interfaces accept the
same `output_video_name` option.
