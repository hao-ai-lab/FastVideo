# V1 API

## Video Generator

This class will be the primary Python API for generating videos and images.

```{autodoc2-summary}
    fastvideo.VideoGenerator
```

VideoGenerator.from_pretrained() should be the primary way of creating a new video generator.

````{py:method} from_pretrained(model_path: str, device: typing.optional[str] = none, torch_dtype: typing.optional[torch.dtype] = none, pipeline_config: typing.optional[typing.union[str | fastvideo.v1.configs.pipelines.pipelineconfig]] = none, **kwargs) -> v1.entrypoints.video_generator.videogenerator
:canonical: v1.entrypoints.video_generator.videogenerator.from_pretrained
:classmethod:

```{autodoc2-docstring} v1.entrypoints.video_generator.videogenerator.from_pretrained
:parser: docs.source.autodoc2_docstring_parser
```

## Initialization Configuration
```{autodoc2-summary}
    fastvideo.v1.configs.pipelines.PipelineConfig
```

## Sampling Configuration
```{autodoc2-summary}
    fastvideo.v1.configs.sample.SamplingParam
```
