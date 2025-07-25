# V1 API

FastVideo's V1 API provides a streamlined interface for video generation tasks with powerful customization options. This page documents the primary components of the API.

## Video Generator

This class will be the primary Python API for generating videos and images.

```{autodoc2-summary}
    fastvideo.VideoGenerator
```

`````{py:class} VideoGenerator(fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs, executor_class: type[fastvideo.worker.executor.Executor], log_stats: bool)
:canonical: fastvideo.entrypoints.video_generator.VideoGenerator

```{autodoc2-docstring} fastvideo.entrypoints.video_generator.VideoGenerator
:parser: docs.source.autodoc2_docstring_parser
```

`VideoGenerator.from_pretrained()` should be the primary way of creating a new video generator.

````{py:method} from_pretrained(model_path: str, device: typing.Optional[str] = None, torch_dtype: typing.Optional[torch.dtype] = None, pipeline_config: typing.Optional[typing.Union[str | fastvideo.configs.pipelines.PipelineConfig]] = None, **kwargs) -> fastvideo.entrypoints.video_generator.VideoGenerator
:canonical: fastvideo.entrypoints.video_generator.VideoGenerator.from_pretrained
:classmethod:

```{autodoc2-docstring} fastvideo.entrypoints.video_generator.VideoGenerator.from_pretrained
:parser: docs.source.autodoc2_docstring_parser
```


## Configuring FastVideo

The follow two classes `PipelineConfig` and `SamplingParam` are used to configure initialization and sampling parameters, respectively.

### PipelineConfig
```{autodoc2-summary}
    fastvideo.PipelineConfig
```

`````{py:class} PipelineConfig
:canonical: fastvideo.configs.pipelines.base.PipelineConfig

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} from_pretrained(model_path: str) -> fastvideo.configs.pipelines.base.PipelineConfig
:canonical: fastvideo.configs.pipelines.base.PipelineConfig.from_pretrained
:classmethod:

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig.from_pretrained
:parser: docs.source.autodoc2_docstring_parser
```


````{py:method} dump_to_json(file_path: str)
:canonical: fastvideo.configs.pipelines.base.PipelineConfig.dump_to_json

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig.dump_to_json
:parser: docs.source.autodoc2_docstring_parser
```


### SamplingParam

```{autodoc2-summary}
    fastvideo.SamplingParam
```

`````{py:class} SamplingParam
:canonical: fastvideo.configs.sample.base.SamplingParam

```{autodoc2-docstring} fastvideo.configs.sample.base.SamplingParam
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} from_pretrained(model_path: str) -> fastvideo.configs.sample.base.SamplingParam
:canonical: fastvideo.configs.sample.base.SamplingParam.from_pretrained
:classmethod:

```{autodoc2-docstring} fastvideo.configs.sample.base.SamplingParam.from_pretrained
:parser: docs.source.autodoc2_docstring_parser
```
