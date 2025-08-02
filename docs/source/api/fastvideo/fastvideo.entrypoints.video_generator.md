# {py:mod}`fastvideo.entrypoints.video_generator`

```{py:module} fastvideo.entrypoints.video_generator
```

```{autodoc2-docstring} fastvideo.entrypoints.video_generator
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VideoGenerator <fastvideo.entrypoints.video_generator.VideoGenerator>`
  - ```{autodoc2-docstring} fastvideo.entrypoints.video_generator.VideoGenerator
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.entrypoints.video_generator.logger>`
  - ```{autodoc2-docstring} fastvideo.entrypoints.video_generator.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} VideoGenerator(fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs, executor_class: type[fastvideo.worker.executor.Executor], log_stats: bool)
:canonical: fastvideo.entrypoints.video_generator.VideoGenerator

```{autodoc2-docstring} fastvideo.entrypoints.video_generator.VideoGenerator
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.entrypoints.video_generator.VideoGenerator.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} from_fastvideo_args(fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.entrypoints.video_generator.VideoGenerator
:canonical: fastvideo.entrypoints.video_generator.VideoGenerator.from_fastvideo_args
:classmethod:

```{autodoc2-docstring} fastvideo.entrypoints.video_generator.VideoGenerator.from_fastvideo_args
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} from_pretrained(model_path: str, device: str | None = None, torch_dtype: torch.dtype | None = None, **kwargs) -> fastvideo.entrypoints.video_generator.VideoGenerator
:canonical: fastvideo.entrypoints.video_generator.VideoGenerator.from_pretrained
:classmethod:

```{autodoc2-docstring} fastvideo.entrypoints.video_generator.VideoGenerator.from_pretrained
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} generate_video(prompt: str | None = None, sampling_param: fastvideo.configs.sample.SamplingParam | None = None, **kwargs) -> dict[str, typing.Any] | list[numpy.ndarray] | list[dict[str, typing.Any]]
:canonical: fastvideo.entrypoints.video_generator.VideoGenerator.generate_video

```{autodoc2-docstring} fastvideo.entrypoints.video_generator.VideoGenerator.generate_video
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} set_lora_adapter(lora_nickname: str, lora_path: str | None = None) -> None
:canonical: fastvideo.entrypoints.video_generator.VideoGenerator.set_lora_adapter

```{autodoc2-docstring} fastvideo.entrypoints.video_generator.VideoGenerator.set_lora_adapter
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} shutdown()
:canonical: fastvideo.entrypoints.video_generator.VideoGenerator.shutdown

```{autodoc2-docstring} fastvideo.entrypoints.video_generator.VideoGenerator.shutdown
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} logger
:canonical: fastvideo.entrypoints.video_generator.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.entrypoints.video_generator.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
