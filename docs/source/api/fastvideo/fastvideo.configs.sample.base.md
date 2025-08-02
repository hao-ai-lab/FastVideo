# {py:mod}`fastvideo.configs.sample.base`

```{py:module} fastvideo.configs.sample.base
```

```{autodoc2-docstring} fastvideo.configs.sample.base
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CacheParams <fastvideo.configs.sample.base.CacheParams>`
  - ```{autodoc2-docstring} fastvideo.configs.sample.base.CacheParams
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`SamplingParam <fastvideo.configs.sample.base.SamplingParam>`
  - ```{autodoc2-docstring} fastvideo.configs.sample.base.SamplingParam
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.configs.sample.base.logger>`
  - ```{autodoc2-docstring} fastvideo.configs.sample.base.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} CacheParams
:canonical: fastvideo.configs.sample.base.CacheParams

```{autodoc2-docstring} fastvideo.configs.sample.base.CacheParams
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} cache_type
:canonical: fastvideo.configs.sample.base.CacheParams.cache_type
:type: str
:value: >
   'none'

```{autodoc2-docstring} fastvideo.configs.sample.base.CacheParams.cache_type
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} SamplingParam
:canonical: fastvideo.configs.sample.base.SamplingParam

```{autodoc2-docstring} fastvideo.configs.sample.base.SamplingParam
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} add_cli_args(parser: typing.Any) -> typing.Any
:canonical: fastvideo.configs.sample.base.SamplingParam.add_cli_args
:staticmethod:

```{autodoc2-docstring} fastvideo.configs.sample.base.SamplingParam.add_cli_args
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} check_sampling_param()
:canonical: fastvideo.configs.sample.base.SamplingParam.check_sampling_param

```{autodoc2-docstring} fastvideo.configs.sample.base.SamplingParam.check_sampling_param
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} data_type
:canonical: fastvideo.configs.sample.base.SamplingParam.data_type
:type: str
:value: >
   'video'

```{autodoc2-docstring} fastvideo.configs.sample.base.SamplingParam.data_type
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} enable_teacache
:canonical: fastvideo.configs.sample.base.SamplingParam.enable_teacache
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.configs.sample.base.SamplingParam.enable_teacache
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} fps
:canonical: fastvideo.configs.sample.base.SamplingParam.fps
:type: int
:value: >
   24

```{autodoc2-docstring} fastvideo.configs.sample.base.SamplingParam.fps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} from_pretrained(model_path: str) -> fastvideo.configs.sample.base.SamplingParam
:canonical: fastvideo.configs.sample.base.SamplingParam.from_pretrained
:classmethod:

```{autodoc2-docstring} fastvideo.configs.sample.base.SamplingParam.from_pretrained
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} guidance_rescale
:canonical: fastvideo.configs.sample.base.SamplingParam.guidance_rescale
:type: float
:value: >
   0.0

```{autodoc2-docstring} fastvideo.configs.sample.base.SamplingParam.guidance_rescale
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} guidance_scale
:canonical: fastvideo.configs.sample.base.SamplingParam.guidance_scale
:type: float
:value: >
   1.0

```{autodoc2-docstring} fastvideo.configs.sample.base.SamplingParam.guidance_scale
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} height
:canonical: fastvideo.configs.sample.base.SamplingParam.height
:type: int
:value: >
   720

```{autodoc2-docstring} fastvideo.configs.sample.base.SamplingParam.height
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} image_path
:canonical: fastvideo.configs.sample.base.SamplingParam.image_path
:type: str | None
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.sample.base.SamplingParam.image_path
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} negative_prompt
:canonical: fastvideo.configs.sample.base.SamplingParam.negative_prompt
:type: str | None
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.sample.base.SamplingParam.negative_prompt
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_frames
:canonical: fastvideo.configs.sample.base.SamplingParam.num_frames
:type: int
:value: >
   125

```{autodoc2-docstring} fastvideo.configs.sample.base.SamplingParam.num_frames
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_frames_round_down
:canonical: fastvideo.configs.sample.base.SamplingParam.num_frames_round_down
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.configs.sample.base.SamplingParam.num_frames_round_down
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_inference_steps
:canonical: fastvideo.configs.sample.base.SamplingParam.num_inference_steps
:type: int
:value: >
   50

```{autodoc2-docstring} fastvideo.configs.sample.base.SamplingParam.num_inference_steps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_videos_per_prompt
:canonical: fastvideo.configs.sample.base.SamplingParam.num_videos_per_prompt
:type: int
:value: >
   1

```{autodoc2-docstring} fastvideo.configs.sample.base.SamplingParam.num_videos_per_prompt
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} output_path
:canonical: fastvideo.configs.sample.base.SamplingParam.output_path
:type: str
:value: >
   'outputs/'

```{autodoc2-docstring} fastvideo.configs.sample.base.SamplingParam.output_path
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} output_video_name
:canonical: fastvideo.configs.sample.base.SamplingParam.output_video_name
:type: str | None
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.sample.base.SamplingParam.output_video_name
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} prompt
:canonical: fastvideo.configs.sample.base.SamplingParam.prompt
:type: str | list[str] | None
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.sample.base.SamplingParam.prompt
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} prompt_path
:canonical: fastvideo.configs.sample.base.SamplingParam.prompt_path
:type: str | None
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.sample.base.SamplingParam.prompt_path
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} return_frames
:canonical: fastvideo.configs.sample.base.SamplingParam.return_frames
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.configs.sample.base.SamplingParam.return_frames
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} save_video
:canonical: fastvideo.configs.sample.base.SamplingParam.save_video
:type: bool
:value: >
   True

```{autodoc2-docstring} fastvideo.configs.sample.base.SamplingParam.save_video
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} seed
:canonical: fastvideo.configs.sample.base.SamplingParam.seed
:type: int
:value: >
   1024

```{autodoc2-docstring} fastvideo.configs.sample.base.SamplingParam.seed
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} update(source_dict: dict[str, typing.Any]) -> None
:canonical: fastvideo.configs.sample.base.SamplingParam.update

```{autodoc2-docstring} fastvideo.configs.sample.base.SamplingParam.update
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} width
:canonical: fastvideo.configs.sample.base.SamplingParam.width
:type: int
:value: >
   1280

```{autodoc2-docstring} fastvideo.configs.sample.base.SamplingParam.width
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} logger
:canonical: fastvideo.configs.sample.base.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.configs.sample.base.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
