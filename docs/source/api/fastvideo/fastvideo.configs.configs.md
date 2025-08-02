# {py:mod}`fastvideo.configs.configs`

```{py:module} fastvideo.configs.configs
```

```{autodoc2-docstring} fastvideo.configs.configs
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PreprocessConfig <fastvideo.configs.configs.PreprocessConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.configs.PreprocessConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} PreprocessConfig
:canonical: fastvideo.configs.configs.PreprocessConfig

```{autodoc2-docstring} fastvideo.configs.configs.PreprocessConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} add_cli_args(parser: fastvideo.utils.FlexibleArgumentParser, prefix: str = 'preprocess') -> fastvideo.utils.FlexibleArgumentParser
:canonical: fastvideo.configs.configs.PreprocessConfig.add_cli_args
:staticmethod:

```{autodoc2-docstring} fastvideo.configs.configs.PreprocessConfig.add_cli_args
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} check_preprocess_config() -> None
:canonical: fastvideo.configs.configs.PreprocessConfig.check_preprocess_config

```{autodoc2-docstring} fastvideo.configs.configs.PreprocessConfig.check_preprocess_config
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} dataloader_num_workers
:canonical: fastvideo.configs.configs.PreprocessConfig.dataloader_num_workers
:type: int
:value: >
   1

```{autodoc2-docstring} fastvideo.configs.configs.PreprocessConfig.dataloader_num_workers
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} dataset_output_dir
:canonical: fastvideo.configs.configs.PreprocessConfig.dataset_output_dir
:type: str
:value: >
   './output'

```{autodoc2-docstring} fastvideo.configs.configs.PreprocessConfig.dataset_output_dir
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} dataset_path
:canonical: fastvideo.configs.configs.PreprocessConfig.dataset_path
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} fastvideo.configs.configs.PreprocessConfig.dataset_path
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} do_temporal_sample
:canonical: fastvideo.configs.configs.PreprocessConfig.do_temporal_sample
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.configs.configs.PreprocessConfig.do_temporal_sample
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} drop_short_ratio
:canonical: fastvideo.configs.configs.PreprocessConfig.drop_short_ratio
:type: float
:value: >
   1.0

```{autodoc2-docstring} fastvideo.configs.configs.PreprocessConfig.drop_short_ratio
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} flush_frequency
:canonical: fastvideo.configs.configs.PreprocessConfig.flush_frequency
:type: int
:value: >
   256

```{autodoc2-docstring} fastvideo.configs.configs.PreprocessConfig.flush_frequency
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} from_kwargs(kwargs: dict[str, typing.Any]) -> typing.Optional[fastvideo.configs.configs.PreprocessConfig]
:canonical: fastvideo.configs.configs.PreprocessConfig.from_kwargs
:classmethod:

```{autodoc2-docstring} fastvideo.configs.configs.PreprocessConfig.from_kwargs
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} max_height
:canonical: fastvideo.configs.configs.PreprocessConfig.max_height
:type: int
:value: >
   480

```{autodoc2-docstring} fastvideo.configs.configs.PreprocessConfig.max_height
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} max_width
:canonical: fastvideo.configs.configs.PreprocessConfig.max_width
:type: int
:value: >
   848

```{autodoc2-docstring} fastvideo.configs.configs.PreprocessConfig.max_width
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} model_path
:canonical: fastvideo.configs.configs.PreprocessConfig.model_path
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} fastvideo.configs.configs.PreprocessConfig.model_path
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_frames
:canonical: fastvideo.configs.configs.PreprocessConfig.num_frames
:type: int
:value: >
   163

```{autodoc2-docstring} fastvideo.configs.configs.PreprocessConfig.num_frames
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} preprocess_video_batch_size
:canonical: fastvideo.configs.configs.PreprocessConfig.preprocess_video_batch_size
:type: int
:value: >
   2

```{autodoc2-docstring} fastvideo.configs.configs.PreprocessConfig.preprocess_video_batch_size
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} samples_per_file
:canonical: fastvideo.configs.configs.PreprocessConfig.samples_per_file
:type: int
:value: >
   64

```{autodoc2-docstring} fastvideo.configs.configs.PreprocessConfig.samples_per_file
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} speed_factor
:canonical: fastvideo.configs.configs.PreprocessConfig.speed_factor
:type: float
:value: >
   1.0

```{autodoc2-docstring} fastvideo.configs.configs.PreprocessConfig.speed_factor
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} train_fps
:canonical: fastvideo.configs.configs.PreprocessConfig.train_fps
:type: int
:value: >
   30

```{autodoc2-docstring} fastvideo.configs.configs.PreprocessConfig.train_fps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} training_cfg_rate
:canonical: fastvideo.configs.configs.PreprocessConfig.training_cfg_rate
:type: float
:value: >
   0.0

```{autodoc2-docstring} fastvideo.configs.configs.PreprocessConfig.training_cfg_rate
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} video_length_tolerance_range
:canonical: fastvideo.configs.configs.PreprocessConfig.video_length_tolerance_range
:type: float
:value: >
   2.0

```{autodoc2-docstring} fastvideo.configs.configs.PreprocessConfig.video_length_tolerance_range
:parser: docs.source.autodoc2_docstring_parser
```

````

`````
