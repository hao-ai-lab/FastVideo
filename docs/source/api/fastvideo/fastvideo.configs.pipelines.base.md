# {py:mod}`fastvideo.configs.pipelines.base`

```{py:module} fastvideo.configs.pipelines.base
```

```{autodoc2-docstring} fastvideo.configs.pipelines.base
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PipelineConfig <fastvideo.configs.pipelines.base.PipelineConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`STA_Mode <fastvideo.configs.pipelines.base.STA_Mode>`
  - ```{autodoc2-docstring} fastvideo.configs.pipelines.base.STA_Mode
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`SlidingTileAttnConfig <fastvideo.configs.pipelines.base.SlidingTileAttnConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.pipelines.base.SlidingTileAttnConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`parse_int_list <fastvideo.configs.pipelines.base.parse_int_list>`
  - ```{autodoc2-docstring} fastvideo.configs.pipelines.base.parse_int_list
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`postprocess_text <fastvideo.configs.pipelines.base.postprocess_text>`
  - ```{autodoc2-docstring} fastvideo.configs.pipelines.base.postprocess_text
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`preprocess_text <fastvideo.configs.pipelines.base.preprocess_text>`
  - ```{autodoc2-docstring} fastvideo.configs.pipelines.base.preprocess_text
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.configs.pipelines.base.logger>`
  - ```{autodoc2-docstring} fastvideo.configs.pipelines.base.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} PipelineConfig
:canonical: fastvideo.configs.pipelines.base.PipelineConfig

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} DEFAULT_TEXT_ENCODER_PRECISIONS
:canonical: fastvideo.configs.pipelines.base.PipelineConfig.DEFAULT_TEXT_ENCODER_PRECISIONS
:value: >
   ('fp32',)

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig.DEFAULT_TEXT_ENCODER_PRECISIONS
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} STA_mode
:canonical: fastvideo.configs.pipelines.base.PipelineConfig.STA_mode
:type: fastvideo.configs.pipelines.base.STA_Mode
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig.STA_mode
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} add_cli_args(parser: fastvideo.utils.FlexibleArgumentParser, prefix: str = '') -> fastvideo.utils.FlexibleArgumentParser
:canonical: fastvideo.configs.pipelines.base.PipelineConfig.add_cli_args
:staticmethod:

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig.add_cli_args
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} check_pipeline_config() -> None
:canonical: fastvideo.configs.pipelines.base.PipelineConfig.check_pipeline_config

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig.check_pipeline_config
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} disable_autocast
:canonical: fastvideo.configs.pipelines.base.PipelineConfig.disable_autocast
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig.disable_autocast
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} dit_config
:canonical: fastvideo.configs.pipelines.base.PipelineConfig.dit_config
:type: fastvideo.configs.models.DiTConfig
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig.dit_config
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} dit_precision
:canonical: fastvideo.configs.pipelines.base.PipelineConfig.dit_precision
:type: str
:value: >
   'bf16'

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig.dit_precision
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} dmd_denoising_steps
:canonical: fastvideo.configs.pipelines.base.PipelineConfig.dmd_denoising_steps
:type: list[int] | None
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig.dmd_denoising_steps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} dump_to_json(file_path: str)
:canonical: fastvideo.configs.pipelines.base.PipelineConfig.dump_to_json

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig.dump_to_json
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} embedded_cfg_scale
:canonical: fastvideo.configs.pipelines.base.PipelineConfig.embedded_cfg_scale
:type: float
:value: >
   6.0

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig.embedded_cfg_scale
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} flow_shift
:canonical: fastvideo.configs.pipelines.base.PipelineConfig.flow_shift
:type: float | None
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig.flow_shift
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} from_kwargs(kwargs: dict[str, typing.Any], config_cli_prefix: str = '') -> fastvideo.configs.pipelines.base.PipelineConfig
:canonical: fastvideo.configs.pipelines.base.PipelineConfig.from_kwargs
:classmethod:

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig.from_kwargs
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} from_pretrained(model_path: str) -> fastvideo.configs.pipelines.base.PipelineConfig
:canonical: fastvideo.configs.pipelines.base.PipelineConfig.from_pretrained
:classmethod:

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig.from_pretrained
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} image_encoder_config
:canonical: fastvideo.configs.pipelines.base.PipelineConfig.image_encoder_config
:type: fastvideo.configs.models.EncoderConfig
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig.image_encoder_config
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} image_encoder_precision
:canonical: fastvideo.configs.pipelines.base.PipelineConfig.image_encoder_precision
:type: str
:value: >
   'fp32'

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig.image_encoder_precision
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} load_from_json(file_path: str)
:canonical: fastvideo.configs.pipelines.base.PipelineConfig.load_from_json

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig.load_from_json
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} mask_strategy_file_path
:canonical: fastvideo.configs.pipelines.base.PipelineConfig.mask_strategy_file_path
:type: str | None
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig.mask_strategy_file_path
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} model_path
:canonical: fastvideo.configs.pipelines.base.PipelineConfig.model_path
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig.model_path
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} neg_magic
:canonical: fastvideo.configs.pipelines.base.PipelineConfig.neg_magic
:type: str | None
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig.neg_magic
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} pipeline_config_path
:canonical: fastvideo.configs.pipelines.base.PipelineConfig.pipeline_config_path
:type: str | None
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig.pipeline_config_path
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} pos_magic
:canonical: fastvideo.configs.pipelines.base.PipelineConfig.pos_magic
:type: str | None
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig.pos_magic
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} postprocess_text_funcs
:canonical: fastvideo.configs.pipelines.base.PipelineConfig.postprocess_text_funcs
:type: tuple[collections.abc.Callable[[fastvideo.configs.models.encoders.BaseEncoderOutput], torch.tensor], ...]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig.postprocess_text_funcs
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} preprocess_text_funcs
:canonical: fastvideo.configs.pipelines.base.PipelineConfig.preprocess_text_funcs
:type: tuple[collections.abc.Callable[[str], str], ...]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig.preprocess_text_funcs
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} skip_time_steps
:canonical: fastvideo.configs.pipelines.base.PipelineConfig.skip_time_steps
:type: int
:value: >
   15

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig.skip_time_steps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} text_encoder_configs
:canonical: fastvideo.configs.pipelines.base.PipelineConfig.text_encoder_configs
:type: tuple[fastvideo.configs.models.EncoderConfig, ...]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig.text_encoder_configs
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} text_encoder_precisions
:canonical: fastvideo.configs.pipelines.base.PipelineConfig.text_encoder_precisions
:type: tuple[str, ...]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig.text_encoder_precisions
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} timesteps_scale
:canonical: fastvideo.configs.pipelines.base.PipelineConfig.timesteps_scale
:type: bool | None
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig.timesteps_scale
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} update_config_from_dict(args: dict[str, typing.Any], prefix: str = '') -> None
:canonical: fastvideo.configs.pipelines.base.PipelineConfig.update_config_from_dict

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig.update_config_from_dict
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} update_pipeline_config(source_pipeline_dict: dict[str, typing.Any]) -> None
:canonical: fastvideo.configs.pipelines.base.PipelineConfig.update_pipeline_config

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig.update_pipeline_config
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} vae_config
:canonical: fastvideo.configs.pipelines.base.PipelineConfig.vae_config
:type: fastvideo.configs.models.VAEConfig
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig.vae_config
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} vae_precision
:canonical: fastvideo.configs.pipelines.base.PipelineConfig.vae_precision
:type: str
:value: >
   'fp32'

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig.vae_precision
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} vae_sp
:canonical: fastvideo.configs.pipelines.base.PipelineConfig.vae_sp
:type: bool
:value: >
   True

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig.vae_sp
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} vae_tiling
:canonical: fastvideo.configs.pipelines.base.PipelineConfig.vae_tiling
:type: bool
:value: >
   True

```{autodoc2-docstring} fastvideo.configs.pipelines.base.PipelineConfig.vae_tiling
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} STA_Mode()
:canonical: fastvideo.configs.pipelines.base.STA_Mode

Bases: {py:obj}`str`, {py:obj}`enum.Enum`

```{autodoc2-docstring} fastvideo.configs.pipelines.base.STA_Mode
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.configs.pipelines.base.STA_Mode.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} NONE
:canonical: fastvideo.configs.pipelines.base.STA_Mode.NONE
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.pipelines.base.STA_Mode.NONE
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} STA_INFERENCE
:canonical: fastvideo.configs.pipelines.base.STA_Mode.STA_INFERENCE
:value: >
   'STA_inference'

```{autodoc2-docstring} fastvideo.configs.pipelines.base.STA_Mode.STA_INFERENCE
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} STA_SEARCHING
:canonical: fastvideo.configs.pipelines.base.STA_Mode.STA_SEARCHING
:value: >
   'STA_searching'

```{autodoc2-docstring} fastvideo.configs.pipelines.base.STA_Mode.STA_SEARCHING
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} STA_TUNING
:canonical: fastvideo.configs.pipelines.base.STA_Mode.STA_TUNING
:value: >
   'STA_tuning'

```{autodoc2-docstring} fastvideo.configs.pipelines.base.STA_Mode.STA_TUNING
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} STA_TUNING_CFG
:canonical: fastvideo.configs.pipelines.base.STA_Mode.STA_TUNING_CFG
:value: >
   'STA_tuning_cfg'

```{autodoc2-docstring} fastvideo.configs.pipelines.base.STA_Mode.STA_TUNING_CFG
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} SlidingTileAttnConfig
:canonical: fastvideo.configs.pipelines.base.SlidingTileAttnConfig

Bases: {py:obj}`fastvideo.configs.pipelines.base.PipelineConfig`

```{autodoc2-docstring} fastvideo.configs.pipelines.base.SlidingTileAttnConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} height
:canonical: fastvideo.configs.pipelines.base.SlidingTileAttnConfig.height
:type: int
:value: >
   576

```{autodoc2-docstring} fastvideo.configs.pipelines.base.SlidingTileAttnConfig.height
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} pad_to_square
:canonical: fastvideo.configs.pipelines.base.SlidingTileAttnConfig.pad_to_square
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.configs.pipelines.base.SlidingTileAttnConfig.pad_to_square
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} stride
:canonical: fastvideo.configs.pipelines.base.SlidingTileAttnConfig.stride
:type: int
:value: >
   8

```{autodoc2-docstring} fastvideo.configs.pipelines.base.SlidingTileAttnConfig.stride
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} use_overlap_optimization
:canonical: fastvideo.configs.pipelines.base.SlidingTileAttnConfig.use_overlap_optimization
:type: bool
:value: >
   True

```{autodoc2-docstring} fastvideo.configs.pipelines.base.SlidingTileAttnConfig.use_overlap_optimization
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} width
:canonical: fastvideo.configs.pipelines.base.SlidingTileAttnConfig.width
:type: int
:value: >
   1024

```{autodoc2-docstring} fastvideo.configs.pipelines.base.SlidingTileAttnConfig.width
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} window_size
:canonical: fastvideo.configs.pipelines.base.SlidingTileAttnConfig.window_size
:type: int
:value: >
   16

```{autodoc2-docstring} fastvideo.configs.pipelines.base.SlidingTileAttnConfig.window_size
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} logger
:canonical: fastvideo.configs.pipelines.base.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.configs.pipelines.base.logger
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:function} parse_int_list(value: str) -> list[int]
:canonical: fastvideo.configs.pipelines.base.parse_int_list

```{autodoc2-docstring} fastvideo.configs.pipelines.base.parse_int_list
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} postprocess_text(output: fastvideo.configs.models.encoders.BaseEncoderOutput) -> torch.tensor
:canonical: fastvideo.configs.pipelines.base.postprocess_text

```{autodoc2-docstring} fastvideo.configs.pipelines.base.postprocess_text
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} preprocess_text(prompt: str) -> str
:canonical: fastvideo.configs.pipelines.base.preprocess_text

```{autodoc2-docstring} fastvideo.configs.pipelines.base.preprocess_text
:parser: docs.source.autodoc2_docstring_parser
```
````
