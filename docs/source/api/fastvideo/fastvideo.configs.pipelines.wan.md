# {py:mod}`fastvideo.configs.pipelines.wan`

```{py:module} fastvideo.configs.pipelines.wan
```

```{autodoc2-docstring} fastvideo.configs.pipelines.wan
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FastWanT2V480PConfig <fastvideo.configs.pipelines.wan.FastWanT2V480PConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.pipelines.wan.FastWanT2V480PConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`WanI2V480PConfig <fastvideo.configs.pipelines.wan.WanI2V480PConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.pipelines.wan.WanI2V480PConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`WanI2V720PConfig <fastvideo.configs.pipelines.wan.WanI2V720PConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.pipelines.wan.WanI2V720PConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`WanT2V480PConfig <fastvideo.configs.pipelines.wan.WanT2V480PConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.pipelines.wan.WanT2V480PConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`WanT2V720PConfig <fastvideo.configs.pipelines.wan.WanT2V720PConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.pipelines.wan.WanT2V720PConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`t5_postprocess_text <fastvideo.configs.pipelines.wan.t5_postprocess_text>`
  - ```{autodoc2-docstring} fastvideo.configs.pipelines.wan.t5_postprocess_text
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} FastWanT2V480PConfig
:canonical: fastvideo.configs.pipelines.wan.FastWanT2V480PConfig

Bases: {py:obj}`fastvideo.configs.pipelines.wan.WanT2V480PConfig`

```{autodoc2-docstring} fastvideo.configs.pipelines.wan.FastWanT2V480PConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} dmd_denoising_steps
:canonical: fastvideo.configs.pipelines.wan.FastWanT2V480PConfig.dmd_denoising_steps
:type: list[int] | None
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.pipelines.wan.FastWanT2V480PConfig.dmd_denoising_steps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} flow_shift
:canonical: fastvideo.configs.pipelines.wan.FastWanT2V480PConfig.flow_shift
:type: int
:value: >
   8

```{autodoc2-docstring} fastvideo.configs.pipelines.wan.FastWanT2V480PConfig.flow_shift
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} WanI2V480PConfig
:canonical: fastvideo.configs.pipelines.wan.WanI2V480PConfig

Bases: {py:obj}`fastvideo.configs.pipelines.wan.WanT2V480PConfig`

```{autodoc2-docstring} fastvideo.configs.pipelines.wan.WanI2V480PConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} image_encoder_config
:canonical: fastvideo.configs.pipelines.wan.WanI2V480PConfig.image_encoder_config
:type: fastvideo.configs.models.EncoderConfig
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.pipelines.wan.WanI2V480PConfig.image_encoder_config
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} image_encoder_precision
:canonical: fastvideo.configs.pipelines.wan.WanI2V480PConfig.image_encoder_precision
:type: str
:value: >
   'fp32'

```{autodoc2-docstring} fastvideo.configs.pipelines.wan.WanI2V480PConfig.image_encoder_precision
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} WanI2V720PConfig
:canonical: fastvideo.configs.pipelines.wan.WanI2V720PConfig

Bases: {py:obj}`fastvideo.configs.pipelines.wan.WanI2V480PConfig`

```{autodoc2-docstring} fastvideo.configs.pipelines.wan.WanI2V720PConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} flow_shift
:canonical: fastvideo.configs.pipelines.wan.WanI2V720PConfig.flow_shift
:type: int
:value: >
   5

```{autodoc2-docstring} fastvideo.configs.pipelines.wan.WanI2V720PConfig.flow_shift
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} WanT2V480PConfig
:canonical: fastvideo.configs.pipelines.wan.WanT2V480PConfig

Bases: {py:obj}`fastvideo.configs.pipelines.base.PipelineConfig`

```{autodoc2-docstring} fastvideo.configs.pipelines.wan.WanT2V480PConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} dit_config
:canonical: fastvideo.configs.pipelines.wan.WanT2V480PConfig.dit_config
:type: fastvideo.configs.models.DiTConfig
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.pipelines.wan.WanT2V480PConfig.dit_config
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} flow_shift
:canonical: fastvideo.configs.pipelines.wan.WanT2V480PConfig.flow_shift
:type: int
:value: >
   3

```{autodoc2-docstring} fastvideo.configs.pipelines.wan.WanT2V480PConfig.flow_shift
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} postprocess_text_funcs
:canonical: fastvideo.configs.pipelines.wan.WanT2V480PConfig.postprocess_text_funcs
:type: tuple[collections.abc.Callable[[fastvideo.configs.models.encoders.BaseEncoderOutput], torch.tensor], ...]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.pipelines.wan.WanT2V480PConfig.postprocess_text_funcs
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} precision
:canonical: fastvideo.configs.pipelines.wan.WanT2V480PConfig.precision
:type: str
:value: >
   'bf16'

```{autodoc2-docstring} fastvideo.configs.pipelines.wan.WanT2V480PConfig.precision
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} text_encoder_configs
:canonical: fastvideo.configs.pipelines.wan.WanT2V480PConfig.text_encoder_configs
:type: tuple[fastvideo.configs.models.EncoderConfig, ...]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.pipelines.wan.WanT2V480PConfig.text_encoder_configs
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} text_encoder_precisions
:canonical: fastvideo.configs.pipelines.wan.WanT2V480PConfig.text_encoder_precisions
:type: tuple[str, ...]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.pipelines.wan.WanT2V480PConfig.text_encoder_precisions
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} vae_config
:canonical: fastvideo.configs.pipelines.wan.WanT2V480PConfig.vae_config
:type: fastvideo.configs.models.VAEConfig
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.pipelines.wan.WanT2V480PConfig.vae_config
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} vae_precision
:canonical: fastvideo.configs.pipelines.wan.WanT2V480PConfig.vae_precision
:type: str
:value: >
   'fp32'

```{autodoc2-docstring} fastvideo.configs.pipelines.wan.WanT2V480PConfig.vae_precision
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} vae_sp
:canonical: fastvideo.configs.pipelines.wan.WanT2V480PConfig.vae_sp
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.configs.pipelines.wan.WanT2V480PConfig.vae_sp
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} vae_tiling
:canonical: fastvideo.configs.pipelines.wan.WanT2V480PConfig.vae_tiling
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.configs.pipelines.wan.WanT2V480PConfig.vae_tiling
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} WanT2V720PConfig
:canonical: fastvideo.configs.pipelines.wan.WanT2V720PConfig

Bases: {py:obj}`fastvideo.configs.pipelines.wan.WanT2V480PConfig`

```{autodoc2-docstring} fastvideo.configs.pipelines.wan.WanT2V720PConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} flow_shift
:canonical: fastvideo.configs.pipelines.wan.WanT2V720PConfig.flow_shift
:type: int
:value: >
   5

```{autodoc2-docstring} fastvideo.configs.pipelines.wan.WanT2V720PConfig.flow_shift
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:function} t5_postprocess_text(outputs: fastvideo.configs.models.encoders.BaseEncoderOutput) -> torch.tensor
:canonical: fastvideo.configs.pipelines.wan.t5_postprocess_text

```{autodoc2-docstring} fastvideo.configs.pipelines.wan.t5_postprocess_text
:parser: docs.source.autodoc2_docstring_parser
```
````
