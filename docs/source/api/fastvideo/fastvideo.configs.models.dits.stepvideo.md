# {py:mod}`fastvideo.configs.models.dits.stepvideo`

```{py:module} fastvideo.configs.models.dits.stepvideo
```

```{autodoc2-docstring} fastvideo.configs.models.dits.stepvideo
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`StepVideoArchConfig <fastvideo.configs.models.dits.stepvideo.StepVideoArchConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.models.dits.stepvideo.StepVideoArchConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`StepVideoConfig <fastvideo.configs.models.dits.stepvideo.StepVideoConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.models.dits.stepvideo.StepVideoConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} StepVideoArchConfig
:canonical: fastvideo.configs.models.dits.stepvideo.StepVideoArchConfig

Bases: {py:obj}`fastvideo.configs.models.dits.base.DiTArchConfig`

```{autodoc2-docstring} fastvideo.configs.models.dits.stepvideo.StepVideoArchConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} attention_head_dim
:canonical: fastvideo.configs.models.dits.stepvideo.StepVideoArchConfig.attention_head_dim
:type: int
:value: >
   128

```{autodoc2-docstring} fastvideo.configs.models.dits.stepvideo.StepVideoArchConfig.attention_head_dim
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} attention_type
:canonical: fastvideo.configs.models.dits.stepvideo.StepVideoArchConfig.attention_type
:type: str | None
:value: >
   'torch'

```{autodoc2-docstring} fastvideo.configs.models.dits.stepvideo.StepVideoArchConfig.attention_type
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} caption_channels
:canonical: fastvideo.configs.models.dits.stepvideo.StepVideoArchConfig.caption_channels
:type: int | list[int] | tuple[int, ...] | None
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.dits.stepvideo.StepVideoArchConfig.caption_channels
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} dropout
:canonical: fastvideo.configs.models.dits.stepvideo.StepVideoArchConfig.dropout
:type: float
:value: >
   0.0

```{autodoc2-docstring} fastvideo.configs.models.dits.stepvideo.StepVideoArchConfig.dropout
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} exclude_lora_layers
:canonical: fastvideo.configs.models.dits.stepvideo.StepVideoArchConfig.exclude_lora_layers
:type: list[str]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.dits.stepvideo.StepVideoArchConfig.exclude_lora_layers
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} in_channels
:canonical: fastvideo.configs.models.dits.stepvideo.StepVideoArchConfig.in_channels
:type: int
:value: >
   64

```{autodoc2-docstring} fastvideo.configs.models.dits.stepvideo.StepVideoArchConfig.in_channels
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} norm_elementwise_affine
:canonical: fastvideo.configs.models.dits.stepvideo.StepVideoArchConfig.norm_elementwise_affine
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.configs.models.dits.stepvideo.StepVideoArchConfig.norm_elementwise_affine
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} norm_eps
:canonical: fastvideo.configs.models.dits.stepvideo.StepVideoArchConfig.norm_eps
:type: float
:value: >
   1e-06

```{autodoc2-docstring} fastvideo.configs.models.dits.stepvideo.StepVideoArchConfig.norm_eps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} norm_type
:canonical: fastvideo.configs.models.dits.stepvideo.StepVideoArchConfig.norm_type
:type: str
:value: >
   'ada_norm_single'

```{autodoc2-docstring} fastvideo.configs.models.dits.stepvideo.StepVideoArchConfig.norm_type
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_attention_heads
:canonical: fastvideo.configs.models.dits.stepvideo.StepVideoArchConfig.num_attention_heads
:type: int
:value: >
   48

```{autodoc2-docstring} fastvideo.configs.models.dits.stepvideo.StepVideoArchConfig.num_attention_heads
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_layers
:canonical: fastvideo.configs.models.dits.stepvideo.StepVideoArchConfig.num_layers
:type: int
:value: >
   48

```{autodoc2-docstring} fastvideo.configs.models.dits.stepvideo.StepVideoArchConfig.num_layers
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} out_channels
:canonical: fastvideo.configs.models.dits.stepvideo.StepVideoArchConfig.out_channels
:type: int | None
:value: >
   64

```{autodoc2-docstring} fastvideo.configs.models.dits.stepvideo.StepVideoArchConfig.out_channels
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} param_names_mapping
:canonical: fastvideo.configs.models.dits.stepvideo.StepVideoArchConfig.param_names_mapping
:type: dict
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.dits.stepvideo.StepVideoArchConfig.param_names_mapping
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} patch_size
:canonical: fastvideo.configs.models.dits.stepvideo.StepVideoArchConfig.patch_size
:type: int
:value: >
   1

```{autodoc2-docstring} fastvideo.configs.models.dits.stepvideo.StepVideoArchConfig.patch_size
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} use_additional_conditions
:canonical: fastvideo.configs.models.dits.stepvideo.StepVideoArchConfig.use_additional_conditions
:type: bool | None
:value: >
   False

```{autodoc2-docstring} fastvideo.configs.models.dits.stepvideo.StepVideoArchConfig.use_additional_conditions
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} StepVideoConfig
:canonical: fastvideo.configs.models.dits.stepvideo.StepVideoConfig

Bases: {py:obj}`fastvideo.configs.models.dits.base.DiTConfig`

```{autodoc2-docstring} fastvideo.configs.models.dits.stepvideo.StepVideoConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} arch_config
:canonical: fastvideo.configs.models.dits.stepvideo.StepVideoConfig.arch_config
:type: fastvideo.configs.models.dits.base.DiTArchConfig
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.dits.stepvideo.StepVideoConfig.arch_config
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} prefix
:canonical: fastvideo.configs.models.dits.stepvideo.StepVideoConfig.prefix
:type: str
:value: >
   'StepVideo'

```{autodoc2-docstring} fastvideo.configs.models.dits.stepvideo.StepVideoConfig.prefix
:parser: docs.source.autodoc2_docstring_parser
```

````

`````
