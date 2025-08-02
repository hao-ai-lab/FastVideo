# {py:mod}`fastvideo.configs.models.dits.base`

```{py:module} fastvideo.configs.models.dits.base
```

```{autodoc2-docstring} fastvideo.configs.models.dits.base
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DiTArchConfig <fastvideo.configs.models.dits.base.DiTArchConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.models.dits.base.DiTArchConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`DiTConfig <fastvideo.configs.models.dits.base.DiTConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.models.dits.base.DiTConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} DiTArchConfig
:canonical: fastvideo.configs.models.dits.base.DiTArchConfig

Bases: {py:obj}`fastvideo.configs.models.base.ArchConfig`

```{autodoc2-docstring} fastvideo.configs.models.dits.base.DiTArchConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} exclude_lora_layers
:canonical: fastvideo.configs.models.dits.base.DiTArchConfig.exclude_lora_layers
:type: list[str]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.dits.base.DiTArchConfig.exclude_lora_layers
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} hidden_size
:canonical: fastvideo.configs.models.dits.base.DiTArchConfig.hidden_size
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.configs.models.dits.base.DiTArchConfig.hidden_size
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} lora_param_names_mapping
:canonical: fastvideo.configs.models.dits.base.DiTArchConfig.lora_param_names_mapping
:type: dict
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.dits.base.DiTArchConfig.lora_param_names_mapping
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_attention_heads
:canonical: fastvideo.configs.models.dits.base.DiTArchConfig.num_attention_heads
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.configs.models.dits.base.DiTArchConfig.num_attention_heads
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_channels_latents
:canonical: fastvideo.configs.models.dits.base.DiTArchConfig.num_channels_latents
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.configs.models.dits.base.DiTArchConfig.num_channels_latents
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} param_names_mapping
:canonical: fastvideo.configs.models.dits.base.DiTArchConfig.param_names_mapping
:type: dict
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.dits.base.DiTArchConfig.param_names_mapping
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} reverse_param_names_mapping
:canonical: fastvideo.configs.models.dits.base.DiTArchConfig.reverse_param_names_mapping
:type: dict
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.dits.base.DiTArchConfig.reverse_param_names_mapping
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} DiTConfig
:canonical: fastvideo.configs.models.dits.base.DiTConfig

Bases: {py:obj}`fastvideo.configs.models.base.ModelConfig`

```{autodoc2-docstring} fastvideo.configs.models.dits.base.DiTConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} add_cli_args(parser: typing.Any, prefix: str = 'dit-config') -> typing.Any
:canonical: fastvideo.configs.models.dits.base.DiTConfig.add_cli_args
:staticmethod:

```{autodoc2-docstring} fastvideo.configs.models.dits.base.DiTConfig.add_cli_args
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} arch_config
:canonical: fastvideo.configs.models.dits.base.DiTConfig.arch_config
:type: fastvideo.configs.models.dits.base.DiTArchConfig
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.dits.base.DiTConfig.arch_config
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} prefix
:canonical: fastvideo.configs.models.dits.base.DiTConfig.prefix
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} fastvideo.configs.models.dits.base.DiTConfig.prefix
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} quant_config
:canonical: fastvideo.configs.models.dits.base.DiTConfig.quant_config
:type: fastvideo.layers.quantization.QuantizationConfig | None
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.models.dits.base.DiTConfig.quant_config
:parser: docs.source.autodoc2_docstring_parser
```

````

`````
