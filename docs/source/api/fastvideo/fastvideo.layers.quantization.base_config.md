# {py:mod}`fastvideo.layers.quantization.base_config`

```{py:module} fastvideo.layers.quantization.base_config
```

```{autodoc2-docstring} fastvideo.layers.quantization.base_config
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`QuantizationConfig <fastvideo.layers.quantization.base_config.QuantizationConfig>`
  - ```{autodoc2-docstring} fastvideo.layers.quantization.base_config.QuantizationConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`QuantizeMethodBase <fastvideo.layers.quantization.base_config.QuantizeMethodBase>`
  - ```{autodoc2-docstring} fastvideo.layers.quantization.base_config.QuantizeMethodBase
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`method_has_implemented_embedding <fastvideo.layers.quantization.base_config.method_has_implemented_embedding>`
  - ```{autodoc2-docstring} fastvideo.layers.quantization.base_config.method_has_implemented_embedding
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} QuantizationConfig()
:canonical: fastvideo.layers.quantization.base_config.QuantizationConfig

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} fastvideo.layers.quantization.base_config.QuantizationConfig
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.layers.quantization.base_config.QuantizationConfig.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} from_config(config: dict[str, typing.Any]) -> fastvideo.layers.quantization.base_config.QuantizationConfig
:canonical: fastvideo.layers.quantization.base_config.QuantizationConfig.from_config
:abstractmethod:
:classmethod:

```{autodoc2-docstring} fastvideo.layers.quantization.base_config.QuantizationConfig.from_config
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_cache_scale(name: str) -> str | None
:canonical: fastvideo.layers.quantization.base_config.QuantizationConfig.get_cache_scale

```{autodoc2-docstring} fastvideo.layers.quantization.base_config.QuantizationConfig.get_cache_scale
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_config_filenames() -> list[str]
:canonical: fastvideo.layers.quantization.base_config.QuantizationConfig.get_config_filenames
:abstractmethod:
:staticmethod:

```{autodoc2-docstring} fastvideo.layers.quantization.base_config.QuantizationConfig.get_config_filenames
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_from_keys(config: dict[str, typing.Any], keys: list[str]) -> typing.Any
:canonical: fastvideo.layers.quantization.base_config.QuantizationConfig.get_from_keys
:staticmethod:

```{autodoc2-docstring} fastvideo.layers.quantization.base_config.QuantizationConfig.get_from_keys
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_from_keys_or(config: dict[str, typing.Any], keys: list[str], default: typing.Any) -> typing.Any
:canonical: fastvideo.layers.quantization.base_config.QuantizationConfig.get_from_keys_or
:staticmethod:

```{autodoc2-docstring} fastvideo.layers.quantization.base_config.QuantizationConfig.get_from_keys_or
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_min_capability() -> int
:canonical: fastvideo.layers.quantization.base_config.QuantizationConfig.get_min_capability
:abstractmethod:
:classmethod:

```{autodoc2-docstring} fastvideo.layers.quantization.base_config.QuantizationConfig.get_min_capability
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_name() -> fastvideo.layers.quantization.QuantizationMethods
:canonical: fastvideo.layers.quantization.base_config.QuantizationConfig.get_name
:abstractmethod:

```{autodoc2-docstring} fastvideo.layers.quantization.base_config.QuantizationConfig.get_name
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_quant_method(layer: torch.nn.Module, prefix: str) -> fastvideo.layers.quantization.base_config.QuantizeMethodBase | None
:canonical: fastvideo.layers.quantization.base_config.QuantizationConfig.get_quant_method
:abstractmethod:

```{autodoc2-docstring} fastvideo.layers.quantization.base_config.QuantizationConfig.get_quant_method
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_supported_act_dtypes() -> list[torch.dtype]
:canonical: fastvideo.layers.quantization.base_config.QuantizationConfig.get_supported_act_dtypes
:abstractmethod:

```{autodoc2-docstring} fastvideo.layers.quantization.base_config.QuantizationConfig.get_supported_act_dtypes
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} override_quantization_method(hf_quant_cfg, user_quant) -> fastvideo.layers.quantization.QuantizationMethods | None
:canonical: fastvideo.layers.quantization.base_config.QuantizationConfig.override_quantization_method
:classmethod:

```{autodoc2-docstring} fastvideo.layers.quantization.base_config.QuantizationConfig.override_quantization_method
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} QuantizeMethodBase
:canonical: fastvideo.layers.quantization.base_config.QuantizeMethodBase

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} fastvideo.layers.quantization.base_config.QuantizeMethodBase
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} apply(layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor
:canonical: fastvideo.layers.quantization.base_config.QuantizeMethodBase.apply
:abstractmethod:

```{autodoc2-docstring} fastvideo.layers.quantization.base_config.QuantizeMethodBase.apply
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} create_weights(layer: torch.nn.Module, *weight_args, **extra_weight_attrs)
:canonical: fastvideo.layers.quantization.base_config.QuantizeMethodBase.create_weights
:abstractmethod:

```{autodoc2-docstring} fastvideo.layers.quantization.base_config.QuantizeMethodBase.create_weights
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} embedding(layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor
:canonical: fastvideo.layers.quantization.base_config.QuantizeMethodBase.embedding
:abstractmethod:

```{autodoc2-docstring} fastvideo.layers.quantization.base_config.QuantizeMethodBase.embedding
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} process_weights_after_loading(layer: torch.nn.Module) -> None
:canonical: fastvideo.layers.quantization.base_config.QuantizeMethodBase.process_weights_after_loading

```{autodoc2-docstring} fastvideo.layers.quantization.base_config.QuantizeMethodBase.process_weights_after_loading
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:function} method_has_implemented_embedding(method_class: type[fastvideo.layers.quantization.base_config.QuantizeMethodBase]) -> bool
:canonical: fastvideo.layers.quantization.base_config.method_has_implemented_embedding

```{autodoc2-docstring} fastvideo.layers.quantization.base_config.method_has_implemented_embedding
:parser: docs.source.autodoc2_docstring_parser
```
````
