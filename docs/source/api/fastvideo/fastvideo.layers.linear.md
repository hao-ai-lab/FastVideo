# {py:mod}`fastvideo.layers.linear`

```{py:module} fastvideo.layers.linear
```

```{autodoc2-docstring} fastvideo.layers.linear
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ColumnParallelLinear <fastvideo.layers.linear.ColumnParallelLinear>`
  - ```{autodoc2-docstring} fastvideo.layers.linear.ColumnParallelLinear
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`LinearBase <fastvideo.layers.linear.LinearBase>`
  - ```{autodoc2-docstring} fastvideo.layers.linear.LinearBase
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`LinearMethodBase <fastvideo.layers.linear.LinearMethodBase>`
  - ```{autodoc2-docstring} fastvideo.layers.linear.LinearMethodBase
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`MergedColumnParallelLinear <fastvideo.layers.linear.MergedColumnParallelLinear>`
  - ```{autodoc2-docstring} fastvideo.layers.linear.MergedColumnParallelLinear
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`QKVParallelLinear <fastvideo.layers.linear.QKVParallelLinear>`
  - ```{autodoc2-docstring} fastvideo.layers.linear.QKVParallelLinear
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`ReplicatedLinear <fastvideo.layers.linear.ReplicatedLinear>`
  - ```{autodoc2-docstring} fastvideo.layers.linear.ReplicatedLinear
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`RowParallelLinear <fastvideo.layers.linear.RowParallelLinear>`
  - ```{autodoc2-docstring} fastvideo.layers.linear.RowParallelLinear
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`UnquantizedLinearMethod <fastvideo.layers.linear.UnquantizedLinearMethod>`
  - ```{autodoc2-docstring} fastvideo.layers.linear.UnquantizedLinearMethod
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`adjust_scalar_to_fused_array <fastvideo.layers.linear.adjust_scalar_to_fused_array>`
  - ```{autodoc2-docstring} fastvideo.layers.linear.adjust_scalar_to_fused_array
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`WEIGHT_LOADER_V2_SUPPORTED <fastvideo.layers.linear.WEIGHT_LOADER_V2_SUPPORTED>`
  - ```{autodoc2-docstring} fastvideo.layers.linear.WEIGHT_LOADER_V2_SUPPORTED
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`logger <fastvideo.layers.linear.logger>`
  - ```{autodoc2-docstring} fastvideo.layers.linear.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} ColumnParallelLinear(input_size: int, output_size: int, bias: bool = True, gather_output: bool = False, skip_bias_add: bool = False, params_dtype: torch.dtype | None = None, quant_config: fastvideo.layers.quantization.base_config.QuantizationConfig | None = None, output_sizes: list[int] | None = None, prefix: str = '')
:canonical: fastvideo.layers.linear.ColumnParallelLinear

Bases: {py:obj}`fastvideo.layers.linear.LinearBase`

```{autodoc2-docstring} fastvideo.layers.linear.ColumnParallelLinear
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.layers.linear.ColumnParallelLinear.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} extra_repr() -> str
:canonical: fastvideo.layers.linear.ColumnParallelLinear.extra_repr

````

````{py:method} forward(input_: torch.Tensor) -> tuple[torch.Tensor, torch.nn.parameter.Parameter | None]
:canonical: fastvideo.layers.linear.ColumnParallelLinear.forward

```{autodoc2-docstring} fastvideo.layers.linear.ColumnParallelLinear.forward
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} weight_loader(param: torch.nn.parameter.Parameter, loaded_weight: torch.Tensor) -> None
:canonical: fastvideo.layers.linear.ColumnParallelLinear.weight_loader

```{autodoc2-docstring} fastvideo.layers.linear.ColumnParallelLinear.weight_loader
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} weight_loader_v2(param: torch.nn.parameter.Parameter, loaded_weight: torch.Tensor) -> None
:canonical: fastvideo.layers.linear.ColumnParallelLinear.weight_loader_v2

```{autodoc2-docstring} fastvideo.layers.linear.ColumnParallelLinear.weight_loader_v2
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} LinearBase(input_size: int, output_size: int, skip_bias_add: bool = False, params_dtype: torch.dtype | None = None, quant_config: fastvideo.layers.quantization.base_config.QuantizationConfig | None = None, prefix: str = '')
:canonical: fastvideo.layers.linear.LinearBase

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} fastvideo.layers.linear.LinearBase
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.layers.linear.LinearBase.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} forward(x: torch.Tensor) -> tuple[torch.Tensor, torch.nn.parameter.Parameter | None]
:canonical: fastvideo.layers.linear.LinearBase.forward
:abstractmethod:

```{autodoc2-docstring} fastvideo.layers.linear.LinearBase.forward
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} LinearMethodBase
:canonical: fastvideo.layers.linear.LinearMethodBase

Bases: {py:obj}`fastvideo.layers.quantization.base_config.QuantizeMethodBase`

```{autodoc2-docstring} fastvideo.layers.linear.LinearMethodBase
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} apply(layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor
:canonical: fastvideo.layers.linear.LinearMethodBase.apply
:abstractmethod:

```{autodoc2-docstring} fastvideo.layers.linear.LinearMethodBase.apply
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} create_weights(layer: torch.nn.Module, input_size_per_partition: int, output_partition_sizes: list[int], input_size: int, output_size: int, params_dtype: torch.dtype, **extra_weight_attrs) -> None
:canonical: fastvideo.layers.linear.LinearMethodBase.create_weights
:abstractmethod:

```{autodoc2-docstring} fastvideo.layers.linear.LinearMethodBase.create_weights
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} MergedColumnParallelLinear(input_size: int, output_sizes: list[int], bias: bool = True, gather_output: bool = False, skip_bias_add: bool = False, params_dtype: torch.dtype | None = None, quant_config: fastvideo.layers.quantization.base_config.QuantizationConfig | None = None, prefix: str = '')
:canonical: fastvideo.layers.linear.MergedColumnParallelLinear

Bases: {py:obj}`fastvideo.layers.linear.ColumnParallelLinear`

```{autodoc2-docstring} fastvideo.layers.linear.MergedColumnParallelLinear
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.layers.linear.MergedColumnParallelLinear.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} weight_loader(param: torch.nn.parameter.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int | None = None) -> None
:canonical: fastvideo.layers.linear.MergedColumnParallelLinear.weight_loader

```{autodoc2-docstring} fastvideo.layers.linear.MergedColumnParallelLinear.weight_loader
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} weight_loader_v2(param: fastvideo.models.parameter.BasevLLMParameter, loaded_weight: torch.Tensor, loaded_shard_id: int | None = None) -> None
:canonical: fastvideo.layers.linear.MergedColumnParallelLinear.weight_loader_v2

```{autodoc2-docstring} fastvideo.layers.linear.MergedColumnParallelLinear.weight_loader_v2
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} QKVParallelLinear(hidden_size: int, head_size: int, total_num_heads: int, total_num_kv_heads: int | None = None, bias: bool = True, skip_bias_add: bool = False, params_dtype: torch.dtype | None = None, quant_config: fastvideo.layers.quantization.base_config.QuantizationConfig | None = None, prefix: str = '')
:canonical: fastvideo.layers.linear.QKVParallelLinear

Bases: {py:obj}`fastvideo.layers.linear.ColumnParallelLinear`

```{autodoc2-docstring} fastvideo.layers.linear.QKVParallelLinear
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.layers.linear.QKVParallelLinear.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} weight_loader(param: torch.nn.parameter.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str | None = None)
:canonical: fastvideo.layers.linear.QKVParallelLinear.weight_loader

```{autodoc2-docstring} fastvideo.layers.linear.QKVParallelLinear.weight_loader
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} weight_loader_v2(param: fastvideo.models.parameter.BasevLLMParameter, loaded_weight: torch.Tensor, loaded_shard_id: str | None = None)
:canonical: fastvideo.layers.linear.QKVParallelLinear.weight_loader_v2

```{autodoc2-docstring} fastvideo.layers.linear.QKVParallelLinear.weight_loader_v2
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} ReplicatedLinear(input_size: int, output_size: int, bias: bool = True, skip_bias_add: bool = False, params_dtype: torch.dtype | None = None, quant_config: fastvideo.layers.quantization.base_config.QuantizationConfig | None = None, prefix: str = '')
:canonical: fastvideo.layers.linear.ReplicatedLinear

Bases: {py:obj}`fastvideo.layers.linear.LinearBase`

```{autodoc2-docstring} fastvideo.layers.linear.ReplicatedLinear
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.layers.linear.ReplicatedLinear.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} extra_repr() -> str
:canonical: fastvideo.layers.linear.ReplicatedLinear.extra_repr

````

````{py:method} forward(x: torch.Tensor) -> tuple[torch.Tensor, torch.nn.parameter.Parameter | None]
:canonical: fastvideo.layers.linear.ReplicatedLinear.forward

```{autodoc2-docstring} fastvideo.layers.linear.ReplicatedLinear.forward
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} weight_loader(param: torch.nn.parameter.Parameter, loaded_weight: torch.Tensor) -> None
:canonical: fastvideo.layers.linear.ReplicatedLinear.weight_loader

```{autodoc2-docstring} fastvideo.layers.linear.ReplicatedLinear.weight_loader
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} RowParallelLinear(input_size: int, output_size: int, bias: bool = True, input_is_parallel: bool = True, skip_bias_add: bool = False, params_dtype: torch.dtype | None = None, reduce_results: bool = True, quant_config: fastvideo.layers.quantization.base_config.QuantizationConfig | None = None, prefix: str = '')
:canonical: fastvideo.layers.linear.RowParallelLinear

Bases: {py:obj}`fastvideo.layers.linear.LinearBase`

```{autodoc2-docstring} fastvideo.layers.linear.RowParallelLinear
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.layers.linear.RowParallelLinear.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} extra_repr() -> str
:canonical: fastvideo.layers.linear.RowParallelLinear.extra_repr

````

````{py:method} forward(input_) -> tuple[torch.Tensor, torch.nn.parameter.Parameter | None]
:canonical: fastvideo.layers.linear.RowParallelLinear.forward

```{autodoc2-docstring} fastvideo.layers.linear.RowParallelLinear.forward
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} weight_loader(param: torch.nn.parameter.Parameter, loaded_weight: torch.Tensor)
:canonical: fastvideo.layers.linear.RowParallelLinear.weight_loader

```{autodoc2-docstring} fastvideo.layers.linear.RowParallelLinear.weight_loader
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} weight_loader_v2(param: fastvideo.models.parameter.BasevLLMParameter, loaded_weight: torch.Tensor)
:canonical: fastvideo.layers.linear.RowParallelLinear.weight_loader_v2

```{autodoc2-docstring} fastvideo.layers.linear.RowParallelLinear.weight_loader_v2
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} UnquantizedLinearMethod
:canonical: fastvideo.layers.linear.UnquantizedLinearMethod

Bases: {py:obj}`fastvideo.layers.linear.LinearMethodBase`

```{autodoc2-docstring} fastvideo.layers.linear.UnquantizedLinearMethod
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} apply(layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor
:canonical: fastvideo.layers.linear.UnquantizedLinearMethod.apply

````

````{py:method} create_weights(layer: torch.nn.Module, input_size_per_partition: int, output_partition_sizes: list[int], input_size: int, output_size: int, params_dtype: torch.dtype, **extra_weight_attrs) -> None
:canonical: fastvideo.layers.linear.UnquantizedLinearMethod.create_weights

````

`````

````{py:data} WEIGHT_LOADER_V2_SUPPORTED
:canonical: fastvideo.layers.linear.WEIGHT_LOADER_V2_SUPPORTED
:value: >
   ['CompressedTensorsLinearMethod', 'AWQMarlinLinearMethod', 'AWQLinearMethod', 'GPTQMarlinLinearMetho...

```{autodoc2-docstring} fastvideo.layers.linear.WEIGHT_LOADER_V2_SUPPORTED
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:function} adjust_scalar_to_fused_array(param: torch.Tensor, loaded_weight: torch.Tensor, shard_id: str | int) -> tuple[torch.Tensor, torch.Tensor]
:canonical: fastvideo.layers.linear.adjust_scalar_to_fused_array

```{autodoc2-docstring} fastvideo.layers.linear.adjust_scalar_to_fused_array
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:data} logger
:canonical: fastvideo.layers.linear.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.layers.linear.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
