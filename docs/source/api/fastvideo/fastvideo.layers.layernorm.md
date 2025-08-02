# {py:mod}`fastvideo.layers.layernorm`

```{py:module} fastvideo.layers.layernorm
```

```{autodoc2-docstring} fastvideo.layers.layernorm
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FP32LayerNorm <fastvideo.layers.layernorm.FP32LayerNorm>`
  -
* - {py:obj}`LayerNormScaleShift <fastvideo.layers.layernorm.LayerNormScaleShift>`
  - ```{autodoc2-docstring} fastvideo.layers.layernorm.LayerNormScaleShift
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`RMSNorm <fastvideo.layers.layernorm.RMSNorm>`
  - ```{autodoc2-docstring} fastvideo.layers.layernorm.RMSNorm
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`ScaleResidual <fastvideo.layers.layernorm.ScaleResidual>`
  - ```{autodoc2-docstring} fastvideo.layers.layernorm.ScaleResidual
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`ScaleResidualLayerNormScaleShift <fastvideo.layers.layernorm.ScaleResidualLayerNormScaleShift>`
  - ```{autodoc2-docstring} fastvideo.layers.layernorm.ScaleResidualLayerNormScaleShift
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} FP32LayerNorm(normalized_shape: torch.nn.modules.normalization._shape_t, eps: float = 1e-05, elementwise_affine: bool = True, bias: bool = True, device=None, dtype=None)
:canonical: fastvideo.layers.layernorm.FP32LayerNorm

Bases: {py:obj}`torch.nn.LayerNorm`

````{py:method} forward(inputs: torch.Tensor) -> torch.Tensor
:canonical: fastvideo.layers.layernorm.FP32LayerNorm.forward

```{autodoc2-docstring} fastvideo.layers.layernorm.FP32LayerNorm.forward
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} LayerNormScaleShift(hidden_size: int, norm_type: str = 'rms', eps: float = 1e-06, elementwise_affine: bool = False, dtype: torch.dtype = torch.float32, compute_dtype: torch.dtype | None = None, prefix: str = '')
:canonical: fastvideo.layers.layernorm.LayerNormScaleShift

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} fastvideo.layers.layernorm.LayerNormScaleShift
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.layers.layernorm.LayerNormScaleShift.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} forward(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor
:canonical: fastvideo.layers.layernorm.LayerNormScaleShift.forward

```{autodoc2-docstring} fastvideo.layers.layernorm.LayerNormScaleShift.forward
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} RMSNorm(hidden_size: int, eps: float = 1e-06, dtype: torch.dtype = torch.float32, var_hidden_size: int | None = None, has_weight: bool = True)
:canonical: fastvideo.layers.layernorm.RMSNorm

Bases: {py:obj}`fastvideo.layers.custom_op.CustomOp`

```{autodoc2-docstring} fastvideo.layers.layernorm.RMSNorm
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.layers.layernorm.RMSNorm.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} extra_repr() -> str
:canonical: fastvideo.layers.layernorm.RMSNorm.extra_repr

````

````{py:method} forward_native(x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]
:canonical: fastvideo.layers.layernorm.RMSNorm.forward_native

```{autodoc2-docstring} fastvideo.layers.layernorm.RMSNorm.forward_native
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} ScaleResidual(prefix: str = '')
:canonical: fastvideo.layers.layernorm.ScaleResidual

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} fastvideo.layers.layernorm.ScaleResidual
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.layers.layernorm.ScaleResidual.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} forward(residual: torch.Tensor, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor
:canonical: fastvideo.layers.layernorm.ScaleResidual.forward

```{autodoc2-docstring} fastvideo.layers.layernorm.ScaleResidual.forward
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} ScaleResidualLayerNormScaleShift(hidden_size: int, norm_type: str = 'rms', eps: float = 1e-06, elementwise_affine: bool = False, dtype: torch.dtype = torch.float32, compute_dtype: torch.dtype | None = None, prefix: str = '')
:canonical: fastvideo.layers.layernorm.ScaleResidualLayerNormScaleShift

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} fastvideo.layers.layernorm.ScaleResidualLayerNormScaleShift
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.layers.layernorm.ScaleResidualLayerNormScaleShift.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} forward(residual: torch.Tensor, x: torch.Tensor, gate: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
:canonical: fastvideo.layers.layernorm.ScaleResidualLayerNormScaleShift.forward

```{autodoc2-docstring} fastvideo.layers.layernorm.ScaleResidualLayerNormScaleShift.forward
:parser: docs.source.autodoc2_docstring_parser
```

````

`````
