# {py:mod}`fastvideo.layers.activation`

```{py:module} fastvideo.layers.activation
```

```{autodoc2-docstring} fastvideo.layers.activation
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GeluAndMul <fastvideo.layers.activation.GeluAndMul>`
  - ```{autodoc2-docstring} fastvideo.layers.activation.GeluAndMul
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`NewGELU <fastvideo.layers.activation.NewGELU>`
  -
* - {py:obj}`QuickGELU <fastvideo.layers.activation.QuickGELU>`
  -
* - {py:obj}`SiluAndMul <fastvideo.layers.activation.SiluAndMul>`
  - ```{autodoc2-docstring} fastvideo.layers.activation.SiluAndMul
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_act_and_mul_fn <fastvideo.layers.activation.get_act_and_mul_fn>`
  - ```{autodoc2-docstring} fastvideo.layers.activation.get_act_and_mul_fn
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`get_act_fn <fastvideo.layers.activation.get_act_fn>`
  - ```{autodoc2-docstring} fastvideo.layers.activation.get_act_fn
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} GeluAndMul(approximate: str = 'none')
:canonical: fastvideo.layers.activation.GeluAndMul

Bases: {py:obj}`fastvideo.layers.custom_op.CustomOp`

```{autodoc2-docstring} fastvideo.layers.activation.GeluAndMul
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.layers.activation.GeluAndMul.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} extra_repr() -> str
:canonical: fastvideo.layers.activation.GeluAndMul.extra_repr

````

````{py:method} forward_native(x: torch.Tensor) -> torch.Tensor
:canonical: fastvideo.layers.activation.GeluAndMul.forward_native

```{autodoc2-docstring} fastvideo.layers.activation.GeluAndMul.forward_native
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} NewGELU()
:canonical: fastvideo.layers.activation.NewGELU

Bases: {py:obj}`fastvideo.layers.custom_op.CustomOp`

````{py:method} forward_native(x: torch.Tensor) -> torch.Tensor
:canonical: fastvideo.layers.activation.NewGELU.forward_native

```{autodoc2-docstring} fastvideo.layers.activation.NewGELU.forward_native
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} QuickGELU()
:canonical: fastvideo.layers.activation.QuickGELU

Bases: {py:obj}`fastvideo.layers.custom_op.CustomOp`

````{py:method} forward_native(x: torch.Tensor) -> torch.Tensor
:canonical: fastvideo.layers.activation.QuickGELU.forward_native

```{autodoc2-docstring} fastvideo.layers.activation.QuickGELU.forward_native
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} SiluAndMul()
:canonical: fastvideo.layers.activation.SiluAndMul

Bases: {py:obj}`fastvideo.layers.custom_op.CustomOp`

```{autodoc2-docstring} fastvideo.layers.activation.SiluAndMul
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.layers.activation.SiluAndMul.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} forward_native(x: torch.Tensor) -> torch.Tensor
:canonical: fastvideo.layers.activation.SiluAndMul.forward_native

```{autodoc2-docstring} fastvideo.layers.activation.SiluAndMul.forward_native
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:function} get_act_and_mul_fn(act_fn_name: str) -> torch.nn.Module
:canonical: fastvideo.layers.activation.get_act_and_mul_fn

```{autodoc2-docstring} fastvideo.layers.activation.get_act_and_mul_fn
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} get_act_fn(act_fn_name: str) -> torch.nn.Module
:canonical: fastvideo.layers.activation.get_act_fn

```{autodoc2-docstring} fastvideo.layers.activation.get_act_fn
:parser: docs.source.autodoc2_docstring_parser
```
````
