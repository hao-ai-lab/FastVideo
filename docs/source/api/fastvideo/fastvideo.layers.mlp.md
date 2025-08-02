# {py:mod}`fastvideo.layers.mlp`

```{py:module} fastvideo.layers.mlp
```

```{autodoc2-docstring} fastvideo.layers.mlp
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MLP <fastvideo.layers.mlp.MLP>`
  - ```{autodoc2-docstring} fastvideo.layers.mlp.MLP
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} MLP(input_dim: int, mlp_hidden_dim: int, output_dim: int | None = None, bias: bool = True, act_type: str = 'gelu_pytorch_tanh', dtype: torch.dtype | None = None, prefix: str = '')
:canonical: fastvideo.layers.mlp.MLP

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} fastvideo.layers.mlp.MLP
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.layers.mlp.MLP.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} forward(x: torch.Tensor) -> torch.Tensor
:canonical: fastvideo.layers.mlp.MLP.forward

```{autodoc2-docstring} fastvideo.layers.mlp.MLP.forward
:parser: docs.source.autodoc2_docstring_parser
```

````

`````
