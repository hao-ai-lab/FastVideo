# {py:mod}`fastvideo.attention.layer`

```{py:module} fastvideo.attention.layer
```

```{autodoc2-docstring} fastvideo.attention.layer
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DistributedAttention <fastvideo.attention.layer.DistributedAttention>`
  - ```{autodoc2-docstring} fastvideo.attention.layer.DistributedAttention
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`DistributedAttention_VSA <fastvideo.attention.layer.DistributedAttention_VSA>`
  - ```{autodoc2-docstring} fastvideo.attention.layer.DistributedAttention_VSA
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`LocalAttention <fastvideo.attention.layer.LocalAttention>`
  - ```{autodoc2-docstring} fastvideo.attention.layer.LocalAttention
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} DistributedAttention(num_heads: int, head_size: int, num_kv_heads: int | None = None, softmax_scale: float | None = None, causal: bool = False, supported_attention_backends: tuple[fastvideo.platforms.AttentionBackendEnum, ...] | None = None, prefix: str = '', **extra_impl_args)
:canonical: fastvideo.attention.layer.DistributedAttention

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} fastvideo.attention.layer.DistributedAttention
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.attention.layer.DistributedAttention.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, replicated_q: torch.Tensor | None = None, replicated_k: torch.Tensor | None = None, replicated_v: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]
:canonical: fastvideo.attention.layer.DistributedAttention.forward

```{autodoc2-docstring} fastvideo.attention.layer.DistributedAttention.forward
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} DistributedAttention_VSA(num_heads: int, head_size: int, num_kv_heads: int | None = None, softmax_scale: float | None = None, causal: bool = False, supported_attention_backends: tuple[fastvideo.platforms.AttentionBackendEnum, ...] | None = None, prefix: str = '', **extra_impl_args)
:canonical: fastvideo.attention.layer.DistributedAttention_VSA

Bases: {py:obj}`fastvideo.attention.layer.DistributedAttention`

```{autodoc2-docstring} fastvideo.attention.layer.DistributedAttention_VSA
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.attention.layer.DistributedAttention_VSA.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, replicated_q: torch.Tensor | None = None, replicated_k: torch.Tensor | None = None, replicated_v: torch.Tensor | None = None, gate_compress: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]
:canonical: fastvideo.attention.layer.DistributedAttention_VSA.forward

```{autodoc2-docstring} fastvideo.attention.layer.DistributedAttention_VSA.forward
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} LocalAttention(num_heads: int, head_size: int, num_kv_heads: int | None = None, softmax_scale: float | None = None, causal: bool = False, supported_attention_backends: tuple[fastvideo.platforms.AttentionBackendEnum, ...] | None = None, **extra_impl_args)
:canonical: fastvideo.attention.layer.LocalAttention

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} fastvideo.attention.layer.LocalAttention
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.attention.layer.LocalAttention.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor
:canonical: fastvideo.attention.layer.LocalAttention.forward

```{autodoc2-docstring} fastvideo.attention.layer.LocalAttention.forward
:parser: docs.source.autodoc2_docstring_parser
```

````

`````
