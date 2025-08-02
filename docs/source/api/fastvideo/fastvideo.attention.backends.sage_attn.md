# {py:mod}`fastvideo.attention.backends.sage_attn`

```{py:module} fastvideo.attention.backends.sage_attn
```

```{autodoc2-docstring} fastvideo.attention.backends.sage_attn
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SageAttentionBackend <fastvideo.attention.backends.sage_attn.SageAttentionBackend>`
  -
* - {py:obj}`SageAttentionImpl <fastvideo.attention.backends.sage_attn.SageAttentionImpl>`
  -
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.attention.backends.sage_attn.logger>`
  - ```{autodoc2-docstring} fastvideo.attention.backends.sage_attn.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} SageAttentionBackend
:canonical: fastvideo.attention.backends.sage_attn.SageAttentionBackend

Bases: {py:obj}`fastvideo.attention.backends.abstract.AttentionBackend`

````{py:attribute} accept_output_buffer
:canonical: fastvideo.attention.backends.sage_attn.SageAttentionBackend.accept_output_buffer
:type: bool
:value: >
   True

```{autodoc2-docstring} fastvideo.attention.backends.sage_attn.SageAttentionBackend.accept_output_buffer
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_impl_cls() -> type[fastvideo.attention.backends.sage_attn.SageAttentionImpl]
:canonical: fastvideo.attention.backends.sage_attn.SageAttentionBackend.get_impl_cls
:staticmethod:

```{autodoc2-docstring} fastvideo.attention.backends.sage_attn.SageAttentionBackend.get_impl_cls
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_name() -> str
:canonical: fastvideo.attention.backends.sage_attn.SageAttentionBackend.get_name
:staticmethod:

```{autodoc2-docstring} fastvideo.attention.backends.sage_attn.SageAttentionBackend.get_name
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_supported_head_sizes() -> list[int]
:canonical: fastvideo.attention.backends.sage_attn.SageAttentionBackend.get_supported_head_sizes
:staticmethod:

```{autodoc2-docstring} fastvideo.attention.backends.sage_attn.SageAttentionBackend.get_supported_head_sizes
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} SageAttentionImpl(num_heads: int, head_size: int, causal: bool, softmax_scale: float, num_kv_heads: int | None = None, prefix: str = '', **extra_impl_args)
:canonical: fastvideo.attention.backends.sage_attn.SageAttentionImpl

Bases: {py:obj}`fastvideo.attention.backends.abstract.AttentionImpl`

````{py:method} forward(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_metadata: fastvideo.attention.backends.abstract.AttentionMetadata) -> torch.Tensor
:canonical: fastvideo.attention.backends.sage_attn.SageAttentionImpl.forward

```{autodoc2-docstring} fastvideo.attention.backends.sage_attn.SageAttentionImpl.forward
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} logger
:canonical: fastvideo.attention.backends.sage_attn.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.attention.backends.sage_attn.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
