# {py:mod}`fastvideo.attention.backends.sdpa`

```{py:module} fastvideo.attention.backends.sdpa
```

```{autodoc2-docstring} fastvideo.attention.backends.sdpa
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SDPABackend <fastvideo.attention.backends.sdpa.SDPABackend>`
  -
* - {py:obj}`SDPAImpl <fastvideo.attention.backends.sdpa.SDPAImpl>`
  -
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.attention.backends.sdpa.logger>`
  - ```{autodoc2-docstring} fastvideo.attention.backends.sdpa.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} SDPABackend
:canonical: fastvideo.attention.backends.sdpa.SDPABackend

Bases: {py:obj}`fastvideo.attention.backends.abstract.AttentionBackend`

````{py:attribute} accept_output_buffer
:canonical: fastvideo.attention.backends.sdpa.SDPABackend.accept_output_buffer
:type: bool
:value: >
   True

```{autodoc2-docstring} fastvideo.attention.backends.sdpa.SDPABackend.accept_output_buffer
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_impl_cls() -> type[fastvideo.attention.backends.sdpa.SDPAImpl]
:canonical: fastvideo.attention.backends.sdpa.SDPABackend.get_impl_cls
:staticmethod:

```{autodoc2-docstring} fastvideo.attention.backends.sdpa.SDPABackend.get_impl_cls
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_name() -> str
:canonical: fastvideo.attention.backends.sdpa.SDPABackend.get_name
:staticmethod:

```{autodoc2-docstring} fastvideo.attention.backends.sdpa.SDPABackend.get_name
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_supported_head_sizes() -> list[int]
:canonical: fastvideo.attention.backends.sdpa.SDPABackend.get_supported_head_sizes
:staticmethod:

```{autodoc2-docstring} fastvideo.attention.backends.sdpa.SDPABackend.get_supported_head_sizes
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} SDPAImpl(num_heads: int, head_size: int, causal: bool, softmax_scale: float, num_kv_heads: int | None = None, prefix: str = '', **extra_impl_args)
:canonical: fastvideo.attention.backends.sdpa.SDPAImpl

Bases: {py:obj}`fastvideo.attention.backends.abstract.AttentionImpl`

````{py:method} forward(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_metadata: fastvideo.attention.backends.abstract.AttentionMetadata) -> torch.Tensor
:canonical: fastvideo.attention.backends.sdpa.SDPAImpl.forward

```{autodoc2-docstring} fastvideo.attention.backends.sdpa.SDPAImpl.forward
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} logger
:canonical: fastvideo.attention.backends.sdpa.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.attention.backends.sdpa.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
