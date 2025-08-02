# {py:mod}`fastvideo.attention.backends.flash_attn`

```{py:module} fastvideo.attention.backends.flash_attn
```

```{autodoc2-docstring} fastvideo.attention.backends.flash_attn
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FlashAttentionBackend <fastvideo.attention.backends.flash_attn.FlashAttentionBackend>`
  -
* - {py:obj}`FlashAttentionImpl <fastvideo.attention.backends.flash_attn.FlashAttentionImpl>`
  -
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.attention.backends.flash_attn.logger>`
  - ```{autodoc2-docstring} fastvideo.attention.backends.flash_attn.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} FlashAttentionBackend
:canonical: fastvideo.attention.backends.flash_attn.FlashAttentionBackend

Bases: {py:obj}`fastvideo.attention.backends.abstract.AttentionBackend`

````{py:attribute} accept_output_buffer
:canonical: fastvideo.attention.backends.flash_attn.FlashAttentionBackend.accept_output_buffer
:type: bool
:value: >
   True

```{autodoc2-docstring} fastvideo.attention.backends.flash_attn.FlashAttentionBackend.accept_output_buffer
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_builder_cls() -> type[fastvideo.attention.backends.abstract.AttentionMetadataBuilder]
:canonical: fastvideo.attention.backends.flash_attn.FlashAttentionBackend.get_builder_cls
:abstractmethod:
:staticmethod:

```{autodoc2-docstring} fastvideo.attention.backends.flash_attn.FlashAttentionBackend.get_builder_cls
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_impl_cls() -> type[fastvideo.attention.backends.flash_attn.FlashAttentionImpl]
:canonical: fastvideo.attention.backends.flash_attn.FlashAttentionBackend.get_impl_cls
:staticmethod:

```{autodoc2-docstring} fastvideo.attention.backends.flash_attn.FlashAttentionBackend.get_impl_cls
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_metadata_cls() -> type[fastvideo.attention.backends.abstract.AttentionMetadata]
:canonical: fastvideo.attention.backends.flash_attn.FlashAttentionBackend.get_metadata_cls
:abstractmethod:
:staticmethod:

```{autodoc2-docstring} fastvideo.attention.backends.flash_attn.FlashAttentionBackend.get_metadata_cls
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_name() -> str
:canonical: fastvideo.attention.backends.flash_attn.FlashAttentionBackend.get_name
:staticmethod:

```{autodoc2-docstring} fastvideo.attention.backends.flash_attn.FlashAttentionBackend.get_name
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_supported_head_sizes() -> list[int]
:canonical: fastvideo.attention.backends.flash_attn.FlashAttentionBackend.get_supported_head_sizes
:staticmethod:

```{autodoc2-docstring} fastvideo.attention.backends.flash_attn.FlashAttentionBackend.get_supported_head_sizes
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} FlashAttentionImpl(num_heads: int, head_size: int, causal: bool, softmax_scale: float, num_kv_heads: int | None = None, prefix: str = '', **extra_impl_args)
:canonical: fastvideo.attention.backends.flash_attn.FlashAttentionImpl

Bases: {py:obj}`fastvideo.attention.backends.abstract.AttentionImpl`

````{py:method} forward(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_metadata: fastvideo.attention.backends.abstract.AttentionMetadata)
:canonical: fastvideo.attention.backends.flash_attn.FlashAttentionImpl.forward

```{autodoc2-docstring} fastvideo.attention.backends.flash_attn.FlashAttentionImpl.forward
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} logger
:canonical: fastvideo.attention.backends.flash_attn.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.attention.backends.flash_attn.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
