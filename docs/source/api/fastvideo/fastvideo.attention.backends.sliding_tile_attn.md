# {py:mod}`fastvideo.attention.backends.sliding_tile_attn`

```{py:module} fastvideo.attention.backends.sliding_tile_attn
```

```{autodoc2-docstring} fastvideo.attention.backends.sliding_tile_attn
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RangeDict <fastvideo.attention.backends.sliding_tile_attn.RangeDict>`
  -
* - {py:obj}`SlidingTileAttentionBackend <fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionBackend>`
  -
* - {py:obj}`SlidingTileAttentionImpl <fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionImpl>`
  -
* - {py:obj}`SlidingTileAttentionMetadata <fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionMetadata>`
  -
* - {py:obj}`SlidingTileAttentionMetadataBuilder <fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionMetadataBuilder>`
  -
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.attention.backends.sliding_tile_attn.logger>`
  - ```{autodoc2-docstring} fastvideo.attention.backends.sliding_tile_attn.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

```{py:class} RangeDict()
:canonical: fastvideo.attention.backends.sliding_tile_attn.RangeDict

Bases: {py:obj}`dict`

```

`````{py:class} SlidingTileAttentionBackend
:canonical: fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionBackend

Bases: {py:obj}`fastvideo.attention.backends.abstract.AttentionBackend`

````{py:attribute} accept_output_buffer
:canonical: fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionBackend.accept_output_buffer
:type: bool
:value: >
   True

```{autodoc2-docstring} fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionBackend.accept_output_buffer
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_builder_cls() -> type[fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionMetadataBuilder]
:canonical: fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionBackend.get_builder_cls
:staticmethod:

```{autodoc2-docstring} fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionBackend.get_builder_cls
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_impl_cls() -> type[fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionImpl]
:canonical: fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionBackend.get_impl_cls
:staticmethod:

```{autodoc2-docstring} fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionBackend.get_impl_cls
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_metadata_cls() -> type[fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionMetadata]
:canonical: fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionBackend.get_metadata_cls
:staticmethod:

```{autodoc2-docstring} fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionBackend.get_metadata_cls
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_name() -> str
:canonical: fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionBackend.get_name
:staticmethod:

```{autodoc2-docstring} fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionBackend.get_name
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_supported_head_sizes() -> list[int]
:canonical: fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionBackend.get_supported_head_sizes
:staticmethod:

```{autodoc2-docstring} fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionBackend.get_supported_head_sizes
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} SlidingTileAttentionImpl(num_heads: int, head_size: int, causal: bool, softmax_scale: float, num_kv_heads: int | None = None, prefix: str = '', **extra_impl_args)
:canonical: fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionImpl

Bases: {py:obj}`fastvideo.attention.backends.abstract.AttentionImpl`

````{py:method} forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_metadata: fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionMetadata) -> torch.Tensor
:canonical: fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionImpl.forward

```{autodoc2-docstring} fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionImpl.forward
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} postprocess_output(output: torch.Tensor, attn_metadata: fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionMetadata) -> torch.Tensor
:canonical: fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionImpl.postprocess_output

````

````{py:method} preprocess_qkv(qkv: torch.Tensor, attn_metadata: fastvideo.attention.backends.abstract.AttentionMetadata) -> torch.Tensor
:canonical: fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionImpl.preprocess_qkv

````

````{py:method} tile(x: torch.Tensor) -> torch.Tensor
:canonical: fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionImpl.tile

```{autodoc2-docstring} fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionImpl.tile
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} untile(x: torch.Tensor) -> torch.Tensor
:canonical: fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionImpl.untile

```{autodoc2-docstring} fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionImpl.untile
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} SlidingTileAttentionMetadata
:canonical: fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionMetadata

Bases: {py:obj}`fastvideo.attention.backends.abstract.AttentionMetadata`

````{py:attribute} STA_param
:canonical: fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionMetadata.STA_param
:type: list[list[typing.Any]]
:value: >
   None

```{autodoc2-docstring} fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionMetadata.STA_param
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} current_timestep
:canonical: fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionMetadata.current_timestep
:type: int
:value: >
   None

```{autodoc2-docstring} fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionMetadata.current_timestep
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} SlidingTileAttentionMetadataBuilder()
:canonical: fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionMetadataBuilder

Bases: {py:obj}`fastvideo.attention.backends.abstract.AttentionMetadataBuilder`

````{py:method} build(STA_param: list[list[typing.Any]], current_timestep: int, **kwargs: dict[str, typing.Any]) -> fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionMetadata
:canonical: fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionMetadataBuilder.build

````

````{py:method} prepare()
:canonical: fastvideo.attention.backends.sliding_tile_attn.SlidingTileAttentionMetadataBuilder.prepare

````

`````

````{py:data} logger
:canonical: fastvideo.attention.backends.sliding_tile_attn.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.attention.backends.sliding_tile_attn.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
