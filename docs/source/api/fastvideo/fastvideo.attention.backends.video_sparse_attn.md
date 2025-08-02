# {py:mod}`fastvideo.attention.backends.video_sparse_attn`

```{py:module} fastvideo.attention.backends.video_sparse_attn
```

```{autodoc2-docstring} fastvideo.attention.backends.video_sparse_attn
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VideoSparseAttentionBackend <fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionBackend>`
  -
* - {py:obj}`VideoSparseAttentionImpl <fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionImpl>`
  -
* - {py:obj}`VideoSparseAttentionMetadata <fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionMetadata>`
  -
* - {py:obj}`VideoSparseAttentionMetadataBuilder <fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionMetadataBuilder>`
  -
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`construct_variable_block_sizes <fastvideo.attention.backends.video_sparse_attn.construct_variable_block_sizes>`
  - ```{autodoc2-docstring} fastvideo.attention.backends.video_sparse_attn.construct_variable_block_sizes
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`get_non_pad_index <fastvideo.attention.backends.video_sparse_attn.get_non_pad_index>`
  - ```{autodoc2-docstring} fastvideo.attention.backends.video_sparse_attn.get_non_pad_index
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`get_reverse_tile_partition_indices <fastvideo.attention.backends.video_sparse_attn.get_reverse_tile_partition_indices>`
  - ```{autodoc2-docstring} fastvideo.attention.backends.video_sparse_attn.get_reverse_tile_partition_indices
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`get_tile_partition_indices <fastvideo.attention.backends.video_sparse_attn.get_tile_partition_indices>`
  - ```{autodoc2-docstring} fastvideo.attention.backends.video_sparse_attn.get_tile_partition_indices
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VSA_TILE_SIZE <fastvideo.attention.backends.video_sparse_attn.VSA_TILE_SIZE>`
  - ```{autodoc2-docstring} fastvideo.attention.backends.video_sparse_attn.VSA_TILE_SIZE
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`logger <fastvideo.attention.backends.video_sparse_attn.logger>`
  - ```{autodoc2-docstring} fastvideo.attention.backends.video_sparse_attn.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

````{py:data} VSA_TILE_SIZE
:canonical: fastvideo.attention.backends.video_sparse_attn.VSA_TILE_SIZE
:value: >
   (4, 4, 4)

```{autodoc2-docstring} fastvideo.attention.backends.video_sparse_attn.VSA_TILE_SIZE
:parser: docs.source.autodoc2_docstring_parser
```

````

`````{py:class} VideoSparseAttentionBackend
:canonical: fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionBackend

Bases: {py:obj}`fastvideo.attention.backends.abstract.AttentionBackend`

````{py:attribute} accept_output_buffer
:canonical: fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionBackend.accept_output_buffer
:type: bool
:value: >
   True

```{autodoc2-docstring} fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionBackend.accept_output_buffer
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_builder_cls() -> type[fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionMetadataBuilder]
:canonical: fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionBackend.get_builder_cls
:staticmethod:

```{autodoc2-docstring} fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionBackend.get_builder_cls
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_impl_cls() -> type[fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionImpl]
:canonical: fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionBackend.get_impl_cls
:staticmethod:

```{autodoc2-docstring} fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionBackend.get_impl_cls
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_metadata_cls() -> type[fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionMetadata]
:canonical: fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionBackend.get_metadata_cls
:staticmethod:

```{autodoc2-docstring} fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionBackend.get_metadata_cls
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_name() -> str
:canonical: fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionBackend.get_name
:staticmethod:

```{autodoc2-docstring} fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionBackend.get_name
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_supported_head_sizes() -> list[int]
:canonical: fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionBackend.get_supported_head_sizes
:staticmethod:

```{autodoc2-docstring} fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionBackend.get_supported_head_sizes
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} VideoSparseAttentionImpl(num_heads: int, head_size: int, causal: bool, softmax_scale: float, num_kv_heads: int | None = None, prefix: str = '', **extra_impl_args)
:canonical: fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionImpl

Bases: {py:obj}`fastvideo.attention.backends.abstract.AttentionImpl`

````{py:method} forward(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, gate_compress: torch.Tensor, attn_metadata: fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionMetadata) -> torch.Tensor
:canonical: fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionImpl.forward

```{autodoc2-docstring} fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionImpl.forward
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} postprocess_output(output: torch.Tensor, attn_metadata: fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionMetadata) -> torch.Tensor
:canonical: fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionImpl.postprocess_output

````

````{py:method} preprocess_qkv(qkv: torch.Tensor, attn_metadata: fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionMetadata) -> torch.Tensor
:canonical: fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionImpl.preprocess_qkv

````

````{py:method} tile(x: torch.Tensor, num_tiles: list[int], tile_partition_indices: torch.LongTensor, non_pad_index: torch.LongTensor) -> torch.Tensor
:canonical: fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionImpl.tile

```{autodoc2-docstring} fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionImpl.tile
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} untile(x: torch.Tensor, reverse_tile_partition_indices: torch.LongTensor, non_pad_index: torch.LongTensor) -> torch.Tensor
:canonical: fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionImpl.untile

```{autodoc2-docstring} fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionImpl.untile
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} VideoSparseAttentionMetadata
:canonical: fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionMetadata

Bases: {py:obj}`fastvideo.attention.backends.abstract.AttentionMetadata`

````{py:attribute} VSA_sparsity
:canonical: fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionMetadata.VSA_sparsity
:type: float
:value: >
   None

```{autodoc2-docstring} fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionMetadata.VSA_sparsity
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} current_timestep
:canonical: fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionMetadata.current_timestep
:type: int
:value: >
   None

```{autodoc2-docstring} fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionMetadata.current_timestep
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} dit_seq_shape
:canonical: fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionMetadata.dit_seq_shape
:type: list[int]
:value: >
   None

```{autodoc2-docstring} fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionMetadata.dit_seq_shape
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} non_pad_index
:canonical: fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionMetadata.non_pad_index
:type: torch.LongTensor
:value: >
   None

```{autodoc2-docstring} fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionMetadata.non_pad_index
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_tiles
:canonical: fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionMetadata.num_tiles
:type: list[int]
:value: >
   None

```{autodoc2-docstring} fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionMetadata.num_tiles
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} reverse_tile_partition_indices
:canonical: fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionMetadata.reverse_tile_partition_indices
:type: torch.LongTensor
:value: >
   None

```{autodoc2-docstring} fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionMetadata.reverse_tile_partition_indices
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} tile_partition_indices
:canonical: fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionMetadata.tile_partition_indices
:type: torch.LongTensor
:value: >
   None

```{autodoc2-docstring} fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionMetadata.tile_partition_indices
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} total_seq_length
:canonical: fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionMetadata.total_seq_length
:type: int
:value: >
   None

```{autodoc2-docstring} fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionMetadata.total_seq_length
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} variable_block_sizes
:canonical: fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionMetadata.variable_block_sizes
:type: torch.LongTensor
:value: >
   None

```{autodoc2-docstring} fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionMetadata.variable_block_sizes
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} VideoSparseAttentionMetadataBuilder()
:canonical: fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionMetadataBuilder

Bases: {py:obj}`fastvideo.attention.backends.abstract.AttentionMetadataBuilder`

````{py:method} build(current_timestep: int, raw_latent_shape: tuple[int, int, int], patch_size: tuple[int, int, int], VSA_sparsity: float, device: torch.device, **kwargs: dict[str, typing.Any]) -> fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionMetadata
:canonical: fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionMetadataBuilder.build

````

````{py:method} prepare()
:canonical: fastvideo.attention.backends.video_sparse_attn.VideoSparseAttentionMetadataBuilder.prepare

````

`````

````{py:function} construct_variable_block_sizes(dit_seq_shape: tuple[int, int, int], num_tiles: tuple[int, int, int], device: torch.device) -> torch.LongTensor
:canonical: fastvideo.attention.backends.video_sparse_attn.construct_variable_block_sizes

```{autodoc2-docstring} fastvideo.attention.backends.video_sparse_attn.construct_variable_block_sizes
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} get_non_pad_index(variable_block_sizes: torch.LongTensor, max_block_size: int)
:canonical: fastvideo.attention.backends.video_sparse_attn.get_non_pad_index

```{autodoc2-docstring} fastvideo.attention.backends.video_sparse_attn.get_non_pad_index
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} get_reverse_tile_partition_indices(dit_seq_shape: tuple[int, int, int], tile_size: tuple[int, int, int], device: torch.device) -> torch.LongTensor
:canonical: fastvideo.attention.backends.video_sparse_attn.get_reverse_tile_partition_indices

```{autodoc2-docstring} fastvideo.attention.backends.video_sparse_attn.get_reverse_tile_partition_indices
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} get_tile_partition_indices(dit_seq_shape: tuple[int, int, int], tile_size: tuple[int, int, int], device: torch.device) -> torch.LongTensor
:canonical: fastvideo.attention.backends.video_sparse_attn.get_tile_partition_indices

```{autodoc2-docstring} fastvideo.attention.backends.video_sparse_attn.get_tile_partition_indices
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:data} logger
:canonical: fastvideo.attention.backends.video_sparse_attn.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.attention.backends.video_sparse_attn.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
