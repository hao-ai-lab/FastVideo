# {py:mod}`fastvideo.layers.rotary_embedding`

```{py:module} fastvideo.layers.rotary_embedding
```

```{autodoc2-docstring} fastvideo.layers.rotary_embedding
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RotaryEmbedding <fastvideo.layers.rotary_embedding.RotaryEmbedding>`
  - ```{autodoc2-docstring} fastvideo.layers.rotary_embedding.RotaryEmbedding
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_1d_rotary_pos_embed <fastvideo.layers.rotary_embedding.get_1d_rotary_pos_embed>`
  - ```{autodoc2-docstring} fastvideo.layers.rotary_embedding.get_1d_rotary_pos_embed
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`get_meshgrid_nd <fastvideo.layers.rotary_embedding.get_meshgrid_nd>`
  - ```{autodoc2-docstring} fastvideo.layers.rotary_embedding.get_meshgrid_nd
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`get_nd_rotary_pos_embed <fastvideo.layers.rotary_embedding.get_nd_rotary_pos_embed>`
  - ```{autodoc2-docstring} fastvideo.layers.rotary_embedding.get_nd_rotary_pos_embed
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`get_rope <fastvideo.layers.rotary_embedding.get_rope>`
  - ```{autodoc2-docstring} fastvideo.layers.rotary_embedding.get_rope
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`get_rotary_pos_embed <fastvideo.layers.rotary_embedding.get_rotary_pos_embed>`
  - ```{autodoc2-docstring} fastvideo.layers.rotary_embedding.get_rotary_pos_embed
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} RotaryEmbedding(head_size: int, rotary_dim: int, max_position_embeddings: int, base: int | float, is_neox_style: bool, dtype: torch.dtype)
:canonical: fastvideo.layers.rotary_embedding.RotaryEmbedding

Bases: {py:obj}`fastvideo.layers.custom_op.CustomOp`

```{autodoc2-docstring} fastvideo.layers.rotary_embedding.RotaryEmbedding
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.layers.rotary_embedding.RotaryEmbedding.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} extra_repr() -> str
:canonical: fastvideo.layers.rotary_embedding.RotaryEmbedding.extra_repr

````

````{py:method} forward_native(positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor, offsets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]
:canonical: fastvideo.layers.rotary_embedding.RotaryEmbedding.forward_native

```{autodoc2-docstring} fastvideo.layers.rotary_embedding.RotaryEmbedding.forward_native
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:function} get_1d_rotary_pos_embed(dim: int, pos: torch.FloatTensor | int, theta: float = 10000.0, theta_rescale_factor: float = 1.0, interpolation_factor: float = 1.0, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]
:canonical: fastvideo.layers.rotary_embedding.get_1d_rotary_pos_embed

```{autodoc2-docstring} fastvideo.layers.rotary_embedding.get_1d_rotary_pos_embed
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} get_meshgrid_nd(start: int | tuple[int, ...], *args: int | tuple[int, ...], dim: int = 2) -> torch.Tensor
:canonical: fastvideo.layers.rotary_embedding.get_meshgrid_nd

```{autodoc2-docstring} fastvideo.layers.rotary_embedding.get_meshgrid_nd
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} get_nd_rotary_pos_embed(rope_dim_list, start, *args, theta=10000.0, theta_rescale_factor: float | list[float] = 1.0, interpolation_factor: float | list[float] = 1.0, shard_dim: int = 0, sp_rank: int = 0, sp_world_size: int = 1, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]
:canonical: fastvideo.layers.rotary_embedding.get_nd_rotary_pos_embed

```{autodoc2-docstring} fastvideo.layers.rotary_embedding.get_nd_rotary_pos_embed
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} get_rope(head_size: int, rotary_dim: int, max_position: int, base: int | float, is_neox_style: bool = True, rope_scaling: dict[str, typing.Any] | None = None, dtype: torch.dtype | None = None, partial_rotary_factor: float = 1.0) -> fastvideo.layers.rotary_embedding.RotaryEmbedding
:canonical: fastvideo.layers.rotary_embedding.get_rope

```{autodoc2-docstring} fastvideo.layers.rotary_embedding.get_rope
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} get_rotary_pos_embed(rope_sizes, hidden_size, heads_num, rope_dim_list, rope_theta, theta_rescale_factor=1.0, interpolation_factor=1.0, shard_dim: int = 0, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]
:canonical: fastvideo.layers.rotary_embedding.get_rotary_pos_embed

```{autodoc2-docstring} fastvideo.layers.rotary_embedding.get_rotary_pos_embed
:parser: docs.source.autodoc2_docstring_parser
```
````
