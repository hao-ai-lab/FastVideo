# {py:mod}`fastvideo.layers.visual_embedding`

```{py:module} fastvideo.layers.visual_embedding
```

```{autodoc2-docstring} fastvideo.layers.visual_embedding
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ModulateProjection <fastvideo.layers.visual_embedding.ModulateProjection>`
  - ```{autodoc2-docstring} fastvideo.layers.visual_embedding.ModulateProjection
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`PatchEmbed <fastvideo.layers.visual_embedding.PatchEmbed>`
  - ```{autodoc2-docstring} fastvideo.layers.visual_embedding.PatchEmbed
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`TimestepEmbedder <fastvideo.layers.visual_embedding.TimestepEmbedder>`
  - ```{autodoc2-docstring} fastvideo.layers.visual_embedding.TimestepEmbedder
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`timestep_embedding <fastvideo.layers.visual_embedding.timestep_embedding>`
  - ```{autodoc2-docstring} fastvideo.layers.visual_embedding.timestep_embedding
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`unpatchify <fastvideo.layers.visual_embedding.unpatchify>`
  - ```{autodoc2-docstring} fastvideo.layers.visual_embedding.unpatchify
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} ModulateProjection(hidden_size: int, factor: int = 2, act_layer: str = 'silu', dtype: torch.dtype | None = None, prefix: str = '')
:canonical: fastvideo.layers.visual_embedding.ModulateProjection

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} fastvideo.layers.visual_embedding.ModulateProjection
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.layers.visual_embedding.ModulateProjection.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} forward(x: torch.Tensor) -> torch.Tensor
:canonical: fastvideo.layers.visual_embedding.ModulateProjection.forward

```{autodoc2-docstring} fastvideo.layers.visual_embedding.ModulateProjection.forward
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} PatchEmbed(patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, bias=True, dtype=None, prefix: str = '')
:canonical: fastvideo.layers.visual_embedding.PatchEmbed

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} fastvideo.layers.visual_embedding.PatchEmbed
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.layers.visual_embedding.PatchEmbed.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} forward(x)
:canonical: fastvideo.layers.visual_embedding.PatchEmbed.forward

```{autodoc2-docstring} fastvideo.layers.visual_embedding.PatchEmbed.forward
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} TimestepEmbedder(hidden_size, act_layer='silu', frequency_embedding_size=256, max_period=10000, dtype=None, freq_dtype=torch.float32, prefix: str = '')
:canonical: fastvideo.layers.visual_embedding.TimestepEmbedder

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} fastvideo.layers.visual_embedding.TimestepEmbedder
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.layers.visual_embedding.TimestepEmbedder.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} forward(t: torch.Tensor) -> torch.Tensor
:canonical: fastvideo.layers.visual_embedding.TimestepEmbedder.forward

```{autodoc2-docstring} fastvideo.layers.visual_embedding.TimestepEmbedder.forward
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:function} timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000, dtype: torch.dtype = torch.float32) -> torch.Tensor
:canonical: fastvideo.layers.visual_embedding.timestep_embedding

```{autodoc2-docstring} fastvideo.layers.visual_embedding.timestep_embedding
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} unpatchify(x, t, h, w, patch_size, channels) -> torch.Tensor
:canonical: fastvideo.layers.visual_embedding.unpatchify

```{autodoc2-docstring} fastvideo.layers.visual_embedding.unpatchify
:parser: docs.source.autodoc2_docstring_parser
```
````
