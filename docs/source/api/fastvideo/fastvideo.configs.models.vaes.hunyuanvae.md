# {py:mod}`fastvideo.configs.models.vaes.hunyuanvae`

```{py:module} fastvideo.configs.models.vaes.hunyuanvae
```

```{autodoc2-docstring} fastvideo.configs.models.vaes.hunyuanvae
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HunyuanVAEArchConfig <fastvideo.configs.models.vaes.hunyuanvae.HunyuanVAEArchConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.models.vaes.hunyuanvae.HunyuanVAEArchConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`HunyuanVAEConfig <fastvideo.configs.models.vaes.hunyuanvae.HunyuanVAEConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.models.vaes.hunyuanvae.HunyuanVAEConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} HunyuanVAEArchConfig
:canonical: fastvideo.configs.models.vaes.hunyuanvae.HunyuanVAEArchConfig

Bases: {py:obj}`fastvideo.configs.models.vaes.base.VAEArchConfig`

```{autodoc2-docstring} fastvideo.configs.models.vaes.hunyuanvae.HunyuanVAEArchConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} act_fn
:canonical: fastvideo.configs.models.vaes.hunyuanvae.HunyuanVAEArchConfig.act_fn
:type: str
:value: >
   'silu'

```{autodoc2-docstring} fastvideo.configs.models.vaes.hunyuanvae.HunyuanVAEArchConfig.act_fn
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} block_out_channels
:canonical: fastvideo.configs.models.vaes.hunyuanvae.HunyuanVAEArchConfig.block_out_channels
:type: tuple[int, ...]
:value: >
   (128, 256, 512, 512)

```{autodoc2-docstring} fastvideo.configs.models.vaes.hunyuanvae.HunyuanVAEArchConfig.block_out_channels
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} down_block_types
:canonical: fastvideo.configs.models.vaes.hunyuanvae.HunyuanVAEArchConfig.down_block_types
:type: tuple[str, ...]
:value: >
   ('HunyuanVideoDownBlock3D', 'HunyuanVideoDownBlock3D', 'HunyuanVideoDownBlock3D', 'HunyuanVideoDownB...

```{autodoc2-docstring} fastvideo.configs.models.vaes.hunyuanvae.HunyuanVAEArchConfig.down_block_types
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} in_channels
:canonical: fastvideo.configs.models.vaes.hunyuanvae.HunyuanVAEArchConfig.in_channels
:type: int
:value: >
   3

```{autodoc2-docstring} fastvideo.configs.models.vaes.hunyuanvae.HunyuanVAEArchConfig.in_channels
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} latent_channels
:canonical: fastvideo.configs.models.vaes.hunyuanvae.HunyuanVAEArchConfig.latent_channels
:type: int
:value: >
   16

```{autodoc2-docstring} fastvideo.configs.models.vaes.hunyuanvae.HunyuanVAEArchConfig.latent_channels
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} layers_per_block
:canonical: fastvideo.configs.models.vaes.hunyuanvae.HunyuanVAEArchConfig.layers_per_block
:type: int
:value: >
   2

```{autodoc2-docstring} fastvideo.configs.models.vaes.hunyuanvae.HunyuanVAEArchConfig.layers_per_block
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} mid_block_add_attention
:canonical: fastvideo.configs.models.vaes.hunyuanvae.HunyuanVAEArchConfig.mid_block_add_attention
:type: bool
:value: >
   True

```{autodoc2-docstring} fastvideo.configs.models.vaes.hunyuanvae.HunyuanVAEArchConfig.mid_block_add_attention
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} norm_num_groups
:canonical: fastvideo.configs.models.vaes.hunyuanvae.HunyuanVAEArchConfig.norm_num_groups
:type: int
:value: >
   32

```{autodoc2-docstring} fastvideo.configs.models.vaes.hunyuanvae.HunyuanVAEArchConfig.norm_num_groups
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} out_channels
:canonical: fastvideo.configs.models.vaes.hunyuanvae.HunyuanVAEArchConfig.out_channels
:type: int
:value: >
   3

```{autodoc2-docstring} fastvideo.configs.models.vaes.hunyuanvae.HunyuanVAEArchConfig.out_channels
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} scaling_factor
:canonical: fastvideo.configs.models.vaes.hunyuanvae.HunyuanVAEArchConfig.scaling_factor
:type: float
:value: >
   0.476986

```{autodoc2-docstring} fastvideo.configs.models.vaes.hunyuanvae.HunyuanVAEArchConfig.scaling_factor
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} spatial_compression_ratio
:canonical: fastvideo.configs.models.vaes.hunyuanvae.HunyuanVAEArchConfig.spatial_compression_ratio
:type: int
:value: >
   8

```{autodoc2-docstring} fastvideo.configs.models.vaes.hunyuanvae.HunyuanVAEArchConfig.spatial_compression_ratio
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} temporal_compression_ratio
:canonical: fastvideo.configs.models.vaes.hunyuanvae.HunyuanVAEArchConfig.temporal_compression_ratio
:type: int
:value: >
   4

```{autodoc2-docstring} fastvideo.configs.models.vaes.hunyuanvae.HunyuanVAEArchConfig.temporal_compression_ratio
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} up_block_types
:canonical: fastvideo.configs.models.vaes.hunyuanvae.HunyuanVAEArchConfig.up_block_types
:type: tuple[str, ...]
:value: >
   ('HunyuanVideoUpBlock3D', 'HunyuanVideoUpBlock3D', 'HunyuanVideoUpBlock3D', 'HunyuanVideoUpBlock3D')

```{autodoc2-docstring} fastvideo.configs.models.vaes.hunyuanvae.HunyuanVAEArchConfig.up_block_types
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} HunyuanVAEConfig
:canonical: fastvideo.configs.models.vaes.hunyuanvae.HunyuanVAEConfig

Bases: {py:obj}`fastvideo.configs.models.vaes.base.VAEConfig`

```{autodoc2-docstring} fastvideo.configs.models.vaes.hunyuanvae.HunyuanVAEConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} arch_config
:canonical: fastvideo.configs.models.vaes.hunyuanvae.HunyuanVAEConfig.arch_config
:type: fastvideo.configs.models.vaes.base.VAEArchConfig
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.vaes.hunyuanvae.HunyuanVAEConfig.arch_config
:parser: docs.source.autodoc2_docstring_parser
```

````

`````
