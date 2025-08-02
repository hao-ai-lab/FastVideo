# {py:mod}`fastvideo.configs.models.vaes.wanvae`

```{py:module} fastvideo.configs.models.vaes.wanvae
```

```{autodoc2-docstring} fastvideo.configs.models.vaes.wanvae
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`WanVAEArchConfig <fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`WanVAEConfig <fastvideo.configs.models.vaes.wanvae.WanVAEConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.models.vaes.wanvae.WanVAEConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} WanVAEArchConfig
:canonical: fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig

Bases: {py:obj}`fastvideo.configs.models.vaes.base.VAEArchConfig`

```{autodoc2-docstring} fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} attn_scales
:canonical: fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig.attn_scales
:type: tuple[float, ...]
:value: >
   ()

```{autodoc2-docstring} fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig.attn_scales
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} base_dim
:canonical: fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig.base_dim
:type: int
:value: >
   96

```{autodoc2-docstring} fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig.base_dim
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} clip_output
:canonical: fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig.clip_output
:type: bool
:value: >
   True

```{autodoc2-docstring} fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig.clip_output
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} decoder_base_dim
:canonical: fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig.decoder_base_dim
:type: int | None
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig.decoder_base_dim
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} dim_mult
:canonical: fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig.dim_mult
:type: tuple[int, ...]
:value: >
   (1, 2, 4, 4)

```{autodoc2-docstring} fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig.dim_mult
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} dropout
:canonical: fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig.dropout
:type: float
:value: >
   0.0

```{autodoc2-docstring} fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig.dropout
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} in_channels
:canonical: fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig.in_channels
:type: int
:value: >
   3

```{autodoc2-docstring} fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig.in_channels
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} is_residual
:canonical: fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig.is_residual
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig.is_residual
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} latents_mean
:canonical: fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig.latents_mean
:type: tuple[float, ...]
:value: >
   ()

```{autodoc2-docstring} fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig.latents_mean
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} latents_std
:canonical: fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig.latents_std
:type: tuple[float, ...]
:value: >
   (2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6...

```{autodoc2-docstring} fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig.latents_std
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_res_blocks
:canonical: fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig.num_res_blocks
:type: int
:value: >
   2

```{autodoc2-docstring} fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig.num_res_blocks
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} out_channels
:canonical: fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig.out_channels
:type: int
:value: >
   3

```{autodoc2-docstring} fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig.out_channels
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} patch_size
:canonical: fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig.patch_size
:type: int | None
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig.patch_size
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} scale_factor_spatial
:canonical: fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig.scale_factor_spatial
:type: int
:value: >
   8

```{autodoc2-docstring} fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig.scale_factor_spatial
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} scale_factor_temporal
:canonical: fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig.scale_factor_temporal
:type: int
:value: >
   4

```{autodoc2-docstring} fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig.scale_factor_temporal
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} temperal_downsample
:canonical: fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig.temperal_downsample
:type: tuple[bool, ...]
:value: >
   (False, True, True)

```{autodoc2-docstring} fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig.temperal_downsample
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} z_dim
:canonical: fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig.z_dim
:type: int
:value: >
   16

```{autodoc2-docstring} fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig.z_dim
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} WanVAEConfig
:canonical: fastvideo.configs.models.vaes.wanvae.WanVAEConfig

Bases: {py:obj}`fastvideo.configs.models.vaes.base.VAEConfig`

```{autodoc2-docstring} fastvideo.configs.models.vaes.wanvae.WanVAEConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} arch_config
:canonical: fastvideo.configs.models.vaes.wanvae.WanVAEConfig.arch_config
:type: fastvideo.configs.models.vaes.wanvae.WanVAEArchConfig
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.vaes.wanvae.WanVAEConfig.arch_config
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} use_feature_cache
:canonical: fastvideo.configs.models.vaes.wanvae.WanVAEConfig.use_feature_cache
:type: bool
:value: >
   True

```{autodoc2-docstring} fastvideo.configs.models.vaes.wanvae.WanVAEConfig.use_feature_cache
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} use_parallel_tiling
:canonical: fastvideo.configs.models.vaes.wanvae.WanVAEConfig.use_parallel_tiling
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.configs.models.vaes.wanvae.WanVAEConfig.use_parallel_tiling
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} use_temporal_tiling
:canonical: fastvideo.configs.models.vaes.wanvae.WanVAEConfig.use_temporal_tiling
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.configs.models.vaes.wanvae.WanVAEConfig.use_temporal_tiling
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} use_tiling
:canonical: fastvideo.configs.models.vaes.wanvae.WanVAEConfig.use_tiling
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.configs.models.vaes.wanvae.WanVAEConfig.use_tiling
:parser: docs.source.autodoc2_docstring_parser
```

````

`````
