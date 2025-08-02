# {py:mod}`fastvideo.configs.models.vaes.stepvideovae`

```{py:module} fastvideo.configs.models.vaes.stepvideovae
```

```{autodoc2-docstring} fastvideo.configs.models.vaes.stepvideovae
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`StepVideoVAEArchConfig <fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEArchConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEArchConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`StepVideoVAEConfig <fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} StepVideoVAEArchConfig
:canonical: fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEArchConfig

Bases: {py:obj}`fastvideo.configs.models.vaes.base.VAEArchConfig`

```{autodoc2-docstring} fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEArchConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} frame_len
:canonical: fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEArchConfig.frame_len
:type: int
:value: >
   17

```{autodoc2-docstring} fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEArchConfig.frame_len
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} in_channels
:canonical: fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEArchConfig.in_channels
:type: int
:value: >
   3

```{autodoc2-docstring} fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEArchConfig.in_channels
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_res_blocks
:canonical: fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEArchConfig.num_res_blocks
:type: int
:value: >
   2

```{autodoc2-docstring} fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEArchConfig.num_res_blocks
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} out_channels
:canonical: fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEArchConfig.out_channels
:type: int
:value: >
   3

```{autodoc2-docstring} fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEArchConfig.out_channels
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} scaling_factor
:canonical: fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEArchConfig.scaling_factor
:type: float
:value: >
   1.0

```{autodoc2-docstring} fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEArchConfig.scaling_factor
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} spatial_compression_ratio
:canonical: fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEArchConfig.spatial_compression_ratio
:type: int
:value: >
   16

```{autodoc2-docstring} fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEArchConfig.spatial_compression_ratio
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} temporal_compression_ratio
:canonical: fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEArchConfig.temporal_compression_ratio
:type: int
:value: >
   8

```{autodoc2-docstring} fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEArchConfig.temporal_compression_ratio
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} version
:canonical: fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEArchConfig.version
:type: int
:value: >
   2

```{autodoc2-docstring} fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEArchConfig.version
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} world_size
:canonical: fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEArchConfig.world_size
:type: int
:value: >
   1

```{autodoc2-docstring} fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEArchConfig.world_size
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} z_channels
:canonical: fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEArchConfig.z_channels
:type: int
:value: >
   64

```{autodoc2-docstring} fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEArchConfig.z_channels
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} StepVideoVAEConfig
:canonical: fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEConfig

Bases: {py:obj}`fastvideo.configs.models.vaes.base.VAEConfig`

```{autodoc2-docstring} fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} arch_config
:canonical: fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEConfig.arch_config
:type: fastvideo.configs.models.vaes.base.VAEArchConfig
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEConfig.arch_config
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} use_parallel_tiling
:canonical: fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEConfig.use_parallel_tiling
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEConfig.use_parallel_tiling
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} use_temporal_scaling_frames
:canonical: fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEConfig.use_temporal_scaling_frames
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEConfig.use_temporal_scaling_frames
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} use_temporal_tiling
:canonical: fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEConfig.use_temporal_tiling
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEConfig.use_temporal_tiling
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} use_tiling
:canonical: fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEConfig.use_tiling
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.configs.models.vaes.stepvideovae.StepVideoVAEConfig.use_tiling
:parser: docs.source.autodoc2_docstring_parser
```

````

`````
