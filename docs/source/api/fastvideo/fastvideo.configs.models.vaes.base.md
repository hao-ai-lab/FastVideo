# {py:mod}`fastvideo.configs.models.vaes.base`

```{py:module} fastvideo.configs.models.vaes.base
```

```{autodoc2-docstring} fastvideo.configs.models.vaes.base
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VAEArchConfig <fastvideo.configs.models.vaes.base.VAEArchConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.models.vaes.base.VAEArchConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`VAEConfig <fastvideo.configs.models.vaes.base.VAEConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.models.vaes.base.VAEConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} VAEArchConfig
:canonical: fastvideo.configs.models.vaes.base.VAEArchConfig

Bases: {py:obj}`fastvideo.configs.models.base.ArchConfig`

```{autodoc2-docstring} fastvideo.configs.models.vaes.base.VAEArchConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} scaling_factor
:canonical: fastvideo.configs.models.vaes.base.VAEArchConfig.scaling_factor
:type: float | torch.Tensor
:value: >
   0

```{autodoc2-docstring} fastvideo.configs.models.vaes.base.VAEArchConfig.scaling_factor
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} spatial_compression_ratio
:canonical: fastvideo.configs.models.vaes.base.VAEArchConfig.spatial_compression_ratio
:type: int
:value: >
   8

```{autodoc2-docstring} fastvideo.configs.models.vaes.base.VAEArchConfig.spatial_compression_ratio
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} temporal_compression_ratio
:canonical: fastvideo.configs.models.vaes.base.VAEArchConfig.temporal_compression_ratio
:type: int
:value: >
   4

```{autodoc2-docstring} fastvideo.configs.models.vaes.base.VAEArchConfig.temporal_compression_ratio
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} VAEConfig
:canonical: fastvideo.configs.models.vaes.base.VAEConfig

Bases: {py:obj}`fastvideo.configs.models.base.ModelConfig`

```{autodoc2-docstring} fastvideo.configs.models.vaes.base.VAEConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} add_cli_args(parser: typing.Any, prefix: str = 'vae-config') -> typing.Any
:canonical: fastvideo.configs.models.vaes.base.VAEConfig.add_cli_args
:staticmethod:

```{autodoc2-docstring} fastvideo.configs.models.vaes.base.VAEConfig.add_cli_args
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} arch_config
:canonical: fastvideo.configs.models.vaes.base.VAEConfig.arch_config
:type: fastvideo.configs.models.vaes.base.VAEArchConfig
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.vaes.base.VAEConfig.arch_config
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} blend_num_frames
:canonical: fastvideo.configs.models.vaes.base.VAEConfig.blend_num_frames
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.configs.models.vaes.base.VAEConfig.blend_num_frames
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} from_cli_args(args: argparse.Namespace) -> fastvideo.configs.models.vaes.base.VAEConfig
:canonical: fastvideo.configs.models.vaes.base.VAEConfig.from_cli_args
:classmethod:

```{autodoc2-docstring} fastvideo.configs.models.vaes.base.VAEConfig.from_cli_args
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} load_decoder
:canonical: fastvideo.configs.models.vaes.base.VAEConfig.load_decoder
:type: bool
:value: >
   True

```{autodoc2-docstring} fastvideo.configs.models.vaes.base.VAEConfig.load_decoder
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} load_encoder
:canonical: fastvideo.configs.models.vaes.base.VAEConfig.load_encoder
:type: bool
:value: >
   True

```{autodoc2-docstring} fastvideo.configs.models.vaes.base.VAEConfig.load_encoder
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} tile_sample_min_height
:canonical: fastvideo.configs.models.vaes.base.VAEConfig.tile_sample_min_height
:type: int
:value: >
   256

```{autodoc2-docstring} fastvideo.configs.models.vaes.base.VAEConfig.tile_sample_min_height
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} tile_sample_min_num_frames
:canonical: fastvideo.configs.models.vaes.base.VAEConfig.tile_sample_min_num_frames
:type: int
:value: >
   16

```{autodoc2-docstring} fastvideo.configs.models.vaes.base.VAEConfig.tile_sample_min_num_frames
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} tile_sample_min_width
:canonical: fastvideo.configs.models.vaes.base.VAEConfig.tile_sample_min_width
:type: int
:value: >
   256

```{autodoc2-docstring} fastvideo.configs.models.vaes.base.VAEConfig.tile_sample_min_width
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} tile_sample_stride_height
:canonical: fastvideo.configs.models.vaes.base.VAEConfig.tile_sample_stride_height
:type: int
:value: >
   192

```{autodoc2-docstring} fastvideo.configs.models.vaes.base.VAEConfig.tile_sample_stride_height
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} tile_sample_stride_num_frames
:canonical: fastvideo.configs.models.vaes.base.VAEConfig.tile_sample_stride_num_frames
:type: int
:value: >
   12

```{autodoc2-docstring} fastvideo.configs.models.vaes.base.VAEConfig.tile_sample_stride_num_frames
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} tile_sample_stride_width
:canonical: fastvideo.configs.models.vaes.base.VAEConfig.tile_sample_stride_width
:type: int
:value: >
   192

```{autodoc2-docstring} fastvideo.configs.models.vaes.base.VAEConfig.tile_sample_stride_width
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} use_parallel_tiling
:canonical: fastvideo.configs.models.vaes.base.VAEConfig.use_parallel_tiling
:type: bool
:value: >
   True

```{autodoc2-docstring} fastvideo.configs.models.vaes.base.VAEConfig.use_parallel_tiling
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} use_temporal_scaling_frames
:canonical: fastvideo.configs.models.vaes.base.VAEConfig.use_temporal_scaling_frames
:type: bool
:value: >
   True

```{autodoc2-docstring} fastvideo.configs.models.vaes.base.VAEConfig.use_temporal_scaling_frames
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} use_temporal_tiling
:canonical: fastvideo.configs.models.vaes.base.VAEConfig.use_temporal_tiling
:type: bool
:value: >
   True

```{autodoc2-docstring} fastvideo.configs.models.vaes.base.VAEConfig.use_temporal_tiling
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} use_tiling
:canonical: fastvideo.configs.models.vaes.base.VAEConfig.use_tiling
:type: bool
:value: >
   True

```{autodoc2-docstring} fastvideo.configs.models.vaes.base.VAEConfig.use_tiling
:parser: docs.source.autodoc2_docstring_parser
```

````

`````
