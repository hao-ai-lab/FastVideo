# {py:mod}`fastvideo.configs.models.dits.hunyuanvideo`

```{py:module} fastvideo.configs.models.dits.hunyuanvideo
```

```{autodoc2-docstring} fastvideo.configs.models.dits.hunyuanvideo
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HunyuanVideoArchConfig <fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`HunyuanVideoConfig <fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`is_double_block <fastvideo.configs.models.dits.hunyuanvideo.is_double_block>`
  - ```{autodoc2-docstring} fastvideo.configs.models.dits.hunyuanvideo.is_double_block
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`is_refiner_block <fastvideo.configs.models.dits.hunyuanvideo.is_refiner_block>`
  - ```{autodoc2-docstring} fastvideo.configs.models.dits.hunyuanvideo.is_refiner_block
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`is_single_block <fastvideo.configs.models.dits.hunyuanvideo.is_single_block>`
  - ```{autodoc2-docstring} fastvideo.configs.models.dits.hunyuanvideo.is_single_block
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`is_txt_in <fastvideo.configs.models.dits.hunyuanvideo.is_txt_in>`
  - ```{autodoc2-docstring} fastvideo.configs.models.dits.hunyuanvideo.is_txt_in
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} HunyuanVideoArchConfig
:canonical: fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig

Bases: {py:obj}`fastvideo.configs.models.dits.base.DiTArchConfig`

```{autodoc2-docstring} fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} attention_head_dim
:canonical: fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.attention_head_dim
:type: int
:value: >
   128

```{autodoc2-docstring} fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.attention_head_dim
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} dtype
:canonical: fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.dtype
:type: torch.dtype | None
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.dtype
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} exclude_lora_layers
:canonical: fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.exclude_lora_layers
:type: list[str]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.exclude_lora_layers
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} guidance_embeds
:canonical: fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.guidance_embeds
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.guidance_embeds
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} in_channels
:canonical: fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.in_channels
:type: int
:value: >
   16

```{autodoc2-docstring} fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.in_channels
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} mlp_ratio
:canonical: fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.mlp_ratio
:type: float
:value: >
   4.0

```{autodoc2-docstring} fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.mlp_ratio
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_attention_heads
:canonical: fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.num_attention_heads
:type: int
:value: >
   24

```{autodoc2-docstring} fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.num_attention_heads
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_layers
:canonical: fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.num_layers
:type: int
:value: >
   20

```{autodoc2-docstring} fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.num_layers
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_refiner_layers
:canonical: fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.num_refiner_layers
:type: int
:value: >
   2

```{autodoc2-docstring} fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.num_refiner_layers
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_single_layers
:canonical: fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.num_single_layers
:type: int
:value: >
   40

```{autodoc2-docstring} fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.num_single_layers
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} out_channels
:canonical: fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.out_channels
:type: int
:value: >
   16

```{autodoc2-docstring} fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.out_channels
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} param_names_mapping
:canonical: fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.param_names_mapping
:type: dict
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.param_names_mapping
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} patch_size
:canonical: fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.patch_size
:type: int
:value: >
   2

```{autodoc2-docstring} fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.patch_size
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} patch_size_t
:canonical: fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.patch_size_t
:type: int
:value: >
   1

```{autodoc2-docstring} fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.patch_size_t
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} pooled_projection_dim
:canonical: fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.pooled_projection_dim
:type: int
:value: >
   768

```{autodoc2-docstring} fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.pooled_projection_dim
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} qk_norm
:canonical: fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.qk_norm
:type: str
:value: >
   'rms_norm'

```{autodoc2-docstring} fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.qk_norm
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} reverse_param_names_mapping
:canonical: fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.reverse_param_names_mapping
:type: dict
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.reverse_param_names_mapping
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} rope_axes_dim
:canonical: fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.rope_axes_dim
:type: tuple[int, int, int]
:value: >
   (16, 56, 56)

```{autodoc2-docstring} fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.rope_axes_dim
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} rope_theta
:canonical: fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.rope_theta
:type: int
:value: >
   256

```{autodoc2-docstring} fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.rope_theta
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} text_embed_dim
:canonical: fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.text_embed_dim
:type: int
:value: >
   4096

```{autodoc2-docstring} fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoArchConfig.text_embed_dim
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} HunyuanVideoConfig
:canonical: fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoConfig

Bases: {py:obj}`fastvideo.configs.models.dits.base.DiTConfig`

```{autodoc2-docstring} fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} arch_config
:canonical: fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoConfig.arch_config
:type: fastvideo.configs.models.dits.base.DiTArchConfig
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoConfig.arch_config
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} prefix
:canonical: fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoConfig.prefix
:type: str
:value: >
   'Hunyuan'

```{autodoc2-docstring} fastvideo.configs.models.dits.hunyuanvideo.HunyuanVideoConfig.prefix
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:function} is_double_block(n: str, m) -> bool
:canonical: fastvideo.configs.models.dits.hunyuanvideo.is_double_block

```{autodoc2-docstring} fastvideo.configs.models.dits.hunyuanvideo.is_double_block
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} is_refiner_block(n: str, m) -> bool
:canonical: fastvideo.configs.models.dits.hunyuanvideo.is_refiner_block

```{autodoc2-docstring} fastvideo.configs.models.dits.hunyuanvideo.is_refiner_block
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} is_single_block(n: str, m) -> bool
:canonical: fastvideo.configs.models.dits.hunyuanvideo.is_single_block

```{autodoc2-docstring} fastvideo.configs.models.dits.hunyuanvideo.is_single_block
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} is_txt_in(n: str, m) -> bool
:canonical: fastvideo.configs.models.dits.hunyuanvideo.is_txt_in

```{autodoc2-docstring} fastvideo.configs.models.dits.hunyuanvideo.is_txt_in
:parser: docs.source.autodoc2_docstring_parser
```
````
