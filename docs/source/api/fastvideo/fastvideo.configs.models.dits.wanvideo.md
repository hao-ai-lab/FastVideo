# {py:mod}`fastvideo.configs.models.dits.wanvideo`

```{py:module} fastvideo.configs.models.dits.wanvideo
```

```{autodoc2-docstring} fastvideo.configs.models.dits.wanvideo
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`WanVideoArchConfig <fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`WanVideoConfig <fastvideo.configs.models.dits.wanvideo.WanVideoConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.models.dits.wanvideo.WanVideoConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`is_blocks <fastvideo.configs.models.dits.wanvideo.is_blocks>`
  - ```{autodoc2-docstring} fastvideo.configs.models.dits.wanvideo.is_blocks
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} WanVideoArchConfig
:canonical: fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig

Bases: {py:obj}`fastvideo.configs.models.dits.base.DiTArchConfig`

```{autodoc2-docstring} fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} added_kv_proj_dim
:canonical: fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.added_kv_proj_dim
:type: int | None
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.added_kv_proj_dim
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} attention_head_dim
:canonical: fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.attention_head_dim
:type: int
:value: >
   128

```{autodoc2-docstring} fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.attention_head_dim
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} cross_attn_norm
:canonical: fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.cross_attn_norm
:type: bool
:value: >
   True

```{autodoc2-docstring} fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.cross_attn_norm
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} eps
:canonical: fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.eps
:type: float
:value: >
   1e-06

```{autodoc2-docstring} fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.eps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} exclude_lora_layers
:canonical: fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.exclude_lora_layers
:type: list[str]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.exclude_lora_layers
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} ffn_dim
:canonical: fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.ffn_dim
:type: int
:value: >
   13824

```{autodoc2-docstring} fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.ffn_dim
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} freq_dim
:canonical: fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.freq_dim
:type: int
:value: >
   256

```{autodoc2-docstring} fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.freq_dim
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} image_dim
:canonical: fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.image_dim
:type: int | None
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.image_dim
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} in_channels
:canonical: fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.in_channels
:type: int
:value: >
   16

```{autodoc2-docstring} fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.in_channels
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} lora_param_names_mapping
:canonical: fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.lora_param_names_mapping
:type: dict
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.lora_param_names_mapping
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_attention_heads
:canonical: fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.num_attention_heads
:type: int
:value: >
   40

```{autodoc2-docstring} fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.num_attention_heads
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_layers
:canonical: fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.num_layers
:type: int
:value: >
   40

```{autodoc2-docstring} fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.num_layers
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} out_channels
:canonical: fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.out_channels
:type: int
:value: >
   16

```{autodoc2-docstring} fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.out_channels
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} param_names_mapping
:canonical: fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.param_names_mapping
:type: dict
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.param_names_mapping
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} patch_size
:canonical: fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.patch_size
:type: tuple[int, int, int]
:value: >
   (1, 2, 2)

```{autodoc2-docstring} fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.patch_size
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} pos_embed_seq_len
:canonical: fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.pos_embed_seq_len
:type: int | None
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.pos_embed_seq_len
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} qk_norm
:canonical: fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.qk_norm
:type: str
:value: >
   'rms_norm_across_heads'

```{autodoc2-docstring} fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.qk_norm
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} reverse_param_names_mapping
:canonical: fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.reverse_param_names_mapping
:type: dict
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.reverse_param_names_mapping
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} rope_max_seq_len
:canonical: fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.rope_max_seq_len
:type: int
:value: >
   1024

```{autodoc2-docstring} fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.rope_max_seq_len
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} text_dim
:canonical: fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.text_dim
:type: int
:value: >
   4096

```{autodoc2-docstring} fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.text_dim
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} text_len
:canonical: fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.text_len
:value: >
   512

```{autodoc2-docstring} fastvideo.configs.models.dits.wanvideo.WanVideoArchConfig.text_len
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} WanVideoConfig
:canonical: fastvideo.configs.models.dits.wanvideo.WanVideoConfig

Bases: {py:obj}`fastvideo.configs.models.dits.base.DiTConfig`

```{autodoc2-docstring} fastvideo.configs.models.dits.wanvideo.WanVideoConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} arch_config
:canonical: fastvideo.configs.models.dits.wanvideo.WanVideoConfig.arch_config
:type: fastvideo.configs.models.dits.base.DiTArchConfig
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.dits.wanvideo.WanVideoConfig.arch_config
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} prefix
:canonical: fastvideo.configs.models.dits.wanvideo.WanVideoConfig.prefix
:type: str
:value: >
   'Wan'

```{autodoc2-docstring} fastvideo.configs.models.dits.wanvideo.WanVideoConfig.prefix
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:function} is_blocks(n: str, m) -> bool
:canonical: fastvideo.configs.models.dits.wanvideo.is_blocks

```{autodoc2-docstring} fastvideo.configs.models.dits.wanvideo.is_blocks
:parser: docs.source.autodoc2_docstring_parser
```
````
