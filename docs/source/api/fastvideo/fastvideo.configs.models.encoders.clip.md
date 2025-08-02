# {py:mod}`fastvideo.configs.models.encoders.clip`

```{py:module} fastvideo.configs.models.encoders.clip
```

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CLIPTextArchConfig <fastvideo.configs.models.encoders.clip.CLIPTextArchConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPTextArchConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`CLIPTextConfig <fastvideo.configs.models.encoders.clip.CLIPTextConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPTextConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`CLIPVisionArchConfig <fastvideo.configs.models.encoders.clip.CLIPVisionArchConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPVisionArchConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`CLIPVisionConfig <fastvideo.configs.models.encoders.clip.CLIPVisionConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPVisionConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} CLIPTextArchConfig
:canonical: fastvideo.configs.models.encoders.clip.CLIPTextArchConfig

Bases: {py:obj}`fastvideo.configs.models.encoders.base.TextEncoderArchConfig`

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPTextArchConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} attention_dropout
:canonical: fastvideo.configs.models.encoders.clip.CLIPTextArchConfig.attention_dropout
:type: float
:value: >
   0.0

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPTextArchConfig.attention_dropout
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} bos_token_id
:canonical: fastvideo.configs.models.encoders.clip.CLIPTextArchConfig.bos_token_id
:type: int
:value: >
   49406

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPTextArchConfig.bos_token_id
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} dropout
:canonical: fastvideo.configs.models.encoders.clip.CLIPTextArchConfig.dropout
:type: float
:value: >
   0.0

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPTextArchConfig.dropout
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} eos_token_id
:canonical: fastvideo.configs.models.encoders.clip.CLIPTextArchConfig.eos_token_id
:type: int
:value: >
   49407

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPTextArchConfig.eos_token_id
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} hidden_act
:canonical: fastvideo.configs.models.encoders.clip.CLIPTextArchConfig.hidden_act
:type: str
:value: >
   'quick_gelu'

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPTextArchConfig.hidden_act
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} hidden_size
:canonical: fastvideo.configs.models.encoders.clip.CLIPTextArchConfig.hidden_size
:type: int
:value: >
   512

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPTextArchConfig.hidden_size
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} initializer_factor
:canonical: fastvideo.configs.models.encoders.clip.CLIPTextArchConfig.initializer_factor
:type: float
:value: >
   1.0

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPTextArchConfig.initializer_factor
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} initializer_range
:canonical: fastvideo.configs.models.encoders.clip.CLIPTextArchConfig.initializer_range
:type: float
:value: >
   0.02

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPTextArchConfig.initializer_range
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} intermediate_size
:canonical: fastvideo.configs.models.encoders.clip.CLIPTextArchConfig.intermediate_size
:type: int
:value: >
   2048

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPTextArchConfig.intermediate_size
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} layer_norm_eps
:canonical: fastvideo.configs.models.encoders.clip.CLIPTextArchConfig.layer_norm_eps
:type: float
:value: >
   1e-05

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPTextArchConfig.layer_norm_eps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} max_position_embeddings
:canonical: fastvideo.configs.models.encoders.clip.CLIPTextArchConfig.max_position_embeddings
:type: int
:value: >
   77

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPTextArchConfig.max_position_embeddings
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_attention_heads
:canonical: fastvideo.configs.models.encoders.clip.CLIPTextArchConfig.num_attention_heads
:type: int
:value: >
   8

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPTextArchConfig.num_attention_heads
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_hidden_layers
:canonical: fastvideo.configs.models.encoders.clip.CLIPTextArchConfig.num_hidden_layers
:type: int
:value: >
   12

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPTextArchConfig.num_hidden_layers
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} pad_token_id
:canonical: fastvideo.configs.models.encoders.clip.CLIPTextArchConfig.pad_token_id
:type: int
:value: >
   1

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPTextArchConfig.pad_token_id
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} projection_dim
:canonical: fastvideo.configs.models.encoders.clip.CLIPTextArchConfig.projection_dim
:type: int
:value: >
   512

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPTextArchConfig.projection_dim
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} stacked_params_mapping
:canonical: fastvideo.configs.models.encoders.clip.CLIPTextArchConfig.stacked_params_mapping
:type: list[tuple[str, str, str]]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPTextArchConfig.stacked_params_mapping
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} text_len
:canonical: fastvideo.configs.models.encoders.clip.CLIPTextArchConfig.text_len
:type: int
:value: >
   77

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPTextArchConfig.text_len
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} vocab_size
:canonical: fastvideo.configs.models.encoders.clip.CLIPTextArchConfig.vocab_size
:type: int
:value: >
   49408

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPTextArchConfig.vocab_size
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} CLIPTextConfig
:canonical: fastvideo.configs.models.encoders.clip.CLIPTextConfig

Bases: {py:obj}`fastvideo.configs.models.encoders.base.TextEncoderConfig`

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPTextConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} arch_config
:canonical: fastvideo.configs.models.encoders.clip.CLIPTextConfig.arch_config
:type: fastvideo.configs.models.encoders.base.TextEncoderArchConfig
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPTextConfig.arch_config
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_hidden_layers_override
:canonical: fastvideo.configs.models.encoders.clip.CLIPTextConfig.num_hidden_layers_override
:type: int | None
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPTextConfig.num_hidden_layers_override
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} prefix
:canonical: fastvideo.configs.models.encoders.clip.CLIPTextConfig.prefix
:type: str
:value: >
   'clip'

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPTextConfig.prefix
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} require_post_norm
:canonical: fastvideo.configs.models.encoders.clip.CLIPTextConfig.require_post_norm
:type: bool | None
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPTextConfig.require_post_norm
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} CLIPVisionArchConfig
:canonical: fastvideo.configs.models.encoders.clip.CLIPVisionArchConfig

Bases: {py:obj}`fastvideo.configs.models.encoders.base.ImageEncoderArchConfig`

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPVisionArchConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} attention_dropout
:canonical: fastvideo.configs.models.encoders.clip.CLIPVisionArchConfig.attention_dropout
:type: float
:value: >
   0.0

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPVisionArchConfig.attention_dropout
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} dropout
:canonical: fastvideo.configs.models.encoders.clip.CLIPVisionArchConfig.dropout
:type: float
:value: >
   0.0

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPVisionArchConfig.dropout
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} hidden_act
:canonical: fastvideo.configs.models.encoders.clip.CLIPVisionArchConfig.hidden_act
:type: str
:value: >
   'quick_gelu'

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPVisionArchConfig.hidden_act
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} hidden_size
:canonical: fastvideo.configs.models.encoders.clip.CLIPVisionArchConfig.hidden_size
:type: int
:value: >
   768

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPVisionArchConfig.hidden_size
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} image_size
:canonical: fastvideo.configs.models.encoders.clip.CLIPVisionArchConfig.image_size
:type: int
:value: >
   224

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPVisionArchConfig.image_size
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} initializer_factor
:canonical: fastvideo.configs.models.encoders.clip.CLIPVisionArchConfig.initializer_factor
:type: float
:value: >
   1.0

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPVisionArchConfig.initializer_factor
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} initializer_range
:canonical: fastvideo.configs.models.encoders.clip.CLIPVisionArchConfig.initializer_range
:type: float
:value: >
   0.02

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPVisionArchConfig.initializer_range
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} intermediate_size
:canonical: fastvideo.configs.models.encoders.clip.CLIPVisionArchConfig.intermediate_size
:type: int
:value: >
   3072

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPVisionArchConfig.intermediate_size
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} layer_norm_eps
:canonical: fastvideo.configs.models.encoders.clip.CLIPVisionArchConfig.layer_norm_eps
:type: float
:value: >
   1e-05

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPVisionArchConfig.layer_norm_eps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_attention_heads
:canonical: fastvideo.configs.models.encoders.clip.CLIPVisionArchConfig.num_attention_heads
:type: int
:value: >
   12

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPVisionArchConfig.num_attention_heads
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_channels
:canonical: fastvideo.configs.models.encoders.clip.CLIPVisionArchConfig.num_channels
:type: int
:value: >
   3

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPVisionArchConfig.num_channels
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_hidden_layers
:canonical: fastvideo.configs.models.encoders.clip.CLIPVisionArchConfig.num_hidden_layers
:type: int
:value: >
   12

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPVisionArchConfig.num_hidden_layers
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} patch_size
:canonical: fastvideo.configs.models.encoders.clip.CLIPVisionArchConfig.patch_size
:type: int
:value: >
   32

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPVisionArchConfig.patch_size
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} projection_dim
:canonical: fastvideo.configs.models.encoders.clip.CLIPVisionArchConfig.projection_dim
:type: int
:value: >
   512

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPVisionArchConfig.projection_dim
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} stacked_params_mapping
:canonical: fastvideo.configs.models.encoders.clip.CLIPVisionArchConfig.stacked_params_mapping
:type: list[tuple[str, str, str]]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPVisionArchConfig.stacked_params_mapping
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} CLIPVisionConfig
:canonical: fastvideo.configs.models.encoders.clip.CLIPVisionConfig

Bases: {py:obj}`fastvideo.configs.models.encoders.base.ImageEncoderConfig`

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPVisionConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} arch_config
:canonical: fastvideo.configs.models.encoders.clip.CLIPVisionConfig.arch_config
:type: fastvideo.configs.models.encoders.base.ImageEncoderArchConfig
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPVisionConfig.arch_config
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_hidden_layers_override
:canonical: fastvideo.configs.models.encoders.clip.CLIPVisionConfig.num_hidden_layers_override
:type: int | None
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPVisionConfig.num_hidden_layers_override
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} prefix
:canonical: fastvideo.configs.models.encoders.clip.CLIPVisionConfig.prefix
:type: str
:value: >
   'clip'

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPVisionConfig.prefix
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} require_post_norm
:canonical: fastvideo.configs.models.encoders.clip.CLIPVisionConfig.require_post_norm
:type: bool | None
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.models.encoders.clip.CLIPVisionConfig.require_post_norm
:parser: docs.source.autodoc2_docstring_parser
```

````

`````
