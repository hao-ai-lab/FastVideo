# {py:mod}`fastvideo.configs.models.encoders.t5`

```{py:module} fastvideo.configs.models.encoders.t5
```

```{autodoc2-docstring} fastvideo.configs.models.encoders.t5
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`T5ArchConfig <fastvideo.configs.models.encoders.t5.T5ArchConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.models.encoders.t5.T5ArchConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`T5Config <fastvideo.configs.models.encoders.t5.T5Config>`
  - ```{autodoc2-docstring} fastvideo.configs.models.encoders.t5.T5Config
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} T5ArchConfig
:canonical: fastvideo.configs.models.encoders.t5.T5ArchConfig

Bases: {py:obj}`fastvideo.configs.models.encoders.base.TextEncoderArchConfig`

```{autodoc2-docstring} fastvideo.configs.models.encoders.t5.T5ArchConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} classifier_dropout
:canonical: fastvideo.configs.models.encoders.t5.T5ArchConfig.classifier_dropout
:type: float
:value: >
   0.0

```{autodoc2-docstring} fastvideo.configs.models.encoders.t5.T5ArchConfig.classifier_dropout
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} d_ff
:canonical: fastvideo.configs.models.encoders.t5.T5ArchConfig.d_ff
:type: int
:value: >
   2048

```{autodoc2-docstring} fastvideo.configs.models.encoders.t5.T5ArchConfig.d_ff
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} d_kv
:canonical: fastvideo.configs.models.encoders.t5.T5ArchConfig.d_kv
:type: int
:value: >
   64

```{autodoc2-docstring} fastvideo.configs.models.encoders.t5.T5ArchConfig.d_kv
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} d_model
:canonical: fastvideo.configs.models.encoders.t5.T5ArchConfig.d_model
:type: int
:value: >
   512

```{autodoc2-docstring} fastvideo.configs.models.encoders.t5.T5ArchConfig.d_model
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} dense_act_fn
:canonical: fastvideo.configs.models.encoders.t5.T5ArchConfig.dense_act_fn
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} fastvideo.configs.models.encoders.t5.T5ArchConfig.dense_act_fn
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} dropout_rate
:canonical: fastvideo.configs.models.encoders.t5.T5ArchConfig.dropout_rate
:type: float
:value: >
   0.1

```{autodoc2-docstring} fastvideo.configs.models.encoders.t5.T5ArchConfig.dropout_rate
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} eos_token_id
:canonical: fastvideo.configs.models.encoders.t5.T5ArchConfig.eos_token_id
:type: int
:value: >
   1

```{autodoc2-docstring} fastvideo.configs.models.encoders.t5.T5ArchConfig.eos_token_id
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} feed_forward_proj
:canonical: fastvideo.configs.models.encoders.t5.T5ArchConfig.feed_forward_proj
:type: str
:value: >
   'relu'

```{autodoc2-docstring} fastvideo.configs.models.encoders.t5.T5ArchConfig.feed_forward_proj
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} initializer_factor
:canonical: fastvideo.configs.models.encoders.t5.T5ArchConfig.initializer_factor
:type: float
:value: >
   1.0

```{autodoc2-docstring} fastvideo.configs.models.encoders.t5.T5ArchConfig.initializer_factor
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} is_encoder_decoder
:canonical: fastvideo.configs.models.encoders.t5.T5ArchConfig.is_encoder_decoder
:type: bool
:value: >
   True

```{autodoc2-docstring} fastvideo.configs.models.encoders.t5.T5ArchConfig.is_encoder_decoder
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} is_gated_act
:canonical: fastvideo.configs.models.encoders.t5.T5ArchConfig.is_gated_act
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.configs.models.encoders.t5.T5ArchConfig.is_gated_act
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} layer_norm_epsilon
:canonical: fastvideo.configs.models.encoders.t5.T5ArchConfig.layer_norm_epsilon
:type: float
:value: >
   1e-06

```{autodoc2-docstring} fastvideo.configs.models.encoders.t5.T5ArchConfig.layer_norm_epsilon
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_decoder_layers
:canonical: fastvideo.configs.models.encoders.t5.T5ArchConfig.num_decoder_layers
:type: int | None
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.models.encoders.t5.T5ArchConfig.num_decoder_layers
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_heads
:canonical: fastvideo.configs.models.encoders.t5.T5ArchConfig.num_heads
:type: int
:value: >
   8

```{autodoc2-docstring} fastvideo.configs.models.encoders.t5.T5ArchConfig.num_heads
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_layers
:canonical: fastvideo.configs.models.encoders.t5.T5ArchConfig.num_layers
:type: int
:value: >
   6

```{autodoc2-docstring} fastvideo.configs.models.encoders.t5.T5ArchConfig.num_layers
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} pad_token_id
:canonical: fastvideo.configs.models.encoders.t5.T5ArchConfig.pad_token_id
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.configs.models.encoders.t5.T5ArchConfig.pad_token_id
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} relative_attention_max_distance
:canonical: fastvideo.configs.models.encoders.t5.T5ArchConfig.relative_attention_max_distance
:type: int
:value: >
   128

```{autodoc2-docstring} fastvideo.configs.models.encoders.t5.T5ArchConfig.relative_attention_max_distance
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} relative_attention_num_buckets
:canonical: fastvideo.configs.models.encoders.t5.T5ArchConfig.relative_attention_num_buckets
:type: int
:value: >
   32

```{autodoc2-docstring} fastvideo.configs.models.encoders.t5.T5ArchConfig.relative_attention_num_buckets
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} stacked_params_mapping
:canonical: fastvideo.configs.models.encoders.t5.T5ArchConfig.stacked_params_mapping
:type: list[tuple[str, str, str]]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.encoders.t5.T5ArchConfig.stacked_params_mapping
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} text_len
:canonical: fastvideo.configs.models.encoders.t5.T5ArchConfig.text_len
:type: int
:value: >
   512

```{autodoc2-docstring} fastvideo.configs.models.encoders.t5.T5ArchConfig.text_len
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} use_cache
:canonical: fastvideo.configs.models.encoders.t5.T5ArchConfig.use_cache
:type: bool
:value: >
   True

```{autodoc2-docstring} fastvideo.configs.models.encoders.t5.T5ArchConfig.use_cache
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} vocab_size
:canonical: fastvideo.configs.models.encoders.t5.T5ArchConfig.vocab_size
:type: int
:value: >
   32128

```{autodoc2-docstring} fastvideo.configs.models.encoders.t5.T5ArchConfig.vocab_size
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} T5Config
:canonical: fastvideo.configs.models.encoders.t5.T5Config

Bases: {py:obj}`fastvideo.configs.models.encoders.base.TextEncoderConfig`

```{autodoc2-docstring} fastvideo.configs.models.encoders.t5.T5Config
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} arch_config
:canonical: fastvideo.configs.models.encoders.t5.T5Config.arch_config
:type: fastvideo.configs.models.encoders.base.TextEncoderArchConfig
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.encoders.t5.T5Config.arch_config
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} prefix
:canonical: fastvideo.configs.models.encoders.t5.T5Config.prefix
:type: str
:value: >
   't5'

```{autodoc2-docstring} fastvideo.configs.models.encoders.t5.T5Config.prefix
:parser: docs.source.autodoc2_docstring_parser
```

````

`````
