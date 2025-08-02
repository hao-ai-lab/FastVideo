# {py:mod}`fastvideo.configs.models.encoders.llama`

```{py:module} fastvideo.configs.models.encoders.llama
```

```{autodoc2-docstring} fastvideo.configs.models.encoders.llama
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LlamaArchConfig <fastvideo.configs.models.encoders.llama.LlamaArchConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.models.encoders.llama.LlamaArchConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`LlamaConfig <fastvideo.configs.models.encoders.llama.LlamaConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.models.encoders.llama.LlamaConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} LlamaArchConfig
:canonical: fastvideo.configs.models.encoders.llama.LlamaArchConfig

Bases: {py:obj}`fastvideo.configs.models.encoders.base.TextEncoderArchConfig`

```{autodoc2-docstring} fastvideo.configs.models.encoders.llama.LlamaArchConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} attention_bias
:canonical: fastvideo.configs.models.encoders.llama.LlamaArchConfig.attention_bias
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.configs.models.encoders.llama.LlamaArchConfig.attention_bias
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} attention_dropout
:canonical: fastvideo.configs.models.encoders.llama.LlamaArchConfig.attention_dropout
:type: float
:value: >
   0.0

```{autodoc2-docstring} fastvideo.configs.models.encoders.llama.LlamaArchConfig.attention_dropout
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} bos_token_id
:canonical: fastvideo.configs.models.encoders.llama.LlamaArchConfig.bos_token_id
:type: int
:value: >
   1

```{autodoc2-docstring} fastvideo.configs.models.encoders.llama.LlamaArchConfig.bos_token_id
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} eos_token_id
:canonical: fastvideo.configs.models.encoders.llama.LlamaArchConfig.eos_token_id
:type: int
:value: >
   2

```{autodoc2-docstring} fastvideo.configs.models.encoders.llama.LlamaArchConfig.eos_token_id
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} head_dim
:canonical: fastvideo.configs.models.encoders.llama.LlamaArchConfig.head_dim
:type: int | None
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.models.encoders.llama.LlamaArchConfig.head_dim
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} hidden_act
:canonical: fastvideo.configs.models.encoders.llama.LlamaArchConfig.hidden_act
:type: str
:value: >
   'silu'

```{autodoc2-docstring} fastvideo.configs.models.encoders.llama.LlamaArchConfig.hidden_act
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} hidden_size
:canonical: fastvideo.configs.models.encoders.llama.LlamaArchConfig.hidden_size
:type: int
:value: >
   4096

```{autodoc2-docstring} fastvideo.configs.models.encoders.llama.LlamaArchConfig.hidden_size
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} hidden_state_skip_layer
:canonical: fastvideo.configs.models.encoders.llama.LlamaArchConfig.hidden_state_skip_layer
:type: int
:value: >
   2

```{autodoc2-docstring} fastvideo.configs.models.encoders.llama.LlamaArchConfig.hidden_state_skip_layer
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} initializer_range
:canonical: fastvideo.configs.models.encoders.llama.LlamaArchConfig.initializer_range
:type: float
:value: >
   0.02

```{autodoc2-docstring} fastvideo.configs.models.encoders.llama.LlamaArchConfig.initializer_range
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} intermediate_size
:canonical: fastvideo.configs.models.encoders.llama.LlamaArchConfig.intermediate_size
:type: int
:value: >
   11008

```{autodoc2-docstring} fastvideo.configs.models.encoders.llama.LlamaArchConfig.intermediate_size
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} max_position_embeddings
:canonical: fastvideo.configs.models.encoders.llama.LlamaArchConfig.max_position_embeddings
:type: int
:value: >
   2048

```{autodoc2-docstring} fastvideo.configs.models.encoders.llama.LlamaArchConfig.max_position_embeddings
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} mlp_bias
:canonical: fastvideo.configs.models.encoders.llama.LlamaArchConfig.mlp_bias
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.configs.models.encoders.llama.LlamaArchConfig.mlp_bias
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_attention_heads
:canonical: fastvideo.configs.models.encoders.llama.LlamaArchConfig.num_attention_heads
:type: int
:value: >
   32

```{autodoc2-docstring} fastvideo.configs.models.encoders.llama.LlamaArchConfig.num_attention_heads
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_hidden_layers
:canonical: fastvideo.configs.models.encoders.llama.LlamaArchConfig.num_hidden_layers
:type: int
:value: >
   32

```{autodoc2-docstring} fastvideo.configs.models.encoders.llama.LlamaArchConfig.num_hidden_layers
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_key_value_heads
:canonical: fastvideo.configs.models.encoders.llama.LlamaArchConfig.num_key_value_heads
:type: int | None
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.models.encoders.llama.LlamaArchConfig.num_key_value_heads
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} pad_token_id
:canonical: fastvideo.configs.models.encoders.llama.LlamaArchConfig.pad_token_id
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.configs.models.encoders.llama.LlamaArchConfig.pad_token_id
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} pretraining_tp
:canonical: fastvideo.configs.models.encoders.llama.LlamaArchConfig.pretraining_tp
:type: int
:value: >
   1

```{autodoc2-docstring} fastvideo.configs.models.encoders.llama.LlamaArchConfig.pretraining_tp
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} rms_norm_eps
:canonical: fastvideo.configs.models.encoders.llama.LlamaArchConfig.rms_norm_eps
:type: float
:value: >
   1e-06

```{autodoc2-docstring} fastvideo.configs.models.encoders.llama.LlamaArchConfig.rms_norm_eps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} rope_scaling
:canonical: fastvideo.configs.models.encoders.llama.LlamaArchConfig.rope_scaling
:type: float | None
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.models.encoders.llama.LlamaArchConfig.rope_scaling
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} rope_theta
:canonical: fastvideo.configs.models.encoders.llama.LlamaArchConfig.rope_theta
:type: float
:value: >
   10000.0

```{autodoc2-docstring} fastvideo.configs.models.encoders.llama.LlamaArchConfig.rope_theta
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} stacked_params_mapping
:canonical: fastvideo.configs.models.encoders.llama.LlamaArchConfig.stacked_params_mapping
:type: list[tuple[str, str, str]]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.encoders.llama.LlamaArchConfig.stacked_params_mapping
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} text_len
:canonical: fastvideo.configs.models.encoders.llama.LlamaArchConfig.text_len
:type: int
:value: >
   256

```{autodoc2-docstring} fastvideo.configs.models.encoders.llama.LlamaArchConfig.text_len
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} tie_word_embeddings
:canonical: fastvideo.configs.models.encoders.llama.LlamaArchConfig.tie_word_embeddings
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.configs.models.encoders.llama.LlamaArchConfig.tie_word_embeddings
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} use_cache
:canonical: fastvideo.configs.models.encoders.llama.LlamaArchConfig.use_cache
:type: bool
:value: >
   True

```{autodoc2-docstring} fastvideo.configs.models.encoders.llama.LlamaArchConfig.use_cache
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} vocab_size
:canonical: fastvideo.configs.models.encoders.llama.LlamaArchConfig.vocab_size
:type: int
:value: >
   32000

```{autodoc2-docstring} fastvideo.configs.models.encoders.llama.LlamaArchConfig.vocab_size
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} LlamaConfig
:canonical: fastvideo.configs.models.encoders.llama.LlamaConfig

Bases: {py:obj}`fastvideo.configs.models.encoders.base.TextEncoderConfig`

```{autodoc2-docstring} fastvideo.configs.models.encoders.llama.LlamaConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} arch_config
:canonical: fastvideo.configs.models.encoders.llama.LlamaConfig.arch_config
:type: fastvideo.configs.models.encoders.base.TextEncoderArchConfig
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.encoders.llama.LlamaConfig.arch_config
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} prefix
:canonical: fastvideo.configs.models.encoders.llama.LlamaConfig.prefix
:type: str
:value: >
   'llama'

```{autodoc2-docstring} fastvideo.configs.models.encoders.llama.LlamaConfig.prefix
:parser: docs.source.autodoc2_docstring_parser
```

````

`````
