# {py:mod}`fastvideo.configs.models.encoders.base`

```{py:module} fastvideo.configs.models.encoders.base
```

```{autodoc2-docstring} fastvideo.configs.models.encoders.base
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BaseEncoderOutput <fastvideo.configs.models.encoders.base.BaseEncoderOutput>`
  - ```{autodoc2-docstring} fastvideo.configs.models.encoders.base.BaseEncoderOutput
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`EncoderArchConfig <fastvideo.configs.models.encoders.base.EncoderArchConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.models.encoders.base.EncoderArchConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`EncoderConfig <fastvideo.configs.models.encoders.base.EncoderConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.models.encoders.base.EncoderConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`ImageEncoderArchConfig <fastvideo.configs.models.encoders.base.ImageEncoderArchConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.models.encoders.base.ImageEncoderArchConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`ImageEncoderConfig <fastvideo.configs.models.encoders.base.ImageEncoderConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.models.encoders.base.ImageEncoderConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`TextEncoderArchConfig <fastvideo.configs.models.encoders.base.TextEncoderArchConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.models.encoders.base.TextEncoderArchConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`TextEncoderConfig <fastvideo.configs.models.encoders.base.TextEncoderConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.models.encoders.base.TextEncoderConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} BaseEncoderOutput
:canonical: fastvideo.configs.models.encoders.base.BaseEncoderOutput

```{autodoc2-docstring} fastvideo.configs.models.encoders.base.BaseEncoderOutput
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} attention_mask
:canonical: fastvideo.configs.models.encoders.base.BaseEncoderOutput.attention_mask
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.models.encoders.base.BaseEncoderOutput.attention_mask
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} attentions
:canonical: fastvideo.configs.models.encoders.base.BaseEncoderOutput.attentions
:type: tuple[torch.FloatTensor, ...] | None
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.models.encoders.base.BaseEncoderOutput.attentions
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} hidden_states
:canonical: fastvideo.configs.models.encoders.base.BaseEncoderOutput.hidden_states
:type: tuple[torch.FloatTensor, ...] | None
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.models.encoders.base.BaseEncoderOutput.hidden_states
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} last_hidden_state
:canonical: fastvideo.configs.models.encoders.base.BaseEncoderOutput.last_hidden_state
:type: torch.FloatTensor | None
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.models.encoders.base.BaseEncoderOutput.last_hidden_state
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} pooler_output
:canonical: fastvideo.configs.models.encoders.base.BaseEncoderOutput.pooler_output
:type: torch.FloatTensor | None
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.models.encoders.base.BaseEncoderOutput.pooler_output
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} EncoderArchConfig
:canonical: fastvideo.configs.models.encoders.base.EncoderArchConfig

Bases: {py:obj}`fastvideo.configs.models.base.ArchConfig`

```{autodoc2-docstring} fastvideo.configs.models.encoders.base.EncoderArchConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} architectures
:canonical: fastvideo.configs.models.encoders.base.EncoderArchConfig.architectures
:type: list[str]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.encoders.base.EncoderArchConfig.architectures
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} output_hidden_states
:canonical: fastvideo.configs.models.encoders.base.EncoderArchConfig.output_hidden_states
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.configs.models.encoders.base.EncoderArchConfig.output_hidden_states
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} use_return_dict
:canonical: fastvideo.configs.models.encoders.base.EncoderArchConfig.use_return_dict
:type: bool
:value: >
   True

```{autodoc2-docstring} fastvideo.configs.models.encoders.base.EncoderArchConfig.use_return_dict
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} EncoderConfig
:canonical: fastvideo.configs.models.encoders.base.EncoderConfig

Bases: {py:obj}`fastvideo.configs.models.base.ModelConfig`

```{autodoc2-docstring} fastvideo.configs.models.encoders.base.EncoderConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} arch_config
:canonical: fastvideo.configs.models.encoders.base.EncoderConfig.arch_config
:type: fastvideo.configs.models.base.ArchConfig
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.encoders.base.EncoderConfig.arch_config
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} lora_config
:canonical: fastvideo.configs.models.encoders.base.EncoderConfig.lora_config
:type: typing.Any | None
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.models.encoders.base.EncoderConfig.lora_config
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} prefix
:canonical: fastvideo.configs.models.encoders.base.EncoderConfig.prefix
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} fastvideo.configs.models.encoders.base.EncoderConfig.prefix
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} quant_config
:canonical: fastvideo.configs.models.encoders.base.EncoderConfig.quant_config
:type: fastvideo.layers.quantization.QuantizationConfig | None
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.models.encoders.base.EncoderConfig.quant_config
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:class} ImageEncoderArchConfig
:canonical: fastvideo.configs.models.encoders.base.ImageEncoderArchConfig

Bases: {py:obj}`fastvideo.configs.models.encoders.base.EncoderArchConfig`

```{autodoc2-docstring} fastvideo.configs.models.encoders.base.ImageEncoderArchConfig
:parser: docs.source.autodoc2_docstring_parser
```

````

`````{py:class} ImageEncoderConfig
:canonical: fastvideo.configs.models.encoders.base.ImageEncoderConfig

Bases: {py:obj}`fastvideo.configs.models.encoders.base.EncoderConfig`

```{autodoc2-docstring} fastvideo.configs.models.encoders.base.ImageEncoderConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} arch_config
:canonical: fastvideo.configs.models.encoders.base.ImageEncoderConfig.arch_config
:type: fastvideo.configs.models.base.ArchConfig
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.encoders.base.ImageEncoderConfig.arch_config
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} TextEncoderArchConfig
:canonical: fastvideo.configs.models.encoders.base.TextEncoderArchConfig

Bases: {py:obj}`fastvideo.configs.models.encoders.base.EncoderArchConfig`

```{autodoc2-docstring} fastvideo.configs.models.encoders.base.TextEncoderArchConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} decoder_start_token_id
:canonical: fastvideo.configs.models.encoders.base.TextEncoderArchConfig.decoder_start_token_id
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.configs.models.encoders.base.TextEncoderArchConfig.decoder_start_token_id
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} eos_token_id
:canonical: fastvideo.configs.models.encoders.base.TextEncoderArchConfig.eos_token_id
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.configs.models.encoders.base.TextEncoderArchConfig.eos_token_id
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} hidden_size
:canonical: fastvideo.configs.models.encoders.base.TextEncoderArchConfig.hidden_size
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.configs.models.encoders.base.TextEncoderArchConfig.hidden_size
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} hidden_state_skip_layer
:canonical: fastvideo.configs.models.encoders.base.TextEncoderArchConfig.hidden_state_skip_layer
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.configs.models.encoders.base.TextEncoderArchConfig.hidden_state_skip_layer
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_attention_heads
:canonical: fastvideo.configs.models.encoders.base.TextEncoderArchConfig.num_attention_heads
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.configs.models.encoders.base.TextEncoderArchConfig.num_attention_heads
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_hidden_layers
:canonical: fastvideo.configs.models.encoders.base.TextEncoderArchConfig.num_hidden_layers
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.configs.models.encoders.base.TextEncoderArchConfig.num_hidden_layers
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} output_past
:canonical: fastvideo.configs.models.encoders.base.TextEncoderArchConfig.output_past
:type: bool
:value: >
   True

```{autodoc2-docstring} fastvideo.configs.models.encoders.base.TextEncoderArchConfig.output_past
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} pad_token_id
:canonical: fastvideo.configs.models.encoders.base.TextEncoderArchConfig.pad_token_id
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.configs.models.encoders.base.TextEncoderArchConfig.pad_token_id
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} scalable_attention
:canonical: fastvideo.configs.models.encoders.base.TextEncoderArchConfig.scalable_attention
:type: bool
:value: >
   True

```{autodoc2-docstring} fastvideo.configs.models.encoders.base.TextEncoderArchConfig.scalable_attention
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} stacked_params_mapping
:canonical: fastvideo.configs.models.encoders.base.TextEncoderArchConfig.stacked_params_mapping
:type: list[tuple[str, str, str]]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.encoders.base.TextEncoderArchConfig.stacked_params_mapping
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} text_len
:canonical: fastvideo.configs.models.encoders.base.TextEncoderArchConfig.text_len
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.configs.models.encoders.base.TextEncoderArchConfig.text_len
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} tie_word_embeddings
:canonical: fastvideo.configs.models.encoders.base.TextEncoderArchConfig.tie_word_embeddings
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.configs.models.encoders.base.TextEncoderArchConfig.tie_word_embeddings
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} tokenizer_kwargs
:canonical: fastvideo.configs.models.encoders.base.TextEncoderArchConfig.tokenizer_kwargs
:type: dict[str, typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.encoders.base.TextEncoderArchConfig.tokenizer_kwargs
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} vocab_size
:canonical: fastvideo.configs.models.encoders.base.TextEncoderArchConfig.vocab_size
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.configs.models.encoders.base.TextEncoderArchConfig.vocab_size
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} TextEncoderConfig
:canonical: fastvideo.configs.models.encoders.base.TextEncoderConfig

Bases: {py:obj}`fastvideo.configs.models.encoders.base.EncoderConfig`

```{autodoc2-docstring} fastvideo.configs.models.encoders.base.TextEncoderConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} arch_config
:canonical: fastvideo.configs.models.encoders.base.TextEncoderConfig.arch_config
:type: fastvideo.configs.models.base.ArchConfig
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.encoders.base.TextEncoderConfig.arch_config
:parser: docs.source.autodoc2_docstring_parser
```

````

`````
