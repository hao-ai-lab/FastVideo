# {py:mod}`fastvideo.configs.pipelines.hunyuan`

```{py:module} fastvideo.configs.pipelines.hunyuan
```

```{autodoc2-docstring} fastvideo.configs.pipelines.hunyuan
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FastHunyuanConfig <fastvideo.configs.pipelines.hunyuan.FastHunyuanConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.pipelines.hunyuan.FastHunyuanConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`HunyuanConfig <fastvideo.configs.pipelines.hunyuan.HunyuanConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.pipelines.hunyuan.HunyuanConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`PromptTemplate <fastvideo.configs.pipelines.hunyuan.PromptTemplate>`
  -
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`clip_postprocess_text <fastvideo.configs.pipelines.hunyuan.clip_postprocess_text>`
  - ```{autodoc2-docstring} fastvideo.configs.pipelines.hunyuan.clip_postprocess_text
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`clip_preprocess_text <fastvideo.configs.pipelines.hunyuan.clip_preprocess_text>`
  - ```{autodoc2-docstring} fastvideo.configs.pipelines.hunyuan.clip_preprocess_text
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`llama_postprocess_text <fastvideo.configs.pipelines.hunyuan.llama_postprocess_text>`
  - ```{autodoc2-docstring} fastvideo.configs.pipelines.hunyuan.llama_postprocess_text
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`llama_preprocess_text <fastvideo.configs.pipelines.hunyuan.llama_preprocess_text>`
  - ```{autodoc2-docstring} fastvideo.configs.pipelines.hunyuan.llama_preprocess_text
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PROMPT_TEMPLATE_ENCODE_VIDEO <fastvideo.configs.pipelines.hunyuan.PROMPT_TEMPLATE_ENCODE_VIDEO>`
  - ```{autodoc2-docstring} fastvideo.configs.pipelines.hunyuan.PROMPT_TEMPLATE_ENCODE_VIDEO
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`prompt_template_video <fastvideo.configs.pipelines.hunyuan.prompt_template_video>`
  - ```{autodoc2-docstring} fastvideo.configs.pipelines.hunyuan.prompt_template_video
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} FastHunyuanConfig
:canonical: fastvideo.configs.pipelines.hunyuan.FastHunyuanConfig

Bases: {py:obj}`fastvideo.configs.pipelines.hunyuan.HunyuanConfig`

```{autodoc2-docstring} fastvideo.configs.pipelines.hunyuan.FastHunyuanConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} flow_shift
:canonical: fastvideo.configs.pipelines.hunyuan.FastHunyuanConfig.flow_shift
:type: int
:value: >
   17

```{autodoc2-docstring} fastvideo.configs.pipelines.hunyuan.FastHunyuanConfig.flow_shift
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} HunyuanConfig
:canonical: fastvideo.configs.pipelines.hunyuan.HunyuanConfig

Bases: {py:obj}`fastvideo.configs.pipelines.base.PipelineConfig`

```{autodoc2-docstring} fastvideo.configs.pipelines.hunyuan.HunyuanConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} dit_config
:canonical: fastvideo.configs.pipelines.hunyuan.HunyuanConfig.dit_config
:type: fastvideo.configs.models.DiTConfig
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.pipelines.hunyuan.HunyuanConfig.dit_config
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} dit_precision
:canonical: fastvideo.configs.pipelines.hunyuan.HunyuanConfig.dit_precision
:type: str
:value: >
   'bf16'

```{autodoc2-docstring} fastvideo.configs.pipelines.hunyuan.HunyuanConfig.dit_precision
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} embedded_cfg_scale
:canonical: fastvideo.configs.pipelines.hunyuan.HunyuanConfig.embedded_cfg_scale
:type: int
:value: >
   6

```{autodoc2-docstring} fastvideo.configs.pipelines.hunyuan.HunyuanConfig.embedded_cfg_scale
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} flow_shift
:canonical: fastvideo.configs.pipelines.hunyuan.HunyuanConfig.flow_shift
:type: int
:value: >
   7

```{autodoc2-docstring} fastvideo.configs.pipelines.hunyuan.HunyuanConfig.flow_shift
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} postprocess_text_funcs
:canonical: fastvideo.configs.pipelines.hunyuan.HunyuanConfig.postprocess_text_funcs
:type: tuple[collections.abc.Callable[[fastvideo.configs.models.encoders.BaseEncoderOutput], torch.tensor], ...]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.pipelines.hunyuan.HunyuanConfig.postprocess_text_funcs
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} preprocess_text_funcs
:canonical: fastvideo.configs.pipelines.hunyuan.HunyuanConfig.preprocess_text_funcs
:type: tuple[collections.abc.Callable[[str], str], ...]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.pipelines.hunyuan.HunyuanConfig.preprocess_text_funcs
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} text_encoder_configs
:canonical: fastvideo.configs.pipelines.hunyuan.HunyuanConfig.text_encoder_configs
:type: tuple[fastvideo.configs.models.EncoderConfig, ...]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.pipelines.hunyuan.HunyuanConfig.text_encoder_configs
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} text_encoder_precisions
:canonical: fastvideo.configs.pipelines.hunyuan.HunyuanConfig.text_encoder_precisions
:type: tuple[str, ...]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.pipelines.hunyuan.HunyuanConfig.text_encoder_precisions
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} vae_config
:canonical: fastvideo.configs.pipelines.hunyuan.HunyuanConfig.vae_config
:type: fastvideo.configs.models.VAEConfig
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.pipelines.hunyuan.HunyuanConfig.vae_config
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} vae_precision
:canonical: fastvideo.configs.pipelines.hunyuan.HunyuanConfig.vae_precision
:type: str
:value: >
   'fp16'

```{autodoc2-docstring} fastvideo.configs.pipelines.hunyuan.HunyuanConfig.vae_precision
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} PROMPT_TEMPLATE_ENCODE_VIDEO
:canonical: fastvideo.configs.pipelines.hunyuan.PROMPT_TEMPLATE_ENCODE_VIDEO
:value: <Multiline-String>

```{autodoc2-docstring} fastvideo.configs.pipelines.hunyuan.PROMPT_TEMPLATE_ENCODE_VIDEO
:parser: docs.source.autodoc2_docstring_parser
```

````

`````{py:class} PromptTemplate()
:canonical: fastvideo.configs.pipelines.hunyuan.PromptTemplate

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} crop_start
:canonical: fastvideo.configs.pipelines.hunyuan.PromptTemplate.crop_start
:type: int
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.pipelines.hunyuan.PromptTemplate.crop_start
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} template
:canonical: fastvideo.configs.pipelines.hunyuan.PromptTemplate.template
:type: str
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.pipelines.hunyuan.PromptTemplate.template
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:function} clip_postprocess_text(outputs: fastvideo.configs.models.encoders.BaseEncoderOutput) -> torch.tensor
:canonical: fastvideo.configs.pipelines.hunyuan.clip_postprocess_text

```{autodoc2-docstring} fastvideo.configs.pipelines.hunyuan.clip_postprocess_text
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} clip_preprocess_text(prompt: str) -> str
:canonical: fastvideo.configs.pipelines.hunyuan.clip_preprocess_text

```{autodoc2-docstring} fastvideo.configs.pipelines.hunyuan.clip_preprocess_text
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} llama_postprocess_text(outputs: fastvideo.configs.models.encoders.BaseEncoderOutput) -> torch.tensor
:canonical: fastvideo.configs.pipelines.hunyuan.llama_postprocess_text

```{autodoc2-docstring} fastvideo.configs.pipelines.hunyuan.llama_postprocess_text
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} llama_preprocess_text(prompt: str) -> str
:canonical: fastvideo.configs.pipelines.hunyuan.llama_preprocess_text

```{autodoc2-docstring} fastvideo.configs.pipelines.hunyuan.llama_preprocess_text
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:data} prompt_template_video
:canonical: fastvideo.configs.pipelines.hunyuan.prompt_template_video
:type: fastvideo.configs.pipelines.hunyuan.PromptTemplate
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.pipelines.hunyuan.prompt_template_video
:parser: docs.source.autodoc2_docstring_parser
```

````
