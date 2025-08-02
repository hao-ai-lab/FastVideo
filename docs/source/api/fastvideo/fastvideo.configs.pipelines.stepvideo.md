# {py:mod}`fastvideo.configs.pipelines.stepvideo`

```{py:module} fastvideo.configs.pipelines.stepvideo
```

```{autodoc2-docstring} fastvideo.configs.pipelines.stepvideo
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`StepVideoT2VConfig <fastvideo.configs.pipelines.stepvideo.StepVideoT2VConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.pipelines.stepvideo.StepVideoT2VConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} StepVideoT2VConfig
:canonical: fastvideo.configs.pipelines.stepvideo.StepVideoT2VConfig

Bases: {py:obj}`fastvideo.configs.pipelines.base.PipelineConfig`

```{autodoc2-docstring} fastvideo.configs.pipelines.stepvideo.StepVideoT2VConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} dit_config
:canonical: fastvideo.configs.pipelines.stepvideo.StepVideoT2VConfig.dit_config
:type: fastvideo.configs.models.DiTConfig
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.pipelines.stepvideo.StepVideoT2VConfig.dit_config
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} flow_shift
:canonical: fastvideo.configs.pipelines.stepvideo.StepVideoT2VConfig.flow_shift
:type: int
:value: >
   13

```{autodoc2-docstring} fastvideo.configs.pipelines.stepvideo.StepVideoT2VConfig.flow_shift
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} neg_magic
:canonical: fastvideo.configs.pipelines.stepvideo.StepVideoT2VConfig.neg_magic
:type: str
:value: >
   '画面暗、低分辨率、不良手、文本、缺少手指、多余的手指、裁剪、低质量、颗粒状、签名、水印、用户名、模糊。'

```{autodoc2-docstring} fastvideo.configs.pipelines.stepvideo.StepVideoT2VConfig.neg_magic
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} pos_magic
:canonical: fastvideo.configs.pipelines.stepvideo.StepVideoT2VConfig.pos_magic
:type: str
:value: >
   '超高清、HDR 视频、环境光、杜比全景声、画面稳定、流畅动作、逼真的细节、专业级构图、超现实主义、自然、生动、超细节、清晰。'

```{autodoc2-docstring} fastvideo.configs.pipelines.stepvideo.StepVideoT2VConfig.pos_magic
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} precision
:canonical: fastvideo.configs.pipelines.stepvideo.StepVideoT2VConfig.precision
:type: str
:value: >
   'bf16'

```{autodoc2-docstring} fastvideo.configs.pipelines.stepvideo.StepVideoT2VConfig.precision
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} timesteps_scale
:canonical: fastvideo.configs.pipelines.stepvideo.StepVideoT2VConfig.timesteps_scale
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.configs.pipelines.stepvideo.StepVideoT2VConfig.timesteps_scale
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} vae_config
:canonical: fastvideo.configs.pipelines.stepvideo.StepVideoT2VConfig.vae_config
:type: fastvideo.configs.models.VAEConfig
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.pipelines.stepvideo.StepVideoT2VConfig.vae_config
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} vae_precision
:canonical: fastvideo.configs.pipelines.stepvideo.StepVideoT2VConfig.vae_precision
:type: str
:value: >
   'bf16'

```{autodoc2-docstring} fastvideo.configs.pipelines.stepvideo.StepVideoT2VConfig.vae_precision
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} vae_sp
:canonical: fastvideo.configs.pipelines.stepvideo.StepVideoT2VConfig.vae_sp
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.configs.pipelines.stepvideo.StepVideoT2VConfig.vae_sp
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} vae_tiling
:canonical: fastvideo.configs.pipelines.stepvideo.StepVideoT2VConfig.vae_tiling
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.configs.pipelines.stepvideo.StepVideoT2VConfig.vae_tiling
:parser: docs.source.autodoc2_docstring_parser
```

````

`````
