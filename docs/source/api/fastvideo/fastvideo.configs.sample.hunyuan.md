# {py:mod}`fastvideo.configs.sample.hunyuan`

```{py:module} fastvideo.configs.sample.hunyuan
```

```{autodoc2-docstring} fastvideo.configs.sample.hunyuan
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FastHunyuanSamplingParam <fastvideo.configs.sample.hunyuan.FastHunyuanSamplingParam>`
  -
* - {py:obj}`HunyuanSamplingParam <fastvideo.configs.sample.hunyuan.HunyuanSamplingParam>`
  -
````

### API

`````{py:class} FastHunyuanSamplingParam
:canonical: fastvideo.configs.sample.hunyuan.FastHunyuanSamplingParam

Bases: {py:obj}`fastvideo.configs.sample.hunyuan.HunyuanSamplingParam`

````{py:attribute} num_inference_steps
:canonical: fastvideo.configs.sample.hunyuan.FastHunyuanSamplingParam.num_inference_steps
:type: int
:value: >
   6

```{autodoc2-docstring} fastvideo.configs.sample.hunyuan.FastHunyuanSamplingParam.num_inference_steps
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} HunyuanSamplingParam
:canonical: fastvideo.configs.sample.hunyuan.HunyuanSamplingParam

Bases: {py:obj}`fastvideo.configs.sample.base.SamplingParam`

````{py:attribute} fps
:canonical: fastvideo.configs.sample.hunyuan.HunyuanSamplingParam.fps
:type: int
:value: >
   24

```{autodoc2-docstring} fastvideo.configs.sample.hunyuan.HunyuanSamplingParam.fps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} guidance_scale
:canonical: fastvideo.configs.sample.hunyuan.HunyuanSamplingParam.guidance_scale
:type: float
:value: >
   1.0

```{autodoc2-docstring} fastvideo.configs.sample.hunyuan.HunyuanSamplingParam.guidance_scale
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} height
:canonical: fastvideo.configs.sample.hunyuan.HunyuanSamplingParam.height
:type: int
:value: >
   720

```{autodoc2-docstring} fastvideo.configs.sample.hunyuan.HunyuanSamplingParam.height
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_frames
:canonical: fastvideo.configs.sample.hunyuan.HunyuanSamplingParam.num_frames
:type: int
:value: >
   125

```{autodoc2-docstring} fastvideo.configs.sample.hunyuan.HunyuanSamplingParam.num_frames
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_inference_steps
:canonical: fastvideo.configs.sample.hunyuan.HunyuanSamplingParam.num_inference_steps
:type: int
:value: >
   50

```{autodoc2-docstring} fastvideo.configs.sample.hunyuan.HunyuanSamplingParam.num_inference_steps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} teacache_params
:canonical: fastvideo.configs.sample.hunyuan.HunyuanSamplingParam.teacache_params
:type: fastvideo.configs.sample.teacache.TeaCacheParams
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.sample.hunyuan.HunyuanSamplingParam.teacache_params
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} width
:canonical: fastvideo.configs.sample.hunyuan.HunyuanSamplingParam.width
:type: int
:value: >
   1280

```{autodoc2-docstring} fastvideo.configs.sample.hunyuan.HunyuanSamplingParam.width
:parser: docs.source.autodoc2_docstring_parser
```

````

`````
