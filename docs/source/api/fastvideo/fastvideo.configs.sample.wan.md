# {py:mod}`fastvideo.configs.sample.wan`

```{py:module} fastvideo.configs.sample.wan
```

```{autodoc2-docstring} fastvideo.configs.sample.wan
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FastWanT2V480PConfig <fastvideo.configs.sample.wan.FastWanT2V480PConfig>`
  -
* - {py:obj}`Wan2_2_Base_SamplingParam <fastvideo.configs.sample.wan.Wan2_2_Base_SamplingParam>`
  - ```{autodoc2-docstring} fastvideo.configs.sample.wan.Wan2_2_Base_SamplingParam
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`Wan2_2_I2V_A14B_SamplingParam <fastvideo.configs.sample.wan.Wan2_2_I2V_A14B_SamplingParam>`
  -
* - {py:obj}`Wan2_2_T2V_A14B_SamplingParam <fastvideo.configs.sample.wan.Wan2_2_T2V_A14B_SamplingParam>`
  -
* - {py:obj}`Wan2_2_TI2V_5B_SamplingParam <fastvideo.configs.sample.wan.Wan2_2_TI2V_5B_SamplingParam>`
  - ```{autodoc2-docstring} fastvideo.configs.sample.wan.Wan2_2_TI2V_5B_SamplingParam
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`WanI2V_14B_480P_SamplingParam <fastvideo.configs.sample.wan.WanI2V_14B_480P_SamplingParam>`
  -
* - {py:obj}`WanI2V_14B_720P_SamplingParam <fastvideo.configs.sample.wan.WanI2V_14B_720P_SamplingParam>`
  -
* - {py:obj}`WanT2V_14B_SamplingParam <fastvideo.configs.sample.wan.WanT2V_14B_SamplingParam>`
  -
* - {py:obj}`WanT2V_1_3B_SamplingParam <fastvideo.configs.sample.wan.WanT2V_1_3B_SamplingParam>`
  -
````

### API

`````{py:class} FastWanT2V480PConfig
:canonical: fastvideo.configs.sample.wan.FastWanT2V480PConfig

Bases: {py:obj}`fastvideo.configs.sample.wan.WanT2V_1_3B_SamplingParam`

````{py:attribute} fps
:canonical: fastvideo.configs.sample.wan.FastWanT2V480PConfig.fps
:type: int
:value: >
   16

```{autodoc2-docstring} fastvideo.configs.sample.wan.FastWanT2V480PConfig.fps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} height
:canonical: fastvideo.configs.sample.wan.FastWanT2V480PConfig.height
:type: int
:value: >
   448

```{autodoc2-docstring} fastvideo.configs.sample.wan.FastWanT2V480PConfig.height
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_frames
:canonical: fastvideo.configs.sample.wan.FastWanT2V480PConfig.num_frames
:type: int
:value: >
   61

```{autodoc2-docstring} fastvideo.configs.sample.wan.FastWanT2V480PConfig.num_frames
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_inference_steps
:canonical: fastvideo.configs.sample.wan.FastWanT2V480PConfig.num_inference_steps
:type: int
:value: >
   3

```{autodoc2-docstring} fastvideo.configs.sample.wan.FastWanT2V480PConfig.num_inference_steps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} width
:canonical: fastvideo.configs.sample.wan.FastWanT2V480PConfig.width
:type: int
:value: >
   832

```{autodoc2-docstring} fastvideo.configs.sample.wan.FastWanT2V480PConfig.width
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} Wan2_2_Base_SamplingParam
:canonical: fastvideo.configs.sample.wan.Wan2_2_Base_SamplingParam

Bases: {py:obj}`fastvideo.configs.sample.base.SamplingParam`

```{autodoc2-docstring} fastvideo.configs.sample.wan.Wan2_2_Base_SamplingParam
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} negative_prompt
:canonical: fastvideo.configs.sample.wan.Wan2_2_Base_SamplingParam.negative_prompt
:type: str | None
:value: >
   '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸...'

```{autodoc2-docstring} fastvideo.configs.sample.wan.Wan2_2_Base_SamplingParam.negative_prompt
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

```{py:class} Wan2_2_I2V_A14B_SamplingParam
:canonical: fastvideo.configs.sample.wan.Wan2_2_I2V_A14B_SamplingParam

Bases: {py:obj}`fastvideo.configs.sample.wan.Wan2_2_Base_SamplingParam`

```

```{py:class} Wan2_2_T2V_A14B_SamplingParam
:canonical: fastvideo.configs.sample.wan.Wan2_2_T2V_A14B_SamplingParam

Bases: {py:obj}`fastvideo.configs.sample.wan.Wan2_2_Base_SamplingParam`

```

`````{py:class} Wan2_2_TI2V_5B_SamplingParam
:canonical: fastvideo.configs.sample.wan.Wan2_2_TI2V_5B_SamplingParam

Bases: {py:obj}`fastvideo.configs.sample.wan.Wan2_2_Base_SamplingParam`

```{autodoc2-docstring} fastvideo.configs.sample.wan.Wan2_2_TI2V_5B_SamplingParam
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} fps
:canonical: fastvideo.configs.sample.wan.Wan2_2_TI2V_5B_SamplingParam.fps
:type: int
:value: >
   24

```{autodoc2-docstring} fastvideo.configs.sample.wan.Wan2_2_TI2V_5B_SamplingParam.fps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} guidance_scale
:canonical: fastvideo.configs.sample.wan.Wan2_2_TI2V_5B_SamplingParam.guidance_scale
:type: float
:value: >
   5.0

```{autodoc2-docstring} fastvideo.configs.sample.wan.Wan2_2_TI2V_5B_SamplingParam.guidance_scale
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} height
:canonical: fastvideo.configs.sample.wan.Wan2_2_TI2V_5B_SamplingParam.height
:type: int
:value: >
   704

```{autodoc2-docstring} fastvideo.configs.sample.wan.Wan2_2_TI2V_5B_SamplingParam.height
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_frames
:canonical: fastvideo.configs.sample.wan.Wan2_2_TI2V_5B_SamplingParam.num_frames
:type: int
:value: >
   121

```{autodoc2-docstring} fastvideo.configs.sample.wan.Wan2_2_TI2V_5B_SamplingParam.num_frames
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_inference_steps
:canonical: fastvideo.configs.sample.wan.Wan2_2_TI2V_5B_SamplingParam.num_inference_steps
:type: int
:value: >
   50

```{autodoc2-docstring} fastvideo.configs.sample.wan.Wan2_2_TI2V_5B_SamplingParam.num_inference_steps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} width
:canonical: fastvideo.configs.sample.wan.Wan2_2_TI2V_5B_SamplingParam.width
:type: int
:value: >
   1280

```{autodoc2-docstring} fastvideo.configs.sample.wan.Wan2_2_TI2V_5B_SamplingParam.width
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} WanI2V_14B_480P_SamplingParam
:canonical: fastvideo.configs.sample.wan.WanI2V_14B_480P_SamplingParam

Bases: {py:obj}`fastvideo.configs.sample.wan.WanT2V_1_3B_SamplingParam`

````{py:attribute} guidance_scale
:canonical: fastvideo.configs.sample.wan.WanI2V_14B_480P_SamplingParam.guidance_scale
:type: float
:value: >
   5.0

```{autodoc2-docstring} fastvideo.configs.sample.wan.WanI2V_14B_480P_SamplingParam.guidance_scale
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_inference_steps
:canonical: fastvideo.configs.sample.wan.WanI2V_14B_480P_SamplingParam.num_inference_steps
:type: int
:value: >
   40

```{autodoc2-docstring} fastvideo.configs.sample.wan.WanI2V_14B_480P_SamplingParam.num_inference_steps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} teacache_params
:canonical: fastvideo.configs.sample.wan.WanI2V_14B_480P_SamplingParam.teacache_params
:type: fastvideo.configs.sample.teacache.WanTeaCacheParams
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.sample.wan.WanI2V_14B_480P_SamplingParam.teacache_params
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} WanI2V_14B_720P_SamplingParam
:canonical: fastvideo.configs.sample.wan.WanI2V_14B_720P_SamplingParam

Bases: {py:obj}`fastvideo.configs.sample.wan.WanT2V_14B_SamplingParam`

````{py:attribute} guidance_scale
:canonical: fastvideo.configs.sample.wan.WanI2V_14B_720P_SamplingParam.guidance_scale
:type: float
:value: >
   5.0

```{autodoc2-docstring} fastvideo.configs.sample.wan.WanI2V_14B_720P_SamplingParam.guidance_scale
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_inference_steps
:canonical: fastvideo.configs.sample.wan.WanI2V_14B_720P_SamplingParam.num_inference_steps
:type: int
:value: >
   40

```{autodoc2-docstring} fastvideo.configs.sample.wan.WanI2V_14B_720P_SamplingParam.num_inference_steps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} teacache_params
:canonical: fastvideo.configs.sample.wan.WanI2V_14B_720P_SamplingParam.teacache_params
:type: fastvideo.configs.sample.teacache.WanTeaCacheParams
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.sample.wan.WanI2V_14B_720P_SamplingParam.teacache_params
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} WanT2V_14B_SamplingParam
:canonical: fastvideo.configs.sample.wan.WanT2V_14B_SamplingParam

Bases: {py:obj}`fastvideo.configs.sample.base.SamplingParam`

````{py:attribute} fps
:canonical: fastvideo.configs.sample.wan.WanT2V_14B_SamplingParam.fps
:type: int
:value: >
   16

```{autodoc2-docstring} fastvideo.configs.sample.wan.WanT2V_14B_SamplingParam.fps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} guidance_scale
:canonical: fastvideo.configs.sample.wan.WanT2V_14B_SamplingParam.guidance_scale
:type: float
:value: >
   5.0

```{autodoc2-docstring} fastvideo.configs.sample.wan.WanT2V_14B_SamplingParam.guidance_scale
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} height
:canonical: fastvideo.configs.sample.wan.WanT2V_14B_SamplingParam.height
:type: int
:value: >
   720

```{autodoc2-docstring} fastvideo.configs.sample.wan.WanT2V_14B_SamplingParam.height
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} negative_prompt
:canonical: fastvideo.configs.sample.wan.WanT2V_14B_SamplingParam.negative_prompt
:type: str
:value: >
   'Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, stat...'

```{autodoc2-docstring} fastvideo.configs.sample.wan.WanT2V_14B_SamplingParam.negative_prompt
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_frames
:canonical: fastvideo.configs.sample.wan.WanT2V_14B_SamplingParam.num_frames
:type: int
:value: >
   81

```{autodoc2-docstring} fastvideo.configs.sample.wan.WanT2V_14B_SamplingParam.num_frames
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_inference_steps
:canonical: fastvideo.configs.sample.wan.WanT2V_14B_SamplingParam.num_inference_steps
:type: int
:value: >
   50

```{autodoc2-docstring} fastvideo.configs.sample.wan.WanT2V_14B_SamplingParam.num_inference_steps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} teacache_params
:canonical: fastvideo.configs.sample.wan.WanT2V_14B_SamplingParam.teacache_params
:type: fastvideo.configs.sample.teacache.WanTeaCacheParams
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.sample.wan.WanT2V_14B_SamplingParam.teacache_params
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} width
:canonical: fastvideo.configs.sample.wan.WanT2V_14B_SamplingParam.width
:type: int
:value: >
   1280

```{autodoc2-docstring} fastvideo.configs.sample.wan.WanT2V_14B_SamplingParam.width
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} WanT2V_1_3B_SamplingParam
:canonical: fastvideo.configs.sample.wan.WanT2V_1_3B_SamplingParam

Bases: {py:obj}`fastvideo.configs.sample.base.SamplingParam`

````{py:attribute} fps
:canonical: fastvideo.configs.sample.wan.WanT2V_1_3B_SamplingParam.fps
:type: int
:value: >
   16

```{autodoc2-docstring} fastvideo.configs.sample.wan.WanT2V_1_3B_SamplingParam.fps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} guidance_scale
:canonical: fastvideo.configs.sample.wan.WanT2V_1_3B_SamplingParam.guidance_scale
:type: float
:value: >
   3.0

```{autodoc2-docstring} fastvideo.configs.sample.wan.WanT2V_1_3B_SamplingParam.guidance_scale
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} height
:canonical: fastvideo.configs.sample.wan.WanT2V_1_3B_SamplingParam.height
:type: int
:value: >
   480

```{autodoc2-docstring} fastvideo.configs.sample.wan.WanT2V_1_3B_SamplingParam.height
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} negative_prompt
:canonical: fastvideo.configs.sample.wan.WanT2V_1_3B_SamplingParam.negative_prompt
:type: str
:value: >
   'Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, stat...'

```{autodoc2-docstring} fastvideo.configs.sample.wan.WanT2V_1_3B_SamplingParam.negative_prompt
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_frames
:canonical: fastvideo.configs.sample.wan.WanT2V_1_3B_SamplingParam.num_frames
:type: int
:value: >
   81

```{autodoc2-docstring} fastvideo.configs.sample.wan.WanT2V_1_3B_SamplingParam.num_frames
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_inference_steps
:canonical: fastvideo.configs.sample.wan.WanT2V_1_3B_SamplingParam.num_inference_steps
:type: int
:value: >
   50

```{autodoc2-docstring} fastvideo.configs.sample.wan.WanT2V_1_3B_SamplingParam.num_inference_steps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} teacache_params
:canonical: fastvideo.configs.sample.wan.WanT2V_1_3B_SamplingParam.teacache_params
:type: fastvideo.configs.sample.teacache.WanTeaCacheParams
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.sample.wan.WanT2V_1_3B_SamplingParam.teacache_params
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} width
:canonical: fastvideo.configs.sample.wan.WanT2V_1_3B_SamplingParam.width
:type: int
:value: >
   832

```{autodoc2-docstring} fastvideo.configs.sample.wan.WanT2V_1_3B_SamplingParam.width
:parser: docs.source.autodoc2_docstring_parser
```

````

`````
