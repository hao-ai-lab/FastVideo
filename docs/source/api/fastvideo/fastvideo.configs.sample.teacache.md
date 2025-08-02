# {py:mod}`fastvideo.configs.sample.teacache`

```{py:module} fastvideo.configs.sample.teacache
```

```{autodoc2-docstring} fastvideo.configs.sample.teacache
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TeaCacheParams <fastvideo.configs.sample.teacache.TeaCacheParams>`
  - ```{autodoc2-docstring} fastvideo.configs.sample.teacache.TeaCacheParams
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`WanTeaCacheParams <fastvideo.configs.sample.teacache.WanTeaCacheParams>`
  - ```{autodoc2-docstring} fastvideo.configs.sample.teacache.WanTeaCacheParams
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} TeaCacheParams
:canonical: fastvideo.configs.sample.teacache.TeaCacheParams

Bases: {py:obj}`fastvideo.configs.sample.base.CacheParams`

```{autodoc2-docstring} fastvideo.configs.sample.teacache.TeaCacheParams
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} cache_type
:canonical: fastvideo.configs.sample.teacache.TeaCacheParams.cache_type
:type: str
:value: >
   'teacache'

```{autodoc2-docstring} fastvideo.configs.sample.teacache.TeaCacheParams.cache_type
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} coefficients
:canonical: fastvideo.configs.sample.teacache.TeaCacheParams.coefficients
:type: list[float]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.sample.teacache.TeaCacheParams.coefficients
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} teacache_thresh
:canonical: fastvideo.configs.sample.teacache.TeaCacheParams.teacache_thresh
:type: float
:value: >
   0.0

```{autodoc2-docstring} fastvideo.configs.sample.teacache.TeaCacheParams.teacache_thresh
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} WanTeaCacheParams
:canonical: fastvideo.configs.sample.teacache.WanTeaCacheParams

Bases: {py:obj}`fastvideo.configs.sample.base.CacheParams`

```{autodoc2-docstring} fastvideo.configs.sample.teacache.WanTeaCacheParams
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} cache_type
:canonical: fastvideo.configs.sample.teacache.WanTeaCacheParams.cache_type
:type: str
:value: >
   'teacache'

```{autodoc2-docstring} fastvideo.configs.sample.teacache.WanTeaCacheParams.cache_type
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:property} coefficients
:canonical: fastvideo.configs.sample.teacache.WanTeaCacheParams.coefficients
:type: list[float]

```{autodoc2-docstring} fastvideo.configs.sample.teacache.WanTeaCacheParams.coefficients
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_cutoff_steps(num_inference_steps: int) -> int
:canonical: fastvideo.configs.sample.teacache.WanTeaCacheParams.get_cutoff_steps

```{autodoc2-docstring} fastvideo.configs.sample.teacache.WanTeaCacheParams.get_cutoff_steps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} non_ret_steps_coeffs
:canonical: fastvideo.configs.sample.teacache.WanTeaCacheParams.non_ret_steps_coeffs
:type: list[float]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.sample.teacache.WanTeaCacheParams.non_ret_steps_coeffs
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:property} ret_steps
:canonical: fastvideo.configs.sample.teacache.WanTeaCacheParams.ret_steps
:type: int

```{autodoc2-docstring} fastvideo.configs.sample.teacache.WanTeaCacheParams.ret_steps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} ret_steps_coeffs
:canonical: fastvideo.configs.sample.teacache.WanTeaCacheParams.ret_steps_coeffs
:type: list[float]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.sample.teacache.WanTeaCacheParams.ret_steps_coeffs
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} teacache_thresh
:canonical: fastvideo.configs.sample.teacache.WanTeaCacheParams.teacache_thresh
:type: float
:value: >
   0.0

```{autodoc2-docstring} fastvideo.configs.sample.teacache.WanTeaCacheParams.teacache_thresh
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} use_ret_steps
:canonical: fastvideo.configs.sample.teacache.WanTeaCacheParams.use_ret_steps
:type: bool
:value: >
   True

```{autodoc2-docstring} fastvideo.configs.sample.teacache.WanTeaCacheParams.use_ret_steps
:parser: docs.source.autodoc2_docstring_parser
```

````

`````
