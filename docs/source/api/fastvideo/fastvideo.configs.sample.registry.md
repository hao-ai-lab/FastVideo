# {py:mod}`fastvideo.configs.sample.registry`

```{py:module} fastvideo.configs.sample.registry
```

```{autodoc2-docstring} fastvideo.configs.sample.registry
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_sampling_param_cls_for_name <fastvideo.configs.sample.registry.get_sampling_param_cls_for_name>`
  - ```{autodoc2-docstring} fastvideo.configs.sample.registry.get_sampling_param_cls_for_name
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SAMPLING_FALLBACK_PARAM <fastvideo.configs.sample.registry.SAMPLING_FALLBACK_PARAM>`
  - ```{autodoc2-docstring} fastvideo.configs.sample.registry.SAMPLING_FALLBACK_PARAM
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`SAMPLING_PARAM_DETECTOR <fastvideo.configs.sample.registry.SAMPLING_PARAM_DETECTOR>`
  - ```{autodoc2-docstring} fastvideo.configs.sample.registry.SAMPLING_PARAM_DETECTOR
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`SAMPLING_PARAM_REGISTRY <fastvideo.configs.sample.registry.SAMPLING_PARAM_REGISTRY>`
  - ```{autodoc2-docstring} fastvideo.configs.sample.registry.SAMPLING_PARAM_REGISTRY
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`logger <fastvideo.configs.sample.registry.logger>`
  - ```{autodoc2-docstring} fastvideo.configs.sample.registry.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

````{py:data} SAMPLING_FALLBACK_PARAM
:canonical: fastvideo.configs.sample.registry.SAMPLING_FALLBACK_PARAM
:type: dict[str, typing.Any]
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.sample.registry.SAMPLING_FALLBACK_PARAM
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:data} SAMPLING_PARAM_DETECTOR
:canonical: fastvideo.configs.sample.registry.SAMPLING_PARAM_DETECTOR
:type: dict[str, collections.abc.Callable[[str], bool]]
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.sample.registry.SAMPLING_PARAM_DETECTOR
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:data} SAMPLING_PARAM_REGISTRY
:canonical: fastvideo.configs.sample.registry.SAMPLING_PARAM_REGISTRY
:type: dict[str, typing.Any]
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.sample.registry.SAMPLING_PARAM_REGISTRY
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:function} get_sampling_param_cls_for_name(pipeline_name_or_path: str) -> typing.Any | None
:canonical: fastvideo.configs.sample.registry.get_sampling_param_cls_for_name

```{autodoc2-docstring} fastvideo.configs.sample.registry.get_sampling_param_cls_for_name
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:data} logger
:canonical: fastvideo.configs.sample.registry.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.configs.sample.registry.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
