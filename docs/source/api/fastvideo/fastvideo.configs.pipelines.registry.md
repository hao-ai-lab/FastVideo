# {py:mod}`fastvideo.configs.pipelines.registry`

```{py:module} fastvideo.configs.pipelines.registry
```

```{autodoc2-docstring} fastvideo.configs.pipelines.registry
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_pipeline_config_cls_from_name <fastvideo.configs.pipelines.registry.get_pipeline_config_cls_from_name>`
  - ```{autodoc2-docstring} fastvideo.configs.pipelines.registry.get_pipeline_config_cls_from_name
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PIPELINE_DETECTOR <fastvideo.configs.pipelines.registry.PIPELINE_DETECTOR>`
  - ```{autodoc2-docstring} fastvideo.configs.pipelines.registry.PIPELINE_DETECTOR
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`PIPELINE_FALLBACK_CONFIG <fastvideo.configs.pipelines.registry.PIPELINE_FALLBACK_CONFIG>`
  - ```{autodoc2-docstring} fastvideo.configs.pipelines.registry.PIPELINE_FALLBACK_CONFIG
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`PIPE_NAME_TO_CONFIG <fastvideo.configs.pipelines.registry.PIPE_NAME_TO_CONFIG>`
  - ```{autodoc2-docstring} fastvideo.configs.pipelines.registry.PIPE_NAME_TO_CONFIG
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`logger <fastvideo.configs.pipelines.registry.logger>`
  - ```{autodoc2-docstring} fastvideo.configs.pipelines.registry.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

````{py:data} PIPELINE_DETECTOR
:canonical: fastvideo.configs.pipelines.registry.PIPELINE_DETECTOR
:type: dict[str, collections.abc.Callable[[str], bool]]
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.pipelines.registry.PIPELINE_DETECTOR
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:data} PIPELINE_FALLBACK_CONFIG
:canonical: fastvideo.configs.pipelines.registry.PIPELINE_FALLBACK_CONFIG
:type: dict[str, type[fastvideo.configs.pipelines.base.PipelineConfig]]
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.pipelines.registry.PIPELINE_FALLBACK_CONFIG
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:data} PIPE_NAME_TO_CONFIG
:canonical: fastvideo.configs.pipelines.registry.PIPE_NAME_TO_CONFIG
:type: dict[str, type[fastvideo.configs.pipelines.base.PipelineConfig]]
:value: >
   None

```{autodoc2-docstring} fastvideo.configs.pipelines.registry.PIPE_NAME_TO_CONFIG
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:function} get_pipeline_config_cls_from_name(pipeline_name_or_path: str) -> type[fastvideo.configs.pipelines.base.PipelineConfig]
:canonical: fastvideo.configs.pipelines.registry.get_pipeline_config_cls_from_name

```{autodoc2-docstring} fastvideo.configs.pipelines.registry.get_pipeline_config_cls_from_name
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:data} logger
:canonical: fastvideo.configs.pipelines.registry.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.configs.pipelines.registry.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
