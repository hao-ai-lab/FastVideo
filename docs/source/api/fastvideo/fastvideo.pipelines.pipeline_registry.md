# {py:mod}`fastvideo.pipelines.pipeline_registry`

```{py:module} fastvideo.pipelines.pipeline_registry
```

```{autodoc2-docstring} fastvideo.pipelines.pipeline_registry
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PipelineType <fastvideo.pipelines.pipeline_registry.PipelineType>`
  - ```{autodoc2-docstring} fastvideo.pipelines.pipeline_registry.PipelineType
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_pipeline_registry <fastvideo.pipelines.pipeline_registry.get_pipeline_registry>`
  - ```{autodoc2-docstring} fastvideo.pipelines.pipeline_registry.get_pipeline_registry
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`import_pipeline_classes <fastvideo.pipelines.pipeline_registry.import_pipeline_classes>`
  - ```{autodoc2-docstring} fastvideo.pipelines.pipeline_registry.import_pipeline_classes
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.pipelines.pipeline_registry.logger>`
  - ```{autodoc2-docstring} fastvideo.pipelines.pipeline_registry.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} PipelineType()
:canonical: fastvideo.pipelines.pipeline_registry.PipelineType

Bases: {py:obj}`str`, {py:obj}`enum.Enum`

```{autodoc2-docstring} fastvideo.pipelines.pipeline_registry.PipelineType
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.pipelines.pipeline_registry.PipelineType.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} BASIC
:canonical: fastvideo.pipelines.pipeline_registry.PipelineType.BASIC
:value: >
   'basic'

```{autodoc2-docstring} fastvideo.pipelines.pipeline_registry.PipelineType.BASIC
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} PREPROCESS
:canonical: fastvideo.pipelines.pipeline_registry.PipelineType.PREPROCESS
:value: >
   'preprocess'

```{autodoc2-docstring} fastvideo.pipelines.pipeline_registry.PipelineType.PREPROCESS
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} TRAINING
:canonical: fastvideo.pipelines.pipeline_registry.PipelineType.TRAINING
:value: >
   'training'

```{autodoc2-docstring} fastvideo.pipelines.pipeline_registry.PipelineType.TRAINING
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} choices() -> list[str]
:canonical: fastvideo.pipelines.pipeline_registry.PipelineType.choices
:classmethod:

```{autodoc2-docstring} fastvideo.pipelines.pipeline_registry.PipelineType.choices
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} from_string(value: str) -> fastvideo.pipelines.pipeline_registry.PipelineType
:canonical: fastvideo.pipelines.pipeline_registry.PipelineType.from_string
:classmethod:

```{autodoc2-docstring} fastvideo.pipelines.pipeline_registry.PipelineType.from_string
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:function} get_pipeline_registry(pipeline_type: fastvideo.pipelines.pipeline_registry.PipelineType | str | None = None) -> fastvideo.pipelines.pipeline_registry._PipelineRegistry
:canonical: fastvideo.pipelines.pipeline_registry.get_pipeline_registry

```{autodoc2-docstring} fastvideo.pipelines.pipeline_registry.get_pipeline_registry
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} import_pipeline_classes(pipeline_types: list[fastvideo.pipelines.pipeline_registry.PipelineType] | fastvideo.pipelines.pipeline_registry.PipelineType | None = None) -> dict[str, dict[str, dict[str, type[fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase] | None]]]
:canonical: fastvideo.pipelines.pipeline_registry.import_pipeline_classes

```{autodoc2-docstring} fastvideo.pipelines.pipeline_registry.import_pipeline_classes
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:data} logger
:canonical: fastvideo.pipelines.pipeline_registry.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.pipelines.pipeline_registry.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
