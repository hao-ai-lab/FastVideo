# {py:mod}`fastvideo.pipelines.basic.wan.wan_pipeline`

```{py:module} fastvideo.pipelines.basic.wan.wan_pipeline
```

```{autodoc2-docstring} fastvideo.pipelines.basic.wan.wan_pipeline
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`WanPipeline <fastvideo.pipelines.basic.wan.wan_pipeline.WanPipeline>`
  - ```{autodoc2-docstring} fastvideo.pipelines.basic.wan.wan_pipeline.WanPipeline
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EntryClass <fastvideo.pipelines.basic.wan.wan_pipeline.EntryClass>`
  - ```{autodoc2-docstring} fastvideo.pipelines.basic.wan.wan_pipeline.EntryClass
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`logger <fastvideo.pipelines.basic.wan.wan_pipeline.logger>`
  - ```{autodoc2-docstring} fastvideo.pipelines.basic.wan.wan_pipeline.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

````{py:data} EntryClass
:canonical: fastvideo.pipelines.basic.wan.wan_pipeline.EntryClass
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.basic.wan.wan_pipeline.EntryClass
:parser: docs.source.autodoc2_docstring_parser
```

````

`````{py:class} WanPipeline(*args, **kwargs)
:canonical: fastvideo.pipelines.basic.wan.wan_pipeline.WanPipeline

Bases: {py:obj}`fastvideo.pipelines.LoRAPipeline`, {py:obj}`fastvideo.pipelines.ComposedPipelineBase`

```{autodoc2-docstring} fastvideo.pipelines.basic.wan.wan_pipeline.WanPipeline
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.pipelines.basic.wan.wan_pipeline.WanPipeline.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} create_pipeline_stages(fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> None
:canonical: fastvideo.pipelines.basic.wan.wan_pipeline.WanPipeline.create_pipeline_stages

```{autodoc2-docstring} fastvideo.pipelines.basic.wan.wan_pipeline.WanPipeline.create_pipeline_stages
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} initialize_pipeline(fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs)
:canonical: fastvideo.pipelines.basic.wan.wan_pipeline.WanPipeline.initialize_pipeline

````

`````

````{py:data} logger
:canonical: fastvideo.pipelines.basic.wan.wan_pipeline.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.pipelines.basic.wan.wan_pipeline.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
