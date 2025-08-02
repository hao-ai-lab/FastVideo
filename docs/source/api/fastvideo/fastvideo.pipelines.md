# {py:mod}`fastvideo.pipelines`

```{py:module} fastvideo.pipelines
```

```{autodoc2-docstring} fastvideo.pipelines
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Subpackages

```{toctree}
:titlesonly:
:maxdepth: 3

fastvideo.pipelines.basic
fastvideo.pipelines.stages
fastvideo.pipelines.training
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

fastvideo.pipelines.composed_pipeline_base
fastvideo.pipelines.lora_pipeline
fastvideo.pipelines.pipeline_batch_info
fastvideo.pipelines.pipeline_registry
```

## Package Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PipelineWithLoRA <fastvideo.pipelines.PipelineWithLoRA>`
  - ```{autodoc2-docstring} fastvideo.pipelines.PipelineWithLoRA
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`build_pipeline <fastvideo.pipelines.build_pipeline>`
  - ```{autodoc2-docstring} fastvideo.pipelines.build_pipeline
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.pipelines.logger>`
  - ```{autodoc2-docstring} fastvideo.pipelines.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

````{py:class} PipelineWithLoRA(*args, **kwargs)
:canonical: fastvideo.pipelines.PipelineWithLoRA

Bases: {py:obj}`fastvideo.pipelines.lora_pipeline.LoRAPipeline`, {py:obj}`fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase`

```{autodoc2-docstring} fastvideo.pipelines.PipelineWithLoRA
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.pipelines.PipelineWithLoRA.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:function} build_pipeline(fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs, pipeline_type: fastvideo.pipelines.pipeline_registry.PipelineType | str = PipelineType.BASIC) -> fastvideo.pipelines.PipelineWithLoRA
:canonical: fastvideo.pipelines.build_pipeline

```{autodoc2-docstring} fastvideo.pipelines.build_pipeline
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:data} logger
:canonical: fastvideo.pipelines.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.pipelines.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
