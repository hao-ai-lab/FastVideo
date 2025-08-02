# {py:mod}`fastvideo.pipelines.stages.conditioning`

```{py:module} fastvideo.pipelines.stages.conditioning
```

```{autodoc2-docstring} fastvideo.pipelines.stages.conditioning
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ConditioningStage <fastvideo.pipelines.stages.conditioning.ConditioningStage>`
  - ```{autodoc2-docstring} fastvideo.pipelines.stages.conditioning.ConditioningStage
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.pipelines.stages.conditioning.logger>`
  - ```{autodoc2-docstring} fastvideo.pipelines.stages.conditioning.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} ConditioningStage
:canonical: fastvideo.pipelines.stages.conditioning.ConditioningStage

Bases: {py:obj}`fastvideo.pipelines.stages.base.PipelineStage`

```{autodoc2-docstring} fastvideo.pipelines.stages.conditioning.ConditioningStage
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} forward(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.pipeline_batch_info.ForwardBatch
:canonical: fastvideo.pipelines.stages.conditioning.ConditioningStage.forward

```{autodoc2-docstring} fastvideo.pipelines.stages.conditioning.ConditioningStage.forward
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} verify_input(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.stages.validators.VerificationResult
:canonical: fastvideo.pipelines.stages.conditioning.ConditioningStage.verify_input

```{autodoc2-docstring} fastvideo.pipelines.stages.conditioning.ConditioningStage.verify_input
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} verify_output(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.stages.validators.VerificationResult
:canonical: fastvideo.pipelines.stages.conditioning.ConditioningStage.verify_output

```{autodoc2-docstring} fastvideo.pipelines.stages.conditioning.ConditioningStage.verify_output
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} logger
:canonical: fastvideo.pipelines.stages.conditioning.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.pipelines.stages.conditioning.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
