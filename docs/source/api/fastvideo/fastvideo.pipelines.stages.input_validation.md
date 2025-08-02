# {py:mod}`fastvideo.pipelines.stages.input_validation`

```{py:module} fastvideo.pipelines.stages.input_validation
```

```{autodoc2-docstring} fastvideo.pipelines.stages.input_validation
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`InputValidationStage <fastvideo.pipelines.stages.input_validation.InputValidationStage>`
  - ```{autodoc2-docstring} fastvideo.pipelines.stages.input_validation.InputValidationStage
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`V <fastvideo.pipelines.stages.input_validation.V>`
  - ```{autodoc2-docstring} fastvideo.pipelines.stages.input_validation.V
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`logger <fastvideo.pipelines.stages.input_validation.logger>`
  - ```{autodoc2-docstring} fastvideo.pipelines.stages.input_validation.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} InputValidationStage
:canonical: fastvideo.pipelines.stages.input_validation.InputValidationStage

Bases: {py:obj}`fastvideo.pipelines.stages.base.PipelineStage`

```{autodoc2-docstring} fastvideo.pipelines.stages.input_validation.InputValidationStage
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} forward(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.pipeline_batch_info.ForwardBatch
:canonical: fastvideo.pipelines.stages.input_validation.InputValidationStage.forward

```{autodoc2-docstring} fastvideo.pipelines.stages.input_validation.InputValidationStage.forward
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} verify_input(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.stages.validators.VerificationResult
:canonical: fastvideo.pipelines.stages.input_validation.InputValidationStage.verify_input

```{autodoc2-docstring} fastvideo.pipelines.stages.input_validation.InputValidationStage.verify_input
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} verify_output(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.stages.validators.VerificationResult
:canonical: fastvideo.pipelines.stages.input_validation.InputValidationStage.verify_output

```{autodoc2-docstring} fastvideo.pipelines.stages.input_validation.InputValidationStage.verify_output
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} V
:canonical: fastvideo.pipelines.stages.input_validation.V
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.stages.input_validation.V
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:data} logger
:canonical: fastvideo.pipelines.stages.input_validation.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.pipelines.stages.input_validation.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
