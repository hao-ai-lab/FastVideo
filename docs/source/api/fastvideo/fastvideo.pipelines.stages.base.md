# {py:mod}`fastvideo.pipelines.stages.base`

```{py:module} fastvideo.pipelines.stages.base
```

```{autodoc2-docstring} fastvideo.pipelines.stages.base
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PipelineStage <fastvideo.pipelines.stages.base.PipelineStage>`
  - ```{autodoc2-docstring} fastvideo.pipelines.stages.base.PipelineStage
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.pipelines.stages.base.logger>`
  - ```{autodoc2-docstring} fastvideo.pipelines.stages.base.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} PipelineStage
:canonical: fastvideo.pipelines.stages.base.PipelineStage

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} fastvideo.pipelines.stages.base.PipelineStage
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} backward(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.pipeline_batch_info.ForwardBatch
:canonical: fastvideo.pipelines.stages.base.PipelineStage.backward
:abstractmethod:

```{autodoc2-docstring} fastvideo.pipelines.stages.base.PipelineStage.backward
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:property} device
:canonical: fastvideo.pipelines.stages.base.PipelineStage.device
:type: torch.device

```{autodoc2-docstring} fastvideo.pipelines.stages.base.PipelineStage.device
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} forward(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.pipeline_batch_info.ForwardBatch
:canonical: fastvideo.pipelines.stages.base.PipelineStage.forward
:abstractmethod:

```{autodoc2-docstring} fastvideo.pipelines.stages.base.PipelineStage.forward
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} set_logging(enable: bool)
:canonical: fastvideo.pipelines.stages.base.PipelineStage.set_logging

```{autodoc2-docstring} fastvideo.pipelines.stages.base.PipelineStage.set_logging
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} verify_input(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.stages.validators.VerificationResult
:canonical: fastvideo.pipelines.stages.base.PipelineStage.verify_input

```{autodoc2-docstring} fastvideo.pipelines.stages.base.PipelineStage.verify_input
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} verify_output(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.stages.validators.VerificationResult
:canonical: fastvideo.pipelines.stages.base.PipelineStage.verify_output

```{autodoc2-docstring} fastvideo.pipelines.stages.base.PipelineStage.verify_output
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:exception} StageVerificationError()
:canonical: fastvideo.pipelines.stages.base.StageVerificationError

Bases: {py:obj}`Exception`

```{autodoc2-docstring} fastvideo.pipelines.stages.base.StageVerificationError
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.pipelines.stages.base.StageVerificationError.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:data} logger
:canonical: fastvideo.pipelines.stages.base.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.pipelines.stages.base.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
