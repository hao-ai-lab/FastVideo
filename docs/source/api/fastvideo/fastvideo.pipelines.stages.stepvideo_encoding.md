# {py:mod}`fastvideo.pipelines.stages.stepvideo_encoding`

```{py:module} fastvideo.pipelines.stages.stepvideo_encoding
```

```{autodoc2-docstring} fastvideo.pipelines.stages.stepvideo_encoding
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`StepvideoPromptEncodingStage <fastvideo.pipelines.stages.stepvideo_encoding.StepvideoPromptEncodingStage>`
  - ```{autodoc2-docstring} fastvideo.pipelines.stages.stepvideo_encoding.StepvideoPromptEncodingStage
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.pipelines.stages.stepvideo_encoding.logger>`
  - ```{autodoc2-docstring} fastvideo.pipelines.stages.stepvideo_encoding.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} StepvideoPromptEncodingStage(stepllm, clip)
:canonical: fastvideo.pipelines.stages.stepvideo_encoding.StepvideoPromptEncodingStage

Bases: {py:obj}`fastvideo.pipelines.stages.base.PipelineStage`

```{autodoc2-docstring} fastvideo.pipelines.stages.stepvideo_encoding.StepvideoPromptEncodingStage
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.pipelines.stages.stepvideo_encoding.StepvideoPromptEncodingStage.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} forward(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args) -> fastvideo.pipelines.pipeline_batch_info.ForwardBatch
:canonical: fastvideo.pipelines.stages.stepvideo_encoding.StepvideoPromptEncodingStage.forward

````

````{py:method} verify_input(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.stages.validators.VerificationResult
:canonical: fastvideo.pipelines.stages.stepvideo_encoding.StepvideoPromptEncodingStage.verify_input

```{autodoc2-docstring} fastvideo.pipelines.stages.stepvideo_encoding.StepvideoPromptEncodingStage.verify_input
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} verify_output(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.stages.validators.VerificationResult
:canonical: fastvideo.pipelines.stages.stepvideo_encoding.StepvideoPromptEncodingStage.verify_output

```{autodoc2-docstring} fastvideo.pipelines.stages.stepvideo_encoding.StepvideoPromptEncodingStage.verify_output
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} logger
:canonical: fastvideo.pipelines.stages.stepvideo_encoding.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.pipelines.stages.stepvideo_encoding.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
