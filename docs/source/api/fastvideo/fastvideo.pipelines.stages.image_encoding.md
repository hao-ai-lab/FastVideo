# {py:mod}`fastvideo.pipelines.stages.image_encoding`

```{py:module} fastvideo.pipelines.stages.image_encoding
```

```{autodoc2-docstring} fastvideo.pipelines.stages.image_encoding
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ImageEncodingStage <fastvideo.pipelines.stages.image_encoding.ImageEncodingStage>`
  - ```{autodoc2-docstring} fastvideo.pipelines.stages.image_encoding.ImageEncodingStage
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.pipelines.stages.image_encoding.logger>`
  - ```{autodoc2-docstring} fastvideo.pipelines.stages.image_encoding.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} ImageEncodingStage(image_encoder, image_processor)
:canonical: fastvideo.pipelines.stages.image_encoding.ImageEncodingStage

Bases: {py:obj}`fastvideo.pipelines.stages.base.PipelineStage`

```{autodoc2-docstring} fastvideo.pipelines.stages.image_encoding.ImageEncodingStage
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.pipelines.stages.image_encoding.ImageEncodingStage.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} forward(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.pipeline_batch_info.ForwardBatch
:canonical: fastvideo.pipelines.stages.image_encoding.ImageEncodingStage.forward

```{autodoc2-docstring} fastvideo.pipelines.stages.image_encoding.ImageEncodingStage.forward
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} verify_input(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.stages.validators.VerificationResult
:canonical: fastvideo.pipelines.stages.image_encoding.ImageEncodingStage.verify_input

```{autodoc2-docstring} fastvideo.pipelines.stages.image_encoding.ImageEncodingStage.verify_input
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} verify_output(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.stages.validators.VerificationResult
:canonical: fastvideo.pipelines.stages.image_encoding.ImageEncodingStage.verify_output

```{autodoc2-docstring} fastvideo.pipelines.stages.image_encoding.ImageEncodingStage.verify_output
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} logger
:canonical: fastvideo.pipelines.stages.image_encoding.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.pipelines.stages.image_encoding.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
