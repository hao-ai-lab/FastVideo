# {py:mod}`fastvideo.pipelines.stages.text_encoding`

```{py:module} fastvideo.pipelines.stages.text_encoding
```

```{autodoc2-docstring} fastvideo.pipelines.stages.text_encoding
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TextEncodingStage <fastvideo.pipelines.stages.text_encoding.TextEncodingStage>`
  - ```{autodoc2-docstring} fastvideo.pipelines.stages.text_encoding.TextEncodingStage
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.pipelines.stages.text_encoding.logger>`
  - ```{autodoc2-docstring} fastvideo.pipelines.stages.text_encoding.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} TextEncodingStage(text_encoders, tokenizers)
:canonical: fastvideo.pipelines.stages.text_encoding.TextEncodingStage

Bases: {py:obj}`fastvideo.pipelines.stages.base.PipelineStage`

```{autodoc2-docstring} fastvideo.pipelines.stages.text_encoding.TextEncodingStage
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.pipelines.stages.text_encoding.TextEncodingStage.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} forward(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.pipeline_batch_info.ForwardBatch
:canonical: fastvideo.pipelines.stages.text_encoding.TextEncodingStage.forward

```{autodoc2-docstring} fastvideo.pipelines.stages.text_encoding.TextEncodingStage.forward
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} verify_input(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.stages.validators.VerificationResult
:canonical: fastvideo.pipelines.stages.text_encoding.TextEncodingStage.verify_input

```{autodoc2-docstring} fastvideo.pipelines.stages.text_encoding.TextEncodingStage.verify_input
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} verify_output(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.stages.validators.VerificationResult
:canonical: fastvideo.pipelines.stages.text_encoding.TextEncodingStage.verify_output

```{autodoc2-docstring} fastvideo.pipelines.stages.text_encoding.TextEncodingStage.verify_output
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} logger
:canonical: fastvideo.pipelines.stages.text_encoding.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.pipelines.stages.text_encoding.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
