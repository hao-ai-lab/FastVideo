# {py:mod}`fastvideo.pipelines.stages.decoding`

```{py:module} fastvideo.pipelines.stages.decoding
```

```{autodoc2-docstring} fastvideo.pipelines.stages.decoding
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DecodingStage <fastvideo.pipelines.stages.decoding.DecodingStage>`
  - ```{autodoc2-docstring} fastvideo.pipelines.stages.decoding.DecodingStage
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.pipelines.stages.decoding.logger>`
  - ```{autodoc2-docstring} fastvideo.pipelines.stages.decoding.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} DecodingStage(vae, pipeline=None)
:canonical: fastvideo.pipelines.stages.decoding.DecodingStage

Bases: {py:obj}`fastvideo.pipelines.stages.base.PipelineStage`

```{autodoc2-docstring} fastvideo.pipelines.stages.decoding.DecodingStage
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.pipelines.stages.decoding.DecodingStage.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} forward(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.pipeline_batch_info.ForwardBatch
:canonical: fastvideo.pipelines.stages.decoding.DecodingStage.forward

```{autodoc2-docstring} fastvideo.pipelines.stages.decoding.DecodingStage.forward
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} verify_input(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.stages.validators.VerificationResult
:canonical: fastvideo.pipelines.stages.decoding.DecodingStage.verify_input

```{autodoc2-docstring} fastvideo.pipelines.stages.decoding.DecodingStage.verify_input
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} verify_output(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.stages.validators.VerificationResult
:canonical: fastvideo.pipelines.stages.decoding.DecodingStage.verify_output

```{autodoc2-docstring} fastvideo.pipelines.stages.decoding.DecodingStage.verify_output
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} logger
:canonical: fastvideo.pipelines.stages.decoding.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.pipelines.stages.decoding.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
