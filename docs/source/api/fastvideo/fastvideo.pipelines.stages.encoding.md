# {py:mod}`fastvideo.pipelines.stages.encoding`

```{py:module} fastvideo.pipelines.stages.encoding
```

```{autodoc2-docstring} fastvideo.pipelines.stages.encoding
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EncodingStage <fastvideo.pipelines.stages.encoding.EncodingStage>`
  - ```{autodoc2-docstring} fastvideo.pipelines.stages.encoding.EncodingStage
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.pipelines.stages.encoding.logger>`
  - ```{autodoc2-docstring} fastvideo.pipelines.stages.encoding.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} EncodingStage(vae: fastvideo.models.vaes.common.ParallelTiledVAE)
:canonical: fastvideo.pipelines.stages.encoding.EncodingStage

Bases: {py:obj}`fastvideo.pipelines.stages.base.PipelineStage`

```{autodoc2-docstring} fastvideo.pipelines.stages.encoding.EncodingStage
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.pipelines.stages.encoding.EncodingStage.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} forward(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.pipeline_batch_info.ForwardBatch
:canonical: fastvideo.pipelines.stages.encoding.EncodingStage.forward

```{autodoc2-docstring} fastvideo.pipelines.stages.encoding.EncodingStage.forward
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} preprocess(image: PIL.Image.Image, vae_scale_factor: int, height: int | None = None, width: int | None = None, resize_mode: str = 'default') -> torch.Tensor
:canonical: fastvideo.pipelines.stages.encoding.EncodingStage.preprocess

```{autodoc2-docstring} fastvideo.pipelines.stages.encoding.EncodingStage.preprocess
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} retrieve_latents(encoder_output: torch.Tensor, generator: torch.Generator | None = None, sample_mode: str = 'sample')
:canonical: fastvideo.pipelines.stages.encoding.EncodingStage.retrieve_latents

```{autodoc2-docstring} fastvideo.pipelines.stages.encoding.EncodingStage.retrieve_latents
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} verify_input(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.stages.validators.VerificationResult
:canonical: fastvideo.pipelines.stages.encoding.EncodingStage.verify_input

```{autodoc2-docstring} fastvideo.pipelines.stages.encoding.EncodingStage.verify_input
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} verify_output(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.stages.validators.VerificationResult
:canonical: fastvideo.pipelines.stages.encoding.EncodingStage.verify_output

```{autodoc2-docstring} fastvideo.pipelines.stages.encoding.EncodingStage.verify_output
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} logger
:canonical: fastvideo.pipelines.stages.encoding.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.pipelines.stages.encoding.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
