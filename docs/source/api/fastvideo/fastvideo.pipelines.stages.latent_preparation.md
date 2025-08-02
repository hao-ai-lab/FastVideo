# {py:mod}`fastvideo.pipelines.stages.latent_preparation`

```{py:module} fastvideo.pipelines.stages.latent_preparation
```

```{autodoc2-docstring} fastvideo.pipelines.stages.latent_preparation
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LatentPreparationStage <fastvideo.pipelines.stages.latent_preparation.LatentPreparationStage>`
  - ```{autodoc2-docstring} fastvideo.pipelines.stages.latent_preparation.LatentPreparationStage
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.pipelines.stages.latent_preparation.logger>`
  - ```{autodoc2-docstring} fastvideo.pipelines.stages.latent_preparation.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} LatentPreparationStage(scheduler, transformer)
:canonical: fastvideo.pipelines.stages.latent_preparation.LatentPreparationStage

Bases: {py:obj}`fastvideo.pipelines.stages.base.PipelineStage`

```{autodoc2-docstring} fastvideo.pipelines.stages.latent_preparation.LatentPreparationStage
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.pipelines.stages.latent_preparation.LatentPreparationStage.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} adjust_video_length(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> int
:canonical: fastvideo.pipelines.stages.latent_preparation.LatentPreparationStage.adjust_video_length

```{autodoc2-docstring} fastvideo.pipelines.stages.latent_preparation.LatentPreparationStage.adjust_video_length
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} forward(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.pipeline_batch_info.ForwardBatch
:canonical: fastvideo.pipelines.stages.latent_preparation.LatentPreparationStage.forward

```{autodoc2-docstring} fastvideo.pipelines.stages.latent_preparation.LatentPreparationStage.forward
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} verify_input(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.stages.validators.VerificationResult
:canonical: fastvideo.pipelines.stages.latent_preparation.LatentPreparationStage.verify_input

```{autodoc2-docstring} fastvideo.pipelines.stages.latent_preparation.LatentPreparationStage.verify_input
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} verify_output(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.stages.validators.VerificationResult
:canonical: fastvideo.pipelines.stages.latent_preparation.LatentPreparationStage.verify_output

```{autodoc2-docstring} fastvideo.pipelines.stages.latent_preparation.LatentPreparationStage.verify_output
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} logger
:canonical: fastvideo.pipelines.stages.latent_preparation.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.pipelines.stages.latent_preparation.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
