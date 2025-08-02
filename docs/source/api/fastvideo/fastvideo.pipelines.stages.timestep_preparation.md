# {py:mod}`fastvideo.pipelines.stages.timestep_preparation`

```{py:module} fastvideo.pipelines.stages.timestep_preparation
```

```{autodoc2-docstring} fastvideo.pipelines.stages.timestep_preparation
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TimestepPreparationStage <fastvideo.pipelines.stages.timestep_preparation.TimestepPreparationStage>`
  - ```{autodoc2-docstring} fastvideo.pipelines.stages.timestep_preparation.TimestepPreparationStage
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.pipelines.stages.timestep_preparation.logger>`
  - ```{autodoc2-docstring} fastvideo.pipelines.stages.timestep_preparation.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} TimestepPreparationStage(scheduler)
:canonical: fastvideo.pipelines.stages.timestep_preparation.TimestepPreparationStage

Bases: {py:obj}`fastvideo.pipelines.stages.base.PipelineStage`

```{autodoc2-docstring} fastvideo.pipelines.stages.timestep_preparation.TimestepPreparationStage
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.pipelines.stages.timestep_preparation.TimestepPreparationStage.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} forward(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.pipeline_batch_info.ForwardBatch
:canonical: fastvideo.pipelines.stages.timestep_preparation.TimestepPreparationStage.forward

```{autodoc2-docstring} fastvideo.pipelines.stages.timestep_preparation.TimestepPreparationStage.forward
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} verify_input(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.stages.validators.VerificationResult
:canonical: fastvideo.pipelines.stages.timestep_preparation.TimestepPreparationStage.verify_input

```{autodoc2-docstring} fastvideo.pipelines.stages.timestep_preparation.TimestepPreparationStage.verify_input
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} verify_output(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.stages.validators.VerificationResult
:canonical: fastvideo.pipelines.stages.timestep_preparation.TimestepPreparationStage.verify_output

```{autodoc2-docstring} fastvideo.pipelines.stages.timestep_preparation.TimestepPreparationStage.verify_output
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} logger
:canonical: fastvideo.pipelines.stages.timestep_preparation.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.pipelines.stages.timestep_preparation.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
