# {py:mod}`fastvideo.training.distillation_pipeline`

```{py:module} fastvideo.training.distillation_pipeline
```

```{autodoc2-docstring} fastvideo.training.distillation_pipeline
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DistillationPipeline <fastvideo.training.distillation_pipeline.DistillationPipeline>`
  - ```{autodoc2-docstring} fastvideo.training.distillation_pipeline.DistillationPipeline
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.training.distillation_pipeline.logger>`
  - ```{autodoc2-docstring} fastvideo.training.distillation_pipeline.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`vsa_available <fastvideo.training.distillation_pipeline.vsa_available>`
  - ```{autodoc2-docstring} fastvideo.training.distillation_pipeline.vsa_available
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} DistillationPipeline(model_path: str, fastvideo_args: fastvideo.fastvideo_args.TrainingArgs, required_config_modules: list[str] | None = None, loaded_modules: dict[str, torch.nn.Module] | None = None)
:canonical: fastvideo.training.distillation_pipeline.DistillationPipeline

Bases: {py:obj}`fastvideo.training.training_pipeline.TrainingPipeline`

```{autodoc2-docstring} fastvideo.training.distillation_pipeline.DistillationPipeline
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.training.distillation_pipeline.DistillationPipeline.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} create_pipeline_stages(fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs)
:canonical: fastvideo.training.distillation_pipeline.DistillationPipeline.create_pipeline_stages

````

````{py:attribute} current_epoch
:canonical: fastvideo.training.distillation_pipeline.DistillationPipeline.current_epoch
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.training.distillation_pipeline.DistillationPipeline.current_epoch
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} current_trainstep
:canonical: fastvideo.training.distillation_pipeline.DistillationPipeline.current_trainstep
:type: int
:value: >
   None

```{autodoc2-docstring} fastvideo.training.distillation_pipeline.DistillationPipeline.current_trainstep
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} faker_score_forward(training_batch: fastvideo.pipelines.TrainingBatch) -> tuple[fastvideo.pipelines.TrainingBatch, torch.Tensor]
:canonical: fastvideo.training.distillation_pipeline.DistillationPipeline.faker_score_forward

```{autodoc2-docstring} fastvideo.training.distillation_pipeline.DistillationPipeline.faker_score_forward
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} init_steps
:canonical: fastvideo.training.distillation_pipeline.DistillationPipeline.init_steps
:type: int
:value: >
   None

```{autodoc2-docstring} fastvideo.training.distillation_pipeline.DistillationPipeline.init_steps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} initialize_training_pipeline(training_args: fastvideo.fastvideo_args.TrainingArgs)
:canonical: fastvideo.training.distillation_pipeline.DistillationPipeline.initialize_training_pipeline

```{autodoc2-docstring} fastvideo.training.distillation_pipeline.DistillationPipeline.initialize_training_pipeline
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} initialize_validation_pipeline(training_args: fastvideo.fastvideo_args.TrainingArgs)
:canonical: fastvideo.training.distillation_pipeline.DistillationPipeline.initialize_validation_pipeline
:abstractmethod:

```{autodoc2-docstring} fastvideo.training.distillation_pipeline.DistillationPipeline.initialize_validation_pipeline
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} train() -> None
:canonical: fastvideo.training.distillation_pipeline.DistillationPipeline.train

```{autodoc2-docstring} fastvideo.training.distillation_pipeline.DistillationPipeline.train
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} train_dataloader
:canonical: fastvideo.training.distillation_pipeline.DistillationPipeline.train_dataloader
:type: torchdata.stateful_dataloader.StatefulDataLoader
:value: >
   None

```{autodoc2-docstring} fastvideo.training.distillation_pipeline.DistillationPipeline.train_dataloader
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} train_loader_iter
:canonical: fastvideo.training.distillation_pipeline.DistillationPipeline.train_loader_iter
:type: collections.abc.Iterator[dict[str, typing.Any]]
:value: >
   None

```{autodoc2-docstring} fastvideo.training.distillation_pipeline.DistillationPipeline.train_loader_iter
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} train_one_step(training_batch: fastvideo.pipelines.TrainingBatch) -> fastvideo.pipelines.TrainingBatch
:canonical: fastvideo.training.distillation_pipeline.DistillationPipeline.train_one_step

```{autodoc2-docstring} fastvideo.training.distillation_pipeline.DistillationPipeline.train_one_step
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} validation_pipeline
:canonical: fastvideo.training.distillation_pipeline.DistillationPipeline.validation_pipeline
:type: fastvideo.pipelines.ComposedPipelineBase
:value: >
   None

```{autodoc2-docstring} fastvideo.training.distillation_pipeline.DistillationPipeline.validation_pipeline
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} video_latent_shape
:canonical: fastvideo.training.distillation_pipeline.DistillationPipeline.video_latent_shape
:type: tuple[int, ...]
:value: >
   None

```{autodoc2-docstring} fastvideo.training.distillation_pipeline.DistillationPipeline.video_latent_shape
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} video_latent_shape_sp
:canonical: fastvideo.training.distillation_pipeline.DistillationPipeline.video_latent_shape_sp
:type: tuple[int, ...]
:value: >
   None

```{autodoc2-docstring} fastvideo.training.distillation_pipeline.DistillationPipeline.video_latent_shape_sp
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} visualize_intermediate_latents(training_batch: fastvideo.pipelines.TrainingBatch, training_args: fastvideo.fastvideo_args.TrainingArgs, step: int)
:canonical: fastvideo.training.distillation_pipeline.DistillationPipeline.visualize_intermediate_latents

```{autodoc2-docstring} fastvideo.training.distillation_pipeline.DistillationPipeline.visualize_intermediate_latents
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} logger
:canonical: fastvideo.training.distillation_pipeline.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.training.distillation_pipeline.logger
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:data} vsa_available
:canonical: fastvideo.training.distillation_pipeline.vsa_available
:value: >
   'is_vsa_available(...)'

```{autodoc2-docstring} fastvideo.training.distillation_pipeline.vsa_available
:parser: docs.source.autodoc2_docstring_parser
```

````
