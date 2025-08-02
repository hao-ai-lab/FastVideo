# {py:mod}`fastvideo.training.training_pipeline`

```{py:module} fastvideo.training.training_pipeline
```

```{autodoc2-docstring} fastvideo.training.training_pipeline
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TrainingPipeline <fastvideo.training.training_pipeline.TrainingPipeline>`
  - ```{autodoc2-docstring} fastvideo.training.training_pipeline.TrainingPipeline
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.training.training_pipeline.logger>`
  - ```{autodoc2-docstring} fastvideo.training.training_pipeline.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`vsa_available <fastvideo.training.training_pipeline.vsa_available>`
  - ```{autodoc2-docstring} fastvideo.training.training_pipeline.vsa_available
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} TrainingPipeline(model_path: str, fastvideo_args: fastvideo.fastvideo_args.TrainingArgs, required_config_modules: list[str] | None = None, loaded_modules: dict[str, torch.nn.Module] | None = None)
:canonical: fastvideo.training.training_pipeline.TrainingPipeline

Bases: {py:obj}`fastvideo.pipelines.LoRAPipeline`, {py:obj}`abc.ABC`

```{autodoc2-docstring} fastvideo.training.training_pipeline.TrainingPipeline
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.training.training_pipeline.TrainingPipeline.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} create_pipeline_stages(fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs)
:canonical: fastvideo.training.training_pipeline.TrainingPipeline.create_pipeline_stages

````

````{py:attribute} current_epoch
:canonical: fastvideo.training.training_pipeline.TrainingPipeline.current_epoch
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.training.training_pipeline.TrainingPipeline.current_epoch
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} initialize_training_pipeline(training_args: fastvideo.fastvideo_args.TrainingArgs)
:canonical: fastvideo.training.training_pipeline.TrainingPipeline.initialize_training_pipeline

```{autodoc2-docstring} fastvideo.training.training_pipeline.TrainingPipeline.initialize_training_pipeline
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} initialize_validation_pipeline(training_args: fastvideo.fastvideo_args.TrainingArgs)
:canonical: fastvideo.training.training_pipeline.TrainingPipeline.initialize_validation_pipeline
:abstractmethod:

```{autodoc2-docstring} fastvideo.training.training_pipeline.TrainingPipeline.initialize_validation_pipeline
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} set_schemas() -> None
:canonical: fastvideo.training.training_pipeline.TrainingPipeline.set_schemas

```{autodoc2-docstring} fastvideo.training.training_pipeline.TrainingPipeline.set_schemas
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} train() -> None
:canonical: fastvideo.training.training_pipeline.TrainingPipeline.train

```{autodoc2-docstring} fastvideo.training.training_pipeline.TrainingPipeline.train
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} train_dataloader
:canonical: fastvideo.training.training_pipeline.TrainingPipeline.train_dataloader
:type: torchdata.stateful_dataloader.StatefulDataLoader
:value: >
   None

```{autodoc2-docstring} fastvideo.training.training_pipeline.TrainingPipeline.train_dataloader
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} train_loader_iter
:canonical: fastvideo.training.training_pipeline.TrainingPipeline.train_loader_iter
:type: collections.abc.Iterator[dict[str, typing.Any]]
:value: >
   None

```{autodoc2-docstring} fastvideo.training.training_pipeline.TrainingPipeline.train_loader_iter
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} train_one_step(training_batch: fastvideo.pipelines.TrainingBatch) -> fastvideo.pipelines.TrainingBatch
:canonical: fastvideo.training.training_pipeline.TrainingPipeline.train_one_step

```{autodoc2-docstring} fastvideo.training.training_pipeline.TrainingPipeline.train_one_step
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} validation_pipeline
:canonical: fastvideo.training.training_pipeline.TrainingPipeline.validation_pipeline
:type: fastvideo.pipelines.ComposedPipelineBase
:value: >
   None

```{autodoc2-docstring} fastvideo.training.training_pipeline.TrainingPipeline.validation_pipeline
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} logger
:canonical: fastvideo.training.training_pipeline.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.training.training_pipeline.logger
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:data} vsa_available
:canonical: fastvideo.training.training_pipeline.vsa_available
:value: >
   'is_vsa_available(...)'

```{autodoc2-docstring} fastvideo.training.training_pipeline.vsa_available
:parser: docs.source.autodoc2_docstring_parser
```

````
