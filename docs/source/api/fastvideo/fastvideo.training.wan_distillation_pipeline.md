# {py:mod}`fastvideo.training.wan_distillation_pipeline`

```{py:module} fastvideo.training.wan_distillation_pipeline
```

```{autodoc2-docstring} fastvideo.training.wan_distillation_pipeline
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`WanDistillationPipeline <fastvideo.training.wan_distillation_pipeline.WanDistillationPipeline>`
  - ```{autodoc2-docstring} fastvideo.training.wan_distillation_pipeline.WanDistillationPipeline
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`main <fastvideo.training.wan_distillation_pipeline.main>`
  - ```{autodoc2-docstring} fastvideo.training.wan_distillation_pipeline.main
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.training.wan_distillation_pipeline.logger>`
  - ```{autodoc2-docstring} fastvideo.training.wan_distillation_pipeline.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`vsa_available <fastvideo.training.wan_distillation_pipeline.vsa_available>`
  - ```{autodoc2-docstring} fastvideo.training.wan_distillation_pipeline.vsa_available
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} WanDistillationPipeline(model_path: str, fastvideo_args: fastvideo.fastvideo_args.TrainingArgs, required_config_modules: list[str] | None = None, loaded_modules: dict[str, torch.nn.Module] | None = None)
:canonical: fastvideo.training.wan_distillation_pipeline.WanDistillationPipeline

Bases: {py:obj}`fastvideo.training.distillation_pipeline.DistillationPipeline`

```{autodoc2-docstring} fastvideo.training.wan_distillation_pipeline.WanDistillationPipeline
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.training.wan_distillation_pipeline.WanDistillationPipeline.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} create_training_stages(training_args: fastvideo.fastvideo_args.TrainingArgs)
:canonical: fastvideo.training.wan_distillation_pipeline.WanDistillationPipeline.create_training_stages

```{autodoc2-docstring} fastvideo.training.wan_distillation_pipeline.WanDistillationPipeline.create_training_stages
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} initialize_pipeline(fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs)
:canonical: fastvideo.training.wan_distillation_pipeline.WanDistillationPipeline.initialize_pipeline

```{autodoc2-docstring} fastvideo.training.wan_distillation_pipeline.WanDistillationPipeline.initialize_pipeline
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} initialize_validation_pipeline(training_args: fastvideo.fastvideo_args.TrainingArgs)
:canonical: fastvideo.training.wan_distillation_pipeline.WanDistillationPipeline.initialize_validation_pipeline

````

`````

````{py:data} logger
:canonical: fastvideo.training.wan_distillation_pipeline.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.training.wan_distillation_pipeline.logger
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:function} main(args) -> None
:canonical: fastvideo.training.wan_distillation_pipeline.main

```{autodoc2-docstring} fastvideo.training.wan_distillation_pipeline.main
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:data} vsa_available
:canonical: fastvideo.training.wan_distillation_pipeline.vsa_available
:value: >
   'is_vsa_available(...)'

```{autodoc2-docstring} fastvideo.training.wan_distillation_pipeline.vsa_available
:parser: docs.source.autodoc2_docstring_parser
```

````
