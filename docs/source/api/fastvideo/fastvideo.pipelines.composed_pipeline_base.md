# {py:mod}`fastvideo.pipelines.composed_pipeline_base`

```{py:module} fastvideo.pipelines.composed_pipeline_base
```

```{autodoc2-docstring} fastvideo.pipelines.composed_pipeline_base
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ComposedPipelineBase <fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase>`
  - ```{autodoc2-docstring} fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.pipelines.composed_pipeline_base.logger>`
  - ```{autodoc2-docstring} fastvideo.pipelines.composed_pipeline_base.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} ComposedPipelineBase(model_path: str, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs | fastvideo.fastvideo_args.TrainingArgs, required_config_modules: list[str] | None = None, loaded_modules: dict[str, torch.nn.Module] | None = None)
:canonical: fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} add_module(module_name: str, module: typing.Any)
:canonical: fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.add_module

```{autodoc2-docstring} fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.add_module
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} add_stage(stage_name: str, stage: fastvideo.pipelines.stages.PipelineStage)
:canonical: fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.add_stage

```{autodoc2-docstring} fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.add_stage
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} create_pipeline_stages(fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs)
:canonical: fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.create_pipeline_stages
:abstractmethod:

```{autodoc2-docstring} fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.create_pipeline_stages
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} create_training_stages(training_args: fastvideo.fastvideo_args.TrainingArgs)
:canonical: fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.create_training_stages
:abstractmethod:

```{autodoc2-docstring} fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.create_training_stages
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} fastvideo_args
:canonical: fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.fastvideo_args
:type: fastvideo.fastvideo_args.FastVideoArgs | fastvideo.fastvideo_args.TrainingArgs | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.fastvideo_args
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} forward(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.pipeline_batch_info.ForwardBatch
:canonical: fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.forward

```{autodoc2-docstring} fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.forward
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} from_pretrained(model_path: str, device: str | None = None, torch_dtype: torch.dtype | None = None, pipeline_config: str | fastvideo.configs.pipelines.PipelineConfig | None = None, args: argparse.Namespace | None = None, required_config_modules: list[str] | None = None, loaded_modules: dict[str, torch.nn.Module] | None = None, **kwargs) -> fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase
:canonical: fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.from_pretrained
:classmethod:

```{autodoc2-docstring} fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.from_pretrained
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_module(module_name: str, default_value: typing.Any = None) -> typing.Any
:canonical: fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.get_module

```{autodoc2-docstring} fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.get_module
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} initialize_pipeline(fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs)
:canonical: fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.initialize_pipeline

```{autodoc2-docstring} fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.initialize_pipeline
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} initialize_training_pipeline(training_args: fastvideo.fastvideo_args.TrainingArgs)
:canonical: fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.initialize_training_pipeline
:abstractmethod:

```{autodoc2-docstring} fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.initialize_training_pipeline
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} initialize_validation_pipeline(training_args: fastvideo.fastvideo_args.TrainingArgs)
:canonical: fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.initialize_validation_pipeline
:abstractmethod:

```{autodoc2-docstring} fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.initialize_validation_pipeline
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} is_video_pipeline
:canonical: fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.is_video_pipeline
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.is_video_pipeline
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} load_modules(fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs, loaded_modules: dict[str, torch.nn.Module] | None = None) -> dict[str, typing.Any]
:canonical: fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.load_modules

```{autodoc2-docstring} fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.load_modules
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} modules
:canonical: fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.modules
:type: dict[str, torch.nn.Module]
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.modules
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} post_init() -> None
:canonical: fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.post_init

```{autodoc2-docstring} fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.post_init
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} post_init_called
:canonical: fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.post_init_called
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.post_init_called
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:property} required_config_modules
:canonical: fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.required_config_modules
:type: list[str]

```{autodoc2-docstring} fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.required_config_modules
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} set_trainable() -> None
:canonical: fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.set_trainable

```{autodoc2-docstring} fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.set_trainable
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:property} stages
:canonical: fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.stages
:type: list[fastvideo.pipelines.stages.PipelineStage]

```{autodoc2-docstring} fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.stages
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} train() -> None
:canonical: fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.train
:abstractmethod:

```{autodoc2-docstring} fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.train
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} training_args
:canonical: fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.training_args
:type: fastvideo.fastvideo_args.TrainingArgs | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase.training_args
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} logger
:canonical: fastvideo.pipelines.composed_pipeline_base.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.pipelines.composed_pipeline_base.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
