# {py:mod}`fastvideo.pipelines.basic.stepvideo.stepvideo_pipeline`

```{py:module} fastvideo.pipelines.basic.stepvideo.stepvideo_pipeline
```

```{autodoc2-docstring} fastvideo.pipelines.basic.stepvideo.stepvideo_pipeline
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`StepVideoPipeline <fastvideo.pipelines.basic.stepvideo.stepvideo_pipeline.StepVideoPipeline>`
  -
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EntryClass <fastvideo.pipelines.basic.stepvideo.stepvideo_pipeline.EntryClass>`
  - ```{autodoc2-docstring} fastvideo.pipelines.basic.stepvideo.stepvideo_pipeline.EntryClass
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`logger <fastvideo.pipelines.basic.stepvideo.stepvideo_pipeline.logger>`
  - ```{autodoc2-docstring} fastvideo.pipelines.basic.stepvideo.stepvideo_pipeline.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

````{py:data} EntryClass
:canonical: fastvideo.pipelines.basic.stepvideo.stepvideo_pipeline.EntryClass
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.basic.stepvideo.stepvideo_pipeline.EntryClass
:parser: docs.source.autodoc2_docstring_parser
```

````

`````{py:class} StepVideoPipeline(*args, **kwargs)
:canonical: fastvideo.pipelines.basic.stepvideo.stepvideo_pipeline.StepVideoPipeline

Bases: {py:obj}`fastvideo.pipelines.lora_pipeline.LoRAPipeline`, {py:obj}`fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase`

````{py:method} build_clip(model_dir, device) -> fastvideo.models.encoders.bert.HunyuanClip
:canonical: fastvideo.pipelines.basic.stepvideo.stepvideo_pipeline.StepVideoPipeline.build_clip

```{autodoc2-docstring} fastvideo.pipelines.basic.stepvideo.stepvideo_pipeline.StepVideoPipeline.build_clip
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} build_llm(model_dir, device) -> torch.nn.Module
:canonical: fastvideo.pipelines.basic.stepvideo.stepvideo_pipeline.StepVideoPipeline.build_llm

```{autodoc2-docstring} fastvideo.pipelines.basic.stepvideo.stepvideo_pipeline.StepVideoPipeline.build_llm
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} create_pipeline_stages(fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs)
:canonical: fastvideo.pipelines.basic.stepvideo.stepvideo_pipeline.StepVideoPipeline.create_pipeline_stages

```{autodoc2-docstring} fastvideo.pipelines.basic.stepvideo.stepvideo_pipeline.StepVideoPipeline.create_pipeline_stages
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} initialize_pipeline(fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs)
:canonical: fastvideo.pipelines.basic.stepvideo.stepvideo_pipeline.StepVideoPipeline.initialize_pipeline

```{autodoc2-docstring} fastvideo.pipelines.basic.stepvideo.stepvideo_pipeline.StepVideoPipeline.initialize_pipeline
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} load_modules(fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> dict[str, typing.Any]
:canonical: fastvideo.pipelines.basic.stepvideo.stepvideo_pipeline.StepVideoPipeline.load_modules

```{autodoc2-docstring} fastvideo.pipelines.basic.stepvideo.stepvideo_pipeline.StepVideoPipeline.load_modules
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} logger
:canonical: fastvideo.pipelines.basic.stepvideo.stepvideo_pipeline.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.pipelines.basic.stepvideo.stepvideo_pipeline.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
