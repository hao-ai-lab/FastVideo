# {py:mod}`fastvideo.pipelines.basic.hunyuan.hunyuan_pipeline`

```{py:module} fastvideo.pipelines.basic.hunyuan.hunyuan_pipeline
```

```{autodoc2-docstring} fastvideo.pipelines.basic.hunyuan.hunyuan_pipeline
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HunyuanVideoPipeline <fastvideo.pipelines.basic.hunyuan.hunyuan_pipeline.HunyuanVideoPipeline>`
  -
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EntryClass <fastvideo.pipelines.basic.hunyuan.hunyuan_pipeline.EntryClass>`
  - ```{autodoc2-docstring} fastvideo.pipelines.basic.hunyuan.hunyuan_pipeline.EntryClass
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`logger <fastvideo.pipelines.basic.hunyuan.hunyuan_pipeline.logger>`
  - ```{autodoc2-docstring} fastvideo.pipelines.basic.hunyuan.hunyuan_pipeline.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

````{py:data} EntryClass
:canonical: fastvideo.pipelines.basic.hunyuan.hunyuan_pipeline.EntryClass
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.basic.hunyuan.hunyuan_pipeline.EntryClass
:parser: docs.source.autodoc2_docstring_parser
```

````

`````{py:class} HunyuanVideoPipeline(model_path: str, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs | fastvideo.fastvideo_args.TrainingArgs, required_config_modules: list[str] | None = None, loaded_modules: dict[str, torch.nn.Module] | None = None)
:canonical: fastvideo.pipelines.basic.hunyuan.hunyuan_pipeline.HunyuanVideoPipeline

Bases: {py:obj}`fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase`

````{py:method} create_pipeline_stages(fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs)
:canonical: fastvideo.pipelines.basic.hunyuan.hunyuan_pipeline.HunyuanVideoPipeline.create_pipeline_stages

```{autodoc2-docstring} fastvideo.pipelines.basic.hunyuan.hunyuan_pipeline.HunyuanVideoPipeline.create_pipeline_stages
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} logger
:canonical: fastvideo.pipelines.basic.hunyuan.hunyuan_pipeline.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.pipelines.basic.hunyuan.hunyuan_pipeline.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
