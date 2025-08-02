# {py:mod}`fastvideo.pipelines.basic.wan.wan_i2v_pipeline`

```{py:module} fastvideo.pipelines.basic.wan.wan_i2v_pipeline
```

```{autodoc2-docstring} fastvideo.pipelines.basic.wan.wan_i2v_pipeline
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`WanImageToVideoPipeline <fastvideo.pipelines.basic.wan.wan_i2v_pipeline.WanImageToVideoPipeline>`
  -
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EntryClass <fastvideo.pipelines.basic.wan.wan_i2v_pipeline.EntryClass>`
  - ```{autodoc2-docstring} fastvideo.pipelines.basic.wan.wan_i2v_pipeline.EntryClass
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`logger <fastvideo.pipelines.basic.wan.wan_i2v_pipeline.logger>`
  - ```{autodoc2-docstring} fastvideo.pipelines.basic.wan.wan_i2v_pipeline.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

````{py:data} EntryClass
:canonical: fastvideo.pipelines.basic.wan.wan_i2v_pipeline.EntryClass
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.basic.wan.wan_i2v_pipeline.EntryClass
:parser: docs.source.autodoc2_docstring_parser
```

````

`````{py:class} WanImageToVideoPipeline(*args, **kwargs)
:canonical: fastvideo.pipelines.basic.wan.wan_i2v_pipeline.WanImageToVideoPipeline

Bases: {py:obj}`fastvideo.pipelines.lora_pipeline.LoRAPipeline`, {py:obj}`fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase`

````{py:method} create_pipeline_stages(fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs)
:canonical: fastvideo.pipelines.basic.wan.wan_i2v_pipeline.WanImageToVideoPipeline.create_pipeline_stages

```{autodoc2-docstring} fastvideo.pipelines.basic.wan.wan_i2v_pipeline.WanImageToVideoPipeline.create_pipeline_stages
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} initialize_pipeline(fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs)
:canonical: fastvideo.pipelines.basic.wan.wan_i2v_pipeline.WanImageToVideoPipeline.initialize_pipeline

````

`````

````{py:data} logger
:canonical: fastvideo.pipelines.basic.wan.wan_i2v_pipeline.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.pipelines.basic.wan.wan_i2v_pipeline.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
