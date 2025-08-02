# {py:mod}`fastvideo.pipelines.lora_pipeline`

```{py:module} fastvideo.pipelines.lora_pipeline
```

```{autodoc2-docstring} fastvideo.pipelines.lora_pipeline
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LoRAPipeline <fastvideo.pipelines.lora_pipeline.LoRAPipeline>`
  - ```{autodoc2-docstring} fastvideo.pipelines.lora_pipeline.LoRAPipeline
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.pipelines.lora_pipeline.logger>`
  - ```{autodoc2-docstring} fastvideo.pipelines.lora_pipeline.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} LoRAPipeline(*args, **kwargs)
:canonical: fastvideo.pipelines.lora_pipeline.LoRAPipeline

Bases: {py:obj}`fastvideo.pipelines.composed_pipeline_base.ComposedPipelineBase`

```{autodoc2-docstring} fastvideo.pipelines.lora_pipeline.LoRAPipeline
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.pipelines.lora_pipeline.LoRAPipeline.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} convert_to_lora_layers() -> None
:canonical: fastvideo.pipelines.lora_pipeline.LoRAPipeline.convert_to_lora_layers

```{autodoc2-docstring} fastvideo.pipelines.lora_pipeline.LoRAPipeline.convert_to_lora_layers
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} cur_adapter_name
:canonical: fastvideo.pipelines.lora_pipeline.LoRAPipeline.cur_adapter_name
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} fastvideo.pipelines.lora_pipeline.LoRAPipeline.cur_adapter_name
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} cur_adapter_path
:canonical: fastvideo.pipelines.lora_pipeline.LoRAPipeline.cur_adapter_path
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} fastvideo.pipelines.lora_pipeline.LoRAPipeline.cur_adapter_path
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} device
:canonical: fastvideo.pipelines.lora_pipeline.LoRAPipeline.device
:type: torch.device
:value: >
   'get_local_torch_device(...)'

```{autodoc2-docstring} fastvideo.pipelines.lora_pipeline.LoRAPipeline.device
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} exclude_lora_layers
:canonical: fastvideo.pipelines.lora_pipeline.LoRAPipeline.exclude_lora_layers
:type: list[str]
:value: >
   []

```{autodoc2-docstring} fastvideo.pipelines.lora_pipeline.LoRAPipeline.exclude_lora_layers
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} fastvideo_args
:canonical: fastvideo.pipelines.lora_pipeline.LoRAPipeline.fastvideo_args
:type: fastvideo.fastvideo_args.FastVideoArgs | fastvideo.fastvideo_args.TrainingArgs
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.lora_pipeline.LoRAPipeline.fastvideo_args
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} is_target_layer(module_name: str) -> bool
:canonical: fastvideo.pipelines.lora_pipeline.LoRAPipeline.is_target_layer

```{autodoc2-docstring} fastvideo.pipelines.lora_pipeline.LoRAPipeline.is_target_layer
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} lora_adapters
:canonical: fastvideo.pipelines.lora_pipeline.LoRAPipeline.lora_adapters
:type: dict[str, dict[str, torch.Tensor]]
:value: >
   'defaultdict(...)'

```{autodoc2-docstring} fastvideo.pipelines.lora_pipeline.LoRAPipeline.lora_adapters
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} lora_alpha
:canonical: fastvideo.pipelines.lora_pipeline.LoRAPipeline.lora_alpha
:type: int | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.lora_pipeline.LoRAPipeline.lora_alpha
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} lora_initialized
:canonical: fastvideo.pipelines.lora_pipeline.LoRAPipeline.lora_initialized
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.pipelines.lora_pipeline.LoRAPipeline.lora_initialized
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} lora_layers
:canonical: fastvideo.pipelines.lora_pipeline.LoRAPipeline.lora_layers
:type: dict[str, fastvideo.layers.lora.linear.BaseLayerWithLoRA]
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.lora_pipeline.LoRAPipeline.lora_layers
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} lora_nickname
:canonical: fastvideo.pipelines.lora_pipeline.LoRAPipeline.lora_nickname
:type: str
:value: >
   'default'

```{autodoc2-docstring} fastvideo.pipelines.lora_pipeline.LoRAPipeline.lora_nickname
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} lora_path
:canonical: fastvideo.pipelines.lora_pipeline.LoRAPipeline.lora_path
:type: str | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.lora_pipeline.LoRAPipeline.lora_path
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} lora_rank
:canonical: fastvideo.pipelines.lora_pipeline.LoRAPipeline.lora_rank
:type: int | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.lora_pipeline.LoRAPipeline.lora_rank
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} lora_target_modules
:canonical: fastvideo.pipelines.lora_pipeline.LoRAPipeline.lora_target_modules
:type: list[str] | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.lora_pipeline.LoRAPipeline.lora_target_modules
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} set_lora_adapter(lora_nickname: str, lora_path: str | None = None)
:canonical: fastvideo.pipelines.lora_pipeline.LoRAPipeline.set_lora_adapter

```{autodoc2-docstring} fastvideo.pipelines.lora_pipeline.LoRAPipeline.set_lora_adapter
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} set_trainable() -> None
:canonical: fastvideo.pipelines.lora_pipeline.LoRAPipeline.set_trainable

```{autodoc2-docstring} fastvideo.pipelines.lora_pipeline.LoRAPipeline.set_trainable
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} logger
:canonical: fastvideo.pipelines.lora_pipeline.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.pipelines.lora_pipeline.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
