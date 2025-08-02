# {py:mod}`fastvideo.forward_context`

```{py:module} fastvideo.forward_context
```

```{autodoc2-docstring} fastvideo.forward_context
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ForwardContext <fastvideo.forward_context.ForwardContext>`
  - ```{autodoc2-docstring} fastvideo.forward_context.ForwardContext
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_forward_context <fastvideo.forward_context.get_forward_context>`
  - ```{autodoc2-docstring} fastvideo.forward_context.get_forward_context
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`set_forward_context <fastvideo.forward_context.set_forward_context>`
  - ```{autodoc2-docstring} fastvideo.forward_context.set_forward_context
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`batchsize_forward_time <fastvideo.forward_context.batchsize_forward_time>`
  - ```{autodoc2-docstring} fastvideo.forward_context.batchsize_forward_time
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`batchsize_logging_interval <fastvideo.forward_context.batchsize_logging_interval>`
  - ```{autodoc2-docstring} fastvideo.forward_context.batchsize_logging_interval
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`forward_start_time <fastvideo.forward_context.forward_start_time>`
  - ```{autodoc2-docstring} fastvideo.forward_context.forward_start_time
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`last_logging_time <fastvideo.forward_context.last_logging_time>`
  - ```{autodoc2-docstring} fastvideo.forward_context.last_logging_time
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`logger <fastvideo.forward_context.logger>`
  - ```{autodoc2-docstring} fastvideo.forward_context.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`track_batchsize <fastvideo.forward_context.track_batchsize>`
  - ```{autodoc2-docstring} fastvideo.forward_context.track_batchsize
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} ForwardContext
:canonical: fastvideo.forward_context.ForwardContext

```{autodoc2-docstring} fastvideo.forward_context.ForwardContext
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} attn_metadata
:canonical: fastvideo.forward_context.ForwardContext.attn_metadata
:type: fastvideo.attention.AttentionMetadata
:value: >
   None

```{autodoc2-docstring} fastvideo.forward_context.ForwardContext.attn_metadata
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} current_timestep
:canonical: fastvideo.forward_context.ForwardContext.current_timestep
:type: int
:value: >
   None

```{autodoc2-docstring} fastvideo.forward_context.ForwardContext.current_timestep
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} forward_batch
:canonical: fastvideo.forward_context.ForwardContext.forward_batch
:type: typing.Optional[fastvideo.pipelines.ForwardBatch]
:value: >
   None

```{autodoc2-docstring} fastvideo.forward_context.ForwardContext.forward_batch
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} batchsize_forward_time
:canonical: fastvideo.forward_context.batchsize_forward_time
:type: collections.defaultdict
:value: >
   'defaultdict(...)'

```{autodoc2-docstring} fastvideo.forward_context.batchsize_forward_time
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:data} batchsize_logging_interval
:canonical: fastvideo.forward_context.batchsize_logging_interval
:type: float
:value: >
   1000

```{autodoc2-docstring} fastvideo.forward_context.batchsize_logging_interval
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:data} forward_start_time
:canonical: fastvideo.forward_context.forward_start_time
:type: float
:value: >
   0

```{autodoc2-docstring} fastvideo.forward_context.forward_start_time
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:function} get_forward_context() -> fastvideo.forward_context.ForwardContext
:canonical: fastvideo.forward_context.get_forward_context

```{autodoc2-docstring} fastvideo.forward_context.get_forward_context
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:data} last_logging_time
:canonical: fastvideo.forward_context.last_logging_time
:type: float
:value: >
   0

```{autodoc2-docstring} fastvideo.forward_context.last_logging_time
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:data} logger
:canonical: fastvideo.forward_context.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.forward_context.logger
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:function} set_forward_context(current_timestep, attn_metadata, forward_batch: typing.Optional[fastvideo.pipelines.ForwardBatch] = None)
:canonical: fastvideo.forward_context.set_forward_context

```{autodoc2-docstring} fastvideo.forward_context.set_forward_context
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:data} track_batchsize
:canonical: fastvideo.forward_context.track_batchsize
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.forward_context.track_batchsize
:parser: docs.source.autodoc2_docstring_parser
```

````
