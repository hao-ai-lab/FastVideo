# {py:mod}`fastvideo.worker.executor`

```{py:module} fastvideo.worker.executor
```

```{autodoc2-docstring} fastvideo.worker.executor
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Executor <fastvideo.worker.executor.Executor>`
  -
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.worker.executor.logger>`
  - ```{autodoc2-docstring} fastvideo.worker.executor.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} Executor(fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs)
:canonical: fastvideo.worker.executor.Executor

Bases: {py:obj}`abc.ABC`

````{py:method} collective_rpc(method: str | collections.abc.Callable[..., fastvideo.worker.executor._R], timeout: float | None = None, args: tuple = (), kwargs: dict[str, typing.Any] | None = None) -> list[fastvideo.worker.executor._R]
:canonical: fastvideo.worker.executor.Executor.collective_rpc
:abstractmethod:

```{autodoc2-docstring} fastvideo.worker.executor.Executor.collective_rpc
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} execute_forward(forward_batch: fastvideo.pipelines.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.ForwardBatch
:canonical: fastvideo.worker.executor.Executor.execute_forward

```{autodoc2-docstring} fastvideo.worker.executor.Executor.execute_forward
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_class(fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> type[fastvideo.worker.executor.Executor]
:canonical: fastvideo.worker.executor.Executor.get_class
:classmethod:

```{autodoc2-docstring} fastvideo.worker.executor.Executor.get_class
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} set_lora_adapter(lora_nickname: str, lora_path: str | None = None) -> None
:canonical: fastvideo.worker.executor.Executor.set_lora_adapter
:abstractmethod:

```{autodoc2-docstring} fastvideo.worker.executor.Executor.set_lora_adapter
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} shutdown() -> None
:canonical: fastvideo.worker.executor.Executor.shutdown
:abstractmethod:

```{autodoc2-docstring} fastvideo.worker.executor.Executor.shutdown
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} logger
:canonical: fastvideo.worker.executor.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.worker.executor.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
