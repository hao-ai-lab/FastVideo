# {py:mod}`fastvideo.worker.multiproc_executor`

```{py:module} fastvideo.worker.multiproc_executor
```

```{autodoc2-docstring} fastvideo.worker.multiproc_executor
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MultiprocExecutor <fastvideo.worker.multiproc_executor.MultiprocExecutor>`
  -
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.worker.multiproc_executor.logger>`
  - ```{autodoc2-docstring} fastvideo.worker.multiproc_executor.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} MultiprocExecutor(fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs)
:canonical: fastvideo.worker.multiproc_executor.MultiprocExecutor

Bases: {py:obj}`fastvideo.worker.executor.Executor`

````{py:method} collective_rpc(method: str | collections.abc.Callable, timeout: float | None = None, args: tuple = (), kwargs: dict | None = None) -> list[typing.Any]
:canonical: fastvideo.worker.multiproc_executor.MultiprocExecutor.collective_rpc

````

````{py:method} execute_forward(forward_batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.pipeline_batch_info.ForwardBatch
:canonical: fastvideo.worker.multiproc_executor.MultiprocExecutor.execute_forward

```{autodoc2-docstring} fastvideo.worker.multiproc_executor.MultiprocExecutor.execute_forward
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} set_lora_adapter(lora_nickname: str, lora_path: str | None = None) -> None
:canonical: fastvideo.worker.multiproc_executor.MultiprocExecutor.set_lora_adapter

````

````{py:method} shutdown() -> None
:canonical: fastvideo.worker.multiproc_executor.MultiprocExecutor.shutdown

```{autodoc2-docstring} fastvideo.worker.multiproc_executor.MultiprocExecutor.shutdown
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} logger
:canonical: fastvideo.worker.multiproc_executor.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.worker.multiproc_executor.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
