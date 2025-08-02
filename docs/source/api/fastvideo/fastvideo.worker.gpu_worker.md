# {py:mod}`fastvideo.worker.gpu_worker`

```{py:module} fastvideo.worker.gpu_worker
```

```{autodoc2-docstring} fastvideo.worker.gpu_worker
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Worker <fastvideo.worker.gpu_worker.Worker>`
  - ```{autodoc2-docstring} fastvideo.worker.gpu_worker.Worker
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_worker_process <fastvideo.worker.gpu_worker.run_worker_process>`
  - ```{autodoc2-docstring} fastvideo.worker.gpu_worker.run_worker_process
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CYAN <fastvideo.worker.gpu_worker.CYAN>`
  - ```{autodoc2-docstring} fastvideo.worker.gpu_worker.CYAN
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`RESET <fastvideo.worker.gpu_worker.RESET>`
  - ```{autodoc2-docstring} fastvideo.worker.gpu_worker.RESET
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`logger <fastvideo.worker.gpu_worker.logger>`
  - ```{autodoc2-docstring} fastvideo.worker.gpu_worker.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

````{py:data} CYAN
:canonical: fastvideo.worker.gpu_worker.CYAN
:value: >
   '\x1b[1;36m'

```{autodoc2-docstring} fastvideo.worker.gpu_worker.CYAN
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:data} RESET
:canonical: fastvideo.worker.gpu_worker.RESET
:value: >
   '\x1b[0;0m'

```{autodoc2-docstring} fastvideo.worker.gpu_worker.RESET
:parser: docs.source.autodoc2_docstring_parser
```

````

`````{py:class} Worker(fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs, local_rank: int, rank: int, pipe: multiprocessing.connection.Connection, master_port: int)
:canonical: fastvideo.worker.gpu_worker.Worker

```{autodoc2-docstring} fastvideo.worker.gpu_worker.Worker
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.worker.gpu_worker.Worker.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} event_loop() -> None
:canonical: fastvideo.worker.gpu_worker.Worker.event_loop

```{autodoc2-docstring} fastvideo.worker.gpu_worker.Worker.event_loop
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} execute_forward(forward_batch: fastvideo.pipelines.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.ForwardBatch
:canonical: fastvideo.worker.gpu_worker.Worker.execute_forward

```{autodoc2-docstring} fastvideo.worker.gpu_worker.Worker.execute_forward
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} init_device() -> None
:canonical: fastvideo.worker.gpu_worker.Worker.init_device

```{autodoc2-docstring} fastvideo.worker.gpu_worker.Worker.init_device
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} set_lora_adapter(lora_nickname: str, lora_path: str | None = None) -> None
:canonical: fastvideo.worker.gpu_worker.Worker.set_lora_adapter

```{autodoc2-docstring} fastvideo.worker.gpu_worker.Worker.set_lora_adapter
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} shutdown() -> dict[str, typing.Any]
:canonical: fastvideo.worker.gpu_worker.Worker.shutdown

```{autodoc2-docstring} fastvideo.worker.gpu_worker.Worker.shutdown
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} logger
:canonical: fastvideo.worker.gpu_worker.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.worker.gpu_worker.logger
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:function} run_worker_process(fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs, local_rank: int, rank: int, pipe: multiprocessing.connection.Connection, master_port: int)
:canonical: fastvideo.worker.gpu_worker.run_worker_process

```{autodoc2-docstring} fastvideo.worker.gpu_worker.run_worker_process
:parser: docs.source.autodoc2_docstring_parser
```
````
