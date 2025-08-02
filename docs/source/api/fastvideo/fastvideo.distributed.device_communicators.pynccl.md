# {py:mod}`fastvideo.distributed.device_communicators.pynccl`

```{py:module} fastvideo.distributed.device_communicators.pynccl
```

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PyNcclCommunicator <fastvideo.distributed.device_communicators.pynccl.PyNcclCommunicator>`
  - ```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl.PyNcclCommunicator
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.distributed.device_communicators.pynccl.logger>`
  - ```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} PyNcclCommunicator(group: torch.distributed.ProcessGroup | fastvideo.distributed.utils.StatelessProcessGroup, device: int | str | torch.device, library_path: str | None = None)
:canonical: fastvideo.distributed.device_communicators.pynccl.PyNcclCommunicator

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl.PyNcclCommunicator
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl.PyNcclCommunicator.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} all_gather(output_tensor: torch.Tensor, input_tensor: torch.Tensor, stream=None)
:canonical: fastvideo.distributed.device_communicators.pynccl.PyNcclCommunicator.all_gather

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl.PyNcclCommunicator.all_gather
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} all_reduce(in_tensor: torch.Tensor, op: torch.distributed.ReduceOp = ReduceOp.SUM, stream=None) -> torch.Tensor
:canonical: fastvideo.distributed.device_communicators.pynccl.PyNcclCommunicator.all_reduce

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl.PyNcclCommunicator.all_reduce
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} broadcast(tensor: torch.Tensor, src: int, stream=None)
:canonical: fastvideo.distributed.device_communicators.pynccl.PyNcclCommunicator.broadcast

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl.PyNcclCommunicator.broadcast
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} recv(tensor: torch.Tensor, src: int, stream=None)
:canonical: fastvideo.distributed.device_communicators.pynccl.PyNcclCommunicator.recv

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl.PyNcclCommunicator.recv
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} reduce_scatter(output_tensor: torch.Tensor, input_tensor: torch.Tensor, op: torch.distributed.ReduceOp = ReduceOp.SUM, stream=None)
:canonical: fastvideo.distributed.device_communicators.pynccl.PyNcclCommunicator.reduce_scatter

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl.PyNcclCommunicator.reduce_scatter
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} send(tensor: torch.Tensor, dst: int, stream=None)
:canonical: fastvideo.distributed.device_communicators.pynccl.PyNcclCommunicator.send

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl.PyNcclCommunicator.send
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} logger
:canonical: fastvideo.distributed.device_communicators.pynccl.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
