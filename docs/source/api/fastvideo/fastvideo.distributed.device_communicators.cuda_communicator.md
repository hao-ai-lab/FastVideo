# {py:mod}`fastvideo.distributed.device_communicators.cuda_communicator`

```{py:module} fastvideo.distributed.device_communicators.cuda_communicator
```

```{autodoc2-docstring} fastvideo.distributed.device_communicators.cuda_communicator
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CudaCommunicator <fastvideo.distributed.device_communicators.cuda_communicator.CudaCommunicator>`
  -
````

### API

`````{py:class} CudaCommunicator(cpu_group: torch.distributed.ProcessGroup, device: torch.device | None = None, device_group: torch.distributed.ProcessGroup | None = None, unique_name: str = '')
:canonical: fastvideo.distributed.device_communicators.cuda_communicator.CudaCommunicator

Bases: {py:obj}`fastvideo.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase`

````{py:method} all_reduce(input_, op: torch.distributed.ReduceOp | None = None)
:canonical: fastvideo.distributed.device_communicators.cuda_communicator.CudaCommunicator.all_reduce

````

````{py:method} destroy() -> None
:canonical: fastvideo.distributed.device_communicators.cuda_communicator.CudaCommunicator.destroy

```{autodoc2-docstring} fastvideo.distributed.device_communicators.cuda_communicator.CudaCommunicator.destroy
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} recv(size: torch.Size, dtype: torch.dtype, src: int | None = None) -> torch.Tensor
:canonical: fastvideo.distributed.device_communicators.cuda_communicator.CudaCommunicator.recv

```{autodoc2-docstring} fastvideo.distributed.device_communicators.cuda_communicator.CudaCommunicator.recv
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} send(tensor: torch.Tensor, dst: int | None = None) -> None
:canonical: fastvideo.distributed.device_communicators.cuda_communicator.CudaCommunicator.send

```{autodoc2-docstring} fastvideo.distributed.device_communicators.cuda_communicator.CudaCommunicator.send
:parser: docs.source.autodoc2_docstring_parser
```

````

`````
