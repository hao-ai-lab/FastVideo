# {py:mod}`fastvideo.distributed.device_communicators.cpu_communicator`

```{py:module} fastvideo.distributed.device_communicators.cpu_communicator
```

```{autodoc2-docstring} fastvideo.distributed.device_communicators.cpu_communicator
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CpuCommunicator <fastvideo.distributed.device_communicators.cpu_communicator.CpuCommunicator>`
  -
````

### API

`````{py:class} CpuCommunicator(cpu_group: torch.distributed.ProcessGroup, device: torch.device | None = None, device_group: torch.distributed.ProcessGroup | None = None, unique_name: str = '')
:canonical: fastvideo.distributed.device_communicators.cpu_communicator.CpuCommunicator

Bases: {py:obj}`fastvideo.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase`

````{py:method} all_gather(input_: torch.Tensor, dim: int = -1) -> torch.Tensor
:canonical: fastvideo.distributed.device_communicators.cpu_communicator.CpuCommunicator.all_gather

````

````{py:method} all_reduce(input_: torch.Tensor, op: torch.distributed.ReduceOp | None = torch.distributed.ReduceOp.SUM) -> torch.Tensor
:canonical: fastvideo.distributed.device_communicators.cpu_communicator.CpuCommunicator.all_reduce

````

````{py:method} gather(input_: torch.Tensor, dst: int = 0, dim: int = -1) -> torch.Tensor | None
:canonical: fastvideo.distributed.device_communicators.cpu_communicator.CpuCommunicator.gather

```{autodoc2-docstring} fastvideo.distributed.device_communicators.cpu_communicator.CpuCommunicator.gather
:parser: docs.source.autodoc2_docstring_parser
```

````

`````
