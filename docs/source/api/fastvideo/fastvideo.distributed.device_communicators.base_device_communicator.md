# {py:mod}`fastvideo.distributed.device_communicators.base_device_communicator`

```{py:module} fastvideo.distributed.device_communicators.base_device_communicator
```

```{autodoc2-docstring} fastvideo.distributed.device_communicators.base_device_communicator
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DeviceCommunicatorBase <fastvideo.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase>`
  - ```{autodoc2-docstring} fastvideo.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`DistributedAutograd <fastvideo.distributed.device_communicators.base_device_communicator.DistributedAutograd>`
  - ```{autodoc2-docstring} fastvideo.distributed.device_communicators.base_device_communicator.DistributedAutograd
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} DeviceCommunicatorBase(cpu_group: torch.distributed.ProcessGroup, device: torch.device | None = None, device_group: torch.distributed.ProcessGroup | None = None, unique_name: str = '')
:canonical: fastvideo.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase

```{autodoc2-docstring} fastvideo.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} all_gather(input_: torch.Tensor, dim: int = -1) -> torch.Tensor
:canonical: fastvideo.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase.all_gather

```{autodoc2-docstring} fastvideo.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase.all_gather
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} all_reduce(input_: torch.Tensor, op: torch.distributed.ReduceOp | None = ReduceOp.SUM) -> torch.Tensor
:canonical: fastvideo.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase.all_reduce

```{autodoc2-docstring} fastvideo.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase.all_reduce
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} all_to_all_4D(input_: torch.Tensor, scatter_dim: int = 2, gather_dim: int = 1) -> torch.Tensor
:canonical: fastvideo.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase.all_to_all_4D

```{autodoc2-docstring} fastvideo.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase.all_to_all_4D
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} destroy() -> None
:canonical: fastvideo.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase.destroy

```{autodoc2-docstring} fastvideo.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase.destroy
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} gather(input_: torch.Tensor, dst: int = 0, dim: int = -1) -> torch.Tensor | None
:canonical: fastvideo.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase.gather

```{autodoc2-docstring} fastvideo.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase.gather
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} recv(size: torch.Size, dtype: torch.dtype, src: int | None = None) -> torch.Tensor
:canonical: fastvideo.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase.recv

```{autodoc2-docstring} fastvideo.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase.recv
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} send(tensor: torch.Tensor, dst: int | None = None) -> None
:canonical: fastvideo.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase.send

```{autodoc2-docstring} fastvideo.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase.send
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

``````{py:class} DistributedAutograd
:canonical: fastvideo.distributed.device_communicators.base_device_communicator.DistributedAutograd

```{autodoc2-docstring} fastvideo.distributed.device_communicators.base_device_communicator.DistributedAutograd
:parser: docs.source.autodoc2_docstring_parser
```

`````{py:class} AllGather(*args, **kwargs)
:canonical: fastvideo.distributed.device_communicators.base_device_communicator.DistributedAutograd.AllGather

Bases: {py:obj}`torch.autograd.Function`

```{autodoc2-docstring} fastvideo.distributed.device_communicators.base_device_communicator.DistributedAutograd.AllGather
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.distributed.device_communicators.base_device_communicator.DistributedAutograd.AllGather.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} backward(ctx: typing.Any, grad_output: torch.Tensor) -> tuple[None, torch.Tensor, None, None]
:canonical: fastvideo.distributed.device_communicators.base_device_communicator.DistributedAutograd.AllGather.backward
:staticmethod:

````

````{py:method} forward(ctx: typing.Any, group: torch.distributed.ProcessGroup, input_: torch.Tensor, world_size: int, dim: int) -> torch.Tensor
:canonical: fastvideo.distributed.device_communicators.base_device_communicator.DistributedAutograd.AllGather.forward
:staticmethod:

````

`````

`````{py:class} AllReduce(*args, **kwargs)
:canonical: fastvideo.distributed.device_communicators.base_device_communicator.DistributedAutograd.AllReduce

Bases: {py:obj}`torch.autograd.Function`

```{autodoc2-docstring} fastvideo.distributed.device_communicators.base_device_communicator.DistributedAutograd.AllReduce
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.distributed.device_communicators.base_device_communicator.DistributedAutograd.AllReduce.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} backward(ctx: typing.Any, grad_output: torch.Tensor) -> tuple[None, torch.Tensor, None]
:canonical: fastvideo.distributed.device_communicators.base_device_communicator.DistributedAutograd.AllReduce.backward
:staticmethod:

````

````{py:method} forward(ctx: typing.Any, group: torch.distributed.ProcessGroup, input_: torch.Tensor, op: torch.distributed.ReduceOp | None = None) -> torch.Tensor
:canonical: fastvideo.distributed.device_communicators.base_device_communicator.DistributedAutograd.AllReduce.forward
:staticmethod:

````

`````

`````{py:class} AllToAll4D(*args, **kwargs)
:canonical: fastvideo.distributed.device_communicators.base_device_communicator.DistributedAutograd.AllToAll4D

Bases: {py:obj}`torch.autograd.Function`

```{autodoc2-docstring} fastvideo.distributed.device_communicators.base_device_communicator.DistributedAutograd.AllToAll4D
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.distributed.device_communicators.base_device_communicator.DistributedAutograd.AllToAll4D.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} backward(ctx: typing.Any, grad_output: torch.Tensor) -> tuple[None, torch.Tensor, None, None, None]
:canonical: fastvideo.distributed.device_communicators.base_device_communicator.DistributedAutograd.AllToAll4D.backward
:staticmethod:

````

````{py:method} forward(ctx: typing.Any, group: torch.distributed.ProcessGroup, input_: torch.Tensor, world_size: int, scatter_dim: int, gather_dim: int) -> torch.Tensor
:canonical: fastvideo.distributed.device_communicators.base_device_communicator.DistributedAutograd.AllToAll4D.forward
:staticmethod:

````

`````

``````
