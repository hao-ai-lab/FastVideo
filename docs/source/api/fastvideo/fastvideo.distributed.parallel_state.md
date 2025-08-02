# {py:mod}`fastvideo.distributed.parallel_state`

```{py:module} fastvideo.distributed.parallel_state
```

```{autodoc2-docstring} fastvideo.distributed.parallel_state
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GraphCaptureContext <fastvideo.distributed.parallel_state.GraphCaptureContext>`
  - ```{autodoc2-docstring} fastvideo.distributed.parallel_state.GraphCaptureContext
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`GroupCoordinator <fastvideo.distributed.parallel_state.GroupCoordinator>`
  - ```{autodoc2-docstring} fastvideo.distributed.parallel_state.GroupCoordinator
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`all_reduce <fastvideo.distributed.parallel_state.all_reduce>`
  - ```{autodoc2-docstring} fastvideo.distributed.parallel_state.all_reduce
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`all_reduce_fake <fastvideo.distributed.parallel_state.all_reduce_fake>`
  - ```{autodoc2-docstring} fastvideo.distributed.parallel_state.all_reduce_fake
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`cleanup_dist_env_and_memory <fastvideo.distributed.parallel_state.cleanup_dist_env_and_memory>`
  - ```{autodoc2-docstring} fastvideo.distributed.parallel_state.cleanup_dist_env_and_memory
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`destroy_distributed_environment <fastvideo.distributed.parallel_state.destroy_distributed_environment>`
  - ```{autodoc2-docstring} fastvideo.distributed.parallel_state.destroy_distributed_environment
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`destroy_model_parallel <fastvideo.distributed.parallel_state.destroy_model_parallel>`
  - ```{autodoc2-docstring} fastvideo.distributed.parallel_state.destroy_model_parallel
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`get_dp_group <fastvideo.distributed.parallel_state.get_dp_group>`
  - ```{autodoc2-docstring} fastvideo.distributed.parallel_state.get_dp_group
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`get_dp_rank <fastvideo.distributed.parallel_state.get_dp_rank>`
  - ```{autodoc2-docstring} fastvideo.distributed.parallel_state.get_dp_rank
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`get_dp_world_size <fastvideo.distributed.parallel_state.get_dp_world_size>`
  - ```{autodoc2-docstring} fastvideo.distributed.parallel_state.get_dp_world_size
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`get_local_torch_device <fastvideo.distributed.parallel_state.get_local_torch_device>`
  - ```{autodoc2-docstring} fastvideo.distributed.parallel_state.get_local_torch_device
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`get_sp_group <fastvideo.distributed.parallel_state.get_sp_group>`
  - ```{autodoc2-docstring} fastvideo.distributed.parallel_state.get_sp_group
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`get_sp_parallel_rank <fastvideo.distributed.parallel_state.get_sp_parallel_rank>`
  - ```{autodoc2-docstring} fastvideo.distributed.parallel_state.get_sp_parallel_rank
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`get_sp_world_size <fastvideo.distributed.parallel_state.get_sp_world_size>`
  - ```{autodoc2-docstring} fastvideo.distributed.parallel_state.get_sp_world_size
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`get_tp_group <fastvideo.distributed.parallel_state.get_tp_group>`
  - ```{autodoc2-docstring} fastvideo.distributed.parallel_state.get_tp_group
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`get_tp_rank <fastvideo.distributed.parallel_state.get_tp_rank>`
  - ```{autodoc2-docstring} fastvideo.distributed.parallel_state.get_tp_rank
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`get_tp_world_size <fastvideo.distributed.parallel_state.get_tp_world_size>`
  - ```{autodoc2-docstring} fastvideo.distributed.parallel_state.get_tp_world_size
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`get_world_group <fastvideo.distributed.parallel_state.get_world_group>`
  - ```{autodoc2-docstring} fastvideo.distributed.parallel_state.get_world_group
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`get_world_rank <fastvideo.distributed.parallel_state.get_world_rank>`
  - ```{autodoc2-docstring} fastvideo.distributed.parallel_state.get_world_rank
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`get_world_size <fastvideo.distributed.parallel_state.get_world_size>`
  - ```{autodoc2-docstring} fastvideo.distributed.parallel_state.get_world_size
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`init_distributed_environment <fastvideo.distributed.parallel_state.init_distributed_environment>`
  - ```{autodoc2-docstring} fastvideo.distributed.parallel_state.init_distributed_environment
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`init_model_parallel_group <fastvideo.distributed.parallel_state.init_model_parallel_group>`
  - ```{autodoc2-docstring} fastvideo.distributed.parallel_state.init_model_parallel_group
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`init_world_group <fastvideo.distributed.parallel_state.init_world_group>`
  - ```{autodoc2-docstring} fastvideo.distributed.parallel_state.init_world_group
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`initialize_model_parallel <fastvideo.distributed.parallel_state.initialize_model_parallel>`
  - ```{autodoc2-docstring} fastvideo.distributed.parallel_state.initialize_model_parallel
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`initialize_sequence_parallel_group <fastvideo.distributed.parallel_state.initialize_sequence_parallel_group>`
  - ```{autodoc2-docstring} fastvideo.distributed.parallel_state.initialize_sequence_parallel_group
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`initialize_tensor_parallel_group <fastvideo.distributed.parallel_state.initialize_tensor_parallel_group>`
  - ```{autodoc2-docstring} fastvideo.distributed.parallel_state.initialize_tensor_parallel_group
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`is_the_same_node_as <fastvideo.distributed.parallel_state.is_the_same_node_as>`
  - ```{autodoc2-docstring} fastvideo.distributed.parallel_state.is_the_same_node_as
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`maybe_init_distributed_environment_and_model_parallel <fastvideo.distributed.parallel_state.maybe_init_distributed_environment_and_model_parallel>`
  - ```{autodoc2-docstring} fastvideo.distributed.parallel_state.maybe_init_distributed_environment_and_model_parallel
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`model_parallel_is_initialized <fastvideo.distributed.parallel_state.model_parallel_is_initialized>`
  - ```{autodoc2-docstring} fastvideo.distributed.parallel_state.model_parallel_is_initialized
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`patch_tensor_parallel_group <fastvideo.distributed.parallel_state.patch_tensor_parallel_group>`
  - ```{autodoc2-docstring} fastvideo.distributed.parallel_state.patch_tensor_parallel_group
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`set_custom_all_reduce <fastvideo.distributed.parallel_state.set_custom_all_reduce>`
  - ```{autodoc2-docstring} fastvideo.distributed.parallel_state.set_custom_all_reduce
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TensorMetadata <fastvideo.distributed.parallel_state.TensorMetadata>`
  - ```{autodoc2-docstring} fastvideo.distributed.parallel_state.TensorMetadata
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`logger <fastvideo.distributed.parallel_state.logger>`
  - ```{autodoc2-docstring} fastvideo.distributed.parallel_state.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} GraphCaptureContext
:canonical: fastvideo.distributed.parallel_state.GraphCaptureContext

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GraphCaptureContext
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} stream
:canonical: fastvideo.distributed.parallel_state.GraphCaptureContext.stream
:type: torch.cuda.Stream | None
:value: >
   None

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GraphCaptureContext.stream
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} GroupCoordinator(group_ranks: list[list[int]], local_rank: int, torch_distributed_backend: str | torch.distributed.Backend, use_device_communicator: bool, use_message_queue_broadcaster: bool = False, group_name: str | None = None)
:canonical: fastvideo.distributed.parallel_state.GroupCoordinator

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GroupCoordinator
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GroupCoordinator.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} all_gather(input_: torch.Tensor, dim: int = -1) -> torch.Tensor
:canonical: fastvideo.distributed.parallel_state.GroupCoordinator.all_gather

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GroupCoordinator.all_gather
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} all_reduce(input_: torch.Tensor, op: torch.distributed.ReduceOp | None = ReduceOp.SUM) -> torch.Tensor
:canonical: fastvideo.distributed.parallel_state.GroupCoordinator.all_reduce

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GroupCoordinator.all_reduce
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} all_to_all_4D(input_: torch.Tensor, scatter_dim: int = 2, gather_dim: int = 1) -> torch.Tensor
:canonical: fastvideo.distributed.parallel_state.GroupCoordinator.all_to_all_4D

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GroupCoordinator.all_to_all_4D
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} barrier() -> None
:canonical: fastvideo.distributed.parallel_state.GroupCoordinator.barrier

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GroupCoordinator.barrier
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} broadcast(input_: torch.Tensor, src: int = 0)
:canonical: fastvideo.distributed.parallel_state.GroupCoordinator.broadcast

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GroupCoordinator.broadcast
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} broadcast_object(obj: typing.Any | None = None, src: int = 0)
:canonical: fastvideo.distributed.parallel_state.GroupCoordinator.broadcast_object

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GroupCoordinator.broadcast_object
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} broadcast_object_list(obj_list: list[typing.Any], src: int = 0, group: torch.distributed.ProcessGroup | None = None)
:canonical: fastvideo.distributed.parallel_state.GroupCoordinator.broadcast_object_list

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GroupCoordinator.broadcast_object_list
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} broadcast_tensor_dict(tensor_dict: dict[str, torch.Tensor | typing.Any] | None = None, src: int = 0, group: torch.distributed.ProcessGroup | None = None, metadata_group: torch.distributed.ProcessGroup | None = None) -> dict[str, torch.Tensor | typing.Any] | None
:canonical: fastvideo.distributed.parallel_state.GroupCoordinator.broadcast_tensor_dict

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GroupCoordinator.broadcast_tensor_dict
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} cpu_group
:canonical: fastvideo.distributed.parallel_state.GroupCoordinator.cpu_group
:type: torch.distributed.ProcessGroup
:value: >
   None

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GroupCoordinator.cpu_group
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} destroy() -> None
:canonical: fastvideo.distributed.parallel_state.GroupCoordinator.destroy

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GroupCoordinator.destroy
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} device_communicator
:canonical: fastvideo.distributed.parallel_state.GroupCoordinator.device_communicator
:type: fastvideo.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase
:value: >
   None

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GroupCoordinator.device_communicator
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} device_group
:canonical: fastvideo.distributed.parallel_state.GroupCoordinator.device_group
:type: torch.distributed.ProcessGroup
:value: >
   None

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GroupCoordinator.device_group
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:property} first_rank
:canonical: fastvideo.distributed.parallel_state.GroupCoordinator.first_rank

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GroupCoordinator.first_rank
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} gather(input_: torch.Tensor, dst: int = 0, dim: int = -1) -> torch.Tensor | None
:canonical: fastvideo.distributed.parallel_state.GroupCoordinator.gather

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GroupCoordinator.gather
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} graph_capture(graph_capture_context: fastvideo.distributed.parallel_state.GraphCaptureContext | None = None)
:canonical: fastvideo.distributed.parallel_state.GroupCoordinator.graph_capture

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GroupCoordinator.graph_capture
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:property} is_first_rank
:canonical: fastvideo.distributed.parallel_state.GroupCoordinator.is_first_rank

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GroupCoordinator.is_first_rank
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:property} is_last_rank
:canonical: fastvideo.distributed.parallel_state.GroupCoordinator.is_last_rank

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GroupCoordinator.is_last_rank
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:property} last_rank
:canonical: fastvideo.distributed.parallel_state.GroupCoordinator.last_rank

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GroupCoordinator.last_rank
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} local_rank
:canonical: fastvideo.distributed.parallel_state.GroupCoordinator.local_rank
:type: int
:value: >
   None

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GroupCoordinator.local_rank
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} mq_broadcaster
:canonical: fastvideo.distributed.parallel_state.GroupCoordinator.mq_broadcaster
:type: typing.Any | None
:value: >
   None

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GroupCoordinator.mq_broadcaster
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:property} next_rank
:canonical: fastvideo.distributed.parallel_state.GroupCoordinator.next_rank

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GroupCoordinator.next_rank
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:property} prev_rank
:canonical: fastvideo.distributed.parallel_state.GroupCoordinator.prev_rank

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GroupCoordinator.prev_rank
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} rank
:canonical: fastvideo.distributed.parallel_state.GroupCoordinator.rank
:type: int
:value: >
   None

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GroupCoordinator.rank
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} rank_in_group
:canonical: fastvideo.distributed.parallel_state.GroupCoordinator.rank_in_group
:type: int
:value: >
   None

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GroupCoordinator.rank_in_group
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} ranks
:canonical: fastvideo.distributed.parallel_state.GroupCoordinator.ranks
:type: list[int]
:value: >
   None

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GroupCoordinator.ranks
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} recv(size: torch.Size, dtype: torch.dtype, src: int | None = None) -> torch.Tensor
:canonical: fastvideo.distributed.parallel_state.GroupCoordinator.recv

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GroupCoordinator.recv
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} recv_object(src: int) -> typing.Any
:canonical: fastvideo.distributed.parallel_state.GroupCoordinator.recv_object

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GroupCoordinator.recv_object
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} recv_tensor_dict(src: int | None = None, all_gather_group: typing.Optional[fastvideo.distributed.parallel_state.GroupCoordinator] = None) -> dict[str, torch.Tensor | typing.Any] | None
:canonical: fastvideo.distributed.parallel_state.GroupCoordinator.recv_tensor_dict

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GroupCoordinator.recv_tensor_dict
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} send(tensor: torch.Tensor, dst: int | None = None) -> None
:canonical: fastvideo.distributed.parallel_state.GroupCoordinator.send

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GroupCoordinator.send
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} send_object(obj: typing.Any, dst: int) -> None
:canonical: fastvideo.distributed.parallel_state.GroupCoordinator.send_object

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GroupCoordinator.send_object
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} send_tensor_dict(tensor_dict: dict[str, torch.Tensor | typing.Any], dst: int | None = None, all_gather_group: typing.Optional[fastvideo.distributed.parallel_state.GroupCoordinator] = None) -> dict[str, torch.Tensor | typing.Any] | None
:canonical: fastvideo.distributed.parallel_state.GroupCoordinator.send_tensor_dict

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GroupCoordinator.send_tensor_dict
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} use_device_communicator
:canonical: fastvideo.distributed.parallel_state.GroupCoordinator.use_device_communicator
:type: bool
:value: >
   None

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GroupCoordinator.use_device_communicator
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} world_size
:canonical: fastvideo.distributed.parallel_state.GroupCoordinator.world_size
:type: int
:value: >
   None

```{autodoc2-docstring} fastvideo.distributed.parallel_state.GroupCoordinator.world_size
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} TensorMetadata
:canonical: fastvideo.distributed.parallel_state.TensorMetadata
:value: >
   'namedtuple(...)'

```{autodoc2-docstring} fastvideo.distributed.parallel_state.TensorMetadata
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:function} all_reduce(tensor: torch.Tensor, group_name: str) -> torch.Tensor
:canonical: fastvideo.distributed.parallel_state.all_reduce

```{autodoc2-docstring} fastvideo.distributed.parallel_state.all_reduce
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} all_reduce_fake(tensor: torch.Tensor, group_name: str) -> torch.Tensor
:canonical: fastvideo.distributed.parallel_state.all_reduce_fake

```{autodoc2-docstring} fastvideo.distributed.parallel_state.all_reduce_fake
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} cleanup_dist_env_and_memory(shutdown_ray: bool = False)
:canonical: fastvideo.distributed.parallel_state.cleanup_dist_env_and_memory

```{autodoc2-docstring} fastvideo.distributed.parallel_state.cleanup_dist_env_and_memory
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} destroy_distributed_environment() -> None
:canonical: fastvideo.distributed.parallel_state.destroy_distributed_environment

```{autodoc2-docstring} fastvideo.distributed.parallel_state.destroy_distributed_environment
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} destroy_model_parallel() -> None
:canonical: fastvideo.distributed.parallel_state.destroy_model_parallel

```{autodoc2-docstring} fastvideo.distributed.parallel_state.destroy_model_parallel
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} get_dp_group() -> fastvideo.distributed.parallel_state.GroupCoordinator
:canonical: fastvideo.distributed.parallel_state.get_dp_group

```{autodoc2-docstring} fastvideo.distributed.parallel_state.get_dp_group
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} get_dp_rank() -> int
:canonical: fastvideo.distributed.parallel_state.get_dp_rank

```{autodoc2-docstring} fastvideo.distributed.parallel_state.get_dp_rank
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} get_dp_world_size() -> int
:canonical: fastvideo.distributed.parallel_state.get_dp_world_size

```{autodoc2-docstring} fastvideo.distributed.parallel_state.get_dp_world_size
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} get_local_torch_device() -> torch.device
:canonical: fastvideo.distributed.parallel_state.get_local_torch_device

```{autodoc2-docstring} fastvideo.distributed.parallel_state.get_local_torch_device
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} get_sp_group() -> fastvideo.distributed.parallel_state.GroupCoordinator
:canonical: fastvideo.distributed.parallel_state.get_sp_group

```{autodoc2-docstring} fastvideo.distributed.parallel_state.get_sp_group
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} get_sp_parallel_rank() -> int
:canonical: fastvideo.distributed.parallel_state.get_sp_parallel_rank

```{autodoc2-docstring} fastvideo.distributed.parallel_state.get_sp_parallel_rank
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} get_sp_world_size() -> int
:canonical: fastvideo.distributed.parallel_state.get_sp_world_size

```{autodoc2-docstring} fastvideo.distributed.parallel_state.get_sp_world_size
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} get_tp_group() -> fastvideo.distributed.parallel_state.GroupCoordinator
:canonical: fastvideo.distributed.parallel_state.get_tp_group

```{autodoc2-docstring} fastvideo.distributed.parallel_state.get_tp_group
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} get_tp_rank() -> int
:canonical: fastvideo.distributed.parallel_state.get_tp_rank

```{autodoc2-docstring} fastvideo.distributed.parallel_state.get_tp_rank
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} get_tp_world_size() -> int
:canonical: fastvideo.distributed.parallel_state.get_tp_world_size

```{autodoc2-docstring} fastvideo.distributed.parallel_state.get_tp_world_size
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} get_world_group() -> fastvideo.distributed.parallel_state.GroupCoordinator
:canonical: fastvideo.distributed.parallel_state.get_world_group

```{autodoc2-docstring} fastvideo.distributed.parallel_state.get_world_group
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} get_world_rank() -> int
:canonical: fastvideo.distributed.parallel_state.get_world_rank

```{autodoc2-docstring} fastvideo.distributed.parallel_state.get_world_rank
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} get_world_size() -> int
:canonical: fastvideo.distributed.parallel_state.get_world_size

```{autodoc2-docstring} fastvideo.distributed.parallel_state.get_world_size
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} init_distributed_environment(world_size: int = 1, rank: int = 0, distributed_init_method: str = 'env://', local_rank: int = 0, backend: str = 'nccl', device_id: torch.device | None = None)
:canonical: fastvideo.distributed.parallel_state.init_distributed_environment

```{autodoc2-docstring} fastvideo.distributed.parallel_state.init_distributed_environment
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} init_model_parallel_group(group_ranks: list[list[int]], local_rank: int, backend: str, use_message_queue_broadcaster: bool = False, group_name: str | None = None) -> fastvideo.distributed.parallel_state.GroupCoordinator
:canonical: fastvideo.distributed.parallel_state.init_model_parallel_group

```{autodoc2-docstring} fastvideo.distributed.parallel_state.init_model_parallel_group
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} init_world_group(ranks: list[int], local_rank: int, backend: str) -> fastvideo.distributed.parallel_state.GroupCoordinator
:canonical: fastvideo.distributed.parallel_state.init_world_group

```{autodoc2-docstring} fastvideo.distributed.parallel_state.init_world_group
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} initialize_model_parallel(tensor_model_parallel_size: int = 1, sequence_model_parallel_size: int = 1, data_parallel_size: int = 1, backend: str | None = None) -> None
:canonical: fastvideo.distributed.parallel_state.initialize_model_parallel

```{autodoc2-docstring} fastvideo.distributed.parallel_state.initialize_model_parallel
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} initialize_sequence_parallel_group(sequence_model_parallel_size: int = 1, backend: str | None = None, group_name_suffix: str = '') -> fastvideo.distributed.parallel_state.GroupCoordinator
:canonical: fastvideo.distributed.parallel_state.initialize_sequence_parallel_group

```{autodoc2-docstring} fastvideo.distributed.parallel_state.initialize_sequence_parallel_group
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} initialize_tensor_parallel_group(tensor_model_parallel_size: int = 1, backend: str | None = None, group_name_suffix: str = '') -> fastvideo.distributed.parallel_state.GroupCoordinator
:canonical: fastvideo.distributed.parallel_state.initialize_tensor_parallel_group

```{autodoc2-docstring} fastvideo.distributed.parallel_state.initialize_tensor_parallel_group
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} is_the_same_node_as(pg: torch.distributed.ProcessGroup | fastvideo.distributed.utils.StatelessProcessGroup, source_rank: int = 0) -> list[int]
:canonical: fastvideo.distributed.parallel_state.is_the_same_node_as

```{autodoc2-docstring} fastvideo.distributed.parallel_state.is_the_same_node_as
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:data} logger
:canonical: fastvideo.distributed.parallel_state.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.distributed.parallel_state.logger
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:function} maybe_init_distributed_environment_and_model_parallel(tp_size: int, sp_size: int, distributed_init_method: str = 'env://')
:canonical: fastvideo.distributed.parallel_state.maybe_init_distributed_environment_and_model_parallel

```{autodoc2-docstring} fastvideo.distributed.parallel_state.maybe_init_distributed_environment_and_model_parallel
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} model_parallel_is_initialized() -> bool
:canonical: fastvideo.distributed.parallel_state.model_parallel_is_initialized

```{autodoc2-docstring} fastvideo.distributed.parallel_state.model_parallel_is_initialized
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} patch_tensor_parallel_group(tp_group: fastvideo.distributed.parallel_state.GroupCoordinator)
:canonical: fastvideo.distributed.parallel_state.patch_tensor_parallel_group

```{autodoc2-docstring} fastvideo.distributed.parallel_state.patch_tensor_parallel_group
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} set_custom_all_reduce(enable: bool)
:canonical: fastvideo.distributed.parallel_state.set_custom_all_reduce

```{autodoc2-docstring} fastvideo.distributed.parallel_state.set_custom_all_reduce
:parser: docs.source.autodoc2_docstring_parser
```
````
