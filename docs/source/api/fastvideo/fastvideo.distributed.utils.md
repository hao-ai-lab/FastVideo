# {py:mod}`fastvideo.distributed.utils`

```{py:module} fastvideo.distributed.utils
```

```{autodoc2-docstring} fastvideo.distributed.utils
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`StatelessProcessGroup <fastvideo.distributed.utils.StatelessProcessGroup>`
  - ```{autodoc2-docstring} fastvideo.distributed.utils.StatelessProcessGroup
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`divide <fastvideo.distributed.utils.divide>`
  - ```{autodoc2-docstring} fastvideo.distributed.utils.divide
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`ensure_divisibility <fastvideo.distributed.utils.ensure_divisibility>`
  - ```{autodoc2-docstring} fastvideo.distributed.utils.ensure_divisibility
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`split_tensor_along_last_dim <fastvideo.distributed.utils.split_tensor_along_last_dim>`
  - ```{autodoc2-docstring} fastvideo.distributed.utils.split_tensor_along_last_dim
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.distributed.utils.logger>`
  - ```{autodoc2-docstring} fastvideo.distributed.utils.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} StatelessProcessGroup
:canonical: fastvideo.distributed.utils.StatelessProcessGroup

```{autodoc2-docstring} fastvideo.distributed.utils.StatelessProcessGroup
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} all_gather_obj(obj: typing.Any) -> list[typing.Any]
:canonical: fastvideo.distributed.utils.StatelessProcessGroup.all_gather_obj

```{autodoc2-docstring} fastvideo.distributed.utils.StatelessProcessGroup.all_gather_obj
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} barrier()
:canonical: fastvideo.distributed.utils.StatelessProcessGroup.barrier

```{autodoc2-docstring} fastvideo.distributed.utils.StatelessProcessGroup.barrier
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} broadcast_obj(obj: typing.Any | None, src: int) -> typing.Any
:canonical: fastvideo.distributed.utils.StatelessProcessGroup.broadcast_obj

```{autodoc2-docstring} fastvideo.distributed.utils.StatelessProcessGroup.broadcast_obj
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} broadcast_recv_src_counter
:canonical: fastvideo.distributed.utils.StatelessProcessGroup.broadcast_recv_src_counter
:type: dict[int, int]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.distributed.utils.StatelessProcessGroup.broadcast_recv_src_counter
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} broadcast_send_counter
:canonical: fastvideo.distributed.utils.StatelessProcessGroup.broadcast_send_counter
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.distributed.utils.StatelessProcessGroup.broadcast_send_counter
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} create(host: str, port: int, rank: int, world_size: int, data_expiration_seconds: int = 3600) -> fastvideo.distributed.utils.StatelessProcessGroup
:canonical: fastvideo.distributed.utils.StatelessProcessGroup.create
:staticmethod:

```{autodoc2-docstring} fastvideo.distributed.utils.StatelessProcessGroup.create
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} data_expiration_seconds
:canonical: fastvideo.distributed.utils.StatelessProcessGroup.data_expiration_seconds
:type: int
:value: >
   3600

```{autodoc2-docstring} fastvideo.distributed.utils.StatelessProcessGroup.data_expiration_seconds
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} entries
:canonical: fastvideo.distributed.utils.StatelessProcessGroup.entries
:type: collections.deque[tuple[str, float]]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.distributed.utils.StatelessProcessGroup.entries
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} expire_data() -> None
:canonical: fastvideo.distributed.utils.StatelessProcessGroup.expire_data

```{autodoc2-docstring} fastvideo.distributed.utils.StatelessProcessGroup.expire_data
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} rank
:canonical: fastvideo.distributed.utils.StatelessProcessGroup.rank
:type: int
:value: >
   None

```{autodoc2-docstring} fastvideo.distributed.utils.StatelessProcessGroup.rank
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} recv_obj(src: int) -> typing.Any
:canonical: fastvideo.distributed.utils.StatelessProcessGroup.recv_obj

```{autodoc2-docstring} fastvideo.distributed.utils.StatelessProcessGroup.recv_obj
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} recv_src_counter
:canonical: fastvideo.distributed.utils.StatelessProcessGroup.recv_src_counter
:type: dict[int, int]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.distributed.utils.StatelessProcessGroup.recv_src_counter
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} send_dst_counter
:canonical: fastvideo.distributed.utils.StatelessProcessGroup.send_dst_counter
:type: dict[int, int]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.distributed.utils.StatelessProcessGroup.send_dst_counter
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} send_obj(obj: typing.Any, dst: int)
:canonical: fastvideo.distributed.utils.StatelessProcessGroup.send_obj

```{autodoc2-docstring} fastvideo.distributed.utils.StatelessProcessGroup.send_obj
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} store
:canonical: fastvideo.distributed.utils.StatelessProcessGroup.store
:type: torch._C._distributed_c10d.Store
:value: >
   None

```{autodoc2-docstring} fastvideo.distributed.utils.StatelessProcessGroup.store
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} world_size
:canonical: fastvideo.distributed.utils.StatelessProcessGroup.world_size
:type: int
:value: >
   None

```{autodoc2-docstring} fastvideo.distributed.utils.StatelessProcessGroup.world_size
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:function} divide(numerator: int, denominator: int) -> int
:canonical: fastvideo.distributed.utils.divide

```{autodoc2-docstring} fastvideo.distributed.utils.divide
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} ensure_divisibility(numerator, denominator) -> None
:canonical: fastvideo.distributed.utils.ensure_divisibility

```{autodoc2-docstring} fastvideo.distributed.utils.ensure_divisibility
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:data} logger
:canonical: fastvideo.distributed.utils.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.distributed.utils.logger
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:function} split_tensor_along_last_dim(tensor: torch.Tensor, num_partitions: int, contiguous_split_chunks: bool = False) -> collections.abc.Sequence[torch.Tensor]
:canonical: fastvideo.distributed.utils.split_tensor_along_last_dim

```{autodoc2-docstring} fastvideo.distributed.utils.split_tensor_along_last_dim
:parser: docs.source.autodoc2_docstring_parser
```
````
