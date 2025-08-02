# {py:mod}`fastvideo.distributed.device_communicators.pynccl_wrapper`

```{py:module} fastvideo.distributed.device_communicators.pynccl_wrapper
```

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Function <fastvideo.distributed.device_communicators.pynccl_wrapper.Function>`
  - ```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.Function
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`NCCLLibrary <fastvideo.distributed.device_communicators.pynccl_wrapper.NCCLLibrary>`
  - ```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.NCCLLibrary
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`ncclDataTypeEnum <fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum>`
  - ```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`ncclRedOpTypeEnum <fastvideo.distributed.device_communicators.pynccl_wrapper.ncclRedOpTypeEnum>`
  - ```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.ncclRedOpTypeEnum
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`ncclUniqueId <fastvideo.distributed.device_communicators.pynccl_wrapper.ncclUniqueId>`
  -
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`buffer_type <fastvideo.distributed.device_communicators.pynccl_wrapper.buffer_type>`
  - ```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.buffer_type
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`cudaStream_t <fastvideo.distributed.device_communicators.pynccl_wrapper.cudaStream_t>`
  - ```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.cudaStream_t
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`logger <fastvideo.distributed.device_communicators.pynccl_wrapper.logger>`
  - ```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`ncclComm_t <fastvideo.distributed.device_communicators.pynccl_wrapper.ncclComm_t>`
  - ```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.ncclComm_t
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`ncclDataType_t <fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataType_t>`
  - ```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataType_t
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`ncclRedOp_t <fastvideo.distributed.device_communicators.pynccl_wrapper.ncclRedOp_t>`
  - ```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.ncclRedOp_t
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`ncclResult_t <fastvideo.distributed.device_communicators.pynccl_wrapper.ncclResult_t>`
  - ```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.ncclResult_t
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} Function
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.Function

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.Function
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} argtypes
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.Function.argtypes
:type: list[typing.Any]
:value: >
   None

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.Function.argtypes
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} name
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.Function.name
:type: str
:value: >
   None

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.Function.name
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} restype
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.Function.restype
:type: typing.Any
:value: >
   None

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.Function.restype
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} NCCLLibrary(so_file: str | None = None)
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.NCCLLibrary

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.NCCLLibrary
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.NCCLLibrary.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} NCCL_CHECK(result: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclResult_t) -> None
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.NCCLLibrary.NCCL_CHECK

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.NCCLLibrary.NCCL_CHECK
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} exported_functions
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.NCCLLibrary.exported_functions
:value: >
   None

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.NCCLLibrary.exported_functions
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} ncclAllGather(sendbuff: fastvideo.distributed.device_communicators.pynccl_wrapper.buffer_type, recvbuff: fastvideo.distributed.device_communicators.pynccl_wrapper.buffer_type, count: int, datatype: int, comm: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclComm_t, stream: fastvideo.distributed.device_communicators.pynccl_wrapper.cudaStream_t) -> None
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.NCCLLibrary.ncclAllGather

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.NCCLLibrary.ncclAllGather
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} ncclAllReduce(sendbuff: fastvideo.distributed.device_communicators.pynccl_wrapper.buffer_type, recvbuff: fastvideo.distributed.device_communicators.pynccl_wrapper.buffer_type, count: int, datatype: int, op: int, comm: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclComm_t, stream: fastvideo.distributed.device_communicators.pynccl_wrapper.cudaStream_t) -> None
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.NCCLLibrary.ncclAllReduce

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.NCCLLibrary.ncclAllReduce
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} ncclBroadcast(sendbuff: fastvideo.distributed.device_communicators.pynccl_wrapper.buffer_type, recvbuff: fastvideo.distributed.device_communicators.pynccl_wrapper.buffer_type, count: int, datatype: int, root: int, comm: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclComm_t, stream: fastvideo.distributed.device_communicators.pynccl_wrapper.cudaStream_t) -> None
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.NCCLLibrary.ncclBroadcast

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.NCCLLibrary.ncclBroadcast
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} ncclCommDestroy(comm: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclComm_t) -> None
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.NCCLLibrary.ncclCommDestroy

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.NCCLLibrary.ncclCommDestroy
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} ncclCommInitRank(world_size: int, unique_id: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclUniqueId, rank: int) -> fastvideo.distributed.device_communicators.pynccl_wrapper.ncclComm_t
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.NCCLLibrary.ncclCommInitRank

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.NCCLLibrary.ncclCommInitRank
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} ncclGetErrorString(result: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclResult_t) -> str
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.NCCLLibrary.ncclGetErrorString

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.NCCLLibrary.ncclGetErrorString
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} ncclGetUniqueId() -> fastvideo.distributed.device_communicators.pynccl_wrapper.ncclUniqueId
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.NCCLLibrary.ncclGetUniqueId

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.NCCLLibrary.ncclGetUniqueId
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} ncclGetVersion() -> str
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.NCCLLibrary.ncclGetVersion

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.NCCLLibrary.ncclGetVersion
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} ncclRecv(recvbuff: fastvideo.distributed.device_communicators.pynccl_wrapper.buffer_type, count: int, datatype: int, src: int, comm: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclComm_t, stream: fastvideo.distributed.device_communicators.pynccl_wrapper.cudaStream_t) -> None
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.NCCLLibrary.ncclRecv

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.NCCLLibrary.ncclRecv
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} ncclReduceScatter(sendbuff: fastvideo.distributed.device_communicators.pynccl_wrapper.buffer_type, recvbuff: fastvideo.distributed.device_communicators.pynccl_wrapper.buffer_type, count: int, datatype: int, op: int, comm: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclComm_t, stream: fastvideo.distributed.device_communicators.pynccl_wrapper.cudaStream_t) -> None
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.NCCLLibrary.ncclReduceScatter

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.NCCLLibrary.ncclReduceScatter
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} ncclSend(sendbuff: fastvideo.distributed.device_communicators.pynccl_wrapper.buffer_type, count: int, datatype: int, dest: int, comm: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclComm_t, stream: fastvideo.distributed.device_communicators.pynccl_wrapper.cudaStream_t) -> None
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.NCCLLibrary.ncclSend

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.NCCLLibrary.ncclSend
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} path_to_dict_mapping
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.NCCLLibrary.path_to_dict_mapping
:type: dict[str, dict[str, typing.Any]]
:value: >
   None

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.NCCLLibrary.path_to_dict_mapping
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} path_to_library_cache
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.NCCLLibrary.path_to_library_cache
:type: dict[str, typing.Any]
:value: >
   None

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.NCCLLibrary.path_to_library_cache
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} buffer_type
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.buffer_type
:value: >
   None

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.buffer_type
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:data} cudaStream_t
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.cudaStream_t
:value: >
   None

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.cudaStream_t
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:data} logger
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.logger
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:data} ncclComm_t
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclComm_t
:value: >
   None

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.ncclComm_t
:parser: docs.source.autodoc2_docstring_parser
```

````

`````{py:class} ncclDataTypeEnum
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} from_torch(dtype: torch.dtype) -> int
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum.from_torch
:classmethod:

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum.from_torch
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} ncclBfloat16
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum.ncclBfloat16
:value: >
   9

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum.ncclBfloat16
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} ncclChar
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum.ncclChar
:value: >
   0

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum.ncclChar
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} ncclDouble
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum.ncclDouble
:value: >
   8

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum.ncclDouble
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} ncclFloat
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum.ncclFloat
:value: >
   7

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum.ncclFloat
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} ncclFloat16
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum.ncclFloat16
:value: >
   6

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum.ncclFloat16
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} ncclFloat32
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum.ncclFloat32
:value: >
   7

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum.ncclFloat32
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} ncclFloat64
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum.ncclFloat64
:value: >
   8

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum.ncclFloat64
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} ncclHalf
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum.ncclHalf
:value: >
   6

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum.ncclHalf
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} ncclInt
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum.ncclInt
:value: >
   2

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum.ncclInt
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} ncclInt32
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum.ncclInt32
:value: >
   2

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum.ncclInt32
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} ncclInt64
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum.ncclInt64
:value: >
   4

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum.ncclInt64
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} ncclInt8
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum.ncclInt8
:value: >
   0

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum.ncclInt8
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} ncclNumTypes
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum.ncclNumTypes
:value: >
   10

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum.ncclNumTypes
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} ncclUint32
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum.ncclUint32
:value: >
   3

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum.ncclUint32
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} ncclUint64
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum.ncclUint64
:value: >
   5

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum.ncclUint64
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} ncclUint8
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum.ncclUint8
:value: >
   1

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataTypeEnum.ncclUint8
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} ncclDataType_t
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataType_t
:value: >
   None

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.ncclDataType_t
:parser: docs.source.autodoc2_docstring_parser
```

````

`````{py:class} ncclRedOpTypeEnum
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclRedOpTypeEnum

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.ncclRedOpTypeEnum
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} from_torch(op: torch.distributed.ReduceOp) -> int
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclRedOpTypeEnum.from_torch
:classmethod:

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.ncclRedOpTypeEnum.from_torch
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} ncclAvg
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclRedOpTypeEnum.ncclAvg
:value: >
   4

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.ncclRedOpTypeEnum.ncclAvg
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} ncclMax
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclRedOpTypeEnum.ncclMax
:value: >
   2

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.ncclRedOpTypeEnum.ncclMax
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} ncclMin
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclRedOpTypeEnum.ncclMin
:value: >
   3

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.ncclRedOpTypeEnum.ncclMin
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} ncclNumOps
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclRedOpTypeEnum.ncclNumOps
:value: >
   5

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.ncclRedOpTypeEnum.ncclNumOps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} ncclProd
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclRedOpTypeEnum.ncclProd
:value: >
   1

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.ncclRedOpTypeEnum.ncclProd
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} ncclSum
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclRedOpTypeEnum.ncclSum
:value: >
   0

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.ncclRedOpTypeEnum.ncclSum
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} ncclRedOp_t
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclRedOp_t
:value: >
   None

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.ncclRedOp_t
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:data} ncclResult_t
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclResult_t
:value: >
   None

```{autodoc2-docstring} fastvideo.distributed.device_communicators.pynccl_wrapper.ncclResult_t
:parser: docs.source.autodoc2_docstring_parser
```

````

```{py:class} ncclUniqueId()
:canonical: fastvideo.distributed.device_communicators.pynccl_wrapper.ncclUniqueId

Bases: {py:obj}`ctypes.Structure`

```
