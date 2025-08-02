# {py:mod}`fastvideo.platforms.cuda`

```{py:module} fastvideo.platforms.cuda
```

```{autodoc2-docstring} fastvideo.platforms.cuda
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CudaPlatformBase <fastvideo.platforms.cuda.CudaPlatformBase>`
  - ```{autodoc2-docstring} fastvideo.platforms.cuda.CudaPlatformBase
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`NonNvmlCudaPlatform <fastvideo.platforms.cuda.NonNvmlCudaPlatform>`
  - ```{autodoc2-docstring} fastvideo.platforms.cuda.NonNvmlCudaPlatform
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`NvmlCudaPlatform <fastvideo.platforms.cuda.NvmlCudaPlatform>`
  - ```{autodoc2-docstring} fastvideo.platforms.cuda.NvmlCudaPlatform
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`device_id_to_physical_device_id <fastvideo.platforms.cuda.device_id_to_physical_device_id>`
  - ```{autodoc2-docstring} fastvideo.platforms.cuda.device_id_to_physical_device_id
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`with_nvml_context <fastvideo.platforms.cuda.with_nvml_context>`
  - ```{autodoc2-docstring} fastvideo.platforms.cuda.with_nvml_context
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CudaPlatform <fastvideo.platforms.cuda.CudaPlatform>`
  - ```{autodoc2-docstring} fastvideo.platforms.cuda.CudaPlatform
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`logger <fastvideo.platforms.cuda.logger>`
  - ```{autodoc2-docstring} fastvideo.platforms.cuda.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`nvml_available <fastvideo.platforms.cuda.nvml_available>`
  - ```{autodoc2-docstring} fastvideo.platforms.cuda.nvml_available
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`pynvml <fastvideo.platforms.cuda.pynvml>`
  - ```{autodoc2-docstring} fastvideo.platforms.cuda.pynvml
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

````{py:data} CudaPlatform
:canonical: fastvideo.platforms.cuda.CudaPlatform
:value: >
   None

```{autodoc2-docstring} fastvideo.platforms.cuda.CudaPlatform
:parser: docs.source.autodoc2_docstring_parser
```

````

`````{py:class} CudaPlatformBase
:canonical: fastvideo.platforms.cuda.CudaPlatformBase

Bases: {py:obj}`fastvideo.platforms.interface.Platform`

```{autodoc2-docstring} fastvideo.platforms.cuda.CudaPlatformBase
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} device_control_env_var
:canonical: fastvideo.platforms.cuda.CudaPlatformBase.device_control_env_var
:type: str
:value: >
   'CUDA_VISIBLE_DEVICES'

```{autodoc2-docstring} fastvideo.platforms.cuda.CudaPlatformBase.device_control_env_var
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} device_name
:canonical: fastvideo.platforms.cuda.CudaPlatformBase.device_name
:type: str
:value: >
   'cuda'

```{autodoc2-docstring} fastvideo.platforms.cuda.CudaPlatformBase.device_name
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} device_type
:canonical: fastvideo.platforms.cuda.CudaPlatformBase.device_type
:type: str
:value: >
   'cuda'

```{autodoc2-docstring} fastvideo.platforms.cuda.CudaPlatformBase.device_type
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} dispatch_key
:canonical: fastvideo.platforms.cuda.CudaPlatformBase.dispatch_key
:type: str
:value: >
   'CUDA'

```{autodoc2-docstring} fastvideo.platforms.cuda.CudaPlatformBase.dispatch_key
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_attn_backend_cls(selected_backend: fastvideo.platforms.interface.AttentionBackendEnum | None, head_size: int, dtype: torch.dtype) -> str
:canonical: fastvideo.platforms.cuda.CudaPlatformBase.get_attn_backend_cls
:classmethod:

````

````{py:method} get_current_memory_usage(device: torch.types.Device | None = None) -> float
:canonical: fastvideo.platforms.cuda.CudaPlatformBase.get_current_memory_usage
:classmethod:

````

````{py:method} get_device_capability(device_id: int = 0) -> fastvideo.platforms.interface.DeviceCapability | None
:canonical: fastvideo.platforms.cuda.CudaPlatformBase.get_device_capability
:abstractmethod:
:classmethod:

````

````{py:method} get_device_communicator_cls() -> str
:canonical: fastvideo.platforms.cuda.CudaPlatformBase.get_device_communicator_cls
:classmethod:

````

````{py:method} get_device_name(device_id: int = 0) -> str
:canonical: fastvideo.platforms.cuda.CudaPlatformBase.get_device_name
:abstractmethod:
:classmethod:

````

````{py:method} get_device_total_memory(device_id: int = 0) -> int
:canonical: fastvideo.platforms.cuda.CudaPlatformBase.get_device_total_memory
:abstractmethod:
:classmethod:

````

````{py:method} is_async_output_supported(enforce_eager: bool | None) -> bool
:canonical: fastvideo.platforms.cuda.CudaPlatformBase.is_async_output_supported
:classmethod:

````

````{py:method} is_full_nvlink(device_ids: list[int]) -> bool
:canonical: fastvideo.platforms.cuda.CudaPlatformBase.is_full_nvlink
:abstractmethod:
:classmethod:

```{autodoc2-docstring} fastvideo.platforms.cuda.CudaPlatformBase.is_full_nvlink
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} log_warnings() -> None
:canonical: fastvideo.platforms.cuda.CudaPlatformBase.log_warnings
:classmethod:

```{autodoc2-docstring} fastvideo.platforms.cuda.CudaPlatformBase.log_warnings
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} NonNvmlCudaPlatform
:canonical: fastvideo.platforms.cuda.NonNvmlCudaPlatform

Bases: {py:obj}`fastvideo.platforms.cuda.CudaPlatformBase`

```{autodoc2-docstring} fastvideo.platforms.cuda.NonNvmlCudaPlatform
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} get_device_capability(device_id: int = 0) -> fastvideo.platforms.interface.DeviceCapability
:canonical: fastvideo.platforms.cuda.NonNvmlCudaPlatform.get_device_capability
:classmethod:

````

````{py:method} get_device_name(device_id: int = 0) -> str
:canonical: fastvideo.platforms.cuda.NonNvmlCudaPlatform.get_device_name
:classmethod:

````

````{py:method} get_device_total_memory(device_id: int = 0) -> int
:canonical: fastvideo.platforms.cuda.NonNvmlCudaPlatform.get_device_total_memory
:classmethod:

````

````{py:method} is_full_nvlink(physical_device_ids: list[int]) -> bool
:canonical: fastvideo.platforms.cuda.NonNvmlCudaPlatform.is_full_nvlink
:classmethod:

```{autodoc2-docstring} fastvideo.platforms.cuda.NonNvmlCudaPlatform.is_full_nvlink
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} NvmlCudaPlatform
:canonical: fastvideo.platforms.cuda.NvmlCudaPlatform

Bases: {py:obj}`fastvideo.platforms.cuda.CudaPlatformBase`

```{autodoc2-docstring} fastvideo.platforms.cuda.NvmlCudaPlatform
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} get_device_capability(device_id: int = 0) -> fastvideo.platforms.interface.DeviceCapability | None
:canonical: fastvideo.platforms.cuda.NvmlCudaPlatform.get_device_capability
:classmethod:

````

````{py:method} get_device_name(device_id: int = 0) -> str
:canonical: fastvideo.platforms.cuda.NvmlCudaPlatform.get_device_name
:classmethod:

````

````{py:method} get_device_total_memory(device_id: int = 0) -> int
:canonical: fastvideo.platforms.cuda.NvmlCudaPlatform.get_device_total_memory
:classmethod:

````

````{py:method} get_device_uuid(device_id: int = 0) -> str
:canonical: fastvideo.platforms.cuda.NvmlCudaPlatform.get_device_uuid
:classmethod:

````

````{py:method} has_device_capability(capability: tuple[int, int] | int, device_id: int = 0) -> bool
:canonical: fastvideo.platforms.cuda.NvmlCudaPlatform.has_device_capability
:classmethod:

````

````{py:method} is_full_nvlink(physical_device_ids: list[int]) -> bool
:canonical: fastvideo.platforms.cuda.NvmlCudaPlatform.is_full_nvlink
:classmethod:

```{autodoc2-docstring} fastvideo.platforms.cuda.NvmlCudaPlatform.is_full_nvlink
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} log_warnings() -> None
:canonical: fastvideo.platforms.cuda.NvmlCudaPlatform.log_warnings
:classmethod:

```{autodoc2-docstring} fastvideo.platforms.cuda.NvmlCudaPlatform.log_warnings
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:function} device_id_to_physical_device_id(device_id: int) -> int
:canonical: fastvideo.platforms.cuda.device_id_to_physical_device_id

```{autodoc2-docstring} fastvideo.platforms.cuda.device_id_to_physical_device_id
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:data} logger
:canonical: fastvideo.platforms.cuda.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.platforms.cuda.logger
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:data} nvml_available
:canonical: fastvideo.platforms.cuda.nvml_available
:value: >
   False

```{autodoc2-docstring} fastvideo.platforms.cuda.nvml_available
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:data} pynvml
:canonical: fastvideo.platforms.cuda.pynvml
:value: >
   'import_pynvml(...)'

```{autodoc2-docstring} fastvideo.platforms.cuda.pynvml
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:function} with_nvml_context(fn: collections.abc.Callable[fastvideo.platforms.cuda._P, fastvideo.platforms.cuda._R]) -> collections.abc.Callable[fastvideo.platforms.cuda._P, fastvideo.platforms.cuda._R]
:canonical: fastvideo.platforms.cuda.with_nvml_context

```{autodoc2-docstring} fastvideo.platforms.cuda.with_nvml_context
:parser: docs.source.autodoc2_docstring_parser
```
````
