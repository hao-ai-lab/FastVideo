# {py:mod}`fastvideo.platforms.interface`

```{py:module} fastvideo.platforms.interface
```

```{autodoc2-docstring} fastvideo.platforms.interface
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AttentionBackendEnum <fastvideo.platforms.interface.AttentionBackendEnum>`
  -
* - {py:obj}`CpuArchEnum <fastvideo.platforms.interface.CpuArchEnum>`
  -
* - {py:obj}`DeviceCapability <fastvideo.platforms.interface.DeviceCapability>`
  - ```{autodoc2-docstring} fastvideo.platforms.interface.DeviceCapability
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`Platform <fastvideo.platforms.interface.Platform>`
  - ```{autodoc2-docstring} fastvideo.platforms.interface.Platform
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`PlatformEnum <fastvideo.platforms.interface.PlatformEnum>`
  -
* - {py:obj}`UnspecifiedPlatform <fastvideo.platforms.interface.UnspecifiedPlatform>`
  - ```{autodoc2-docstring} fastvideo.platforms.interface.UnspecifiedPlatform
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.platforms.interface.logger>`
  - ```{autodoc2-docstring} fastvideo.platforms.interface.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} AttentionBackendEnum
:canonical: fastvideo.platforms.interface.AttentionBackendEnum

Bases: {py:obj}`enum.Enum`

````{py:attribute} FLASH_ATTN
:canonical: fastvideo.platforms.interface.AttentionBackendEnum.FLASH_ATTN
:value: >
   'auto(...)'

```{autodoc2-docstring} fastvideo.platforms.interface.AttentionBackendEnum.FLASH_ATTN
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} NO_ATTENTION
:canonical: fastvideo.platforms.interface.AttentionBackendEnum.NO_ATTENTION
:value: >
   'auto(...)'

```{autodoc2-docstring} fastvideo.platforms.interface.AttentionBackendEnum.NO_ATTENTION
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} SAGE_ATTN
:canonical: fastvideo.platforms.interface.AttentionBackendEnum.SAGE_ATTN
:value: >
   'auto(...)'

```{autodoc2-docstring} fastvideo.platforms.interface.AttentionBackendEnum.SAGE_ATTN
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} SLIDING_TILE_ATTN
:canonical: fastvideo.platforms.interface.AttentionBackendEnum.SLIDING_TILE_ATTN
:value: >
   'auto(...)'

```{autodoc2-docstring} fastvideo.platforms.interface.AttentionBackendEnum.SLIDING_TILE_ATTN
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} TORCH_SDPA
:canonical: fastvideo.platforms.interface.AttentionBackendEnum.TORCH_SDPA
:value: >
   'auto(...)'

```{autodoc2-docstring} fastvideo.platforms.interface.AttentionBackendEnum.TORCH_SDPA
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} VIDEO_SPARSE_ATTN
:canonical: fastvideo.platforms.interface.AttentionBackendEnum.VIDEO_SPARSE_ATTN
:value: >
   'auto(...)'

```{autodoc2-docstring} fastvideo.platforms.interface.AttentionBackendEnum.VIDEO_SPARSE_ATTN
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} CpuArchEnum
:canonical: fastvideo.platforms.interface.CpuArchEnum

Bases: {py:obj}`enum.Enum`

````{py:attribute} ARM
:canonical: fastvideo.platforms.interface.CpuArchEnum.ARM
:value: >
   'auto(...)'

```{autodoc2-docstring} fastvideo.platforms.interface.CpuArchEnum.ARM
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} UNSPECIFIED
:canonical: fastvideo.platforms.interface.CpuArchEnum.UNSPECIFIED
:value: >
   'auto(...)'

```{autodoc2-docstring} fastvideo.platforms.interface.CpuArchEnum.UNSPECIFIED
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} X86
:canonical: fastvideo.platforms.interface.CpuArchEnum.X86
:value: >
   'auto(...)'

```{autodoc2-docstring} fastvideo.platforms.interface.CpuArchEnum.X86
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} DeviceCapability
:canonical: fastvideo.platforms.interface.DeviceCapability

Bases: {py:obj}`typing.NamedTuple`

```{autodoc2-docstring} fastvideo.platforms.interface.DeviceCapability
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} as_version_str() -> str
:canonical: fastvideo.platforms.interface.DeviceCapability.as_version_str

```{autodoc2-docstring} fastvideo.platforms.interface.DeviceCapability.as_version_str
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} major
:canonical: fastvideo.platforms.interface.DeviceCapability.major
:type: int
:value: >
   None

```{autodoc2-docstring} fastvideo.platforms.interface.DeviceCapability.major
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} minor
:canonical: fastvideo.platforms.interface.DeviceCapability.minor
:type: int
:value: >
   None

```{autodoc2-docstring} fastvideo.platforms.interface.DeviceCapability.minor
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} to_int() -> int
:canonical: fastvideo.platforms.interface.DeviceCapability.to_int

```{autodoc2-docstring} fastvideo.platforms.interface.DeviceCapability.to_int
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} Platform
:canonical: fastvideo.platforms.interface.Platform

```{autodoc2-docstring} fastvideo.platforms.interface.Platform
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} device_name
:canonical: fastvideo.platforms.interface.Platform.device_name
:type: str
:value: >
   None

```{autodoc2-docstring} fastvideo.platforms.interface.Platform.device_name
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} device_type
:canonical: fastvideo.platforms.interface.Platform.device_type
:type: str
:value: >
   None

```{autodoc2-docstring} fastvideo.platforms.interface.Platform.device_type
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} dispatch_key
:canonical: fastvideo.platforms.interface.Platform.dispatch_key
:type: str
:value: >
   'CPU'

```{autodoc2-docstring} fastvideo.platforms.interface.Platform.dispatch_key
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_attn_backend_cls(selected_backend: fastvideo.platforms.interface.AttentionBackendEnum | None, head_size: int, dtype: torch.dtype) -> str
:canonical: fastvideo.platforms.interface.Platform.get_attn_backend_cls
:classmethod:

```{autodoc2-docstring} fastvideo.platforms.interface.Platform.get_attn_backend_cls
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_cpu_architecture() -> fastvideo.platforms.interface.CpuArchEnum
:canonical: fastvideo.platforms.interface.Platform.get_cpu_architecture
:classmethod:

```{autodoc2-docstring} fastvideo.platforms.interface.Platform.get_cpu_architecture
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_current_memory_usage(device: torch.types.Device | None = None) -> float
:canonical: fastvideo.platforms.interface.Platform.get_current_memory_usage
:abstractmethod:
:classmethod:

```{autodoc2-docstring} fastvideo.platforms.interface.Platform.get_current_memory_usage
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_device_capability(device_id: int = 0) -> fastvideo.platforms.interface.DeviceCapability | None
:canonical: fastvideo.platforms.interface.Platform.get_device_capability
:classmethod:

```{autodoc2-docstring} fastvideo.platforms.interface.Platform.get_device_capability
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_device_communicator_cls() -> str
:canonical: fastvideo.platforms.interface.Platform.get_device_communicator_cls
:classmethod:

```{autodoc2-docstring} fastvideo.platforms.interface.Platform.get_device_communicator_cls
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_device_name(device_id: int = 0) -> str
:canonical: fastvideo.platforms.interface.Platform.get_device_name
:abstractmethod:
:classmethod:

```{autodoc2-docstring} fastvideo.platforms.interface.Platform.get_device_name
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_device_total_memory(device_id: int = 0) -> int
:canonical: fastvideo.platforms.interface.Platform.get_device_total_memory
:abstractmethod:
:classmethod:

```{autodoc2-docstring} fastvideo.platforms.interface.Platform.get_device_total_memory
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_device_uuid(device_id: int = 0) -> str
:canonical: fastvideo.platforms.interface.Platform.get_device_uuid
:abstractmethod:
:classmethod:

```{autodoc2-docstring} fastvideo.platforms.interface.Platform.get_device_uuid
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} has_device_capability(capability: tuple[int, int] | int, device_id: int = 0) -> bool
:canonical: fastvideo.platforms.interface.Platform.has_device_capability
:classmethod:

```{autodoc2-docstring} fastvideo.platforms.interface.Platform.has_device_capability
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} inference_mode()
:canonical: fastvideo.platforms.interface.Platform.inference_mode
:classmethod:

```{autodoc2-docstring} fastvideo.platforms.interface.Platform.inference_mode
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} is_async_output_supported(enforce_eager: bool | None) -> bool
:canonical: fastvideo.platforms.interface.Platform.is_async_output_supported
:abstractmethod:
:classmethod:

```{autodoc2-docstring} fastvideo.platforms.interface.Platform.is_async_output_supported
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} is_cpu() -> bool
:canonical: fastvideo.platforms.interface.Platform.is_cpu

```{autodoc2-docstring} fastvideo.platforms.interface.Platform.is_cpu
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} is_cuda() -> bool
:canonical: fastvideo.platforms.interface.Platform.is_cuda

```{autodoc2-docstring} fastvideo.platforms.interface.Platform.is_cuda
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} is_cuda_alike() -> bool
:canonical: fastvideo.platforms.interface.Platform.is_cuda_alike

```{autodoc2-docstring} fastvideo.platforms.interface.Platform.is_cuda_alike
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} is_mps() -> bool
:canonical: fastvideo.platforms.interface.Platform.is_mps

```{autodoc2-docstring} fastvideo.platforms.interface.Platform.is_mps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} is_out_of_tree() -> bool
:canonical: fastvideo.platforms.interface.Platform.is_out_of_tree

```{autodoc2-docstring} fastvideo.platforms.interface.Platform.is_out_of_tree
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} is_rocm() -> bool
:canonical: fastvideo.platforms.interface.Platform.is_rocm

```{autodoc2-docstring} fastvideo.platforms.interface.Platform.is_rocm
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} is_tpu() -> bool
:canonical: fastvideo.platforms.interface.Platform.is_tpu

```{autodoc2-docstring} fastvideo.platforms.interface.Platform.is_tpu
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} seed_everything(seed: int | None = None) -> None
:canonical: fastvideo.platforms.interface.Platform.seed_everything
:classmethod:

```{autodoc2-docstring} fastvideo.platforms.interface.Platform.seed_everything
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} simple_compile_backend
:canonical: fastvideo.platforms.interface.Platform.simple_compile_backend
:type: str
:value: >
   'inductor'

```{autodoc2-docstring} fastvideo.platforms.interface.Platform.simple_compile_backend
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} supported_quantization
:canonical: fastvideo.platforms.interface.Platform.supported_quantization
:type: list[str]
:value: >
   []

```{autodoc2-docstring} fastvideo.platforms.interface.Platform.supported_quantization
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} verify_model_arch(model_arch: str) -> None
:canonical: fastvideo.platforms.interface.Platform.verify_model_arch
:classmethod:

```{autodoc2-docstring} fastvideo.platforms.interface.Platform.verify_model_arch
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} verify_quantization(quant: str) -> None
:canonical: fastvideo.platforms.interface.Platform.verify_quantization
:classmethod:

```{autodoc2-docstring} fastvideo.platforms.interface.Platform.verify_quantization
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} PlatformEnum
:canonical: fastvideo.platforms.interface.PlatformEnum

Bases: {py:obj}`enum.Enum`

````{py:attribute} CPU
:canonical: fastvideo.platforms.interface.PlatformEnum.CPU
:value: >
   'auto(...)'

```{autodoc2-docstring} fastvideo.platforms.interface.PlatformEnum.CPU
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} CUDA
:canonical: fastvideo.platforms.interface.PlatformEnum.CUDA
:value: >
   'auto(...)'

```{autodoc2-docstring} fastvideo.platforms.interface.PlatformEnum.CUDA
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} MPS
:canonical: fastvideo.platforms.interface.PlatformEnum.MPS
:value: >
   'auto(...)'

```{autodoc2-docstring} fastvideo.platforms.interface.PlatformEnum.MPS
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} OOT
:canonical: fastvideo.platforms.interface.PlatformEnum.OOT
:value: >
   'auto(...)'

```{autodoc2-docstring} fastvideo.platforms.interface.PlatformEnum.OOT
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} ROCM
:canonical: fastvideo.platforms.interface.PlatformEnum.ROCM
:value: >
   'auto(...)'

```{autodoc2-docstring} fastvideo.platforms.interface.PlatformEnum.ROCM
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} TPU
:canonical: fastvideo.platforms.interface.PlatformEnum.TPU
:value: >
   'auto(...)'

```{autodoc2-docstring} fastvideo.platforms.interface.PlatformEnum.TPU
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} UNSPECIFIED
:canonical: fastvideo.platforms.interface.PlatformEnum.UNSPECIFIED
:value: >
   'auto(...)'

```{autodoc2-docstring} fastvideo.platforms.interface.PlatformEnum.UNSPECIFIED
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} UnspecifiedPlatform
:canonical: fastvideo.platforms.interface.UnspecifiedPlatform

Bases: {py:obj}`fastvideo.platforms.interface.Platform`

```{autodoc2-docstring} fastvideo.platforms.interface.UnspecifiedPlatform
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} device_type
:canonical: fastvideo.platforms.interface.UnspecifiedPlatform.device_type
:value: <Multiline-String>

```{autodoc2-docstring} fastvideo.platforms.interface.UnspecifiedPlatform.device_type
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} logger
:canonical: fastvideo.platforms.interface.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.platforms.interface.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
