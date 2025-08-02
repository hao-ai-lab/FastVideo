# {py:mod}`fastvideo.platforms.cpu`

```{py:module} fastvideo.platforms.cpu
```

```{autodoc2-docstring} fastvideo.platforms.cpu
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CpuPlatform <fastvideo.platforms.cpu.CpuPlatform>`
  - ```{autodoc2-docstring} fastvideo.platforms.cpu.CpuPlatform
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} CpuPlatform
:canonical: fastvideo.platforms.cpu.CpuPlatform

Bases: {py:obj}`fastvideo.platforms.interface.Platform`

```{autodoc2-docstring} fastvideo.platforms.cpu.CpuPlatform
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} device_name
:canonical: fastvideo.platforms.cpu.CpuPlatform.device_name
:value: >
   'CPU'

```{autodoc2-docstring} fastvideo.platforms.cpu.CpuPlatform.device_name
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} device_type
:canonical: fastvideo.platforms.cpu.CpuPlatform.device_type
:value: >
   'cpu'

```{autodoc2-docstring} fastvideo.platforms.cpu.CpuPlatform.device_type
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} dispatch_key
:canonical: fastvideo.platforms.cpu.CpuPlatform.dispatch_key
:value: >
   'CPU'

```{autodoc2-docstring} fastvideo.platforms.cpu.CpuPlatform.dispatch_key
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_cpu_architecture() -> fastvideo.platforms.interface.CpuArchEnum
:canonical: fastvideo.platforms.cpu.CpuPlatform.get_cpu_architecture
:classmethod:

```{autodoc2-docstring} fastvideo.platforms.cpu.CpuPlatform.get_cpu_architecture
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_current_memory_usage(device: torch.types.Device | None = None) -> float
:canonical: fastvideo.platforms.cpu.CpuPlatform.get_current_memory_usage
:classmethod:

````

````{py:method} get_device_communicator_cls() -> str
:canonical: fastvideo.platforms.cpu.CpuPlatform.get_device_communicator_cls
:classmethod:

````

````{py:method} get_device_name(device_id: int = 0) -> str
:canonical: fastvideo.platforms.cpu.CpuPlatform.get_device_name
:classmethod:

````

````{py:method} get_device_total_memory(device_id: int = 0) -> int
:canonical: fastvideo.platforms.cpu.CpuPlatform.get_device_total_memory
:classmethod:

````

````{py:method} get_device_uuid(device_id: int = 0) -> str
:canonical: fastvideo.platforms.cpu.CpuPlatform.get_device_uuid
:classmethod:

````

````{py:method} is_async_output_supported(enforce_eager: bool | None) -> bool
:canonical: fastvideo.platforms.cpu.CpuPlatform.is_async_output_supported
:classmethod:

````

`````
