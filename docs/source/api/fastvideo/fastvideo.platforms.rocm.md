# {py:mod}`fastvideo.platforms.rocm`

```{py:module} fastvideo.platforms.rocm
```

```{autodoc2-docstring} fastvideo.platforms.rocm
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RocmPlatform <fastvideo.platforms.rocm.RocmPlatform>`
  - ```{autodoc2-docstring} fastvideo.platforms.rocm.RocmPlatform
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.platforms.rocm.logger>`
  - ```{autodoc2-docstring} fastvideo.platforms.rocm.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} RocmPlatform
:canonical: fastvideo.platforms.rocm.RocmPlatform

Bases: {py:obj}`fastvideo.platforms.interface.Platform`

```{autodoc2-docstring} fastvideo.platforms.rocm.RocmPlatform
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} device_control_env_var
:canonical: fastvideo.platforms.rocm.RocmPlatform.device_control_env_var
:type: str
:value: >
   'CUDA_VISIBLE_DEVICES'

```{autodoc2-docstring} fastvideo.platforms.rocm.RocmPlatform.device_control_env_var
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} device_name
:canonical: fastvideo.platforms.rocm.RocmPlatform.device_name
:type: str
:value: >
   'rocm'

```{autodoc2-docstring} fastvideo.platforms.rocm.RocmPlatform.device_name
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} device_type
:canonical: fastvideo.platforms.rocm.RocmPlatform.device_type
:type: str
:value: >
   'cuda'

```{autodoc2-docstring} fastvideo.platforms.rocm.RocmPlatform.device_type
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} dispatch_key
:canonical: fastvideo.platforms.rocm.RocmPlatform.dispatch_key
:type: str
:value: >
   'CUDA'

```{autodoc2-docstring} fastvideo.platforms.rocm.RocmPlatform.dispatch_key
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_attn_backend_cls(selected_backend: fastvideo.platforms.interface.AttentionBackendEnum | None, head_size: int, dtype: torch.dtype) -> str
:canonical: fastvideo.platforms.rocm.RocmPlatform.get_attn_backend_cls
:classmethod:

````

````{py:method} get_current_memory_usage(device: torch.device | None = None) -> float
:canonical: fastvideo.platforms.rocm.RocmPlatform.get_current_memory_usage
:classmethod:

````

````{py:method} get_device_capability(device_id: int = 0) -> fastvideo.platforms.interface.DeviceCapability
:canonical: fastvideo.platforms.rocm.RocmPlatform.get_device_capability
:classmethod:

````

````{py:method} get_device_communicator_cls() -> str
:canonical: fastvideo.platforms.rocm.RocmPlatform.get_device_communicator_cls
:classmethod:

````

````{py:method} get_device_name(device_id: int = 0) -> str
:canonical: fastvideo.platforms.rocm.RocmPlatform.get_device_name
:classmethod:

````

````{py:method} get_device_total_memory(device_id: int = 0) -> int
:canonical: fastvideo.platforms.rocm.RocmPlatform.get_device_total_memory
:classmethod:

````

````{py:method} is_async_output_supported(enforce_eager: bool | None) -> bool
:canonical: fastvideo.platforms.rocm.RocmPlatform.is_async_output_supported
:classmethod:

````

````{py:method} log_warnings() -> None
:canonical: fastvideo.platforms.rocm.RocmPlatform.log_warnings
:classmethod:

```{autodoc2-docstring} fastvideo.platforms.rocm.RocmPlatform.log_warnings
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} logger
:canonical: fastvideo.platforms.rocm.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.platforms.rocm.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
