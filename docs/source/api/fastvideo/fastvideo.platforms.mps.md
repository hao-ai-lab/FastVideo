# {py:mod}`fastvideo.platforms.mps`

```{py:module} fastvideo.platforms.mps
```

```{autodoc2-docstring} fastvideo.platforms.mps
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MpsPlatform <fastvideo.platforms.mps.MpsPlatform>`
  - ```{autodoc2-docstring} fastvideo.platforms.mps.MpsPlatform
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.platforms.mps.logger>`
  - ```{autodoc2-docstring} fastvideo.platforms.mps.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} MpsPlatform
:canonical: fastvideo.platforms.mps.MpsPlatform

Bases: {py:obj}`fastvideo.platforms.interface.Platform`

```{autodoc2-docstring} fastvideo.platforms.mps.MpsPlatform
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} device_control_env_var
:canonical: fastvideo.platforms.mps.MpsPlatform.device_control_env_var
:type: str
:value: >
   'MPS_VISIBLE_DEVICES'

```{autodoc2-docstring} fastvideo.platforms.mps.MpsPlatform.device_control_env_var
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} device_name
:canonical: fastvideo.platforms.mps.MpsPlatform.device_name
:type: str
:value: >
   'mps'

```{autodoc2-docstring} fastvideo.platforms.mps.MpsPlatform.device_name
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} device_type
:canonical: fastvideo.platforms.mps.MpsPlatform.device_type
:type: str
:value: >
   'mps'

```{autodoc2-docstring} fastvideo.platforms.mps.MpsPlatform.device_type
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} dispatch_key
:canonical: fastvideo.platforms.mps.MpsPlatform.dispatch_key
:type: str
:value: >
   'MPS'

```{autodoc2-docstring} fastvideo.platforms.mps.MpsPlatform.dispatch_key
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_attn_backend_cls(selected_backend: fastvideo.platforms.AttentionBackendEnum | None, head_size: int, dtype: torch.dtype) -> str
:canonical: fastvideo.platforms.mps.MpsPlatform.get_attn_backend_cls
:classmethod:

````

````{py:method} get_current_memory_usage(device: torch.types.Device | None = None) -> float
:canonical: fastvideo.platforms.mps.MpsPlatform.get_current_memory_usage
:classmethod:

````

````{py:method} get_device_capability(device_id: int = 0) -> fastvideo.platforms.interface.DeviceCapability | None
:canonical: fastvideo.platforms.mps.MpsPlatform.get_device_capability
:abstractmethod:
:classmethod:

````

````{py:method} get_device_communicator_cls() -> str
:canonical: fastvideo.platforms.mps.MpsPlatform.get_device_communicator_cls
:classmethod:

````

````{py:method} get_device_name(device_id: int = 0) -> str
:canonical: fastvideo.platforms.mps.MpsPlatform.get_device_name
:abstractmethod:
:classmethod:

````

````{py:method} get_device_total_memory(device_id: int = 0) -> int
:canonical: fastvideo.platforms.mps.MpsPlatform.get_device_total_memory
:abstractmethod:
:classmethod:

````

````{py:method} get_device_uuid(device_id: int = 0) -> str
:canonical: fastvideo.platforms.mps.MpsPlatform.get_device_uuid
:abstractmethod:
:classmethod:

````

````{py:method} is_async_output_supported(enforce_eager: bool | None) -> bool
:canonical: fastvideo.platforms.mps.MpsPlatform.is_async_output_supported
:classmethod:

````

````{py:method} seed_everything(seed: int | None = None) -> None
:canonical: fastvideo.platforms.mps.MpsPlatform.seed_everything
:classmethod:

```{autodoc2-docstring} fastvideo.platforms.mps.MpsPlatform.seed_everything
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} logger
:canonical: fastvideo.platforms.mps.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.platforms.mps.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
