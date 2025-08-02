# {py:mod}`fastvideo.attention.selector`

```{py:module} fastvideo.attention.selector
```

```{autodoc2-docstring} fastvideo.attention.selector
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`backend_name_to_enum <fastvideo.attention.selector.backend_name_to_enum>`
  - ```{autodoc2-docstring} fastvideo.attention.selector.backend_name_to_enum
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`get_attn_backend <fastvideo.attention.selector.get_attn_backend>`
  - ```{autodoc2-docstring} fastvideo.attention.selector.get_attn_backend
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`get_env_variable_attn_backend <fastvideo.attention.selector.get_env_variable_attn_backend>`
  - ```{autodoc2-docstring} fastvideo.attention.selector.get_env_variable_attn_backend
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`get_global_forced_attn_backend <fastvideo.attention.selector.get_global_forced_attn_backend>`
  - ```{autodoc2-docstring} fastvideo.attention.selector.get_global_forced_attn_backend
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`global_force_attn_backend <fastvideo.attention.selector.global_force_attn_backend>`
  - ```{autodoc2-docstring} fastvideo.attention.selector.global_force_attn_backend
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`global_force_attn_backend_context_manager <fastvideo.attention.selector.global_force_attn_backend_context_manager>`
  - ```{autodoc2-docstring} fastvideo.attention.selector.global_force_attn_backend_context_manager
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`forced_attn_backend <fastvideo.attention.selector.forced_attn_backend>`
  - ```{autodoc2-docstring} fastvideo.attention.selector.forced_attn_backend
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`logger <fastvideo.attention.selector.logger>`
  - ```{autodoc2-docstring} fastvideo.attention.selector.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

````{py:function} backend_name_to_enum(backend_name: str) -> fastvideo.platforms.AttentionBackendEnum | None
:canonical: fastvideo.attention.selector.backend_name_to_enum

```{autodoc2-docstring} fastvideo.attention.selector.backend_name_to_enum
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:data} forced_attn_backend
:canonical: fastvideo.attention.selector.forced_attn_backend
:type: fastvideo.platforms.AttentionBackendEnum | None
:value: >
   None

```{autodoc2-docstring} fastvideo.attention.selector.forced_attn_backend
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:function} get_attn_backend(head_size: int, dtype: torch.dtype, supported_attention_backends: tuple[fastvideo.platforms.AttentionBackendEnum, ...] | None = None) -> type[fastvideo.attention.backends.abstract.AttentionBackend]
:canonical: fastvideo.attention.selector.get_attn_backend

```{autodoc2-docstring} fastvideo.attention.selector.get_attn_backend
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} get_env_variable_attn_backend() -> fastvideo.platforms.AttentionBackendEnum | None
:canonical: fastvideo.attention.selector.get_env_variable_attn_backend

```{autodoc2-docstring} fastvideo.attention.selector.get_env_variable_attn_backend
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} get_global_forced_attn_backend() -> fastvideo.platforms.AttentionBackendEnum | None
:canonical: fastvideo.attention.selector.get_global_forced_attn_backend

```{autodoc2-docstring} fastvideo.attention.selector.get_global_forced_attn_backend
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} global_force_attn_backend(attn_backend: fastvideo.platforms.AttentionBackendEnum | None) -> None
:canonical: fastvideo.attention.selector.global_force_attn_backend

```{autodoc2-docstring} fastvideo.attention.selector.global_force_attn_backend
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} global_force_attn_backend_context_manager(attn_backend: fastvideo.platforms.AttentionBackendEnum) -> collections.abc.Generator[None, None, None]
:canonical: fastvideo.attention.selector.global_force_attn_backend_context_manager

```{autodoc2-docstring} fastvideo.attention.selector.global_force_attn_backend_context_manager
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:data} logger
:canonical: fastvideo.attention.selector.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.attention.selector.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
