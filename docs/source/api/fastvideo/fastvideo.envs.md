# {py:mod}`fastvideo.envs`

```{py:module} fastvideo.envs
```

```{autodoc2-docstring} fastvideo.envs
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_default_cache_root <fastvideo.envs.get_default_cache_root>`
  - ```{autodoc2-docstring} fastvideo.envs.get_default_cache_root
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`get_default_config_root <fastvideo.envs.get_default_config_root>`
  - ```{autodoc2-docstring} fastvideo.envs.get_default_config_root
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`maybe_convert_int <fastvideo.envs.maybe_convert_int>`
  - ```{autodoc2-docstring} fastvideo.envs.maybe_convert_int
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`environment_variables <fastvideo.envs.environment_variables>`
  - ```{autodoc2-docstring} fastvideo.envs.environment_variables
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

````{py:data} environment_variables
:canonical: fastvideo.envs.environment_variables
:type: dict[str, collections.abc.Callable[[], typing.Any]]
:value: >
   None

```{autodoc2-docstring} fastvideo.envs.environment_variables
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:function} get_default_cache_root() -> str
:canonical: fastvideo.envs.get_default_cache_root

```{autodoc2-docstring} fastvideo.envs.get_default_cache_root
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} get_default_config_root() -> str
:canonical: fastvideo.envs.get_default_config_root

```{autodoc2-docstring} fastvideo.envs.get_default_config_root
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} maybe_convert_int(value: str | None) -> int | None
:canonical: fastvideo.envs.maybe_convert_int

```{autodoc2-docstring} fastvideo.envs.maybe_convert_int
:parser: docs.source.autodoc2_docstring_parser
```
````
