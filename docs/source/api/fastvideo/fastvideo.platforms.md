# {py:mod}`fastvideo.platforms`

```{py:module} fastvideo.platforms
```

```{autodoc2-docstring} fastvideo.platforms
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

fastvideo.platforms.cpu
fastvideo.platforms.cuda
fastvideo.platforms.interface
fastvideo.platforms.mps
fastvideo.platforms.rocm
```

## Package Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`cpu_platform_plugin <fastvideo.platforms.cpu_platform_plugin>`
  - ```{autodoc2-docstring} fastvideo.platforms.cpu_platform_plugin
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`cuda_platform_plugin <fastvideo.platforms.cuda_platform_plugin>`
  - ```{autodoc2-docstring} fastvideo.platforms.cuda_platform_plugin
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`mps_platform_plugin <fastvideo.platforms.mps_platform_plugin>`
  - ```{autodoc2-docstring} fastvideo.platforms.mps_platform_plugin
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`resolve_current_platform_cls_qualname <fastvideo.platforms.resolve_current_platform_cls_qualname>`
  - ```{autodoc2-docstring} fastvideo.platforms.resolve_current_platform_cls_qualname
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`rocm_platform_plugin <fastvideo.platforms.rocm_platform_plugin>`
  - ```{autodoc2-docstring} fastvideo.platforms.rocm_platform_plugin
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`builtin_platform_plugins <fastvideo.platforms.builtin_platform_plugins>`
  - ```{autodoc2-docstring} fastvideo.platforms.builtin_platform_plugins
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`logger <fastvideo.platforms.logger>`
  - ```{autodoc2-docstring} fastvideo.platforms.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

````{py:data} builtin_platform_plugins
:canonical: fastvideo.platforms.builtin_platform_plugins
:value: >
   None

```{autodoc2-docstring} fastvideo.platforms.builtin_platform_plugins
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:function} cpu_platform_plugin() -> str | None
:canonical: fastvideo.platforms.cpu_platform_plugin

```{autodoc2-docstring} fastvideo.platforms.cpu_platform_plugin
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} cuda_platform_plugin() -> str | None
:canonical: fastvideo.platforms.cuda_platform_plugin

```{autodoc2-docstring} fastvideo.platforms.cuda_platform_plugin
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:data} logger
:canonical: fastvideo.platforms.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.platforms.logger
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:function} mps_platform_plugin() -> str | None
:canonical: fastvideo.platforms.mps_platform_plugin

```{autodoc2-docstring} fastvideo.platforms.mps_platform_plugin
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} resolve_current_platform_cls_qualname() -> str
:canonical: fastvideo.platforms.resolve_current_platform_cls_qualname

```{autodoc2-docstring} fastvideo.platforms.resolve_current_platform_cls_qualname
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} rocm_platform_plugin() -> str | None
:canonical: fastvideo.platforms.rocm_platform_plugin

```{autodoc2-docstring} fastvideo.platforms.rocm_platform_plugin
:parser: docs.source.autodoc2_docstring_parser
```
````
