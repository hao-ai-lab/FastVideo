# {py:mod}`fastvideo.configs.models.base`

```{py:module} fastvideo.configs.models.base
```

```{autodoc2-docstring} fastvideo.configs.models.base
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ArchConfig <fastvideo.configs.models.base.ArchConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.models.base.ArchConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`ModelConfig <fastvideo.configs.models.base.ModelConfig>`
  - ```{autodoc2-docstring} fastvideo.configs.models.base.ModelConfig
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.configs.models.base.logger>`
  - ```{autodoc2-docstring} fastvideo.configs.models.base.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} ArchConfig
:canonical: fastvideo.configs.models.base.ArchConfig

```{autodoc2-docstring} fastvideo.configs.models.base.ArchConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} stacked_params_mapping
:canonical: fastvideo.configs.models.base.ArchConfig.stacked_params_mapping
:type: list[tuple[str, str, str]]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.base.ArchConfig.stacked_params_mapping
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} ModelConfig
:canonical: fastvideo.configs.models.base.ModelConfig

```{autodoc2-docstring} fastvideo.configs.models.base.ModelConfig
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} arch_config
:canonical: fastvideo.configs.models.base.ModelConfig.arch_config
:type: fastvideo.configs.models.base.ArchConfig
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.configs.models.base.ModelConfig.arch_config
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} update_model_arch(source_model_dict: dict[str, typing.Any]) -> None
:canonical: fastvideo.configs.models.base.ModelConfig.update_model_arch

```{autodoc2-docstring} fastvideo.configs.models.base.ModelConfig.update_model_arch
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} update_model_config(source_model_dict: dict[str, typing.Any]) -> None
:canonical: fastvideo.configs.models.base.ModelConfig.update_model_config

```{autodoc2-docstring} fastvideo.configs.models.base.ModelConfig.update_model_config
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} logger
:canonical: fastvideo.configs.models.base.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.configs.models.base.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
