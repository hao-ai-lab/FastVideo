# {py:mod}`fastvideo.layers.quantization`

```{py:module} fastvideo.layers.quantization
```

```{autodoc2-docstring} fastvideo.layers.quantization
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

fastvideo.layers.quantization.base_config
```

## Package Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_quantization_config <fastvideo.layers.quantization.get_quantization_config>`
  - ```{autodoc2-docstring} fastvideo.layers.quantization.get_quantization_config
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`register_quantization_config <fastvideo.layers.quantization.register_quantization_config>`
  - ```{autodoc2-docstring} fastvideo.layers.quantization.register_quantization_config
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`QUANTIZATION_METHODS <fastvideo.layers.quantization.QUANTIZATION_METHODS>`
  - ```{autodoc2-docstring} fastvideo.layers.quantization.QUANTIZATION_METHODS
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`QuantizationMethods <fastvideo.layers.quantization.QuantizationMethods>`
  - ```{autodoc2-docstring} fastvideo.layers.quantization.QuantizationMethods
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`all <fastvideo.layers.quantization.all>`
  - ```{autodoc2-docstring} fastvideo.layers.quantization.all
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

````{py:data} QUANTIZATION_METHODS
:canonical: fastvideo.layers.quantization.QUANTIZATION_METHODS
:type: list[str]
:value: >
   'list(...)'

```{autodoc2-docstring} fastvideo.layers.quantization.QUANTIZATION_METHODS
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:data} QuantizationMethods
:canonical: fastvideo.layers.quantization.QuantizationMethods
:value: >
   None

```{autodoc2-docstring} fastvideo.layers.quantization.QuantizationMethods
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:data} all
:canonical: fastvideo.layers.quantization.all
:value: >
   ['QuantizationMethods', 'QuantizationConfig', 'get_quantization_config', 'QUANTIZATION_METHODS']

```{autodoc2-docstring} fastvideo.layers.quantization.all
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:function} get_quantization_config(quantization: str) -> type[fastvideo.layers.quantization.base_config.QuantizationConfig]
:canonical: fastvideo.layers.quantization.get_quantization_config

```{autodoc2-docstring} fastvideo.layers.quantization.get_quantization_config
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} register_quantization_config(quantization: str)
:canonical: fastvideo.layers.quantization.register_quantization_config

```{autodoc2-docstring} fastvideo.layers.quantization.register_quantization_config
:parser: docs.source.autodoc2_docstring_parser
```
````
