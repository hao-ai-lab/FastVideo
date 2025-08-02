# {py:mod}`fastvideo.layers.custom_op`

```{py:module} fastvideo.layers.custom_op
```

```{autodoc2-docstring} fastvideo.layers.custom_op
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CustomOp <fastvideo.layers.custom_op.CustomOp>`
  - ```{autodoc2-docstring} fastvideo.layers.custom_op.CustomOp
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.layers.custom_op.logger>`
  - ```{autodoc2-docstring} fastvideo.layers.custom_op.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} CustomOp()
:canonical: fastvideo.layers.custom_op.CustomOp

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} fastvideo.layers.custom_op.CustomOp
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.layers.custom_op.CustomOp.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} default_on() -> bool
:canonical: fastvideo.layers.custom_op.CustomOp.default_on
:abstractmethod:
:staticmethod:

```{autodoc2-docstring} fastvideo.layers.custom_op.CustomOp.default_on
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} dispatch_forward() -> collections.abc.Callable
:canonical: fastvideo.layers.custom_op.CustomOp.dispatch_forward

```{autodoc2-docstring} fastvideo.layers.custom_op.CustomOp.dispatch_forward
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} enabled() -> bool
:canonical: fastvideo.layers.custom_op.CustomOp.enabled
:classmethod:

```{autodoc2-docstring} fastvideo.layers.custom_op.CustomOp.enabled
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} forward(*args, **kwargs) -> typing.Any
:canonical: fastvideo.layers.custom_op.CustomOp.forward

```{autodoc2-docstring} fastvideo.layers.custom_op.CustomOp.forward
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} forward_cpu(*args, **kwargs) -> typing.Any
:canonical: fastvideo.layers.custom_op.CustomOp.forward_cpu

```{autodoc2-docstring} fastvideo.layers.custom_op.CustomOp.forward_cpu
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} forward_cuda(*args, **kwargs) -> typing.Any
:canonical: fastvideo.layers.custom_op.CustomOp.forward_cuda
:abstractmethod:

```{autodoc2-docstring} fastvideo.layers.custom_op.CustomOp.forward_cuda
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} forward_native(*args, **kwargs) -> typing.Any
:canonical: fastvideo.layers.custom_op.CustomOp.forward_native
:abstractmethod:

```{autodoc2-docstring} fastvideo.layers.custom_op.CustomOp.forward_native
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} forward_oot(*args, **kwargs) -> typing.Any
:canonical: fastvideo.layers.custom_op.CustomOp.forward_oot

```{autodoc2-docstring} fastvideo.layers.custom_op.CustomOp.forward_oot
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} forward_tpu(*args, **kwargs) -> typing.Any
:canonical: fastvideo.layers.custom_op.CustomOp.forward_tpu

```{autodoc2-docstring} fastvideo.layers.custom_op.CustomOp.forward_tpu
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} op_registry
:canonical: fastvideo.layers.custom_op.CustomOp.op_registry
:type: dict[str, type[fastvideo.layers.custom_op.CustomOp]]
:value: >
   None

```{autodoc2-docstring} fastvideo.layers.custom_op.CustomOp.op_registry
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} register(name: str) -> collections.abc.Callable
:canonical: fastvideo.layers.custom_op.CustomOp.register
:classmethod:

```{autodoc2-docstring} fastvideo.layers.custom_op.CustomOp.register
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} logger
:canonical: fastvideo.layers.custom_op.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.layers.custom_op.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
