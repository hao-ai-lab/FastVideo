# {py:mod}`fastvideo.entrypoints.cli.utils`

```{py:module} fastvideo.entrypoints.cli.utils
```

```{autodoc2-docstring} fastvideo.entrypoints.cli.utils
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RaiseNotImplementedAction <fastvideo.entrypoints.cli.utils.RaiseNotImplementedAction>`
  -
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`launch_distributed <fastvideo.entrypoints.cli.utils.launch_distributed>`
  - ```{autodoc2-docstring} fastvideo.entrypoints.cli.utils.launch_distributed
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.entrypoints.cli.utils.logger>`
  - ```{autodoc2-docstring} fastvideo.entrypoints.cli.utils.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

```{py:class} RaiseNotImplementedAction(option_strings, dest, nargs=None, const=None, default=None, type=None, choices=None, required=False, help=None, metavar=None)
:canonical: fastvideo.entrypoints.cli.utils.RaiseNotImplementedAction

Bases: {py:obj}`argparse.Action`

```

````{py:function} launch_distributed(num_gpus: int, args: list[str], master_port: int | None = None) -> int
:canonical: fastvideo.entrypoints.cli.utils.launch_distributed

```{autodoc2-docstring} fastvideo.entrypoints.cli.utils.launch_distributed
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:data} logger
:canonical: fastvideo.entrypoints.cli.utils.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.entrypoints.cli.utils.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
