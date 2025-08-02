# {py:mod}`fastvideo.entrypoints.cli.generate`

```{py:module} fastvideo.entrypoints.cli.generate
```

```{autodoc2-docstring} fastvideo.entrypoints.cli.generate
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GenerateSubcommand <fastvideo.entrypoints.cli.generate.GenerateSubcommand>`
  - ```{autodoc2-docstring} fastvideo.entrypoints.cli.generate.GenerateSubcommand
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`cmd_init <fastvideo.entrypoints.cli.generate.cmd_init>`
  - ```{autodoc2-docstring} fastvideo.entrypoints.cli.generate.cmd_init
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.entrypoints.cli.generate.logger>`
  - ```{autodoc2-docstring} fastvideo.entrypoints.cli.generate.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} GenerateSubcommand()
:canonical: fastvideo.entrypoints.cli.generate.GenerateSubcommand

Bases: {py:obj}`fastvideo.entrypoints.cli.cli_types.CLISubcommand`

```{autodoc2-docstring} fastvideo.entrypoints.cli.generate.GenerateSubcommand
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.entrypoints.cli.generate.GenerateSubcommand.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} cmd(args: argparse.Namespace) -> None
:canonical: fastvideo.entrypoints.cli.generate.GenerateSubcommand.cmd

````

````{py:method} subparser_init(subparsers: argparse._SubParsersAction) -> fastvideo.utils.FlexibleArgumentParser
:canonical: fastvideo.entrypoints.cli.generate.GenerateSubcommand.subparser_init

````

````{py:method} validate(args: argparse.Namespace) -> None
:canonical: fastvideo.entrypoints.cli.generate.GenerateSubcommand.validate

```{autodoc2-docstring} fastvideo.entrypoints.cli.generate.GenerateSubcommand.validate
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:function} cmd_init() -> list[fastvideo.entrypoints.cli.cli_types.CLISubcommand]
:canonical: fastvideo.entrypoints.cli.generate.cmd_init

```{autodoc2-docstring} fastvideo.entrypoints.cli.generate.cmd_init
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:data} logger
:canonical: fastvideo.entrypoints.cli.generate.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.entrypoints.cli.generate.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
