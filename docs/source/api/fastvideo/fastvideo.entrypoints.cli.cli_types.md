# {py:mod}`fastvideo.entrypoints.cli.cli_types`

```{py:module} fastvideo.entrypoints.cli.cli_types
```

```{autodoc2-docstring} fastvideo.entrypoints.cli.cli_types
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CLISubcommand <fastvideo.entrypoints.cli.cli_types.CLISubcommand>`
  - ```{autodoc2-docstring} fastvideo.entrypoints.cli.cli_types.CLISubcommand
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} CLISubcommand
:canonical: fastvideo.entrypoints.cli.cli_types.CLISubcommand

```{autodoc2-docstring} fastvideo.entrypoints.cli.cli_types.CLISubcommand
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} cmd(args: argparse.Namespace) -> None
:canonical: fastvideo.entrypoints.cli.cli_types.CLISubcommand.cmd
:abstractmethod:

```{autodoc2-docstring} fastvideo.entrypoints.cli.cli_types.CLISubcommand.cmd
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} name
:canonical: fastvideo.entrypoints.cli.cli_types.CLISubcommand.name
:type: str
:value: >
   None

```{autodoc2-docstring} fastvideo.entrypoints.cli.cli_types.CLISubcommand.name
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} subparser_init(subparsers: argparse._SubParsersAction) -> fastvideo.utils.FlexibleArgumentParser
:canonical: fastvideo.entrypoints.cli.cli_types.CLISubcommand.subparser_init
:abstractmethod:

```{autodoc2-docstring} fastvideo.entrypoints.cli.cli_types.CLISubcommand.subparser_init
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} validate(args: argparse.Namespace) -> None
:canonical: fastvideo.entrypoints.cli.cli_types.CLISubcommand.validate

```{autodoc2-docstring} fastvideo.entrypoints.cli.cli_types.CLISubcommand.validate
:parser: docs.source.autodoc2_docstring_parser
```

````

`````
