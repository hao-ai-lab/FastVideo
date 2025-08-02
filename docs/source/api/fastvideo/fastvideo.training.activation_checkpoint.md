# {py:mod}`fastvideo.training.activation_checkpoint`

```{py:module} fastvideo.training.activation_checkpoint
```

```{autodoc2-docstring} fastvideo.training.activation_checkpoint
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CheckpointType <fastvideo.training.activation_checkpoint.CheckpointType>`
  -
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`apply_activation_checkpointing <fastvideo.training.activation_checkpoint.apply_activation_checkpointing>`
  - ```{autodoc2-docstring} fastvideo.training.activation_checkpoint.apply_activation_checkpointing
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TRANSFORMER_BLOCK_NAMES <fastvideo.training.activation_checkpoint.TRANSFORMER_BLOCK_NAMES>`
  - ```{autodoc2-docstring} fastvideo.training.activation_checkpoint.TRANSFORMER_BLOCK_NAMES
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} CheckpointType()
:canonical: fastvideo.training.activation_checkpoint.CheckpointType

Bases: {py:obj}`str`, {py:obj}`enum.Enum`

````{py:attribute} BLOCK_SKIP
:canonical: fastvideo.training.activation_checkpoint.CheckpointType.BLOCK_SKIP
:value: >
   'block_skip'

```{autodoc2-docstring} fastvideo.training.activation_checkpoint.CheckpointType.BLOCK_SKIP
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} FULL
:canonical: fastvideo.training.activation_checkpoint.CheckpointType.FULL
:value: >
   'full'

```{autodoc2-docstring} fastvideo.training.activation_checkpoint.CheckpointType.FULL
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} OPS
:canonical: fastvideo.training.activation_checkpoint.CheckpointType.OPS
:value: >
   'ops'

```{autodoc2-docstring} fastvideo.training.activation_checkpoint.CheckpointType.OPS
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} TRANSFORMER_BLOCK_NAMES
:canonical: fastvideo.training.activation_checkpoint.TRANSFORMER_BLOCK_NAMES
:value: >
   ['blocks', 'double_blocks', 'single_blocks', 'transformer_blocks', 'temporal_transformer_blocks', 't...

```{autodoc2-docstring} fastvideo.training.activation_checkpoint.TRANSFORMER_BLOCK_NAMES
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:function} apply_activation_checkpointing(module: torch.nn.Module, checkpointing_type: str = CheckpointType.FULL, n_layer: int = 1) -> torch.nn.Module
:canonical: fastvideo.training.activation_checkpoint.apply_activation_checkpointing

```{autodoc2-docstring} fastvideo.training.activation_checkpoint.apply_activation_checkpointing
:parser: docs.source.autodoc2_docstring_parser
```
````
