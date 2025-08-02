# {py:mod}`fastvideo.distributed.communication_op`

```{py:module} fastvideo.distributed.communication_op
```

```{autodoc2-docstring} fastvideo.distributed.communication_op
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`sequence_model_parallel_all_gather <fastvideo.distributed.communication_op.sequence_model_parallel_all_gather>`
  - ```{autodoc2-docstring} fastvideo.distributed.communication_op.sequence_model_parallel_all_gather
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`sequence_model_parallel_all_to_all_4D <fastvideo.distributed.communication_op.sequence_model_parallel_all_to_all_4D>`
  - ```{autodoc2-docstring} fastvideo.distributed.communication_op.sequence_model_parallel_all_to_all_4D
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`tensor_model_parallel_all_gather <fastvideo.distributed.communication_op.tensor_model_parallel_all_gather>`
  - ```{autodoc2-docstring} fastvideo.distributed.communication_op.tensor_model_parallel_all_gather
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`tensor_model_parallel_all_reduce <fastvideo.distributed.communication_op.tensor_model_parallel_all_reduce>`
  - ```{autodoc2-docstring} fastvideo.distributed.communication_op.tensor_model_parallel_all_reduce
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

````{py:function} sequence_model_parallel_all_gather(input_: torch.Tensor, dim: int = -1) -> torch.Tensor
:canonical: fastvideo.distributed.communication_op.sequence_model_parallel_all_gather

```{autodoc2-docstring} fastvideo.distributed.communication_op.sequence_model_parallel_all_gather
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} sequence_model_parallel_all_to_all_4D(input_: torch.Tensor, scatter_dim: int = 2, gather_dim: int = 1) -> torch.Tensor
:canonical: fastvideo.distributed.communication_op.sequence_model_parallel_all_to_all_4D

```{autodoc2-docstring} fastvideo.distributed.communication_op.sequence_model_parallel_all_to_all_4D
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} tensor_model_parallel_all_gather(input_: torch.Tensor, dim: int = -1) -> torch.Tensor
:canonical: fastvideo.distributed.communication_op.tensor_model_parallel_all_gather

```{autodoc2-docstring} fastvideo.distributed.communication_op.tensor_model_parallel_all_gather
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor
:canonical: fastvideo.distributed.communication_op.tensor_model_parallel_all_reduce

```{autodoc2-docstring} fastvideo.distributed.communication_op.tensor_model_parallel_all_reduce
:parser: docs.source.autodoc2_docstring_parser
```
````
