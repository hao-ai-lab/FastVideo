# {py:mod}`fastvideo.training.checkpointing_utils`

```{py:module} fastvideo.training.checkpointing_utils
```

```{autodoc2-docstring} fastvideo.training.checkpointing_utils
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ModelWrapper <fastvideo.training.checkpointing_utils.ModelWrapper>`
  -
* - {py:obj}`OptimizerWrapper <fastvideo.training.checkpointing_utils.OptimizerWrapper>`
  -
* - {py:obj}`RandomStateWrapper <fastvideo.training.checkpointing_utils.RandomStateWrapper>`
  -
* - {py:obj}`SchedulerWrapper <fastvideo.training.checkpointing_utils.SchedulerWrapper>`
  -
````

### API

`````{py:class} ModelWrapper(model: torch.nn.Module)
:canonical: fastvideo.training.checkpointing_utils.ModelWrapper

Bases: {py:obj}`torch.distributed.checkpoint.stateful.Stateful`

````{py:method} load_state_dict(state_dict: dict[str, typing.Any]) -> None
:canonical: fastvideo.training.checkpointing_utils.ModelWrapper.load_state_dict

````

````{py:method} state_dict() -> dict[str, typing.Any]
:canonical: fastvideo.training.checkpointing_utils.ModelWrapper.state_dict

````

`````

`````{py:class} OptimizerWrapper(model: torch.nn.Module, optimizer: torch.optim.Optimizer)
:canonical: fastvideo.training.checkpointing_utils.OptimizerWrapper

Bases: {py:obj}`torch.distributed.checkpoint.stateful.Stateful`

````{py:method} load_state_dict(state_dict: dict[str, typing.Any]) -> None
:canonical: fastvideo.training.checkpointing_utils.OptimizerWrapper.load_state_dict

````

````{py:method} state_dict() -> dict[str, typing.Any]
:canonical: fastvideo.training.checkpointing_utils.OptimizerWrapper.state_dict

````

`````

`````{py:class} RandomStateWrapper(noise_generator: torch.Generator | None = None)
:canonical: fastvideo.training.checkpointing_utils.RandomStateWrapper

Bases: {py:obj}`torch.distributed.checkpoint.stateful.Stateful`

````{py:method} load_state_dict(state_dict: dict[str, typing.Any]) -> None
:canonical: fastvideo.training.checkpointing_utils.RandomStateWrapper.load_state_dict

````

````{py:method} state_dict() -> dict[str, typing.Any]
:canonical: fastvideo.training.checkpointing_utils.RandomStateWrapper.state_dict

````

`````

`````{py:class} SchedulerWrapper(scheduler)
:canonical: fastvideo.training.checkpointing_utils.SchedulerWrapper

Bases: {py:obj}`torch.distributed.checkpoint.stateful.Stateful`

````{py:method} load_state_dict(state_dict: dict[str, typing.Any]) -> None
:canonical: fastvideo.training.checkpointing_utils.SchedulerWrapper.load_state_dict

````

````{py:method} state_dict() -> dict[str, typing.Any]
:canonical: fastvideo.training.checkpointing_utils.SchedulerWrapper.state_dict

````

`````
