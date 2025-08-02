# {py:mod}`fastvideo.STA_configuration`

```{py:module} fastvideo.STA_configuration
```

```{autodoc2-docstring} fastvideo.STA_configuration
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`average_head_losses <fastvideo.STA_configuration.average_head_losses>`
  - ```{autodoc2-docstring} fastvideo.STA_configuration.average_head_losses
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`configure_sta <fastvideo.STA_configuration.configure_sta>`
  - ```{autodoc2-docstring} fastvideo.STA_configuration.configure_sta
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`read_specific_json_files <fastvideo.STA_configuration.read_specific_json_files>`
  - ```{autodoc2-docstring} fastvideo.STA_configuration.read_specific_json_files
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`save_mask_search_results <fastvideo.STA_configuration.save_mask_search_results>`
  - ```{autodoc2-docstring} fastvideo.STA_configuration.save_mask_search_results
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`select_best_mask_strategy <fastvideo.STA_configuration.select_best_mask_strategy>`
  - ```{autodoc2-docstring} fastvideo.STA_configuration.select_best_mask_strategy
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

````{py:function} average_head_losses(results: list[dict[str, typing.Any]], selected_masks: list[list[int]]) -> dict[str, dict[str, numpy.ndarray]]
:canonical: fastvideo.STA_configuration.average_head_losses

```{autodoc2-docstring} fastvideo.STA_configuration.average_head_losses
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} configure_sta(mode: str = 'STA_searching', layer_num: int = 40, time_step_num: int = 50, head_num: int = 40, **kwargs) -> list[list[list[typing.Any]]]
:canonical: fastvideo.STA_configuration.configure_sta

```{autodoc2-docstring} fastvideo.STA_configuration.configure_sta
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} read_specific_json_files(folder_path: str) -> list[dict[str, typing.Any]]
:canonical: fastvideo.STA_configuration.read_specific_json_files

```{autodoc2-docstring} fastvideo.STA_configuration.read_specific_json_files
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} save_mask_search_results(mask_search_final_result: list[dict[str, list[float]]], prompt: str, mask_strategies: list[str], output_dir: str = 'output/mask_search_result/') -> str | None
:canonical: fastvideo.STA_configuration.save_mask_search_results

```{autodoc2-docstring} fastvideo.STA_configuration.save_mask_search_results
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} select_best_mask_strategy(averaged_results: dict[str, dict[str, numpy.ndarray]], selected_masks: list[list[int]], skip_time_steps: int = 12, timesteps: int = 50, head_num: int = 40) -> tuple[dict[str, list[int]], float, dict[str, int]]
:canonical: fastvideo.STA_configuration.select_best_mask_strategy

```{autodoc2-docstring} fastvideo.STA_configuration.select_best_mask_strategy
:parser: docs.source.autodoc2_docstring_parser
```
````
