# {py:mod}`fastvideo.pipelines.stages.validators`

```{py:module} fastvideo.pipelines.stages.validators
```

```{autodoc2-docstring} fastvideo.pipelines.stages.validators
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`StageValidators <fastvideo.pipelines.stages.validators.StageValidators>`
  - ```{autodoc2-docstring} fastvideo.pipelines.stages.validators.StageValidators
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`ValidationFailure <fastvideo.pipelines.stages.validators.ValidationFailure>`
  - ```{autodoc2-docstring} fastvideo.pipelines.stages.validators.ValidationFailure
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`VerificationResult <fastvideo.pipelines.stages.validators.VerificationResult>`
  - ```{autodoc2-docstring} fastvideo.pipelines.stages.validators.VerificationResult
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`V <fastvideo.pipelines.stages.validators.V>`
  - ```{autodoc2-docstring} fastvideo.pipelines.stages.validators.V
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} StageValidators
:canonical: fastvideo.pipelines.stages.validators.StageValidators

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.StageValidators
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} bool_value(value: typing.Any) -> bool
:canonical: fastvideo.pipelines.stages.validators.StageValidators.bool_value
:staticmethod:

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.StageValidators.bool_value
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} divisible(divisor: int) -> collections.abc.Callable[[typing.Any], bool]
:canonical: fastvideo.pipelines.stages.validators.StageValidators.divisible
:staticmethod:

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.StageValidators.divisible
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} divisible_by(value: typing.Any, divisor: int) -> bool
:canonical: fastvideo.pipelines.stages.validators.StageValidators.divisible_by
:staticmethod:

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.StageValidators.divisible_by
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} generator_or_list_generators(value: typing.Any) -> bool
:canonical: fastvideo.pipelines.stages.validators.StageValidators.generator_or_list_generators
:staticmethod:

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.StageValidators.generator_or_list_generators
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} is_list(value: typing.Any) -> bool
:canonical: fastvideo.pipelines.stages.validators.StageValidators.is_list
:staticmethod:

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.StageValidators.is_list
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} is_tensor(value: typing.Any) -> bool
:canonical: fastvideo.pipelines.stages.validators.StageValidators.is_tensor
:staticmethod:

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.StageValidators.is_tensor
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} is_tuple(value: typing.Any) -> bool
:canonical: fastvideo.pipelines.stages.validators.StageValidators.is_tuple
:staticmethod:

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.StageValidators.is_tuple
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} list_length(value: typing.Any, length: int) -> bool
:canonical: fastvideo.pipelines.stages.validators.StageValidators.list_length
:staticmethod:

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.StageValidators.list_length
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} list_min_length(value: typing.Any, min_length: int) -> bool
:canonical: fastvideo.pipelines.stages.validators.StageValidators.list_min_length
:staticmethod:

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.StageValidators.list_min_length
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} list_not_empty(value: typing.Any) -> bool
:canonical: fastvideo.pipelines.stages.validators.StageValidators.list_not_empty
:staticmethod:

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.StageValidators.list_not_empty
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} list_of_tensors(value: typing.Any) -> bool
:canonical: fastvideo.pipelines.stages.validators.StageValidators.list_of_tensors
:staticmethod:

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.StageValidators.list_of_tensors
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} list_of_tensors_dims(dims: int) -> collections.abc.Callable[[typing.Any], bool]
:canonical: fastvideo.pipelines.stages.validators.StageValidators.list_of_tensors_dims
:staticmethod:

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.StageValidators.list_of_tensors_dims
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} list_of_tensors_min_dims(min_dims: int) -> collections.abc.Callable[[typing.Any], bool]
:canonical: fastvideo.pipelines.stages.validators.StageValidators.list_of_tensors_min_dims
:staticmethod:

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.StageValidators.list_of_tensors_min_dims
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} list_of_tensors_with_dims(value: typing.Any, dims: int) -> bool
:canonical: fastvideo.pipelines.stages.validators.StageValidators.list_of_tensors_with_dims
:staticmethod:

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.StageValidators.list_of_tensors_with_dims
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} list_of_tensors_with_min_dims(value: typing.Any, min_dims: int) -> bool
:canonical: fastvideo.pipelines.stages.validators.StageValidators.list_of_tensors_with_min_dims
:staticmethod:

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.StageValidators.list_of_tensors_with_min_dims
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} min_dims(min_dims: int) -> collections.abc.Callable[[typing.Any], bool]
:canonical: fastvideo.pipelines.stages.validators.StageValidators.min_dims
:staticmethod:

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.StageValidators.min_dims
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} non_negative_float(value: typing.Any) -> bool
:canonical: fastvideo.pipelines.stages.validators.StageValidators.non_negative_float
:staticmethod:

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.StageValidators.non_negative_float
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} none_or_list(value: typing.Any) -> bool
:canonical: fastvideo.pipelines.stages.validators.StageValidators.none_or_list
:staticmethod:

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.StageValidators.none_or_list
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} none_or_positive_int(value: typing.Any) -> bool
:canonical: fastvideo.pipelines.stages.validators.StageValidators.none_or_positive_int
:staticmethod:

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.StageValidators.none_or_positive_int
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} none_or_tensor(value: typing.Any) -> bool
:canonical: fastvideo.pipelines.stages.validators.StageValidators.none_or_tensor
:staticmethod:

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.StageValidators.none_or_tensor
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} none_or_tensor_with_dims(dims: int) -> collections.abc.Callable[[typing.Any], bool]
:canonical: fastvideo.pipelines.stages.validators.StageValidators.none_or_tensor_with_dims
:staticmethod:

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.StageValidators.none_or_tensor_with_dims
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} not_none(value: typing.Any) -> bool
:canonical: fastvideo.pipelines.stages.validators.StageValidators.not_none
:staticmethod:

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.StageValidators.not_none
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} positive_float(value: typing.Any) -> bool
:canonical: fastvideo.pipelines.stages.validators.StageValidators.positive_float
:staticmethod:

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.StageValidators.positive_float
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} positive_int(value: typing.Any) -> bool
:canonical: fastvideo.pipelines.stages.validators.StageValidators.positive_int
:staticmethod:

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.StageValidators.positive_int
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} positive_int_divisible(divisor: int) -> collections.abc.Callable[[typing.Any], bool]
:canonical: fastvideo.pipelines.stages.validators.StageValidators.positive_int_divisible
:staticmethod:

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.StageValidators.positive_int_divisible
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} string_not_empty(value: typing.Any) -> bool
:canonical: fastvideo.pipelines.stages.validators.StageValidators.string_not_empty
:staticmethod:

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.StageValidators.string_not_empty
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} string_or_list_strings(value: typing.Any) -> bool
:canonical: fastvideo.pipelines.stages.validators.StageValidators.string_or_list_strings
:staticmethod:

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.StageValidators.string_or_list_strings
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} tensor_min_dims(value: typing.Any, min_dims: int) -> bool
:canonical: fastvideo.pipelines.stages.validators.StageValidators.tensor_min_dims
:staticmethod:

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.StageValidators.tensor_min_dims
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} tensor_shape_matches(value: typing.Any, expected_shape: tuple) -> bool
:canonical: fastvideo.pipelines.stages.validators.StageValidators.tensor_shape_matches
:staticmethod:

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.StageValidators.tensor_shape_matches
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} tensor_with_dims(value: typing.Any, dims: int) -> bool
:canonical: fastvideo.pipelines.stages.validators.StageValidators.tensor_with_dims
:staticmethod:

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.StageValidators.tensor_with_dims
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} with_dims(dims: int) -> collections.abc.Callable[[typing.Any], bool]
:canonical: fastvideo.pipelines.stages.validators.StageValidators.with_dims
:staticmethod:

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.StageValidators.with_dims
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} V
:canonical: fastvideo.pipelines.stages.validators.V
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.V
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:class} ValidationFailure(validator_name: str, actual_value: typing.Any, expected: str | None = None, error_msg: str | None = None)
:canonical: fastvideo.pipelines.stages.validators.ValidationFailure

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.ValidationFailure
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.ValidationFailure.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````

`````{py:class} VerificationResult()
:canonical: fastvideo.pipelines.stages.validators.VerificationResult

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.VerificationResult
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.VerificationResult.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} add_check(field_name: str, value: typing.Any, validators: collections.abc.Callable[[typing.Any], bool] | list[collections.abc.Callable[[typing.Any], bool]]) -> fastvideo.pipelines.stages.validators.VerificationResult
:canonical: fastvideo.pipelines.stages.validators.VerificationResult.add_check

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.VerificationResult.add_check
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_detailed_failures() -> dict[str, list[fastvideo.pipelines.stages.validators.ValidationFailure]]
:canonical: fastvideo.pipelines.stages.validators.VerificationResult.get_detailed_failures

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.VerificationResult.get_detailed_failures
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_failed_fields() -> list[str]
:canonical: fastvideo.pipelines.stages.validators.VerificationResult.get_failed_fields

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.VerificationResult.get_failed_fields
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_failure_summary() -> str
:canonical: fastvideo.pipelines.stages.validators.VerificationResult.get_failure_summary

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.VerificationResult.get_failure_summary
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} is_valid() -> bool
:canonical: fastvideo.pipelines.stages.validators.VerificationResult.is_valid

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.VerificationResult.is_valid
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} to_dict() -> dict
:canonical: fastvideo.pipelines.stages.validators.VerificationResult.to_dict

```{autodoc2-docstring} fastvideo.pipelines.stages.validators.VerificationResult.to_dict
:parser: docs.source.autodoc2_docstring_parser
```

````

`````
