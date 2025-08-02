# {py:mod}`fastvideo.pipelines.stages.denoising`

```{py:module} fastvideo.pipelines.stages.denoising
```

```{autodoc2-docstring} fastvideo.pipelines.stages.denoising
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DenoisingStage <fastvideo.pipelines.stages.denoising.DenoisingStage>`
  - ```{autodoc2-docstring} fastvideo.pipelines.stages.denoising.DenoisingStage
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`DmdDenoisingStage <fastvideo.pipelines.stages.denoising.DmdDenoisingStage>`
  - ```{autodoc2-docstring} fastvideo.pipelines.stages.denoising.DmdDenoisingStage
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.pipelines.stages.denoising.logger>`
  - ```{autodoc2-docstring} fastvideo.pipelines.stages.denoising.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} DenoisingStage(transformer, scheduler, pipeline=None)
:canonical: fastvideo.pipelines.stages.denoising.DenoisingStage

Bases: {py:obj}`fastvideo.pipelines.stages.base.PipelineStage`

```{autodoc2-docstring} fastvideo.pipelines.stages.denoising.DenoisingStage
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.pipelines.stages.denoising.DenoisingStage.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} forward(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.pipeline_batch_info.ForwardBatch
:canonical: fastvideo.pipelines.stages.denoising.DenoisingStage.forward

```{autodoc2-docstring} fastvideo.pipelines.stages.denoising.DenoisingStage.forward
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} prepare_extra_func_kwargs(func, kwargs) -> dict[str, typing.Any]
:canonical: fastvideo.pipelines.stages.denoising.DenoisingStage.prepare_extra_func_kwargs

```{autodoc2-docstring} fastvideo.pipelines.stages.denoising.DenoisingStage.prepare_extra_func_kwargs
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} prepare_sta_param(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs)
:canonical: fastvideo.pipelines.stages.denoising.DenoisingStage.prepare_sta_param

```{autodoc2-docstring} fastvideo.pipelines.stages.denoising.DenoisingStage.prepare_sta_param
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} progress_bar(iterable: collections.abc.Iterable | None = None, total: int | None = None) -> tqdm.auto.tqdm
:canonical: fastvideo.pipelines.stages.denoising.DenoisingStage.progress_bar

```{autodoc2-docstring} fastvideo.pipelines.stages.denoising.DenoisingStage.progress_bar
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0) -> torch.Tensor
:canonical: fastvideo.pipelines.stages.denoising.DenoisingStage.rescale_noise_cfg

```{autodoc2-docstring} fastvideo.pipelines.stages.denoising.DenoisingStage.rescale_noise_cfg
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} save_sta_search_results(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch)
:canonical: fastvideo.pipelines.stages.denoising.DenoisingStage.save_sta_search_results

```{autodoc2-docstring} fastvideo.pipelines.stages.denoising.DenoisingStage.save_sta_search_results
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} verify_input(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.stages.validators.VerificationResult
:canonical: fastvideo.pipelines.stages.denoising.DenoisingStage.verify_input

```{autodoc2-docstring} fastvideo.pipelines.stages.denoising.DenoisingStage.verify_input
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} verify_output(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.stages.validators.VerificationResult
:canonical: fastvideo.pipelines.stages.denoising.DenoisingStage.verify_output

```{autodoc2-docstring} fastvideo.pipelines.stages.denoising.DenoisingStage.verify_output
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} DmdDenoisingStage(transformer, scheduler)
:canonical: fastvideo.pipelines.stages.denoising.DmdDenoisingStage

Bases: {py:obj}`fastvideo.pipelines.stages.denoising.DenoisingStage`

```{autodoc2-docstring} fastvideo.pipelines.stages.denoising.DmdDenoisingStage
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.pipelines.stages.denoising.DmdDenoisingStage.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} forward(batch: fastvideo.pipelines.pipeline_batch_info.ForwardBatch, fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs) -> fastvideo.pipelines.pipeline_batch_info.ForwardBatch
:canonical: fastvideo.pipelines.stages.denoising.DmdDenoisingStage.forward

```{autodoc2-docstring} fastvideo.pipelines.stages.denoising.DmdDenoisingStage.forward
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} logger
:canonical: fastvideo.pipelines.stages.denoising.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.pipelines.stages.denoising.logger
:parser: docs.source.autodoc2_docstring_parser
```

````
