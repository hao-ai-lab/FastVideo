# {py:mod}`fastvideo.tests.utils`

```{py:module} fastvideo.tests.utils
```

```{autodoc2-docstring} fastvideo.tests.utils
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`compare_folders <fastvideo.tests.utils.compare_folders>`
  - ```{autodoc2-docstring} fastvideo.tests.utils.compare_folders
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`compute_video_ssim_torchvision <fastvideo.tests.utils.compute_video_ssim_torchvision>`
  - ```{autodoc2-docstring} fastvideo.tests.utils.compute_video_ssim_torchvision
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`write_ssim_results <fastvideo.tests.utils.write_ssim_results>`
  - ```{autodoc2-docstring} fastvideo.tests.utils.write_ssim_results
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.tests.utils.logger>`
  - ```{autodoc2-docstring} fastvideo.tests.utils.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

````{py:function} compare_folders(reference_folder, generated_folder, use_ms_ssim=True)
:canonical: fastvideo.tests.utils.compare_folders

```{autodoc2-docstring} fastvideo.tests.utils.compare_folders
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} compute_video_ssim_torchvision(video1_path, video2_path, use_ms_ssim=True)
:canonical: fastvideo.tests.utils.compute_video_ssim_torchvision

```{autodoc2-docstring} fastvideo.tests.utils.compute_video_ssim_torchvision
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:data} logger
:canonical: fastvideo.tests.utils.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.tests.utils.logger
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:function} write_ssim_results(output_dir, ssim_values, reference_path, generated_path, num_inference_steps, prompt)
:canonical: fastvideo.tests.utils.write_ssim_results

```{autodoc2-docstring} fastvideo.tests.utils.write_ssim_results
:parser: docs.source.autodoc2_docstring_parser
```
````
