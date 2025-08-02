# {py:mod}`fastvideo.training.training_utils`

```{py:module} fastvideo.training.training_utils
```

```{autodoc2-docstring} fastvideo.training.training_utils
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`clip_grad_norm_ <fastvideo.training.training_utils.clip_grad_norm_>`
  - ```{autodoc2-docstring} fastvideo.training.training_utils.clip_grad_norm_
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`clip_grad_norm_while_handling_failing_dtensor_cases <fastvideo.training.training_utils.clip_grad_norm_while_handling_failing_dtensor_cases>`
  - ```{autodoc2-docstring} fastvideo.training.training_utils.clip_grad_norm_while_handling_failing_dtensor_cases
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`compute_density_for_timestep_sampling <fastvideo.training.training_utils.compute_density_for_timestep_sampling>`
  - ```{autodoc2-docstring} fastvideo.training.training_utils.compute_density_for_timestep_sampling
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`custom_to_hf_state_dict <fastvideo.training.training_utils.custom_to_hf_state_dict>`
  - ```{autodoc2-docstring} fastvideo.training.training_utils.custom_to_hf_state_dict
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`gather_state_dict_on_cpu_rank0 <fastvideo.training.training_utils.gather_state_dict_on_cpu_rank0>`
  - ```{autodoc2-docstring} fastvideo.training.training_utils.gather_state_dict_on_cpu_rank0
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`get_sigmas <fastvideo.training.training_utils.get_sigmas>`
  - ```{autodoc2-docstring} fastvideo.training.training_utils.get_sigmas
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`load_checkpoint <fastvideo.training.training_utils.load_checkpoint>`
  - ```{autodoc2-docstring} fastvideo.training.training_utils.load_checkpoint
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`load_distillation_checkpoint <fastvideo.training.training_utils.load_distillation_checkpoint>`
  - ```{autodoc2-docstring} fastvideo.training.training_utils.load_distillation_checkpoint
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`normalize_dit_input <fastvideo.training.training_utils.normalize_dit_input>`
  - ```{autodoc2-docstring} fastvideo.training.training_utils.normalize_dit_input
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`pred_noise_to_pred_video <fastvideo.training.training_utils.pred_noise_to_pred_video>`
  - ```{autodoc2-docstring} fastvideo.training.training_utils.pred_noise_to_pred_video
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`save_checkpoint <fastvideo.training.training_utils.save_checkpoint>`
  - ```{autodoc2-docstring} fastvideo.training.training_utils.save_checkpoint
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`save_distillation_checkpoint <fastvideo.training.training_utils.save_distillation_checkpoint>`
  - ```{autodoc2-docstring} fastvideo.training.training_utils.save_distillation_checkpoint
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`shard_latents_across_sp <fastvideo.training.training_utils.shard_latents_across_sp>`
  - ```{autodoc2-docstring} fastvideo.training.training_utils.shard_latents_across_sp
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`shift_timestep <fastvideo.training.training_utils.shift_timestep>`
  - ```{autodoc2-docstring} fastvideo.training.training_utils.shift_timestep
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.training.training_utils.logger>`
  - ```{autodoc2-docstring} fastvideo.training.training_utils.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

````{py:function} clip_grad_norm_(parameters: torch.Tensor | list[torch.Tensor], max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False, foreach: bool | None = None, pp_mesh: torch.distributed.device_mesh.DeviceMesh | None = None) -> torch.Tensor
:canonical: fastvideo.training.training_utils.clip_grad_norm_

```{autodoc2-docstring} fastvideo.training.training_utils.clip_grad_norm_
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} clip_grad_norm_while_handling_failing_dtensor_cases(parameters: torch.Tensor | list[torch.Tensor], max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False, foreach: bool | None = None, pp_mesh: torch.distributed.device_mesh.DeviceMesh | None = None) -> torch.Tensor | None
:canonical: fastvideo.training.training_utils.clip_grad_norm_while_handling_failing_dtensor_cases

```{autodoc2-docstring} fastvideo.training.training_utils.clip_grad_norm_while_handling_failing_dtensor_cases
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} compute_density_for_timestep_sampling(weighting_scheme: str, batch_size: int, generator, logit_mean: float | None = None, logit_std: float | None = None, mode_scale: float | None = None)
:canonical: fastvideo.training.training_utils.compute_density_for_timestep_sampling

```{autodoc2-docstring} fastvideo.training.training_utils.compute_density_for_timestep_sampling
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} custom_to_hf_state_dict(state_dict: dict[str, typing.Any] | collections.abc.Iterator[tuple[str, torch.Tensor]], reverse_param_names_mapping: dict[str, tuple[str, int, int]]) -> dict[str, typing.Any]
:canonical: fastvideo.training.training_utils.custom_to_hf_state_dict

```{autodoc2-docstring} fastvideo.training.training_utils.custom_to_hf_state_dict
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} gather_state_dict_on_cpu_rank0(model, device: torch.device | None = None) -> dict[str, typing.Any]
:canonical: fastvideo.training.training_utils.gather_state_dict_on_cpu_rank0

```{autodoc2-docstring} fastvideo.training.training_utils.gather_state_dict_on_cpu_rank0
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} get_sigmas(noise_scheduler, device, timesteps, n_dim=4, dtype=torch.float32) -> torch.Tensor
:canonical: fastvideo.training.training_utils.get_sigmas

```{autodoc2-docstring} fastvideo.training.training_utils.get_sigmas
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} load_checkpoint(transformer, rank, checkpoint_path, optimizer=None, dataloader=None, scheduler=None, noise_generator=None) -> int
:canonical: fastvideo.training.training_utils.load_checkpoint

```{autodoc2-docstring} fastvideo.training.training_utils.load_checkpoint
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} load_distillation_checkpoint(generator_transformer, fake_score_transformer, rank, checkpoint_path, generator_optimizer=None, fake_score_optimizer=None, dataloader=None, generator_scheduler=None, fake_score_scheduler=None, noise_generator=None) -> int
:canonical: fastvideo.training.training_utils.load_distillation_checkpoint

```{autodoc2-docstring} fastvideo.training.training_utils.load_distillation_checkpoint
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:data} logger
:canonical: fastvideo.training.training_utils.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.training.training_utils.logger
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:function} normalize_dit_input(model_type, latents, vae) -> torch.Tensor
:canonical: fastvideo.training.training_utils.normalize_dit_input

```{autodoc2-docstring} fastvideo.training.training_utils.normalize_dit_input
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} pred_noise_to_pred_video(pred_noise: torch.Tensor, noise_input_latent: torch.Tensor, timestep: torch.Tensor, scheduler: typing.Any) -> torch.Tensor
:canonical: fastvideo.training.training_utils.pred_noise_to_pred_video

```{autodoc2-docstring} fastvideo.training.training_utils.pred_noise_to_pred_video
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} save_checkpoint(transformer, rank, output_dir, step, optimizer=None, dataloader=None, scheduler=None, noise_generator=None) -> None
:canonical: fastvideo.training.training_utils.save_checkpoint

```{autodoc2-docstring} fastvideo.training.training_utils.save_checkpoint
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} save_distillation_checkpoint(generator_transformer, fake_score_transformer, rank, output_dir, step, generator_optimizer=None, fake_score_optimizer=None, dataloader=None, generator_scheduler=None, fake_score_scheduler=None, noise_generator=None, only_save_generator_weight=False) -> None
:canonical: fastvideo.training.training_utils.save_distillation_checkpoint

```{autodoc2-docstring} fastvideo.training.training_utils.save_distillation_checkpoint
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} shard_latents_across_sp(latents: torch.Tensor, num_latent_t: int) -> torch.Tensor
:canonical: fastvideo.training.training_utils.shard_latents_across_sp

```{autodoc2-docstring} fastvideo.training.training_utils.shard_latents_across_sp
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} shift_timestep(timestep: torch.Tensor, shift: float, num_train_timestep: float) -> torch.Tensor
:canonical: fastvideo.training.training_utils.shift_timestep

```{autodoc2-docstring} fastvideo.training.training_utils.shift_timestep
:parser: docs.source.autodoc2_docstring_parser
```
````
