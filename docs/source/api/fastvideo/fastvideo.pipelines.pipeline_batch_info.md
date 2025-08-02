# {py:mod}`fastvideo.pipelines.pipeline_batch_info`

```{py:module} fastvideo.pipelines.pipeline_batch_info
```

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ForwardBatch <fastvideo.pipelines.pipeline_batch_info.ForwardBatch>`
  - ```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`TrainingBatch <fastvideo.pipelines.pipeline_batch_info.TrainingBatch>`
  - ```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.TrainingBatch
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} ForwardBatch
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} STA_param
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.STA_param
:type: list | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.STA_param
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} VSA_sparsity
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.VSA_sparsity
:type: float
:value: >
   0.0

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.VSA_sparsity
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} batch_size
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.batch_size
:type: int | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.batch_size
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} clip_embedding_neg
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.clip_embedding_neg
:type: list[torch.Tensor] | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.clip_embedding_neg
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} clip_embedding_pos
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.clip_embedding_pos
:type: list[torch.Tensor] | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.clip_embedding_pos
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} data_type
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.data_type
:type: str
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.data_type
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} do_classifier_free_guidance
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.do_classifier_free_guidance
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.do_classifier_free_guidance
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} enable_teacache
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.enable_teacache
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.enable_teacache
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} eta
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.eta
:type: float
:value: >
   0.0

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.eta
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} extra
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.extra
:type: dict[str, typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.extra
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} extra_step_kwargs
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.extra_step_kwargs
:type: dict[str, typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.extra_step_kwargs
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} fps
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.fps
:type: int | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.fps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} generator
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.generator
:type: torch.Generator | list[torch.Generator] | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.generator
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} guidance_rescale
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.guidance_rescale
:type: float
:value: >
   0.0

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.guidance_rescale
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} guidance_scale
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.guidance_scale
:type: float
:value: >
   1.0

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.guidance_scale
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} height
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.height
:type: int | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.height
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} height_latents
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.height_latents
:type: int | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.height_latents
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} image_embeds
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.image_embeds
:type: list[torch.Tensor]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.image_embeds
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} image_latent
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.image_latent
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.image_latent
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} image_path
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.image_path
:type: str | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.image_path
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} is_cfg_negative
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.is_cfg_negative
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.is_cfg_negative
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} is_prompt_processed
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.is_prompt_processed
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.is_prompt_processed
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} latents
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.latents
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.latents
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} mask_search_final_result_neg
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.mask_search_final_result_neg
:type: list[list] | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.mask_search_final_result_neg
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} mask_search_final_result_pos
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.mask_search_final_result_pos
:type: list[list] | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.mask_search_final_result_pos
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} max_sequence_length
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.max_sequence_length
:type: int | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.max_sequence_length
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} modules
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.modules
:type: dict[str, typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.modules
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} n_tokens
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.n_tokens
:type: int | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.n_tokens
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} negative_attention_mask
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.negative_attention_mask
:type: list[torch.Tensor] | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.negative_attention_mask
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} negative_prompt
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.negative_prompt
:type: str | list[str] | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.negative_prompt
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} negative_prompt_embeds
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.negative_prompt_embeds
:type: list[torch.Tensor] | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.negative_prompt_embeds
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} noise_pred
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.noise_pred
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.noise_pred
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_frames
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.num_frames
:type: int
:value: >
   1

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.num_frames
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_frames_round_down
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.num_frames_round_down
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.num_frames_round_down
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_inference_steps
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.num_inference_steps
:type: int
:value: >
   50

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.num_inference_steps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_videos_per_prompt
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.num_videos_per_prompt
:type: int
:value: >
   1

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.num_videos_per_prompt
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} output
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.output
:type: typing.Any
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.output
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} output_path
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.output_path
:type: str
:value: >
   'outputs/'

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.output_path
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} output_video_name
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.output_video_name
:type: str | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.output_video_name
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} pil_image
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.pil_image
:type: PIL.Image.Image | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.pil_image
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} preprocessed_image
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.preprocessed_image
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.preprocessed_image
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} prompt
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.prompt
:type: str | list[str] | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.prompt
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} prompt_attention_mask
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.prompt_attention_mask
:type: list[torch.Tensor] | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.prompt_attention_mask
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} prompt_embeds
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.prompt_embeds
:type: list[torch.Tensor]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.prompt_embeds
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} prompt_path
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.prompt_path
:type: str | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.prompt_path
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} prompt_template
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.prompt_template
:type: dict[str, typing.Any] | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.prompt_template
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} raw_latent_shape
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.raw_latent_shape
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.raw_latent_shape
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} return_frames
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.return_frames
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.return_frames
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} save_video
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.save_video
:type: bool
:value: >
   True

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.save_video
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} seed
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.seed
:type: int | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.seed
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} seeds
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.seeds
:type: list[int] | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.seeds
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} sigmas
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.sigmas
:type: list[float] | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.sigmas
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} step_index
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.step_index
:type: int | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.step_index
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} teacache_params
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.teacache_params
:type: fastvideo.configs.sample.teacache.TeaCacheParams | fastvideo.configs.sample.teacache.WanTeaCacheParams | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.teacache_params
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} timestep
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.timestep
:type: torch.Tensor | float | int | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.timestep
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} timesteps
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.timesteps
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.timesteps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} width
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.width
:type: int | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.width
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} width_latents
:canonical: fastvideo.pipelines.pipeline_batch_info.ForwardBatch.width_latents
:type: int | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.ForwardBatch.width_latents
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} TrainingBatch
:canonical: fastvideo.pipelines.pipeline_batch_info.TrainingBatch

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.TrainingBatch
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} attn_metadata
:canonical: fastvideo.pipelines.pipeline_batch_info.TrainingBatch.attn_metadata
:type: fastvideo.attention.AttentionMetadata | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.TrainingBatch.attn_metadata
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} attn_metadata_vsa
:canonical: fastvideo.pipelines.pipeline_batch_info.TrainingBatch.attn_metadata_vsa
:type: fastvideo.attention.AttentionMetadata | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.TrainingBatch.attn_metadata_vsa
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} conditional_dict
:canonical: fastvideo.pipelines.pipeline_batch_info.TrainingBatch.conditional_dict
:type: dict[str, typing.Any] | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.TrainingBatch.conditional_dict
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} current_timestep
:canonical: fastvideo.pipelines.pipeline_batch_info.TrainingBatch.current_timestep
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.TrainingBatch.current_timestep
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} current_vsa_sparsity
:canonical: fastvideo.pipelines.pipeline_batch_info.TrainingBatch.current_vsa_sparsity
:type: float
:value: >
   0.0

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.TrainingBatch.current_vsa_sparsity
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} dmd_latent_vis_dict
:canonical: fastvideo.pipelines.pipeline_batch_info.TrainingBatch.dmd_latent_vis_dict
:type: dict[str, typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.TrainingBatch.dmd_latent_vis_dict
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} encoder_attention_mask
:canonical: fastvideo.pipelines.pipeline_batch_info.TrainingBatch.encoder_attention_mask
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.TrainingBatch.encoder_attention_mask
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} encoder_attention_mask_neg
:canonical: fastvideo.pipelines.pipeline_batch_info.TrainingBatch.encoder_attention_mask_neg
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.TrainingBatch.encoder_attention_mask_neg
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} encoder_hidden_states
:canonical: fastvideo.pipelines.pipeline_batch_info.TrainingBatch.encoder_hidden_states
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.TrainingBatch.encoder_hidden_states
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} encoder_hidden_states_neg
:canonical: fastvideo.pipelines.pipeline_batch_info.TrainingBatch.encoder_hidden_states_neg
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.TrainingBatch.encoder_hidden_states_neg
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} fake_score_latent_vis_dict
:canonical: fastvideo.pipelines.pipeline_batch_info.TrainingBatch.fake_score_latent_vis_dict
:type: dict[str, typing.Any]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.TrainingBatch.fake_score_latent_vis_dict
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} fake_score_loss
:canonical: fastvideo.pipelines.pipeline_batch_info.TrainingBatch.fake_score_loss
:type: float
:value: >
   0.0

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.TrainingBatch.fake_score_loss
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} generator_loss
:canonical: fastvideo.pipelines.pipeline_batch_info.TrainingBatch.generator_loss
:type: float
:value: >
   0.0

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.TrainingBatch.generator_loss
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} grad_norm
:canonical: fastvideo.pipelines.pipeline_batch_info.TrainingBatch.grad_norm
:type: float | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.TrainingBatch.grad_norm
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} image_embeds
:canonical: fastvideo.pipelines.pipeline_batch_info.TrainingBatch.image_embeds
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.TrainingBatch.image_embeds
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} image_latents
:canonical: fastvideo.pipelines.pipeline_batch_info.TrainingBatch.image_latents
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.TrainingBatch.image_latents
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} infos
:canonical: fastvideo.pipelines.pipeline_batch_info.TrainingBatch.infos
:type: list[dict[str, typing.Any]] | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.TrainingBatch.infos
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} input_kwargs
:canonical: fastvideo.pipelines.pipeline_batch_info.TrainingBatch.input_kwargs
:type: dict[str, typing.Any] | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.TrainingBatch.input_kwargs
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} latents
:canonical: fastvideo.pipelines.pipeline_batch_info.TrainingBatch.latents
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.TrainingBatch.latents
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} loss
:canonical: fastvideo.pipelines.pipeline_batch_info.TrainingBatch.loss
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.TrainingBatch.loss
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} mask_lat_size
:canonical: fastvideo.pipelines.pipeline_batch_info.TrainingBatch.mask_lat_size
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.TrainingBatch.mask_lat_size
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} noise
:canonical: fastvideo.pipelines.pipeline_batch_info.TrainingBatch.noise
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.TrainingBatch.noise
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} noise_latents
:canonical: fastvideo.pipelines.pipeline_batch_info.TrainingBatch.noise_latents
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.TrainingBatch.noise_latents
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} noisy_model_input
:canonical: fastvideo.pipelines.pipeline_batch_info.TrainingBatch.noisy_model_input
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.TrainingBatch.noisy_model_input
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} preprocessed_image
:canonical: fastvideo.pipelines.pipeline_batch_info.TrainingBatch.preprocessed_image
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.TrainingBatch.preprocessed_image
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} raw_latent_shape
:canonical: fastvideo.pipelines.pipeline_batch_info.TrainingBatch.raw_latent_shape
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.TrainingBatch.raw_latent_shape
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} sigmas
:canonical: fastvideo.pipelines.pipeline_batch_info.TrainingBatch.sigmas
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.TrainingBatch.sigmas
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} timesteps
:canonical: fastvideo.pipelines.pipeline_batch_info.TrainingBatch.timesteps
:type: torch.Tensor | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.TrainingBatch.timesteps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} total_loss
:canonical: fastvideo.pipelines.pipeline_batch_info.TrainingBatch.total_loss
:type: float | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.TrainingBatch.total_loss
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} unconditional_dict
:canonical: fastvideo.pipelines.pipeline_batch_info.TrainingBatch.unconditional_dict
:type: dict[str, typing.Any] | None
:value: >
   None

```{autodoc2-docstring} fastvideo.pipelines.pipeline_batch_info.TrainingBatch.unconditional_dict
:parser: docs.source.autodoc2_docstring_parser
```

````

`````
