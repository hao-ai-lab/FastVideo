# {py:mod}`fastvideo.fastvideo_args`

```{py:module} fastvideo.fastvideo_args
```

```{autodoc2-docstring} fastvideo.fastvideo_args
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ExecutionMode <fastvideo.fastvideo_args.ExecutionMode>`
  - ```{autodoc2-docstring} fastvideo.fastvideo_args.ExecutionMode
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`FastVideoArgs <fastvideo.fastvideo_args.FastVideoArgs>`
  - ```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`TrainingArgs <fastvideo.fastvideo_args.TrainingArgs>`
  - ```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`WorkloadType <fastvideo.fastvideo_args.WorkloadType>`
  - ```{autodoc2-docstring} fastvideo.fastvideo_args.WorkloadType
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_current_fastvideo_args <fastvideo.fastvideo_args.get_current_fastvideo_args>`
  - ```{autodoc2-docstring} fastvideo.fastvideo_args.get_current_fastvideo_args
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`parse_int_list <fastvideo.fastvideo_args.parse_int_list>`
  - ```{autodoc2-docstring} fastvideo.fastvideo_args.parse_int_list
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`prepare_fastvideo_args <fastvideo.fastvideo_args.prepare_fastvideo_args>`
  - ```{autodoc2-docstring} fastvideo.fastvideo_args.prepare_fastvideo_args
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`set_current_fastvideo_args <fastvideo.fastvideo_args.set_current_fastvideo_args>`
  - ```{autodoc2-docstring} fastvideo.fastvideo_args.set_current_fastvideo_args
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <fastvideo.fastvideo_args.logger>`
  - ```{autodoc2-docstring} fastvideo.fastvideo_args.logger
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} ExecutionMode()
:canonical: fastvideo.fastvideo_args.ExecutionMode

Bases: {py:obj}`str`, {py:obj}`enum.Enum`

```{autodoc2-docstring} fastvideo.fastvideo_args.ExecutionMode
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.fastvideo_args.ExecutionMode.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} DISTILLATION
:canonical: fastvideo.fastvideo_args.ExecutionMode.DISTILLATION
:value: >
   'distillation'

```{autodoc2-docstring} fastvideo.fastvideo_args.ExecutionMode.DISTILLATION
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} FINETUNING
:canonical: fastvideo.fastvideo_args.ExecutionMode.FINETUNING
:value: >
   'finetuning'

```{autodoc2-docstring} fastvideo.fastvideo_args.ExecutionMode.FINETUNING
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} INFERENCE
:canonical: fastvideo.fastvideo_args.ExecutionMode.INFERENCE
:value: >
   'inference'

```{autodoc2-docstring} fastvideo.fastvideo_args.ExecutionMode.INFERENCE
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} PREPROCESS
:canonical: fastvideo.fastvideo_args.ExecutionMode.PREPROCESS
:value: >
   'preprocess'

```{autodoc2-docstring} fastvideo.fastvideo_args.ExecutionMode.PREPROCESS
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} choices() -> list[str]
:canonical: fastvideo.fastvideo_args.ExecutionMode.choices
:classmethod:

```{autodoc2-docstring} fastvideo.fastvideo_args.ExecutionMode.choices
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} from_string(value: str) -> fastvideo.fastvideo_args.ExecutionMode
:canonical: fastvideo.fastvideo_args.ExecutionMode.from_string
:classmethod:

```{autodoc2-docstring} fastvideo.fastvideo_args.ExecutionMode.from_string
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} FastVideoArgs
:canonical: fastvideo.fastvideo_args.FastVideoArgs

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} STA_mode
:canonical: fastvideo.fastvideo_args.FastVideoArgs.STA_mode
:type: fastvideo.configs.pipelines.base.STA_Mode
:value: >
   None

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.STA_mode
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} VSA_sparsity
:canonical: fastvideo.fastvideo_args.FastVideoArgs.VSA_sparsity
:type: float
:value: >
   0.0

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.VSA_sparsity
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} add_cli_args(parser: fastvideo.utils.FlexibleArgumentParser) -> fastvideo.utils.FlexibleArgumentParser
:canonical: fastvideo.fastvideo_args.FastVideoArgs.add_cli_args
:staticmethod:

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.add_cli_args
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} cache_strategy
:canonical: fastvideo.fastvideo_args.FastVideoArgs.cache_strategy
:type: str
:value: >
   'none'

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.cache_strategy
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} check_fastvideo_args() -> None
:canonical: fastvideo.fastvideo_args.FastVideoArgs.check_fastvideo_args

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.check_fastvideo_args
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} disable_autocast
:canonical: fastvideo.fastvideo_args.FastVideoArgs.disable_autocast
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.disable_autocast
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} dist_timeout
:canonical: fastvideo.fastvideo_args.FastVideoArgs.dist_timeout
:type: int | None
:value: >
   None

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.dist_timeout
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} distributed_executor_backend
:canonical: fastvideo.fastvideo_args.FastVideoArgs.distributed_executor_backend
:type: str
:value: >
   'mp'

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.distributed_executor_backend
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} dit_cpu_offload
:canonical: fastvideo.fastvideo_args.FastVideoArgs.dit_cpu_offload
:type: bool
:value: >
   True

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.dit_cpu_offload
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} enable_stage_verification
:canonical: fastvideo.fastvideo_args.FastVideoArgs.enable_stage_verification
:type: bool
:value: >
   True

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.enable_stage_verification
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} enable_torch_compile
:canonical: fastvideo.fastvideo_args.FastVideoArgs.enable_torch_compile
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.enable_torch_compile
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} from_cli_args(args: argparse.Namespace) -> fastvideo.fastvideo_args.FastVideoArgs
:canonical: fastvideo.fastvideo_args.FastVideoArgs.from_cli_args
:classmethod:

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.from_cli_args
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} from_kwargs(**kwargs: typing.Any) -> fastvideo.fastvideo_args.FastVideoArgs
:canonical: fastvideo.fastvideo_args.FastVideoArgs.from_kwargs
:classmethod:

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.from_kwargs
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} hsdp_replicate_dim
:canonical: fastvideo.fastvideo_args.FastVideoArgs.hsdp_replicate_dim
:type: int
:value: >
   1

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.hsdp_replicate_dim
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} hsdp_shard_dim
:canonical: fastvideo.fastvideo_args.FastVideoArgs.hsdp_shard_dim
:type: int
:value: >
   None

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.hsdp_shard_dim
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} image_encoder_cpu_offload
:canonical: fastvideo.fastvideo_args.FastVideoArgs.image_encoder_cpu_offload
:type: bool
:value: >
   True

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.image_encoder_cpu_offload
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} inference_mode
:canonical: fastvideo.fastvideo_args.FastVideoArgs.inference_mode
:type: bool
:value: >
   True

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.inference_mode
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} lora_nickname
:canonical: fastvideo.fastvideo_args.FastVideoArgs.lora_nickname
:type: str
:value: >
   'default'

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.lora_nickname
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} lora_path
:canonical: fastvideo.fastvideo_args.FastVideoArgs.lora_path
:type: str | None
:value: >
   None

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.lora_path
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} lora_target_modules
:canonical: fastvideo.fastvideo_args.FastVideoArgs.lora_target_modules
:type: list[str] | None
:value: >
   None

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.lora_target_modules
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} mask_strategy_file_path
:canonical: fastvideo.fastvideo_args.FastVideoArgs.mask_strategy_file_path
:type: str | None
:value: >
   None

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.mask_strategy_file_path
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} master_port
:canonical: fastvideo.fastvideo_args.FastVideoArgs.master_port
:type: int | None
:value: >
   None

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.master_port
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} mode
:canonical: fastvideo.fastvideo_args.FastVideoArgs.mode
:type: fastvideo.fastvideo_args.ExecutionMode
:value: >
   None

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.mode
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} model_loaded
:canonical: fastvideo.fastvideo_args.FastVideoArgs.model_loaded
:type: dict[str, bool]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.model_loaded
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} model_path
:canonical: fastvideo.fastvideo_args.FastVideoArgs.model_path
:type: str
:value: >
   None

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.model_path
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} model_paths
:canonical: fastvideo.fastvideo_args.FastVideoArgs.model_paths
:type: dict[str, str]
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.model_paths
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_gpus
:canonical: fastvideo.fastvideo_args.FastVideoArgs.num_gpus
:type: int
:value: >
   1

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.num_gpus
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} output_type
:canonical: fastvideo.fastvideo_args.FastVideoArgs.output_type
:type: str
:value: >
   'pil'

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.output_type
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} pin_cpu_memory
:canonical: fastvideo.fastvideo_args.FastVideoArgs.pin_cpu_memory
:type: bool
:value: >
   True

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.pin_cpu_memory
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} pipeline_config
:canonical: fastvideo.fastvideo_args.FastVideoArgs.pipeline_config
:type: fastvideo.configs.pipelines.base.PipelineConfig
:value: >
   'field(...)'

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.pipeline_config
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} preprocess_config
:canonical: fastvideo.fastvideo_args.FastVideoArgs.preprocess_config
:type: fastvideo.configs.configs.PreprocessConfig | None
:value: >
   None

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.preprocess_config
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} prompt_txt
:canonical: fastvideo.fastvideo_args.FastVideoArgs.prompt_txt
:type: str | None
:value: >
   None

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.prompt_txt
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} revision
:canonical: fastvideo.fastvideo_args.FastVideoArgs.revision
:type: str | None
:value: >
   None

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.revision
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} skip_time_steps
:canonical: fastvideo.fastvideo_args.FastVideoArgs.skip_time_steps
:type: int
:value: >
   15

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.skip_time_steps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} sp_size
:canonical: fastvideo.fastvideo_args.FastVideoArgs.sp_size
:type: int
:value: >
   None

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.sp_size
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} text_encoder_cpu_offload
:canonical: fastvideo.fastvideo_args.FastVideoArgs.text_encoder_cpu_offload
:type: bool
:value: >
   True

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.text_encoder_cpu_offload
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} tp_size
:canonical: fastvideo.fastvideo_args.FastVideoArgs.tp_size
:type: int
:value: >
   None

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.tp_size
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:property} training_mode
:canonical: fastvideo.fastvideo_args.FastVideoArgs.training_mode
:type: bool

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.training_mode
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} trust_remote_code
:canonical: fastvideo.fastvideo_args.FastVideoArgs.trust_remote_code
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.trust_remote_code
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} use_fsdp_inference
:canonical: fastvideo.fastvideo_args.FastVideoArgs.use_fsdp_inference
:type: bool
:value: >
   True

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.use_fsdp_inference
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} vae_cpu_offload
:canonical: fastvideo.fastvideo_args.FastVideoArgs.vae_cpu_offload
:type: bool
:value: >
   True

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.vae_cpu_offload
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} workload_type
:canonical: fastvideo.fastvideo_args.FastVideoArgs.workload_type
:type: fastvideo.fastvideo_args.WorkloadType
:value: >
   None

```{autodoc2-docstring} fastvideo.fastvideo_args.FastVideoArgs.workload_type
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} TrainingArgs
:canonical: fastvideo.fastvideo_args.TrainingArgs

Bases: {py:obj}`fastvideo.fastvideo_args.FastVideoArgs`

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} VSA_decay_interval_steps
:canonical: fastvideo.fastvideo_args.TrainingArgs.VSA_decay_interval_steps
:type: int
:value: >
   1

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.VSA_decay_interval_steps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} VSA_decay_rate
:canonical: fastvideo.fastvideo_args.TrainingArgs.VSA_decay_rate
:type: float
:value: >
   0.01

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.VSA_decay_rate
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} add_cli_args(parser: fastvideo.utils.FlexibleArgumentParser) -> fastvideo.utils.FlexibleArgumentParser
:canonical: fastvideo.fastvideo_args.TrainingArgs.add_cli_args
:staticmethod:

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.add_cli_args
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} allow_tf32
:canonical: fastvideo.fastvideo_args.TrainingArgs.allow_tf32
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.allow_tf32
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} checkpointing_steps
:canonical: fastvideo.fastvideo_args.TrainingArgs.checkpointing_steps
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.checkpointing_steps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} checkpoints_total_limit
:canonical: fastvideo.fastvideo_args.TrainingArgs.checkpoints_total_limit
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.checkpoints_total_limit
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} data_path
:canonical: fastvideo.fastvideo_args.TrainingArgs.data_path
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.data_path
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} dataloader_num_workers
:canonical: fastvideo.fastvideo_args.TrainingArgs.dataloader_num_workers
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.dataloader_num_workers
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} distill_cfg
:canonical: fastvideo.fastvideo_args.TrainingArgs.distill_cfg
:type: float
:value: >
   0.0

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.distill_cfg
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} dit_model_name_or_path
:canonical: fastvideo.fastvideo_args.TrainingArgs.dit_model_name_or_path
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.dit_model_name_or_path
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} ema_decay
:canonical: fastvideo.fastvideo_args.TrainingArgs.ema_decay
:type: float
:value: >
   0.0

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.ema_decay
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} ema_start_step
:canonical: fastvideo.fastvideo_args.TrainingArgs.ema_start_step
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.ema_start_step
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} enable_gradient_checkpointing_type
:canonical: fastvideo.fastvideo_args.TrainingArgs.enable_gradient_checkpointing_type
:type: str | None
:value: >
   None

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.enable_gradient_checkpointing_type
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} from_cli_args(args: argparse.Namespace) -> fastvideo.fastvideo_args.TrainingArgs
:canonical: fastvideo.fastvideo_args.TrainingArgs.from_cli_args
:classmethod:

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.from_cli_args
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} fsdp_sharding_startegy
:canonical: fastvideo.fastvideo_args.TrainingArgs.fsdp_sharding_startegy
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.fsdp_sharding_startegy
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} generator_update_interval
:canonical: fastvideo.fastvideo_args.TrainingArgs.generator_update_interval
:type: int
:value: >
   5

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.generator_update_interval
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} gradient_accumulation_steps
:canonical: fastvideo.fastvideo_args.TrainingArgs.gradient_accumulation_steps
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.gradient_accumulation_steps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} group_frame
:canonical: fastvideo.fastvideo_args.TrainingArgs.group_frame
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.group_frame
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} group_resolution
:canonical: fastvideo.fastvideo_args.TrainingArgs.group_resolution
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.group_resolution
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} hunyuan_teacher_disable_cfg
:canonical: fastvideo.fastvideo_args.TrainingArgs.hunyuan_teacher_disable_cfg
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.hunyuan_teacher_disable_cfg
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} learning_rate
:canonical: fastvideo.fastvideo_args.TrainingArgs.learning_rate
:type: float
:value: >
   0.0

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.learning_rate
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} linear_quadratic_threshold
:canonical: fastvideo.fastvideo_args.TrainingArgs.linear_quadratic_threshold
:type: float
:value: >
   0.0

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.linear_quadratic_threshold
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} linear_range
:canonical: fastvideo.fastvideo_args.TrainingArgs.linear_range
:type: float
:value: >
   0.0

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.linear_range
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} log_validation
:canonical: fastvideo.fastvideo_args.TrainingArgs.log_validation
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.log_validation
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} log_visualization
:canonical: fastvideo.fastvideo_args.TrainingArgs.log_visualization
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.log_visualization
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} logit_mean
:canonical: fastvideo.fastvideo_args.TrainingArgs.logit_mean
:type: float
:value: >
   0.0

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.logit_mean
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} logit_std
:canonical: fastvideo.fastvideo_args.TrainingArgs.logit_std
:type: float
:value: >
   1.0

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.logit_std
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} lora_alpha
:canonical: fastvideo.fastvideo_args.TrainingArgs.lora_alpha
:type: int | None
:value: >
   None

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.lora_alpha
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} lora_rank
:canonical: fastvideo.fastvideo_args.TrainingArgs.lora_rank
:type: int | None
:value: >
   None

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.lora_rank
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} lora_training
:canonical: fastvideo.fastvideo_args.TrainingArgs.lora_training
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.lora_training
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} lr_num_cycles
:canonical: fastvideo.fastvideo_args.TrainingArgs.lr_num_cycles
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.lr_num_cycles
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} lr_power
:canonical: fastvideo.fastvideo_args.TrainingArgs.lr_power
:type: float
:value: >
   0.0

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.lr_power
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} lr_scheduler
:canonical: fastvideo.fastvideo_args.TrainingArgs.lr_scheduler
:type: str
:value: >
   'constant'

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.lr_scheduler
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} lr_warmup_steps
:canonical: fastvideo.fastvideo_args.TrainingArgs.lr_warmup_steps
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.lr_warmup_steps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} master_weight_type
:canonical: fastvideo.fastvideo_args.TrainingArgs.master_weight_type
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.master_weight_type
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} max_grad_norm
:canonical: fastvideo.fastvideo_args.TrainingArgs.max_grad_norm
:type: float
:value: >
   0.0

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.max_grad_norm
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} max_timestep_ratio
:canonical: fastvideo.fastvideo_args.TrainingArgs.max_timestep_ratio
:type: float
:value: >
   0.98

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.max_timestep_ratio
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} max_train_steps
:canonical: fastvideo.fastvideo_args.TrainingArgs.max_train_steps
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.max_train_steps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} min_timestep_ratio
:canonical: fastvideo.fastvideo_args.TrainingArgs.min_timestep_ratio
:type: float
:value: >
   0.2

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.min_timestep_ratio
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} mixed_precision
:canonical: fastvideo.fastvideo_args.TrainingArgs.mixed_precision
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.mixed_precision
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} mode_scale
:canonical: fastvideo.fastvideo_args.TrainingArgs.mode_scale
:type: float
:value: >
   0.0

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.mode_scale
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} multi_phased_distill_schedule
:canonical: fastvideo.fastvideo_args.TrainingArgs.multi_phased_distill_schedule
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.multi_phased_distill_schedule
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} not_apply_cfg_solver
:canonical: fastvideo.fastvideo_args.TrainingArgs.not_apply_cfg_solver
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.not_apply_cfg_solver
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_euler_timesteps
:canonical: fastvideo.fastvideo_args.TrainingArgs.num_euler_timesteps
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.num_euler_timesteps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_frames
:canonical: fastvideo.fastvideo_args.TrainingArgs.num_frames
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.num_frames
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_height
:canonical: fastvideo.fastvideo_args.TrainingArgs.num_height
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.num_height
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_latent_t
:canonical: fastvideo.fastvideo_args.TrainingArgs.num_latent_t
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.num_latent_t
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_train_epochs
:canonical: fastvideo.fastvideo_args.TrainingArgs.num_train_epochs
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.num_train_epochs
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} num_width
:canonical: fastvideo.fastvideo_args.TrainingArgs.num_width
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.num_width
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} output_dir
:canonical: fastvideo.fastvideo_args.TrainingArgs.output_dir
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.output_dir
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} precondition_outputs
:canonical: fastvideo.fastvideo_args.TrainingArgs.precondition_outputs
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.precondition_outputs
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} pred_decay_type
:canonical: fastvideo.fastvideo_args.TrainingArgs.pred_decay_type
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.pred_decay_type
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} pred_decay_weight
:canonical: fastvideo.fastvideo_args.TrainingArgs.pred_decay_weight
:type: float
:value: >
   0.0

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.pred_decay_weight
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} pretrained_model_name_or_path
:canonical: fastvideo.fastvideo_args.TrainingArgs.pretrained_model_name_or_path
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.pretrained_model_name_or_path
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} real_score_guidance_scale
:canonical: fastvideo.fastvideo_args.TrainingArgs.real_score_guidance_scale
:type: float
:value: >
   3.5

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.real_score_guidance_scale
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} resume_from_checkpoint
:canonical: fastvideo.fastvideo_args.TrainingArgs.resume_from_checkpoint
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.resume_from_checkpoint
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} scale_lr
:canonical: fastvideo.fastvideo_args.TrainingArgs.scale_lr
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.scale_lr
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} scheduler_type
:canonical: fastvideo.fastvideo_args.TrainingArgs.scheduler_type
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.scheduler_type
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} seed
:canonical: fastvideo.fastvideo_args.TrainingArgs.seed
:type: int | None
:value: >
   None

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.seed
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} selective_checkpointing
:canonical: fastvideo.fastvideo_args.TrainingArgs.selective_checkpointing
:type: float
:value: >
   0.0

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.selective_checkpointing
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} tracker_project_name
:canonical: fastvideo.fastvideo_args.TrainingArgs.tracker_project_name
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.tracker_project_name
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} train_batch_size
:canonical: fastvideo.fastvideo_args.TrainingArgs.train_batch_size
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.train_batch_size
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} train_sp_batch_size
:canonical: fastvideo.fastvideo_args.TrainingArgs.train_sp_batch_size
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.train_sp_batch_size
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} training_cfg_rate
:canonical: fastvideo.fastvideo_args.TrainingArgs.training_cfg_rate
:type: float
:value: >
   0.0

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.training_cfg_rate
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} training_state_checkpointing_steps
:canonical: fastvideo.fastvideo_args.TrainingArgs.training_state_checkpointing_steps
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.training_state_checkpointing_steps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} use_ema
:canonical: fastvideo.fastvideo_args.TrainingArgs.use_ema
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.use_ema
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} validation_dataset_file
:canonical: fastvideo.fastvideo_args.TrainingArgs.validation_dataset_file
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.validation_dataset_file
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} validation_guidance_scale
:canonical: fastvideo.fastvideo_args.TrainingArgs.validation_guidance_scale
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.validation_guidance_scale
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} validation_preprocessed_path
:canonical: fastvideo.fastvideo_args.TrainingArgs.validation_preprocessed_path
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.validation_preprocessed_path
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} validation_sampling_steps
:canonical: fastvideo.fastvideo_args.TrainingArgs.validation_sampling_steps
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.validation_sampling_steps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} validation_steps
:canonical: fastvideo.fastvideo_args.TrainingArgs.validation_steps
:type: float
:value: >
   0.0

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.validation_steps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} wandb_run_name
:canonical: fastvideo.fastvideo_args.TrainingArgs.wandb_run_name
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.wandb_run_name
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} weight_decay
:canonical: fastvideo.fastvideo_args.TrainingArgs.weight_decay
:type: float
:value: >
   0.0

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.weight_decay
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} weight_only_checkpointing_steps
:canonical: fastvideo.fastvideo_args.TrainingArgs.weight_only_checkpointing_steps
:type: int
:value: >
   0

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.weight_only_checkpointing_steps
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} weighting_scheme
:canonical: fastvideo.fastvideo_args.TrainingArgs.weighting_scheme
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} fastvideo.fastvideo_args.TrainingArgs.weighting_scheme
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} WorkloadType()
:canonical: fastvideo.fastvideo_args.WorkloadType

Bases: {py:obj}`str`, {py:obj}`enum.Enum`

```{autodoc2-docstring} fastvideo.fastvideo_args.WorkloadType
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.fastvideo_args.WorkloadType.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} I2I
:canonical: fastvideo.fastvideo_args.WorkloadType.I2I
:value: >
   'i2i'

```{autodoc2-docstring} fastvideo.fastvideo_args.WorkloadType.I2I
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} I2V
:canonical: fastvideo.fastvideo_args.WorkloadType.I2V
:value: >
   'i2v'

```{autodoc2-docstring} fastvideo.fastvideo_args.WorkloadType.I2V
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} T2I
:canonical: fastvideo.fastvideo_args.WorkloadType.T2I
:value: >
   't2i'

```{autodoc2-docstring} fastvideo.fastvideo_args.WorkloadType.T2I
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} T2V
:canonical: fastvideo.fastvideo_args.WorkloadType.T2V
:value: >
   't2v'

```{autodoc2-docstring} fastvideo.fastvideo_args.WorkloadType.T2V
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} choices() -> list[str]
:canonical: fastvideo.fastvideo_args.WorkloadType.choices
:classmethod:

```{autodoc2-docstring} fastvideo.fastvideo_args.WorkloadType.choices
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} from_string(value: str) -> fastvideo.fastvideo_args.WorkloadType
:canonical: fastvideo.fastvideo_args.WorkloadType.from_string
:classmethod:

```{autodoc2-docstring} fastvideo.fastvideo_args.WorkloadType.from_string
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:function} get_current_fastvideo_args() -> fastvideo.fastvideo_args.FastVideoArgs
:canonical: fastvideo.fastvideo_args.get_current_fastvideo_args

```{autodoc2-docstring} fastvideo.fastvideo_args.get_current_fastvideo_args
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:data} logger
:canonical: fastvideo.fastvideo_args.logger
:value: >
   'init_logger(...)'

```{autodoc2-docstring} fastvideo.fastvideo_args.logger
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:function} parse_int_list(value: str) -> list[int]
:canonical: fastvideo.fastvideo_args.parse_int_list

```{autodoc2-docstring} fastvideo.fastvideo_args.parse_int_list
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} prepare_fastvideo_args(argv: list[str]) -> fastvideo.fastvideo_args.FastVideoArgs
:canonical: fastvideo.fastvideo_args.prepare_fastvideo_args

```{autodoc2-docstring} fastvideo.fastvideo_args.prepare_fastvideo_args
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} set_current_fastvideo_args(fastvideo_args: fastvideo.fastvideo_args.FastVideoArgs)
:canonical: fastvideo.fastvideo_args.set_current_fastvideo_args

```{autodoc2-docstring} fastvideo.fastvideo_args.set_current_fastvideo_args
:parser: docs.source.autodoc2_docstring_parser
```
````
