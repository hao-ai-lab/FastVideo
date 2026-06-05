import json
import os
import fire
from dataclasses import asdict
from functools import partial
import torch
import torch.distributed as dist
from .model.qwen2vl_trainer import (
    Qwen2VLRewardModelBT,
    VLMRewardTrainer,
    compute_multi_attr_accuracy,
    PartialEmbeddingUpdateCallback,
)
from .dataset.pairwise_dataset import PairwiseOriginalDataset
from .dataset.data_collator_qwen import QWen2VLDataCollator
from .utils.parser import ModelConfig, PEFTLoraConfig, TrainingConfig, DataConfig
from .utils.training_utils import load_model_from_checkpoint, find_target_linear_names
from .utils.parser import parse_args_with_yaml
from transformers import AutoProcessor
from peft import LoraConfig, get_peft_model
from trl import get_kbit_device_map, get_quantization_config
from .model.differentiable_image_processor import Qwen2VLImageProcessor
try:
    import flash_attn
except ImportError:
    flash_attn = None
    print("Flash Attention is not installed. Falling to SDPA.")

def create_model_and_processor(
    model_config,
    peft_lora_config,
    training_args,
    cache_dir=None,
    differentiable=False,
):
    # create model
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_cache=False
    )

    # create processor and set padding

    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path, padding_side="right", cache_dir=cache_dir
    )

    if differentiable:
        processor.image_processor = Qwen2VLImageProcessor()

    special_token_ids = None
    if model_config.use_special_tokens:
        special_tokens = ["<|Reward|>"]
        processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens}
        )
        special_token_ids = processor.tokenizer.convert_tokens_to_ids(special_tokens)

    model = Qwen2VLRewardModelBT.from_pretrained(
        model_config.model_name_or_path,
        output_dim=model_config.output_dim,
        reward_token=model_config.reward_token,
        special_token_ids=special_token_ids,
        torch_dtype=torch_dtype,
        attn_implementation=(
            "flash_attention_2" if not training_args.disable_flash_attn2 and flash_attn is not None else "sdpa"
        ),
        cache_dir=cache_dir,
        rm_head_type=model_config.rm_head_type,
        rm_head_kwargs=model_config.rm_head_kwargs,
        **model_kwargs,
    )

    if model_config.use_special_tokens:
        model.resize_token_embeddings(len(processor.tokenizer))

    if training_args.bf16:
        model.to(torch.bfloat16)
    if training_args.fp16:
        model.to(torch.float16)

    model.rm_head.to(torch.float32)
    
    # create lora and peft model
    if peft_lora_config.lora_enable:
        target_modules = find_target_linear_names(
            model,
            num_lora_modules=peft_lora_config.num_lora_modules,
            lora_namespan_exclude=peft_lora_config.lora_namespan_exclude,
        )
        peft_config = LoraConfig(
            target_modules=target_modules,
            r=peft_lora_config.lora_r,
            lora_alpha=peft_lora_config.lora_alpha,
            lora_dropout=peft_lora_config.lora_dropout,
            task_type=peft_lora_config.lora_task_type,
            use_rslora=peft_lora_config.use_rslora,
            bias="none",
            modules_to_save=peft_lora_config.lora_modules_to_save,
        )
        model = get_peft_model(model, peft_config)
    else:
        peft_config = None

    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    return model, processor, peft_config


def save_configs_to_json(data_config, training_args, model_config, peft_lora_config):
    """
    Save all configurations to a JSON file.
    """
    config_dict = {
        "data_config": asdict(data_config),
        "training_args": asdict(training_args),
        "model_config": asdict(model_config),
        "peft_lora_config": asdict(peft_lora_config),
    }
    # del information about local device
    del config_dict["training_args"]["local_rank"]
    del config_dict["training_args"]["_n_gpu"]

    save_path = os.path.join(training_args.output_dir, "model_config.json")

    os.makedirs(training_args.output_dir, exist_ok=True)
    print(training_args.output_dir)

    with open(save_path, "w") as f:
        json.dump(config_dict, f, indent=4)


def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

def train(config, local_rank=0, debug=False):

    ## ===> Step 1: Parse arguments
    (data_config, training_args, model_config, peft_lora_config), config_path = (
        parse_args_with_yaml(
            (DataConfig, TrainingConfig, ModelConfig, PEFTLoraConfig), config, is_train=True
        )
    )
    training_args.output_dir = os.path.join(
        training_args.output_dir, config.split("/")[-1].split(".")[0]
    )
    training_args.logging_dir = training_args.output_dir
    # check valid (lora config)
    assert not (
        peft_lora_config.lora_enable and model_config.freeze_llm
    ), "When using LoRA, the LLM should not be frozen. If you want to freeze the LLM, please disable LoRA."
    if not peft_lora_config.lora_enable:
        assert (
            not peft_lora_config.vision_lora
        ), "Error: model_config.lora_enable is not enabled, but model_config.vision_lora is enabled."
    else:
        if peft_lora_config.lora_namespan_exclude is None:
            peft_lora_config.lora_namespan_exclude = []
        if not peft_lora_config.vision_lora:
            peft_lora_config.lora_namespan_exclude += ["visual"]

    ## ===> Step 2: Load model and configure
    model, processor, peft_config = create_model_and_processor(
        model_config=model_config,
        peft_lora_config=peft_lora_config,
        training_args=training_args,
    )

    ## load model
    if training_args.load_from_pretrained is not None:
        model, checkpoint_step = load_model_from_checkpoint(
            model,
            training_args.load_from_pretrained,
            training_args.load_from_pretrained_step,
        )
    model.train()

    if peft_lora_config.lora_enable:
        model_to_configure = model.model
    else:
        model_to_configure = model
        # set requires_grad for LLM
        set_requires_grad(
            model_to_configure.model.parameters(), not model_config.freeze_llm
        )
        set_requires_grad(model_to_configure.model.embed_tokens.parameters(), False)
    if not peft_lora_config.vision_lora:
        # set requires_grad for visual encoder and merger
        set_requires_grad(
            model_to_configure.visual.parameters(), not model_config.freeze_vision_tower
        )
        set_requires_grad(
            model_to_configure.visual.merger.parameters(), model_config.tune_merger
        )
    
    if model_config.trainable_visual_layers: # This is inverse order to index of model.visual.blocks, set -1 to unfreeze all layers
        assert model_config.trainable_visual_layers <= len(model_to_configure.visual.blocks), "trainable_visual_layers should be less than or equal to the number of visual blocks"
        freeze_layer_num = len(model_to_configure.visual.blocks) - model_config.trainable_visual_layers if model_config.trainable_visual_layers > 0 else 0
        for index, layer in enumerate(model_to_configure.visual.blocks):
            if index < freeze_layer_num:
                set_requires_grad(layer.parameters(), False)
            else:
                set_requires_grad(layer.parameters(), True)
    
    # set requires_grad for regression head
    set_requires_grad(model_to_configure.rm_head.parameters(), True)

    ## ===> Step 3: Load Dataset and configure
    train_dataset = PairwiseOriginalDataset(
        data_config.train_json_list,
        data_config.soft_label,
        data_config.confidence_threshold,
    )
    test_set_dict = {}
    for item in data_config.test_json_list:
        test_set_dict[item[0]] = PairwiseOriginalDataset(
            item[1],
            data_config.soft_label,
            data_config.confidence_threshold,
        )

    print(f"===> Selected {len(train_dataset)} samples for training.")
    for key, value in test_set_dict.items():
        print(f"===> Selected {len(value)} samples for {key} testing.")

    num_gpu = int(os.environ.get("WORLD_SIZE", 1))
    data_collator = QWen2VLDataCollator(
        processor,
        max_pixels=data_config.max_pixels,
        min_pixels=data_config.min_pixels,
        with_instruction=data_config.with_instruction,
        use_special_tokens=model_config.use_special_tokens,
    )
    compute_metrics = partial(compute_multi_attr_accuracy)

    actual_batch_size = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * num_gpu
    )
    total_steps = (
        training_args.num_train_epochs * len(train_dataset) // actual_batch_size
    )
    if training_args.save_epochs is not None:
        training_args.save_steps = round(
            training_args.save_epochs * len(train_dataset) / actual_batch_size
        )
    if training_args.eval_epochs is not None:
        training_args.eval_steps = round(
            training_args.eval_epochs * len(train_dataset) / actual_batch_size
        )
    if training_args.logging_epochs is not None:
        training_args.logging_steps = round(
            training_args.logging_epochs * len(train_dataset) / actual_batch_size
        )

    if training_args.local_rank == -1 or training_args.local_rank == 0:
        print(f"===> Using {num_gpu} GPUs.")
        print(f"===> Total Batch Size: {actual_batch_size}")
        print(f"===> Training Epochs: {training_args.num_train_epochs}")
        print(f"===> Total Steps: {total_steps}")
        print(f"===> Save Steps: {training_args.save_steps}")
        print(f"===> Eval Steps: {training_args.eval_steps}")
        print(f"===> Logging Steps: {training_args.logging_steps}")

    ## ===> Step 4: Save configs for re-check
    if training_args.local_rank == -1 or training_args.local_rank == 0:
        save_configs_to_json(data_config, training_args, model_config, peft_lora_config)

    print(train_dataset)
    ## ===> Step 5: Start Training!

    special_token_ids = model.special_token_ids
    callbacks = []
    if special_token_ids is not None:
        callbacks.append(PartialEmbeddingUpdateCallback(special_token_ids))

    trainer = VLMRewardTrainer(
        model=model,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=(test_set_dict if training_args.conduct_eval else None),
        peft_config=peft_config,
        callbacks=callbacks,
        loss_type=model_config.loss_type,
        loss_hyperparameters=model_config.loss_hyperparameters,
        tokenizer=processor.tokenizer,
        tied_threshold=data_config.tied_threshold,
        visualization_steps=training_args.visualization_steps,
        max_viz_samples=training_args.max_viz_samples,
    )
    trainer.train()

    if training_args.local_rank == -1 or training_args.local_rank == 0:
        model_state_dict = model.state_dict()
        torch.save(
            model_state_dict, os.path.join(training_args.output_dir, "final_model.pth")
        )
        model.config.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    fire.Fire(train)
