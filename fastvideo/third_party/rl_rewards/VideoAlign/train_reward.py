import ast
import json
import os
import random
from dataclasses import asdict
from functools import partial

import torch
from data import DataConfig, QWen2VLDataCollator, convert_GSB_csv_to_reward_data
from peft import LoraConfig, get_peft_model
from trainer import (
    PartialEmbeddingUpdateCallback,
    Qwen2VLRewardModelBT,
    VideoVLMRewardTrainer,
    compute_multi_attr_accuracy,
)
from transformers import AutoProcessor, HfArgumentParser
from trl import get_kbit_device_map, get_quantization_config
from utils import ModelConfig, PEFTLoraConfig, TrainingConfig, load_model_from_checkpoint

from datasets import load_dataset


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


def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=False):
    """
    Find the target linear modules for LoRA.
    """
    linear_cls = torch.nn.Linear
    embedding_cls = torch.nn.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            # print(f"Excluding module: {name}")
            continue

        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)

    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names


def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad


def create_model_and_processor(
    model_config,
    peft_lora_config,
    training_args,
    cache_dir=None,
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
        use_cache=True if training_args.gradient_checkpointing else False,
    )
    # pdb.set_trace()

    # create processor and set padding
    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path, padding_side="right", cache_dir=cache_dir
    )

    special_token_ids = None
    if model_config.use_special_tokens:
        special_tokens = ["<|VQ_reward|>", "<|MQ_reward|>", "<|TA_reward|>"]
        processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        special_token_ids = processor.tokenizer.convert_tokens_to_ids(special_tokens)

    model = Qwen2VLRewardModelBT.from_pretrained(
        model_config.model_name_or_path,
        output_dim=model_config.output_dim,
        reward_token=model_config.reward_token,
        special_token_ids=special_token_ids,
        torch_dtype=torch_dtype,
        attn_implementation=("flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa"),
        cache_dir=cache_dir,
        **model_kwargs,
    )
    if model_config.use_special_tokens:
        model.resize_token_embeddings(len(processor.tokenizer))

    if training_args.bf16:
        model.to(torch.bfloat16)
    if training_args.fp16:
        model.to(torch.float16)

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


def create_dataset(data_config, meta_file=None):
    if meta_file is None:
        meta_file = data_config.meta_data
    dataset = load_dataset("csv", data_files=meta_file)

    def add_idx(example, idx):
        example["metainfo_idx"] = idx
        return example

    dataset["train"] = dataset["train"].map(lambda example, idx: add_idx(example, idx), with_indices=True)

    if not data_config.use_tied_data:
        filter_func = lambda example: any(example[f"{dim}"] != "same" for dim in data_config.eval_dim)
        dataset = dataset.filter(filter_func)

    # convert data to reward data
    convert_func = lambda example: convert_GSB_csv_to_reward_data(
        example,
        data_config.data_dir,
        data_config.eval_dim,
        data_config.max_frame_pixels,
        data_config.fps,
        data_config.num_frames,
        data_config.prompt_template_type,
        sample_type=data_config.sample_type,
    )
    dataset = dataset.map(
        convert_func,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
    )
    dataset = dataset["train"]
    # pdb.set_trace()
    return dataset


def train():
    ## ===> Step 1: Parse arguments
    parser = HfArgumentParser((DataConfig, TrainingConfig, ModelConfig, PEFTLoraConfig))
    (
        data_config,
        training_args,
        model_config,
        peft_lora_config,
    ) = parser.parse_args_into_dataclasses()
    # pdb.set_trace()

    # check valid (lora config)
    assert not (peft_lora_config.lora_enable and model_config.freeze_llm), (
        "When using LoRA, the LLM should not be frozen. If you want to freeze the LLM, please disable LoRA."
    )
    if not peft_lora_config.lora_enable:
        assert not peft_lora_config.vision_lora, (
            "Error: model_config.lora_enable is not enabled, but model_config.vision_lora is enabled."
        )
    else:
        if peft_lora_config.lora_namespan_exclude is not None:
            peft_lora_config.lora_namespan_exclude = ast.literal_eval(peft_lora_config.lora_namespan_exclude)
        else:
            peft_lora_config.lora_namespan_exclude = []
        if not peft_lora_config.vision_lora:
            peft_lora_config.lora_namespan_exclude += ["visual"]

    # pdb.set_trace()

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
        set_requires_grad(model_to_configure.model.parameters(), not model_config.freeze_llm)

    if not peft_lora_config.vision_lora:
        # set requires_grad for visual encoder and merger
        set_requires_grad(model_to_configure.visual.parameters(), not model_config.freeze_vision_tower)
        set_requires_grad(model_to_configure.visual.merger.parameters(), model_config.tune_merger)

    # set requires_grad for regression head
    set_requires_grad(model_to_configure.rm_head.parameters(), True)

    ## ===> Step 3: Load Dataset and configure
    if isinstance(data_config.eval_dim, str):
        data_config.eval_dim = [data_config.eval_dim]
    # datasets = create_dataset(data_config)
    # train_dataset = concatenate_datasets([datasets[dim] for dim in data_config.eval_dim])
    train_dataset = create_dataset(data_config)
    train_dataset = train_dataset.shuffle(seed=42)

    if training_args.conduct_eval:
        if data_config.meta_data_test is not None:
            random.seed(42)
            valid_dataset = create_dataset(data_config, meta_file=data_config.meta_data_test)
            # indices = random.sample(range(len(valid_dataset)), 1000)
            # valid_dataset = valid_dataset.select(indices)
        else:
            dataset = train_dataset.train_test_split(test_size=0.02)
            train_dataset = dataset["train"]
            valid_dataset = dataset["test"]
    else:
        valid_dataset = None

    print(f"===> Selected {len(train_dataset)} samples for training.")
    print(f"===> Selected {len(valid_dataset)} samples for testing.")

    num_gpu = int(os.environ.get("WORLD_SIZE", 1))
    data_collator = QWen2VLDataCollator(
        processor,
        add_noise=data_config.add_noise,
        p_shuffle_frames=data_config.p_shuffle_frames,
        p_color_jitter=data_config.p_color_jitter,
    )
    compute_metrics = partial(compute_multi_attr_accuracy, eval_dims=data_config.eval_dim)

    actual_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_gpu
    total_steps = training_args.num_train_epochs * len(train_dataset) // actual_batch_size
    if training_args.save_epochs is not None:
        training_args.save_steps = round(training_args.save_epochs * len(train_dataset) / actual_batch_size)
    if training_args.eval_epochs is not None:
        training_args.eval_steps = round(training_args.eval_epochs * len(train_dataset) / actual_batch_size)
    if training_args.logging_epochs is not None:
        training_args.logging_steps = round(training_args.logging_epochs * len(train_dataset) / actual_batch_size)

    if training_args.local_rank == -1 or training_args.local_rank == 0:
        print(f"===> Using {num_gpu} GPUs.")
        print(f"===> Total Batch Size: {actual_batch_size}")
        print(f"===> Training Epochs: {training_args.num_train_epochs}")
        print(f"===> Total Steps: {total_steps}")
        print(f"===> Save Steps: {training_args.save_steps}")
        print(f"===> Eval Steps: {training_args.eval_steps}")
        print(f"===> Logging Steps: {training_args.logging_steps}")

    # pdb.set_trace()

    ## ===> Step 4: Save configs for re-check
    if training_args.local_rank == -1 or training_args.local_rank == 0:
        save_configs_to_json(data_config, training_args, model_config, peft_lora_config)

    print(train_dataset)
    ## ===> Step 5: Start Training!

    special_token_ids = model.special_token_ids
    callbacks = []
    if special_token_ids is not None:
        callbacks.append(PartialEmbeddingUpdateCallback(special_token_ids))

    trainer = VideoVLMRewardTrainer(
        model=model,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset if training_args.conduct_eval else None,
        peft_config=peft_config,
        callbacks=callbacks,
        loss_type=model_config.loss_type,
        tokenizer=processor.tokenizer,
    )

    trainer.train()

    if training_args.local_rank == -1 or training_args.local_rank == 0:
        model_state_dict = model.state_dict()
        torch.save(model_state_dict, os.path.join(training_args.output_dir, "final_model.pth"))
        model.config.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
