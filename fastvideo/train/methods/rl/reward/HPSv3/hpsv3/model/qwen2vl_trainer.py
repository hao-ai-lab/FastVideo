import os
import pdb
import warnings
import time
import math
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

from typing import List, Optional, Dict, Union, Any
import pandas as pd
import safetensors
import numpy as np
import torch
import torch.nn as nn
import datasets
from torch.utils.data import Dataset, DataLoader
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration
from transformers import AutoConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer import TrainerCallback
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    is_peft_available,
    is_datasets_available,
    WEIGHTS_NAME,
    TRAINING_ARGS_NAME,
    SAFE_WEIGHTS_NAME,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
    speed_metrics,
    deepspeed_init,
    speed_metrics,
    has_length,
    EvalPrediction,
    EvalLoopContainer,
    PredictionOutput,
    is_torch_xla_available,
    denumpify_detensorize,
    PredictionOutput,
    EvalLoopOutput,
    DistributedTensorGatherer,
    SequentialDistributedSampler,
    nested_concat,
)
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.trainer_callback import TrainerControl, TrainerState

from transformers.trainer_pt_utils import nested_detach, find_batch_size
from transformers.training_args import TrainingArguments
from trl import RewardTrainer
from ..utils.training_utils import get_peft_state_non_lora_maybe_zero_3

class Qwen2VLRewardModelBT(Qwen2VLForConditionalGeneration):
    def __init__(
        self,
        config,
        output_dim=4,
        reward_token="last",
        special_token_ids=None,
        rm_head_type="default",
        rm_head_kwargs=None,
    ):
        super().__init__(config)
        # pdb.set_trace()
        self.output_dim = output_dim
        if rm_head_type == "default":
            self.rm_head = nn.Linear(config.hidden_size, output_dim, bias=False)
        elif rm_head_type == "ranknet":
            if rm_head_kwargs is not None:
                for layer in range(rm_head_kwargs.get("num_layers", 3)):
                    if layer == 0:
                        self.rm_head = nn.Sequential(
                            nn.Linear(config.hidden_size, rm_head_kwargs["hidden_size"]),
                            nn.ReLU(),
                            nn.Dropout(rm_head_kwargs.get("dropout", 0.1)),
                        )
                    elif layer < rm_head_kwargs.get("num_layers", 3) - 1:
                        self.rm_head.add_module(
                            f"layer_{layer}",
                            nn.Sequential(
                                nn.Linear(rm_head_kwargs["hidden_size"], rm_head_kwargs["hidden_size"]),
                                nn.ReLU(),
                                nn.Dropout(rm_head_kwargs.get("dropout", 0.1)),
                            ),
                        )
                    else:
                        self.rm_head.add_module(
                            f"output_layer",
                            nn.Linear(rm_head_kwargs["hidden_size"], output_dim, bias=rm_head_kwargs.get("bias", False)),
                        )

            else:
                self.rm_head = nn.Sequential(
                    nn.Linear(config.hidden_size, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.05),
                    nn.Linear(1024, 16),
                    nn.ReLU(),
                    nn.Linear(16, output_dim),
                )

        self.rm_head.to(torch.float32)
        self.reward_token = reward_token

        self.special_token_ids = special_token_ids
        if self.special_token_ids is not None:
            self.reward_token = "special"

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
    ):
        ## modified from the origin class Qwen2VLForConditionalGeneration
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # pdb.set_trace()
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                )
                image_embeds = image_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                )
                video_embeds = video_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]  # [B, L, D]
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            logits = self.rm_head(hidden_states)  # [B, L, N]

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        ## get sequence length
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = (
                    torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                )
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        ## get the last token's logits
        if self.reward_token == "last":
            pooled_logits = logits[
                torch.arange(batch_size, device=logits.device), sequence_lengths
            ]
        elif self.reward_token == "mean":
            ## get the mean of all valid tokens' logits
            valid_lengths = torch.clamp(sequence_lengths, min=0, max=logits.size(1) - 1)
            pooled_logits = torch.stack(
                [logits[i, : valid_lengths[i]].mean(dim=0) for i in range(batch_size)]
            )
        elif self.reward_token == "special":
            # special_token_ids = self.tokenizer.convert_tokens_to_ids(self.special_tokens)
            # create a mask for special tokens
            special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for special_token_id in self.special_token_ids:
                special_token_mask = special_token_mask | (
                    input_ids == special_token_id
                )
            pooled_logits = logits[special_token_mask, ...]
            pooled_logits = pooled_logits.view(
                batch_size, 1, -1
            )  # [B, 3, N] assert 3 attributes
            pooled_logits = pooled_logits.view(batch_size, -1)

            # pdb.set_trace()
        else:
            raise ValueError("Invalid reward_token")

        return {"logits": pooled_logits}


def _convert_A_B_to_chosen_rejected(
    rewards_A,
    rewards_B,
    tied_threshold=None,
    choice_dist=None,
):
    """
    Inputs:
        rewards_A: [B, 1]
        rewards_B: [B, 1]
    Outputs:
        rewards_chosen: [B, 1]
        rewards_rejected: [B, 1]
        nontied_mask: [B, 1] (preference labels that is not tied)
    """
    chosen_label = torch.ones_like(rewards_A, dtype=torch.int64).to(
        rewards_A.device
    )  # [B, 1]
    chosen_mask = chosen_label == 1
    rejected_mask = chosen_label != 1

    rewards_chosen = rewards_A 
    rewards_rejected = rewards_B

    if tied_threshold is None:
        nontied_mask = torch.ones_like(chosen_label, dtype=torch.float32).to(
            rewards_A.device
        )
    else:
        nontied_mask = (
            torch.abs(
                (choice_dist[:, 0] - choice_dist[:, 1]) / torch.sum(choice_dist, dim=-1)
            )
            > tied_threshold
        )
        print(nontied_mask)
    return (
        rewards_chosen,
        rewards_rejected,
        nontied_mask,
    )


class PartialEmbeddingUpdateCallback(TrainerCallback):
    """
    Callback to update the embedding of special tokens
    Only the special tokens are updated, the rest of the embeddings are kept fixed
    """

    def __init__(self, special_token_ids):
        super().__init__()
        self.special_token_ids = special_token_ids
        self.orig_embeds_params = None

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        self.orig_embeds_params = model.get_input_embeddings().weight.clone().detach()

    def on_step_end(self, args, state, control, **kwargs):
        # pdb.set_trace()
        model = kwargs.get("model")
        tokenizer = kwargs.get("tokenizer")

        index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
        index_no_updates[self.special_token_ids] = False
        with torch.no_grad():
            model.get_input_embeddings().weight[index_no_updates] = (
                self.orig_embeds_params[index_no_updates]
            )


class VLMRewardTrainer(RewardTrainer):
    def __init__(self, loss_type="regular", loss_hyperparameters={}, tied_threshold=None, 
                 visualization_steps=500, max_viz_samples=4, *args, **kwargs):
        super(VLMRewardTrainer, self).__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.tied_threshold = tied_threshold
        self.rewards_chosen_accumulated = []
        self.rewards_rejected_accumulated = []
        self.loss_hyperparameters = loss_hyperparameters
        self.visualization_steps = visualization_steps
        self.max_viz_samples = max_viz_samples

    def get_eval_dataloader(
        self, eval_dataset: Optional[Union[str, Dataset]] = None
    ) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                If a `str`, will use `self.eval_dataset[eval_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.eval_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset if eval_dataset is not None else self.eval_dataset
        )

        data_collator = self.data_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(
                eval_dataset, description="evaluation"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="evaluation"
            )

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            visual_parameters = []
            merger_parameters = []
            rm_head_parameters = []

            if self.args.vision_lr is not None:
                lr_mapper["visual"] = self.args.vision_lr
                visual_parameters = [
                    name
                    for name, _ in opt_model.named_parameters()
                    if "visual" in name and "merger" not in name
                ]
            if self.args.merger_lr is not None:
                lr_mapper["merger"] = self.args.merger_lr
                merger_parameters = [
                    name for name, _ in opt_model.named_parameters() if "merger" in name
                ]
                
            if self.args.rm_head_lr is not None:
                lr_mapper["rm_head"] = self.args.rm_head_lr
                rm_head_parameters = [
                    name for name, _ in opt_model.named_parameters() if "rm_head" in name
                ]

            if len(lr_mapper) > 0:
                special_lr_parameters = merger_parameters + visual_parameters + rm_head_parameters

                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in special_lr_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in special_lr_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                ]

                if visual_parameters:
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (
                                        n in decay_parameters
                                        and n in visual_parameters
                                        and p.requires_grad
                                    )
                                ],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.vision_lr,
                            },
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (
                                        n not in decay_parameters
                                        and n in visual_parameters
                                        and p.requires_grad
                                    )
                                ],
                                "weight_decay": 0.0,
                                "lr": self.args.vision_lr,
                            },
                        ]
                    )

                if merger_parameters:
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (
                                        n in decay_parameters
                                        and n in merger_parameters
                                        and p.requires_grad
                                    )
                                ],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.merger_lr,
                            },
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (
                                        n not in decay_parameters
                                        and n in merger_parameters
                                        and p.requires_grad
                                    )
                                ],
                                "weight_decay": 0.0,
                                "lr": self.args.merger_lr,
                            },
                        ]
                    )
                
                if rm_head_parameters:
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (
                                        n in decay_parameters
                                        and n in rm_head_parameters
                                        and p.requires_grad
                                    )
                                ],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.rm_head_lr,
                            },
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (
                                        n not in decay_parameters
                                        and n in rm_head_parameters
                                        and p.requires_grad
                                    )
                                ],
                                "weight_decay": 0.0,
                                "lr": self.args.rm_head_lr,
                            },
                        ]
                    )

            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            if self.model.special_token_ids:
                special_token_embeddings = opt_model.get_input_embeddings().weight

                special_token_embeddings.requires_grad = True

                optimizer_grouped_parameters.extend(
                    [
                        {
                            # "params": [p for n, p in opt_model.get_input_embeddings().named_parameters() if (p.requires_grad)],
                            "params": [special_token_embeddings],
                            "lr": self.args.special_token_lr,
                            "weight_decay": 0.0,
                        },
                    ]
                )

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                self.args, opt_model
            )

            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )

        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        rewards_A = model(return_dict=True, **inputs["batch_1"])["logits"]
        rewards_B = model(return_dict=True, **inputs["batch_2"])["logits"]
        
        # Log to TensorBoard for visualization
        if (hasattr(self.state, 'global_step') and 
            self.state.global_step % self.visualization_steps == 0 and
            self.state.global_step > 0):
            # Pass the original inputs which should contain the text prompts
            self._log_training_visualization(inputs, rewards_A, rewards_B)
        
        # calculate loss, optionally modulate with margin
        # get chosen and rejected rewards from the chosen label
        (
            rewards_chosen,
            rewards_rejected,
            nontied_mask,
        ) = _convert_A_B_to_chosen_rejected(
            rewards_A,
            rewards_B,
            tied_threshold=self.tied_threshold,
            choice_dist=inputs["choice_dist"],
        )

        loss_dict = {}

        if self.loss_type == "bt":
            # Bradley-Terry model
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected)
            out_mask = nontied_mask
            loss = loss * out_mask
            loss = loss.mean()
        elif self.loss_type == "likelihood_displacement":
            # Bradley-Terry model
            loss = -nn.functional.logsigmoid(rewards_chosen - self.loss_hyperparameters['tau'] * rewards_rejected)
            out_mask = nontied_mask
            loss = loss * out_mask
            loss = loss.mean()

        elif self.loss_type == "constant_margin":
            # Bradley-Terry model with constant margin
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected - 0.57)
            out_mask = nontied_mask
            loss = loss * out_mask
            loss = loss.mean()
        elif self.loss_type == "btt":
            # Bradley-Terry-With-Ties model
            k = 5.0
            log_k = math.log(k)
            log_k2_sub_1 = math.log(k**2 - 1)
            bt_loss = -nn.functional.logsigmoid(
                rewards_chosen - rewards_rejected - log_k
            )
            same_loss = (
                -nn.functional.logsigmoid(rewards_chosen - rewards_rejected - log_k)
                - nn.functional.logsigmoid(rewards_rejected - rewards_chosen - log_k)
                - log_k2_sub_1
            )
            loss = bt_loss * nontied_mask.float() + same_loss * (
                1 - nontied_mask.float()
            )
            out_mask = torch.ones_like(nontied_mask, dtype=torch.float32).to(
                rewards_A.device
            )  # [B, 1]
            loss = loss * out_mask

            loss = loss.mean()
        elif self.loss_type == "hpsv2":
            device = rewards_A.device
            rewards = torch.nn.functional.softmax(
                torch.cat([rewards_A, rewards_B], dim=-1), dim=-1
            )
            text_0_logits, text_1_logits = rewards[:, 0], rewards[:, 1]
            label_0, label_1 = torch.ones_like(text_0_logits), torch.zeros_like(
                text_0_logits
            )

            text_logits = torch.stack([text_0_logits, text_1_logits], dim=-1)
            text_0_labels = torch.zeros(
                text_logits.shape[0], device=device, dtype=torch.long
            )
            text_1_labels = text_0_labels + 1

            text_0_loss = torch.nn.functional.cross_entropy(
                text_logits, text_0_labels, reduction="none"
            )
            text_1_loss = torch.nn.functional.cross_entropy(
                text_logits, text_1_labels, reduction="none"
            )

            loss = label_0 * text_0_loss + label_1 * text_1_loss

            # absolute_example_weight = 1 / num_per_prompt
            # denominator = absolute_example_weight.sum()
            # weight_per_example = absolute_example_weight / denominator
            # text_loss *= weight_per_example
            loss = loss.sum()
        elif self.loss_type == "uncertainty":
            batch_size = rewards_A.shape[0]
            mean_chosen = rewards_A[:, 0]
            mean_rejected = rewards_B[:, 0]
            sigma_chosen = torch.exp(rewards_A[:, 1])
            sigma_rejected = torch.exp(rewards_B[:, 1])

            mean_z = mean_chosen - mean_rejected
            sigma_z = torch.sqrt(sigma_chosen**2 + sigma_rejected**2)

            z_samples = torch.randn(batch_size, 1000).to(sigma_z.device).to(
                torch.float16
            ) * sigma_z.unsqueeze(1).repeat(1, 1000) + mean_z.unsqueeze(1).repeat(
                1, 1000
            )
            loss = -torch.nn.functional.logsigmoid(z_samples).mean()
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented.")

        loss_dict.update({"loss": loss.item()})

        if return_outputs:
            ## return rewards_A/B instead of chosen/rejected
            ## easier to calculate metrics for multi-attribute
            return loss, {
                "rewards_A": rewards_A,
                "rewards_B": rewards_B,
            }
        return loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        model.eval()
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, logits_dict = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)
        loss = loss.detach()
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = nested_detach(logits)
        if self.loss_type != "uncertainty":
            logits = torch.cat(logits, dim=1)  # [B, 2]
        else:
            logits = torch.cat([p[:, [0]] for p in logits], dim=1)

        labels = torch.ones((logits.shape[0], 1)).to(logits.device)

        return loss, logits, labels

    def _log_training_visualization(self, inputs, rewards_A, rewards_B):
        """Log training samples and predictions to TensorBoard"""
        try:
            # Get tensorboard writer from trainer
            writer = None
            if hasattr(self, 'log_metrics'):
                # Try to get the writer from the logger
                if hasattr(self.args, 'report_to') and 'tensorboard' in self.args.report_to:
                    from torch.utils.tensorboard import SummaryWriter
                    if not hasattr(self, '_tb_writer'):
                        self._tb_writer = SummaryWriter(log_dir=self.args.logging_dir)
                    writer = self._tb_writer
            
            if writer is None:
                return
            
            step = self.state.global_step
            batch_size = min(len(rewards_A), self.max_viz_samples)
            
            # Log scalar metrics
            for i in range(batch_size):
                score_A = rewards_A[i].float().detach().cpu().numpy()
                score_B = rewards_B[i].float().detach().cpu().numpy()
                
                # Convert to float for logging
                score_A_val = float(score_A.mean()) if score_A.ndim > 0 else float(score_A)
                score_B_val = float(score_B.mean()) if score_B.ndim > 0 else float(score_B)
                score_diff = score_A_val - score_B_val
                
                writer.add_scalar(f'train_viz/sample_{i}/score_A', score_A_val, step)
                writer.add_scalar(f'train_viz/sample_{i}/score_B', score_B_val, step)
                writer.add_scalar(f'train_viz/sample_{i}/score_diff', score_diff, step)
                
                try:
                    # Get image data from inputs
                    image_A = inputs['image_1'][i] if 'image_1' in inputs else None
                    image_B = inputs['image_2'][i] if 'image_2' in inputs else None
                    
                    # Get prompt text from the original batch (now properly stored)
                    prompt_A = inputs.get('text_1', ['Unknown prompt'])[i] if 'text_1' in inputs else 'Unknown prompt'
                    
                    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
                    fig.text(0.05, 0.05, f'Prompt:\n{prompt_A[:200]}{"..." if len(prompt_A) > 200 else ""}', 
                                 ha='left', va='bottom', fontsize=8, wrap=True,
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
                    img_A_np = np.array(image_A)
                    if img_A_np.ndim == 3 and img_A_np.shape[0] == 3:  # CHW format
                        img_A_np = np.transpose(img_A_np, (1, 2, 0))
                    img_A_np = np.clip(img_A_np, 0, 1)  # Ensure values are in [0,1]
                    axes[0].imshow(img_A_np)
                    axes[0].set_title(f'Image A - Score: {score_A_val:.3f}')
                    axes[0].axis('off')
                    
                    img_B_np = np.array(image_B)
                    if img_B_np.ndim == 3 and img_B_np.shape[0] == 3:  # CHW format
                        img_B_np = np.transpose(img_B_np, (1, 2, 0))
                    img_B_np = np.clip(img_B_np, 0, 1)  # Ensure values are in [0,1]
                    axes[1].imshow(img_B_np)
                
                    axes[1].set_title(f'Image B - Score: {score_B_val:.3f}')
                    axes[1].axis('off')
                    
                    # Add prediction info
                    winner = "A" if score_diff > 0 else "B"
                    plt.suptitle(f'Step {step} - Sample {i} | Predicted Winner: Image {winner} | Diff: {score_diff:.3f}', fontsize=14)
                    plt.tight_layout()

                    # Log figure to tensorboard
                    writer.add_figure(f'train_viz/sample_{i}_comparison', fig, step)
                    plt.close(fig)
                except Exception as viz_error:
                    print(f"Warning: Could not extract images for visualization: {viz_error}")
                    continue

            # Log aggregate statistics
            all_scores_A = rewards_A.float().detach().cpu().numpy()
            all_scores_B = rewards_B.float().detach().cpu().numpy()
            
            writer.add_histogram('train_viz/all_scores_A', all_scores_A, step)
            writer.add_histogram('train_viz/all_scores_B', all_scores_B, step)
            writer.add_scalar('train_viz/mean_score_A', float(all_scores_A.mean()), step)
            writer.add_scalar('train_viz/mean_score_B', float(all_scores_B.mean()), step)
            writer.add_scalar('train_viz/mean_score_diff', float((all_scores_A - all_scores_B).mean()), step)
            
        except Exception as e:
            print(f"Error in training visualization: {e}")

    def _save_checkpoint(self, model, trial, metrics=None):

        if isinstance(self.model, PeftModel):
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is None and trial is None:
                self.store_flos()

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            os.makedirs(output_dir, exist_ok=True)

            # TODO: Just Temp
            self.save_model(output_dir, _internal_call=True)
            # pdb.set_trace()

            if not self.args.save_full_model:
                non_lora_weights = get_peft_state_non_lora_maybe_zero_3(
                    self.model.named_parameters(), require_grad_only=True
                )
                torch.save(
                    non_lora_weights,
                    os.path.join(output_dir, "non_lora_state_dict.pth"),
                )
            # safetensors.torch.save(non_lora_weights, os.path.join(output_dir, "non_lora_model.safetensors"))

            if not self.args.save_only_model:
                # Save optimizer and scheduler
                self._save_optimizer_and_scheduler(output_dir)
                # Save RNG state
                self._save_rng_state(output_dir)

        else:
            super(RewardTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # pdb.set_trace()

        supported_classes = (
            (PreTrainedModel,)
            if not is_peft_available()
            else (PreTrainedModel, PeftModel)
        )
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
                self.accelerator.unwrap_model(self.model).save_pretrained(
                    output_dir,
                    state_dict=state_dict,
                    safe_serialization=self.args.save_safetensors,
                )
            else:
                logger.info(
                    "Trainer.model is not a `PreTrainedModel`, only saving its state dict."
                )
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict,
                        os.path.join(output_dir, SAFE_WEIGHTS_NAME),
                        metadata={"format": "pt"},
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            if not self.args.save_full_model:
                state_dict = {k: v for k, v in state_dict.items() if "wte" not in k}
                self.model.save_pretrained(
                    output_dir,
                    state_dict=state_dict,
                    safe_serialization=self.args.save_safetensors,
                )
            else:
                torch.save(state_dict, os.path.join(output_dir, "model.pth"))

        if self.tokenizer is not None:
            os.makedirs(os.path.join(output_dir, "tokenizer"), exist_ok=True)
            self.tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        # pdb.set_trace()


def compute_multi_attr_accuracy(eval_pred, metainfo_idxs=None) -> Dict[str, float]:
    predictions, labels = eval_pred
    metrics = {}

    pred_curr = predictions
    label_curr = labels.squeeze(1)
    total_count = np.sum(label_curr != 0)

    rewards_chosen = pred_curr[:, 0]
    rewards_rejected = pred_curr[:, 1]

    rewards_chosen_avg = np.sum(rewards_chosen) / total_count
    rewards_rejected_avg = np.sum(rewards_rejected) / total_count

    accuracy = np.sum(rewards_chosen > rewards_rejected) / total_count

    metrics.update(
        {
            f"Acc": accuracy,
            f"R_chosen_avg": rewards_chosen_avg,
            f"R_rejected_avg": rewards_rejected_avg,
        }
    )
    return metrics
