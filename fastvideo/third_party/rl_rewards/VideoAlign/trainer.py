import math
import os

# from training.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3
import numpy as np
import pandas as pd
import safetensors
import torch
from peft import PeftModel
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import Qwen2VLForConditionalGeneration
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer import (
    PREFIX_CHECKPOINT_DIR,
    SAFE_WEIGHTS_NAME,
    TRAINING_ARGS_NAME,
    WEIGHTS_NAME,
    TrainerCallback,
    is_datasets_available,
    is_peft_available,
    is_sagemaker_mp_enabled,
    is_torch_xla_available,
    logger,
)
from transformers.trainer_pt_utils import nested_detach
from trl import RewardTrainer
from utils import get_peft_state_non_lora_maybe_zero_3

import datasets

if is_torch_xla_available():
    pass
else:
    IS_XLA_FSDPV2_POST_2_2 = False


class Qwen2VLRewardModelBT(Qwen2VLForConditionalGeneration):
    def __init__(self, config, output_dim=4, reward_token="last", special_token_ids=None):
        super().__init__(config)
        # pdb.set_trace()
        self.output_dim = output_dim
        self.rm_head = nn.Linear(config.hidden_size, output_dim, bias=False)
        self.reward_token = reward_token

        self.special_token_ids = special_token_ids
        if self.special_token_ids is not None:
            self.reward_token = "special"

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        rope_deltas: torch.LongTensor | None = None,
    ):
        ## modified from the origin class Qwen2VLForConditionalGeneration
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # pdb.set_trace()
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
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

        logits = self.rm_head(hidden_states)  # [B, L, N]

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        ## get sequence length
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        elif input_ids is not None:
            # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
            sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            sequence_lengths = sequence_lengths.to(logits.device)
        else:
            sequence_lengths = -1

        ## get the last token's logits
        if self.reward_token == "last":
            pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        elif self.reward_token == "mean":
            ## get the mean of all valid tokens' logits
            valid_lengths = torch.clamp(sequence_lengths, min=0, max=logits.size(1) - 1)
            pooled_logits = torch.stack([logits[i, : valid_lengths[i]].mean(dim=0) for i in range(batch_size)])
        elif self.reward_token == "special":
            # special_token_ids = self.tokenizer.convert_tokens_to_ids(self.special_tokens)
            # create a mask for special tokens
            special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for special_token_id in self.special_token_ids:
                special_token_mask = special_token_mask | (input_ids == special_token_id)
            pooled_logits = logits[special_token_mask, ...]
            pooled_logits = pooled_logits.view(batch_size, 3, -1)  # [B, 3, N] assert 3 attributes
            if self.output_dim == 3:
                pooled_logits = pooled_logits.diagonal(dim1=1, dim2=2)
            pooled_logits = pooled_logits.view(batch_size, -1)

            # pdb.set_trace()
        else:
            raise ValueError("Invalid reward_token")

        return {"logits": pooled_logits}


def _convert_A_B_to_chosen_rejected(rewards_A, rewards_B, scores_A, scores_B, chosen_label, label_dim=None):
    """
    Inputs:
        rewards_A: [B, N]
        rewards_B: [B, N]
        scores_A: [B, N]
        scores_B: [B, N]
        chosen_label: [B, N]
    Outputs:
        rewards_chosen: [B, N]
        rewards_rejected: [B, N]
        scores_chosen: [B, N]
        scores_rejected: [B, N]
        nontied_mask: [B, N] (preference labels that is not tied)
        valid_mask: [B, N]  (all valid labels)
    """
    chosen_mask = chosen_label == 1
    # rejected_mask = (chosen_label == -1)
    rejected_mask = chosen_label != 1
    if label_dim is not None:
        N = chosen_label.size(1)
        chosen_mask = chosen_mask[:, label_dim].unsqueeze(1).expand(-1, N)
        rejected_mask = rejected_mask[:, label_dim].unsqueeze(1).expand(-1, N)

    rewards_chosen = torch.where(chosen_mask, rewards_A, rewards_B)
    rewards_rejected = torch.where(rejected_mask, rewards_A, rewards_B)
    scores_chosen = torch.where(chosen_mask, scores_A, scores_B)
    scores_rejected = torch.where(rejected_mask, scores_A, scores_B)

    nontied_mask = ((chosen_label == 1) | (chosen_label == -1)).float()
    if label_dim is not None:
        nontied_mask = nontied_mask[:, label_dim].unsqueeze(1).expand(-1, N)

    valid_mask = (chosen_label != 22).float()
    if label_dim is not None:
        valid_mask = valid_mask[:, label_dim].unsqueeze(1).expand(-1, N)
    # rewards_chosen = rewards_chosen * valid_mask
    # rewards_rejected = rewards_rejected * valid_mask

    return (
        rewards_chosen,
        rewards_rejected,
        scores_chosen,
        scores_rejected,
        nontied_mask,
        valid_mask,
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
            model.get_input_embeddings().weight[index_no_updates] = self.orig_embeds_params[index_no_updates]


class VideoVLMRewardTrainer(RewardTrainer):
    def __init__(self, loss_type="regular", enable_noise_in_eval=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.loss_type = loss_type
        self.enable_noise_in_eval = enable_noise_in_eval

        self.rewards_chosen_accumulated = []
        self.rewards_rejected_accumulated = []
        self.scores_chosen_accumulated = []
        self.scores_rejected_accumulated = []

    def get_eval_dataloader(self, eval_dataset: str | Dataset | None = None) -> DataLoader:
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
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )
        data_collator = lambda features: self.data_collator(features, enable_noise=self.enable_noise_in_eval)

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

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

            if self.args.vision_lr is not None:
                lr_mapper["visual"] = self.args.vision_lr
                visual_parameters = [
                    name for name, _ in opt_model.named_parameters() if "visual" in name and "merger" not in name
                ]
            if self.args.merger_lr is not None:
                lr_mapper["merger"] = self.args.merger_lr
                merger_parameters = [name for name, _ in opt_model.named_parameters() if "merger" in name]

            if len(lr_mapper) > 0:
                special_lr_parameters = merger_parameters + visual_parameters

                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)
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
                                    if (n in decay_parameters and n in visual_parameters and p.requires_grad)
                                ],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.vision_lr,
                            },
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (n not in decay_parameters and n in visual_parameters and p.requires_grad)
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
                                    if (n in decay_parameters and n in merger_parameters and p.requires_grad)
                                ],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.merger_lr,
                            },
                            {
                                "params": [
                                    p
                                    for n, p in opt_model.named_parameters()
                                    if (n not in decay_parameters and n in merger_parameters and p.requires_grad)
                                ],
                                "weight_decay": 0.0,
                                "lr": self.args.merger_lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
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

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    # def training_step(self, model, inputs, num_items_in_batch=None):
    #     pdb.set_trace()
    #     return super(VideoVLMRewardTrainer, self).training_step(model, inputs, num_items_in_batch)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
    ):
        rewards_A = model(return_dict=True, **inputs["A"])["logits"]
        rewards_B = model(return_dict=True, **inputs["B"])["logits"]
        # calculate loss, optionally modulate with margin
        # get chosen and rejected rewards from the chosen label
        (
            rewards_chosen,
            rewards_rejected,
            scores_chosen,
            scores_rejected,
            nontied_mask,
            valid_mask,
        ) = _convert_A_B_to_chosen_rejected(
            rewards_A,
            rewards_B,
            inputs["A_scores"],
            inputs["B_scores"],
            inputs["chosen_label"],
        )
        # pdb.set_trace()
        inputs["margin"] = scores_chosen - scores_rejected

        loss_dict = {}

        if self.loss_type == "bt":
            # Bradley-Terry model
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected)
            out_mask = nontied_mask
        elif self.loss_type == "margin":
            # Bradley-Terry model with margin
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected - inputs["margin"])
            out_mask = nontied_mask
        elif self.loss_type == "constant_margin":
            # Bradley-Terry model with constant margin
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected - 0.57)
            out_mask = nontied_mask
        elif self.loss_type == "scaled":
            # Bradley-Terry model with scaled margin
            loss = -(inputs["margin"] + 0.0) * nn.functional.logsigmoid(rewards_chosen - rewards_rejected)
            out_mask = nontied_mask
        elif self.loss_type == "reg":
            # regression loss
            rewards = torch.stack([rewards_A, rewards_B], dim=1)
            scores = torch.stack([inputs["A_scores"], inputs["B_scores"]], dim=1)
            out_mask = scores != 0.0
            scores = scores - 3.0  # rescale
            # pdb.set_trace()
            loss = nn.functional.mse_loss(rewards, scores, reduction="none")
        elif self.loss_type == "btt":
            # Bradley-Terry-With-Ties model
            k = 5.0
            log_k = math.log(k)
            log_k2_sub_1 = math.log(k**2 - 1)
            bt_loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected - log_k)
            same_loss = (
                -nn.functional.logsigmoid(rewards_chosen - rewards_rejected - log_k)
                - nn.functional.logsigmoid(rewards_rejected - rewards_chosen - log_k)
                - log_k2_sub_1
            )
            loss = bt_loss * nontied_mask + same_loss * (1 - nontied_mask)
            out_mask = valid_mask
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented.")

        loss = loss * out_mask

        loss = loss.mean()
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
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, logits_dict = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)
        loss = loss.detach()
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = nested_detach(logits)
        logits = torch.stack(logits).permute(1, 0, 2)  # [B, 2, N]

        labels = inputs["chosen_label"]  # [B, N], values in {-1, 0, 1}

        return loss, logits, labels

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
            super()._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: str | None = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # pdb.set_trace()

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
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
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict,
                        os.path.join(output_dir, SAFE_WEIGHTS_NAME),
                        metadata={"format": "pt"},
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        elif not self.args.save_full_model:
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


def compute_multi_attr_accuracy(eval_pred, metainfo_idxs=None, eval_dims=None, save_path=None) -> dict[str, float]:
    predictions, labels = eval_pred
    metrics = {}
    for idx, eval_dim in enumerate(eval_dims):
        pred_curr = predictions[:, :, idx]
        label_curr = labels[:, idx]
        # pdb.set_trace()
        ## calculate the average scores of rewards_chosen and rewards_rejected
        valid_mask = label_curr != 0

        rewards_chosen = np.where(label_curr == 1, pred_curr[:, 0], pred_curr[:, 1])
        rewards_rejected = np.where(label_curr == -1, pred_curr[:, 0], pred_curr[:, 1])

        rewards_chosen_avg = np.sum(rewards_chosen * valid_mask) / np.sum(valid_mask)
        rewards_rejected_avg = np.sum(rewards_rejected * valid_mask) / np.sum(valid_mask)

        pred_curr = np.argmax(pred_curr, axis=1)
        pred_curr = np.where(pred_curr == 0, 1, -1)
        accuracy = np.array(pred_curr == label_curr, dtype=float)
        accuracy = np.sum(accuracy * valid_mask) / np.sum(valid_mask)

        metrics.update(
            {
                f"accuracy_{eval_dim}": accuracy,
                f"rewards_chosen_avg_{eval_dim}": rewards_chosen_avg,
                f"rewards_rejected_avg_{eval_dim}": rewards_rejected_avg,
            }
        )

    if save_path is not None and metainfo_idxs is not None:
        df = pd.DataFrame(metainfo_idxs, columns=["metainfo_idx"])
        for idx, eval_dim in enumerate(eval_dims):
            rewards_A = predictions[:, 0, idx]
            rewards_B = predictions[:, 1, idx]
            df[f"reward_A_{eval_dim}"] = rewards_A
            df[f"reward_B_{eval_dim}"] = rewards_B

        df.to_csv(save_path, index=False)
        print(f"===> Inference results saved to {save_path}")

    # pdb.set_trace()
    return metrics
