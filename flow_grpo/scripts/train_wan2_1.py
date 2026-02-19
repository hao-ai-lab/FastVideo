from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json

from six import print_
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler, WanPipeline
from diffusers.loaders import AttnProcsLayers
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
import numpy as np
import flow_grpo.prompts
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.wan_pipeline_with_logprob import wan_pipeline_with_logprob, sde_step_with_logprob
from flow_grpo.diffusers_patch.wan_prompt_embedding import encode_prompt
import torch
import wandb
from functools import partial
import tqdm
import tempfile
import itertools
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
from peft.utils import get_peft_model_state_dict
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper
import imageio
from safetensors.torch import save_file as save_safetensors
tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__, log_level="INFO")

# Align-debug logs: under ALIGN_LOGS_ROOT (env var) or flow_grpo root. Structure: <root>/align_logs/flow_logs
_FLOW_GRPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_align_root = os.environ.get("ALIGN_LOGS_ROOT")
ALIGN_FLOW_LOGS_DIR = os.path.join(
    os.path.abspath(_align_root) if _align_root else _FLOW_GRPO_ROOT,
    "align_logs",
    "flow_logs",
)
# Decoded videos for FastVideo reward verification: <root>/align_logs/fv_logs/decoded_videos
DECODED_VIDEOS_DIR = os.path.join(
    os.path.abspath(_align_root) if _align_root else _FLOW_GRPO_ROOT,
    "align_logs",
    "fv_logs",
    "decoded_videos",
)

# Debug: log rewards/advantages for first batch only once, then stop
_debug_rewards_advantages_logged = False

# Same as FastVideo rl_prompt_dataset.FIXED_DEBUG_PROMPT for alignment
FIXED_DEBUG_PROMPT = (
    "\"A detailed, vintage-style alien abduction pamphlet titled Probing FAQs, featuring illustrations of extraterrestrial beings and spacecraft, alongside text explaining common questions and procedures.\""
)

class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size  # 每卡的batch大小
        self.k = k                    # 每个样本重复的次数
        self.num_replicas = num_replicas  # 总卡数
        self.rank = rank              # 当前卡编号
        self.seed = seed              # 随机种子，用于同步
        
        # 计算每个迭代需要的不同样本数
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not div n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # 不同样本数
        self.epoch=0
        self.hardcoded_manual_seed = 20

    def __iter__(self):
        while True:
            # 生成确定性的随机序列，确保所有卡同步
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            print(f"sampler manual_seed: {self.seed + self.epoch}")
            # g.manual_seed(self.hardcoded_manual_seed)
            # self.hardcoded_manual_seed += 1
            # print('epoch', self.epoch)
            # 随机选择m个不同的样本
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            # print(self.rank, 'indices', indices)
            # 每个样本重复k次，生成总样本数n*b
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # 打乱顺序确保均匀分配
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            # print(self.rank, 'shuffled_samples', shuffled_samples)
            # 将样本分割到各个卡
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            # print(self.rank, 'per_card_samples', per_card_samples[self.rank])
            # 返回当前卡的样本索引
            yield per_card_samples[self.rank]
    
    def set_epoch(self, epoch):
        self.epoch = epoch  # 用于同步不同 epoch 的随机状态

def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        # pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds

def set_adapter_and_freeze_params(transformer, adapter_name):
    transformer.module.set_adapter(adapter_name)
    for name, param in transformer.named_parameters():
        if "learner" in name:
            param.requires_grad_(True)
        elif "ref" in name:
            param.requires_grad_(False)

def calculate_zero_std_ratio(prompts, gathered_rewards):
    """
    计算每个唯一提示词对应奖励值的标准差为零的比例
    
    参数:
        prompts: 提示词列表
        gathered_rewards: 包含奖励值的字典，须包含'ori_avg'键
        
    返回:
        zero_std_ratio: 标准差为零的比例
        prompt_std_devs: 每个唯一提示词对应的标准差数组
    """
    # 将提示词列表转换为NumPy数组
    prompt_array = np.array(prompts)
    
    # 获取唯一提示词及其分组信息
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, 
        return_inverse=True,
        return_counts=True
    )
    
    # 分组获取每个提示词对应的奖励值
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    
    # 计算每个分组的标准差
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    
    # 计算零标准差的比例
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    
    return zero_std_ratio
    

def get_sigmas(noise_scheduler, timesteps, accelerator, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
    timesteps = timesteps.to(accelerator.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma
        
def compute_log_prob(transformer, pipeline, sample, j, embeds, negative_embeds, config, **kwargs):
    attention_kwargs = kwargs.get('attention_kwargs', getattr(config, 'attention_kwargs', None))
    if config.train.cfg:
        noise_pred_text = transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j],
            encoder_hidden_states=embeds,  # Should contain both neg and pos embeds
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]
        noise_pred_uncond = transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j],
            encoder_hidden_states=negative_embeds,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]
        noise_pred = (
            noise_pred_uncond
            + config.sample.guidance_scale
            * (noise_pred_text - noise_pred_uncond)
        )
    else:
        noise_pred = transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j],
            encoder_hidden_states=embeds,
            return_dict=False,
        )[0]
    
    # compute the log prob of next_latents given latents under the current model
    prev_sample, log_prob, prev_sample_mean, std_dev_t, dt = sde_step_with_logprob(
        pipeline.scheduler,
        noise_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
        return_dt_and_std_dev_t=True
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t, dt

def eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    neg_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=512, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)

    all_rewards = defaultdict(list)
    for test_batch in tqdm(
            test_dataloader,
            desc="Eval: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
        prompts, prompt_metadata = test_batch
        prompt_embeds = compute_text_embeddings(
            prompts, 
            text_encoders, 
            tokenizers, 
            max_sequence_length=512,
            device=accelerator.device
        )
        # 最后一个batch可能不够batch_size
        if len(prompt_embeds)<len(sample_neg_prompt_embeds):
            sample_neg_prompt_embeds = sample_neg_prompt_embeds[:len(prompt_embeds)]

        with autocast():
            with torch.no_grad():
                videos, latents, log_probs, _ = wan_pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    return_dict=False,
                    num_frames=config.frames,
                    height=config.height,
                    width=config.width, 
                    determistic=True,
                )
        rewards = executor.submit(reward_fn, videos, prompts, prompt_metadata, only_strict=False)
        # yield to to make sure reward computation starts
        time.sleep(0)
        rewards, reward_metadata = rewards.result()

        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)
    
    last_batch_videos_gather = accelerator.gather(torch.as_tensor(videos, device=accelerator.device)).cpu().numpy()
    last_batch_prompt_ids = tokenizers[0](
        prompts,
        padding="max_length",
        max_length=512, 
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(accelerator.device)

    last_batch_prompt_ids_gather = accelerator.gather(last_batch_prompt_ids).cpu().numpy()
    last_batch_prompts_gather = pipeline.tokenizer.batch_decode(
        last_batch_prompt_ids_gather, skip_special_tokens=True
    )
    last_batch_rewards_gather = {}
    for key, value in rewards.items():
        last_batch_rewards_gather[key] = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    if accelerator.is_main_process:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(15, len(last_batch_videos_gather))
            sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                video = last_batch_videos_gather[index].transpose(0, 2, 3, 1)
                frames = [img for img in video]
                frames = [(frame * 255).astype(np.uint8) for frame in frames]
                imageio.mimsave(os.path.join(tmpdir, f"{idx}.mp4"), frames, fps=8, codec="libx264", format='FFMPEG')

            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            sampled_rewards = [{k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather} for index in sample_indices]
            for key, value in all_rewards.items():
                print(key, value.shape)
            accelerator.log(
                {
                    "eval_images": [
                        wandb.Video(
                            os.path.join(tmpdir, f"{idx}.mp4"),
                            caption=f"{prompt:.1000} | " + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                            format="mp4",
                            fps=8 
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                    ],
                    **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()},
                },
                step=global_step,
            )
    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(
                filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from))
            )
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    train_timesteps = [step_index  for step_index in range(num_train_timesteps)]
    gradient_accumulation_steps = config.train.gradient_accumulation_steps * num_train_timesteps

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    wandb_project_name = "wan_flow_grpo"
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=wandb_project_name,
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name}},
        )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    pipeline = WanPipeline.from_pretrained(
        config.pretrained.model
    )
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)

    pipeline.transformer.requires_grad_(not config.use_lora)

    text_encoders = [pipeline.text_encoder]
    tokenizers = [pipeline.tokenizer]

    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move transformer, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    # pipeline.scheduler.to(accelerator.device, dtype=inference_dtype)

    if config.use_lora:
        # pipeline.transformer.to(accelerator.device, dtype=inference_dtype)
        pipeline.transformer.to(accelerator.device)
        
        # pipeline.transformer.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if config.use_lora:
        # Set correct lora layers
        target_modules = [
            "add_k_proj",
            "add_q_proj",
            "add_v_proj",
            "to_add_out",
            "to_k",
            "to_out.0",
            "to_q",
            "to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path)
            # 使用PeftModel.from_pretrained load后所有参数的requires_grad都是False，需要set_adapter来使得adapter参数梯度为True
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)
    
    transformer = pipeline.transformer
    transformer.enable_gradient_checkpointing()
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    # 平均影响到之前的20*8=160个step
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # prepare prompt and reward fn
    reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
    eval_reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)

    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, 'train')
        test_dataset = TextPromptDataset(config.dataset, 'test')

        # 创建无限循环的DataLoader
        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,  # 你的k值
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        # 创建DataLoader，注意这里不需要shuffle，由Sampler控制
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=0,
            collate_fn=TextPromptDataset.collate_fn,
            # persistent_workers=True
        )
        # 创建正常的DataLoader
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=TextPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )
    
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, 'train')
        test_dataset = GenevalPromptDataset(config.dataset, 'test')
        # 创建无限循环的DataLoader
        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,  # 你的k值
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        # 创建DataLoader，注意这里不需要shuffle，由Sampler控制
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=GenevalPromptDataset.collate_fn,
            # persistent_workers=True
        )
        # 创建正常的DataLoader
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=GenevalPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )
    else:
        raise NotImplementedError("Only general_ocr is supported with dataset")


    neg_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=512, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size * config.sample.sample_time_per_prompt, 1, 1)

    if config.sample.num_image_per_prompt * config.sample.sample_time_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    # initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, test_dataloader = accelerator.prepare(transformer, optimizer, train_dataloader, test_dataloader)

    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=8)

    # Train!
    samples_per_epoch = (
        config.sample.train_batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")
    # assert config.sample.train_batch_size >= config.train.batch_size
    # assert config.sample.train_batch_size % config.train.batch_size == 0
    # assert samples_per_epoch % total_train_batch_size == 0

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
    else:
        first_epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)

    for epoch in range(first_epoch, config.num_epochs):
        #################### SAMPLING ####################
        pipeline.transformer.eval()
        samples = []
        prompts = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)

            if accelerator.is_main_process and epoch >= 2:
                print(f"===local prompts===")
                for p in prompts:
                    print(p)

            prompt_embeds = compute_text_embeddings(
                prompts, 
                text_encoders, 
                tokenizers, 
                max_sequence_length=512,
                device=accelerator.device
            )
            prompt_ids = tokenizers[0](
                prompts,
                padding="max_length",
                max_length=512, 
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)
            # Gather prompts across all ranks for this batch and write to prompts.txt (before any continue)
            # Only log when epoch >= 2; first 2 epochs are skipped so we should not log in skipped epochs.
            prompt_ids_gathered = accelerator.gather(prompt_ids).cpu().numpy()
            if accelerator.is_main_process and epoch >= 2:
                out_dir = ALIGN_FLOW_LOGS_DIR
                os.makedirs(out_dir, exist_ok=True)
                prompts_gathered = pipeline.tokenizer.batch_decode(
                    prompt_ids_gathered, skip_special_tokens=True
                )
                prompts_path = os.path.join(out_dir, "prompts.txt")
                if i == 0:
                    open(prompts_path, "w").close()
                with open(prompts_path, "a") as f:
                    if i > 0:
                        f.write("\n")
                    f.write(f"=== batch {i} ===\n")
                    f.write("\n".join(prompts_gathered) + "\n")
            if i==0 and epoch % config.eval_freq == 0: # and epoch>0:
                eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, eval_reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters)
            if i==0 and epoch % config.save_freq == 0 and epoch>0 and accelerator.is_main_process:
                save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)
            # 这里是故意的，因为前两个epoch收集的group size会有bug,经过两个epoch后，group_size稳定成指定的
            if epoch < 2:
                continue
            # sample
            for j in tqdm(
                range(config.sample.sample_time_per_prompt),
                desc=f"Epoch {epoch}: sampling | multi sample per prompt",
                disable=not accelerator.is_local_main_process,
                position=1,
            ):
                # myregion debug: save videos and log metrics for every batch (epoch >= 2 only), aligned with FastVideo
                do_debug_save = accelerator.is_main_process and epoch >= 2
                debug_gen = torch.Generator("cpu").manual_seed(42 + i * 100 + j) if do_debug_save else None
                with autocast():
                    with torch.no_grad():
                        out = wan_pipeline_with_logprob(
                            pipeline,
                            prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=sample_neg_prompt_embeds,
                            num_inference_steps=config.sample.num_steps,
                            guidance_scale=config.sample.guidance_scale,
                            output_type="pt",
                            return_dict=False,
                            num_frames=config.frames,
                            height=config.height,
                            width=config.width,
                            kl_reward=config.sample.kl_reward,
                            generator=debug_gen,
                            return_debug_metrics=do_debug_save,
                        )
                if do_debug_save:
                    videos, latents, log_probs, kls, debug_metrics = out
                    out_dir = ALIGN_FLOW_LOGS_DIR
                    os.makedirs(out_dir, exist_ok=True)
                    fps = 24
                    for batch_idx in range(videos.shape[0]):
                        vid = videos[batch_idx].float().cpu()
                        vid_np = vid.numpy()
                        if vid_np.min() < 0:
                            vid_np = (vid_np + 1.0) / 2.0
                        vid_np = np.clip(vid_np, 0.0, 1.0)
                        vid_np = (np.round(vid_np * 255.0)).astype(np.uint8)
                        frames = [vid_np[t].transpose(1, 2, 0) for t in range(vid_np.shape[0])]
                        imageio.mimsave(
                            os.path.join(out_dir, f"debug_batch{i}_j{j}_sample_{batch_idx}.mp4"),
                            frames,
                            fps=fps,
                        )
                    metrics_path = os.path.join(out_dir, "debug_metrics.txt")
                    mode = "w" if (i == 0 and j == 0) else "a"
                    with open(metrics_path, mode) as f:
                        f.write(f"=== batch {i} (j={j}) ===\n")
                        pred_per = debug_metrics.pop("debug_model_pred_per_step", [])
                        latents_per = debug_metrics.pop("debug_intermediate_latents_per_step", [])
                        var_noise_per = debug_metrics.pop("debug_variance_noise_sum_per_step", [])
                        for k, v in debug_metrics.items():
                            f.write(f"{k}: {v}\n")
                        for step_i in range(max(len(pred_per), len(latents_per), len(var_noise_per))):
                            if step_i < len(pred_per):
                                f.write(f"step_{step_i} sum_model_pred_fp64: {pred_per[step_i]}\n")
                            if step_i < len(latents_per):
                                f.write(f"step_{step_i} sum_intermediate_latents_fp64: {latents_per[step_i]}\n")
                            if step_i < len(var_noise_per):
                                f.write(f"step_{step_i} sum_variance_noise_fp64: {var_noise_per[step_i]}\n")
                    if i == 0 and j == 0:
                        logger.info("RL_METRIC: Debug region saving flow_logs and metrics per batch; will log rewards/advantages then stop.")
                videos, latents, log_probs, kls = out[:4]
                # Save decoded videos to safetensors for FastVideo reward verification (each batch, each rank)
                if epoch >= 2 and j == 0:
                    os.makedirs(DECODED_VIDEOS_DIR, exist_ok=True)
                    path = os.path.join(DECODED_VIDEOS_DIR, f"batch_{i}_rank_{accelerator.process_index}.safetensors")
                    save_safetensors({"decoded_videos": videos.detach().cpu()}, path)
                # endregion

                latents = torch.stack(
                    latents, dim=1
                )  # (batch_size, num_steps + 1, 16, 96, 96)
                log_probs = torch.stack(log_probs, dim=1)  # shape after stack (batch_size, num_steps)
                kls = torch.stack(kls, dim=1) 
                kl = kls.detach()

                timesteps = pipeline.scheduler.timesteps.repeat(
                    config.sample.train_batch_size, 1
                )  # (batch_size, num_steps)

                # compute rewards asynchronously
                rewards = executor.submit(reward_fn, videos, prompts, prompt_metadata, only_strict=True)
                # images b, 3, 512, 512
                # yield to to make sure reward computation starts
                time.sleep(0)
                
                samples.append(
                    {
                        "prompt_ids": prompt_ids,   # b, 77
                        "prompt_embeds": prompt_embeds,    # b, 205, 4096
                        "negative_prompt_embeds": sample_neg_prompt_embeds,
                        "timesteps": timesteps,
                        "latents": latents[
                            :, :-1
                        ],  # each entry is the latent before timestep t.   b, 11, 16, 64, 64
                        "next_latents": latents[
                            :, 1:
                        ],  # each entry is the latent after timestep t
                        "log_probs": log_probs,   # b, t + 1
                        "kl": kl,
                        "rewards": rewards,
                    }
                )

        if epoch < 2:
            continue
        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            # accelerator.print(reward_metadata)
            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        if epoch % 10 == 0 and accelerator.is_main_process:
            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(videos))
                sample_indices = random.sample(range(len(videos)), num_samples)

                for idx, i in enumerate(sample_indices):
                    video = videos[i]
                    frames = [img for img in video.cpu().numpy().transpose(0, 2, 3, 1)]
                    frames = [(frame * 255).astype(np.uint8) for frame in frames]
                    imageio.mimsave(os.path.join(tmpdir, f"{idx}.mp4"), frames, fps=8, codec="libx264", format='FFMPEG')         

                sampled_prompts = [prompts[i] for i in sample_indices]
                sampled_rewards = [rewards['avg'][i] for i in sample_indices]

                accelerator.log(
                    {
                        "video": [
                            wandb.Video(
                                os.path.join(tmpdir, f"{idx}.mp4"),
                                caption=f"{prompt:.100} | avg: {avg_reward:.2f}",
                                format="mp4",
                                fps=8 
                            )
                            for idx, (prompt, avg_reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                        ],
                    },
                    step=global_step,
                )
        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(-1) - config.sample.kl_reward*samples["kl"]
        # gather rewards across processes
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}
        # log rewards and images
        accelerator.log(
            {
                "epoch": epoch,
                **{f"reward_{key}": value.mean() for key, value in gathered_rewards.items() if '_strict_accuracy' not in key and '_accuracy' not in key},
                "kl": samples["kl"].mean().cpu().numpy(),
                "kl_abs": samples["kl"].abs().mean().cpu().numpy()
            },
            step=global_step,
        )

        # per-prompt mean/std tracking
        if config.per_prompt_stat_tracking:
            # gather the prompts across processes
            # print(f"[Rank {accelerator.process_index}] prompt_ids shape before gather: {samples['prompt_ids'].shape}")
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = pipeline.tokenizer.batch_decode(
                prompt_ids, skip_special_tokens=True
            )

            advantages = stat_tracker.update(prompts, gathered_rewards['avg'])
            if accelerator.is_local_main_process:
                print("len(prompts)", len(prompts))
                print("len unique prompts", len(set(prompts)))

            group_size, trained_prompt_num = stat_tracker.get_stats()

            zero_std_ratio = calculate_zero_std_ratio(prompts, gathered_rewards)

            accelerator.log(
                {
                    "group_size": group_size,
                    "trained_prompt_num": trained_prompt_num,
                    "zero_std_ratio": zero_std_ratio,
                },
                step=global_step,
            )
            stat_tracker.clear()
        else:
            advantages = (gathered_rewards['avg'] - gathered_rewards['avg'].mean()) / (gathered_rewards['avg'].std() + 1e-4)

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        advantages = torch.as_tensor(advantages)

        # myregion debug: log reward, advantage, prompt per sample (64 lines) to flow_logs/debug_metrics.txt then stop
        global _debug_rewards_advantages_logged
        if not _debug_rewards_advantages_logged and accelerator.is_main_process:
            out_dir = ALIGN_FLOW_LOGS_DIR
            os.makedirs(out_dir, exist_ok=True)
            rewards_arr = np.asarray(gathered_rewards['avg'])
            advantages_arr = advantages.cpu().numpy()
            # Take first element per row if broadcast to (64, 20); otherwise use ravel
            if rewards_arr.ndim > 1:
                rewards_1d = rewards_arr[:, 0].ravel()
                advantages_1d = advantages_arr[:, 0].ravel()
            else:
                rewards_1d = rewards_arr.ravel()
                advantages_1d = advantages_arr.ravel()
            n = len(rewards_1d)
            # Reorder to match prompts.txt: batch 0 (all ranks), then batch 1 (all ranks)
            local_N = (
                config.sample.num_batches_per_epoch
                * config.sample.sample_time_per_prompt
                * config.sample.train_batch_size
            )
            batch_size = config.sample.train_batch_size
            reorder_idx = np.array(
                [
                    r * local_N + b * (config.sample.sample_time_per_prompt * batch_size) + i
                    for b in range(config.sample.num_batches_per_epoch)
                    for r in range(accelerator.num_processes)
                    for i in range(batch_size)
                ]
            )
            rewards_1d = rewards_1d[reorder_idx]
            advantages_1d = advantages_1d[reorder_idx]
            prompts_ordered = [prompts[j] if j < len(prompts) else "" for j in reorder_idx]
            with open(os.path.join(out_dir, "debug_metrics.txt"), "a") as f:
                f.write("\n=== reward, advantage, prompt (one per line, same order as prompts.txt) ===\n")
                for i in range(n):
                    f.write(f"{float(rewards_1d[i])}, {float(advantages_1d[i])}, {prompts_ordered[i]}\n")
            logger.info("RL_METRIC: Debug region appended full rewards/advantages to flow_logs/debug_metrics.txt; stopping.")
            _debug_rewards_advantages_logged = True
            
        # endregion


        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            .to(accelerator.device)
        )

        if accelerator.is_local_main_process:
            print("advantages: ", samples["advantages"].abs().mean())
            print("kl: ", samples["kl"].mean())

        del samples["rewards"]
        del samples["prompt_ids"]

        # Get the mask for samples where all advantages are zero across the time dimension
        mask = (samples["advantages"].abs().sum(dim=1) != 0)
        
        # If the number of True values in mask is not divisible by config.sample.num_batches_per_epoch,
        # randomly change some False values to True to make it divisible
        num_batches = config.sample.num_batches_per_epoch * config.sample.sample_time_per_prompt
        true_count = mask.sum()
        if true_count == 0:
            print("advantages: ", samples["advantages"].abs().mean())
            print("mask.sum() == 0. revise in this rank")
            samples["advantages"] = samples["advantages"] + 1e-6
            print("after revise advantages: ", samples["advantages"].abs().mean())
            mask = (samples["advantages"].abs().sum(dim=1) != 0)

        if true_count % num_batches != 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches - (true_count % num_batches)
            if len(false_indices) >= num_to_change:
                random_indices = torch.randperm(len(false_indices))[:num_to_change]
                mask[false_indices[random_indices]] = True
        accelerator.log(
            {
                "actual_batch_size": mask.sum().item()// (config.sample.num_batches_per_epoch * config.sample.sample_time_per_prompt),
            },
            step=global_step,
        )
        # Filter out samples where the entire time dimension of advantages is zero
        samples = {k: v[mask] for k, v in samples.items()}

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert num_timesteps == config.sample.num_steps
        logger.info(f"total_batch_size: {total_batch_size}")

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples along batch dimension
            g = torch.Generator(device='cuda').manual_seed(50)
            # perm = torch.randperm(total_batch_size, device=accelerator.device, generator=g)
            perm = torch.randperm(total_batch_size, device='cuda', generator=g)
            if accelerator.is_main_process:
                print(f"shuffle_training_batches perm: {perm}")
            # perm = torch.arange(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            # shuffle along time dimension independently for each sample
            perms = torch.stack(
                [
                    # torch.randperm(num_timesteps, device=accelerator.device)
                    torch.arange(num_timesteps, device=accelerator.device)
                    for _ in range(total_batch_size)
                ]
            )
            for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                samples[key] = samples[key][
                    torch.arange(total_batch_size, device=accelerator.device)[:, None],
                    perms,
                ]

            micoe_batch = total_batch_size // (config.sample.num_batches_per_epoch * config.sample.sample_time_per_prompt)

            samples_batched = {
                k: v.reshape(-1, micoe_batch, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]
            # myregion debug
            for batch_index, sample in enumerate(samples_batched):
                mean_adv = sample["advantages"].mean(dim=1)  # [micoe_batch] per rank
                gathered_mean_adv = accelerator.gather(mean_adv).cpu().tolist()
                if accelerator.is_main_process:
                    print(f"==== training batch {batch_index} ====")
                    print(f"mean(advantages) (global batch size {len(gathered_mean_adv)}): {gathered_mean_adv}")
            # endregion 


            # train
            pipeline.transformer.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                if config.train.cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds = sample["prompt_embeds"]
                    negative_embeds = train_neg_prompt_embeds[:len(sample["prompt_embeds"])]
                else:
                    embeds = sample["prompt_embeds"]
                    negative_embeds = None

                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(transformer):
                        with autocast():
                            prev_sample, log_prob, prev_sample_mean, std_dev_t, dt = compute_log_prob(transformer, pipeline, sample, j, embeds, negative_embeds, config)
                            if config.train.beta > 0:
                                with torch.no_grad():
                                    with transformer.module.disable_adapter():
                                        prev_sample_ref, log_prob_ref, prev_sample_mean_ref, std_dev_t_ref, dt_ref = compute_log_prob(transformer, pipeline, sample, j, embeds, negative_embeds, config)

                        # grpo logic
                        advantages = torch.clamp(
                            sample["advantages"][:, j],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        if config.train.beta > 0:
                            kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2,3), keepdim=True) / (2 * (std_dev_t * dt_ref) ** 2)
                            kl_loss = torch.mean(kl_loss)
                            loss = policy_loss + config.train.beta * kl_loss
                        else:
                            loss = policy_loss

                        info["approx_kl"].append(
                            0.5
                            * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (
                                    torch.abs(ratio - 1.0) > config.train.clip_range
                                ).float()
                            )
                        )
                        info["policy_loss"].append(policy_loss)
                        if config.train.beta > 0:
                            info["kl_loss"].append(kl_loss)

                        info["loss"].append(loss)

                        # backward pass
                        accelerator.backward(loss)
                        
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                transformer.parameters(), config.train.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        # assert (j == train_timesteps[-1]) and (
                        #     i + 1
                        # ) % config.train.gradient_accumulation_steps == 0
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        accelerator.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)
                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)
            # make sure we did an optimization step at the end of the inner epoch
            # assert accelerator.sync_gradients
        raise KeyboardInterrupt("Debug stop after training one step ends.")
if __name__ == "__main__":
    app.run(main)


