from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
import hashlib
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator, load_checkpoint_and_dispatch, init_empty_weights
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from flow_grpo.fsdp_utils import save_fsdp_checkpoint, register_optimizer_offload_hooks
# diffusers
from diffusers.utils.torch_utils import is_compiled_module

# bagel
from flow_grpo.bagel.data.data_utils import add_special_tokens
from flow_grpo.bagel.data.transforms import ImageTransform
from flow_grpo.bagel.modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from flow_grpo.bagel.modeling.qwen2 import Qwen2Tokenizer
from flow_grpo.bagel.modeling.autoencoder import load_ae
from flow_grpo.bagel.modeling.bagel.qwen2_navit import NaiveCache
from flow_grpo.bagel.inferencer import InterleaveInferencer

import numpy as np
import flow_grpo.prompts
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from huggingface_hub import snapshot_download


tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)

class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]
            # 限制测试集样本数量,bagel推理比较慢,只取前512个样本
            if split == 'test':
                self.prompts = self.prompts[:512]
        
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
        self.batch_size = batch_size  # Batch size per replica
        self.k = k                    # Number of repetitions per sample
        self.num_replicas = num_replicas  # Total number of replicas
        self.rank = rank              # Current replica rank
        self.seed = seed              # Random seed for synchronization
        
        # Compute the number of unique samples needed per iteration
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # Number of unique samples
        self.epoch = 0

    def __iter__(self):
        while True:
            # Generate a deterministic random sequence to ensure all replicas are synchronized
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            # Randomly select m unique samples
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            
            # Repeat each sample k times to generate n*b total samples
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # Shuffle to ensure uniform distribution
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            
            # Split samples to each replica
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            
            # Return current replica's sample indices
            yield per_card_samples[self.rank]
    
    def set_epoch(self, epoch):
        self.epoch = epoch  # Used to synchronize random state across epochs

def calculate_zero_std_ratio(prompts, gathered_rewards):
    """
    Calculate the proportion of unique prompts whose reward standard deviation is zero.
    
    Args:
        prompts: List of prompts.
        gathered_rewards: Dictionary containing rewards, must include the key 'ori_avg'.
        
    Returns:
        zero_std_ratio: Proportion of prompts with zero standard deviation.
        prompt_std_devs: Mean standard deviation across all unique prompts.
    """
    # Convert prompt list to NumPy array
    prompt_array = np.array(prompts)
    
    # Get unique prompts and their group information
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, 
        return_inverse=True,
        return_counts=True
    )
    
    # Group rewards for each prompt
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    
    # Calculate standard deviation for each group
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    
    # Calculate the ratio of zero standard deviation
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    
    return zero_std_ratio, prompt_std_devs.mean()

def create_generators(prompts, base_seed):
    generators = []
    for prompt in prompts:
        # Use a stable hash (SHA256), then convert it to an integer seed
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')  # Take the first 4 bytes as part of the seed
        seed = (base_seed + prompt_hash_int) % (2**31) # Ensure the number is within a valid range
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators


def eval(inferencer, inference_hyper, test_dataloader, tokenizer, config, accelerator, global_step, eval_reward_fn, executor, autocast, ema, transformer_trainable_parameters, prefix=''):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    all_rewards = defaultdict(list)
    for test_batch in tqdm(
            test_dataloader,
            desc="Eval: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
        prompts, prompt_metadata = test_batch
        images = []
        with autocast():
            for idx, prompt in enumerate(prompts):
                with torch.no_grad():
                    output_dict = inferencer(
                        text=prompt, 
                        noise_level=0, 
                        grpo_config=config, 
                        accelerator=accelerator, 
                        cfg_text_scale=config.sample.eval_guidance_scale, 
                        num_timesteps=config.sample.eval_num_steps,**inference_hyper)
                images.append(output_dict['image'])
            images = torch.stack(images, dim=0)  # shape after stack (batch_size, 3, resolution, resolution)
        rewards = executor.submit(eval_reward_fn, images, prompts, prompt_metadata, only_strict=False)
        # yield to to make sure reward computation starts
        time.sleep(0)
        rewards, reward_metadata = rewards.result()

        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)
    
    last_batch_images_gather = accelerator.gather(torch.as_tensor(images, device=accelerator.device)).cpu().numpy()
    last_batch_prompt_ids = tokenizer(
        prompts,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(accelerator.device)
    last_batch_prompt_ids_gather = accelerator.gather(last_batch_prompt_ids).cpu().numpy()
    last_batch_prompts_gather = prompts = tokenizer.batch_decode(last_batch_prompt_ids_gather, skip_special_tokens=True)
    last_batch_rewards_gather = {}
    for key, value in rewards.items():
        last_batch_rewards_gather[key] = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    if accelerator.is_main_process:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(25, len(last_batch_images_gather))
            # sample_indices = random.sample(range(len(images)), num_samples)
            sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                image = last_batch_images_gather[index]
                pil = Image.fromarray(
                    (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))  # 使用新的索引
            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            sampled_rewards = [{k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather} for index in sample_indices]
            wandb.log(
                {
                    f"{prefix} eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | " + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                    ],
                    **{f"{prefix}_eval_reward_{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()},
                },
                step=global_step,
            )
    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        # log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # 训练的bs=1，每次训练sample_sde_window_size个timestep
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * config.sample.train_batch_size*config.sample.sde_window_size
    )
    # accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1
    accelerator.state.fsdp_plugin.activation_checkpointing = config.activation_checkpointing
    accelerator.state.fsdp_plugin.transformer_cls_names_to_wrap = ['Qwen2MoTDecoderLayer']

    if accelerator.is_main_process:
        wandb.init(
            project="flow_grpo",
            name=config.run_name,
            config=config.to_dict(),
        )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    model_path = config.pretrained.model  # Download from https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT
    if not os.path.exists(model_path):
        model_local_dir = snapshot_download(repo_id=model_path)
    else:
        model_local_dir = model_path


    # LLM config preparing
    llm_config = Qwen2Config.from_json_file(os.path.join(model_local_dir, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    # # ViT config preparing
    # vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_local_dir, "vit_config.json"))
    # vit_config.rope = False
    # vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    # VAE loading
    vae_model, vae_config = load_ae(local_path=os.path.join(model_local_dir, "ae.safetensors"))

    # Bagel config preparing
    bagel_config = BagelConfig(
        visual_gen=True,
        visual_und=False,
        llm_config=llm_config, 
        vit_config=None,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        # vit_model      = SiglipVisionModel(vit_config)
        model          = Bagel(language_model, None, bagel_config)
        # model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    # Tokenizer Preparing
    tokenizer = Qwen2Tokenizer.from_pretrained(model_local_dir)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # Image Transform Preparing
    # TODO: change to original size
    # vae_transform = ImageTransform(1024, 512, 16)
    # vit_transform = ImageTransform(980, 224, 14)

    vae_transform = ImageTransform(512, 256, 8)
    vit_transform = ImageTransform(490, 112, 7)

    # Thanks @onion-liu: https://github.com/ByteDance-Seed/Bagel/pull/8
    print('***************accelerator.process_index**********', accelerator.process_index)
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(model_local_dir, "ema.safetensors"),
        device_map={"": f"cuda:{accelerator.local_process_index}"},  # 关键修改：映射所有层到当前设备
        offload_buffers=False,            # 关闭缓冲区卸载（确保完全加载）
        dtype=inference_dtype,
        force_hooks=True,
        offload_folder="/tmp/offload"
    )
    model = model.eval()

    if config.train.beta>0:
        language_model_ref = Qwen2ForCausalLM(llm_config)
        language_model_ref.load_state_dict(model.language_model.state_dict())
        language_model_ref.to(device=f"cuda:{accelerator.local_process_index}", dtype=inference_dtype)
        language_model_ref.eval()
        language_model_ref.requires_grad_(False)
    
    # freeze parameters of models to save more memory
    vae_model.requires_grad_(False)
    model.requires_grad_(False)
    # model.language_model.requires_grad_(False)
    # model.time_embedder.requires_grad_(False)
    # model.vae2llm.requires_grad_(False)
    # model.llm2vae.requires_grad_(False)
    # model.vit_model.requires_grad_(False)
    # model.connector.requires_grad_(False)

    inference_hyper=dict(
        cfg_img_scale=1.0,
        cfg_interval=[0, 1.0],
        timestep_shift=config.train.timestep_shift,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(config.resolution, config.resolution),
    )

    inferencer = InterleaveInferencer(
        model=model, 
        vae_model=vae_model, 
        tokenizer=tokenizer, 
        vae_transform=vae_transform, 
        vit_transform=vit_transform, 
        new_token_ids=new_token_ids
    )


    # Move transformer, vae and text_encoder to device and cast to inference_dtype
    vae_model.to(accelerator.device, dtype=inference_dtype)
    # vit_model.to(accelerator.device, dtype=inference_dtype)
    model.to(accelerator.device, dtype=inference_dtype)
    # if accelerator.is_main_process:
    #     print(model)
    if config.use_lora:
        # Set correct lora layers
        target_modules = [
            "self_attn.q_proj_moe_gen",
            "self_attn.k_proj_moe_gen",
            "self_attn.v_proj_moe_gen",
            "self_attn.o_proj_moe_gen",
            "mlp_moe_gen.gate_proj",
            "mlp_moe_gen.up_proj",
            "mlp_moe_gen.down_proj",
        ]
        transformer_lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        # 保持进入prepare之前lora和base的模型参数dype一致
        model.language_model = get_peft_model(model.language_model, transformer_lora_config)
        for name, param in model.language_model.named_parameters():
            if 'lora' in name:
                param.data = param.data.to(dtype=inference_dtype)
    else:
        # for循环给参数名里面有moe的设置requires_grad=True
        for name, param in model.language_model.named_parameters():
            if 'moe_gen' in name:
                param.requires_grad = True

    transformer = model.language_model
    # vit_model = model.vit_model
    transformer.config.use_cache = False
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    ema=None

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )
    if config.fsdp_optimizer_offload:
        register_optimizer_offload_hooks(optimizer)
    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, 'train')
        test_dataset = TextPromptDataset(config.dataset, 'test')

        # Create an infinite-loop DataLoader
        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        # Create a DataLoader; note that shuffling is not needed here because it’s controlled by the Sampler.
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=TextPromptDataset.collate_fn,
            # persistent_workers=True
        )

        # Create a regular DataLoader
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

        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=GenevalPromptDataset.collate_fn,
            # persistent_workers=True
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=GenevalPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )
    else:
        raise NotImplementedError("Only general_ocr is supported with dataset")
    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    # initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    # accelerator.state.fsdp_plugin.ignored_modules=[model.language_model.model.embed_tokens] 
    # prepare prompt and reward fn
    reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
    eval_reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)

    if config.train.beta>0:
        transformer, language_model_ref, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
            transformer, 
            language_model_ref,
            optimizer, 
            train_dataloader, 
            test_dataloader, 
        )
        model.language_model_ref = language_model_ref
    else:
        transformer, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
            transformer, 
            optimizer, 
            train_dataloader, 
            test_dataloader, 
        )
    model.language_model = transformer
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

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
    else:
        first_epoch = 0
    train_iter = iter(train_dataloader)
    global_step = 0

    for epoch in range(first_epoch, config.num_epochs):
        #################### EVAL ####################
        transformer.eval()
        if not config.debug and epoch % config.save_freq == 0 and epoch > 0:
            save_fsdp_checkpoint(config.save_dir, transformer, global_step, accelerator.process_index)
        if not config.debug and epoch % config.eval_freq == 0 and epoch > 0:
            eval(inferencer, inference_hyper, test_dataloader, tokenizer, config, accelerator, global_step, eval_reward_fn, executor, autocast, ema, transformer_trainable_parameters)
        #################### SAMPLING ####################
        transformer.eval()
        samples = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)

            prompt_ids = tokenizer(
                prompts,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)

            # sample
            generators = create_generators(prompts, base_seed=42)
            with autocast():
                images=[]
                latents=[]
                log_probs=[]
                timesteps=[]
                for idx, prompt in enumerate(prompts):
                    if config.sample.same_latent:
                        generator = generators[idx:idx+1]
                    else:
                        generator = None
                    with torch.no_grad():
                        output_dict = inferencer(
                            text=prompt,
                            noise_level=config.sample.noise_level,
                            grpo_config=config,
                            accelerator=accelerator,
                            num_timesteps=config.sample.num_steps,
                            # TODO: check it 
                            cfg_text_scale=config.sample.guidance_scale,
                            generators=generator,
                            **inference_hyper)
                    images.append(output_dict['image'])
                    latents.append(output_dict['all_latents'])
                    log_probs.append(output_dict['all_log_probs'])
                    timesteps.append(output_dict['timesteps'])
            stacked_inner_latents = [torch.stack(inner_list, dim=0) for inner_list in latents]
            latents = torch.stack(stacked_inner_latents, dim=0) # (batch_size, num_steps + 1, 4096, 64)
            stacked_inner_log_probs = [torch.stack(inner_list, dim=0) for inner_list in log_probs]
            log_probs = torch.stack(stacked_inner_log_probs, dim=0)  # shape after stack (batch_size, num_steps)
            timesteps = torch.stack(timesteps, dim=0)  # shape after stack (batch_size, num_steps)
            images = torch.stack(images, dim=0)  # shape after stack (batch_size, 3, resolution, resolution)

            # compute rewards asynchronously
            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            # yield to to make sure reward computation starts
            time.sleep(0)

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "timesteps": timesteps,
                    "latents": latents[
                        :, :-1
                    ],  # each entry is the latent before timestep t
                    "prev_latents": latents[
                        :, 1:
                    ],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "rewards": rewards,
                }
            )

        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
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

        if epoch % 5 == 0 and accelerator.is_main_process:
            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(images))
                sample_indices = random.sample(range(len(images)), num_samples)

                for idx, i in enumerate(sample_indices):
                    image = images[i]
                    pil = Image.fromarray(
                        (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))  # 使用新的索引

                sampled_prompts = [prompts[i] for i in sample_indices]
                sampled_rewards = [rewards['avg'][i] for i in sample_indices]

                wandb.log(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompt:.100} | avg: {avg_reward:.2f}",
                            )
                            for idx, (prompt, avg_reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                        ],
                    },
                    step=global_step,
                )
        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(-1)
        # gather rewards across processes
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}
        # log rewards and images
        if accelerator.is_main_process:
            wandb.log(
                {
                    "epoch": epoch,
                    **{f"reward_{key}": value.mean() for key, value in gathered_rewards.items() if '_strict_accuracy' not in key and '_accuracy' not in key},
                },
                step=global_step,
            )

        # per-prompt mean/std tracking
        if config.per_prompt_stat_tracking:
            # gather the prompts across processes
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
            advantages = stat_tracker.update(prompts, gathered_rewards['avg'])

            group_size, trained_prompt_num = stat_tracker.get_stats()

            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts, gathered_rewards)

            if accelerator.is_main_process:
                wandb.log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                    },
                    step=global_step,
                )
            stat_tracker.clear()
        else:
            advantages = (gathered_rewards['avg'] - gathered_rewards['avg'].mean()) / (gathered_rewards['avg'].std() + 1e-4)

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        advantages = torch.as_tensor(advantages)
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            .to(accelerator.device)
        )

        del samples["rewards"]

        total_batch_size, num_timesteps = samples["timesteps"].shape

        #################### TRAINING ####################
        transformer.train()
        for inner_epoch in range(config.train.num_inner_epochs):
            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, total_batch_size//config.sample.num_batches_per_epoch, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]
            # train

            transformer.train()
            transformer.module.training = False
            transformer.module.model.training = False
            if config.use_lora:
                transformer.module.model.model.training = False
                for layer in transformer.module.model.model.layers:
                    layer.module.training = False
                    layer.module.self_attn.training = False
            else:
                for layer in transformer.module.model.layers:
                    layer.module.training = False
                    layer.module.self_attn.training = False
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                sample["dtimesteps"] = torch.cat([sample["timesteps"][:,:-1] - sample["timesteps"][:, 1:], sample["timesteps"][:,-1].unsqueeze(1)], dim=1)
                bs = sample["timesteps"].shape[0]
                prompts = tokenizer.batch_decode(sample['prompt_ids'], skip_special_tokens=True)
                for j in tqdm(
                    range(bs),
                    desc="Batch Size",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):  
                    cur_sample = {k: v[j] for k, v in sample.items()}
                    
                    with autocast():
                        output_dict = inferencer(text=prompts[j], 
                                                 noise_level=config.sample.noise_level,
                                                 learn=True,
                                                 sample=cur_sample,
                                                 grpo_config=config,
                                                 accelerator=accelerator,
                                                 optimizer=optimizer,
                                                 transformer=transformer,
                                                 num_timesteps=config.sample.num_steps,
                                                 cfg_text_scale=config.sample.guidance_scale,
                                                 **inference_hyper)
                info["clipfrac"].append(
                    output_dict["clipfrac"]
                )
                info["clipfrac_gt_one"].append(
                    output_dict["clipfrac_gt_one"]
                )
                info["clipfrac_lt_one"].append(
                    output_dict["clipfrac_lt_one"]
                )
                # print('output_dict["clipfrac"]:', output_dict["clipfrac"])
                info["policy_loss"].append(output_dict["policy_loss"])
                info["kl_loss"].append(output_dict["kl_loss"])
                info["loss"].append(output_dict["loss"])

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                    info = accelerator.reduce(info, reduction="mean")
                    info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                    if accelerator.is_main_process:
                        wandb.log(info, step=global_step)
                    global_step += 1
                    info = defaultdict(list)

                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)

    if accelerator.is_main_process:
        wandb.finish()     


if __name__ == "__main__":
    app.run(main)
