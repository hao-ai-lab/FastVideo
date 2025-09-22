# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import pytest
import torch
from torch.distributed.tensor import DTensor
from torch.testing import assert_close
from transformers import AutoConfig, AutoTokenizer, UMT5EncoderModel

from fastvideo.wan.modules.tokenizers import HuggingfaceTokenizer
from fastvideo.wan.modules.t5 import umt5_xxl

from fastvideo.configs.pipelines import PipelineConfig
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import TextEncoderLoader
from fastvideo.utils import maybe_download_model, PRECISION_TO_TYPE
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.configs.models.encoders import T5Config

logger = init_logger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29503"

BASE_MODEL_PATH = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
MODEL_PATH = maybe_download_model(BASE_MODEL_PATH,
                                  local_dir=os.path.join(
                                      'data', BASE_MODEL_PATH))
TEXT_ENCODER_PATH = os.path.join(MODEL_PATH, "text_encoder")
TOKENIZER_PATH = os.path.join(MODEL_PATH, "tokenizer")


@pytest.mark.usefixtures("distributed_setup")
def test_t5_encoder():
    # Initialize the two model implementations
    hf_config = AutoConfig.from_pretrained(TEXT_ENCODER_PATH)
    print(hf_config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision_str = "fp32"
    precision = PRECISION_TO_TYPE[precision_str]
    # model1 = UMT5EncoderModel.from_pretrained(TEXT_ENCODER_PATH).to(
    #     precision).to(device).eval()
    # tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model1 = umt5_xxl(
        encoder_only=True,
        return_tokenizer=False,
        dtype=torch.float32,
        device=device,
    ).eval().requires_grad_(False)
    model1.load_state_dict(
        torch.load("/mnt/weka/home/hao.zhang/wei/Self-Forcing-clean/wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
                    map_location='cpu', weights_only=False)
    )

    tokenizer1 = HuggingfaceTokenizer(
        name="/mnt/weka/home/hao.zhang/wei/Self-Forcing-clean/wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/", seq_len=512, clean='whitespace')


    args = FastVideoArgs(model_path=TEXT_ENCODER_PATH,
                        pipeline_config=PipelineConfig(text_encoder_configs=(T5Config(),),
                        text_encoder_precisions=(precision_str,)),
                        pin_cpu_memory=False)
    loader = TextEncoderLoader()
    model2 = loader.load(TEXT_ENCODER_PATH, args)
    model2 = model2.to(dtype=torch.bfloat16).to(precision)
    model2.eval()
    tokenizer2 = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    # Sanity check weights between the two models
    logger.info("Comparing model weights for sanity check...")
    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())

    # Check number of parameters
    logger.info("Model1 has %s parameters", len(params1))
    logger.info("Model2 has %s parameters", len(params2))

    model1_weight_sum = sum(p.float().sum().item() for p in model1.parameters())
    model2_weight_sum = sum(p.float().sum().item() for p in model2.parameters())
    logger.info("Model1 weight sum: %s", model1_weight_sum)
    logger.info("Model2 weight sum: %s", model2_weight_sum)

    # weight_diffs = []
    # # check if embed_tokens are the same
    weights = ["encoder.block.{}.layer.0.layer_norm.weight", "encoder.block.{}.layer.0.SelfAttention.relative_attention_bias.weight", \
               "encoder.block.{}.layer.0.SelfAttention.o.weight", "encoder.block.{}.layer.1.DenseReluDense.wi_0.weight", "encoder.block.{}.layer.1.DenseReluDense.wi_1.weight",\
                "encoder.block.{}.layer.1.DenseReluDense.wo.weight", \
                "encoder.block.{}.layer.1.layer_norm.weight", "encoder.final_layer_norm.weight"]
    
    for idx in range(hf_config.num_hidden_layers):
        for w in weights:
            # name1 = w.format(idx)
            name2 = w.format(idx)
            # p1 = params1[name1]
            p2 = params2[name2]
            p2 = (p2.to_local() if isinstance(p2, DTensor) else p2).to(device)
            # assert_close(p1, p2, atol=1e-4, rtol=1e-4)
    

    # Test with some sample prompts
    # prompts = [
    #     "Once upon a time", "The quick brown fox jumps over",
    #     "In a galaxy far, far away"
    # ]
    prompts = [
        "A vibrant scene of Kenyan golfers at a lush green golf course on a sunny day. The golfers, dressed in casual yet stylish attire, are teeing off with animated expressions, showcasing their enthusiasm for the game. Rolling hills and pristine greens stretch out behind them, creating a picturesque backdrop. In the foreground, a golf buggy and a caddy stand ready, adding to the serene atmosphere. The camera captures the action from a mid-shot angle, focusing on the golfers' dynamic motions as they swing their clubs."
    ]

    logger.info("Testing T5 encoder with sample prompts")

    with torch.no_grad():
        for prompt in prompts:
            logger.info("Testing prompt: %s", prompt)

            # Tokenize the prompt
            tokens1, mask = tokenizer1(prompt, return_mask=True, add_special_tokens=True)
            tokens2 = tokenizer2(prompt,
                               padding="max_length",
                               max_length=512,
                               truncation=True,
                               return_tensors="pt").to(device)

            # Get outputs from HuggingFace implementation
            # filter out padding input_ids
            # tokens.input_ids = tokens.input_ids[tokens.attention_mask==1]
            # tokens.attention_mask = tokens.attention_mask[tokens.attention_mask==1]
            outputs1 = model1(tokens1.to(device),
                              mask.to(device))
            print("--------------------------------")
            logger.info("Testing model2")

            # Get outputs from our implementation
            with set_forward_context(current_timestep=0, attn_metadata=None):
                outputs2 = model2(
                    input_ids=tokens2.input_ids,
                    attention_mask=tokens2.attention_mask,
                ).last_hidden_state

            # Compare last hidden states
            last_hidden_state1 = outputs1[mask == 1]
            last_hidden_state2 = outputs2[tokens2.attention_mask == 1]
            logger.info("last_hidden_state1 sum: %s", last_hidden_state1.float().sum())
            logger.info("last_hidden_state2 sum: %s", last_hidden_state2.float().sum())

            assert last_hidden_state1.shape == last_hidden_state2.shape, \
                f"Hidden state shapes don't match: {last_hidden_state1.shape} vs {last_hidden_state2.shape}"

            max_diff_hidden = torch.max(
                torch.abs(last_hidden_state1 - last_hidden_state2))
            mean_diff_hidden = torch.mean(
                torch.abs(last_hidden_state1 - last_hidden_state2))

            logger.info("Maximum difference in last hidden states: %s",
                        max_diff_hidden.item())
            logger.info("Mean difference in last hidden states: %s",
                        mean_diff_hidden.item())
            logger.info("Max memory allocated: %s GB", torch.cuda.max_memory_allocated() / 1024**3)
            # Check if outputs are similar (allowing for small numerical differences)
            assert mean_diff_hidden < 1e-4, \
                f"Hidden states differ significantly: mean diff = {mean_diff_hidden.item()}"
            assert max_diff_hidden < 1e-4, \
                f"Hidden states differ significantly: max diff = {max_diff_hidden.item()}"