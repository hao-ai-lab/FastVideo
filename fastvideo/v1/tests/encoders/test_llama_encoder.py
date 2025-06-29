# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import pytest
import torch
from transformers import AutoConfig

from fastvideo.models.hunyuan.text_encoder import (load_text_encoder,
                                                   load_tokenizer)
from fastvideo.v1.configs.pipelines import PipelineConfig
from fastvideo.v1.forward_context import set_forward_context
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.models.loader.component_loader import TextEncoderLoader
from fastvideo.v1.utils import maybe_download_model
from fastvideo.v1.configs.models.encoders import LlamaConfig

logger = init_logger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29503"

BASE_MODEL_PATH = "hunyuanvideo-community/HunyuanVideo"
MODEL_PATH = maybe_download_model(BASE_MODEL_PATH,
                                  local_dir=os.path.join(
                                      'data', BASE_MODEL_PATH))
TEXT_ENCODER_PATH = os.path.join(MODEL_PATH, "text_encoder")
TOKENIZER_PATH = os.path.join(MODEL_PATH, "tokenizer")


@pytest.mark.usefixtures("distributed_setup")
def test_llama_encoder():
    """
    Tests compatibility between two different implementations for loading text encoders:
    1. load_text_encoder from fastvideo.models.hunyuan.text_encoder
    2. TextEncoderLoader from fastvideo.v1.models.loader
    
    The test verifies that both implementations:
    - Load models with the same weights and parameters
    - Produce nearly identical outputs for the same input prompts
    """
    args = FastVideoArgs(model_path="meta-llama/Llama-2-7b-hf",
                         pipeline_config=PipelineConfig(text_encoder_configs=(LlamaConfig(),), text_encoder_precisions=("fp16",)))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize the two model implementations
    logger.info("Loading models from %s", args.model_path)
    hf_config = AutoConfig.from_pretrained(TEXT_ENCODER_PATH)
    print(hf_config)

    # Load our implementation using the loader from text_encoder/__init__.py
    model1, _ = load_text_encoder(text_encoder_type="llm",
                                  text_encoder_precision='fp16',
                                  text_encoder_path=TEXT_ENCODER_PATH,
                                  logger=logger,
                                  device=device)
    loader = TextEncoderLoader()
    device = torch.device("cuda:0")
    model2 = loader.load(TEXT_ENCODER_PATH, "", args)

    # Convert to float16 and move to device
    # model2 = model2.to(torch.float16)
    model2 = model2.to(device)
    model2.eval()

    # Sanity check weights between the two models
    logger.info("Comparing model weights for sanity check...")
    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())

    # Check number of parameters
    logger.info("Model1 has %d parameters", len(params1))
    logger.info("Model2 has %d parameters", len(params2))

    # Compare a few key parameters
    weight_diffs = []
    # check if embed_tokens are the same
    print(model1.embed_tokens.weight.shape, model2.embed_tokens.weight.shape)
    assert torch.allclose(model1.embed_tokens.weight,
                          model2.embed_tokens.weight)
    weights = [
        "layers.{}.input_layernorm.weight",
        "layers.{}.post_attention_layernorm.weight"
    ]
    # for (name1, param1), (name2, param2) in zip(
    #     sorted(params1.items()), sorted(params2.items())
    # ):
    for layer_idx in range(hf_config.num_hidden_layers):
        for w in weights:
            name1 = w.format(layer_idx)
            name2 = w.format(layer_idx)
            p1 = params1[name1]
            p2 = params2[name2]
            # print(type(p2))
            if "gate_up" in name2:
                # print("skipping gate_up")
                continue
            try:
                # logger.info(f"Parameter: {name1} vs {name2}")
                max_diff = torch.max(torch.abs(p1 - p2)).item()
                mean_diff = torch.mean(torch.abs(p1 - p2)).item()
                weight_diffs.append((name1, name2, max_diff, mean_diff))
                # logger.info(f"  Max diff: {max_diff}, Mean diff: {mean_diff}")
            except Exception as e:
                logger.info("Error comparing %s and %s: %s", name1, name2, e)

    tokenizer, _ = load_tokenizer(tokenizer_type="llm",
                                  tokenizer_path=TOKENIZER_PATH,
                                  logger=logger)

    # Test with some sample prompts
    prompts = [
        "Once upon a time",
        # "The quick brown fox jumps over",
        # "In a galaxy far, far away"
    ]

    logger.info("Testing LLaMA encoder with sample prompts")

    with torch.no_grad():
        for prompt in prompts:
            logger.info("Testing prompt: '%s'", prompt)

            # Tokenize the prompt
            tokens = tokenizer(prompt, return_tensors="pt").to(device)

            # Get outputs from our implementation
            # filter out padding input_ids
            # tokens.input_ids = tokens.input_ids[tokens.attention_mask==1]
            # tokens.attention_mask = tokens.attention_mask[tokens.attention_mask==1]
            outputs1 = model1(input_ids=tokens.input_ids,
                              output_hidden_states=True)
            print("--------------------------------")
            logger.info("Testing model2")

            # Get outputs from HuggingFace implementation
            with set_forward_context(current_timestep=0, attn_metadata=None):
                outputs2 = model2(input_ids=tokens.input_ids,
                                  attention_mask=tokens.attention_mask,
                                  output_hidden_states=True)

            # Compare last hidden states
            last_hidden_state1 = outputs1.last_hidden_state[
                tokens.attention_mask == 1]
            last_hidden_state2 = outputs2.last_hidden_state[
                tokens.attention_mask == 1]

            assert last_hidden_state1.shape == last_hidden_state2.shape, \
                f"Hidden state shapes don't match: {last_hidden_state1.shape} vs {last_hidden_state2.shape}"

            max_diff_hidden = torch.max(
                torch.abs(last_hidden_state1 - last_hidden_state2))
            mean_diff_hidden = torch.mean(
                torch.abs(last_hidden_state1 - last_hidden_state2))

            logger.info("Maximum difference in last hidden states: %f",
                        max_diff_hidden.item())
            logger.info("Mean difference in last hidden states: %f",
                        mean_diff_hidden.item())

            # Check if outputs are similar (allowing for small numerical differences)
            assert mean_diff_hidden < 1e-2, \
                f"Hidden states differ significantly: mean diff = {mean_diff_hidden.item()}"
            assert max_diff_hidden < 1e-1, \
                f"Hidden states differ significantly: max diff = {max_diff_hidden.item()}"
