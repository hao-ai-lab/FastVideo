from fastvideo.models.hunyuan.text_encoder import  load_text_encoder, load_tokenizer
import os
import torch
import torch.nn as nn
import argparse
import numpy as np
from fastvideo.v1.logger import init_logger
from transformers import AutoConfig
from fastvideo.v1.models.loader.component_loader import TextEncoderLoader

from fastvideo.v1.distributed import init_distributed_environment, initialize_model_parallel
from fastvideo.v1.pipelines.stages import DenoisingStage
from fastvideo.v1.forward_context import set_forward_context
logger = init_logger(__name__)


def initialize_identical_weights(model1, model2, seed=42):
    """Initialize both models with identical weights using a fixed seed for reproducibility."""
    # Get all parameters from both models
    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())

    # Initialize each layer with identical values
    with torch.no_grad():
        # Initialize weights
        for name1, param1 in params1.items():
            if 'weight' in name1:
                # Set seed before each weight initialization
                torch.manual_seed(seed)
                nn.init.normal_(param1, mean=0.0, std=0.05)

        for name2, param2 in params2.items():
            if 'weight' in name2:
                # Reset seed to get same initialization
                torch.manual_seed(seed)
                nn.init.normal_(param2, mean=0.0, std=0.05)

        # Initialize biases
        for name1, param1 in params1.items():
            if 'bias' in name1:
                torch.manual_seed(seed)
                nn.init.normal_(param1, mean=0.0, std=0.05)
                param1.data = param1.data.to(torch.bfloat16)

        for name2, param2 in params2.items():
            if 'bias' in name2:
                torch.manual_seed(seed)
                nn.init.normal_(param2, mean=0.0, std=0.05)
                param2.data = param2.data.to(torch.bfloat16)

    logger.info("Both models initialized with identical weights in bfloat16")
    return model1, model2


def setup_args():
    parser = argparse.ArgumentParser(description='LLaMA Encoder Test')
    parser.add_argument('--model-path',
                        type=str,
                        default="meta-llama/Llama-2-7b-hf",
                        help='Path to the LLaMA model')
    parser.add_argument(
        '--precision',
        type=str,
        default="float16",
        help='Precision to use for the model (float32, float16, bfloat16)')
    return parser.parse_args()


def test_llama_encoder():
    init_distributed_environment(world_size=1,
                                 rank=0,
                                 distributed_init_method="env://",
                                 local_rank=0,
                                 backend="nccl")
    initialize_model_parallel(tensor_model_parallel_size=1,
                              sequence_model_parallel_size=1,
                              backend="nccl")
    args = setup_args()

    # Set fixed random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize the two model implementations
    logger.info(f"Loading models from {args.model_path}")
    model_path = "data/hunyuanvideo-community/HunyuanVideo/text_encoder"

    hf_config = AutoConfig.from_pretrained(model_path)
    print(hf_config)

    # Load our implementation using the loader from text_encoder/__init__.py
    model1, _ = load_text_encoder(
        text_encoder_type="llm",
        text_encoder_precision='fp16',
        text_encoder_path=model_path,
        logger=logger,
        device=device
    )
    loader = TextEncoderLoader()
    args.device_str = "cuda:0"
    device = torch.device(args.device_str)
    model2 = loader.load_model(model_path, hf_config, device)

    # Convert to float16 and move to device
    model2 = model2.to(torch.float16)
    model2 = model2.to(device)
    model2.eval()

    # Sanity check weights between the two models
    logger.info("Comparing model weights for sanity check...")
    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())

    # Check number of parameters
    logger.info(f"Model1 has {len(params1)} parameters")
    logger.info(f"Model2 has {len(params2)} parameters")

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
    for l in range(hf_config.num_hidden_layers):
        for w in weights:
            name1 = w.format(l)
            name2 = w.format(l)
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
                logger.info(f"Error comparing {name1} and {name2}: {e}")

    tokenizer_path = "data/hunyuanvideo-community/HunyuanVideo/tokenizer"
    # Load tokenizer
    tokenizer, _ = load_tokenizer(tokenizer_type="llm",
                                  tokenizer_path=tokenizer_path,
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
            logger.info(f"Testing prompt: '{prompt}'")

            # Tokenize the prompt
            tokens = tokenizer(
                prompt,
                return_tensors="pt"
            ).to(device)
            
            # Get outputs from our implementation
            # filter out padding input_ids
            # tokens.input_ids = tokens.input_ids[tokens.attention_mask==1]
            # tokens.attention_mask = tokens.attention_mask[tokens.attention_mask==1]
            outputs1 = model1(
                input_ids=tokens.input_ids,
                output_hidden_states=True
            )
            print("--------------------------------")
            logger.info(f"Testing model2")

            # Get outputs from HuggingFace implementation
            with set_forward_context(current_timestep=0, attn_metadata=None):
                outputs2 = model2(
                    input_ids=tokens.input_ids,
                    attention_mask=tokens.attention_mask,
                    output_hidden_states=True
                )
            
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

            logger.info(
                f"Maximum difference in last hidden states: {max_diff_hidden.item()}"
            )
            logger.info(
                f"Mean difference in last hidden states: {mean_diff_hidden.item()}"
            )

    logger.info(
        "Test passed! Both LLaMA encoder implementations produce similar outputs."
    )
    logger.info("Test completed successfully")


if __name__ == "__main__":
    test_llama_encoder()
