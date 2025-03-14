from fastvideo.v1.models.encoders.clip import CLIPTextModel
from fastvideo.v1.models.hunyuan.text_encoder import TextEncoder
import os
import torch
import torch.nn as nn
import argparse
import numpy as np
from transformers import CLIPTextModel as HFCLIPTextModel
from fastvideo.v1.logger import init_logger
from fastvideo.v1.models.hunyuan.text_encoder import load_text_encoder, load_tokenizer

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
    parser = argparse.ArgumentParser(description='CLIP Text Encoder Test')
    parser.add_argument('--model-path', type=str, default="openai/clip-vit-large-patch14",
                        help='Path to the CLIP model')
    parser.add_argument('--precision', type=str, default="bfloat16",
                        help='Precision to use for the model (float32, float16, bfloat16)')
    return parser.parse_args()

def test_clip_encoder():
    args = setup_args()
    
    # Set fixed random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Initialize the two model implementations
    logger.info(f"Loading models from {args.model_path}")
    
    # Load our implementation using the loader from text_encoder/__init__.py
    model1, _ = load_text_encoder(
        text_encoder_type="clipL",
        text_encoder_precision=args.precision,
        text_encoder_path=args.model_path,
        logger=logger,
        device=device
    )
    
    # Load the HuggingFace implementation directly
    model2 = HFCLIPTextModel.from_pretrained(args.model_path)
    if args.precision == "bfloat16":
        model2 = model2.to(torch.bfloat16)
    elif args.precision == "float16":
        model2 = model2.to(torch.float16)
    model2 = model2.to(device)
    model2.eval()
    
    # Load tokenizer
    tokenizer, _ = load_tokenizer(
        tokenizer_type="clipL",
        tokenizer_path=args.model_path,
        logger=logger
    )
    
    # Test with some sample prompts
    prompts = [
        "a photo of a cat",
        "a beautiful landscape with mountains",
        "an astronaut riding a horse on the moon"
    ]
    
    logger.info("Testing CLIP text encoder with sample prompts")
    
    with torch.no_grad():
        for prompt in prompts:
            logger.info(f"Testing prompt: '{prompt}'")
            
            # Tokenize the prompt
            tokens = tokenizer(
                prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            # Get embeddings from our implementation
            outputs1 = model1(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                output_hidden_states=True
            )
            
            # Get embeddings from HuggingFace implementation
            outputs2 = model2(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                output_hidden_states=True
            )
            
            # Compare last hidden states
            last_hidden_state1 = outputs1.last_hidden_state
            last_hidden_state2 = outputs2.last_hidden_state
            
            assert last_hidden_state1.shape == last_hidden_state2.shape, \
                f"Hidden state shapes don't match: {last_hidden_state1.shape} vs {last_hidden_state2.shape}"
            
            max_diff_hidden = torch.max(torch.abs(last_hidden_state1 - last_hidden_state2))
            mean_diff_hidden = torch.mean(torch.abs(last_hidden_state1 - last_hidden_state2))
            
            logger.info(f"Maximum difference in last hidden states: {max_diff_hidden.item()}")
            logger.info(f"Mean difference in last hidden states: {mean_diff_hidden.item()}")
            
            # Compare pooler outputs
            pooler_output1 = outputs1.pooler_output
            pooler_output2 = outputs2.pooler_output
            
            assert pooler_output1.shape == pooler_output2.shape, \
                f"Pooler output shapes don't match: {pooler_output1.shape} vs {pooler_output2.shape}"
            
            max_diff_pooler = torch.max(torch.abs(pooler_output1 - pooler_output2))
            mean_diff_pooler = torch.mean(torch.abs(pooler_output1 - pooler_output2))
            
            logger.info(f"Maximum difference in pooler outputs: {max_diff_pooler.item()}")
            logger.info(f"Mean difference in pooler outputs: {mean_diff_pooler.item()}")
            
            # Check if outputs are similar (allowing for small numerical differences)
            assert max_diff_hidden < 1e-4, \
                f"Hidden states differ significantly: max diff = {max_diff_hidden.item()}"
            assert max_diff_pooler < 1e-4, \
                f"Pooler outputs differ significantly: max diff = {max_diff_pooler.item()}"
    
    logger.info("Test passed! Both CLIP text encoder implementations produce similar outputs.")
    logger.info("Test completed successfully")

if __name__ == "__main__":
    test_clip_encoder()