import os

import torch
from transformers import AutoTokenizer

from fastvideo.configs.pipelines import LTX2T2VConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.models.loader.component_loader import (
    TextEncoderLoader,
    TransformerLoader,
)


def main() -> None:
    transformer_path = os.getenv(
        "LTX2_FASTVIDEO_PATH", "converted/ltx2/transformer"
    )
    text_encoder_path = os.getenv(
        "LTX2_TEXT_ENCODER_PATH", "converted/ltx2/text_embedding_projection"
    )
    gemma_model_path = os.getenv("LTX2_GEMMA_MODEL_PATH", "")

    if not os.path.isdir(transformer_path):
        raise FileNotFoundError(
            f"Missing LTX-2 transformer weights at {transformer_path}"
        )
    if not os.path.isdir(text_encoder_path):
        raise FileNotFoundError(
            f"Missing LTX-2 text encoder weights at {text_encoder_path}"
        )
    if not gemma_model_path or not os.path.isdir(gemma_model_path):
        raise FileNotFoundError(
            "Set LTX2_GEMMA_MODEL_PATH to a local Gemma model directory."
        )

    os.environ["LTX2_GEMMA_MODEL_PATH"] = gemma_model_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16

    args = FastVideoArgs(
        model_path=transformer_path,
        pipeline_config=LTX2T2VConfig(),
    )

    transformer = TransformerLoader().load(transformer_path, args).to(device)
    text_encoder = TextEncoderLoader().load(text_encoder_path, args).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        gemma_model_path, local_files_only=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = (
        "A curious raccoon peers through a vibrant field of yellow sunflowers."
    )
    inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=1024,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        with set_forward_context(current_timestep=0, attn_metadata=None):
            text_outputs = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        prompt_embeds = text_outputs.last_hidden_state.to(
            device=device, dtype=precision
        )

        batch_size = 1
        frames = 9
        height = 64
        width = 64
        hidden_states = torch.randn(
            batch_size,
            transformer.num_channels_latents,
            frames,
            height,
            width,
            device=device,
            dtype=precision,
        )
        timestep = torch.tensor([500], device=device, dtype=torch.float32)

        with set_forward_context(current_timestep=0, attn_metadata=None):
            output = transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
            )

    print(f"LTX-2 transformer output shape: {output.shape}")


if __name__ == "__main__":
    main()
