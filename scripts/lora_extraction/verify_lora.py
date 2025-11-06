import torch
import diffusers
from diffusers import DiffusionPipeline
from tqdm import tqdm

BASE_MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
FINETUNED_MODEL = "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"
LORA_PATH = "fastwan2.2_transformer_lora.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32
print(f"Using device: {device}")

setattr(diffusers, "WanDMDPipeline", diffusers.WanPipeline)

# --- Load base and LoRA weights ---
print("\nLoading base model (CPU)...")
base_pipe = DiffusionPipeline.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
base_state_dict = base_pipe.transformer.state_dict()
del base_pipe
torch.cuda.empty_cache()

print("Loading LoRA deltas...")
lora_dict = torch.load(LORA_PATH, map_location="cpu")

print("Loading fine-tuned model (CPU)...")
ft_pipe = DiffusionPipeline.from_pretrained(FINETUNED_MODEL, torch_dtype=torch.float32)
ft_state_dict = ft_pipe.transformer.state_dict()
del ft_pipe
torch.cuda.empty_cache()

# --- Verify LoRA reconstruction ---
print("\nApplying LoRA deltas and verifying per-layer similarity...")
mse_total, count = 0.0, 0
progress = tqdm(lora_dict.items(), total=len(lora_dict))

for key, val in progress:
    if key not in base_state_dict or key not in ft_state_dict:
        continue

    W_base = base_state_dict[key].float()
    A = val["A"].float()
    B = val["B"].float()
    W_lora = W_base + A @ B.T
    W_ft = ft_state_dict[key].float()

    mse = torch.mean((W_lora - W_ft) ** 2).item()
    mse_total += mse
    count += 1

    if count % 50 == 0:
        progress.set_description(f"{key}: MSE={mse:.2e}")

avg_mse = mse_total / max(count, 1)
print("\nVerification complete.")
print(f"Average MSE between fine-tuned and LoRA-applied weights: {avg_mse:.6e}")