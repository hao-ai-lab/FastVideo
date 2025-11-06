import os
import torch
from tqdm import tqdm
import diffusers
from diffusers import DiffusionPipeline

# Configuration
BASE_MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
FINETUNED_MODEL = "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"
OUTPUT_PATH = "fastwan2.2_transformer_lora.pt"
CHECKPOINT_PATH = "lora_checkpoint.pt"
RANK = 16

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32
print(f"Using device: {device}")

# Setup for Wan models in Diffusers
setattr(diffusers, "WanDMDPipeline", diffusers.WanPipeline)

# Load base model on CPU
print("\nLoading base model (CPU)")
base_pipe = DiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)
base_pipe.to("cpu")
base_state_dict = base_pipe.transformer.state_dict()
del base_pipe
torch.cuda.empty_cache()
print("Base model weights cached on CPU.")

# Load fine-tuned model on GPU
print("\nLoading fine-tuned model (GPU)")
finetuned_pipe = DiffusionPipeline.from_pretrained(
    FINETUNED_MODEL,
    torch_dtype=torch_dtype,
)
finetuned_pipe.to(device)
ft_transformer = finetuned_pipe.transformer

# Initialize LoRA extraction state
lora_dict = {}
start_idx = 0

if os.path.exists(CHECKPOINT_PATH):
    print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    lora_dict = checkpoint["lora_dict"]
    start_idx = checkpoint["index"]

ft_state_dict = ft_transformer.state_dict()
keys = sorted(ft_state_dict.keys())
print(f"\nTotal parameters to inspect: {len(keys)}")

mean_deltas = []
progress_bar = tqdm(enumerate(keys), total=len(keys), desc="Extracting LoRA deltas")

for i, key in progress_bar:
    if i < start_idx:
        continue
    if key not in base_state_dict or not key.endswith("weight"):
        continue
    if any(substr in key for substr in ["norm", "bias", "embedding"]):
        continue

    W_base = base_state_dict[key].to(device, dtype=torch.float32)
    W_ft = ft_state_dict[key].to(device, dtype=torch.float32)
    if W_base.shape != W_ft.shape:
        continue

    delta = W_ft - W_base
    mean_abs = delta.abs().mean().item()
    mean_deltas.append(mean_abs)

    if i % 25 == 0:
        print(f"\n[{i}/{len(keys)}] {key}: mean|delta|={mean_abs:.5f}")

    try:
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        rank = min(RANK, S.size(0))
        A = (U[:, :rank] * S[:rank].sqrt()).half().cpu()
        B = (Vh[:rank, :].T * S[:rank].sqrt()).half().cpu()
        lora_dict[key] = {"A": A, "B": B}
    except Exception as e:
        print(f"Skipping {key} due to error: {e}")
        continue

    del W_base, W_ft, delta, U, S, Vh
    torch.cuda.empty_cache()

    if i % 50 == 0 and i > 0:
        torch.save({"index": i, "lora_dict": lora_dict}, CHECKPOINT_PATH)
        print(f"Checkpoint saved at {i}")

# Final save
torch.save(lora_dict, OUTPUT_PATH)
print("\nLoRA extraction complete.")
print(f"Average |delta| magnitude: {sum(mean_deltas)/len(mean_deltas):.5f}")
if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)