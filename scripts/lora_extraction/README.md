# LoRA Extraction and Verification Scripts

This folder contains utility scripts for **extracting, verifying, and comparing LoRA adapters** between a base diffusion model and its fine-tuned version.  
It was developed for FastVideo models built on the [Wan 2.2 TI2V architecture](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers).

---

## Scripts Overview

### 1. `extract_lora.py`
Extracts LoRA adapter matrices by computing the weight deltas between the **base model** and the **fine-tuned model**, then performing low-rank decomposition.

**Default configuration:**
- Base model: `Wan-AI/Wan2.2-TI2V-5B-Diffusers`
- Finetuned model: `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers`
- Rank: 16
- Outputs LoRA checkpoint at `fastwan2.2_transformer_lora.pt`

Supports checkpointing for large model runs to avoid restarting on failure.

---

### 2. `verify_lora.py`
Loads the extracted LoRA weights and re-applies them to the base model to verify equivalence to the fine-tuned model numerically.

---

### 3. `compare_lora_outputs.py`
Compares visual outputs (e.g. generated frames or videos) between:
- Base model  
- Fine-tuned model  
- Base + Extracted LoRA  

to validate that the extracted LoRA produces comparable results.

---

## Usage

Run extraction:
```bash
python scripts/lora_extraction/extract_lora.py
```

Resume from a saved checkpoint:
```bash
python scripts/lora_extraction/extract_lora.py --resume
```

Verify:
```bash
python scripts/lora_extraction/verify_lora.py
```

Compare outputs:
```bash
python scripts/lora_extraction/compare_lora_outputs.py
```
