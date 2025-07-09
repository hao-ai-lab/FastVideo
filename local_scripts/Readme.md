# 🧪 FastVideo & DMD Distillation: Development, Debugging, and Reproduction

This guide provides streamlined instructions for working with the **FastVideo distillation** and **DMD (CausVid)** repositories, including development/debug branches, environment setup, dataset/model preparation, and execution scripts.

---

## 🔧 FastVideo Distill

### 🔗 Branches

- **Development branch:**  
  [`yq/distill-dmd`](https://github.com/hao-ai-lab/FastVideo/tree/yq/distill-dmd)

- **Debug branch (fixed seed):**  
  [`yq/distill-dmd-fixseed`](https://github.com/hao-ai-lab/FastVideo/tree/yq/distill-dmd-fixseed)

### 🛠️ Environment Setup

Follow the official FastVideo installation instructions:  
📄 [Create and activate a conda environment](https://hao-ai-lab.github.io/FastVideo/getting_started/installation.html#create-and-activate-a-conda-environment-for-fastvideo)

### 📁 Dataset Preparation

Ensure the dataset is available by running:

```bash
python scripts/huggingface/download_hf.py \
  --repo_id=FastVideo/mini_i2v_dataset \
  --local_dir=mini_i2v_dataset \
  --repo_type=dataset
```

This downloads a minimal example dataset sufficient for running the distillation script.

### 🚀 Distillation Script

Use the same script for both `distill-dmd` and `distill-dmd-fixseed`:

📄 [`v1_distill_dmd_wan.sh`](https://github.com/hao-ai-lab/FastVideo/blob/yq/distill-dmd/scripts/distill/v1_distill_dmd_wan.sh)

### 🐛 Debugging Tip

To set a breakpoint in fastvideo codebase:
```python
torch.distributed.breakpoint()
```

---

## 🧩 DMD Official Repo (Forked)

### 🔗 Branches

- **Reproduce branch:**  
  [`yq/debug`](https://github.com/BrianChen1129/CausVid/tree/yq/debug)

- **Debug branch (with fixed seed):**  
  [`yq/debug-fixseed`](https://github.com/BrianChen1129/CausVid/tree/yq/debug-fixseed)

### 🛠️ Environment Setup

- **For `yq/debug` (reproduce branch):**  
  Follow the official DMD README instructions.

- **For `yq/debug-fixseed`:**  
  Use the same environment as FastVideo and then:

```bash
# Inside the FastVideo directory
python setup.py develop

# Then install any missing packages
pip install <missing-packages>
```

### 📦 Model & Dataset

- Download the **official Wan base model** and dataset from Hugging Face:  
  📦 [mixkit_latents_lmdb](https://huggingface.co/tianweiy/CausVid/tree/main/mixkit_latents_lmdb)  
  ➡️ Place the dataset under `checkpoints/`.

### 🚀 Running Scripts

- **Reproduce branch script (8-node run):**  
  📄 [`dmd_8n_simple.sh`](https://github.com/BrianChen1129/CausVid/blob/yq/debug/local_scripts/dmd_8n_simple.sh)

- **Debug branch script (fixed seed and input):**  
  📄 [`debug.sh`](https://github.com/BrianChen1129/CausVid/blob/yq/debug-fixseed/local_scripts/debug.sh)

---
