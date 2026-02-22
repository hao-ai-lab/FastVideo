# LTX-2 Crush-Smol Example
# TODO: Update this doc.

These are e2e example scripts for finetuning LTX-2 on the crush-smol dataset.

## Execute the following commands from `FastVideo/` to run training:

### Download crush-smol dataset:

`bash examples/training/finetune/ltx2/download_dataset.sh`

#### Or use the scripts at scripts/dataset_preparation to download and prepare the dataset

### Preprocess the videos and captions into latents:

`bash examples/training/finetune/ltx2/preprocess_ltx2_data_t2v_new.sh`

### Edit the following file and run finetuning:

`bash examples/training/finetune/ltx2/finetune_t2v.sh`

Notes:
- Update `DATASET_PATH` in the preprocess script to point to your merged dataset root (`videos/` + `videos2caption.json`).
- `MODEL_PATH` should point to a local LTX-2 diffusers-style directory that contains `model_index.json` and `text_encoder/gemma`.

