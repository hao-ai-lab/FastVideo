(v0-data-preprocess)=

# 🧱 Data Preprocess

To save GPU memory, we precompute text embeddings and VAE latents to eliminate the need to load the text encoder and VAE during training.

We provide a sample dataset to help you get started. Download the source media using the following command:

```bash
python scripts/huggingface/download_hf.py --repo_id=FastVideo/mini_i2v_dataset --local_dir=data/mini_i2v_dataset --repo_type=dataset
```

The folder `crush-smol_raw/` contains raw videos and captions for testing preprocessing, while `crush-smol_preprocessed/` contains latents prepared for testing training.

To preprocess the dataset for fine-tuning or distillation, run:

```
bash scripts/preprocess/v1_preprocess_wan_data_t2v # for wan
```

## Process your own dataset

If you wish to create your own dataset for finetuning or distillation, please refer `mini_i2v_dataset/crush-smol_raw/` to structure you video dataset in the following format:

```
path_to_your_dataset_folder/
├── videos/
│   ├── 0.mp4
│   ├── 1.mp4
├── videos.txt
└── prompt.txt
```

To geranate the `videos2caption.json` and `merge.txt`, run

``` python
python scripts/dataset_preparation/prepare_json_file.py --data_folder mini_i2v_dataset/crush-smol_raw/ --output your_output_folder
```

Adjust the `DATA_MERGE_PATH` and `OUTPUT_DIR` in `scripts/preprocess/v1_preprocess_****.sh` accordingly and run:

```
bash scripts/preprocess/v1_preprocess_****.sh
```

The preprocessed data will be put into the `OUTPUT_DIR` and the `videos2caption.json` can be used in finetune and distill scripts.
