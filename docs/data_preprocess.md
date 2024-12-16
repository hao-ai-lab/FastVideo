


## 🧱 Data Preprocess

To save GPU memory, we precompute text embeddings and VAE latents to eliminate the need to load the text encoder and VAE during training.


We provide a sample dataset to help you get started. Download the source media using the following command:
```bash
python scripts/huggingface/download_hf.py --repo_id=FastVideo/Image-Vid-Finetune-Src --local_dir=data/Image-Vid-Finetune-Src --repo_type=dataset
```
To preprocess the dataset for fine-tuning or distillation, run:
```
bash scripts/preprocess/preprocess_mochi_data.sh # for mochi
bash scripts/preprocess/preprocess_hunyuan_data.sh # for hunyuan
```

The preprocessed dataset will be stored in `Image-Vid-Finetune-Mochi` or `Image-Vid-Finetune-HunYuan` correspondingly.

### Process your own dataset

If you wish to create your own dataset for finetuning or distillation, please structure you video dataset in the following format:

path_to_dataset_folder/
├── media/
│   ├── 0.jpg
│   ├── 1.mp4
│   ├── 2.jpg
├── video2caption.json
└── merge.txt

Format the JSON file as a list, where each item represents a media source:

For image media,
```
{
    "path": "0.jpg",
    "cap": ["captions"]
}
```
For video media, 
```
{
    "path": "1.mp4",
    "resolution": {
      "width": 848,
      "height": 480
    },
    "fps": 30.0,
    "duration": 6.033333333333333,
    "cap": [
      "caption"
    ]
  }
```

Use a txt file (merge.txt) to contain the source folder for media and the JSON file for meta information:

```
path_to_media_source_foder,path_to_json_file
```

Adjust the `DATA_MERGE_PATH` and `OUTPUT_DIR` in `scripts/preprocess/preprocess_****_data.sh` accordingly and run:
```
bash scripts/preprocess/preprocess_****_data.sh
```
The preprocessed data will be put into the `OUTPUT_DIR` and the `videos2caption.json` can be used in finetune and distill scripts.
