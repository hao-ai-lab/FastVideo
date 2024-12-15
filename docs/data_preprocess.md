


## 🧱 Data Preprocess

To avoid loading text encoder and VAE during training, we precompute the text embeddings and latents for the video frames. 


### Sample for Data Preprocess

We provide a small sample dataset for you to start with, download the source media with command:
```
python scripts/huggingface/download_hf.py --repo_id=FastVideo/Image-Vid-Finetune-Src --local_dir=data/Image-Vid-Finetune-Src --repo_type=dataset
```
To preprocess dataset for finetune/distill run:

```
bash scripts/preprocess/preprocess_mochi_data.sh # for mochi
bash scripts/preprocess/preprocess_hunyuan_data.sh # for hunyuan
```

The preprocessed dataset will be stored in `Image-Vid-Finetune-Mochi` or `Image-Vid-Finetune-HunYuan` correspondingly.

### Create Custom Dataset

If you wish to create your own dataset for finetuning or distillation, please pay attention to the following format:

Use a txt file to contain the source folder for media and the JSON file for meta information:

```
path_to_media_source_foder,path_to_json_file
```
The content of the JSON file is a list with each item corresponding to a media source.

For image media, the JSON item needs to follow this format:
```
{
    "path": "0.jpg",
    "cap": ["captions"]
}
```
For video media, the JSON item needs to follow this format:
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
Adjust the `DATA_MERGE_PATH` and `OUTPUT_DIR` in `scripts/preprocess/preprocess_****_data.sh` accordingly and run:
```
bash scripts/preprocess/preprocess_****_data.sh
```
The preprocessed data will be put into the `OUTPUT_DIR` and the `videos2caption.json` can be used in finetune and distill scripts.
