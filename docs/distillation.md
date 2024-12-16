## 🎯 Distill


Our distillation recipe is based on [Phased Consistency Model](https://github.com/G-U-N/Phased-Consistency-Model). We did not find significant improvement using multi-phase distillation, so we keep the one phase setup similar to the original latent consistency model's recipe.

We use the [MixKit](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main/all_mixkit) dataset for distillation. To avoid running the text encoder and VAE during training, we preprocess all data to generate text embeddings and VAE latents.

Preprocessing instructions can be found [data_preprocess.md](#-data-preprocess). For convenience, we also provide preprocessed data that can be downloaded directly using the following command:

```bash
python scripts/huggingface/download_hf.py --repo_id=FastVideo/HD-Mixkit-Finetune-Hunyuan --local_dir=data/HD-Mixkit-Finetune-Hunyuan --repo_type=dataset
```
Next, download the original model weights with:

```bash
python scripts/huggingface/download_hf.py --repo_id=FastVideo/hunyuan --local_dir=data/hunyuan --repo_type=model
```
To launch the distillation process, use the following commands:

```
bash scripts/distill/distill_mochi.sh # for mochi
bash scripts/distill/distill_hunyuan.sh # for hunyuan
```
We also provide an optional script for distillation with adversarial loss, located at `fastvideo/distill_adv.py`. Although we tried adversarial loss, we did not observe significant improvements.
