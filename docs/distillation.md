## 🎯 Distill


Our distillation recipe is based on [Phased Consistency Model](https://github.com/G-U-N/Phased-Consistency-Model). We did not find significant improvement using multi-phase distillation, so we keep the one phase setup similar to the original latent consistency model's recipe.

We source our training data from the [MixKit](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main/all_mixkit). To avoid inference text encoder and vae during training,  we first preprocess all the data to get the text embeddings and VAE latents for the video frames. 

The preprocess instruction can be found in [data_preprocess.md](#-data-preprocess). We also provide the preprocessed data for you to start with, download the dataset with command:

```bash
python scripts/huggingface/download_hf.py --repo_id=FastVideo/HD-Mixkit-Finetune-Hunyuan --local_dir=data/HD-Mixkit-Finetune-Hunyuan --repo_type=dataset
```
Then download the original model weight with command:

```bash
python scripts/huggingface/download_hf.py --repo_id=FastVideo/hunyuan --local_dir=data/hunyuan --repo_type=model
```
Then the distillation can be launched by:

```
bash scripts/distill/distill_mochi.sh # for mochi
bash scripts/distill/distill_hunyuan.sh # for hunyuan
```

We additionaly provide a code to distill with adversarial loss located at `fastvideo/distill_adv.py`, even though we did not find significant improvement using adversarial loss.
