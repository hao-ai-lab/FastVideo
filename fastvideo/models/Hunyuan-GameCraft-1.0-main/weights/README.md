# Download Pretrained Models

All models are stored in `Hunyuan-GameCraft-1.0/weights` by default, and the file structure is as follows
```shell
Hunyuan-GameCraft-1.0
  ├──weights
  │  ├──gamecraft_models
  │  │  │──mp_rank_00_model_states.pt
  │  │  │──mp_rank_00_model_states_distill.pt
  │  │──stdmodels
  │  │  ├──vae_3d
  │  │  │  │──hyvae
  │  │  │  │  ├──pytorch_model.pt
  │  │  │  │  ├──config.json
  │  │  ├──llava-llama-3-8b-v1_1-transformers
  │  │  │  ├──model-00001-of-00004.safatensors
  │  │  │  ├──model-00002-of-00004.safatensors
  │  │  │  ├──model-00003-of-00004.safatensors
  │  │  │  ├──model-00004-of-00004.safatensors
  │  │  │  ├──...
  │  │  ├──openai_clip-vit-large-patch14
```

## Download Hunyuan-GameCraft-1.0 model
To download the Hunyuan-GameCraft-1.0 model, first install the huggingface-cli. (Detailed instructions are available [here](https://huggingface.co/docs/huggingface_hub/guides/cli).)

```shell
python -m pip install "huggingface_hub[cli]"
```

Then download the model using the following commands:

```shell
# Switch to the directory named 'Hunyuan-GameCraft-1.0/weights'
cd Hunyuan-GameCraft-1.0/weights
# Use the huggingface-cli tool to download HunyuanVideo-Avatar model in HunyuanVideo-Avatar/weights dir.
# The download time may vary from 10 minutes to 1 hour depending on network conditions.
huggingface-cli download tencent/Hunyuan-GameCraft-1.0 --local-dir ./
```
