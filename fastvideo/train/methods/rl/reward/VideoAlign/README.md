<h1 align="center"> Improving Video Generation with Human Feedback </h1>
<div align="center">
  <!-- <a href='LICENSE'><img src='https://img.shields.io/badge/license-MIT-yellow'></a> -->
  <a href='https://arxiv.org/abs/2501.13918'><img src='https://img.shields.io/badge/arXiv-VideoAlign-red'></a>  &nbsp;
  <a href='https://gongyeliu.github.io/videoalign/'><img src='https://img.shields.io/badge/Project-VideoAlign-green'></a> &nbsp;
  <a href="https://github.com/KwaiVGI/VideoAlign"><img src="https://img.shields.io/badge/GitHub-VideoAlign-9E95B7?logo=github"></a> &nbsp;
  <a href='https://huggingface.co/KwaiVGI/VideoReward'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Model-VideoReward-blue'></a> &nbsp;
  <br>
  <a href='https://huggingface.co/datasets/KwaiVGI/VideoGen-RewardBench'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Eval%20Dataset-VideoGen--RewardBench-blue'></a> &nbsp;
  <a href='https://huggingface.co/spaces/KwaiVGI/VideoGen-RewardBench'><img src='https://img.shields.io/badge/Space-VideoGen--RewardBench-orange.svg?logo=data:image/svg+xml;charset=utf-8;base64,PHN2ZyB0PSIxNzM5MjA0MzY2MDEwIiBjbGFzcz0iaWNvbiIgdmlld0JveD0iMCAwIDEwMjQgMTAyNCIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHAtaWQ9IjQzNDYiIHdpZHRoPSIyMDAiIGhlaWdodD0iMjAwIj48cGF0aCBkPSJNNjgyLjY2NjY2NyA0NjkuMzMzMzMzVjEyOEgzNDEuMzMzMzMzdjI1Nkg4NS4zMzMzMzN2NTEyaDg1My4zMzMzMzRWNDY5LjMzMzMzM2gtMjU2eiBtLTI1Ni0yNTZoMTcwLjY2NjY2NnY1OTcuMzMzMzM0aC0xNzAuNjY2NjY2VjIxMy4zMzMzMzN6IG0tMjU2IDI1NmgxNzAuNjY2NjY2djM0MS4zMzMzMzRIMTcwLjY2NjY2N3YtMzQxLjMzMzMzNHogbTY4Mi42NjY2NjYgMzQxLjMzMzMzNGgtMTcwLjY2NjY2NnYtMjU2aDE3MC42NjY2NjZ2MjU2eiIgcC1pZD0iNDM0NyIgZmlsbD0iIzhhOGE4YSI+PC9wYXRoPjwvc3ZnPg=='></a> &nbsp;
  <br>
</div>


## 📖 Introduction


This repository open-sources the **VideoReward** component -- our VLM-based reward model introduced in the paper [Improving Video Generation with Human Feedback](https://arxiv.org/abs/2501.13918). For Flow-DPO, we provide an implementation for text-to-image tasks [here](https://github.com/yifan123/flow_grpo/blob/main/scripts/single_node/dpo.sh).


VideoReward evaluates generated videos across three critical dimensions:
* Visual Quality (VQ): The clarity, aesthetics, and single-frame reasonableness.
* Motion Quality (MQ): The dynamic stability, dynamic reasonableness, naturalness, and dynamic degress.
* Text Alignment (TA): The relevance between the generated video and the text prompt.

This versatile reward model can be used for data filtering, guidance, reject sampling, DPO, and other RL methods. <br>

<img src=https://gongyeliu.github.io/videoalign/pics/overview.png width="100%"/>



## 📝 Updates
- __[2025.08.14]__: 🔥 We provide the prompt sets used to evaluate video generation performance in this paper, including VBench, VideoGen-Eval, and TA-Hard. See [`./datasets/video_eval_prompts`](./datasets/video_eval_prompts/README.md) for details.
- __[2025.07.17]__: 🔥 Release the [Flow-DPO](https://github.com/yifan123/flow_grpo/blob/main/scripts/single_node/dpo.sh).
- __[2025.02.08]__: 🔥 Release the [VideoGen-RewardBench](https://huggingface.co/datasets/KwaiVGI/VideoGen-RewardBench) and [Leaderboard](https://huggingface.co/spaces/KwaiVGI/VideoGen-RewardBench).
- __[2025.02.08]__: 🔥 Release the [Code](#) and [Checkpoints](https://huggingface.co/KwaiVGI/VideoReward) of VideoReward.
- __[2025.01.23]__: Release the [Paper](https://arxiv.org/abs/2501.13918) and [Project Page](https://gongyeliu.github.io/videoalign/).


##  🚀 Quick Started

### 1. Environment Set Up
Clone this repository and install packages.
```bash
git clone https://github.com/KwaiVGI/VideoAlign
cd VideoAlign
conda env create -f environment.yaml
conda activate VideoReward
pip install flash-attn==2.5.8 --no-build-isolation
```

### 2. Download Pretrained Weights

Please download our checkpoints from [Huggingface](https://huggingface.co/KwaiVGI/VideoReward) and put it in `./checkpoints/`.

```bash
cd checkpoints
git lfs install
git clone https://huggingface.co/KwaiVGI/VideoReward
cd ..
```

### 3. Scoring for a single prompt-video item.

```bash
python inference.py
```


## ✨ Eval the Performance on VideoGen-RewardBench

### 1. Download the VideoGen-RewardBench and put it in `./datasets/`.

```bash
cd dataset
git lfs install
git clone https://huggingface.co/datasets/KwaiVGI/VideoGen-RewardBench
cd ..
```

### 2. Start inference

```bash
python eval_videogen_rewardbench.py
```

## 🏁 Train RM on Your Own Data
### 1. Prepare your own data as the [instruction](./datasets/train/README.md) stated.

### 2. Start training!
```bash
sh train.sh
```



## 🤗 Acknowledgments

Our reward model is based on [QWen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct), and our code is build upon [TRL](https://github.com/huggingface/trl) and [Qwen2-VL-Finetune](https://github.com/2U1/Qwen2-VL-Finetune), thanks to all the contributors!


## ⭐ Citation

Please leave us a star ⭐ if you find our work helpful.
```bibtex
@article{liu2025improving,
  title={Improving video generation with human feedback},
  author={Liu, Jie and Liu, Gongye and Liang, Jiajun and Yuan, Ziyang and Liu, Xiaokun and Zheng, Mingwu and Wu, Xiele and Wang, Qiulin and Qin, Wenyu and Xia, Menghan and others},
  journal={arXiv preprint arXiv:2501.13918},
  year={2025}
}
```
```bibtex
@article{liu2025flow,
  title={Flow-grpo: Training flow matching models via online rl},
  author={Liu, Jie and Liu, Gongye and Liang, Jiajun and Li, Yangguang and Liu, Jiaheng and Wang, Xintao and Wan, Pengfei and Zhang, Di and Ouyang, Wanli},
  journal={arXiv preprint arXiv:2505.05470},
  year={2025}
}
```