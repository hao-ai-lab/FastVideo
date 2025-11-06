<div align="center">
<img src=assets/logos/logo.svg width="30%"/>
</div>

**FastVideo is a unified post-training and inference framework for accelerated video generation.**

FastVideo features an end-to-end unified pipeline for accelerating diffusion models, starting from data preprocessing to model training, finetuning, distillation, and inference. FastVideo is designed to be modular and extensible, allowing users to easily add new optimizations and techniques. Whether it is training-free optimizations or post-training optimizations, FastVideo has you covered.

<p align="center">
    | 🕹️ <a href="https://fastwan.fastvideo.org/"<b>Online Demo</b></a> | <a href="https://hao-ai-lab.github.io/FastVideo"><b>Documentation</b></a> | <a href="https://hao-ai-lab.github.io/FastVideo/inference/inference_quick_start.html"><b> Quick Start</b></a> | 🤗 <a href="https://huggingface.co/collections/FastVideo/fastwan-6886a305d9799c8cd1496408"  target="_blank"><b>FastWan</b></a>  | 🟣💬 <a href="https://join.slack.com/t/fastvideo/shared_invite/zt-3csdw1isz-Euq8_Q8~baewG8hxjXs2gQ" target="_blank"> <b>Slack</b> </a> |  🟣💬 <a href="https://ibb.co/tMwknPLY" target="_blank"> <b> WeChat </b> </a> |
</p>

<div align="center">
<img src=assets/fastwan.png width="90%"/>
</div>

## NEWS
- ```2025/08/04```: Release [FastWan](https://hao-ai-lab.github.io/FastVideo/distillation/dmd.html) models and [Sparse-Distillation](https://hao-ai-lab.github.io/blogs/fastvideo_post_training/).
- ```2025/06/14```: Release finetuning and inference code for [VSA](https://arxiv.org/pdf/2505.13389)
- ```2025/04/24```: [FastVideo V1](https://hao-ai-lab.github.io/blogs/fastvideo/) is released!
- ```2025/02/18```: Release the inference code for [Sliding Tile Attention](https://hao-ai-lab.github.io/blogs/sta/).

## Key Features

FastVideo has the following features:
- End-to-end post-training support:
  - [Sparse distillation](https://hao-ai-lab.github.io/blogs/fastvideo_post_training/) for Wan2.1 and Wan2.2 to achineve >50x denoising speedup
  - Data preprocessing pipeline for video data
  - Support full finetuning and LoRA finetuning for state-of-the-art open video DiTs
  - Scalable training with FSDP2, sequence parallelism, and selective activation checkpointing, with near linear scaling to 64 GPUs
- State-of-the-art performance optimizations for inference
  - [Video Sparse Attention](https://arxiv.org/pdf/2505.13389)
  - [Sliding Tile Attention](https://arxiv.org/pdf/2502.04507)
  - [TeaCache](https://arxiv.org/pdf/2411.19108)
  - [Sage Attention](https://arxiv.org/abs/2410.02367)
- Diverse hardware and OS support
  - Support H100, A100, 4090
  - Support Linux, Windows, MacOS

## Getting Started
We recommend using an environment manager such as `Conda` to create a clean environment:

```bash
# Create and activate a new conda environment
conda create -n fastvideo python=3.12
conda activate fastvideo

# Install FastVideo
pip install fastvideo
```

Please see our [docs](https://hao-ai-lab.github.io/FastVideo/getting_started/installation.html) for more detailed installation instructions.

## Sparse Distillation
For our sparse distillation techniques, please see our [distillation docs](https://hao-ai-lab.github.io/FastVideo/distillation/dmd.html) and check out our [blog](https://hao-ai-lab.github.io/blogs/fastvideo_post_training/).

See below for recipes and datasets:

|                                            Model                                              |                                               Sparse Distillation                                                 |                                                  Dataset                                                  |
|:-------------------------------------------------------------------------------------------:  |:---------------------------------------------------------------------------------------------------------------:  |:--------------------------------------------------------------------------------------------------------: |
| [FastWan2.1-T2V-1.3B](https://huggingface.co/FastVideo/FastWan2.1-T2V-1.3B-Diffusers)         |    [Recipe](https://github.com/hao-ai-lab/FastVideo/tree/main/examples/distill/Wan2.1-T2V/Wan-Syn-Data-480P)      | [FastVideo Synthetic Wan2.1 480P](https://huggingface.co/datasets/FastVideo/Wan-Syn_77x448x832_600k)      |
| [FastWan2.1-T2V-14B-Preview](https://huggingface.co/FastVideo/FastWan2.1-T2V-14B-Diffusers)   |                                                   Coming soon!                                                    |   [FastVideo Synthetic Wan2.1 720P](https://huggingface.co/datasets/FastVideo/Wan-Syn_77x768x1280_250k)   |
| [FastWan2.2-TI2V-5B](https://huggingface.co/FastVideo/FastWan2.2-TI2V-5B-Diffusers)           | [Recipe](https://github.com/hao-ai-lab/FastVideo/tree/main/examples/distill/Wan2.2-TI2V-5B-Diffusers/Data-free)   | [FastVideo Synthetic Wan2.2 720P](https://huggingface.co/datasets/FastVideo/Wan2.2-Syn-121x704x1280_32k)  |

## Inference
### Generating Your First Video
Here's a minimal example to generate a video using the default settings. Make sure VSA kernels are [installed](https://hao-ai-lab.github.io/FastVideo/video_sparse_attention/installation.html). Create a file called `example.py` with the following code:

```python
import os
from fastvideo import VideoGenerator

def main():
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"

    # Create a video generator with a pre-trained model
    generator = VideoGenerator.from_pretrained(
        "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
        num_gpus=1,  # Adjust based on your hardware
    )

    # Define a prompt for your video
    prompt = "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes wide with interest."

    # Generate the video
    video = generator.generate_video(
        prompt,
        return_frames=True,  # Also return frames from this call (defaults to False)
        output_path="my_videos/",  # Controls where videos are saved
        save_video=True
    )

if __name__ == '__main__':
    main()
```

Run the script with:

```bash
python example.py
```

For a more detailed guide, please see our [inference quick start](https://hao-ai-lab.github.io/FastVideo/inference/inference_quick_start.html).

### Other docs:

- [Design Overview](https://hao-ai-lab.github.io/FastVideo/design/overview.html)
- [Contribution Guide](https://hao-ai-lab.github.io/FastVideo/getting_started/installation.html)

## Distillation and Finetuning
- [Distillation Guide](https://hao-ai-lab.github.io/FastVideo/distillation/dmd.html)
<!-- - [Finetuning Guide](https://hao-ai-lab.github.io/FastVideo/training/finetune.html) -->

## 📑 Development Plan
<!-- - More distillation methods -->
  <!-- - [ ] Add Distribution Matching Distillation -->
More FastWan Models Coming Soon!
- [ ] Add FastWan2.1-T2V-14B
- [ ] Add FastWan2.2-T2V-14B
- [ ] Add FastWan2.2-I2V-14B
<!-- - Optimization features
- Code updates -->
  <!-- - [ ] fp8 support -->
  <!-- - [ ] faster load model and save model support -->

See details in [development roadmap](https://github.com/hao-ai-lab/FastVideo/issues/468).

## 🤝 Contributing

We welcome all contributions. Please check out our guide [here](https://hao-ai-lab.github.io/FastVideo/contributing/overview.html)

## Acknowledgement
We learned and reused code from the following projects:
- [Wan-Video](https://github.com/Wan-Video)
- [ThunderKittens](https://github.com/HazyResearch/ThunderKittens)
- [Triton](https://github.com/triton-lang/triton)
- [DMD2](https://github.com/tianweiy/DMD2)
- [diffusers](https://github.com/huggingface/diffusers)
- [xDiT](https://github.com/xdit-project/xDiT)
- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)

We thank [MBZUAI](https://ifm.mbzuai.ac.ae/), [Anyscale](https://www.anyscale.com/), and [GMI Cloud](https://www.gmicloud.ai/) for their support throughout this project.

## Citation
If you find FastVideo useful, please considering citing our work:

```bibtex
@software{fastvideo2024,
  title        = {FastVideo: A Unified Framework for Accelerated Video Generation},
  author       = {The FastVideo Team},
  url          = {https://github.com/hao-ai-lab/FastVideo},
  month        = apr,
  year         = {2024},
}

@article{zhang2025vsa,
  title={Vsa: Faster video diffusion with trainable sparse attention},
  author={Zhang, Peiyuan and Chen, Yongqi and Huang, Haofeng and Lin, Will and Liu, Zhengzhong and Stoica, Ion and Xing, Eric and Zhang, Hao},
  journal={arXiv preprint arXiv:2505.13389},
  year={2025}
}

@article{zhang2025fast,
  title={Fast video generation with sliding tile attention},
  author={Zhang, Peiyuan and Chen, Yongqi and Su, Runlong and Ding, Hangliang and Stoica, Ion and Liu, Zhengzhong and Zhang, Hao},
  journal={arXiv preprint arXiv:2502.04507},
  year={2025}
}
```
