<div align="center">
<img src=assets/logo.png width="30%"/>
</div>

**FastVideo is a unified framework for accelerated video generation.**

FastVideo is an inference and post-training framework for diffusion models. It features an end-to-end unified pipeline for accelerating diffusion models, starting from data preprocessing to model training, finetuning, distillation, and inference. FastVideo is designed to be modular and extensible, allowing users to easily add new optimizations and techniques. Whether it is training-free optimizations or post-training optimizations, FastVideo has you covered.

<p align="center">
    | <a href="https://hao-ai-lab.github.io/FastVideo"><b>Documentation</b></a> | <a href="https://hao-ai-lab.github.io/FastVideo/inference/inference_quick_start.html"><b> Quick Start</b></a> | 🤗 <a href="https://huggingface.co/FastVideo/FastWan2.1-T2V-1.3B-Diffusers"  target="_blank"><b>FastWan2.1</b></a>  | 🤗 <a href="Wan-AI/Wan2.2-TI2V-5B-Diffusers" target="_blank"><b>FastWan2.2</b></a> | 🟣💬 <a href="https://join.slack.com/t/fastvideo/shared_invite/zt-38u6p1jqe-yDI1QJOCEnbtkLoaI5bjZQ" target="_blank"> <b>Slack</b> </a> |
</p>

<div align="center">
<img src=assets/fastwan.png width="90%"/>
</div>

## NEWS
- ```2025/08/04```: Release [FastWan](https://hao-ai-lab.github.io/blogs/fastvideo_post_training/), achieving 15x end-to-end speedup for video generation with sparse distillation.
- ```2025/06/14```: Release finetuning and inference code for [VSA](https://arxiv.org/pdf/2505.13389)
- ```2025/04/24```: [FastVideo V1](https://hao-ai-lab.github.io/blogs/fastvideo/) is released!
- ```2025/02/18```: Release the inference code for [Sliding Tile Attention](https://hao-ai-lab.github.io/blogs/sta/).

## Key Features

FastVideo has the following features:
- State-of-the-art performance optimizations for inference
  - [Sliding Tile Attention](https://arxiv.org/pdf/2502.04507)
  - [TeaCache](https://arxiv.org/pdf/2411.19108)
  - [Sage Attention](https://arxiv.org/abs/2410.02367)
- E2E post-training support
  - Data preprocessing pipeline for video data.
  - [Sparse distillation](https://hao-ai-lab.github.io/blogs/fastvideo_post_training/) for Wan2.1 and Wan2.2 using [Video Sparse Attention](https://arxiv.org/pdf/2505.13389) and [Distribution Matching Distillation](https://tianweiy.github.io/dmd2/)
  - Support full finetuning and LoRA finetuning for state-of-the-art open video DiTs.
  - Scalable training with FSDP2, sequence parallelism, and selective activation checkpointing, with near linear scaling to 64 GPUs.

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
Here's a minimal example to generate a video using the default settings. Create a file called `example.py` with the following code:

```python
from fastvideo import VideoGenerator

def main():
    # Create a video generator with a pre-trained model
    generator = VideoGenerator.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
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
- [Finetuning Guide](https://hao-ai-lab.github.io/FastVideo/training/finetune.html)

## 📑 Development Plan

<!-- - More distillation methods -->
  <!-- - [ ] Add Distribution Matching Distillation -->
- More models support
  <!-- - [ ] Add CogvideoX model -->
  - [x] Add StepVideo to V1
- Optimization features
  - [x] Teacache in V1
  - [x] SageAttention in V1
- Code updates
  - [x] V1 Configuration API
  - [ ] Support Training in V1
  <!-- - [ ] fp8 support -->
  <!-- - [ ] faster load model and save model support -->

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

We thank MBZUAI and [Anyscale](https://www.anyscale.com/) for their support throughout this project.

## Citation
If you use FastVideo for your research, please cite our work:

```bibtex
@software{fastvideo2024,
  title        = {FastVideo: A Unified Framework for Accelerated Video Generation},
  author       = {The FastVideo Team},
  url          = {https://github.com/hao-ai-lab/FastVideo},
  month        = apr,
  year         = {2024},
}

@misc{zhang2025vsafastervideodiffusion,
      title={VSA: Faster Video Diffusion with Trainable Sparse Attention}, 
      author={Peiyuan Zhang and Haofeng Huang and Yongqi Chen and Will Lin and Zhengzhong Liu and Ion Stoica and Eric Xing and Hao Zhang},
      year={2025},
      eprint={2505.13389},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.13389}, 
}
@misc{zhang2025fastvideogenerationsliding,
      title={Fast Video Generation with Sliding Tile Attention},
      author={Peiyuan Zhang and Yongqi Chen and Runlong Su and Hangliang Ding and Ion Stoica and Zhenghong Liu and Hao Zhang},
      year={2025},
      eprint={2502.04507},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.04507},
}
```
