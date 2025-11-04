# LongCat-Video 融入 FastVideo

## 代办清单

- 核心管线与组件
  - 定义 LongCat 管线配置并注册到 pipeline registry - Done by Alex
  - 新增 FlowMatchEulerDiscreteScheduler 的调度器加载器 - Done by Alex
  - 新增 LongCatVideoTransformer3DModel 的 Transformer 加载器 - Done by Alex
  - 验证并适配 UMT5 文本编码器与分词器加载
  - 复用或适配 Wan VAE 加载与配置
  - 实现 LongCatPipeline 并挂接到 build_pipeline
  - 实现 LongCat 专用去噪阶段支持 T2V/I2V/VC/Refine
  - 实现 I2V/VC 条件编码与潜变量注入阶段
  - 实现 Refine 上采样流程与 t_thresh 逻辑

- 数据结构与 CLI
  - 扩展 ForwardBatch 以支持 LongCat 额外参数（use_distill/kv_cache/num_cond_frames 等）
  - 扩展 SamplingParam 增加 task 与 LongCat 标志位
  - 扩展 generate CLI 支持任务类型与输入源（image/video/long）

- 权重导入与转换
  - 支持直接从目录读取 LongCat 权重与校验（子目录 tokenizer/text_encoder/vae/scheduler/dit）
  - 编写权重转换脚本生成 FastVideo 目录结构（含 model_index.json）

- 性能与可选特性
  - 接入 KV Cache 与可选 CPU offload
  - 接入可选 Block Sparse Attention 后端
  - 接入 LongCat LoRA 装载与启用/禁用接口

- 验证与文档
  - 新增单元与集成测试覆盖构建与一步推理
  - 新增 inference 脚本与文档，给出示例参数
  - 端到端验证各任务并对齐分辨率/帧数约束

简要说明
- LongCat 的组件与接口：`tokenizer=AutoTokenizer`、`text_encoder=UMT5EncoderModel`、`vae=AutoencoderKLWan`、`scheduler=FlowMatchEulerDiscreteScheduler`、`dit=LongCatVideoTransformer3DModel`，其管线提供 4 个入口：`generate_t2v/generate_i2v/generate_vc/generate_refine`，并包含 `use_distill`、KV cache、LoRA、Refine 上采样等特性。
- FastVideo 侧可重用 `ForwardBatch.extra` 承载 LongCat 专属参数；需为 LongCat 建立专用去噪与条件阶段以复刻其时间步与潜变量注入、KV cache 与 Refine 逻辑；并在 `configs/pipelines/` 与 `pipeline_registry` 中建立映射，支持 `--model-path` 到 LongCat 配置的分发。
- 权重支持两路：直接读取 LongCat 目录结构（需校验）或提供转换脚本生成 FastVideo 期望布局与 `model_index.json`。

- 我已经根据 `LongCat-Video` 的 `run_demo_*.py` 与 `pipeline_longcat_video.py`，以及 FastVideo 的 `VideoGenerator`/`ForwardBatch`/管线分阶段设计，整理出上述任务。

## 细节
### 1) 定义 LongCat 管线配置并注册到 pipeline registry
Done by Alex
- 加入了`fastvideo/configs/pipelines/longcat.py`
    - longcat采用umt5, 是t5的后继版本，目前的post process text函数用的是和wan的t5一样的post process procedure
    - LongCatT2V480PConfig内部结构：dit(placeholder，未来要用LongCat3DTransformer替代)(bf16), WAN_VAE(bf16)
    - longcat没有采用flow shift
    - longcat推理需要vae encoder和decoder均可用 
    - use distill设置为了false
    - 自定义参数例如enable_bsa, use_distill, enhance_hf
- 更新 `fastvideo/configs/pipelines/registry.py`
    - PIPE_NAME_TO_CONFIG还没改，之后改
    - PIPELINE_DETECTOR加入了longcat的detect
    - PIPELINE_FALLBACK_CONFIG加入longcat的fallback


### 2) 新增 FlowMatchEulerDiscreteScheduler 的调度器加载器
Done by Alex
- 新增 fastvideo/third_party/longcat/scheduling_flow_match_euler_discrete.py（拷贝 LongCat 实现）。
- 看起来原本Scheduler.load就支持flow match 的scheduler，所以这部分verify为不需要修改


### 3) 新增 LongCatVideoTransformer3DModel 的 Transformer 加载器
Done by Alex
- 将所有LongCat-Video核心代码移植到了third_party/longcat_video下
- 在fastvideo/models/registry.py 新增了行："LongCatVideoTransformer3DModel": ("dits", "longcat_video_dit", "LongCatVideoTransformer3DModel"), 使得TransformerLoader可以正确load LongCat自定义的Transformer3D结构
- TODO: [FUTURE] 后续权重转换的时候确保 LongCat transformer 子目录的 `config.json` 写有 `"_class_name": "LongCatVideoTransformer3DModel"`；并且该目录下有 safetensors 权重，命名与类参数名匹配。
    - 示例（保存为 `<model_root>/dit/config.json`）：
        ```json
        {
        "_class_name": "LongCatVideoTransformer3DModel",
        "_diffusers_version": "0.31.0",
        "in_channels": 16,
        "out_channels": 16,
        "hidden_size": 4096,
        "depth": 48,
        "num_heads": 32,
        "caption_channels": 4096,
        "mlp_ratio": 4,
        "adaln_tembed_dim": 512,
        "frequency_embedding_size": 256,
        "patch_size": [1, 2, 2],
        "enable_flashattn3": false,
        "enable_flashattn2": false,
        "enable_xformers": false,
        "enable_bsa": false,
        "bsa_params": null,
        "cp_split_hw": [1, 1],
        "text_tokens_zero_pad": false
        }
        ```
    - 权重：把 LongCat 的 transformer 权重 `.safetensors` 放在同一个 `dit/` 目录。文件名不限（Loader 会加载该目录下所有 `.safetensors`），但 state_dict 的键必须与模型定义匹配。


### 4) 验证并适配 UMT5 文本编码器与分词器加载

- 分词器（Tokenizer）
  - 不需要新增 Loader。确保传给现有 `TokenizerLoader.load(model_path, fastvideo_args)` 的路径就是 `<model_root>/tokenizer` 子目录（由 `model_index.json` 或目录推断提供）。`AutoTokenizer.from_pretrained(<model_root>/tokenizer)` 即可正常工作。

- 文本编码器（UMT5）
  - 为 UMT5 走 Transformers 的 `from_pretrained` 分支（直接读取 `<model_root>/text_encoder`），绕过本地结构化权重装载。
  - 修改文件：`fastvideo/models/loader/component_loader.py`

1) 顶部 import 增加 UMT5：

```python
from transformers import AutoImageProcessor, AutoTokenizer
from transformers import UMT5EncoderModel  # 新增
```

2) 在 `TextEncoderLoader.load` 开头加入 UMT5 快速路径（检测到 UMT5 就直接用 HF 加载并返回；否则走原逻辑）：

```python
def load(self, model_path: str, fastvideo_args: FastVideoArgs):
    """Load the text encoders based on the model path, and inference args."""
    # 先尝试读取 transformers 配置，判断是否为 UMT5
    try:
        with open(os.path.join(model_path, "config.json")) as f:
            cfg_json = json.load(f)
    except Exception:
        cfg_json = {}

    archs = cfg_json.get("architectures", [])
    model_type = cfg_json.get("model_type", "")
    is_umt5 = ("UMT5EncoderModel" in archs) or (model_type.lower() in {"umt5", "t5"})

    if is_umt5:
        # 设备与精度选择与原逻辑保持一致
        from fastvideo.platforms import current_platform
        if fastvideo_args.text_encoder_cpu_offload:
            target_device = torch.device("mps") if current_platform.is_mps() else torch.device("cpu")
            encoder_precision = fastvideo_args.pipeline_config.text_encoder_precisions[0]
        else:
            target_device = get_local_torch_device()
            encoder_precision = fastvideo_args.pipeline_config.text_encoder_precisions[0]

        dtype = PRECISION_TO_TYPE.get(encoder_precision, torch.bfloat16)
        enc = UMT5EncoderModel.from_pretrained(model_path, torch_dtype=dtype)
        return enc.to(target_device).eval()

    # 非 UMT5 走原有路径
    model_config = get_diffusers_config(model=model_path)
    model_config.pop("_name_or_path", None)
    model_config.pop("transformers_version", None)
    model_config.pop("model_type", None)
    model_config.pop("tokenizer_class", None)
    model_config.pop("torch_dtype", None)
    logger.info("HF Model config: %s", model_config)
    # ... 原始实现的其余部分保持不变 ...
```

- 验收要点
  - `<model_root>/tokenizer/config.json` 存在时，`TokenizerLoader` 直接能加载。
  - `<model_root>/text_encoder/config.json` 包含 `architectures: ["UMT5EncoderModel"]` 或 `model_type: "umt5"` 时，`TextEncoderLoader` 走 HF 分支并返回 `.eval()` 的 `UMT5EncoderModel`。
  - 开启 `--text-encoder-cpu-offload` 时，编码器落在 CPU；否则落在本地 GPU；dtype 由 `pipeline_config.text_encoder_precisions[0]` 控制。

### 5) 复用或适配 Wan VAE 加载与配置

- 直接复用现有 Wan `VAELoader`，仅确认 dtype 可由 `pipeline_config.vae_precision` 控制：
  - 文件：`fastvideo/models/loader/component_loader.py`

```python
# 在 VAELoader.load 内确保：
vae_dtype = torch.bfloat16 if config.vae_precision == "bf16" else torch.float32
vae = AutoencoderKLWan.from_pretrained(model_path, subfolder=subdir, torch_dtype=vae_dtype)
```

### 6) 实现 LongCatPipeline 并挂接到 build_pipeline

- 新增文件：`fastvideo/pipelines/basic/longcat/longcat_pipeline.py`

```python
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.configs.pipelines.longcat import LongCatBaseConfig
from fastvideo.pipelines.stages.input_validation import InputValidationStage
from fastvideo.pipelines.stages.text_encoding import TextEncodingStage
from fastvideo.pipelines.stages.timestep_preparation import TimestepPreparationStage
from fastvideo.pipelines.stages.latent_preparation import LatentPreparationStage
from fastvideo.pipelines.stages.decoding import DecodingStage
from .stages.longcat_conditioning import LongCatConditioningStage
from .stages.longcat_denoising import LongCatDenoisingStage


class LongCatPipeline(ComposedPipelineBase):
    _required_config_modules = ["text_encoder", "tokenizer", "vae", "transformer", "scheduler"]

    def __init__(self, model_path: str, fastvideo_args):
        super().__init__(model_path, fastvideo_args)

    def initialize_pipeline(self, fastvideo_args):
        # LongCat 自带 scheduler，不替换为 FlowUniPC
        pass

    def create_pipeline_stages(self, fastvideo_args):
        self.register_stage(InputValidationStage)
        self.register_stage(TextEncodingStage)
        self.register_stage(LongCatConditioningStage)  # I2V/VC/Refine 条件注入
        self.register_stage(TimestepPreparationStage)
        self.register_stage(LatentPreparationStage)
        self.register_stage(LongCatDenoisingStage)     # LongCat 专用去噪
        self.register_stage(DecodingStage)
```

- 注册到构建器：`fastvideo/pipelines/pipeline_registry.py`

```python
from fastvideo.pipelines.basic.longcat.longcat_pipeline import LongCatPipeline

PIPELINE_CLASS_REGISTRY.update({
    ("LongCatPipeline", "video-generation"): LongCatPipeline,
})
```

### 7) 实现 LongCat 专用去噪阶段支持 T2V/I2V/VC/Refine

- 新增：`fastvideo/pipelines/basic/longcat/stages/longcat_denoising.py`

```python
import torch
from fastvideo.pipelines.stages.base import PipelineStage


class LongCatDenoisingStage(PipelineStage):
    name = "LongCatDenoisingStage"

    def run(self, batch, fastvideo_args):
        dit = batch.modules["transformer"]
        scheduler = batch.modules["scheduler"]
        prompt_embeds = batch.prompt_embeds
        prompt_attention_mask = batch.prompt_attention_mask

        timesteps = scheduler.timesteps
        do_cfg = batch.do_classifier_free_guidance

        latents = batch.latents
        dtype = dit.dtype

        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            latent_model_input = latent_model_input.to(dtype)

            # VC/I2V/Refine 的 timestep 按 LongCat 规则处理
            timestep = t.expand(latent_model_input.shape[0]).to(dtype)
            if batch.data_type in {"i2v", "vc", "refine"}:
                timestep = timestep.unsqueeze(-1).repeat(1, latent_model_input.shape[2])
                if batch.data_type != "vc" or not batch.use_kv_cache:
                    # 保护条件区
                    timestep[:, : batch.extra.get("num_cond_latents", 0)] = 0

            noise_pred = dit(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                num_cond_latents=batch.extra.get("num_cond_latents"),
                kv_cache_dict=batch.extra.get("kv_cache_dict"),
            )

            if do_cfg:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                B = noise_pred_cond.shape[0]
                pos = noise_pred_cond.reshape(B, -1)
                neg = noise_pred_uncond.reshape(B, -1)
                st_star = (pos * neg).sum(dim=1, keepdim=True) / (neg.pow(2).sum(dim=1, keepdim=True) + 1e-8)
                st_star = st_star.view(B, 1, 1, 1)
                noise_pred = noise_pred_uncond * st_star + batch.guidance_scale * (noise_pred_cond - noise_pred_uncond * st_star)

            # LongCat 为与调度器适配，取反
            noise_pred = -noise_pred

            if batch.data_type == "vc" and batch.use_kv_cache is False:
                num_cond = batch.extra.get("num_cond_latents", 0)
                latents[:, :, num_cond:] = scheduler.step(noise_pred[:, :, num_cond:], t, latents[:, :, num_cond:], return_dict=False)[0]
            elif batch.data_type == "i2v":
                latents[:, :, 1:] = scheduler.step(noise_pred[:, :, 1:], t, latents[:, :, 1:], return_dict=False)[0]
            else:
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        batch.latents = latents
        return batch
```

### 8) 实现 I2V/VC 条件编码与潜变量注入阶段

- 新增：`fastvideo/pipelines/basic/longcat/stages/longcat_conditioning.py`

```python
import math
import torch
from fastvideo.pipelines.stages.base import PipelineStage


class LongCatConditioningStage(PipelineStage):
    name = "LongCatConditioningStage"

    def run(self, batch, fastvideo_args):
        vae = batch.modules["vae"]
        data_type = batch.data_type  # "t2v" | "i2v" | "vc" | "refine"

        # 根据分辨率和 CP 切分计算对齐尺度
        scale_spatial = fastvideo_args.pipeline_config.vae_config.arch_config.spatial_compression_ratio * 2
        if batch.extra.get("cp_split_hw"):
            scale_spatial *= max(batch.extra["cp_split_hw"])

        if data_type == "i2v":
            # image -> latent 注入到第一帧
            cond = batch.preprocessed_image  # 由前置 stage 生成
            enc = vae.encode(cond.unsqueeze(2))  # [B, C, 1, H, W]
            latent = self._retrieve_latents(enc)
            latent = self._normalize_latents(latent, vae)
            # 覆盖条件区
            batch.latents[:, :, :1] = latent
            batch.extra["num_cond_latents"] = 1

        if data_type == "vc":
            num_cond_frames = batch.extra.get("num_cond_frames", 13)
            video = batch.preprocessed_video  # [B, T, C, H, W]
            video = video[:, -(num_cond_frames):].permute(0, 2, 1, 3, 4)
            enc = vae.encode(video)
            cond_latent = self._retrieve_latents(enc)
            cond_latent = self._normalize_latents(cond_latent, vae)

            num_cond_latents = 1 + (num_cond_frames - 1) // vae.config.scale_factor_temporal
            batch.latents[:, :, :num_cond_latents] = cond_latent
            batch.extra["num_cond_latents"] = num_cond_latents

        return batch

    def _retrieve_latents(self, encoder_output, sample_mode: str = "sample"):
        if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
            return encoder_output.latent_dist.sample()
        if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
            return encoder_output.latent_dist.mode()
        if hasattr(encoder_output, "latents"):
            return encoder_output.latents
        raise RuntimeError("invalid vae encode output")

    def _normalize_latents(self, latents, vae):
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        return (latents - latents_mean) * latents_std
```

### 9) 实现 Refine 上采样流程与 t_thresh 逻辑

- 在 `LongCatConditioningStage` 或新建 `longcat_refine.py` 中实现（示例以新文件）：
  - 新增：`fastvideo/pipelines/basic/longcat/stages/longcat_refine.py`

```python
import torch
import torch.nn.functional as F
from fastvideo.pipelines.stages.base import PipelineStage


class LongCatRefineStage(PipelineStage):
    name = "LongCatRefineStage"

    def run(self, batch, fastvideo_args):
        vae = batch.modules["vae"]
        stage1_video = batch.extra["stage1_video"]  # list[PIL.Image] 或 np.ndarray
        t_thresh = float(batch.extra.get("t_thresh", 0.5))

        # 变为张量 [T, C, H, W] -> [1, C, T, H, W]
        video = torch.from_numpy(stage1_video).permute(0, 3, 1, 2)
        video = video.to(device=batch.prompt_embeds.device, dtype=batch.prompt_embeds.dtype)
        video = video.permute(1, 0, 2, 3).unsqueeze(0) / 255.0

        # 双三次空间 + 三线性时间上采样
        height, width = batch.height, batch.width
        video_down = F.interpolate(video, size=(video.shape[2], height, width), mode="trilinear", align_corners=True)
        new_frames = video_down.shape[2] * (1 if batch.extra.get("spatial_refine_only") else 2)
        video_up = F.interpolate(video_down, size=(new_frames, height, width), mode="trilinear", align_corners=True)
        video_up = video_up * 2 - 1

        latent_up = self._retrieve_latents(vae.encode(video_up))
        latent_up = self._normalize_latents(latent_up, vae)
        latent_up = (1 - t_thresh) * latent_up + t_thresh * torch.randn_like(latent_up).contiguous()

        batch.latents = latent_up
        return batch

    def _retrieve_latents(self, encoder_output):
        return encoder_output.latent_dist.sample() if hasattr(encoder_output, "latent_dist") else encoder_output.latents

    def _normalize_latents(self, latents, vae):
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        return (latents - latents_mean) * latents_std
```

### 10) 扩展 ForwardBatch 以支持 LongCat 额外参数

- 文件：`fastvideo/pipelines/pipeline_batch_info.py`

```python
@dataclass
class ForwardBatch:
    # ... existing fields ...
    data_type: str = "t2v"  # t2v | i2v | vc | refine
    # LongCat-specific
    use_distill: bool = False
    use_kv_cache: bool = False
    offload_kv_cache: bool = False
    enhance_hf: bool = False
    num_cond_frames: int = 0
    t_thresh: float = 0.5
    spatial_refine_only: bool = False
    # caches and extras
    extra: dict[str, Any] = field(default_factory=dict)
```

### 11) 扩展 SamplingParam 增加 task 与 LongCat 标志位

- 文件：`fastvideo/configs/sample/base.py`（或 `.../wan.py` 旁新增 `longcat.py`）

```python
@dataclass
class SamplingParam:
    # ... existing fields ...
    task: str = "t2v"  # t2v | i2v | vc | refine | long
    resolution: str | None = None  # 480p | 720p
    use_distill: bool = False
    num_cond_frames: int = 13
    enhance_hf: bool = False
    use_kv_cache: bool = True
    offload_kv_cache: bool = False
    t_thresh: float = 0.5
    spatial_refine_only: bool = False
```

### 12) 扩展 generate CLI 支持任务类型与输入源

- 文件：`fastvideo/entrypoints/cli/generate.py`

```python
parser.add_argument("--task", type=str, default="t2v", choices=["t2v", "i2v", "vc", "refine", "long"])
parser.add_argument("--image-path", type=str, default=None)
parser.add_argument("--video-path", type=str, default=None)
parser.add_argument("--resolution", type=str, default=None)
parser.add_argument("--num-cond-frames", type=int, default=13)
parser.add_argument("--use-distill", action="store_true")
parser.add_argument("--use-kv-cache", action="store_true")
parser.add_argument("--offload-kv-cache", action="store_true")
parser.add_argument("--enhance-hf", action="store_true")
parser.add_argument("--t-thresh", type=float, default=0.5)

# cmd() 内拆分到 SamplingParam
generation_args.update(dict(
    task=args.task,
    image_path=args.image_path,
    video_path=args.video_path,
    resolution=args.resolution,
    num_cond_frames=args.num_cond_frames,
    use_distill=args.use_distill,
    use_kv_cache=args.use_kv_cache,
    offload_kv_cache=args.offload_kv_cache,
    enhance_hf=args.enhance_hf,
    t_thresh=args.t_thresh,
))
```

### 13) 支持直接从目录读取 LongCat 权重与校验

- 放宽校验：`fastvideo/utils.py::verify_model_config_and_directory`

```python
def verify_model_config_and_directory(model_path: str) -> dict:
    # ... existing code ...
    if not os.path.exists(model_index_json):
        # LongCat: 允许无 model_index.json，按子目录存在性推断
        subdirs = ["tokenizer", "text_encoder", "vae", "scheduler", "dit"]
        if all(os.path.isdir(os.path.join(model_path, s)) for s in subdirs):
            return {
                "_class_name": "LongCatPipeline",
                "workload_type": "video-generation",
                "tokenizer": {"_class_name": "AutoTokenizer"},
                "text_encoder": {"_class_name": "UMT5EncoderModel"},
                "vae": {"_class_name": "AutoencoderKLWan"},
                "scheduler": {"_class_name": "FlowMatchEulerDiscreteScheduler"},
                "transformer": {"_class_name": "LongCatVideoTransformer3DModel"},
            }
    # ... existing code ...
```

### 14) 编写权重转换脚本生成 FastVideo 目录结构

- 新增：`scripts/convert_longcat_to_fastvideo.py`

```python
import os, json, argparse, shutil

TEMPLATE = {
    "_class_name": "LongCatPipeline",
    "workload_type": "video-generation",
    "tokenizer": {"_class_name": "AutoTokenizer"},
    "text_encoder": {"_class_name": "UMT5EncoderModel"},
    "vae": {"_class_name": "AutoencoderKLWan"},
    "scheduler": {"_class_name": "FlowMatchEulerDiscreteScheduler"},
    "transformer": {"_class_name": "LongCatVideoTransformer3DModel"},
}

def main(src, dst):
    os.makedirs(dst, exist_ok=True)
    for sub in ["tokenizer", "text_encoder", "vae", "scheduler", "dit"]:
        if os.path.isdir(os.path.join(src, sub)):
            shutil.copytree(os.path.join(src, sub), os.path.join(dst, sub), dirs_exist_ok=True)
    with open(os.path.join(dst, "model_index.json"), "w") as f:
        json.dump(TEMPLATE, f, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    main(**vars(ap.parse_args()))
```

### 15) 接入 KV Cache 与可选 CPU offload

- 在 `LongCatDenoisingStage` 前，新增缓存构建逻辑：
  - 文件：`fastvideo/pipelines/basic/longcat/stages/longcat_kvcache.py`

```python
import torch
from fastvideo.pipelines.stages.base import PipelineStage


class LongCatKVCacheStage(PipelineStage):
    name = "LongCatKVCacheStage"

    def run(self, batch, fastvideo_args):
        if not batch.use_kv_cache:
            batch.extra["kv_cache_dict"] = {}
            return batch
        dit = batch.modules["transformer"]
        cond = batch.latents[:, :, : batch.extra["num_cond_latents"]]
        timestep = torch.zeros(cond.shape[0], cond.shape[2], device=cond.device, dtype=dit.dtype)
        empty = torch.zeros([cond.shape[0], 1, batch.max_sequence_length, dit.config.caption_channels], device=cond.device, dtype=dit.dtype)
        _, kv_cache = dit(hidden_states=cond, timestep=timestep, encoder_hidden_states=empty, return_kv=True, skip_crs_attn=True, offload_kv_cache=batch.offload_kv_cache)
        batch.extra["kv_cache_dict"] = kv_cache
        # 去噪前将非条件区取出
        batch.latents = batch.latents[:, :, batch.extra["num_cond_latents"]:]
        return batch
```

- 在 `LongCatPipeline.create_pipeline_stages` 中插入 `LongCatKVCacheStage` 于去噪前。

### 16) 接入可选 Block Sparse Attention 后端

- 方案 A：沿用 FastVideo 统一注意力后端选择，在 `LongCatDenoisingStage` 进入 Transformer 前读取 `FASTVIDEO_ATTENTION_BACKEND`。
- 方案 B：引入 LongCat 的 BSA 实现：
  - 拷贝 `LongCat-Video/longcat_video/block_sparse_attention/*` 至 `fastvideo/third_party/longcat/block_sparse_attention/`
  - 在 `LongCatVideoTransformer3DModel` 初始化时，暴露 `enable_bsa()/disable_bsa()` 并由 `pipeline_config.enable_bsa` 控制。

```python
# 伪代码：在 Loader 加载后
if config.enable_bsa and hasattr(dit, "enable_bsa"):
    dit.enable_bsa()
```

### 17) 接入 LongCat LoRA 装载与启用/禁用接口

- 在 `Executor` 或 `VideoGenerator` 暴露统一接口（参考 Wan LoRA）：

```python
def set_lora_adapter(self, lora_nickname: str, lora_path: str | None = None):
    self.executor.set_lora_adapter(lora_nickname, lora_path)

def enable_loras(self, names: list[str]):
    dit = self.executor.pipeline.modules["transformer"]
    for n in names:
        dit.enable_lora(n)

def disable_all_loras(self):
    dit = self.executor.pipeline.modules["transformer"]
    dit.disable_all_loras()
```

### 18) 新增单元与集成测试覆盖构建与一步推理

- 新增：`fastvideo/tests/test_longcat_pipeline.py`

```python
import pytest, torch
from fastvideo.entrypoints.video_generator import VideoGenerator


@pytest.mark.cuda
def test_build_and_one_step(tmp_path):
    model_path = "path/to/longcat-weights"
    gen = VideoGenerator.from_pretrained(model_path, num_gpus=1)
    out = gen.generate_video(prompt="a cat", height=480, width=832, num_frames=93, num_inference_steps=1, save_video=False)
    assert out["samples"].shape[-3] > 0
```

### 19) 新增 inference 脚本与文档，给出示例参数

- 新增：`scripts/inference/v1_inference_longcat.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

python -m fastvideo generate \
  --model-path /models/LongCat \
  --task t2v \
  --height 480 --width 832 --num-frames 93 \
  --num-inference-steps 50 --guidance-scale 4.0 \
  --use-kv-cache --num-cond-frames 13 \
  --prompt "A cute cat is running in the garden" \
  --output-path outputs_longcat/
```

### 20) 端到端验证各任务并对齐分辨率/帧数约束

- 验证点：
  - 输入尺寸满足 `scale_factor_spatial` 可整除（VAE 压缩 × CP 切分）。
  - `num_frames - 1` 能被 `vae_scale_factor_temporal` 整除；不满足时向下/向上就近对齐并打印警告。
  - T2V/I2V/VC/Refine 四任务均能以 1 步推理通过 smoke test；KV cache 下 VC 性能与显存符合预期。
  - `use_distill/enhance_hf` 互斥；Refine 的 `t_thresh`、`spatial_refine_only` 生效。
