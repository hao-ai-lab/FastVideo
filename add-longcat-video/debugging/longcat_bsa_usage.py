#!/usr/bin/env python3
"""
LongCat BSA (Block Sparse Attention) 使用示例

这个脚本展示了如何在 FastVideo 的 LongCat 实现中使用 BSA。
"""

import os
import sys

# 添加 FastVideo 到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastvideo.entrypoints.video_generator import VideoGenerator


def example_1_basic_usage():
    """示例 1: 基础使用 - 从预配置的权重加载"""
    print("=" * 80)
    print("示例 1: 基础使用 - BSA 在 config.json 中预配置")
    print("=" * 80)
    
    # 假设你的 transformer/config.json 中已经设置了:
    # {
    #   "enable_bsa": true,
    #   "bsa_params": {
    #     "sparsity": 0.9375,
    #     "chunk_3d_shape_q": [4, 4, 4],
    #     "chunk_3d_shape_k": [4, 4, 4]
    #   }
    # }
    
    model_path = "/path/to/longcat-weights-with-bsa"
    
    # 加载模型（BSA 会自动启用）
    generator = VideoGenerator.from_pretrained(
        model_path,
        num_gpus=1
    )
    
    # 生成 720p 视频
    output = generator.generate_video(
        prompt="A majestic eagle soaring through the mountains",
        height=720,
        width=1280,
        num_frames=93,
        num_inference_steps=50,
        guidance_scale=4.0,
        save_video=True,
        output_path="outputs/eagle_720p_bsa.mp4"
    )
    
    print(f"✅ 视频生成完成: {output['output_path']}")
    print(f"📊 生成时间: {output.get('generation_time', 'N/A')}")


def example_2_runtime_enable():
    """示例 2: 运行时启用 BSA"""
    print("\n" + "=" * 80)
    print("示例 2: 运行时动态启用 BSA")
    print("=" * 80)
    
    model_path = "/path/to/longcat-weights"
    
    # 加载模型
    generator = VideoGenerator.from_pretrained(model_path, num_gpus=1)
    
    # 获取 transformer 模块
    transformer = generator.executor.pipeline.get_module("transformer")
    
    # 检查是否支持 BSA
    if hasattr(transformer, 'enable_bsa'):
        print("✅ Transformer 支持 BSA")
        
        # 启用 BSA
        transformer.enable_bsa()
        print("✅ BSA 已启用")
        
        # 检查 BSA 参数
        if hasattr(transformer, 'blocks') and len(transformer.blocks) > 0:
            first_block_attn = transformer.blocks[0].attn
            print(f"📋 enable_bsa: {first_block_attn.enable_bsa}")
            print(f"📋 bsa_params: {first_block_attn.bsa_params}")
    else:
        print("❌ Transformer 不支持 BSA")
        return
    
    # 生成视频
    output = generator.generate_video(
        prompt="A serene lake at sunset with reflections",
        height=720,
        width=1280,
        num_frames=93,
        num_inference_steps=50,
        guidance_scale=4.0,
        save_video=True,
        output_path="outputs/lake_720p_bsa_runtime.mp4"
    )
    
    print(f"✅ 视频生成完成: {output['output_path']}")
    
    # 可选：禁用 BSA（下次生成时不使用）
    # transformer.disable_bsa()
    # print("BSA 已禁用")


def example_3_custom_config():
    """示例 3: 使用自定义 BSA 配置"""
    print("\n" + "=" * 80)
    print("示例 3: 自定义 BSA 配置")
    print("=" * 80)
    
    from fastvideo.configs.pipelines.longcat import (
        LongCatT2V480PConfig,
        LongCatDiTArchConfig
    )
    from fastvideo.configs.models import DiTConfig
    from dataclasses import dataclass, field
    
    # 创建自定义配置
    @dataclass
    class CustomBSAConfig(LongCatT2V480PConfig):
        enable_bsa: bool = True
        
        dit_config: DiTConfig = field(default_factory=lambda: DiTConfig(
            arch_config=LongCatDiTArchConfig(
                enable_bsa=True,
                bsa_params={
                    "sparsity": 0.90,  # 降低稀疏度以提高质量
                    "cdf_threshold": None,
                    "chunk_3d_shape_q": [4, 4, 8],  # 更大的空间块
                    "chunk_3d_shape_k": [4, 4, 8],
                }
            )
        ))
    
    # 注意：这种方式需要在创建 VideoGenerator 之前设置配置
    # 实际使用中，推荐在 config.json 中设置或使用运行时启用方式
    print("📋 自定义 BSA 配置:")
    print(f"  - sparsity: 0.90")
    print(f"  - chunk_3d_shape: [4, 4, 8]")
    print("\n提示: 要使用此配置，请在 transformer/config.json 中设置这些参数")


def example_4_compare_performance():
    """示例 4: 对比 BSA 开启和关闭的性能"""
    print("\n" + "=" * 80)
    print("示例 4: BSA 性能对比")
    print("=" * 80)
    
    import time
    import torch
    
    model_path = "/path/to/longcat-weights"
    generator = VideoGenerator.from_pretrained(model_path, num_gpus=1)
    transformer = generator.executor.pipeline.get_module("transformer")
    
    if not hasattr(transformer, 'enable_bsa'):
        print("❌ Transformer 不支持 BSA，跳过对比")
        return
    
    test_prompt = "A cat playing with a ball"
    test_config = {
        "prompt": test_prompt,
        "height": 720,
        "width": 1280,
        "num_frames": 93,
        "num_inference_steps": 20,  # 使用较少步数以加快测试
        "guidance_scale": 4.0,
        "save_video": False,
    }
    
    # 测试 1: 不使用 BSA
    print("\n🔄 测试 1: 不使用 BSA")
    transformer.disable_bsa()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    output_no_bsa = generator.generate_video(**test_config)
    
    time_no_bsa = time.time() - start_time
    mem_no_bsa = torch.cuda.max_memory_allocated() / 1024**3  # GB
    
    print(f"⏱️  时间: {time_no_bsa:.2f}s")
    print(f"💾 峰值显存: {mem_no_bsa:.2f} GB")
    
    # 清理显存
    torch.cuda.empty_cache()
    
    # 测试 2: 使用 BSA
    print("\n🔄 测试 2: 使用 BSA")
    transformer.enable_bsa()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    output_with_bsa = generator.generate_video(**test_config)
    
    time_with_bsa = time.time() - start_time
    mem_with_bsa = torch.cuda.max_memory_allocated() / 1024**3  # GB
    
    print(f"⏱️  时间: {time_with_bsa:.2f}s")
    print(f"💾 峰值显存: {mem_with_bsa:.2f} GB")
    
    # 对比结果
    print("\n📊 对比结果:")
    print(f"⚡ 速度提升: {time_no_bsa / time_with_bsa:.2f}x")
    print(f"💾 显存节省: {mem_no_bsa - mem_with_bsa:.2f} GB ({(1 - mem_with_bsa/mem_no_bsa)*100:.1f}%)")
    
    if time_with_bsa < time_no_bsa * 0.9:
        print("✅ BSA 带来了显著的性能提升！")
    else:
        print("ℹ️  BSA 在此配置下性能提升有限")


def example_5_adaptive_bsa():
    """示例 5: 根据分辨率自适应使用 BSA"""
    print("\n" + "=" * 80)
    print("示例 5: 自适应 BSA 策略")
    print("=" * 80)
    
    model_path = "/path/to/longcat-weights"
    generator = VideoGenerator.from_pretrained(model_path, num_gpus=1)
    transformer = generator.executor.pipeline.get_module("transformer")
    
    def generate_with_adaptive_bsa(prompt, height, width, **kwargs):
        """根据分辨率自适应启用 BSA"""
        
        # 策略：720p 及以上使用 BSA
        use_bsa = height >= 720
        
        if hasattr(transformer, 'enable_bsa'):
            if use_bsa:
                transformer.enable_bsa()
                print(f"✅ {height}p: 启用 BSA")
            else:
                transformer.disable_bsa()
                print(f"ℹ️  {height}p: 不使用 BSA（分辨率较低）")
        
        return generator.generate_video(
            prompt=prompt,
            height=height,
            width=width,
            **kwargs
        )
    
    # 生成不同分辨率的视频
    test_cases = [
        ("480p", 480, 832),
        ("720p", 720, 1280),
    ]
    
    for name, height, width in test_cases:
        print(f"\n🎬 生成 {name} 视频...")
        output = generate_with_adaptive_bsa(
            prompt="A beautiful sunrise over mountains",
            height=height,
            width=width,
            num_frames=49,
            num_inference_steps=30,
            guidance_scale=4.0,
            save_video=True,
            output_path=f"outputs/sunrise_{name}_adaptive.mp4"
        )
        print(f"✅ {name} 完成: {output['output_path']}")


def main():
    """主函数：运行所有示例"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LongCat BSA 使用示例")
    parser.add_argument(
        "--example",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="运行指定的示例 (1-5)，不指定则显示所有示例说明"
    )
    
    args = parser.parse_args()
    
    examples = {
        1: ("基础使用", example_1_basic_usage),
        2: ("运行时启用", example_2_runtime_enable),
        3: ("自定义配置", example_3_custom_config),
        4: ("性能对比", example_4_compare_performance),
        5: ("自适应策略", example_5_adaptive_bsa),
    }
    
    if args.example:
        # 运行指定示例
        name, func = examples[args.example]
        print(f"\n运行示例 {args.example}: {name}\n")
        func()
    else:
        # 显示所有示例说明
        print("LongCat BSA 使用示例")
        print("=" * 80)
        print("\n可用示例:")
        for num, (name, _) in examples.items():
            print(f"  {num}. {name}")
        print("\n运行方式:")
        print("  python longcat_bsa_usage.py --example <num>")
        print("\n示例:")
        print("  python longcat_bsa_usage.py --example 2  # 运行示例 2")
        print("\n注意: 运行前请修改 model_path 为你的实际模型路径")


if __name__ == "__main__":
    main()

