#!/usr/bin/env python3
"""
BSA 配置管理工具

这个工具帮助你快速启用、禁用和配置 LongCat transformer 的 BSA 功能。
"""

import argparse
import json
import os
import sys
from pathlib import Path


def read_config(config_path):
    """读取 config.json"""
    with open(config_path, 'r') as f:
        return json.load(f)


def write_config(config_path, config):
    """写入 config.json"""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ 配置已保存到: {config_path}")


def enable_bsa(config_path, sparsity=0.9375, cdf_threshold=None, 
               chunk_q=None, chunk_k=None, backup=True):
    """启用 BSA"""
    config = read_config(config_path)
    
    # 备份原配置
    if backup:
        backup_path = str(config_path) + '.bak'
        write_config(backup_path, config)
        print(f"📦 原配置已备份到: {backup_path}")
    
    # 设置 BSA
    config['enable_bsa'] = True
    
    # 设置 BSA 参数
    if chunk_q is None:
        chunk_q = [4, 4, 4]
    if chunk_k is None:
        chunk_k = [4, 4, 4]
    
    config['bsa_params'] = {
        'sparsity': sparsity,
        'cdf_threshold': cdf_threshold,
        'chunk_3d_shape_q': chunk_q,
        'chunk_3d_shape_k': chunk_k
    }
    
    write_config(config_path, config)
    print(f"\n✅ BSA 已启用!")
    print(f"  - sparsity: {sparsity}")
    print(f"  - cdf_threshold: {cdf_threshold}")
    print(f"  - chunk_3d_shape_q: {chunk_q}")
    print(f"  - chunk_3d_shape_k: {chunk_k}")


def disable_bsa(config_path, backup=True):
    """禁用 BSA"""
    config = read_config(config_path)
    
    # 备份原配置
    if backup:
        backup_path = str(config_path) + '.bak'
        write_config(backup_path, config)
        print(f"📦 原配置已备份到: {backup_path}")
    
    # 禁用 BSA
    config['enable_bsa'] = False
    config['bsa_params'] = None
    
    write_config(config_path, config)
    print("\n✅ BSA 已禁用!")


def show_status(config_path):
    """显示当前 BSA 状态"""
    config = read_config(config_path)
    
    print(f"\n📋 配置文件: {config_path}")
    print("=" * 60)
    
    enable_bsa = config.get('enable_bsa', False)
    bsa_params = config.get('bsa_params')
    
    if enable_bsa:
        print("✅ BSA: 已启用")
        if bsa_params:
            print(f"\n参数:")
            print(f"  - sparsity: {bsa_params.get('sparsity', 'N/A')}")
            print(f"  - cdf_threshold: {bsa_params.get('cdf_threshold', 'N/A')}")
            print(f"  - chunk_3d_shape_q: {bsa_params.get('chunk_3d_shape_q', 'N/A')}")
            print(f"  - chunk_3d_shape_k: {bsa_params.get('chunk_3d_shape_k', 'N/A')}")
        else:
            print("⚠️  警告: enable_bsa=true 但 bsa_params=null")
    else:
        print("❌ BSA: 已禁用")
    
    # 显示其他相关配置
    print(f"\n其他配置:")
    print(f"  - _class_name: {config.get('_class_name', 'N/A')}")
    print(f"  - enable_flashattn2: {config.get('enable_flashattn2', 'N/A')}")
    print(f"  - enable_flashattn3: {config.get('enable_flashattn3', 'N/A')}")
    print(f"  - enable_xformers: {config.get('enable_xformers', 'N/A')}")


def apply_preset(config_path, preset, backup=True):
    """应用预设配置"""
    presets = {
        '480p': {
            'enable': False,
            'description': '480p 标准（不使用 BSA）'
        },
        '480p-bsa': {
            'enable': True,
            'sparsity': 0.9375,
            'chunk_q': [4, 4, 4],  # 448×832×64: latent=(64,28,52), VSA同款
            'chunk_k': [4, 4, 4],
            'description': '480p BSA（448×832，VSA 同款分辨率）'
        },
        '704p-balanced': {
            'enable': True,
            'sparsity': 0.9375,
            'chunk_q': [4, 4, 4],  # 704×1280×96: latent=(96,44,80), 原始refinement配置
            'chunk_k': [4, 4, 4],
            'description': '704p 平衡（原始 LongCat refinement 参数）'
        },
        '704p-quality': {
            'enable': True,
            'sparsity': 0.875,
            'chunk_q': [4, 4, 4],
            'chunk_k': [4, 4, 4],
            'description': '704p 质量优先'
        },
        '704p-fast': {
            'enable': True,
            'sparsity': 0.96875,
            'chunk_q': [4, 4, 4],
            'chunk_k': [4, 4, 4],
            'description': '704p 速度优先'
        },
        '768p-balanced': {
            'enable': True,
            'sparsity': 0.9375,
            'chunk_q': [4, 4, 4],  # 768×1216×96: latent=(96,48,76), 48%4=0, 76%4=0
            'chunk_k': [4, 4, 4],
            'description': '768p 平衡（ASPECT_RATIO_960_F64 bucket）'
        },
        'long-video': {
            'enable': True,
            'sparsity': 0.9375,
            'chunk_q': [6, 4, 4],  # T=6 适配更多帧数
            'chunk_k': [6, 4, 4],
            'description': '长视频（>93 帧）'
        }
    }
    
    if preset not in presets:
        print(f"❌ 未知的预设: {preset}")
        print(f"\n可用预设:")
        for name, info in presets.items():
            print(f"  - {name}: {info['description']}")
        sys.exit(1)
    
    preset_config = presets[preset]
    print(f"📝 应用预设: {preset} - {preset_config['description']}")
    
    if preset_config['enable']:
        enable_bsa(
            config_path,
            sparsity=preset_config.get('sparsity', 0.9375),
            cdf_threshold=preset_config.get('cdf_threshold'),
            chunk_q=preset_config.get('chunk_q', [4, 4, 4]),
            chunk_k=preset_config.get('chunk_k', [4, 4, 4]),
            backup=backup
        )
    else:
        disable_bsa(config_path, backup=backup)


def restore_backup(config_path):
    """恢复备份"""
    backup_path = str(config_path) + '.bak'
    if not os.path.exists(backup_path):
        print(f"❌ 备份文件不存在: {backup_path}")
        sys.exit(1)
    
    # 读取备份
    config = read_config(backup_path)
    write_config(config_path, config)
    print(f"✅ 已从备份恢复: {backup_path}")


def main():
    parser = argparse.ArgumentParser(
        description="LongCat BSA 配置管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 查看当前状态
  python manage_bsa.py /path/to/model/transformer/config.json --status
  
  # 启用 BSA（默认 720p 平衡配置）
  python manage_bsa.py /path/to/model/transformer/config.json --enable
  
  # 启用 BSA（自定义参数）
  python manage_bsa.py /path/to/model/transformer/config.json --enable --sparsity 0.875
  
  # 禁用 BSA
  python manage_bsa.py /path/to/model/transformer/config.json --disable
  
  # 应用预设
  python manage_bsa.py /path/to/model/transformer/config.json --preset 720p-quality
  
  # 恢复备份
  python manage_bsa.py /path/to/model/transformer/config.json --restore
  
可用预设:
  - 480p: 480p 标准（不使用 BSA）
  - 720p-balanced: 720p 平衡（推荐）
  - 720p-quality: 720p 质量优先
  - 720p-fast: 720p 速度优先
  - 720p-adaptive: 720p 自适应
  - long-video: 长视频（>93 帧）
        """
    )
    
    parser.add_argument(
        'config_path',
        type=str,
        help='Transformer config.json 的路径'
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--status', '-s',
        action='store_true',
        help='显示当前 BSA 状态'
    )
    group.add_argument(
        '--enable', '-e',
        action='store_true',
        help='启用 BSA'
    )
    group.add_argument(
        '--disable', '-d',
        action='store_true',
        help='禁用 BSA'
    )
    group.add_argument(
        '--preset', '-p',
        type=str,
        help='应用预设配置'
    )
    group.add_argument(
        '--restore', '-r',
        action='store_true',
        help='从备份恢复'
    )
    
    # BSA 参数（用于 --enable）
    parser.add_argument(
        '--sparsity',
        type=float,
        default=0.9375,
        help='稀疏度 (默认: 0.9375)'
    )
    parser.add_argument(
        '--cdf-threshold',
        type=float,
        default=None,
        help='CDF 阈值 (默认: None)'
    )
    parser.add_argument(
        '--chunk-q',
        type=int,
        nargs=3,
        default=[4, 4, 4],
        help='chunk_3d_shape_q (默认: 4 4 4)'
    )
    parser.add_argument(
        '--chunk-k',
        type=int,
        nargs=3,
        default=[4, 4, 4],
        help='chunk_3d_shape_k (默认: 4 4 4)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='不创建备份'
    )
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config_path):
        print(f"❌ 配置文件不存在: {args.config_path}")
        sys.exit(1)
    
    # 执行操作
    if args.status:
        show_status(args.config_path)
    elif args.enable:
        enable_bsa(
            args.config_path,
            sparsity=args.sparsity,
            cdf_threshold=args.cdf_threshold,
            chunk_q=args.chunk_q,
            chunk_k=args.chunk_k,
            backup=not args.no_backup
        )
    elif args.disable:
        disable_bsa(args.config_path, backup=not args.no_backup)
    elif args.preset:
        apply_preset(args.config_path, args.preset, backup=not args.no_backup)
    elif args.restore:
        restore_backup(args.config_path)


if __name__ == '__main__':
    main()

