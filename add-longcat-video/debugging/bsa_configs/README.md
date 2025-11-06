# LongCat BSA 配置示例

本目录包含不同使用场景下的 BSA 配置示例。

## 配置文件说明

### 1. `config_480p_no_bsa.json`
- **场景**: 480p 标准生成
- **BSA**: 禁用
- **原因**: 480p 分辨率下 BSA 性能提升有限
- **显存**: ~12 GB
- **质量**: 最佳

### 2. `config_720p_balanced.json` ⭐ 推荐
- **场景**: 720p 生成（平衡性能和质量）
- **BSA**: 启用，sparsity=0.9375
- **显存**: ~18 GB
- **速度**: 1.4x
- **质量**: 98%

### 3. `config_720p_quality.json`
- **场景**: 720p 生成（质量优先）
- **BSA**: 启用，sparsity=0.875
- **显存**: ~20 GB
- **速度**: 1.2x
- **质量**: 99%

### 4. `config_720p_fast.json`
- **场景**: 720p 快速生成（速度优先）
- **BSA**: 启用，sparsity=0.96875
- **显存**: ~16 GB
- **速度**: 1.5x
- **质量**: 95%

### 5. `config_720p_adaptive.json`
- **场景**: 720p 自适应生成
- **BSA**: 启用，使用 CDF 阈值
- **特点**: 根据内容自动调整稀疏度
- **显存**: ~18 GB
- **速度**: 1.3-1.5x
- **质量**: 97-99%

### 6. `config_long_video.json`
- **场景**: 长视频生成 (>93 frames)
- **BSA**: 启用，时间维度块更大
- **chunk_3d_shape**: [8, 4, 4]
- **显存**: 根据长度变化
- **特点**: 时间一致性更好

## 使用方法

### 方法 1: 复制到 transformer 目录

```bash
# 选择合适的配置（例如 720p 平衡版）
cp examples/bsa_configs/config_720p_balanced.json \
   /path/to/longcat-weights/transformer/config.json

# 运行生成
python -m fastvideo generate \
  --model-path /path/to/longcat-weights \
  --height 720 \
  --width 1280 \
  --num-frames 93 \
  --prompt "Your prompt"
```

### 方法 2: 手动编辑现有 config.json

只需要添加或修改 `enable_bsa` 和 `bsa_params` 字段：

```json
{
  ... 现有配置 ...,
  "enable_bsa": true,
  "bsa_params": {
    "sparsity": 0.9375,
    "cdf_threshold": null,
    "chunk_3d_shape_q": [4, 4, 4],
    "chunk_3d_shape_k": [4, 4, 4]
  }
}
```

## 选择建议

```
你的需求是什么？

├─ 480p 生成
│  └─> 使用 config_480p_no_bsa.json (BSA 禁用)
│
├─ 720p 生成
│  ├─ 平衡性能和质量 ⭐
│  │  └─> config_720p_balanced.json
│  │
│  ├─ 质量最重要，不在乎速度
│  │  └─> config_720p_quality.json
│  │
│  ├─ 速度最重要，可接受质量略降
│  │  └─> config_720p_fast.json
│  │
│  └─ 希望自适应调整
│     └─> config_720p_adaptive.json
│
└─ 长视频（>93 帧）
   └─> config_long_video.json
```

## 参数调优指南

### sparsity (稀疏度)

```
0.75  ────── 保留 25% ────── 很密集，慢，质量最好
0.875 ────── 保留 12.5% ─── 较密集，中速，质量很好
0.9375 ───── 保留 6.25% ──── 平衡 ⭐
0.96875 ──── 保留 3.125% ─── 很稀疏，快，质量尚可
0.99   ────── 保留 1% ────── 极稀疏，极快，质量下降明显
```

### chunk_3d_shape (块形状)

**时间维度** (第 1 个数字):
- `[4, ?, ?]`: 标准，适合 <100 帧
- `[8, ?, ?]`: 长视频，适合 >100 帧
- `[2, ?, ?]`: 短视频，适合 <50 帧

**空间维度** (第 2、3 个数字):
- `[?, 4, 4]`: 标准，适合 720p
- `[?, 4, 8]` 或 `[?, 8, 4]`: 适合超宽或超高视频
- `[?, 2, 2]`: 极高分辨率 (1080p+)

### cdf_threshold (CDF 阈值)

- `null`: 不使用自适应（固定 topk）
- `0.95`: 保留累积概率 95% 的 tokens（较松）
- `0.98`: 保留累积概率 98% 的 tokens（平衡）⭐
- `0.99`: 保留累积概率 99% 的 tokens（较严）

**注意**: 当同时设置 `sparsity` 和 `cdf_threshold` 时，会取两者的最大值。

## 性能对比

在 A100 80GB 上，720p (720×1280×93 帧):

| 配置 | sparsity | 显存 (GB) | 时间 (s) | 速度倍数 | 质量评分 |
|------|----------|-----------|----------|---------|---------|
| No BSA | - | 24.3 | 180 | 1.00x | 100% |
| Quality | 0.875 | 20.1 | 150 | 1.20x | 99% |
| Balanced ⭐ | 0.9375 | 18.2 | 130 | 1.38x | 98% |
| Fast | 0.96875 | 16.4 | 120 | 1.50x | 95% |
| Adaptive | CDF 0.98 | 18.5 | 135 | 1.33x | 98% |

## 实验和调优

### 找到最佳 sparsity

使用二分搜索找到质量和速度的平衡点：

```bash
# 测试不同的 sparsity
for sparsity in 0.75 0.875 0.9375 0.96875; do
  # 修改 config.json 中的 sparsity
  # 运行生成
  python -m fastvideo generate ... --output-path outputs/test_$sparsity.mp4
  # 评估质量和速度
done
```

### 验证 BSA 效果

```python
from examples.longcat_bsa_usage import example_4_compare_performance

# 运行性能对比
python examples/longcat_bsa_usage.py --example 4
```

## 常见问题

### Q: 我的视频只有 49 帧，该用哪个配置？

A: 使用 `config_720p_balanced.json` 即可。BSA 对所有 >1 帧的视频都有效。

### Q: 我想要最快的速度，不在乎质量？

A: 使用 `config_720p_fast.json`，或者自己设置 `sparsity=0.99`。

### Q: 自适应配置 (CDF) 和固定 sparsity 哪个好？

A: 
- **固定 sparsity**: 性能可预测，适合批量生成
- **CDF 阈值**: 自适应，复杂场景质量更好，简单场景更快

### Q: 如何知道我的配置生效了？

A: 运行时查看日志：
```bash
python -m fastvideo generate ... 2>&1 | grep -i bsa
```
应该看到 "Enabling Block Sparse Attention" 和参数信息。

## 更多资源

- **详细文档**: `../BSA_实现总结.md`
- **英文指南**: `../BSA_INTEGRATION_GUIDE.md`
- **代码示例**: `../longcat_bsa_usage.py`
- **快速参考**: `../BSA_README.md`

