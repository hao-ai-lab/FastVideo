# LongCat 推理脚本说明

本目录包含两个 LongCat 推理脚本，分别用于启用和禁用 BSA (Block Sparse Attention)。

## 脚本对比

| 脚本 | BSA 状态 | 分辨率 | 显存 | 速度 | 质量 | 适用场景 |
|-----|---------|--------|------|------|------|---------|
| `v1_inference_longcat.sh` | ❌ 禁用 | 480p | ~12 GB | 基准 | 100% | 480p 标准生成，质量对比 |
| `v1_inference_longcat_BSA.sh` | ✅ 启用 | 720p | ~18 GB | 1.4x | 98% | 720p+ 高分辨率生成 |

## 脚本详细说明

### 1. `v1_inference_longcat.sh` - 标准推理（无 BSA）

**特点**：
- 🔴 **禁用 BSA**：使用标准 Flash Attention
- 📐 **默认 480p**：480×832 分辨率
- 🎯 **最高质量**：没有稀疏化，注意力完整
- 💾 **适中显存**：~12 GB (480p)

**何时使用**：
- 480p 及以下分辨率生成
- 需要最高质量的场景
- 作为 BSA 的对比基准
- 显存充足的情况

**运行方式**：
```bash
bash scripts/inference/v1_inference_longcat.sh
```

**输出目录**：`outputs_video/longcat_no_bsa/`

---

### 2. `v1_inference_longcat_BSA.sh` - BSA 加速推理

**特点**：
- 🟢 **启用 BSA**：使用 Block Sparse Attention
- 📐 **默认 720p**：720×1280 分辨率
- ⚡ **速度提升**：约 1.4倍速度
- 💾 **节省显存**：~18 GB (720p，vs 无 BSA 的 ~24 GB)
- 🎨 **质量保持**：约 98% 质量

**BSA 配置**：
- **预设**：`720p-balanced`
- **稀疏度**：0.9375 (保留 6.25% 注意力)
- **块形状**：[4, 4, 4] (时间×高度×宽度)

**何时使用**：
- 720p 及以上分辨率
- 需要快速生成
- 显存受限的场景
- 批量生成任务

**运行方式**：
```bash
bash scripts/inference/v1_inference_longcat_BSA.sh
```

**输出目录**：`outputs_video/longcat_bsa/`

---

## 工作原理

### 自动配置机制

两个脚本都使用 `scripts/checkpoint_conversion/manage_bsa.py` 工具在运行前自动配置 BSA：

**v1_inference_longcat.sh (禁用 BSA)**：
```bash
# 在推理前执行
python scripts/checkpoint_conversion/manage_bsa.py "$CONFIG_FILE" --disable --no-backup
```

**v1_inference_longcat_BSA.sh (启用 BSA)**：
```bash
# 在推理前执行
python scripts/checkpoint_conversion/manage_bsa.py "$CONFIG_FILE" --preset 720p-balanced --no-backup
```

这样可以确保：
1. ✅ 每次运行都使用正确的配置
2. ✅ 不需要手动修改 config.json
3. ✅ 使用同一个模型权重目录
4. ✅ 方便快速切换和对比

### 配置文件不会永久改变

注意：脚本使用 `--no-backup` 标志，这意味着：
- 配置会立即应用到 `transformer/config.json`
- **下次运行另一个脚本时会被覆盖**
- 如果你想永久保存某个配置，去掉 `--no-backup` 标志

---

## 使用示例

### 场景 1: 快速生成 720p 视频

```bash
# 使用 BSA 快速生成
bash scripts/inference/v1_inference_longcat_BSA.sh

# 结果会保存在 outputs_video/longcat_bsa/
```

### 场景 2: 生成高质量 480p 视频

```bash
# 使用标准推理
bash scripts/inference/v1_inference_longcat.sh

# 结果会保存在 outputs_video/longcat_no_bsa/
```

### 场景 3: 对比 BSA 效果

```bash
# 先运行标准版本（基准）
bash scripts/inference/v1_inference_longcat.sh

# 再运行 BSA 版本（使用相同的 seed）
bash scripts/inference/v1_inference_longcat_BSA.sh

# 对比两个输出目录的结果
# - outputs_video/longcat_no_bsa/
# - outputs_video/longcat_bsa/
```

注意：为了公平对比，BSA 脚本需要改为 480p：
```bash
# 临时修改 v1_inference_longcat_BSA.sh 中的分辨率
--height 480 \
--width 832 \
```

---

## 自定义配置

### 修改 BSA 预设

在 `v1_inference_longcat_BSA.sh` 中修改预设：

```bash
# 720p 质量优先（更密集，更慢，质量更好）
python scripts/checkpoint_conversion/manage_bsa.py "$CONFIG_FILE" --preset 720p-quality --no-backup

# 720p 速度优先（更稀疏，更快，质量略降）
python scripts/checkpoint_conversion/manage_bsa.py "$CONFIG_FILE" --preset 720p-fast --no-backup

# 720p 自适应（使用 CDF 阈值）
python scripts/checkpoint_conversion/manage_bsa.py "$CONFIG_FILE" --preset 720p-adaptive --no-backup

# 长视频（>93 帧）
python scripts/checkpoint_conversion/manage_bsa.py "$CONFIG_FILE" --preset long-video --no-backup
```

可用预设：
- `480p`: 禁用 BSA
- `720p-balanced`: 平衡性能和质量 ⭐ (默认)
- `720p-quality`: 质量优先
- `720p-fast`: 速度优先
- `720p-adaptive`: 自适应
- `long-video`: 长视频优化

### 修改分辨率

**v1_inference_longcat.sh** (480p):
```bash
--height 480 \
--width 832 \
```

**v1_inference_longcat_BSA.sh** (720p):
```bash
--height 720 \
--width 1280 \
```

其他常见分辨率：
- 480p: 480×832
- 720p: 720×1280
- 1080p: 1080×1920 (需要更强的 GPU)

### 修改帧数

```bash
--num-frames 93 \  # 标准 (约 6 秒 @ 15fps)
--num-frames 65 \  # 较短 (约 4 秒 @ 15fps)
--num-frames 129 \ # 较长 (约 8 秒 @ 15fps)
```

### 修改推理步数

```bash
--num-inference-steps 50 \  # 标准质量
--num-inference-steps 30 \  # 快速生成
--num-inference-steps 100 \ # 最高质量
```

---

## 性能参考

基于 A100 80GB，相同 prompt 和 seed (42)：

### 480p (480×832×93 帧)

| 配置 | 显存 | 时间 | 质量 |
|-----|------|------|------|
| 无 BSA | 12.3 GB | 85s | 100% |
| BSA 0.9375 | 10.5 GB | 73s | 99% |

**结论**: 480p 下 BSA 提升有限，建议不使用。

### 720p (720×1280×93 帧)

| 配置 | 显存 | 时间 | 质量 |
|-----|------|------|------|
| 无 BSA | 24.3 GB | 180s | 100% |
| BSA 0.9375 | 18.2 GB | 130s | 98% |

**结论**: 720p 下 BSA 显著提升性能，强烈推荐。

---

## 验证 BSA 是否生效

运行后查看日志：

```bash
bash scripts/inference/v1_inference_longcat_BSA.sh 2>&1 | grep -i bsa
```

应该看到：
```
🔧 Configuring BSA (Block Sparse Attention)...
✅ BSA enabled with 720p-balanced preset
Enabling Block Sparse Attention (BSA) for LongCat transformer
BSA parameters: {'sparsity': 0.9375, ...}
```

---

## 故障排查

### 问题 1: 找不到 manage_bsa.py

**症状**：
```
bash: python: command not found
或
No such file or directory: tools/manage_bsa.py
```

**解决**：
```bash
# 确保在 FastVideo 根目录运行
cd /path/to/FastVideo
bash scripts/inference/v1_inference_longcat_BSA.sh

# 或者使用绝对路径
export FASTVIDEO_ROOT=/path/to/FastVideo
bash $FASTVIDEO_ROOT/scripts/inference/v1_inference_longcat_BSA.sh
```

### 问题 2: Config file not found

**症状**：
```
❌ Error: Config file not found at weights/longcat-native/transformer/config.json
```

**解决**：
1. 检查 `MODEL_BASE` 路径是否正确
2. 确认模型已正确下载和转换
3. 修改脚本中的 `MODEL_BASE` 变量

### 问题 3: BSA 没有生效

**排查步骤**：
```bash
# 1. 查看 config.json
cat weights/longcat-native/transformer/config.json | grep enable_bsa

# 2. 手动启用 BSA
python scripts/checkpoint_conversion/manage_bsa.py weights/longcat-native/transformer/config.json --status
python scripts/checkpoint_conversion/manage_bsa.py weights/longcat-native/transformer/config.json --enable

# 3. 运行推理时查看日志
bash scripts/inference/v1_inference_longcat_BSA.sh 2>&1 | tee inference.log
grep -i bsa inference.log
```

### 问题 4: 显存不足

**解决方案**：

1. 使用更激进的 BSA 配置：
```bash
# 在脚本中修改为 720p-fast
python tools/manage_bsa.py "$CONFIG_FILE" --preset 720p-fast --no-backup
```

2. 降低分辨率或帧数
3. 启用 CPU offload
4. 减少 batch size

---

## 高级用法

### 批量生成对比

创建对比脚本 `compare_bsa.sh`：

```bash
#!/bin/bash

# 设置不同的 seed 进行多次生成
for seed in 42 123 456 789; do
    echo "=== Generating with seed $seed ==="
    
    # 无 BSA
    sed -i "s/--seed .*/--seed $seed \\\\/" scripts/inference/v1_inference_longcat.sh
    bash scripts/inference/v1_inference_longcat.sh
    
    # 有 BSA
    sed -i "s/--seed .*/--seed $seed \\\\/" scripts/inference/v1_inference_longcat_BSA.sh
    bash scripts/inference/v1_inference_longcat_BSA.sh
done

echo "✅ All done! Compare results in:"
echo "  - outputs_video/longcat_no_bsa/"
echo "  - outputs_video/longcat_bsa/"
```

### 使用环境变量覆盖配置

```bash
# 临时使用不同的模型路径
MODEL_BASE=/path/to/other/model bash scripts/inference/v1_inference_longcat_BSA.sh

# 临时使用不同的 attention backend
FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA bash scripts/inference/v1_inference_longcat_BSA.sh
```

---

## BSA 工作原理

BSA 在 attention 层有自动判断：
```python
# fastvideo/third_party/longcat_video/modules/attention.py
if self.enable_bsa and shape[0] > 1:  # 只在多帧时触发
    x = flash_attn_bsa_3d(q, k, v, ...)
```

**与原始 LongCat 的差异**：
- **原始**: 手动调用 `pipe.dit.enable_bsa()`，只在 refinement 使用
- **当前**: 通过 config 自动启用，多帧时自动触发 ✅

## 相关文档

- **权重转换**: `../checkpoint_conversion/LONGCAT_WEIGHT_CONVERSION_README.md`
- **BSA 配置工具**: `../checkpoint_conversion/manage_bsa.py`

---

## 快速参考

```bash
# 查看 BSA 状态
python scripts/checkpoint_conversion/manage_bsa.py weights/longcat-native/transformer/config.json --status

# 480p 标准（无 BSA）
bash scripts/inference/v1_inference_longcat.sh

# 720p BSA 加速
bash scripts/inference/v1_inference_longcat_BSA.sh
```

