### 设定：5×5 图片，用 2×2 的 tile（只看空间）

我用一个**完全具体**的例子走一遍，等价于：

- 把时间维设成 1：`dit_seq_shape = (T=1, H=5, W=5)`
- tile 设成：`tile_size = (1, 2, 2)`（时间 1，不切；空间是 2×2）

原始 flatten 顺序（按行）对应 5×5 网格：

```text
y\x  0  1  2  3  4
0    0  1  2  3  4
1    5  6  7  8  9
2   10 11 12 13 14
3   15 16 17 18 19
4   20 21 22 23 24
```

---

## 1. `tile_partition_indices`：原始 5×5 → 按 tile 顺序

`get_tile_partition_indices` 做的事情是：**按 tile 网格 (t,h,w-tile) 扫描，每个 tile 内再按行优先展开**。

这里 `T=1, H=5, W=5, ts=1, hs=2, ws=2`：

- 时间只有 1 个 tile
- 高方向有 `ceil(5/2)=3` 行 tile：`h_tile = 0,1,2`
- 宽方向有 `ceil(5/2)=3` 列 tile：`w_tile = 0,1,2`

每个 tile 是一个 (最多) 2×2 的小块；最后一行/列的 tile 可能不满。

逐块列举（只看真实 5×5 范围内的像素）：

1. tile (0,0)：覆盖 `y=0..1, x=0..1`  

   → 原始索引 `[0, 1, 5, 6]`

2. tile (0,1)：`y=0..1, x=2..3`  

   → `[2, 3, 7, 8]`

3. tile (0,2)：`y=0..1, x=4..4`（右边缺一列）  

   → `[4, 9]`

4. tile (1,0)：`y=2..3, x=0..1`  

   → `[10, 11, 15, 16]`

5. tile (1,1)：`y=2..3, x=2..3`  

   → `[12, 13, 17, 18]`

6. tile (1,2)：`y=2..3, x=4..4`  

   → `[14, 19]`

7. tile (2,0)：`y=4..4, x=0..1`  

   → `[20, 21]`

8. tile (2,1)：`y=4..4, x=2..3`  

   → `[22, 23]`

9. tile (2,2)：`y=4..4, x=4..4`  

   → `[24]`

按这个顺序把所有 tile flatten 拼起来：

```text
tile_partition_indices =
[ 0, 1, 5, 6,     # tile (0,0)
  2, 3, 7, 8,     # tile (0,1)
  4, 9,           # tile (0,2)
 10,11,15,16,     # tile (1,0)
 12,13,17,18,     # tile (1,1)
 14,19,           # tile (1,2)
 20,21,           # tile (2,0)
 22,23,           # tile (2,1)
 24 ]             # tile (2,2)
```

**含义：**

- 原始 flatten 0..24 按「tile 顺序」重排后，就是这个顺序。

---

## 2. `variable_block_sizes`：每个 tile 里真实 token 数

`construct_variable_block_sizes` 用 `dit_seq_shape` + `num_tiles` 计算每个 tile 的真实体素数。

- 对 H 方向（长度 5，tile 高 2，tile 个数 3）：
  - 前 2 个 tile：完整 2 行
  - 最后 1 个 tile：只剩 1 行  
  → `h_sizes = [2, 2, 1]`
- 对 W 方向同理：`w_sizes = [2, 2, 1]`
- T 方向这里 `t_sizes = [1]`

三者相乘得到 9 个 tile 的大小：

```text
tile (0,0): 1*2*2 = 4
tile (0,1): 1*2*2 = 4
tile (0,2): 1*2*1 = 2

tile (1,0): 4
tile (1,1): 4
tile (1,2): 2

tile (2,0): 2
tile (2,1): 2
tile (2,2): 1
```

也就是：

```text
variable_block_sizes = [4, 4, 2, 4, 4, 2, 2, 2, 1]
```

**含义：**

> 第 j 个 tile 里，有 `variable_block_sizes[j]` 个真实 token，最多占 2×2=4 个格子，其余是 pad。

---

## 3. `non_pad_index`：在「总 padded 轴」里的哪些位置是真实

这里 `max_block_size = tile 的容量 = 1*2*2 = 4`，每个 tile 占一段固定 4 格：

- tile 0 占 `[0,1,2,3]`
- tile 1 占 `[4,5,6,7]`
- tile 2 占 `[8,9,10,11]`
- ...
- tile 8 占 `[32,33,34,35]`  

总长度 = `9 * 4 = 36`。

`get_non_pad_index` 对每个 tile 只取前 `block_sizes[j]` 个位置是真实的：

- tile 0：size=4 → 真实位 `[0,1,2,3]`
- tile 1：size=4 → `[4,5,6,7]`
- tile 2：size=2 → `[8,9]`
- tile 3：size=4 → `[12,13,14,15]`
- tile 4：size=4 → `[16,17,18,19]`
- tile 5：size=2 → `[20,21]`
- tile 6：size=2 → `[24,25]`
- tile 7：size=2 → `[28,29]`
- tile 8：size=1 → `[32]`

拼起来：

```text
non_pad_index =
[ 0,1,2,3,
  4,5,6,7,
  8,9,
  12,13,14,15,
  16,17,18,19,
  20,21,
  24,25,
  28,29,
  32 ]
```

长度也是 25，对应所有真实像素在「总 padded 36 长度轴」上的位置，**每个 tile 都被映射到一段连续的 4 个位置**，只用前 `block_size` 个，其余是 pad。

---

## 4. 真正的 `tile` 映射长什么样

`tile` 做的是：

```python
x_padded[:, non_pad_index] = x[:, tile_partition_indices]
```

- 右边 `x[:, tile_partition_indices]`：  
  把原始 `[0..24]` 的像素按上面的 tile 顺序排：
  ```text
  [0,1,5,6, 2,3,7,8, 4,9, 10,11,15,16, 12,13,17,18, 14,19, 20,21, 22,23, 24]
  ```
- 左边 `x_padded[:, non_pad_index]`：  
  把这 25 个值依次写进上面那 25 个位置 `[0,1,2,3,4,5,6,7,8,9,12,13,...,32]`。

用「tile 粒度」看：

- tile 0：
  - 原始 `[0,1,5,6]`  
  - 写入 padded `[0,1,2,3]`
- tile 1：
  - 原始 `[2,3,7,8]`  
  - 写入 `[4,5,6,7]`
- tile 2：
  - 原始 `[4,9]`  
  - 写入 `[8,9]`，而 `[10,11]` 留着做 pad
- ...
- 最后总长是 36，只有上面这些 index 位置是真实，其它 index 是 0-pad。

**`untile` 反过来**：

```python
x = x[:, non_pad_index][:, reverse_tile_partition_indices]
```

- `x[:, non_pad_index]`：从长度 36 的 padded 向量里，抽回上述 25 个真实位置（顺序仍是 tile 顺序）。
- `reverse_tile_partition_indices`：把 tile 顺序反排回原始 `[0..24]` 的顺序。

---

这样，你可以把「真正 5×5 图」理解成：

- 先按 `(T,H,W)` flatten，序列长度 25；
- 再按 tile 顺序打散重排（`tile_partition_indices`）；
- 再塞进「9 个 tile × 每 tile 4 槽」的总 padded 轴里（`non_pad_index` 决定每个 tile 的局部位置和 pad 位置）；
- kernel 只看这个规则的「tile × 64 槽」世界，完全不用理会 5×5 的不规整边界。