# `fastvideo/distillation/methods/__init__.py`

**目的**
- 提供 method 层（算法层）的统一入口与可发现性。

**当前导出**
- `DistillMethod`：算法基类（抽象）
- `DMD2Method`：distribution matching 目录下的一个具体方法实现

**设计意图**
- method 层应当是 **模型无关** 的（不 import 具体 pipeline/模型实现）；
  任何 family 细节都通过 adapter primitives（protocol）注入。

**实现细节**
- 该模块对 `DMD2Method` 使用 lazy import（`__getattr__`），避免 registry/builder 在
  import 时触发循环依赖（circular import）。
