# `fastvideo/distillation/methods/consistency_model/__init__.py`

**状态**
- 当前是占位目录（`__all__ = []`），用于未来加入 Consistency Model（CM）相关方法。

**期望的演进方向**
- 通过 `@register_method("cm")`（示例）注册具体实现。
- method 只包含算法与 update policy；model plugin/adapter 提供运行时 primitives。
