# `fastvideo/distillation/methods/distribution_matching/__init__.py`

**定位**
- distribution matching 类方法的集合目录。

**当前实现**
- `DMD2Method`（见 `dmd2.md`）

**扩展**
- 新增方法时建议保持：
  - 算法逻辑在 method
  - model plugin 细节通过 adapter protocol 注入
  - 注册通过 `@register_method("<name>")`
