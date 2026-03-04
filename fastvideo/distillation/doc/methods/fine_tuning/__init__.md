# `fastvideo/distillation/methods/fine_tuning/`

**目的**
- 以 “method” 的形式实现 finetuning / SFT：把它看作一种特殊的 distillation recipe（只有 `student` + dataset）。

**当前实现**
- `finetune.py`：`FineTuneMethod`
  - 只要求 `roles.student`
  - loss/policy 在 method 层
  - 复用同一套 trainer/roles/adapter/model plugin/validator/checkpoint 基础设施

**设计要点**
- adapter 仍保持 operation-centric（`prepare_batch / predict_* / backward`），不内置 finetune 的 loss 语义。
- model plugin 负责 build-time：加载 student modules、shared components（VAE/scheduler）、dataloader、validator。
