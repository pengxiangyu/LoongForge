# Validate

模型迁移工作区的校验模块目录。负责检查最终成果的准确性，包括逻辑是否一致、精度是否对齐等，确保迁移成果准确可用，是迁移成果合入前的最后一道关卡。

## 职责

- **逻辑一致性检查**：验证迁移后的模型在结构、数据流、推理路径上与开源原版保持一致。
- **精度对齐检查**：在相同输入与权重下对比迁移前后模型的输出，确认数值精度对齐（在允许误差范围内）。
- **可用性检查**：确认迁移成果可以在 LoongForge 内正常训练、评估与发布，产出可复现的结果。

## 目录内容

- 校验脚本：逻辑对比、精度对比、冒烟测试等检查实现。
- 校验配置：不同模型/任务的校验项与阈值配置。
- 校验报告：每次校验的执行结果与结论，作为合入依据。

## Implemented checker boundary

六个 checker 全部实装，各自只判一个问题：

- `IntakeChecker` / `AnatomyChecker` —— 独立复验各自的封存包。
- `ConstructChecker` —— slot 与目标文件的一一对应、必需 segment、权重覆盖；委托冻结的
  structure judge。
- `ParityChecker` —— 迁移前后的逐 step loss 偏差；委托冻结的 numeric gate。
- `TrainResumeChecker` —— 存档→续训是否改变了轨迹。边界之前的 step 必须完全一致（权重相同、
  采样顺序相同）；边界之后交给同一个冻结 numeric gate 判。
- `EvalChecker` —— 闭环评测报告是否完备（model factory / payload builder / action decoder
  三者都有证据）、每个 task 都有数值指标、且相对基线没有退化。它不跑仿真器，判的是报告。

`numeric_gate` 是数值判决的唯一定义处，`ParityChecker` 与 `TrainResumeChecker` 都调用它，
不各自再推一遍偏差。
`base.py` 提供 checker 协议与源码摘要助手，`gate_runner.py` 以 fail-closed 方式聚合，
`accept_gate.py` 判定最终验收。

复合 validator 哈希绑定 checker 源码路径、`validate/base.py`、
`core/orchestrator/judger.py` 以及声明的依赖源码 —— 包含 checker 委托的那个冻结 judge，
所以换上一个更宽松的 judge 会改变哈希。
v2 与模型相关检查见 `../docs/intake-v2-handoff.md` 与 `../docs/anatomy-spec.md`。

The composite validator hash binds checker source paths, `validate/base.py`,
`core/orchestrator/judger.py` and declared dependency sources.
See `../docs/intake-v2-handoff.md` and `../docs/anatomy-spec.md` for v2 and model-specific checks.

两个已实现的 validator 均显式升至 `validator_version="2"`。
Anatomy 复验必须由调用方传入外部可信 `spec_manifest_sha256`
（CLI：`--spec-manifest-sha256`），不能从待验包的 lock 自取信任值。
v1 task/behavior schema 仍可加载，新生成包使用 v2 校验凭据。
历史 `stages.verify.*` v1 sealed package 必须使用其固定旧 runtime；
本次未实现、未宣称历史封存包的直接复验兼容。
