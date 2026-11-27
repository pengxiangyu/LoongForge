# LoongForge 具身模型适配 RSI 框架

面向 [`baidu-baige/LoongForge`](https://github.com/baidu-baige/LoongForge) 的 5 阶段具身/VLA 模型适配编排框架（`intake` → `anatomy` → `scaffold` → `migrate` → `align`），校验独立成层。验收不是第 6 个阶段，而是 `validate/accept_gate.py` 对四个门禁的判定。

## 架构与当前落地状态

| 模块 / 阶段 | 职责 | 状态 |
| --- | --- | --- |
| [`core/`](core/README.zh-CN.md) | 严格 Pydantic v2 契约（`intake` / `anatomy` / `stage` / `verdict` / `handoff`）、JSON Schema 导出及不可变包原子封存（`publish`/`verify`） | 已完成 |
| `core/orchestrator/` | 纯标准库 + PyYAML 轻量 DAG 编排内核（`graph`、原子 checkpoint 断点续跑/重规划 `scheduler`、传递失效 `invalidation`、`isolation`、确定性 `judger`） | 已完成 |
| `core/stages/f1_intake/` | Stage 1：Git commit/tree/递归 submodule 锁、离线 `git bundle` 打包恢复、固定树 AST 入口发现、只读资产分流（`WAITING_ASSET`/`blocked`）、只读隔离 probe、原子封存与 CLI | 已完成（CPU 测试 + 真实 `openpi` 演练通过） |
| `core/stages/f2_anatomy/` | Stage 2：强制复验 Stage 1 封存包、双独立子进程单步采集与重放、4 级权重快照对比、行为契约与观测计划、`FakeStepAdapter` 与 `OpenPIJaxAdapter` 骨架 | 离线闭环已完成（真机 GPU 待资产就绪） |
| `validate/` | 独立校验层，六个 checker 全部实装：`IntakeChecker` / `AnatomyChecker` 复验封存包，`ConstructChecker`（冻结 structure judge）、`ParityChecker`（冻结 numeric gate）、`TrainResumeChecker`（存档续训一致性）、`EvalChecker`（闭环评测报告）。`GateRunner` 聚合，`accept_gate.py` 判定验收 | 全部门禁已实装 |
| `core/stages/{f3_scaffold,f4_migrate,f5_align}/` | Stage 3 按 `TREE_SPEC.yaml` 生成骨架，Stage 4 用四种写入动作逐 slot 落原实现，Stage 5 驱动对比并路由判决 | 已完成 |
| `core/profiles/loongforge_embodied.yaml` | 声明式 5 阶段 DAG、LoongForge 6 大扩展点映射、3 个具身 Skill 与 5 个无状态确定性工具清单 | 已配置并校验 |

## 快速验证

```bash
python3 -m pytest -q
ruff check .
python3 -m compileall -q core validate tests
```

CLI 入口：

```bash
python3 -m core.stages.f1_intake.cli {init,analyze,freeze} --help
python3 -m validate.intake_check --help
python3 -m core.stages.f2_anatomy.cli {capture,freeze,replay} --help
python3 -m validate.anatomy_check --help
```

## 版本化分析与模型行为规格

- [AnatomySpec 与动作证据](docs/anatomy-spec.md)：采集前冻结相机/RNG/动作规格，支持连续 flow 与离散 token 的 CPU 格式闭环，保持通用硬门禁和独立双进程重放。
- [v2 Intake Skill Handoff](docs/intake-v2-handoff.md)：导入已完成的 Playbook 分析报告，明确 `handoff_only`（不启动 Agent Harness）；提供三侧软槽位、参数溯源与 `ScaffoldPlan` 映射。v1 task pack 继续兼容。

Intake/Anatomy validator 显式升版为 **2**。Spec release 完整内嵌、可搬迁，
复验/replay 必须由调用方提供外部可信 `spec_manifest_sha256`。
v1 schema 仍可加载；历史 `stages.verify.*` v1 sealed package 必须使用固定
旧 runtime，本次不宣称历史封存包直接复验兼容。Scaffold 结构明确区分
已验证 Anatomy 包与未验证的外部裸 SHA 引用。

顶层只保留 `core/`、`validate/`、`skills/`、`examples/`、`docs/`、`tests/`，无旧路径兼容副本。`core/__init__.py` 保持空，编排内核只依赖 stdlib + PyYAML。真实 GPU/模型验收未由 CPU 格式闭环代替。
