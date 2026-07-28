# 00. 执行摘要

vLLM Architecture Agent 是一个面向 VS Code Codex 的轻量 Agent Skill。它读取 vLLM 模型适配器 Python 文件，生成有源码证据、可编辑、少页高密度的 Draw.io 架构图。

## 解决的问题

直接阅读 `vllm/model_executor/models/*.py` 往往需要同时理解模型结构、并行策略、权重加载、外部 runtime 边界和 vLLM 接口。这个项目把这些信息整理成面向人的架构图和报告，而不是把源码简单画成类图。

## 为什么使用 Agent Skill

固定脚本很适合做机械工作，例如定位文件、索引 Class/Method、检查证据路径和验证输出。但“哪些内容值得画”“页面应该如何聚合”“外部行为如何解释”很难完全规则化，因此交给 Codex 判断。

职责分工是：

- 脚本：解析目标、完整索引、生成 checklist、验证 Plan/Evidence/Draw.io。
- Codex：阅读源码、理解模型、决定页面、绘制 Draw.io、进行视觉复查。
- Draw.io MCP：创建、编辑、打开和导出图。

## 为什么不是固定模型模板

当前示例是 HY V3，但生产脚本不硬编码 HY V3 的页面、节点或坐标。Skill 面向 vLLM 模型适配器目录设计，通过通用源码索引和 Agent 自主分析适配 dense、MoE、multimodal、embedding/pooling、hybrid/recurrent 和自定义 weight loading 等模型类型。

真实全目录兼容率需要在指定 vLLM commit 上执行 `vllm-arch scan` 验证，本仓库不声称自动覆盖所有当前和未来模型。

## 当前交付物

HY V3 Golden Example 位于 `examples/hy_v3/`，包含：

- `source-context.json`
- `architecture-plan.json`
- `evidence.json`
- `architecture.drawio`
- `report.md`
- `visual-review.md`
- 4 张 PNG 架构图

当前覆盖结果：

- Class：6 / 6 reviewed
- Method：20 / 20 reviewed
- Mapping Group：12 / 12 covered
- Capability：9 / 9 covered
- Evidence：14 direct, 4 derived, 2 external
- Visual Review：2 轮

## 推荐启动方式

安装后，在 VS Code Codex 中输入：

```text
使用 $vllm-model-architecture-diagram 分析 samples/hy_v3.py，生成默认架构图。
```
