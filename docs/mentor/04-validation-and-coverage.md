# 04. 验证与覆盖

本项目的验证重点不是像素级锁定图的位置，而是保证：

- 目标 Python 文件中的源码结构被完整索引；
- Codex 没有静默遗漏 Class、Method、重要 Branch、Mapping Group 和 Capability；
- 主要架构主张有 Evidence；
- external behavior 不被伪装成 direct；
- Draw.io 页面与 Plan 一致；
- Golden Example 可复制到其他目录后继续验证。

## 当前 HY V3 Golden Example

位置：

```text
examples/hy_v3/
```

验证命令：

```powershell
vllm-arch validate `
  --repo-root . `
  --context examples\hy_v3\source-context.json `
  --plan examples\hy_v3\architecture-plan.json `
  --evidence examples\hy_v3\evidence.json `
  --drawio examples\hy_v3\architecture.drawio `
  --images-dir examples\hy_v3\images
```

当前结果：

- Evidence validation：passed
- Architecture plan validation：passed
- Draw.io validation：passed
- Relocatable example test：passed
- Absolute path scan：passed

## Source Coverage

HY V3 `source-context.json`：

- Classes indexed：6 / 6
- Methods indexed：20 / 20
- Module functions indexed：0 / 0
- Branches catalogued：37
- Weight Mapping Groups：12
- Detected Capabilities：9
- Unindexed nodes：0

## Plan Review Coverage

HY V3 `architecture-plan.json`：

- Class Review：6 / 6
- Method Review：20 / 20
- Branch Coverage：37 / 37
- Mapping Group Coverage：12 / 12
- Capability Coverage：9 / 9
- Unresolved：0
- Silent omission：0

Method 重要性不等于审阅完整性。v2.1.2 中，简单委托方法可以是 supporting；完整性由是否 reviewed 和是否有合理处置状态决定。

## Evidence Summary

HY V3 `evidence.json`：

- Direct claims：14
- Derived claims：4
- External claims：2

`direct` 必须由本地源码直接证明；`derived` 必须说明推导依据；`external` 必须说明本地源码证明了边界，但没有展开外部组件内部实现。

## Draw.io 和 PNG

当前导出 4 张 PNG：

- `model-architecture-and-execution.png`
- `decoder-and-attention.png`
- `moe-architecture-and-routing.png`
- `parallelism-configuration-and-weight-loading.png`

Draw.io validator 检查：

- 页面数量不超过 Plan budget；
- 页面标题与 Plan 完全一致；
- 每个 detail region title 出现在对应页面；
- PNG 文件存在；
- XML 可解析；
- 无 `html=1`；
- 节点尺寸合法；
- 背景为白色。

## Pytest

运行：

```powershell
pytest
```

测试覆盖：

- registry 静态解析；
- source context collector；
- method importance；
- mapping group aggregation；
- capability precision；
- Plan/Evidence validator；
- Draw.io validator；
- Golden Example relocatable validation；
- 中文 README 和 mentor setup 约束。

真实 vLLM 全目录 scan 只有在本地存在 vLLM checkout 时才会执行；否则相关集成测试会 skip。
