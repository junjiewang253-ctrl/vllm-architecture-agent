# 02. Skill 工作流

推荐用户输入：

```text
使用 $vllm-model-architecture-diagram 分析 samples/hy_v3.py，生成默认架构图。
```

用户不需要手工提供 repo root、输出目录、页面数量、Plan 路径、Evidence 路径或验证命令。

## 自动执行流程

1. **解析请求**
   - 识别目标文件 `samples/hy_v3.py`。
   - 推断 workspace root。
   - 生成 model slug `hy-v3`。
   - 准备输出目录 `outputs/hy-v3/`。

2. **准备 Source Context**
   - 调用 `vllm-arch prepare`。
   - 生成 `source-context.json`。
   - 生成空的 `architecture-plan.template.json` 和 `evidence.template.json`。
   - 注意：模板只提供 checklist，不包含模型答案。

3. **Codex 完整阅读源码**
   - 阅读目标 Python 文件。
   - 根据 Class Catalog 审阅所有 Class。
   - 根据 Method Catalog 审阅所有 Method。
   - 审阅 high / medium Branch。
   - 审阅 Mapping Group 和 Capability。
   - 不允许 silent omission。

4. **按需阅读相关文件**
   - 使用 `related_file_candidates` 作为参考。
   - 只在证明关键主张或确定 external boundary 时打开外部文件。
   - 对 vLLM Attention、FusedMoE、AutoWeightsLoader 等稳定外部组件，可以在 API 边界停止。

5. **编写 Architecture Plan**
   - 默认 complete detail level。
   - 默认 4 张复合图，最多 5 张。
   - 每页回答一个工程问题。
   - 每页包含 detail regions。
   - 所有 rendered class/method/branch/mapping/capability 映射到 page + region。

6. **编写 Evidence**
   - 每个核心架构主张必须有源码证据。
   - 使用仓库相对路径。
   - 区分 `direct`、`derived`、`external`。
   - external behavior 不得伪装成 direct。

7. **绘图前验证**
   - 执行 Plan 和 Evidence validator。
   - 验证失败时停止，不开始绘图。

8. **Draw.io MCP 绘图**
   - Codex 使用 Draw.io MCP 创建多页图。
   - 页面名必须与 Plan 一致。
   - 图中使用 component、container、panel、badge、boundary、port 等表达。

9. **PNG 导出和视觉复查**
   - 导出第一稿 PNG。
   - Codex 逐页查看。
   - 编写 `visual-review.md`。
   - 至少修改一轮，最多两轮。

10. **最终验证和报告**
    - 验证 Evidence、Plan、Draw.io 和 PNG。
    - 编写 `report.md`。
    - 向用户报告输出路径和验证结果。

## 脚本与 Codex 的边界

脚本只负责：

- target resolution；
- source context indexing；
- checklist generation；
- Plan/Evidence validation；
- Draw.io structure validation；
- model directory scan。

Codex 负责：

- 架构理解；
- 页面聚合；
- detail region 设计；
- Draw.io 布局；
- 视觉复查；
- 报告解释。

这就是 v2.x Agent-native 的核心：脚本保证事实边界，Agent 负责架构表达。
