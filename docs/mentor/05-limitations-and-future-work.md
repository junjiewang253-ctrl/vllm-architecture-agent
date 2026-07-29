# 05. 限制与后续工作

## 当前限制

1. **外部 runtime 内部行为不默认展开**

   对 vLLM Attention、FusedMoE、AutoWeightsLoader、LoRA runtime 等 imported component，除非 Codex 明确读取对应本地源码，否则只标记为 external boundary。

2. **HY V3 示例不是完整 vLLM checkout**

   `samples/hy_v3.py` 位于本仓库中，可完整索引目标文件，但没有绑定一个真实 vLLM registry checkout。因此示例中的 repository context 可以是 partial，而 target-file coverage 是 full。

3. **真实全目录兼容性需要本地 vLLM 仓库**

   本项目提供 `vllm-arch scan`，但不自动下载或克隆 vLLM。只有在指定真实 vLLM checkout 和 commit 后，才能声明该 commit 的扫描统计。

4. **图形布局由 Agent 判断**

   Draw.io 图由 Codex 通过 MCP 绘制和复查，语义与证据优先于像素级确定性。Plan 2.2 能约束锚点、关系和线型，但不能替代 Agent 的源码理解与视觉判断。相同源码的多次人工运行可能有轻微布局差异。

5. **Validator 不是图像识别器**

   当前 Draw.io validator 检查结构、visual anchor、关系端点、region 挂载、画布、PNG 和 XML 约束，不做完整像素级视觉分析，因此仍要求 Codex 打开 PNG 并进行真实修订。

6. **不声称支持所有未来模型**

   Skill 面向 `vllm/model_executor/models/*.py` 设计，具备通用索引和降级机制，但未来模型仍可能需要更新通用 pattern 或文档指南。

7. **一体化画布适合缩放阅读**

   默认大画布优先保证结构完整，README 缩略图只适合预览。细节阅读应使用 SVG 或可编辑 Draw.io。极端复杂模型允许在两轮布局修订后采用 `composite_pages`，但必须说明拆页原因和跨页顺序。

## 后续工作

- 在固定 vLLM commit 上定期运行全目录 scan，记录兼容率变化。
- 增加更多 portable Golden Example，例如 dense、multimodal、embedding/pooling、hybrid/recurrent。
- 增加可选 PNG 视觉 lint，用于发现明显文字重叠和边穿节点。
- 补充 Claude Code 集成包，但保持 canonical Skill 源目录不变。
- 为 mentor package 增加自动校验脚本，进一步减少手工复现步骤。
