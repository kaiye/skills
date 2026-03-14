---
name: voice-article
description: 语音口述驱动的文章创作流程。用户通过发送语音录音提供素材，AI 转录、整理、多轮迭代，最终输出结构化 Markdown 文章。支持截图标注 review（红色=删除，黄色=修改）和封面图生成。适合 OpenClaw 等对话式 Agent 使用。触发场景：用户发送语音录音要写文章、说"根据这段录音整理成文章"、"帮我把这几段语音整理一下"，或发送截图要求按标注修改文章时使用。不适用于需要多 Agent 协作、素材调研、事实核查的长文创作（请用 writing-team skill）。
---

# voice-article

语音口述 → 结构化 Markdown 文章的完整创作流程。

## 配置

在使用前，确认以下配置（用户可自定义）：

```
OUTPUT_DIR: 文章输出目录（默认：当前工作目录/articles/）
ATTACHMENTS_DIR: 附件目录（默认：OUTPUT_DIR/attachments/）
```

文件命名规范：
- 文章：`<文章标题>.md`
- 封面图：`attachments/<文章标题>-cover.png`
- 临时文件（.bak、.draft-diff.json）：不提交到 git

## 流程

完整 SOP 见 [references/sop.md](references/sop.md)，共 7 步：

1. 素材收集（语音口述）
2. 初稿生成
3. 受众视角 review
4. 内容迭代（含精简 + 标题优化）
5. 封面图生成
6. 输出文章
7. 发布（可选，交由下游 skill 处理）

## 截图标注 review

用户可发送截图，用颜色标注修改意图：

| 颜色 | 含义 |
|------|------|
| 🔴 红色高亮/划线 | 删除该内容 |
| 🟡 黄色高亮/划线 | 修改该内容（需进一步说明改法，或由 AI 判断） |

收到截图后：
1. 识别所有颜色标注
2. 列出"红色：删除 xxx"、"黄色：修改 xxx"
3. 黄色标注若无说明，询问改法或给出建议
4. 确认后执行修改

## 封面图

调用 `gemini-image-gen` skill 生成，比例 2.35:1（适配公众号封面）。保存到 `attachments/<文章标题>-cover.png`，更新 frontmatter `cover` 字段。

## 输出格式

文章输出为带 frontmatter 的 Markdown：

```yaml
---
title: "文章标题"
cover: "./attachments/<文章标题>-cover.png"
---
```
