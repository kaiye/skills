---
name: wechat-publisher
description: 将 Markdown 文章规范化处理后发布到微信公众号草稿箱。依赖 wenyan-mcp（MCP 工具）和 gemini-image-gen skill（封面图生成）。触发场景：用户说"发布到微信"、"发到公众号"、"推送文章"、"发布这篇文章"，或提供 markdown 文件要求发布到微信公众号时使用。
---

# WeChat Publisher

将 Markdown 文章经过预检、规范化、生成封面图后，通过 wenyan-mcp 发布到微信公众号草稿箱。

## 工作流

### 第一步：文章预检与规范化

读取文章内容，依次检查并修复以下问题：

**1. Frontmatter 检查**

必须包含 `title` 和 `cover`。若缺少 frontmatter：
- 从正文第一个 H1（`# 标题`）提取 title
- 添加 frontmatter 块

```yaml
---
title: "从正文提取的标题"
cover: "./cover.png"
---
```

**2. 去除重复标题**

若正文 H1 与 frontmatter title 相同，删除正文 H1，避免发布后出现两个标题。微信文章标题来自 frontmatter，正文应从 H2 开始。

**3. 图片路径修复**

wenyan-mcp 按文件系统路径读取图片，不做 URL 解码。逐一检查：

- 用 Glob 工具验证每个图片引用的文件是否存在
- 文件名含空格：重命名为连字符格式（`Pasted image 123.png` → `pasted-image-123.png`），更新引用
- 路径含 `%20` 编码：解码后查找实际文件，必要时在子目录（如 `attachments/`）中搜索
- 修正后再次确认文件存在

**4. 链接脚注确认**

wenyan-mcp 会将文章中**所有链接**（裸 URL 和 markdown 链接）自动转为文末脚注引用。发布前询问用户是否需要保留这些脚注：
- 保留 → 不处理
- 不保留 → 删除对应链接（纯文本替代或直接去掉）

### 第二步：封面图

若 frontmatter 缺少 cover 或指向的文件不存在：

1. 调用 gemini-image-gen skill 生成封面图
2. 比例 2.35:1（微信公众号封面推荐 900x383）
3. 基于文章标题和内容组装 prompt
4. 用 Read 工具查看生成结果，检查中文文字和构图
5. 保存到文章同目录，更新 frontmatter cover 字段

### 第三步：发布

```
使用 wenyan-mcp 的 publish_article 工具：
- file: 文章绝对路径
- theme_id: 用户选择的主题（默认 pie）
```

可选主题：default、orangeheart、rainbow、lapis、pie、maize、purple、phycat。

发布成功后提醒用户：
- 去公众号后台草稿箱预览排版效果
- 手动设置：摘要、阅读原文链接、作者、原创声明（wenyan-mcp 不支持这些字段）
- 清理之前的旧草稿（每次发布新增一篇，不覆盖）

## 已知限制

详见 [references/known-issues.md](references/known-issues.md)，遇到发布报错时查阅。
