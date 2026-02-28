---
name: page-fetcher
description: 通用网页抓取器（带 UA 伪装 + 选择器抽取）。按 rules.json（hostname→selector/ua/strategy）抓取页面片段，并用 html2md4llm 转成干净 Markdown/JSON。
allowed-tools: Bash, Read, Write, Grep, Glob
---

# page-fetcher

一个更通用的“网页 → Markdown”利器。

## 配置

规则文件：`rules.json`

示例：

```json
{
  "mp.weixin.qq.com": {
    "ua": "Mozilla/5.0 ... Chrome/122 ...",
    "selector": ".rich_media_wrp",
    "strategy": "article"
  }
}
```

说明：
- `ua`：可选；不填则用默认桌面 Chrome UA
- `selector`：可选；目前仅支持 **`.class`** / **`#id`**（不支持组合选择器）
- `strategy`：可选；原样透传给 `html2md4llm`（如 `article` / `list`）

## 安装依赖

```bash
cd $SKILL_DIR
npm i
```

## 用法

URL → Markdown：

```bash
node $SKILL_DIR/scripts/read.js "https://example.com/..." > page.md
```

URL → JSON：

```bash
node $SKILL_DIR/scripts/read.js "https://example.com/..." --json > page.json
```

## 说明

- 本工具 **不会执行页面 JS**，仅处理抓到的静态 HTML。
- 对于 SPA/强动态页面，需要改走无头浏览器或直接调用站点 API。
