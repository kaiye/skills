---
name: markdown-to-pdf
description: 将 Markdown 文件转换为 PDF，支持中文、表格、代码高亮。基于 md2pdf (WeasyPrint)，可自定义 CSS 样式。
homepage: https://github.com/jmaupetit/md2pdf
metadata:
  {
    "openclaw":
      {
        "emoji": "📄",
        "requires": { "bins": ["md2pdf"] },
        "install":
          [
            {
              "id": "pip",
              "kind": "pip",
              "package": "md2pdf[cli]",
              "bins": ["md2pdf"],
              "label": "Install md2pdf with CLI support",
              "flags": ["--break-system-packages"]
            },
          ],
      },
  }
---

# markdown-to-pdf

将 Markdown 文件转换为格式良好的 PDF，支持中文、表格、代码高亮、自定义样式。

## 快速开始

```bash
# 基础用法（使用默认样式）
md2pdf -i input.md -o output.pdf

# 使用自定义 CSS
md2pdf -i input.md -o output.pdf -c custom.css

# 启用 Markdown 扩展（代码高亮等）
md2pdf -i input.md -o output.pdf \
  --extras 'pymdownx.highlight' \
  --extras 'pymdownx.superfences'
```

## 在 OpenClaw 中使用

当用户要求"生成 PDF 报告"、"转成 PDF"、"导出 PDF"时，使用此 skill。

### 标准流程

1. 确认 Markdown 文件路径
2. 使用 skill 目录下的 `style.css`（已优化表格和间距）
3. 生成 PDF 到 `out/` 目录
4. 用 `message` tool 发送给用户

### 示例

```bash
cd /root/.openclaw/workspace/github-repos/kaiye/skills/markdown-to-pdf

# 使用优化后的样式
md2pdf \
  -i /root/.openclaw/workspace/out/report.md \
  -o /root/.openclaw/workspace/out/report.pdf \
  -c style.css
```

## 自定义样式

skill 目录下的 `style.css` 已针对中文报告优化：

- **表格第一列不换行**：`th:first-child, td:first-child { white-space: nowrap; }`
- **修复大段空白**：移除多余的 margin/padding
- **中文字体**：优先使用 Noto Sans CJK SC / Microsoft YaHei
- **打印友好**：A4 纸张，合理的页边距，避免表格跨页断裂

## 支持的 Markdown 扩展

常用扩展（通过 `--extras` 参数启用）：

- `tables`：表格支持（默认启用）
- `fenced_code`：代码块（默认启用）
- `pymdownx.highlight`：代码高亮
- `pymdownx.superfences`：增强代码块
- `pymdownx.emoji`：Emoji 支持
- `toc`：目录生成

## 注意事项

1. **中文支持**：确保系统安装了中文字体（Noto Sans CJK / Microsoft YaHei）
2. **表格宽度**：超宽表格会自动缩小字号，避免溢出
3. **代码块**：长代码会自动换行，不会超出页面
4. **图片**：支持本地图片和 URL，相对路径基于 Markdown 文件位置

## 故障排查

### 中文显示为方块

```bash
# Debian/Ubuntu
apt-get install fonts-noto-cjk

# 或手动指定字体
# 编辑 style.css，修改 font-family
```

### 表格溢出页面

- 减少列数
- 缩短单元格内容
- 使用 `font-size: 11px` 或更小

### 生成失败

```bash
# 检查 WeasyPrint 依赖
python3 -c "import weasyprint; print(weasyprint.__version__)"

# 重新安装
pip install --break-system-packages --upgrade 'md2pdf[cli]'
```

## 技术细节

- **引擎**：WeasyPrint（成熟的 HTML → PDF 渲染器）
- **Markdown 解析**：Python-Markdown + PyMdown Extensions
- **CSS 支持**：完整的 CSS3 支持（Flexbox、Grid 除外）
- **性能**：单个文件通常 < 5 秒

## 相关链接

- [md2pdf GitHub](https://github.com/jmaupetit/md2pdf)（408 stars）
- [WeasyPrint 文档](https://doc.courtbouillon.org/weasyprint/)
- [PyMdown Extensions](https://facelessuser.github.io/pymdown-extensions/)
