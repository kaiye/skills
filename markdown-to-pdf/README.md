# markdown-to-pdf

将 Markdown 转换为格式良好的 PDF，专为中文技术报告优化。

## 特性

- ✅ 表格第一列不换行
- ✅ 修复大段空白
- ✅ 中文字体优化
- ✅ 代码高亮支持
- ✅ 打印友好（避免跨页断裂）

## 快速开始

```bash
# 方式 1: 使用快捷脚本
./convert.sh input.md output.pdf

# 方式 2: 直接调用 md2pdf
md2pdf -i input.md -o output.pdf -c style.css
```

## 文件说明

- `SKILL.md` - OpenClaw skill 定义
- `style.css` - 自定义样式（核心）
- `convert.sh` - 快捷转换脚本
- `README.md` - 本文件

## 依赖安装

```bash
pip install --break-system-packages 'md2pdf[cli]'
```

## 示例

```bash
# 转换技术报告
./convert.sh /root/.openclaw/workspace/out/report.md

# 输出到指定位置
./convert.sh report.md /tmp/output.pdf
```

## CSS 自定义

编辑 `style.css` 可调整：

- 字体、字号、行高
- 表格样式
- 代码块样式
- 页边距

## 技术栈

- **md2pdf**: Python Markdown → PDF 转换器
- **WeasyPrint**: HTML/CSS → PDF 渲染引擎
- **Python-Markdown**: Markdown 解析器

## 许可

MIT License
