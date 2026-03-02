# md2pdf

将 Markdown 文件转换为 PDF，支持中文、表格、代码高亮。基于 marked + Chrome headless，跨平台兼容。

## 触发场景

当用户需要将 Markdown 转换为 PDF 时使用，包括：
- "转成 PDF"、"生成 PDF"、"导出 PDF"
- "把这个 Markdown 转成 PDF"
- "生成报告的 PDF 版本"

## 依赖

- **Node.js**: 已安装
- **marked**: Markdown → HTML 转换器
- **puppeteer-core**: Chrome headless 控制库（不下载 Chromium）
- **Chrome/Chromium**: 系统已安装的浏览器

## 安装

```bash
cd ~/.openclaw/workspace/github-repos/kaiye/skills/md2pdf
npm install
npm link  # 全局安装命令行工具
```

## 使用

### 命令行

```bash
# 基本用法（输出同名 .pdf，竖版）
md2pdf input.md

# 指定输出路径
md2pdf input.md output.pdf

# 生成横版 PDF（适合宽表格）
md2pdf input.md --landscape
md2pdf input.md output.pdf --landscape

# 查看帮助
md2pdf --help
```

### 在 OpenClaw 中使用

当用户要求转换 Markdown 为 PDF 时：

```javascript
const { md2pdf } = require('./md2pdf.js');

// 竖版
await md2pdf('/path/to/input.md', '/path/to/output.pdf');

// 横版
await md2pdf('/path/to/input.md', '/path/to/output.pdf', { landscape: true });
```

或直接调用命令行：

```bash
md2pdf /root/.openclaw/workspace/out/report.md
md2pdf /root/.openclaw/workspace/out/report.md --landscape
```

## 样式

**无内置样式**，使用浏览器默认样式，保持简洁。

## 横版 vs 竖版

- **竖版（默认）**: 适合普通文档、文章
- **横版（--landscape）**: 适合宽表格、代码块、横向内容

## 跨平台支持

自动检测以下路径的 Chrome/Chromium：
- **Linux**: `/usr/bin/google-chrome`, `/usr/bin/chromium`
- **macOS**: `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome`
- **Windows**: `C:\Program Files\Google\Chrome\Application\chrome.exe`

如果未找到浏览器，会提示用户安装。

## 输出格式

- **纸张**: A4
- **边距**: 20mm（上下左右）
- **背景**: 打印背景色（代码块、表格等）

## 示例

```bash
# 转换调研报告（竖版）
md2pdf memu-vs-memos-report.md

# 转换宽表格报告（横版）
md2pdf wide-table-report.md --landscape

# 批量转换（需要 shell 循环）
for f in *.md; do md2pdf "$f"; done
```

## 注意事项

1. **临时文件**: 会生成 `.tmp.html` 临时文件，转换完成后自动删除
2. **Chrome 依赖**: 必须系统已安装 Chrome 或 Chromium
3. **文件路径**: 支持相对路径和绝对路径
4. **中文支持**: 依赖系统字体，Linux 需要安装中文字体（如 `fonts-noto-cjk`）
5. **Emoji 支持**: 需要安装 `fonts-noto-color-emoji`

## 故障排查

**错误: 未找到 Chrome/Chromium**
```bash
# Ubuntu/Debian
sudo apt install chromium-browser

# macOS
brew install --cask google-chrome

# Windows
# 从 https://www.google.com/chrome/ 下载安装
```

**PDF 中文显示为方块**
```bash
# Ubuntu/Debian
sudo apt install fonts-noto-cjk
```

**Emoji 显示为空白框**
```bash
# Ubuntu/Debian
sudo apt install fonts-noto-color-emoji
```

## 未来改进

- [ ] 支持自定义 CSS 样式（`--css style.css`）
- [ ] 支持 GitHub 风格主题（`--github-style`）
- [ ] 支持页眉页脚（`--header "标题"`）
- [ ] 批量转换（`md2pdf *.md`）
- [ ] 支持 YAML front matter（提取标题、作者等元数据）
- [ ] 自动检测表格宽度，智能选择横版/竖版
