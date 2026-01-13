---
name: douyin-extract-text
description: 从抖音视频中提取文案。支持 OCR（硬字幕识别）和 ASR（语音识别）两种模式，并自动进行专业术语订正。当用户需要提取抖音视频文案、字幕、或处理抖音链接时使用。
allowed-tools: Read, Bash, Grep, Glob, Write
---

# 抖音视频文案提取

从抖音视频中提取文案，通过 ASR + OCR 双模式结合，实现高质量文案输出。

## 环境配置

ASR 模式需要阿里云百炼 API 密钥，在 shell 配置文件中设置：

```bash
# 添加到 ~/.zshrc 或 ~/.bashrc
export DASHSCOPE_API_KEY='your-api-key'
```

API 密钥获取：https://dashscope.console.aliyun.com/

## 工作流程（必须按顺序执行）

### 第一步：ASR 语音识别（主体）

**必须首先执行**，ASR 结果作为最终文案的主体（句子连贯）。

```bash
python scripts/douyin_video_processor.py "抖音分享链接" -o ./output
```

输出：`{video_id}_transcript.txt`

### 第二步：OCR 硬字幕识别（辅助）

**必须执行**，OCR 结果用于订正 ASR 中的专业术语错误。

```bash
python scripts/douyin_video_processor.py "抖音分享链接" --ocr -o ./output
```

输出：`{video_id}_transcript.txt`（会覆盖，建议先备份 ASR 结果）

**建议命令**（避免覆盖）：
```bash
# ASR 结果
python scripts/douyin_video_processor.py "链接" -o ./output
mv ./output/*_transcript.txt ./output/asr_result.txt

# OCR 结果
python scripts/douyin_video_processor.py "链接" --ocr -o ./output
mv ./output/*_transcript.txt ./output/ocr_result.txt
```

### 第三步：文案订正

以 ASR 结果为主体，参考 OCR 结果订正专业术语：

```
请基于语音识别（ASR）的效果，进行错别字和专业术语的订正。

订正规则：
1. ASR 结果是主体，保持其句子结构和连贯性
2. OCR 结果仅用于查找正确的专业术语拼写
3. 不要直接使用 OCR 结果，它可能包含视频画面中的无关文字

语音识别结果（ASR）：
[粘贴 asr_result.txt 内容]

OCR 识别结果（仅供参考）：
[粘贴 ocr_result.txt 内容]
```

常见订正词表见 [CORRECTION.md](CORRECTION.md)。

### 第四步：输出最终文案

将订正后的文案保存为 Markdown 文件，输出到用户当前工作目录：

**文件命名规则**：
- 使用视频标题作为文件名
- 格式：`{视频标题}.md`
- 去除标题中的特殊字符（`/\:*?"<>|`）

**Markdown 格式要求**：
```markdown
# {视频标题}

{订正后的完整文案，按内容分段}

---
来源：抖音视频
视频ID：{video_id}
原始链接：{抖音分享链接中的 URL}
```

### 第五步：更新订正词表（可选）

在输出最终文案后，列出本次订正的所有词汇：

```
本次订正列表：
| 原文（ASR） | 订正后 |
|-------------|--------|
| cloud code  | Claude Code |
| 麦塔        | Meta |
| ...         | ... |

是否将这些订正词更新到 CORRECTION.md？(y/n)
```

如果用户确认，将新的订正词追加到 [CORRECTION.md](CORRECTION.md) 对应的分类中。

**注意**：
- 只添加新的订正词，避免重复
- 按类别归类（AI产品、技术术语、项目名称等）
- 这样 skill 可以不断积累学习，提高后续订正的准确性

## 为什么需要双模式？

| 模式 | 优点 | 缺点 |
|------|------|------|
| ASR | 句子连贯完整 | 专业术语易错（cloud → Claude） |
| OCR | 专业术语准确 | 可能包含无关文字，句子不连贯 |

**结合使用**：ASR 提供连贯的句子结构，OCR 提供准确的术语拼写。

## 常用参数

| 参数 | 说明 |
|------|------|
| `--ocr` | 使用 OCR 模式 |
| `--local` | 处理本地视频 |
| `-o, --output` | 指定输出目录 |
| `--info-only` | 仅获取视频信息 |

## 依赖安装

```bash
# 必须安装（ASR 模式）
pip install requests dashscope
brew install ffmpeg

# 必须安装（OCR 模式）
pip install opencv-python numpy paddleocr paddlepaddle
```
