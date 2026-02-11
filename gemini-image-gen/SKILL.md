---
name: gemini-image-gen
description: 使用 Gemini 生成和编辑配图。支持手绘白板风和精致矢量图风两种预设模板，项目可自定义模板覆盖默认值。当用户需要为文章、文档生成插图、配图、概念图，或基于已有图片进行编辑时使用。
allowed-tools: Read, Bash, Grep, Glob, Write
---

# Gemini 配图生成与编辑

通过 Google Gemini API 生成和编辑文章配图，内置两种风格模板，项目可自定义扩展。

## 环境要求

脚本会自动从 `~/.zshrc`、`~/.zshenv`、`~/.bashrc`、`~/.profile` 中读取 `GEMINI_API_KEY`，无需手动 source。代理也会自动探测本地 SOCKS5 端口（1080/7890/7891）。

如需手动配置：

```bash
# 必须：Gemini API Key（https://aistudio.google.com/apikey）
# 通常已在 ~/.zshrc 中设置，脚本会自动读取
export GEMINI_API_KEY='your-key'

# 可选：代理配置（脚本会自动探测本地 SOCKS5 端口）
# 不需要代理时设置为 none
export GEMINI_PROXY='none'

# 可选：模型选择（默认 gemini-3-pro-image-preview）
export GEMINI_MODEL='gemini-2.5-flash-image'
```

## 工作流程

### 第一步：确定模板

读取提示词模板，优先级从高到低：

1. **项目自定义**：项目根目录的 `.gemini-image-templates.yaml`
2. **Skill 默认**：`$SKILL_DIR/templates/default.yaml`

```bash
# 检查项目是否有自定义模板
if [ -f ".gemini-image-templates.yaml" ]; then
  echo "使用项目自定义模板"
else
  echo "使用默认模板: $SKILL_DIR/templates/default.yaml"
fi
```

默认模板包含两种风格：

| 模板 | 名称 | 适用场景 |
|------|------|----------|
| `template_a_handdrawn` | 手绘白板风 | 概念解释、前后对比、简单流程、一句话总结 |
| `template_b_vector` | 精致矢量图风 | 技术概念、工具介绍、系统关系、抽象概念具象化 |

### 第二步：选择模板 + 组装 prompt

根据配图内容选择合适的模板。从 YAML 中取出 `style` 各字段，拼接为完整 prompt：

```
[style.base]
[style.color]
[style.typography]
[style.text_quality]
[style.composition]
[style.resolution]

Scene ([default_aspect_ratio] aspect ratio):
[用户描述的具体画面内容]

Chinese labels: [需要出现的中文文字]
```

**prompt 组装规则**：
- style 各字段原样拼接，作为风格约束（所有同模板的图共享）
- Scene 部分根据每张图的具体内容填写
- Chinese labels 单独列出所有需要渲染的中文，短于 6 字为佳
- 宽高比写入 Scene 描述中（如 "4:3 aspect ratio"），因为 REST API 不支持 imageGenerationConfig 参数

### 第三步：调用生成脚本

```bash
# 生成新图片（指定输出路径）
bash $SKILL_DIR/scripts/gen-image.sh -o <output_path> "<prompt>"

# 生成新图片（不指定路径，默认保存到当前目录 ./generated-YYYYMMDD-HHMMSS.png）
bash $SKILL_DIR/scripts/gen-image.sh "<prompt>"

# 编辑已有图片
bash $SKILL_DIR/scripts/gen-image.sh -i <input_image> -o <output_path> "<prompt>"
```

脚本说明：
- `-o` 输出路径，支持任意 png/jpg 路径，目录不存在会自动创建。省略时默认保存到当前工作目录
- `-i` 输入图片路径（可选），用于基于已有图片进行编辑
- 最后一个参数是第二步组装好的完整 prompt 文本
- 成功时 stdout 输出文件路径和大小，失败时 stderr 输出错误信息
- 默认使用 `gemini-3-pro-image-preview` 模型（中文文字渲染最好）
- **GEMINI_API_KEY 自动探测**：脚本会自动从 ~/.zshrc 等配置文件中提取，无需手动 export
- **代理自动探测**：脚本会自动检测本地 SOCKS5 端口（1080/7890/7891），无需手动配置

### 第四步：验证结果

生成后必须用 Read 工具查看图片，检查：
- [ ] 中文文字是否清晰、正确（无幻觉字符）
- [ ] 构图是否符合描述
- [ ] 风格是否与同批次其他图一致

如果中文有误，优先尝试：缩短文字（6字→2字）、换用更常见的词、调整文字位置到更空旷的区域。

## 批量生成示例

为一篇文章生成多张配图时的典型流程：

```bash
# 读取文章，找出所有 [配图建议: xxx] 标记
# 为每个标记选择模板、组装 prompt、调用脚本

STYLE_A="Hand-drawn whiteboard sketch illustration style. White background. Black ink hand-drawn line art..."
TEXT_RULES="CRITICAL: All Chinese characters must be sharp, crisp..."

# 配图1：手绘风
bash $SKILL_DIR/scripts/gen-image.sh -o ./images/01-concept.png "${STYLE_A} ${TEXT_RULES} Scene (4:3 aspect ratio): ..."

# 配图2：矢量风
bash $SKILL_DIR/scripts/gen-image.sh -o ./images/02-architecture.png "${STYLE_B} ${TEXT_RULES} Scene (4:3 aspect ratio): ..."
```

## 图片编辑

基于已有图片进行修改，用 `-i` 传入原图：

```bash
# 修改配色
bash $SKILL_DIR/scripts/gen-image.sh -i ./images/01-concept.png -o ./images/01-concept-v2.png "Change the accent color from red to blue, keep everything else the same"

# 添加元素
bash $SKILL_DIR/scripts/gen-image.sh -i ./images/02-arch.png -o ./images/02-arch-v2.png "Add a database icon on the right side connected with an arrow"

# 扩展画布
bash $SKILL_DIR/scripts/gen-image.sh -i ./hero.png -o ./hero-wide.png "Extend this image to 21:9 aspect ratio, expanding the background naturally on both sides"
```

编辑时的 prompt 不需要重复完整的风格模板，只需描述要做的修改。模型会保持原图的风格和内容，仅应用指定的变更。

## 项目自定义模板

在项目根目录创建 `.gemini-image-templates.yaml`，格式与默认模板一致：

```yaml
# .gemini-image-templates.yaml
template_a_handdrawn:
  name: "我的手绘风"
  style:
    base: >
      ...自定义风格描述...
    color: >
      ...自定义配色...
    # 其他字段同默认模板结构
  default_aspect_ratio: "16:9"

# 可以添加更多自定义模板
template_c_custom:
  name: "品牌风格"
  style:
    base: >
      ...
```

## 已知限制

1. **REST API 不支持 imageGenerationConfig**：宽高比、分辨率只能写入 prompt 文本，模型大多数时候会遵守
2. **中文渲染非 100%**：Gemini 3 Pro 在 4K 下约 94% 准确率，短文本好于长文本
3. **需要代理**：国内直连 Gemini API 会报 `User location is not supported`，需走 SOCKS5
4. **SynthID 水印**：所有生成图片自动包含不可见的 SynthID 水印，无法关闭

## 模型选择建议

| 需求 | 推荐模型 |
|------|----------|
| 中文文字清晰 | `gemini-3-pro-image-preview`（默认） |
| 快速迭代/草稿 | `gemini-2.5-flash-image` |
| 复杂场景/4K | `gemini-3-pro-image-preview` |
