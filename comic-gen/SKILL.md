---
name: comic-gen
description: 生成四格漫画（起承转合结构）。用户提供故事大纲，自动生成完整的四格漫画。内部调用 gemini-image-gen skill 进行图片生成。
allowed-tools: Read, Bash, Write
dependencies:
  - gemini-image-gen
---

# 四格漫画生成器

基于故事大纲自动生成四格漫画（起承转合结构），适合技术段子、日常吐槽、概念解释等场景。

## 功能特点

- **自动分镜**：根据故事大纲自动拆分为 4 个场景
- **对话优化**：自动精简对话，确保每个气泡不超过 20 字
- **风格一致**：所有角色在 Panel 1-3 保持一致（Panel 4 可以风格突变）
- **视觉元素**：自动添加表情、动作、符号（汗滴、问号、爆炸等）

## 使用方式

### 方式一：交互式创作

直接告诉 AI 你的故事想法，AI 会：
1. 帮你完善故事结构（起承转合）
2. 设计角色和场景
3. 优化对话
4. 生成漫画

示例：
```
我想画一个程序员没有 AI 就下班的故事，最后反转到未来战争
```

### 方式二：使用脚本（高级）

如果你已经有完整的故事大纲，可以直接用脚本生成：

```bash
bash $SKILL_DIR/scripts/gen-comic.sh \
  --output ./my-comic.png \
  --story-file ./story.yaml
```

## 故事大纲格式

使用 YAML 格式描述故事：

```yaml
title: "漫画标题"
style: "handdrawn"  # handdrawn（手绘风）或 realistic（写实风）
aspect_ratio: "16:9"

characters:
  - name: "角色A"
    description: "外观、性格描述"
  - name: "角色B"
    description: "外观、性格描述"

panels:
  panel_1:
    scene: "场景描述"
    characters: ["角色A", "角色B"]
    action: "动作描述"
    dialogue:
      - speaker: "角色A"
        text: "对话内容"
      - speaker: "角色B"
        text: "对话内容"
  
  panel_2:
    # 同上
  
  panel_3:
    # 同上
  
  panel_4:
    # 同上
```

完整示例见 `examples/programmer-war.yaml`

## 内部实现

1. **读取故事大纲** - 解析 YAML 或用户描述
2. **组装 prompt** - 根据 `gemini-image-gen` 的四格漫画模板组装
3. **调用生图** - 调用 `gemini-image-gen/scripts/gen-image.sh`
4. **输出结果** - 保存到 `out/` 目录

## 依赖

- `gemini-image-gen` skill（必须）
- `GEMINI_API_KEY` 环境变量（必须）
- SOCKS5 代理（国内必须，自动探测 1080/7890/7891 端口）

## 示例场景

- **技术段子**：程序员日常、AI 使用场景、技术选型纠结
- **概念解释**：抽象概念拟人化、技术演进历史
- **产品吐槽**：用户 vs 产品经理、需求变更四部曲
- **未来想象**：现在 vs 未来对比、科技发展预测

## 限制

- 每个气泡对话建议不超过 20 字（中文渲染限制）
- Panel 4 可以风格突变，但 Panel 1-3 角色需保持一致
- 生成时间约 30-60 秒（取决于 Gemini API 响应速度）
