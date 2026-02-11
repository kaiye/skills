# Agent Skills

Claude Code 自定义 Skills 集合。

## 可用 Skills

| Skill | 描述 |
|-------|------|
| [douyin-extract-text](./douyin-extract-text/) | 从抖音视频中提取文案。支持 OCR（硬字幕识别）和 ASR（语音识别）两种模式，并自动进行专业术语订正。 |
| [gemini-image-gen](./gemini-image-gen/) | 使用 Gemini 生成和编辑配图。支持手绘白板风和精致矢量图风两种预设模板，项目可自定义模板覆盖默认值。 |

## 安装方法

将对应 skill 目录下的 `SKILL.md` 添加到 Claude Code 的 skills 配置中。

## 目录结构

```
skills/
├── README.md
├── douyin-extract-text/
│   ├── SKILL.md          # Skill 定义文件
│   ├── CORRECTION.md     # 术语订正词表
│   └── scripts/
│       └── douyin_video_processor.py
└── gemini-image-gen/
    ├── SKILL.md          # Skill 定义文件
    ├── templates/
    │   └── default.yaml  # 默认风格模板
    └── scripts/
        └── gen-image.sh  # 图片生成脚本
```
