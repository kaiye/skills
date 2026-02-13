# Agent Skills

Claude Code 自定义 Skills 集合。

## 可用 Skills

| Skill | 描述 |
|-------|------|
| [douyin-extract-text](./douyin-extract-text/) | 从抖音视频中提取文案。支持 OCR（硬字幕识别）和 ASR（语音识别）两种模式，并自动进行专业术语订正。 |
| [gemini-image-gen](./gemini-image-gen/) | 使用 Gemini 生成和编辑配图。支持手绘白板风和精致矢量图风两种预设模板，项目可自定义模板覆盖默认值。 |
| [wechat-publisher](./wechat-publisher/) | 将 Markdown 文章规范化处理后发布到微信公众号草稿箱，支持发布前预检、图片路径修复与封面图生成。 |
| [writing-team](./writing-team/) | 7 人 AI 写作团队编排。素材猎手、主笔、事实核查、风格审计、标题工匠协作完成长文创作，支持迭代修改和按需审核。 |

## 安装方法

将对应 skill 目录下的 `SKILL.md` 添加到 Claude Code 的 skills 配置中。

## 目录结构

```
skills/
├── .gitignore
├── README.md
├── douyin-extract-text/
│   ├── SKILL.md          # Skill 定义文件
│   ├── CORRECTION.md     # 术语订正词表
│   └── scripts/
│       └── douyin_video_processor.py
├── gemini-image-gen/
│   ├── SKILL.md          # Skill 定义文件
│   ├── templates/
│   │   └── default.yaml  # 默认风格模板
│   └── scripts/
│       └── gen-image.sh  # 图片生成脚本
├── wechat-publisher/
│   ├── SKILL.md          # Skill 定义文件（公众号发布流程）
│   └── references/
│       └── known-issues.md
└── writing-team/
    ├── SKILL.md          # Skill 定义文件（团队编排流程）
    └── references/       # Agent 角色定义
        ├── writing-researcher.md
        ├── writing-writer.md
        ├── writing-fact-checker.md
        ├── writing-voice-auditor.md
        └── writing-title-crafter.md
```
