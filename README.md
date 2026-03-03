# Agent Skills

Claude Code 自定义 Skills 集合。

## 可用 Skills

| Skill | 描述 |
|-------|------|
| [comic-gen](./comic-gen/) | 生成四格漫画（起承转合结构）。用户提供故事大纲，自动生成完整的四格漫画。内部调用 gemini-image-gen 进行图片生成。 |
| [douyin-extract-text](./douyin-extract-text/) | 从抖音视频中提取文案。支持 OCR（硬字幕识别）和 ASR（语音识别）两种模式，并自动进行专业术语订正。 |
| [gemini-image-gen](./gemini-image-gen/) | 使用 Gemini 生成和编辑配图。支持手绘白板风、精致矢量图风、四格漫画三种预设模板，项目可自定义模板覆盖默认值。 |
| [md2pdf](./md2pdf/) | 将 Markdown 文件转换为 PDF，支持中文、表格、代码高亮。基于 marked + Chrome headless，跨平台兼容。 |
| [page-fetcher](./page-fetcher/) | 智能网页抓取器（自动降级：HTTP → Puppeteer + Cookie）。支持 OpenClaw browser relay 自动提取 cookie，或通过 mcp-fetch-page 扩展手动提供。按 rules.json 配置选择器和策略。 |
| [wechat-publisher](./wechat-publisher/) | 将 Markdown 文章规范化处理后发布到微信公众号草稿箱，支持发布前预检、图片路径修复与封面图生成。 |
| [writing-team](./writing-team/) | 7 人 AI 写作团队编排。素材猎手、主笔、事实核查、风格审计、标题工匠协作完成长文创作，支持迭代修改和按需审核。 |

## 安装方法

将对应 skill 目录下的 `SKILL.md` 添加到 Claude Code 的 skills 配置中。

## 目录结构

```
skills/
├── .gitignore
├── README.md
├── comic-gen/
│   ├── SKILL.md          # Skill 定义文件
│   ├── examples/
│   │   └── programmer-war.yaml  # 示例故事大纲
│   └── scripts/
│       └── gen-comic.sh  # 漫画生成脚本（WIP）
├── douyin-extract-text/
│   ├── SKILL.md          # Skill 定义文件
│   ├── CORRECTION.md     # 术语订正词表
│   └── scripts/
│       └── douyin_video_processor.py
├── gemini-image-gen/
│   ├── SKILL.md          # Skill 定义文件
│   ├── templates/
│   │   ├── default.yaml       # 默认风格模板（手绘风 + 矢量风）
│   │   └── 4panel-comic.yaml  # 四格漫画模板
│   └── scripts/
│       └── gen-image.sh  # 图片生成脚本
├── md2pdf/
│   ├── SKILL.md          # Skill 定义文件
│   ├── md2pdf.js         # 主脚本（Node.js）
│   ├── package.json      # 依赖配置（marked, puppeteer-core）
│   └── package-lock.json
├── page-fetcher/
│   ├── SKILL.md          # Skill 定义文件（自动降级抓取策略）
│   ├── rules.json        # hostname→ua/selector/strategy 配置
│   ├── package.json      # Node 依赖（puppeteer-core, html2md4llm）
│   ├── cookies/          # Cookie 存储目录
│   ├── pages/            # 抓取结果缓存
│   └── scripts/
│       ├── fetch.js           # 主入口（自动降级）
│       ├── read.js            # 轻量 HTTP 抓取
│       ├── read-spa.js        # Puppeteer 抓取（需 cookie）
│       ├── extract-cookies.js # 从 browser relay 提取 cookie
│       └── save-cookie.js     # 保存用户上传的 cookie
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
