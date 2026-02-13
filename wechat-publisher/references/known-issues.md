# 已知限制与常见问题

## wenyan-mcp 字段限制

MCP 层调用 `publishToDraft` 只透传 `title`/`content`/`cover`。底层 `@wenyan-md/core` 的 `publishToWechatDraft` 支持 `author` 和 `source_url`，但 MCP 层没接上。

以下字段在 frontmatter 中写了也**不会生效**：
- `author` — 作者
- `source_url` — 阅读原文链接
- `digest` — 摘要
- 原创声明 — 微信 API 在「发布」而非「新建草稿」时设置

这些需要去公众号后台手动设置。

## 常见报错

### `invalid ip xxx, not in whitelist`

微信 API 校验调用方 IP。解决：
1. 登录微信开发者平台 https://developers.weixin.qq.com/console/index
2. 公众号 → 基础信息 → IP白名单
3. 添加报错中的 IP 地址

AppID、AppSecret 也在同一页面查看和管理。动态 IP 环境下每次变化都需更新白名单。

### `请通过参数或环境变量提供公众号凭据`

`.mcp.json` 中的 `WECHAT_APP_ID` 或 `WECHAT_APP_SECRET` 为空，或 MCP Server 启动时未加载到新凭据。

解决：在微信开发者平台（https://developers.weixin.qq.com/console/index）的公众号基础信息中获取凭据，填写到 `.mcp.json` 后重启 Claude Code 会话。

### `ENOENT: no such file or directory`

图片文件找不到。常见原因：
- 文件名含空格，wenyan-mcp 不做 URL 解码
- Obsidian 粘贴的图片路径带 `%20`，实际文件在 `attachments/` 子目录

解决：按第一步的图片路径修复流程处理。

### `Can't extract a valid title/cover`

frontmatter 缺少 `title` 或 `cover` 字段。按第一步添加 frontmatter。

## 草稿箱管理

每次 `publish_article` 在草稿箱新增一篇，不覆盖旧的。多次调试后记得提醒用户清理旧草稿。
