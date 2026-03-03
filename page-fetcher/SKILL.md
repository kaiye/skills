---
name: page-fetcher
description: 智能网页抓取器（HTTP + Puppeteer）。统一使用 Sync Your Cookie 格式管理 cookie，支持 Cloudflare KV 自动同步。
allowed-tools: Bash, Read, Write, Grep, Glob
---

# page-fetcher

智能网页抓取器，支持轻量 HTTP 和 Puppeteer + Cookie 两种模式。

## 目录结构

```
page-fetcher/
├── scripts/
│   ├── read.js                  # 轻量 HTTP 抓取
│   ├── read-spa.js              # Puppeteer 抓取（需 cookie）
│   ├── sync-cookies-from-kv.js  # 从 Cloudflare KV 拉取 cookie
│   └── import-cookie.js         # 导入其他格式 cookie
├── cookies/
│   └── sync-your-cookie.json    # 统一 cookie 存储（Sync Your Cookie 格式）
├── pages/                       # 抓取结果缓存
└── rules.json                   # 域名规则（选择器 + UA）
```

## Cookie 管理

### 主路径：Sync Your Cookie 扩展（推荐）

1. **安装扩展**
   - Chrome Store: [Sync Your Cookie](https://chrome.google.com/webstore/detail/bcegpckmgklcpcapnbigfdadedcneopf)

2. **配置 Cloudflare KV**
   - 获取配置信息（见 `~/.openclaw/workspace/.secrets/cloudflare-kv.md`）
   - 在扩展设置中填入 Account ID、Namespace ID、API Token

3. **同步流程**
   - 本地浏览器登录目标网站
   - 点击扩展图标 → **Push**（上传到 Cloudflare KV）
   - VPS 执行拉取：
     ```bash
     node $SKILL_DIR/scripts/sync-cookies-from-kv.js
     ```

### 兜底路径：手动导入 Cookie

如果没有 Sync Your Cookie 扩展，可以通过其他方式导出 cookie 并导入：

```bash
node $SKILL_DIR/scripts/import-cookie.js <cookie-file>
```

**支持的格式**：
- Sync Your Cookie 原生格式（直接合并）
- 单域名格式 `{domain, cookies, localStorage}`（自动转换）

**自动云端同步**：
- 如果有 Cloudflare KV 配置 → 自动同步到云端
- 如果没有配置 → 只保存到本地

## 使用方式

### 轻量 HTTP 抓取

```bash
node $SKILL_DIR/scripts/read.js "https://example.com/..."
```

适用于静态页面，不执行 JS。

### Puppeteer 抓取（需 Cookie）

```bash
node $SKILL_DIR/scripts/read-spa.js "https://example.com/..."
```

自动从 `cookies/sync-your-cookie.json` 读取对应域名的 cookie。

## 配置规则

`rules.json` 示例：

```json
{
  "mp.weixin.qq.com": {
    "ua": "Mozilla/5.0 ...",
    "selector": ".rich_media_wrp",
    "strategy": "article"
  },
  "wx.zsxq.com": {
    "strategy": "article"
  }
}
```

- `ua`：可选，自定义 User-Agent
- `selector`：可选，CSS 选择器（仅支持 `.class` 和 `#id`）
- `strategy`：可选，html2md4llm 策略（`article` / `list`）

## 依赖

```bash
cd $SKILL_DIR
npm install puppeteer-core html2md4llm zod
```

## 输出

- **stdout**：Markdown 内容（带 frontmatter）
- **文件**：`pages/<domain>-<timestamp>.md`（Puppeteer 模式）

## Cookie 格式说明

统一使用 **Sync Your Cookie 格式**：

```json
{
  "updateTime": 1772508093149,
  "createTime": 1772508093149,
  "domainCookieMap": {
    "example.com": {
      "updateTime": 1772508093149,
      "createTime": 1772508093149,
      "cookies": [
        {
          "name": "...",
          "value": "...",
          "domain": "...",
          "path": "/",
          "secure": false,
          "httpOnly": false,
          "sameSite": "unspecified",
          "expirationDate": 1234567890
        }
      ],
      "localStorageItems": [
        {"key": "...", "value": "..."}
      ],
      "userAgent": "Mozilla/5.0 ..."
    }
  }
}
```

**优势**：
- 与 Sync Your Cookie 扩展完全兼容
- 所有域名集中管理，便于备份
- 支持 Cloudflare KV 自动同步

## 注意事项

- Puppeteer 需要系统已安装 Chrome/Chromium
- Cookie 文件统一为 `cookies/sync-your-cookie.json`
- 支持向后兼容：如果 `sync-your-cookie.json` 不存在，会 fallback 到旧格式 `<domain>_cookies.json`
- Cloudflare KV 免费额度：10万次读/天，足够个人使用
