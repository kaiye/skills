#!/usr/bin/env node

/**
 * page-fetcher SPA 抓取脚本（Puppeteer + Cookie）
 * 从 mcp-fetch-page 提取核心逻辑，去掉 MCP 协议部分
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { URL } from 'url';
import os from 'os';
import puppeteer from 'puppeteer-core';
import { main as html2md4llm } from 'html2md4llm';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const SKILL_DIR = path.resolve(__dirname, '..');

const COOKIE_DIR = path.join(SKILL_DIR, 'cookies');
const PAGES_DIR = path.join(SKILL_DIR, 'pages');
const RULES_PATH = path.join(SKILL_DIR, 'rules.json');

function ensureDir(dir) {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

// 优先使用系统已安装的 Chrome，避免依赖 Puppeteer 管理的浏览器下载
function resolveSystemChromePath() {
  try {
    const candidates = [];
    const platform = process.platform;

    if (platform === 'darwin') {
      // 常见的 macOS 安装路径（稳定版 / Beta / Canary）
      candidates.push(
        '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
        '/Applications/Google Chrome Beta.app/Contents/MacOS/Google Chrome Beta',
        '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary'
      );
    } else if (platform === 'win32') {
      const programFiles = process.env['PROGRAMFILES'] || 'C:\\Program Files';
      const programFilesX86 = process.env['PROGRAMFILES(X86)'] || 'C:\\Program Files (x86)';
      const localAppData = process.env['LOCALAPPDATA'] || 'C:\\Users\\%USERNAME%\\AppData\\Local';
      candidates.push(
        path.join(programFiles, 'Google', 'Chrome', 'Application', 'chrome.exe'),
        path.join(programFilesX86, 'Google', 'Chrome', 'Application', 'chrome.exe'),
        path.join(localAppData, 'Google', 'Chrome', 'Application', 'chrome.exe')
      );
    } else {
      // Linux 常见路径
      candidates.push(
        '/usr/bin/google-chrome-stable',
        '/usr/bin/google-chrome',
        '/usr/bin/chromium',
        '/usr/bin/chromium-browser',
        '/snap/bin/chromium'
      );
    }

    for (const p of candidates) {
      try {
        if (fs.existsSync(p)) return p;
      } catch (_) {}
    }
    return null;
  } catch (_) {
    return null;
  }
}

// 加载域名规则配置
let domainRules = {};

function normalizeDomainRules(rawRules) {
  const normalized = {};
  if (!rawRules || typeof rawRules !== 'object') return normalized;

  for (const [domain, value] of Object.entries(rawRules)) {
    if (!domain) continue;
    if (typeof value === 'string') {
      normalized[domain] = {
        selector: value,
        blockedIfContains: [],
        ua: null,
        strategy: null
      };
      continue;
    }
    if (value && typeof value === 'object') {
      const selector = typeof value.selector === 'string' ? value.selector : null;
      const blockedIfContains = Array.isArray(value.blocked_if_contains)
        ? value.blocked_if_contains.filter(item => typeof item === 'string' && item.trim().length > 0)
        : [];
      const ua = typeof value.ua === 'string' && value.ua.trim().length > 0 ? value.ua : null;
      const strategy = typeof value.strategy === 'string' && value.strategy.trim().length > 0 ? value.strategy : null;
      normalized[domain] = { selector, blockedIfContains, ua, strategy };
    }
  }
  return normalized;
}

function loadDomainRules() {
  try {
    const raw = fs.existsSync(RULES_PATH)
      ? JSON.parse(fs.readFileSync(RULES_PATH, 'utf8'))
      : {};
    return normalizeDomainRules(raw);
  } catch (_) {
    return {};
  }
}

domainRules = loadDomainRules();

// 根据URL获取对应的域名规则
function getDomainRuleForUrl(url) {
  try {
    const urlObj = new URL(url);
    const hostname = urlObj.hostname.toLowerCase();

    // 精确匹配
    if (domainRules[hostname]) {
      return domainRules[hostname];
    }

    // 子域名匹配（去掉www等前缀）
    const mainDomain = hostname.replace(/^(www\.|m\.|mobile\.)/, '');
    if (domainRules[mainDomain]) {
      return domainRules[mainDomain];
    }

    // 部分匹配（查找包含的域名）
    for (const [domain, rule] of Object.entries(domainRules)) {
      if (hostname.includes(domain) || domain.includes(mainDomain)) {
        return rule;
      }
    }

    return { selector: null, blockedIfContains: [], ua: null, strategy: null };
  } catch (error) {
    return { selector: null, blockedIfContains: [], ua: null, strategy: null };
  }
}

class CookieManager {
  constructor() {
    this.cookiesCache = {};
  }

  // 列出所有cookie文件路径
  listAllCookieFiles() {
    if (!fs.existsSync(COOKIE_DIR)) {
      return [];
    }
    const files = fs.readdirSync(COOKIE_DIR);
    const result = [];
    for (const file of files) {
      // 仅匹配 *_cookies.json 及可能的重复命名 *_cookies (n).json
      if (/_cookies(\s*\(\d+\))?\.json$/i.test(file)) {
        result.push(path.join(COOKIE_DIR, file));
      }
    }
    return result;
  }

  // 从所有文件加载并合并cookie和localStorage（仅分域名）
  loadAndMergeAllCookies() {
    const files = this.listAllCookieFiles();
    if (files.length === 0) {
      return null;
    }

    const merged = {
      cookies: [],
      localStorageByDomain: {}
    };

    const seenKeys = new Set(); // 用于cookie去重：name|domain|path

    for (const filePath of files) {
      try {
        const data = this.loadCookiesFromFile(filePath);
        if (!data) continue;
        const filename = path.basename(filePath);
        // 从文件名提取来源域名: <domain>_cookies.json 或 <domain>_cookies (n).json
        let sourceDomain = null;
        const m = filename.match(/^(.*?)_cookies(\s*\(\d+\))?\.json$/i);
        if (m && m[1]) {
          sourceDomain = m[1].replace(/^www\./, '');
        }

        // 合并cookies
        const cookies = Array.isArray(data.cookies) ? data.cookies : [];
        for (const c of cookies) {
          if (!c || !c.name || !c.value || !c.domain) continue;
          const pathVal = c.path || '/';
          const key = `${c.name}|${c.domain}|${pathVal}`;
          if (seenKeys.has(key)) continue;
          seenKeys.add(key);
          merged.cookies.push({ ...c, path: pathVal });
        }

        // 合并localStorage到对应域（后读覆盖先读）
        if (data.localStorage && typeof data.localStorage === 'object' && sourceDomain) {
          if (!merged.localStorageByDomain[sourceDomain]) {
            merged.localStorageByDomain[sourceDomain] = {};
          }
          Object.assign(merged.localStorageByDomain[sourceDomain], data.localStorage);
        }
      } catch (err) {
        // 忽略单个文件解析错误
        continue;
      }
    }

    if (merged.cookies.length === 0 && Object.keys(merged.localStorageByDomain).length === 0) {
      return null;
    }
    return merged;
  }

  findCookieFile(domain) {
    const cleanDomain = domain.replace('www.', '');

    if (!fs.existsSync(COOKIE_DIR)) {
      return null;
    }

    // 读取目录中的所有文件
    const files = fs.readdirSync(COOKIE_DIR);

    // 生成可能的文件名模式（包括浏览器重命名的版本）
    const baseNames = [
      `${domain}_cookies`,
      `${cleanDomain}_cookies`,
      `www.${cleanDomain}_cookies`
    ];

    const matchingFiles = [];

    for (const file of files) {
      // 检查文件是否匹配任何基础名称模式
      for (const baseName of baseNames) {
        // 匹配原始文件名或带编号的重复文件名
        // 例如: example.com_cookies.json, example.com_cookies (1).json, example.com_cookies (2).json
        const pattern = new RegExp(`^${baseName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(\\s*\\(\\d+\\))?\\.json$`);

        if (pattern.test(file)) {
          const filePath = path.join(COOKIE_DIR, file);
          const stats = fs.statSync(filePath);

          matchingFiles.push({
            path: filePath,
            filename: file,
            modifiedTime: stats.mtime,
            baseName: baseName
          });
          break; // 避免同一个文件匹配多个baseName
        }
      }
    }

    if (matchingFiles.length === 0) {
      return null;
    }

    // 按修改时间降序排序，返回最新的文件
    matchingFiles.sort((a, b) => b.modifiedTime - a.modifiedTime);

    const latestFile = matchingFiles[0];

    // 如果有多个文件，在控制台输出信息
    if (matchingFiles.length > 1) {
      console.error(`📁 为域名 ${domain} 找到 ${matchingFiles.length} 个cookie文件:`);
      matchingFiles.forEach((file, index) => {
        const isLatest = index === 0 ? ' (最新)' : '';
        console.error(`   ${file.filename} - ${file.modifiedTime.toLocaleString()}${isLatest}`);
      });
      console.error(`🎯 选择最新文件: ${latestFile.filename}`);
    }

    return latestFile.path;
  }

  loadCookiesFromFile(filePath) {
    try {
      const content = fs.readFileSync(filePath, 'utf8');
      return JSON.parse(content);
    } catch (error) {
      console.error(`加载cookie文件失败 ${filePath}:`, error.message);
      return null;
    }
  }

  parseCookieData(cookieData) {
    try {
      return JSON.parse(cookieData);
    } catch (error) {
      throw new Error(`无效的cookie JSON格式: ${error.message}`);
    }
  }

  saveCookiesToFile(domain, cookieData) {
    if (!fs.existsSync(COOKIE_DIR)) {
      fs.mkdirSync(COOKIE_DIR, { recursive: true });
    }

    const cleanDomain = domain.replace('www.', '');
    const filePath = path.join(COOKIE_DIR, `${cleanDomain}_cookies.json`);

    fs.writeFileSync(filePath, JSON.stringify(cookieData, null, 2), 'utf8');
    console.error(`✅ Cookie已保存到: ${filePath}`);
  }

  isCookieExpired(cookieData) {
    try {
      if (!cookieData || !cookieData.cookies) return true;

      const now = new Date();
      let hasExpiredCookies = false;
      let expiredCount = 0;
      let totalWithExpiration = 0;
      const expiredCookieNames = [];

      // 检查每个Cookie的过期时间
      for (const cookie of cookieData.cookies) {
        if (cookie.expirationDate) {
          totalWithExpiration++;
          // expirationDate是Unix时间戳（秒），需要转换为毫秒
          const expireTime = new Date(cookie.expirationDate * 1000);
          if (now > expireTime) {
            hasExpiredCookies = true;
            expiredCount++;
            expiredCookieNames.push(cookie.name);
          }
        }
        // 如果Cookie没有过期时间，认为是会话Cookie，不检查过期
      }

      // 如果有设置过期时间的Cookie，并且其中有些已过期，则返回过期信息
      if (hasExpiredCookies && totalWithExpiration > 0) {
        console.error(`⚠️  检测到 ${expiredCount}/${totalWithExpiration} 个Cookie已过期:`);
        console.error(`   过期Cookie: ${expiredCookieNames.join(', ')}`);
        return true;
      }

      // 如果所有Cookie都没有过期时间，或都未过期，则认为有效
      return false;
    } catch (error) {
      console.error('Cookie过期检测失败:', error);
      return true;
    }
  }

  isCookieExpiredForDomain(cookieData, hostname) {
    try {
      if (!cookieData || !cookieData.cookies || !hostname) return false;

      const now = new Date();
      const cleanHost = String(hostname).toLowerCase().replace(/^www\./, '');
      let hasExpiredCookies = false;
      let expiredCount = 0;
      let totalWithExpiration = 0;
      const expiredCookieNames = [];

      for (const cookie of cookieData.cookies) {
        if (!cookie || !cookie.domain || !cookie.expirationDate) continue;
        const cookieDomain = String(cookie.domain).toLowerCase().replace(/^\./, '').replace(/^www\./, '');
        const matched = cleanHost === cookieDomain || cleanHost.endsWith(`.${cookieDomain}`);
        if (!matched) continue;

        totalWithExpiration++;
        const expireTime = new Date(cookie.expirationDate * 1000);
        if (now > expireTime) {
          hasExpiredCookies = true;
          expiredCount++;
          expiredCookieNames.push(cookie.name);
        }
      }

      if (hasExpiredCookies && totalWithExpiration > 0) {
        console.error(`⚠️  检测到当前域名 ${cleanHost} 的 ${expiredCount}/${totalWithExpiration} 个Cookie已过期:`);
        console.error(`   过期Cookie: ${expiredCookieNames.join(', ')}`);
        return true;
      }

      return false;
    } catch (error) {
      console.error('域名Cookie过期检测失败:', error);
      return false;
    }
  }

  cookiesToString(cookieData) {
    const cookies = [];
    for (const cookie of cookieData.cookies || []) {
      cookies.push(`${cookie.name}=${cookie.value}`);
    }
    return cookies.join('; ');
  }
}

// 保存页面内容到文件（成功或失败都保存）
function savePageContent(url, content, title, isError = false) {
  try {
    ensureDir(PAGES_DIR);

    const urlObj = new URL(url);
    const domain = urlObj.hostname;
    const timestamp = Date.now();
    const filename = `${domain}-${timestamp}.md`;
    const filePath = path.join(PAGES_DIR, filename);

    const textContent = content;
    fs.writeFileSync(filePath, textContent, 'utf8');
    return filePath;
  } catch (error) {
    console.error(`❌ 保存页面内容失败:`, error.message);
    return null;
  }
}

function toYamlPlainString(value) {
  return String(value ?? '').replace(/\r?\n/g, ' ').trim();
}

const cookieManager = new CookieManager();

// 处理SPA页面请求的函数（使用Puppeteer）
async function fetchPage(url, options = {}) {
  const { waitFor, timeout = 30000, headless = true } = options;

  if (!url) {
    throw new Error('URL parameter is required');
  }

  let browser = null;

  try {
    // 解析域名
    const urlObj = new URL(url);
    const domain = urlObj.hostname;

    // 获取cookie数据 - 自动从文件加载所有cookies
    let cookieData = null;

    // 自动合并所有cookie文件，解决短链/跨域跳转漏cookie问题
    const merged = cookieManager.loadAndMergeAllCookies();
    if (merged) {
      const hasExpired = cookieManager.isCookieExpiredForDomain(merged, domain);
      cookieData = merged;
      console.error(`已读取Cookie（合并 ${cookieData.cookies?.length || 0} 个${hasExpired ? '，包含过期项' : ''}）`);
    } else {
      cookieData = null;
      console.error('无Cookie');
    }

    // 启动Puppeteer浏览器，使用系统 Chrome（避免下载受管浏览器）
    const launchOptions = {
      headless: headless,
      defaultViewport: null, // 允许浏览器使用默认视口
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-blink-features=AutomationControlled',
        '--disable-features=VizDisplayCompositor',
        '--disable-extensions',
        '--disable-plugins',
        '--disable-sync',
        '--disable-translate',
        '--disable-default-apps',
        '--no-first-run',
        '--no-default-browser-check',
        '--disable-dev-shm-usage',
        '--disable-gpu',
        '--disable-web-security'
        // 移除 --no-zygote 和 --single-process 参数，这些会导致 frame detached 错误
      ]
    };

    // 直接写死系统 Chrome 路径（若存在），否则尝试使用 channel: 'chrome'
    const systemChrome = resolveSystemChromePath();
    if (systemChrome) {
      launchOptions.executablePath = systemChrome;
    } else {
      // 在 macOS/Windows 上，Puppeteer 可通过 channel 使用系统浏览器
      // 若仍未找到，将回退到默认行为（可能报未安装受管浏览器的错误）
      launchOptions.channel = 'chrome';
    }

    browser = await puppeteer.launch(launchOptions);

    const page = await browser.newPage();

    // 只在无头模式下设置视口大小
    if (headless) {
      await page.setViewport({
        width: 1366,
        height: 768,
        deviceScaleFactor: 1,
        hasTouch: false,
        isLandscape: true,
        isMobile: false,
      });
    }

    // 设置随机用户代理（可被规则覆盖）
    const userAgents = [
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
      'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
      'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15'
    ];
    const randomUserAgent = userAgents[Math.floor(Math.random() * userAgents.length)];
    const domainRule = getDomainRuleForUrl(url);
    const uaToUse = domainRule.ua || randomUserAgent;
    await page.setUserAgent(uaToUse);

    // 禁用自动化检测标志
    await page.evaluateOnNewDocument(() => {
      // 删除webdriver属性
      Object.defineProperty(navigator, 'webdriver', {
        get: () => undefined,
      });

      // 修改plugins长度
      Object.defineProperty(navigator, 'plugins', {
        get: () => [1, 2, 3, 4, 5],
      });

      // 修改语言设置
      Object.defineProperty(navigator, 'languages', {
        get: () => ['en-US', 'en'],
      });

      // 删除自动化控制标志
      delete Object.getPrototypeOf(navigator).webdriver;

      // 覆盖权限查询
      const originalQuery = window.navigator.permissions.query;
      window.navigator.permissions.query = (parameters) => (
        parameters.name === 'notifications' ?
          Promise.resolve({ state: Notification.permission }) :
          originalQuery(parameters)
      );

      // 模拟真实的Chrome运行时
      Object.defineProperty(window, 'chrome', {
        get: () => ({
          runtime: {},
          loadTimes: function() {},
          csi: function() {},
          app: {}
        }),
      });
    });

    // 使用正确的browser.setCookie API设置cookies（带SameSite映射与健壮性）
    if (cookieData && cookieData.cookies && cookieData.cookies.length > 0) {
      try {
        const mapSameSite = (val) => {
          if (!val) return null;
          const lower = String(val).toLowerCase();
          if (lower === 'lax') return 'Lax';
          if (lower === 'strict') return 'Strict';
          if (lower === 'none' || lower === 'no_restriction') return 'None';
          if (lower === 'unspecified' || lower === 'default') return null;
          return null;
        };

        const context = page.browserContext();
        const cookiesToSet = [];
        for (const cookie of cookieData.cookies) {
          if (!cookie || !cookie.name || !cookie.value || !cookie.domain) continue;
          const entry = {
            name: cookie.name,
            value: cookie.value,
            domain: cookie.domain,
            path: cookie.path || '/',
            secure: !!cookie.secure,
            httpOnly: !!cookie.httpOnly
          };
          const mapped = mapSameSite(cookie.sameSite);
          if (mapped) entry.sameSite = mapped;
          if (cookie.expirationDate) entry.expires = cookie.expirationDate;
          cookiesToSet.push(entry);
        }
        if (cookiesToSet.length > 0) {
          await context.setCookie(...cookiesToSet);
          console.error(`已设置 ${cookiesToSet.length} 个Cookie`);
        }
      } catch (error) {
        // 静默处理cookie设置错误（避免泄露敏感信息），但保留简要计数
      }
    } else {
    }

    // 在导航之前设置localStorage（按域名作用域写入，避免污染其他域）
    if (cookieData && cookieData.localStorageByDomain && Object.keys(cookieData.localStorageByDomain).length > 0) {
      await page.evaluateOnNewDocument((byDomain) => {
        try {
          const host = (location.hostname || '').replace(/^www\./, '');
          const candidates = [];
          for (const domain of Object.keys(byDomain)) {
            const d = String(domain).replace(/^www\./, '');
            if (host === d || host.endsWith('.' + d)) {
              candidates.push(d);
            }
          }
          for (const d of candidates) {
            const bucket = byDomain[d] || {};
            for (const [k, v] of Object.entries(bucket)) {
              try { window.localStorage.setItem(k, v); } catch (e) {}
            }
          }
        } catch (e) {
          // 忽略localStorage错误
        }
      }, cookieData.localStorageByDomain);
    }

    // 导航到目标页面（添加更多错误处理）
    let response;
    let finalUrl = url;
    try {
      response = await page.goto(url, {
        waitUntil: 'domcontentloaded',
        timeout: timeout
      });
      finalUrl = response?.url?.() || page.url() || url;

      // 检查页面是否正常加载
      if (response.status() >= 400) {
        throw new Error(`HTTP ${response.status()}: ${response.statusText()}`);
      }

    } catch (error) {
      throw new Error(`页面导航失败: ${error.message}`);
    }

    // 等待JavaScript执行完成
    try {
      await new Promise(r => setTimeout(r, 500));
      if (!page.isClosed()) {
        const readyState = await page.evaluate(() => document.readyState).catch(() => 'unknown');
        if (readyState !== 'complete') {
          await page.waitForFunction(() => document.readyState === 'complete', { timeout: 10000 }).catch(() => {});
        }
      }
    } catch (error) {
      // 继续执行，不抛出异常
    }

    // 提取目标规则：优先用户参数，其次域名预设
    const targetSelector = waitFor || domainRule.selector;

    // 等待动态内容渲染（有 selector 时等 800ms，没有时等 1500ms）
    const initialWait = targetSelector ? 800 : 1500;
    await new Promise(r => setTimeout(r, initialWait));

    // 如果有目标选择器，先等待元素出现
    if (targetSelector) {
      try {
        await page.waitForSelector(targetSelector, { timeout: Math.min(timeout, 10000) });
      } catch (error) {
        // 选择器等待失败时继续处理，后续会回退到body
      }
    }

    // 模拟用户滚动行为
    try {
      if (!page.isClosed()) {
        const scrollHeight = await page.evaluate(() => document.body.scrollHeight);
        const viewportHeight = await page.evaluate(() => window.innerHeight);

        if (scrollHeight > viewportHeight) {
          // 分段滚动，每次检查页面状态
          let currentPosition = 0;
          const stepSize = 300;

          while (currentPosition < scrollHeight - viewportHeight) {
            if (page.isClosed()) break;

            await page.evaluate((position) => {
              window.scrollTo(0, position);
            }, currentPosition);

            currentPosition += stepSize;
            await new Promise(r => setTimeout(r, 100));
          }

          // 滚动回顶部
          if (!page.isClosed()) {
            await page.evaluate(() => window.scrollTo(0, 0));
            await new Promise(r => setTimeout(r, 500));
          }
        }

      }
    } catch (error) {
      // 如果是frame detached错误，不要抛出异常，继续执行
      if (!error.message.includes('detached')) {
        throw error;
      }
    }

    await new Promise(r => setTimeout(r, 500));
    finalUrl = page.url() || finalUrl;

    // 获取页面内容
    const content = await page.content();
    const title = await page.title();
    let debugInfo = {};
    let cleanContent = { title: '', bodyText: '' };

    try {
      if (!page.isClosed()) {
        debugInfo = await page.evaluate(() => {
          const body = document.body;
          const textContent = body.textContent || body.innerText || '';

          return {
            textLength: textContent.length,
            hasApp: !!document.querySelector('#app, #root, .app, .main, main, [data-reactroot]'),
            hasReactElements: document.querySelectorAll('[data-reactid], [data-react-checksum]').length,
            hasVueElements: document.querySelectorAll('[data-v-]').length,
            scriptCount: document.querySelectorAll('script[src]').length,
            stylesheetCount: document.querySelectorAll('link[rel="stylesheet"]').length,
            hasReact: !!window.React,
            hasVue: !!window.Vue,
            hasAngular: !!window.Angular,
            hasJQuery: !!(window.$ || window.jQuery),
            readyState: document.readyState,
            firstTextPreview: textContent.substring(0, 200).replace(/\s+/g, ' ').trim()
          };
        });

        // 按目标选择器提取HTML，未命中时回退到完整body
        const extractedContent = await page.evaluate((selector) => {
          const pageTitle = document.title || '';
          let html = '';

          if (selector) {
            const elements = Array.from(document.querySelectorAll(selector));
            if (elements.length > 0) {
              html = elements.map(el => el.innerHTML || '').join('\n<hr/>\n');
            }
          }
          if (!html) {
            html = document.body?.innerHTML || '';
          }

          return {
            title: pageTitle,
            html
          };
        }, targetSelector);

        const markdownContent = html2md4llm(extractedContent.html || '', {
          outputFormat: 'markdown',
          strategy: domainRule.strategy || undefined
        });
        cleanContent = {
          title: extractedContent.title || title || '',
          bodyText: markdownContent
        };
      } else {
        // 使用已获取的content作为备用
        const title = await page.title().catch(() => '');
        cleanContent = { title: title, bodyText: html2md4llm(content || '') };
      }
    } catch (error) {
      if (error.message.includes('detached')) {
        // 使用已获取的HTML内容作为备用
        const title = await page.title().catch(() => '');
        cleanContent = { title: title, bodyText: html2md4llm(content || '') };
      } else {
        throw error;
      }
    }

    // 压缩连续空行
    const compressedBodyText = cleanContent.bodyText.replace(/\n{3,}/g, '\n\n');
    const blockedIfContains = Array.isArray(domainRule.blockedIfContains) ? domainRule.blockedIfContains : [];
    const htmlForDetection = (content || '').toLowerCase();
    const needsLoginState = blockedIfContains.some(marker => htmlForDetection.includes(String(marker).toLowerCase()));
    const shouldShowCookieExpiredTips = needsLoginState;

    // 在正文顶部添加YAML元信息
    const yamlLines = [
      '---',
      `title: ${toYamlPlainString(cleanContent.title)}`,
      `start_url: ${toYamlPlainString(url)}`
    ];
    if (finalUrl && finalUrl !== url) {
      yamlLines.push(`final_url: ${toYamlPlainString(finalUrl)}`);
    }
    if (shouldShowCookieExpiredTips) {
      yamlLines.push(`cookie_expired_tips: ${toYamlPlainString('页面内容受限，请使用 mcp-fetch-page chrome extension 重新保存登录态。')}`);
    }
    yamlLines.push('---', '', '');

    // 保存Markdown格式内容到文件
    let textContent = `${yamlLines.join('\n')}${compressedBodyText}`;
    const savedFilePath = savePageContent(url, textContent, cleanContent.title);
    if (savedFilePath) {
      console.error(`✓ Saved to ${savedFilePath}`);
    }

    const cleanResult = textContent;

    if (browser) {
      await browser.close();
    }

    return cleanResult;
  } finally {
    if (browser) {
      await browser.close();
    }
  }
}

// CLI 入口
if (import.meta.url === `file://${process.argv[1]}`) {
  const url = process.argv[2];
  if (!url) {
    console.error('Usage: read-spa.js <url>');
    process.exit(1);
  }

  fetchPage(url)
    .then(markdown => {
      // 输出到 stdout
      process.stdout.write(markdown);
    })
    .catch(err => {
      console.error('Error:', err.message);
      try {
        const friendly = [
          '❌ Fetch failed in browser mode.',
          err && err.message ? `Reason: ${err.message}` : null,
          '',
          '建议：使用 Chrome 扩展 “Fetch Page MCP Tools” 写入本地登录信息后重试。',
          '步骤：',
          `1) 打开并登录：${url || '目标网站'}`,
          '2) 点击扩展保存 cookies/localStorage',
          '3) 回到对话中再次调用 read-spa.js',
        ].filter(Boolean).join('\n');

        savePageContent(url || 'about:blank', friendly, 'Fetch Error', true);
      } catch (_) {}
      process.exit(1);
    });
}

export { fetchPage };
