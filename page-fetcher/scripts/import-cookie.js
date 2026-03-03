#!/usr/bin/env node

/**
 * 导入 cookie 数据（支持多种格式）
 * 1. 解析用户输入（Sync Your Cookie 格式或其他格式）
 * 2. 转换并合并到 sync-your-cookie.json
 * 3. 如果有 Cloudflare KV 配置 → 同步回云端
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { z } from 'zod';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const SKILL_DIR = path.resolve(__dirname, '..');
const COOKIE_DIR = path.join(SKILL_DIR, 'cookies');
const SECRETS_DIR = path.join(process.env.HOME, '.openclaw/workspace/.secrets');

// 确保目录存在
if (!fs.existsSync(COOKIE_DIR)) {
  fs.mkdirSync(COOKIE_DIR, { recursive: true });
}

// Zod Schema: Sync Your Cookie 格式
const SyncYourCookieSchema = z.object({
  updateTime: z.number().optional(),
  createTime: z.number().optional(),
  domainCookieMap: z.record(z.object({
    updateTime: z.number().optional(),
    createTime: z.number().optional(),
    cookies: z.array(z.object({
      name: z.string(),
      value: z.string(),
      domain: z.string(),
      path: z.string().optional(),
      secure: z.boolean().optional(),
      httpOnly: z.boolean().optional(),
      sameSite: z.string().optional(),
      expirationDate: z.number().optional(),
      session: z.boolean().optional(),
      storeId: z.string().optional(),
      hostOnly: z.boolean().optional()
    })),
    localStorageItems: z.array(z.object({
      key: z.string(),
      value: z.string()
    })).optional(),
    userAgent: z.string().optional()
  }))
});

/**
 * 读取 Cloudflare KV 配置
 */
function loadCloudflareConfig() {
  const configFile = path.join(SECRETS_DIR, 'cloudflare-kv.md');
  if (!fs.existsSync(configFile)) {
    return null;
  }
  
  try {
    const content = fs.readFileSync(configFile, 'utf-8');
    const accountId = content.match(/Account ID.*?`([^`]+)`/)?.[1];
    const namespaceId = content.match(/Namespace ID.*?`([^`]+)`/)?.[1];
    const apiToken = content.match(/API Token.*?`([^`]+)`/)?.[1];
    
    if (accountId && namespaceId && apiToken) {
      return { accountId, namespaceId, apiToken };
    }
  } catch (err) {
    console.error('Failed to load Cloudflare config:', err.message);
  }
  
  return null;
}

/**
 * 上传到 Cloudflare KV
 */
async function uploadToKV(data, config) {
  const url = `https://api.cloudflare.com/client/v4/accounts/${config.accountId}/storage/kv/namespaces/${config.namespaceId}/values/sync-your-cookie`;
  
  const response = await fetch(url, {
    method: 'PUT',
    headers: {
      'Authorization': `Bearer ${config.apiToken}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
  });

  if (!response.ok) {
    throw new Error(`KV API error: ${response.status} ${response.statusText}`);
  }
  
  return true;
}

/**
 * 转换其他格式为 Sync Your Cookie 格式
 */
function convertToSyncYourCookie(input) {
  // 如果已经是 Sync Your Cookie 格式
  if (input.domainCookieMap) {
    return input;
  }
  
  // 如果是单域名格式 {domain, cookies, localStorage}
  if (input.domain && input.cookies) {
    const domain = input.domain;
    const localStorageItems = [];
    
    if (input.localStorage) {
      for (const [key, value] of Object.entries(input.localStorage)) {
        localStorageItems.push({ key, value });
      }
    }
    
    return {
      updateTime: Date.now(),
      createTime: Date.now(),
      domainCookieMap: {
        [domain]: {
          updateTime: Date.now(),
          createTime: Date.now(),
          cookies: input.cookies,
          localStorageItems,
          userAgent: input.userAgent
        }
      }
    };
  }
  
  throw new Error('Unsupported cookie format');
}

/**
 * 合并 cookie 数据
 */
function mergeCookieData(existing, newData) {
  const merged = {
    updateTime: Date.now(),
    createTime: existing.createTime || Date.now(),
    domainCookieMap: { ...existing.domainCookieMap }
  };
  
  // 合并每个域名的数据
  for (const [domain, data] of Object.entries(newData.domainCookieMap)) {
    merged.domainCookieMap[domain] = {
      ...data,
      updateTime: Date.now(),
      createTime: existing.domainCookieMap?.[domain]?.createTime || Date.now()
    };
  }
  
  return merged;
}

/**
 * 主函数
 */
async function importCookie(inputFile) {
  console.log(`Importing cookie from: ${inputFile}`);
  
  // 1. 读取输入
  let input;
  try {
    const content = fs.readFileSync(inputFile, 'utf-8');
    input = JSON.parse(content);
  } catch (err) {
    console.error('Failed to read input file:', err.message);
    process.exit(1);
  }
  
  // 2. 转换为 Sync Your Cookie 格式
  let newData;
  try {
    newData = convertToSyncYourCookie(input);
  } catch (err) {
    console.error('Failed to convert format:', err.message);
    process.exit(1);
  }
  
  // 3. 验证格式
  try {
    SyncYourCookieSchema.parse(newData);
  } catch (err) {
    console.error('Invalid cookie format:', err.message);
    process.exit(1);
  }
  
  // 4. 读取现有数据并合并
  const syncFile = path.join(COOKIE_DIR, 'sync-your-cookie.json');
  let existing = { domainCookieMap: {} };
  
  if (fs.existsSync(syncFile)) {
    try {
      existing = JSON.parse(fs.readFileSync(syncFile, 'utf-8'));
    } catch (err) {
      console.warn('Failed to read existing file, will overwrite:', err.message);
    }
  }
  
  const merged = mergeCookieData(existing, newData);
  
  // 5. 保存到本地
  fs.writeFileSync(syncFile, JSON.stringify(merged, null, 2));
  console.log(`✓ Saved to ${syncFile}`);
  
  const domains = Object.keys(newData.domainCookieMap);
  for (const domain of domains) {
    const data = newData.domainCookieMap[domain];
    console.log(`  - ${domain}: ${data.cookies.length} cookies, ${data.localStorageItems?.length || 0} localStorage items`);
  }
  
  // 6. 如果有 Cloudflare KV 配置，同步到云端
  const cfConfig = loadCloudflareConfig();
  if (cfConfig) {
    console.log('\nSyncing to Cloudflare KV...');
    try {
      await uploadToKV(merged, cfConfig);
      console.log('✓ Synced to cloud');
    } catch (err) {
      console.error('✗ Failed to sync to cloud:', err.message);
      console.error('  (Local file saved successfully)');
    }
  } else {
    console.log('\n(No Cloudflare KV config found, skipped cloud sync)');
  }
}

// CLI 入口
if (import.meta.url === `file://${process.argv[1]}`) {
  const inputFile = process.argv[2];
  if (!inputFile) {
    console.error('Usage: import-cookie.js <cookie-file>');
    console.error('Example: import-cookie.js ~/Downloads/cookies.json');
    process.exit(1);
  }

  importCookie(inputFile).catch(err => {
    console.error('Error:', err);
    process.exit(1);
  });
}

export { importCookie };
