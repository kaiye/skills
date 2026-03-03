#!/usr/bin/env node

/**
 * 从 Cloudflare KV 拉取 Sync Your Cookie 数据
 * 直接保存原始格式，不做转换
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// 自动加载 .env 文件（Node 20.6+ 原生支持，兼容旧版手动加载）
const __envFilename = fileURLToPath(import.meta.url);
const __envDirname = path.dirname(__envFilename);
const envPath = path.resolve(__envDirname, '..', '.env');
if (fs.existsSync(envPath)) {
  const envContent = fs.readFileSync(envPath, 'utf8');
  for (const line of envContent.split('\n')) {
    const match = line.match(/^([^#=]+)=(.*)$/);
    if (match) process.env[match[1].trim()] = match[2].trim();
  }
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const SKILL_DIR = path.resolve(__dirname, '..');
const COOKIE_DIR = path.join(SKILL_DIR, 'cookies');

// Cloudflare KV 配置（从环境变量或 .env 文件读取，不硬编码敏感信息）
const ACCOUNT_ID = process.env.CF_ACCOUNT_ID;
const NAMESPACE_ID = process.env.CF_NAMESPACE_ID;
const API_TOKEN = process.env.CF_API_TOKEN;

if (!ACCOUNT_ID || !NAMESPACE_ID || !API_TOKEN) {
  console.error('❌ 缺少 Cloudflare KV 配置，请设置环境变量：');
  console.error('   CF_ACCOUNT_ID, CF_NAMESPACE_ID, CF_API_TOKEN');
  console.error('   或在 skills/page-fetcher/ 目录创建 .env 文件');
  process.exit(1);
}

// 确保目录存在
if (!fs.existsSync(COOKIE_DIR)) {
  fs.mkdirSync(COOKIE_DIR, { recursive: true });
}

/**
 * 从 Cloudflare KV 读取数据
 */
async function getFromKV() {
  const url = `https://api.cloudflare.com/client/v4/accounts/${ACCOUNT_ID}/storage/kv/namespaces/${NAMESPACE_ID}/values/sync-your-cookie`;
  
  const response = await fetch(url, {
    headers: {
      'Authorization': `Bearer ${API_TOKEN}`
    }
  });

  if (!response.ok) {
    if (response.status === 404) {
      return null;
    }
    throw new Error(`KV API error: ${response.status} ${response.statusText}`);
  }

  return await response.text();
}

/**
 * 同步 cookie 数据
 */
async function sync() {
  console.log('Fetching cookie data from Cloudflare KV...');
  
  try {
    const text = await getFromKV();
    
    if (!text) {
      console.error('✗ No data found in Cloudflare KV');
      console.error('Please push cookies using Sync Your Cookie extension first.');
      process.exit(1);
    }

    // 验证 JSON 格式
    const data = JSON.parse(text);
    
    if (!data.domainCookieMap) {
      console.error('✗ Invalid data format (missing domainCookieMap)');
      process.exit(1);
    }
    
    const domains = Object.keys(data.domainCookieMap);
    console.log(`Found ${domains.length} domain(s): ${domains.join(', ')}`);
    
    // 直接保存原始格式
    const filename = path.join(COOKIE_DIR, 'sync-your-cookie.json');
    fs.writeFileSync(filename, text);
    
    console.log(`✓ Synced to ${filename}`);
    
    // 显示每个域名的统计
    for (const domain of domains) {
      const domainData = data.domainCookieMap[domain];
      const cookieCount = domainData.cookies?.length || 0;
      const localStorageCount = domainData.localStorageItems?.length || 0;
      console.log(`  - ${domain}: ${cookieCount} cookies, ${localStorageCount} localStorage items`);
    }
    
  } catch (err) {
    console.error('Failed to sync:', err.message);
    process.exit(1);
  }
}

// 执行同步
sync();
