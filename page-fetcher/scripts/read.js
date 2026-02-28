#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { main as html2md4llm } from 'html2md4llm';

function usage() {
  console.error('Usage: read.js <url> [--json]');
  process.exit(2);
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const SKILL_DIR = path.resolve(__dirname, '..');

function hasFlag(flag) {
  return process.argv.includes(flag);
}

const urlStr = process.argv.slice(2).find((a) => !a.startsWith('--'));
const outputJson = hasFlag('--json');

if (!urlStr) usage();

const url = new URL(urlStr);
const host = url.hostname;

const rulesPath = path.join(SKILL_DIR, 'rules.json');
const rules = JSON.parse(fs.readFileSync(rulesPath, 'utf-8'));
const rule = rules[host] || {};

const UA = rule.ua ||
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36';

const res = await fetch(urlStr, {
  headers: {
    'User-Agent': UA,
    Accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Cache-Control': 'no-cache',
    Pragma: 'no-cache',
  },
  redirect: 'follow',
});

const html = await res.text();

// Basic captcha detection for WeChat
const lower = html.toLowerCase();
if (host === 'mp.weixin.qq.com') {
  if (lower.includes('wappoc_appmsgcaptcha')) {
    throw new Error('WeChat returned a captcha/blocked page (wappoc_appmsgcaptcha). Try later, reduce frequency, or change IP/UA.');
  }
}

const title = extractMeta(html, 'og:title') || '';
const author = extractMeta(html, 'og:article:author') || '';

let extractedHtml = html;
if (rule.selector) {
  const bySel = extractBySimpleSelector(html, rule.selector);
  if (!bySel) {
    throw new Error(`Selector not found: ${rule.selector}`);
  }
  extractedHtml = bySel;
}

const strategy = rule.strategy; // optional

if (outputJson) {
  const json = html2md4llm(extractedHtml, { outputFormat: 'json', strategy });
  process.stdout.write(json + '\n');
  process.exit(0);
}

const mdBody = html2md4llm(extractedHtml, { outputFormat: 'markdown', strategy });

const frontMatter = [
  '---',
  `title: ${JSON.stringify(title)}`,
  `author: ${JSON.stringify(author)}`,
  `source_url: ${JSON.stringify(urlStr)}`,
  `fetched_at: ${JSON.stringify(new Date().toISOString())}`,
  '---',
  '',
].join('\n');

process.stdout.write(frontMatter + mdBody.trim() + '\n');

function extractMeta(docHtml, property) {
  // property="og:title" content="..."
  const re = new RegExp(`<meta\\s+[^>]*property=["']${escapeRe(property)}["'][^>]*>`, 'i');
  const m = docHtml.match(re);
  if (!m) return '';
  const tag = m[0];
  const cm = tag.match(/content=["']([^"']+)["']/i);
  return cm ? decodeHtmlEntities(cm[1]) : '';
}

function extractBySimpleSelector(docHtml, selector) {
  // Supports .class and #id only (no combinators)
  selector = selector.trim();
  if (selector.startsWith('.')) {
    const cls = selector.slice(1);
    return extractFirstElementByAttrToken(docHtml, 'class', cls);
  }
  if (selector.startsWith('#')) {
    const id = selector.slice(1);
    return extractFirstElementByAttrExact(docHtml, 'id', id);
  }
  throw new Error(`Unsupported selector (only .class/#id): ${selector}`);
}

function extractFirstElementByAttrExact(docHtml, attr, value) {
  const tagStartRe = new RegExp(`<([a-zA-Z][a-zA-Z0-9]*)\\b[^>]*\\b${escapeRe(attr)}=["']${escapeRe(value)}["'][^>]*>`, 'i');
  const m = docHtml.match(tagStartRe);
  if (!m) return null;
  const tag = m[1].toLowerCase();
  const startIdx = m.index;
  const startTag = m[0];
  const startTagEnd = startIdx + startTag.length;
  const endIdx = findMatchingClosingTag(docHtml, tag, startTagEnd);
  if (endIdx < 0) return null;
  return docHtml.slice(startIdx, endIdx);
}

function extractFirstElementByAttrToken(docHtml, attr, token) {
  const tagStartRe = new RegExp(`<([a-zA-Z][a-zA-Z0-9]*)\\b[^>]*\\b${escapeRe(attr)}=["']([^"']+)["'][^>]*>`, 'ig');
  let m;
  while ((m = tagStartRe.exec(docHtml)) !== null) {
    const tag = m[1].toLowerCase();
    const attrVal = m[2];
    const tokens = attrVal.split(/\\s+/).filter(Boolean);
    if (!tokens.includes(token)) continue;

    const startIdx = m.index;
    const startTag = m[0];
    const startTagEnd = startIdx + startTag.length;
    const endIdx = findMatchingClosingTag(docHtml, tag, startTagEnd);
    if (endIdx < 0) return null;
    return docHtml.slice(startIdx, endIdx);
  }
  return null;
}

function findMatchingClosingTag(docHtml, tag, fromIndex) {
  // Return index right after closing tag
  const openRe = new RegExp(`<${escapeRe(tag)}\\b`, 'ig');
  const closeRe = new RegExp(`</${escapeRe(tag)}\\s*>`, 'ig');

  // We are already inside the first opening tag
  let depth = 1;
  let i = fromIndex;

  while (i < docHtml.length) {
    openRe.lastIndex = i;
    closeRe.lastIndex = i;
    const om = openRe.exec(docHtml);
    const cm = closeRe.exec(docHtml);

    if (!cm) return -1;

    // Determine whether next event is open or close
    if (om && om.index < cm.index) {
      // Check if this open tag is self-closing
      const gt = docHtml.indexOf('>', om.index);
      if (gt < 0) return -1;
      const openTagText = docHtml.slice(om.index, gt + 1);
      const selfClosing = /\/\s*>$/.test(openTagText) || /<\s*(br|hr|img|meta|link|input)\b/i.test(openTagText);
      if (!selfClosing) depth++;
      i = gt + 1;
      continue;
    }

    // close tag
    depth--;
    const closeEnd = cm.index + cm[0].length;
    i = closeEnd;
    if (depth === 0) return closeEnd;
  }

  return -1;
}

function escapeRe(s) {
  return s.replace(/[.*+?^${}()|[\\]\\]/g, '\\$&');
}

function decodeHtmlEntities(str) {
  return str
    .replace(/&amp;/g, '&')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>');
}

