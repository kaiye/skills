#!/usr/bin/env node
/**
 * sync-from-draft.js
 * 从微信草稿箱拉取最新草稿，与本地 Markdown 对比，输出结构化 diff。
 *
 * 用法：
 *   WECHAT_APP_ID=xxx WECHAT_APP_SECRET=yyy \
 *   node sync-from-draft.js <local-md-path> [--apply]
 *
 *   --apply  自动将 diff 结果 patch 回本地文件（不加则只展示）
 */

import https from 'https';
import fs from 'fs';
import path from 'path';
import { html2md4llm } from 'html2md4llm';
import { diffLines, diffWords } from 'diff';

// ── helpers ──────────────────────────────────────────────────────────────────

function httpsPost(url, body) {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify(body);
    const u = new URL(url);
    const req = https.request({
      hostname: u.hostname,
      path: u.pathname + u.search,
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data) },
    }, res => {
      let buf = '';
      res.on('data', c => buf += c);
      res.on('end', () => resolve(JSON.parse(buf)));
    });
    req.on('error', reject);
    req.write(data);
    req.end();
  });
}

function httpsGet(url) {
  return new Promise((resolve, reject) => {
    https.get(url, res => {
      let buf = '';
      res.on('data', c => buf += c);
      res.on('end', () => resolve(JSON.parse(buf)));
    }).on('error', reject);
  });
}

// ── 获取 access_token ──────────────────────────────────────────────────────

async function getToken(appId, appSecret) {
  const res = await httpsGet(
    `https://api.weixin.qq.com/cgi-bin/token?grant_type=client_credential&appid=${appId}&secret=${appSecret}`
  );
  if (!res.access_token) throw new Error(`获取 token 失败: ${JSON.stringify(res)}`);
  return res.access_token;
}

// ── 拉取草稿箱 ─────────────────────────────────────────────────────────────

async function fetchLatestDraft(token) {
  const res = await httpsPost(
    `https://api.weixin.qq.com/cgi-bin/draft/batchget?access_token=${token}`,
    { offset: 0, count: 1, no_content: 0 }
  );
  if (!res.item || res.item.length === 0) throw new Error('草稿箱为空');
  const item = res.item[0];
  const article = item.content.news_item[0];
  return {
    mediaId: item.media_id,
    title: article.title,
    html: article.content,
  };
}

// ── 去除 frontmatter ────────────────────────────────────────────────────────

function stripFrontmatter(md) {
  return md.replace(/^---[\s\S]*?---\n?/, '').trim();
}

// ── diff 并输出结构化结果 ───────────────────────────────────────────────────

function buildDiffReport(localMd, draftMd) {
  const changes = [];

  const localLines = localMd.split('\n').filter(l => l.trim());
  const draftLines = draftMd.split('\n').filter(l => l.trim());

  const lineDiffs = diffLines(localLines.join('\n'), draftLines.join('\n'));

  let i = 0;
  while (i < lineDiffs.length) {
    const chunk = lineDiffs[i];

    if (chunk.removed && i + 1 < lineDiffs.length && lineDiffs[i + 1].added) {
      // 修改：行级 removed + added 配对 → 再做词级 diff 找细节
      const oldText = chunk.value.trim();
      const newText = lineDiffs[i + 1].value.trim();

      // 词级 diff，找出具体改了什么
      const wordDiffs = diffWords(oldText, newText);
      const removedWords = wordDiffs.filter(d => d.removed).map(d => d.value.trim()).filter(Boolean);
      const addedWords = wordDiffs.filter(d => d.added).map(d => d.value.trim()).filter(Boolean);

      // 噪音过滤：词级无差异（只有空格变化）→ 忽略
      if (removedWords.length === 0 && addedWords.length === 0) {
        i += 2;
        continue;
      }

      changes.push({
        type: 'replace',
        old: oldText,
        new: newText,
        detail: { removed: removedWords, added: addedWords },
      });
      i += 2;
    } else if (chunk.removed) {
      changes.push({ type: 'delete', old: chunk.value.trim() });
      i++;
    } else if (chunk.added) {
      changes.push({ type: 'insert', new: chunk.value.trim() });
      i++;
    } else {
      i++;
    }
  }

  return changes;
}

// ── 意图识别：检测括号备注 （...） ──────────────────────────────────────────

const INTENT_RE = /（[^）]+）/g;

function extractIntents(changes) {
  const intents = [];
  changes.forEach((c, idx) => {
    const text = c.new || c.old || '';
    const matches = [...text.matchAll(INTENT_RE)];
    if (matches.length > 0) {
      intents.push({
        changeIndex: idx + 1,
        context: text,
        notes: matches.map(m => m[0]),
      });
    }
  });
  return intents;
}

// ── 打印人类可读报告 ────────────────────────────────────────────────────────

function printReport(changes) {
  if (changes.length === 0) {
    console.log('✅ 草稿与本地文件内容一致，无差异。');
    return;
  }

  console.log(`\n发现 ${changes.length} 处差异：\n`);
  changes.forEach((c, idx) => {
    const label = c.type === 'replace' ? '【修改】' : c.type === 'delete' ? '【删除】' : '【新增】';
    const intentFlag = (c.new || c.old || '').match(INTENT_RE) ? ' ⚠️ 含意图备注' : '';
    console.log(`[${idx + 1}] ${label}${intentFlag}`);
    if (c.type === 'replace') {
      console.log(`  原：${c.old}`);
      console.log(`  改：${c.new}`);
      if (c.detail.removed.length || c.detail.added.length) {
        if (c.detail.removed.length) console.log(`  删词：${c.detail.removed.join(' | ')}`);
        if (c.detail.added.length)  console.log(`  加词：${c.detail.added.join(' | ')}`);
      }
    } else if (c.type === 'delete') {
      console.log(`  内容：${c.old}`);
    } else {
      console.log(`  内容：${c.new}`);
    }
    console.log();
  });

  // 单独汇总意图备注
  const intents = extractIntents(changes);
  if (intents.length > 0) {
    console.log('─────────────────────────────────────');
    console.log('⚠️  发现以下意图备注，apply 前需确认处理方式：\n');
    intents.forEach(it => {
      console.log(`  [差异 ${it.changeIndex}] ${it.notes.join(' / ')}`);
    });
    console.log('\n请告知如何处理后再执行 --apply。');
    console.log('─────────────────────────────────────\n');
  }
}

// ── 主流程 ──────────────────────────────────────────────────────────────────

async function main() {
  const args = process.argv.slice(2);
  const applyFlag = args.includes('--apply');
  const localPath = args.find(a => !a.startsWith('--'));

  if (!localPath) {
    console.error('用法: node sync-from-draft.js <local-md-path> [--apply]');
    process.exit(1);
  }

  const appId = process.env.WECHAT_APP_ID;
  const appSecret = process.env.WECHAT_APP_SECRET;
  if (!appId || !appSecret) {
    console.error('请设置环境变量 WECHAT_APP_ID 和 WECHAT_APP_SECRET');
    process.exit(1);
  }

  console.log('🔑 获取 access_token...');
  const token = await getToken(appId, appSecret);

  console.log('📥 拉取最新草稿...');
  const draft = await fetchLatestDraft(token);
  console.log(`   标题：${draft.title}`);
  console.log(`   Media ID：${draft.mediaId}`);

  console.log('🔄 HTML → Markdown...');
  const draftMd = html2md4llm(draft.html, { strategy: 'article' });

  console.log('📂 读取本地文件...');
  const localRaw = fs.readFileSync(path.resolve(localPath), 'utf8');
  const localMd = stripFrontmatter(localRaw);

  console.log('🔍 对比差异...\n');
  const changes = buildDiffReport(localMd, draftMd);
  printReport(changes);

  // 有意图备注时，阻止 apply，等人工确认
  const intents = extractIntents(changes);
  if (applyFlag && intents.length > 0) {
    console.log('\n❌ apply 已阻止：存在意图备注，请先确认处理方式后再执行 --apply。');
    process.exit(1);
  }

  if (applyFlag && changes.length > 0) {
    // 提取本地 frontmatter（--- ... --- 块）
    const fmMatch = localRaw.match(/^(---[\s\S]*?---\n?)/);
    const frontmatter = fmMatch ? fmMatch[1] : '';

    // 用草稿 MD 替换正文，frontmatter 保持不变
    const patched = frontmatter + draftMd.trim() + '\n';
    fs.writeFileSync(path.resolve(localPath), patched, 'utf8');
    console.log(`✅ 已将草稿内容同步回本地文件：${localPath}`);

    // 同时保存 JSON diff 供归档
    const outPath = localPath.replace(/\.md$/, '.draft-diff.json');
    fs.writeFileSync(outPath, JSON.stringify(changes, null, 2), 'utf8');
    console.log(`💾 diff 记录已保存：${outPath}`);
  } else if (applyFlag && changes.length === 0) {
    console.log('✅ 无差异，本地文件无需更新。');
  }
}

main().catch(err => { console.error('❌', err.message); process.exit(1); });
