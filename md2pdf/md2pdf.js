#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { marked } = require('marked');
const puppeteer = require('puppeteer-core');

function parseFrontmatter(md) {
  // YAML frontmatter must be at the very beginning
  const m = md.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?/);
  if (!m) return { frontmatterRaw: null, frontmatter: {}, body: md };

  const raw = m[1];
  const body = md.slice(m[0].length);

  // Minimal YAML-ish parser: key: value (single-line)
  const fm = {};
  for (const line of raw.split(/\r?\n/)) {
    const mm = line.match(/^\s*([A-Za-z0-9_\-]+)\s*:\s*(.*?)\s*$/);
    if (!mm) continue;
    let v = mm[2];
    if ((v.startsWith('"') && v.endsWith('"')) || (v.startsWith("'") && v.endsWith("'"))) {
      v = v.slice(1, -1);
    }
    fm[mm[1]] = v;
  }

  return { frontmatterRaw: raw, frontmatter: fm, body };
}

function frontmatterToQuote(raw) {
  // Render as a big markdown quote block with fenced yaml
  const fenced = ['```yaml', raw.trimEnd(), '```'].join('\n');
  return fenced
    .split('\n')
    .map((l) => `> ${l}`)
    .join('\n') + '\n\n';
}

async function md2pdf(inputPath, outputPath, options = {}) {
  const markdown = fs.readFileSync(inputPath, 'utf-8');

  const { frontmatterRaw, frontmatter, body } = parseFrontmatter(markdown);

  let mdForRender = markdown;
  if (frontmatterRaw) {
    const mode = options.frontmatter || 'strip';
    if (mode === 'strip') {
      mdForRender = body;
    } else if (mode === 'quote') {
      mdForRender = frontmatterToQuote(frontmatterRaw) + body;
    } else {
      // keep
      mdForRender = markdown;
    }
  }

  // Default theme: mimic the PDF you sent (ChatGPT-like export feel)
  const defaultCss = `
body {
  font-family: -apple-system, "Segoe UI", Helvetica, Arial, "Noto Sans", "PingFang SC", "Microsoft YaHei", sans-serif;
  font-size: 15px;
  line-height: 1.65;
  color: #1a1a1a;
  text-align: justify;
}

p { margin: 0 0 1em 0; }

h1 {
  font-size: 28px;
  font-weight: 700;
  line-height: 1.25;
  margin: 0.4em 0 0.6em 0;
  text-align: left;
}

h2 {
  font-size: 22px;
  font-weight: 700;
  line-height: 1.3;
  margin: 1.4em 0 0.5em 0;
  text-align: left;
}

h3 {
  font-size: 18px;
  font-weight: 700;
  line-height: 1.35;
  margin: 1.2em 0 0.4em 0;
  text-align: left;
}

strong { font-weight: 700; }

code, pre {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
}

sup {
  font-size: 11px;
  vertical-align: super;
  background: #f0f0f0;
  border-radius: 3px;
  padding: 1px 4px;
  margin-left: 2px;
  color: #444;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin: 1em 0;
  font-size: 15px;
  text-align: left;
}

th, td {
  padding: 10px 12px;
  vertical-align: top;
  border-bottom: 1px solid #ddd;
}

th {
  font-weight: 400;
  border-bottom: 1.5px solid #bbb;
}

a { color: inherit; text-decoration: none; }
`;

  const extraCss = options.cssPath ? fs.readFileSync(options.cssPath, 'utf-8') : '';
  const title = frontmatter?.title || path.basename(inputPath);

  const html = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>${String(title).replace(/</g, '&lt;')}</title>
  <style>
${defaultCss}
${extraCss}
  </style>
</head>
<body>
${marked(mdForRender)}
</body>
</html>
`;

  const tempHtml = outputPath.replace(/\.pdf$/i, '.tmp.html');
  fs.writeFileSync(tempHtml, html);

  try {
    const chromePaths = [
      '/usr/bin/google-chrome',
      '/usr/bin/chromium',
      '/usr/bin/chromium-browser',
      '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
      'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
      'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe'
    ];

    let executablePath = null;
    for (const p of chromePaths) {
      if (fs.existsSync(p)) {
        executablePath = p;
        break;
      }
    }
    if (!executablePath) {
      throw new Error('未找到 Chrome/Chromium，请先安装 Google Chrome 或 Chromium');
    }

    const browser = await puppeteer.launch({
      executablePath,
      headless: true,
      args: ['--no-sandbox', '--disable-setuid-sandbox']
    });

    const page = await browser.newPage();
    await page.goto(`file://${path.resolve(tempHtml)}`, { waitUntil: 'networkidle0' });

    // Match the feel of the sample PDF: Letter + generous margins
    await page.pdf({
      path: outputPath,
      format: 'Letter',
      landscape: options.landscape || false,
      printBackground: true,
      margin: {
        top: '16mm',
        right: '19mm',
        bottom: '22mm',
        left: '19mm'
      }
    });

    await browser.close();
    const orientation = options.landscape ? '横版' : '竖版';
    console.log(`✅ PDF 已生成: ${outputPath} (${orientation})`);
  } finally {
    if (fs.existsSync(tempHtml)) fs.unlinkSync(tempHtml);
  }
}

if (require.main === module) {
  const args = process.argv.slice(2);

  if (args.length === 0 || args.includes('--help') || args.includes('-h')) {
    console.log('用法: md2pdf <input.md> [output.pdf] [--landscape] [--css <file.css>] [--frontmatter <strip|quote|keep>]');
    console.log('');
    console.log('选项:');
    console.log('  --landscape, -l                  生成横版 PDF（适合宽表格）');
    console.log('  --css <file.css>                 额外 CSS（在默认主题基础上追加/覆盖）');
    console.log('  --frontmatter <strip|quote|keep> frontmatter 处理方式（默认 strip）');
    console.log('');
    console.log('示例:');
    console.log('  md2pdf report.md');
    console.log('  md2pdf report.md output.pdf');
    console.log('  md2pdf report.md --landscape');
    console.log('  md2pdf report.md --css style.css');
    console.log('  md2pdf report.md --frontmatter quote');
    process.exit(args.includes('--help') || args.includes('-h') ? 0 : 1);
  }

  const landscape = args.includes('--landscape') || args.includes('-l');

  // parse --css <path>
  let cssPath = null;
  const cssIdx = args.findIndex((a) => a === '--css');
  if (cssIdx !== -1) {
    cssPath = args[cssIdx + 1];
    if (!cssPath || cssPath.startsWith('-')) {
      console.error('错误: --css 需要跟一个 css 文件路径');
      process.exit(1);
    }
    if (!fs.existsSync(cssPath)) {
      console.error(`错误: CSS 文件不存在: ${cssPath}`);
      process.exit(1);
    }
  }

  // parse --frontmatter <strip|quote|keep>
  let frontmatter = 'strip';
  const fmIdx = args.findIndex((a) => a === '--frontmatter');
  if (fmIdx !== -1) {
    frontmatter = args[fmIdx + 1];
    if (!frontmatter || frontmatter.startsWith('-')) {
      console.error('错误: --frontmatter 需要跟 strip|quote|keep');
      process.exit(1);
    }
    if (!['strip', 'quote', 'keep'].includes(frontmatter)) {
      console.error('错误: --frontmatter 只支持 strip|quote|keep');
      process.exit(1);
    }
  }

  const fileArgs = args.filter((a, i) => {
    if (a.startsWith('-')) return false;
    if (cssIdx !== -1 && i === cssIdx + 1) return false;
    if (fmIdx !== -1 && i === fmIdx + 1) return false;
    return true;
  });

  const inputPath = fileArgs[0];
  const outputPath = fileArgs[1] || inputPath.replace(/\.md$/i, '.pdf');

  if (!fs.existsSync(inputPath)) {
    console.error(`错误: 文件不存在: ${inputPath}`);
    process.exit(1);
  }

  md2pdf(inputPath, outputPath, { landscape, cssPath, frontmatter }).catch((err) => {
    console.error('错误:', err.message);
    process.exit(1);
  });
}

module.exports = { md2pdf };
