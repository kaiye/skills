#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { marked } = require('marked');
const puppeteer = require('puppeteer-core');

async function md2pdf(inputPath, outputPath, options = {}) {
  // 读取 Markdown 文件
  const markdown = fs.readFileSync(inputPath, 'utf-8');
  
  // 转换为 HTML（无样式）
  const html = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
</head>
<body>
${marked(markdown)}
</body>
</html>
  `;
  
  // 写入临时 HTML 文件
  const tempHtml = outputPath.replace('.pdf', '.tmp.html');
  fs.writeFileSync(tempHtml, html);
  
  try {
    // 查找系统 Chrome
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
    
    // 启动 Chrome headless
    const browser = await puppeteer.launch({
      executablePath,
      headless: true,
      args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    
    const page = await browser.newPage();
    await page.goto(`file://${path.resolve(tempHtml)}`, { waitUntil: 'networkidle0' });
    
    // 生成 PDF
    await page.pdf({
      path: outputPath,
      format: 'A4',
      landscape: options.landscape || false,
      printBackground: true,
      margin: {
        top: '20mm',
        right: '20mm',
        bottom: '20mm',
        left: '20mm'
      }
    });
    
    await browser.close();
    const orientation = options.landscape ? '横版' : '竖版';
    console.log(`✅ PDF 已生成: ${outputPath} (${orientation})`);
    
  } finally {
    // 清理临时文件
    if (fs.existsSync(tempHtml)) {
      fs.unlinkSync(tempHtml);
    }
  }
}

// CLI 入口
if (require.main === module) {
  const args = process.argv.slice(2);
  
  if (args.length === 0 || args.includes('--help') || args.includes('-h')) {
    console.log('用法: md2pdf <input.md> [output.pdf] [--landscape]');
    console.log('');
    console.log('选项:');
    console.log('  --landscape, -l    生成横版 PDF（适合宽表格）');
    console.log('');
    console.log('示例:');
    console.log('  md2pdf report.md');
    console.log('  md2pdf report.md output.pdf');
    console.log('  md2pdf report.md --landscape');
    process.exit(args.includes('--help') || args.includes('-h') ? 0 : 1);
  }
  
  const landscape = args.includes('--landscape') || args.includes('-l');
  const fileArgs = args.filter(a => !a.startsWith('-'));
  
  const inputPath = fileArgs[0];
  const outputPath = fileArgs[1] || inputPath.replace(/\.md$/, '.pdf');
  
  if (!fs.existsSync(inputPath)) {
    console.error(`错误: 文件不存在: ${inputPath}`);
    process.exit(1);
  }
  
  md2pdf(inputPath, outputPath, { landscape }).catch(err => {
    console.error('错误:', err.message);
    process.exit(1);
  });
}

module.exports = { md2pdf };
