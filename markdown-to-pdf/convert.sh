#!/bin/bash
# markdown-to-pdf 快速转换脚本
# 用法: ./convert.sh input.md [output.pdf]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STYLE_CSS="$SCRIPT_DIR/style.css"

if [ $# -lt 1 ]; then
    echo "用法: $0 input.md [output.pdf]"
    echo "示例: $0 report.md report.pdf"
    exit 1
fi

INPUT="$1"
OUTPUT="${2:-${INPUT%.md}.pdf}"

if [ ! -f "$INPUT" ]; then
    echo "错误: 文件不存在: $INPUT"
    exit 1
fi

echo "📄 转换中: $INPUT → $OUTPUT"
echo "🎨 使用样式: $STYLE_CSS"

md2pdf \
    -i "$INPUT" \
    -o "$OUTPUT" \
    -c "$STYLE_CSS" \
    --extras 'tables' \
    --extras 'fenced_code'

echo "✅ 完成: $OUTPUT"
ls -lh "$OUTPUT"
