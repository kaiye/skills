#!/bin/bash
set -e

# ============================================
# 四格漫画生成脚本
# 依赖：gemini-image-gen skill
# ============================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(dirname "$SCRIPT_DIR")"
GEMINI_SKILL_DIR="/root/.openclaw/workspace/github-repos/kaiye/skills/gemini-image-gen"

# 默认参数
OUTPUT_PATH=""
STORY_FILE=""
STYLE="handdrawn"

# 解析参数
while [[ $# -gt 0 ]]; do
  case $1 in
    -o|--output)
      OUTPUT_PATH="$2"
      shift 2
      ;;
    -s|--story-file)
      STORY_FILE="$2"
      shift 2
      ;;
    --style)
      STYLE="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: gen-comic.sh -o <output_path> -s <story_file> [--style handdrawn|realistic]"
      echo ""
      echo "Options:"
      echo "  -o, --output       输出文件路径（必须）"
      echo "  -s, --story-file   故事大纲 YAML 文件（必须）"
      echo "  --style            风格：handdrawn（手绘）或 realistic（写实），默认 handdrawn"
      echo ""
      echo "Example:"
      echo "  gen-comic.sh -o ./my-comic.png -s ./story.yaml"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# 检查必需参数
if [[ -z "$OUTPUT_PATH" ]]; then
  echo "Error: -o/--output is required"
  exit 1
fi

if [[ -z "$STORY_FILE" ]]; then
  echo "Error: -s/--story-file is required"
  exit 1
fi

if [[ ! -f "$STORY_FILE" ]]; then
  echo "Error: Story file not found: $STORY_FILE"
  exit 1
fi

# 检查依赖
if [[ ! -d "$GEMINI_SKILL_DIR" ]]; then
  echo "Error: gemini-image-gen skill not found at $GEMINI_SKILL_DIR"
  exit 1
fi

# TODO: 解析 YAML 并组装 prompt
# 当前版本：直接读取 YAML 内容作为参考，手动组装 prompt
# 未来版本：用 yq 或 Python 解析 YAML

echo "⚠️  当前版本需要手动组装 prompt"
echo "Story file: $STORY_FILE"
echo "Output: $OUTPUT_PATH"
echo ""
echo "请参考 SKILL.md 中的使用方式，通过 AI 交互式生成漫画。"
echo "脚本功能开发中..."

exit 1
