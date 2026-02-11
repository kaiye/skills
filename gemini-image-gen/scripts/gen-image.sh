#!/bin/bash
# Gemini Image Generation & Editing Script
# Usage: gen-image.sh [-o output_path] [-i input_image] <prompt>
#
# Options:
#   -o output_path    Save image to this path (default: ./generated-YYYYMMDD-HHMMSS.png)
#   -i input_image    Input image for editing (supports png/jpg/webp)
#
# Environment variables:
#   GEMINI_API_KEY       (required) Google AI Studio API key
#   GEMINI_MODEL         (optional) defaults to gemini-3-pro-image-preview
#   GEMINI_PROXY         (optional) defaults to auto-detect SOCKS5
#                                   set to "none" to disable proxy
set -euo pipefail

# ── Parse arguments ──
OUTPUT_PATH=""
INPUT_IMAGE=""
while getopts "o:i:" opt; do
  case $opt in
    o) OUTPUT_PATH="$OPTARG" ;;
    i) INPUT_IMAGE="$OPTARG" ;;
    *) echo "Usage: gen-image.sh [-o output_path] [-i input_image] <prompt>" >&2; exit 1 ;;
  esac
done
shift $((OPTIND - 1))

PROMPT="${1:?Usage: gen-image.sh [-o output_path] [-i input_image] <prompt>}"

# Default output path: current directory with timestamp
if [ -z "$OUTPUT_PATH" ]; then
  OUTPUT_PATH="./generated-$(date +%Y%m%d-%H%M%S).png"
fi

# Validate input image if provided
if [ -n "$INPUT_IMAGE" ] && [ ! -f "$INPUT_IMAGE" ]; then
  echo "ERROR: Input image not found: $INPUT_IMAGE" >&2
  exit 1
fi

# ── Auto-detect GEMINI_API_KEY if not in environment ──
if [ -z "${GEMINI_API_KEY:-}" ]; then
  # Try extracting from shell config files
  for rc in ~/.zshrc ~/.zshenv ~/.bashrc ~/.bash_profile ~/.profile; do
    if [ -f "$rc" ]; then
      _key=$(grep -oP "(?<=GEMINI_API_KEY=['\"])[^'\"]+" "$rc" 2>/dev/null \
          || grep "GEMINI_API_KEY=" "$rc" 2>/dev/null | head -1 | sed "s/.*GEMINI_API_KEY=['\"]\\{0,1\\}\\([^'\"]*\\).*/\\1/")
      if [ -n "${_key:-}" ]; then
        export GEMINI_API_KEY="$_key"
        break
      fi
    fi
  done
fi

API_KEY="${GEMINI_API_KEY:?GEMINI_API_KEY not set. Get one at https://aistudio.google.com/apikey}"
MODEL="${GEMINI_MODEL:-gemini-3-pro-image-preview}"

# ── Auto-detect proxy ──
# Default: SOCKS5 on 1080. Set GEMINI_PROXY=none to disable.
if [ "${GEMINI_PROXY:-}" = "none" ]; then
  PROXY_ARGS=""
elif [ -n "${GEMINI_PROXY:-}" ]; then
  PROXY_ARGS="$GEMINI_PROXY"
else
  # Auto-detect: check if SOCKS5 proxy is running on common ports
  PROXY_ARGS=""
  for port in 1080 7890 7891; do
    if nc -z -w1 127.0.0.1 "$port" 2>/dev/null; then
      PROXY_ARGS="--socks5-hostname 127.0.0.1:$port"
      break
    fi
  done
fi

# Ensure output directory exists
mkdir -p "$(dirname "$OUTPUT_PATH")"

# Build JSON payload via python (safe for any prompt content)
PAYLOAD=$(python3 -c "
import json, sys, base64, os

prompt_text = sys.argv[1]
input_image = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None

parts = [{'text': prompt_text}]

if input_image:
    ext = os.path.splitext(input_image)[1].lower()
    mime_map = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp'}
    mime_type = mime_map.get(ext, 'image/png')
    with open(input_image, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()
    parts.append({'inlineData': {'mimeType': mime_type, 'data': img_b64}})

d = {
    'contents': [{'parts': parts}],
    'generationConfig': {'responseModalities': ['TEXT', 'IMAGE']}
}
print(json.dumps(d))
" "$PROMPT" "${INPUT_IMAGE:-}")

# Call Gemini API
RESPONSE=$(curl -s $PROXY_ARGS \
  -X POST \
  -H "Content-Type: application/json" \
  "https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent?key=${API_KEY}" \
  -d "$PAYLOAD" 2>&1)

# Extract image and save
python3 -c "
import json, base64, sys

data = json.load(sys.stdin)

if 'error' in data:
    msg = data['error']['message']
    print(f'ERROR: {msg}', file=sys.stderr)
    sys.exit(1)

candidates = data.get('candidates', [])
if not candidates:
    print('ERROR: No candidates in response', file=sys.stderr)
    sys.exit(1)

parts = candidates[0].get('content', {}).get('parts', [])
saved = False

for p in parts:
    if 'inlineData' in p:
        img_data = base64.b64decode(p['inlineData']['data'])
        with open('$OUTPUT_PATH', 'wb') as f:
            f.write(img_data)
        print(f'$OUTPUT_PATH ({len(img_data)/1024:.0f}KB)')
        saved = True
    elif 'text' in p:
        # Print model commentary to stderr (not mixed with output path)
        print(p['text'].strip()[:200], file=sys.stderr)

if not saved:
    print('ERROR: No image data in response', file=sys.stderr)
    sys.exit(1)
" <<< "$RESPONSE"
