#!/usr/bin/env bash
# Veo text-to-video (or image-to-video) generator
# Usage: gen-video.sh -o output.mp4 [-i ref_image.png] [-d 4-8] [-a 16:9|9:16|1:1] "prompt text"
#
# Environment: GEMINI_API_KEY (auto-read from ~/.zshrc if absent)
# Proxy: socks5h://127.0.0.1:1080 (auto-detected)

set -euo pipefail

OUTPUT=""
INPUT_IMAGE=""
DURATION=5
ASPECT="16:9"
while getopts "o:i:d:a:" opt; do
  case $opt in
    o) OUTPUT="$OPTARG" ;;
    i) INPUT_IMAGE="$OPTARG" ;;
    d) DURATION="$OPTARG" ;;
    a) ASPECT="$OPTARG" ;;
    *) echo "Usage: gen-video.sh [-o out.mp4] [-i image.png] [-d 5] [-a 16:9] <prompt>" >&2; exit 1 ;;
  esac
done
shift $((OPTIND-1))
PROMPT="${1:?Prompt required}"

if [ -z "$OUTPUT" ]; then
  OUTPUT="./veo-$(date +%Y%m%d-%H%M%S).mp4"
fi

# Auto-detect API key
if [ -z "${GEMINI_API_KEY:-}" ]; then
  for rc in ~/.zshrc ~/.zshenv ~/.bashrc ~/.profile; do
    if [ -f "$rc" ]; then
      _key=$(grep -oP "(?<=GEMINI_API_KEY=['\"])[^'\"]+" "$rc" 2>/dev/null \
          || grep "GEMINI_API_KEY=" "$rc" 2>/dev/null | head -1 | sed "s/.*GEMINI_API_KEY=['\"]\\{0,1\\}\\([^'\"]*\\).*/\\1/")
      [ -n "${_key:-}" ] && { export GEMINI_API_KEY="$_key"; break; }
    fi
  done
fi
API_KEY="${GEMINI_API_KEY:?GEMINI_API_KEY not set}"
MODEL="veo-3.1-fast-generate-preview"
BASE="https://generativelanguage.googleapis.com/v1beta"
PROXY="socks5h://127.0.0.1:1080"

mkdir -p "$(dirname "$OUTPUT")"

# Build payload (Python handles base64 for optional image)
PAYLOAD_FILE=$(mktemp /tmp/veo-payload-XXXXXX.json)
RESPONSE_FILE=$(mktemp /tmp/veo-response-XXXXXX.json)
trap "rm -f '$PAYLOAD_FILE' '$RESPONSE_FILE'" EXIT

python3 - <<PYEOF "$PROMPT" "${INPUT_IMAGE:-}" "$PAYLOAD_FILE" "$DURATION" "$ASPECT"
import json, sys, base64, os
prompt, input_image, payload_file, duration, aspect = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
instance = {"prompt": prompt}
if input_image and os.path.exists(input_image):
    ext = os.path.splitext(input_image)[1].lower()
    mime = {".png":"image/png",".jpg":"image/jpeg",".jpeg":"image/jpeg",".webp":"image/webp"}.get(ext,"image/png")
    with open(input_image,"rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    instance["image"] = {"bytesBase64Encoded": b64, "mimeType": mime}
d = {"instances":[instance],"parameters":{"durationSeconds":duration,"aspectRatio":aspect}}
with open(payload_file,"w") as f:
    json.dump(d,f)
PYEOF

echo "→ Submitting to Veo API ($ASPECT, ${DURATION}s)..."
curl -s --socks5-hostname 127.0.0.1:1080 -X POST \
  -H "Content-Type: application/json" \
  "${BASE}/models/${MODEL}:predictLongRunning?key=${API_KEY}" \
  -d @"$PAYLOAD_FILE" -o "$RESPONSE_FILE"

OP=$(python3 -c "import json,sys; d=json.load(open('$RESPONSE_FILE')); print(d.get('name',''))")
if [ -z "$OP" ]; then
  echo "ERROR: no operation name in response:" >&2
  cat "$RESPONSE_FILE" >&2
  exit 1
fi
echo "→ Operation: $OP"

# Poll until done
POLL_URL="${BASE}/${OP}?key=${API_KEY}"
for i in $(seq 1 60); do
  sleep 10
  curl -s --socks5-hostname 127.0.0.1:1080 "$POLL_URL" -o "$RESPONSE_FILE"
  DONE=$(python3 -c "import json; print(json.load(open('$RESPONSE_FILE')).get('done',False))")
  echo "  [${i}0s] done=$DONE"
  [ "$DONE" = "True" ] && break
done

if [ "$DONE" != "True" ]; then
  echo "ERROR: Timeout waiting for video" >&2; exit 1
fi

# Extract and save video
python3 - <<PYEOF "$RESPONSE_FILE" "$OUTPUT"
import json, base64, sys, requests
resp_file, out_path = sys.argv[1], sys.argv[2]
data = json.load(open(resp_file))
if "error" in data:
    print(f"ERROR: {data['error']['message']}", file=sys.stderr); sys.exit(1)
samples = data.get("response",{}).get("generateVideoResponse",{}).get("generatedSamples",[])
if not samples:
    print("ERROR: no generatedSamples", file=sys.stderr)
    print(json.dumps(data, indent=2, ensure_ascii=False)[:1500], file=sys.stderr)
    sys.exit(1)
video = samples[0].get("video", {})
if video.get("bytesBase64Encoded"):
    with open(out_path, "wb") as f: f.write(base64.b64decode(video["bytesBase64Encoded"]))
elif video.get("uri") or video.get("videoUri"):
    import re, os
    vurl = video.get("uri") or video.get("videoUri")
    api_key = os.environ.get("GEMINI_API_KEY", "")
    sep = "&" if "?" in vurl else "?"
    durl = f"{vurl}{sep}key={api_key}" if api_key else vurl
    vr = requests.get(durl, timeout=180,
                      proxies={"https":"socks5h://127.0.0.1:1080","http":"socks5h://127.0.0.1:1080"})
    vr.raise_for_status()
    with open(out_path, "wb") as f: f.write(vr.content)
else:
    print(f"ERROR: no video data in response: {list(video.keys())}", file=sys.stderr); sys.exit(1)
print(f"{out_path} ({os.path.getsize(out_path)//1024}KB)")
PYEOF
