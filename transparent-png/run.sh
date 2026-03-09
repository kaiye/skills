#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run.sh -i <input.png> -o <output.png> [--fuzz <percent>]

Notes:
  - Default fuzz is 25.
  - Requires ImageMagick 7 (magick) and Python3 + Pillow.
EOF
}

INPUT=""
OUTPUT=""
FUZZ="25"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input)
      INPUT="$2"; shift 2 ;;
    -o|--output)
      OUTPUT="$2"; shift 2 ;;
    --fuzz)
      FUZZ="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$INPUT" || -z "$OUTPUT" ]]; then
  usage
  exit 2
fi

if ! command -v magick >/dev/null 2>&1; then
  echo "Error: 'magick' not found. Please install ImageMagick 7." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mapfile -t COLORS < <(python3 "$SCRIPT_DIR/scripts/bg2colors.py" "$INPUT")
C1="${COLORS[0]}"
C2="${COLORS[1]}"

# 8 boundary seed points: 4 corners + 4 edge midpoints.
# Apply alpha floodfill for both clustered background colors.
magick "$INPUT" -alpha set -fuzz "${FUZZ}%" -fill none \
  -draw "alpha 0,0 floodfill ${C1}" -draw "alpha 0,0 floodfill ${C2}" \
  -draw "alpha 0,%[fx:h/2] floodfill ${C1}" -draw "alpha 0,%[fx:h/2] floodfill ${C2}" \
  -draw "alpha 0,%[fx:h-1] floodfill ${C1}" -draw "alpha 0,%[fx:h-1] floodfill ${C2}" \
  -draw "alpha %[fx:w/2],0 floodfill ${C1}" -draw "alpha %[fx:w/2],0 floodfill ${C2}" \
  -draw "alpha %[fx:w-1],0 floodfill ${C1}" -draw "alpha %[fx:w-1],0 floodfill ${C2}" \
  -draw "alpha %[fx:w-1],%[fx:h/2] floodfill ${C1}" -draw "alpha %[fx:w-1],%[fx:h/2] floodfill ${C2}" \
  -draw "alpha %[fx:w/2],%[fx:h-1] floodfill ${C1}" -draw "alpha %[fx:w/2],%[fx:h-1] floodfill ${C2}" \
  -draw "alpha %[fx:w-1],%[fx:h-1] floodfill ${C1}" -draw "alpha %[fx:w-1],%[fx:h-1] floodfill ${C2}" \
  "$OUTPUT"

echo "OK: wrote $OUTPUT (bg colors: $C1 $C2, fuzz=${FUZZ}%)"