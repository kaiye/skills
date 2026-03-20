---
name: veo-video-gen
description: |
  Generate videos using Google Veo API (veo-3.1-fast-generate-preview). Supports three modes:
  (1) text-to-video — user provides a prompt, generate directly;
  (2) video-to-prompt — user uploads a reference video, analyze with Gemini VLM to generate structured Veo prompts, discuss and refine, then generate;
  (3) image-referenced generation — use a reference image to guide composition/style.
  Triggers when user asks to generate a video, create a video from a prompt, use Veo, analyze a video for prompt generation, or produce a short video clip.
  Uses Gemini API key and SOCKS5 proxy automatically.
---

# Veo Video Generation

Generates videos via Google's Veo API. GEMINI_API_KEY auto-detected from `~/.zshrc`. Proxy: `socks5h://127.0.0.1:1080`.

## Skill Directory

```
SKILL_DIR=~/.openclaw/workspace/github-repos/kaiye/skills/veo-video-gen
```

## Mode A — Text-to-Video

User provides a prompt. Ask about:
- **Duration**: 4–8 seconds
- **Aspect ratio**: `16:9` (default), `9:16` (short video), `1:1`
- **Reference image**: "Do you have a reference image for composition or style?"

Then run:
```bash
bash $SKILL_DIR/scripts/gen-video.sh \
  -o /root/.openclaw/workspace/out/veo-OUTPUT.mp4 \
  -d 5 -a 16:9 \
  "YOUR PROMPT HERE"
```

With reference image:
```bash
bash $SKILL_DIR/scripts/gen-video.sh \
  -i /path/to/reference.png \
  -o /root/.openclaw/workspace/out/veo-OUTPUT.mp4 \
  -d 5 -a 16:9 \
  "YOUR PROMPT HERE"
```

## Mode B — Video Analysis → Prompt → Generate

User uploads a reference video. Workflow:

**Step 1: Analyze video**
```bash
python3 $SKILL_DIR/scripts/analyze-video.py /path/to/input.mp4 /tmp/veo-prompt.md
cat /tmp/veo-prompt.md
```

**Step 2: Present structured prompt to user**
Show the full analysis and the "Complete Veo Prompt" section. Ask:
- "Does this capture the style/mood you want?"
- "What should we change — shot type, motion speed, lighting?"

**Step 3: Discuss and refine**
Iterate on the prompt with the user. Key dimensions to adjust:
- Shot type & framing
- Motion descriptors (slow motion, tracking, etc.)
- Lighting & atmosphere
- Style modifiers

**Step 4: Generate**
Once prompt is confirmed, ask about duration/aspect ratio, then run `gen-video.sh`.

## Mode C — Image-Referenced Generation

Can be combined with either Mode A or B. When image is provided:
- Serves as **composition, lighting, or style reference** (not strict adherence)
- Works best for: color palette, framing, object placement
- Pass with `-i` flag to `gen-video.sh`

## Output Delivery

After generation:
1. Copy to web-accessible path: `cp output.mp4 /root/xray/caddy_xray/www/manbo/demo/FILENAME.mp4`
2. Share URL: `https://manbo.im/demo/FILENAME.mp4`

## Prompt Writing Reference

See `references/veo-prompt-guide.md` for:
- Shot type vocabulary
- Motion descriptors
- Lighting terms
- Style modifiers
- Example prompt structures

## Troubleshooting

| Error | Fix |
|-------|-----|
| `durationSeconds out of bound` | Use 4–8 (inclusive) |
| `RESOURCE_EXHAUSTED` | Rate limit — wait 30s and retry |
| `User location not supported` | Ensure SOCKS5 proxy is active on port 1080 |
| No video in response | Check `generatedSamples` key; may need to retry |
