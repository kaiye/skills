#!/usr/bin/env python3
"""
Analyze a reference video using Gemini 2.5 Pro VLM and generate structured prompts for Veo.
Usage: python3 analyze-video.py <video_path> [output_prompt.md]
Output: Structured Veo prompt with scene breakdown, style, lighting, motion, etc.
"""
import base64, json, os, re, sys, pathlib, requests

def get_api_key():
    for rc in ["~/.zshrc","~/.zshenv","~/.bashrc","~/.profile"]:
        p = pathlib.Path(rc).expanduser()
        if p.exists():
            m = re.search(r"GEMINI_API_KEY=['\"]?([^'\"\n]+)", p.read_text(errors='ignore'))
            if m:
                return m.group(1)
    return os.environ.get("GEMINI_API_KEY", "")

def analyze_video(video_path: str) -> str:
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in ~/.zshrc or environment")

    proxy = {"https": "socks5h://127.0.0.1:1080", "http": "socks5h://127.0.0.1:1080"}
    model = "gemini-2.5-pro-preview-03-25"

    with open(video_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode()

    ext = pathlib.Path(video_path).suffix.lower()
    mime_map = {".mp4": "video/mp4", ".mov": "video/quicktime",
                ".avi": "video/avi", ".webm": "video/webm"}
    mime = mime_map.get(ext, "video/mp4")

    system_prompt = """You are a professional video director and Veo prompt engineer.
Your task is to analyze a reference video and generate a detailed, structured prompt that can recreate a similar video using Google's Veo text-to-video model.

Output the prompt in this exact structure:

## Shot Type & Composition
[Describe camera angle, framing, perspective, distance]

## Subject & Action
[Describe the main subject(s), their appearance, and what they're doing]

## Motion & Timing
[Describe movement speed, direction, rhythm, slow-motion if applicable]

## Lighting & Atmosphere
[Describe light source, quality, shadows, color temperature, mood]

## Background & Environment
[Describe the setting, background elements, depth of field]

## Style & Quality
[Describe visual style, realism level, post-processing look, resolution]

## Audio Suggestion
[Suggest sounds or music that would complement this video]

---
## Complete Veo Prompt

[Write a single, cohesive Veo prompt (200-400 words) combining all elements above,
optimized for the Veo model. Start with the most important visual elements.]
"""

    payload = {
        "contents": [{
            "parts": [
                {"text": system_prompt},
                {"inlineData": {"mimeType": mime, "data": video_b64}},
                {"text": "Analyze this video and generate the structured Veo prompt as instructed."}
            ]
        }],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 2048,
            "thinkingConfig": {"thinkingBudget": 1024}
        }
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    resp = requests.post(url, json=payload, proxies=proxy, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        raise RuntimeError(f"API error: {data['error']['message']}")

    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    text = "\n".join(p.get("text", "") for p in parts if "text" in p).strip()
    return text


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <video_path> [output.md]", file=sys.stderr)
        sys.exit(1)

    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(video_path):
        print(f"ERROR: file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    file_mb = os.path.getsize(video_path) / 1024 / 1024
    if file_mb > 20:
        print(f"WARNING: video is {file_mb:.1f}MB, inline upload limit is ~20MB for API.", file=sys.stderr)
        print("Consider trimming to a shorter clip.", file=sys.stderr)

    print(f"Analyzing video: {video_path} ({file_mb:.1f}MB)...", file=sys.stderr)
    result = analyze_video(video_path)

    if output_path:
        pathlib.Path(output_path).write_text(result + "\n")
        print(f"Saved to: {output_path}", file=sys.stderr)
    else:
        print(result)
