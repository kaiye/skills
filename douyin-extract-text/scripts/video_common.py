"""
视频处理共享模块

提供视频下载、切分、音频提取等通用功能。
支持抖音分享链接解析。
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import requests

# 请求头，模拟移动端访问
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "EdgiOS/121.0.2277.107 Version/17.0 Mobile/15E148 Safari/604.1"
    )
}


# ---------------------------------------------------------------------------
# 抖音链接解析
# ---------------------------------------------------------------------------

def parse_douyin_url(share_text: str) -> Dict:
    """从抖音分享文本中解析无水印视频信息。

    Returns:
        {"url": video_url, "title": desc, "video_id": video_id}
    """
    print("🔍 正在解析抖音分享链接...")

    # 提取分享链接
    urls = re.findall(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        share_text,
    )
    if not urls:
        raise ValueError("未找到有效的分享链接")

    share_url = urls[0]
    share_response = requests.get(share_url, headers=HEADERS)
    video_id = share_response.url.split("?")[0].strip("/").split("/")[-1]
    share_url = f"https://www.iesdouyin.com/share/video/{video_id}"

    # 获取视频页面内容
    response = requests.get(share_url, headers=HEADERS)
    response.raise_for_status()

    pattern = re.compile(
        r"window\._ROUTER_DATA\s*=\s*(.*?)</script>",
        flags=re.DOTALL,
    )
    find_res = pattern.search(response.text)

    if not find_res or not find_res.group(1):
        raise ValueError("从 HTML 中解析视频信息失败")

    json_data = json.loads(find_res.group(1).strip())
    VIDEO_ID_PAGE_KEY = "video_(id)/page"
    NOTE_ID_PAGE_KEY = "note_(id)/page"

    if VIDEO_ID_PAGE_KEY in json_data["loaderData"]:
        original_video_info = json_data["loaderData"][VIDEO_ID_PAGE_KEY]["videoInfoRes"]
    elif NOTE_ID_PAGE_KEY in json_data["loaderData"]:
        original_video_info = json_data["loaderData"][NOTE_ID_PAGE_KEY]["videoInfoRes"]
    else:
        raise ValueError("无法从 JSON 中解析视频或图集信息")

    data = original_video_info["item_list"][0]

    video_url = data["video"]["play_addr"]["url_list"][0].replace("playwm", "play")
    desc = data.get("desc", "").strip() or f"douyin_{video_id}"
    # 替换文件名中的非法字符
    desc = re.sub(r'[\\/:*?"<>|]', "_", desc)

    result = {
        "url": video_url,
        "title": desc,
        "video_id": video_id,
    }

    print(f"✅ 解析成功: {desc}")
    print(f"   视频ID: {video_id}")
    return result


# ---------------------------------------------------------------------------
# 视频下载
# ---------------------------------------------------------------------------

def download_video(
    url: str,
    output_dir: str | Path,
    filename: str = "video.mp4",
    headers: Optional[Dict] = None,
) -> Path:
    """下载视频到本地。

    Returns:
        下载后的文件路径
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename

    print(f"⬇️  正在下载视频...")

    response = requests.get(url, headers=headers or HEADERS, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = downloaded / total_size * 100
                    print(f"\r   下载进度: {progress:.1f}%", end="", flush=True)

    print(f"\n✅ 视频下载完成: {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# ffprobe / ffmpeg 工具
# ---------------------------------------------------------------------------

def get_video_duration(video_path: str | Path) -> float:
    """获取视频时长（秒），依赖 ffprobe。"""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def split_video(
    video_path: str | Path,
    segment_duration: int,
    output_dir: str | Path,
) -> List[Path]:
    """将长视频切分为多个片段（无损 copy）。

    Returns:
        切分后的片段文件路径列表
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    duration = get_video_duration(video_path)
    print(f"📊 视频总时长: {duration:.1f}秒 ({duration / 60:.1f}分钟)")

    if duration <= segment_duration:
        print("   视频较短，无需切分")
        return [video_path]

    num_segments = int(duration / segment_duration) + 1
    print(f"✂️  正在将视频切分为 {num_segments} 个片段...")

    segments_dir = output_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    segment_paths: List[Path] = []
    for i in range(num_segments):
        start_time = i * segment_duration
        segment_filename = f"segment_{i:03d}.mp4"
        segment_path = segments_dir / segment_filename

        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-ss", str(start_time),
            "-t", str(segment_duration),
            "-c", "copy",
            str(segment_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True)

        if segment_path.exists() and segment_path.stat().st_size > 0:
            segment_paths.append(segment_path)
            print(f"   片段 {i + 1}/{num_segments}: {segment_filename}")

    print(f"✅ 切分完成，共 {len(segment_paths)} 个片段")
    return segment_paths


def extract_audio(video_path: str | Path) -> Path:
    """从视频文件中提取音频（MP3 格式）。"""
    video_path = Path(video_path)
    audio_path = video_path.with_suffix(".mp3")

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn",
        "-acodec", "libmp3lame",
        "-q:a", "0",
        str(audio_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return audio_path
