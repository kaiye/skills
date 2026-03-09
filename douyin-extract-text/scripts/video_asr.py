# /// script
# requires-python = ">=3.10"
# dependencies = ["requests", "dashscope"]
# ///
"""
视频 ASR 语音识别工具

使用阿里云 DashScope API 进行语音识别。
支持在线识别（短视频）和本地处理（长视频自动切分 → 提取音频 → ASR → 合并）。

用法:
    uv run video_asr.py "抖音分享链接" -o ./output
    uv run video_asr.py "抖音分享链接" --force-local -o ./output
    uv run video_asr.py --local /path/to/video.mp4 -o ./output
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from http import HTTPStatus
from pathlib import Path
from typing import Dict, List
from urllib import request
from concurrent.futures import ThreadPoolExecutor, as_completed

import dashscope

# 将脚本所在目录加入 sys.path，以便导入 video_common
sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_common import (
    HEADERS,
    parse_douyin_url,
    download_video,
    get_video_duration,
    split_video,
    extract_audio,
)

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "paraformer-v2"
DEFAULT_SEGMENT_DURATION = 300  # 5 分钟
MAX_PARALLEL_TRANSCRIPTIONS = 3


# ---------------------------------------------------------------------------
# API Key 自动检测
# ---------------------------------------------------------------------------

def auto_detect_api_key() -> str:
    """从环境变量或 shell 配置文件中自动检测 DASHSCOPE_API_KEY。"""
    key = os.environ.get("DASHSCOPE_API_KEY", "")
    if key:
        return key

    for rc in ("~/.zshrc", "~/.zshenv", "~/.bashrc", "~/.bash_profile", "~/.profile"):
        rc_path = Path(rc).expanduser()
        if not rc_path.is_file():
            continue
        try:
            for line in rc_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("#"):
                    continue
                m = re.search(r"""DASHSCOPE_API_KEY=['\"]?([^'\"#\s]+)""", line)
                if m:
                    return m.group(1)
        except OSError:
            continue

    return ""


# ---------------------------------------------------------------------------
# ASR 核心功能
# ---------------------------------------------------------------------------

def transcribe_video_url(video_url: str, model: str = DEFAULT_MODEL) -> str:
    """直接通过视频 URL 进行转录（适用于短视频）。"""
    print("🎤 正在进行在线语音识别...")

    task_response = dashscope.audio.asr.Transcription.async_call(
        model=model,
        file_urls=[video_url],
        language_hints=["zh", "en"],
    )

    transcription_response = dashscope.audio.asr.Transcription.wait(
        task=task_response.output.task_id
    )

    if transcription_response.status_code == HTTPStatus.OK:
        for transcription in transcription_response.output["results"]:
            url = transcription["transcription_url"]
            result = json.loads(request.urlopen(url).read().decode("utf8"))
            if "transcripts" in result and len(result["transcripts"]) > 0:
                return result["transcripts"][0]["text"]
        return "未识别到文本内容"
    else:
        raise RuntimeError(f"转录失败: {transcription_response.output.message}")


def transcribe_audio(
    audio_path: Path,
    segment_index: int = 0,
    model: str = DEFAULT_MODEL,
) -> Dict:
    """转录单个音频文件。"""
    print(f"🎤 正在识别片段 {segment_index + 1}: {audio_path.name}")

    try:
        task_response = dashscope.audio.asr.Transcription.async_call(
            model=model,
            file_urls=[str(audio_path.absolute())],
            language_hints=["zh", "en"],
        )

        transcription_response = dashscope.audio.asr.Transcription.wait(
            task=task_response.output.task_id
        )

        if transcription_response.status_code == HTTPStatus.OK:
            results = transcription_response.output.get("results", [])
            for transcription in results:
                if "transcription_url" in transcription:
                    url = transcription["transcription_url"]
                    result = json.loads(request.urlopen(url).read().decode("utf8"))
                    if "transcripts" in result and len(result["transcripts"]) > 0:
                        text = result["transcripts"][0]["text"]
                        return {
                            "segment_index": segment_index,
                            "status": "success",
                            "text": text,
                            "file": str(audio_path),
                        }

            return {
                "segment_index": segment_index,
                "status": "success",
                "text": "",
                "file": str(audio_path),
            }
        else:
            return {
                "segment_index": segment_index,
                "status": "error",
                "text": "",
                "error": transcription_response.output.get("message", "未知错误"),
                "file": str(audio_path),
            }

    except Exception as e:
        return {
            "segment_index": segment_index,
            "status": "error",
            "text": "",
            "error": str(e),
            "file": str(audio_path),
        }


# ---------------------------------------------------------------------------
# 长视频处理
# ---------------------------------------------------------------------------

def process_long_video_asr(
    video_path: Path,
    output_dir: Path,
    model: str = DEFAULT_MODEL,
    segment_duration: int = DEFAULT_SEGMENT_DURATION,
) -> tuple[str, List[Dict]]:
    """长视频本地处理：切分 → 提取音频 → ASR → 合并。"""

    # 1. 切分视频
    segments = split_video(video_path, segment_duration, output_dir)

    # 2. 提取每个片段的音频
    print("\n🔊 正在提取音频...")
    audio_files: list[tuple[int, Path]] = []
    for i, segment in enumerate(segments):
        audio_path = extract_audio(segment)
        audio_files.append((i, audio_path))
        print(f"   音频 {i + 1}/{len(segments)}: {audio_path.name}")

    # 3. 并行转录
    print("\n🎤 正在进行语音识别...")
    results: List[Dict] = []

    if len(audio_files) > 1:
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_TRANSCRIPTIONS) as executor:
            futures = {
                executor.submit(transcribe_audio, audio_path, idx, model): idx
                for idx, audio_path in audio_files
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                status = "✅" if result["status"] == "success" else "❌"
                print(f"   {status} 片段 {result['segment_index'] + 1} 识别完成")
    else:
        for idx, audio_path in audio_files:
            result = transcribe_audio(audio_path, idx, model)
            results.append(result)
            status = "✅" if result["status"] == "success" else "❌"
            print(f"   {status} 片段 {idx + 1} 识别完成")

    # 4. 按顺序合并结果
    results.sort(key=lambda x: x["segment_index"])
    full_text = "\n\n".join(
        r["text"] for r in results if r["status"] == "success" and r["text"]
    )

    return full_text, results


# ---------------------------------------------------------------------------
# 主处理流程
# ---------------------------------------------------------------------------

def process_asr(
    share_text: str,
    output_dir: str,
    *,
    force_local: bool = False,
    info_only: bool = False,
    local_file: bool = False,
    model: str = DEFAULT_MODEL,
    segment_duration: int = DEFAULT_SEGMENT_DURATION,
    keep_files: bool = False,
) -> None:
    """ASR 完整处理流程。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if local_file:
        # 处理本地视频文件
        video_path = Path(share_text)
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        print(f"📹 处理本地视频: {video_path}")
        full_text, segment_results = process_long_video_asr(
            video_path, out, model, segment_duration
        )

        text_file = out / f"{video_path.stem}_asr.txt"
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(f"源文件: {video_path}\n")
            f.write(f"{'=' * 50}\n\n")
            f.write(full_text)

        _print_result(full_text, text_file)
        return

    # 1. 解析抖音链接
    video_info = parse_douyin_url(share_text)

    if info_only:
        print("\n📋 视频信息:")
        print(json.dumps(video_info, ensure_ascii=False, indent=2))
        return

    # 2. 先尝试在线识别（短视频）
    if not force_local:
        try:
            print("\n🚀 尝试直接在线识别...")
            text = transcribe_video_url(video_info["url"], model)
            text_file = out / f"{video_info['video_id']}_asr.txt"
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(f"标题: {video_info['title']}\n")
                f.write(f"视频ID: {video_info['video_id']}\n")
                f.write(f"{'=' * 50}\n\n")
                f.write(text)
            _print_result(text, text_file)
            return
        except Exception as e:
            print(f"⚠️  在线识别失败: {e}")
            print("   切换到本地处理模式...")

    # 3. 本地处理
    print("\n📥 启动本地 ASR 处理模式...")
    video_path = download_video(
        video_info["url"], out, f"{video_info['video_id']}.mp4"
    )

    full_text, segment_results = process_long_video_asr(
        video_path, out, model, segment_duration
    )

    # 保存结果
    result_data = {
        "status": "success",
        "video_id": video_info["video_id"],
        "title": video_info["title"],
        "text": full_text,
        "method": "local_asr",
        "segments": segment_results,
        "output_dir": str(out),
    }

    result_file = out / f"{video_info['video_id']}_asr_result.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    text_file = out / f"{video_info['video_id']}_asr.txt"
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(f"标题: {video_info['title']}\n")
        f.write(f"视频ID: {video_info['video_id']}\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(full_text)

    _print_result(full_text, text_file)

    # 清理临时文件
    if not keep_files:
        segments_dir = out / "segments"
        if segments_dir.exists():
            shutil.rmtree(segments_dir)
            print("🧹 已清理临时片段文件")


def _print_result(text: str, text_file: Path) -> None:
    print(f"\n📝 转录结果已保存: {text_file}")
    print("\n" + "=" * 60)
    print("📜 识别结果:")
    print("=" * 60)
    print(text)
    print("=" * 60)
    print("\n✅ 处理完成!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="视频 ASR 语音识别工具 — 使用阿里云 DashScope API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
使用示例:
  uv run video_asr.py "抖音分享链接" -o ./output
  uv run video_asr.py "抖音分享链接" --force-local -o ./output
  uv run video_asr.py --local /path/to/video.mp4 -o ./output
  uv run video_asr.py "抖音分享链接" --info-only

环境变量:
  DASHSCOPE_API_KEY  阿里云百炼 API 密钥（自动从 ~/.zshrc 等配置文件读取）
""",
    )

    parser.add_argument("input", nargs="?", help="抖音分享链接或本地视频路径")
    parser.add_argument("--local", "-l", action="store_true", help="处理本地视频文件")
    parser.add_argument("--force-local", "-f", action="store_true", help="强制本地下载处理（跳过在线识别）")
    parser.add_argument("--output", "-o", type=str, default="./output", help="输出目录（默认 ./output）")
    parser.add_argument("--segment-duration", "-s", type=int, default=DEFAULT_SEGMENT_DURATION, help=f"切分时长（秒），默认 {DEFAULT_SEGMENT_DURATION}")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL, help=f"语音识别模型，默认 {DEFAULT_MODEL}")
    parser.add_argument("--keep-files", "-k", action="store_true", help="保留中间文件")
    parser.add_argument("--info-only", action="store_true", help="仅获取视频信息，不下载和识别")

    args = parser.parse_args()

    if not args.input:
        parser.print_help()
        sys.exit(1)

    # 自动检测 API Key
    api_key = auto_detect_api_key()
    if not api_key and not args.info_only:
        print("❌ 错误: 未找到 DASHSCOPE_API_KEY")
        print("   请在 shell 配置文件中设置:")
        print("   export DASHSCOPE_API_KEY='your-api-key'")
        print("   API 密钥获取: https://bailian.console.aliyun.com/")
        sys.exit(1)

    if api_key:
        dashscope.api_key = api_key

    try:
        process_asr(
            args.input,
            args.output,
            force_local=args.force_local,
            info_only=args.info_only,
            local_file=args.local,
            model=args.model,
            segment_duration=args.segment_duration,
            keep_files=args.keep_files,
        )
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
