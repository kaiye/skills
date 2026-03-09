# /// script
# requires-python = ">=3.10"
# dependencies = ["requests", "rapidocr-onnxruntime", "opencv-python", "numpy"]
# ///
"""
视频 OCR 硬字幕提取工具

使用 RapidOCR (ONNX Runtime) 识别视频中的硬字幕。
无需 API 密钥，纯本地处理。

用法:
    uv run video_ocr.py "抖音分享链接" -o ./output
    uv run video_ocr.py --local /path/to/video.mp4 -o ./output
    uv run video_ocr.py "链接" --subtitle-area 0.7,0.95,0.05,0.95 -o ./output
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# 将脚本所在目录加入 sys.path，以便导入 video_common
sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_common import (
    parse_douyin_url,
    download_video,
    get_video_duration,
    split_video,
)

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
OCR_FRAME_INTERVAL = 1.0         # 每隔多少秒采样一帧
OCR_SIMILARITY_THRESHOLD = 0.85  # 文本相似度阈值，用于去重
OCR_SEGMENT_DURATION = 60        # OCR 模式视频切分时长（秒）
OCR_PARALLEL_WORKERS = 4         # OCR 并行进程数

# 默认字幕区域：画面底部 70%-95%，左右各留 5%
DEFAULT_SUBTITLE_AREA = (0.70, 0.95, 0.05, 0.95)


# ---------------------------------------------------------------------------
# RapidOCR 兼容层
# ---------------------------------------------------------------------------

def _run_ocr(engine, image):
    """调用 RapidOCR 并返回 (texts, scores)，兼容不同版本 API。"""
    result = engine(image)

    # 新版 API (2.x): 返回对象，有 .txts / .scores 属性
    if hasattr(result, "txts"):
        texts = list(result.txts or ())
        scores = list(result.scores or ())
        return texts, scores

    # 旧版 API (1.x): 返回 (result_list, elapse) 元组
    if isinstance(result, tuple):
        items = result[0]
        if items:
            texts = [r[1] for r in items]
            scores = [r[2] for r in items]
            return texts, scores
        return [], []

    return [], []


# ---------------------------------------------------------------------------
# 多进程 worker（顶层函数，可被 pickle）
# ---------------------------------------------------------------------------

def _process_segment_ocr(args: tuple) -> tuple[int, list]:
    """处理单个视频片段的 OCR（用于多进程）。

    Args:
        args: (segment_path, segment_index, time_offset, subtitle_area)

    Returns:
        (segment_index, subtitles_list)
    """
    segment_path, segment_index, time_offset, subtitle_area = args

    # 每个进程独立创建 RapidOCR 实例
    from rapidocr_onnxruntime import RapidOCR
    engine = RapidOCR()

    cap = cv2.VideoCapture(str(segment_path))
    if not cap.isOpened():
        return (segment_index, [])

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    sub_area = subtitle_area or DEFAULT_SUBTITLE_AREA

    def extract_text(frame):
        """从帧中提取文字（支持动态区域检测）。"""
        h, w = frame.shape[:2]
        
        # 先尝试底部字幕区域
        y1, y2 = int(h * sub_area[0]), int(h * sub_area[1])
        x1, x2 = int(w * sub_area[2]), int(w * sub_area[3])
        region = frame[y1:y2, x1:x2]

        # 预处理：灰度 → CLAHE → 二值化
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, binary = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY)
        processed = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        texts, scores = _run_ocr(engine, processed)
        # 仅保留置信度 > 0.7 的文本
        filtered = [t for t, s in zip(texts, scores) if s > 0.7]
        
        # 如果底部没识别到，尝试画面中心区域
        if not filtered:
            # 中心区域：垂直 30%-70%，水平 10%-90%
            y1_center, y2_center = int(h * 0.3), int(h * 0.7)
            x1_center, x2_center = int(w * 0.1), int(w * 0.9)
            region_center = frame[y1_center:y2_center, x1_center:x2_center]
            
            gray_center = cv2.cvtColor(region_center, cv2.COLOR_BGR2GRAY)
            enhanced_center = clahe.apply(gray_center)
            _, binary_center = cv2.threshold(enhanced_center, 200, 255, cv2.THRESH_BINARY)
            processed_center = cv2.cvtColor(binary_center, cv2.COLOR_GRAY2BGR)
            
            texts_center, scores_center = _run_ocr(engine, processed_center)
            filtered = [t for t, s in zip(texts_center, scores_center) if s > 0.7]
        
        return " ".join(filtered)

    def is_similar(t1: str, t2: str) -> bool:
        if not t1 or not t2:
            return False
        return SequenceMatcher(None, t1, t2).ratio() >= OCR_SIMILARITY_THRESHOLD

    # 按时间采样处理
    subtitles: list[dict] = []
    current_text = ""
    current_start = 0.0
    t = 0.0

    while t < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            t += OCR_FRAME_INTERVAL
            continue

        text = extract_text(frame)

        if text and not is_similar(text, current_text):
            if current_text:
                subtitles.append({
                    "start_time": time_offset + current_start,
                    "end_time": time_offset + t,
                    "text": current_text,
                })
            current_text = text
            current_start = t
        elif not text and current_text:
            subtitles.append({
                "start_time": time_offset + current_start,
                "end_time": time_offset + t,
                "text": current_text,
            })
            current_text = ""

        t += OCR_FRAME_INTERVAL

    if current_text:
        subtitles.append({
            "start_time": time_offset + current_start,
            "end_time": time_offset + duration,
            "text": current_text,
        })

    cap.release()
    print(f"   ✅ 片段 {segment_index + 1} 完成，提取 {len(subtitles)} 条字幕")
    return (segment_index, subtitles)


# ---------------------------------------------------------------------------
# 字幕工具函数
# ---------------------------------------------------------------------------

def _subtitles_to_text(subtitles: List[Dict]) -> str:
    """将字幕列表合并为纯文本（去重）。"""
    unique_texts: list[str] = []
    for sub in subtitles:
        text = sub["text"].strip()
        if text and (
            not unique_texts
            or SequenceMatcher(None, text, unique_texts[-1]).ratio() < OCR_SIMILARITY_THRESHOLD
        ):
            unique_texts.append(text)
    return "\n".join(unique_texts)


def _subtitles_to_srt(subtitles: List[Dict]) -> str:
    """将字幕列表转换为 SRT 格式。"""
    def fmt(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines: list[str] = []
    for i, sub in enumerate(subtitles, 1):
        lines.append(str(i))
        lines.append(f"{fmt(sub['start_time'])} --> {fmt(sub['end_time'])}")
        lines.append(sub["text"])
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# OCR 主处理
# ---------------------------------------------------------------------------

def process_video_ocr(
    video_path: Path,
    output_dir: Path,
    subtitle_area: Optional[Tuple[float, float, float, float]] = None,
) -> tuple[str, list]:
    """使用 OCR 提取视频硬字幕（支持并行处理）。"""
    print("\n🔤 正在使用 OCR 提取硬字幕...")

    # 1. 切分视频（复用 video_common.split_video，OCR 用更短的片段）
    segments = split_video(video_path, OCR_SEGMENT_DURATION, output_dir)

    # 2. 准备并行任务参数（从索引计算时间偏移）
    area = subtitle_area or DEFAULT_SUBTITLE_AREA
    tasks = [
        (str(seg_path), idx, idx * OCR_SEGMENT_DURATION, area)
        for idx, seg_path in enumerate(segments)
    ]

    # 3. 并行处理
    num_workers = min(OCR_PARALLEL_WORKERS, len(segments))
    print(f"\n🚀 使用 {num_workers} 进程并行处理...")

    all_subtitles: list[dict] = []
    if len(segments) == 1:
        _, subs = _process_segment_ocr(tasks[0])
        all_subtitles = subs
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_process_segment_ocr, tasks))
        results.sort(key=lambda x: x[0])
        for _, subs in results:
            all_subtitles.extend(subs)

    print(f"\n✅ 识别完成，共提取 {len(all_subtitles)} 条字幕")

    # 4. 转换为纯文本
    full_text = _subtitles_to_text(all_subtitles)

    # 5. 保存 SRT 文件
    srt_content = _subtitles_to_srt(all_subtitles)
    srt_file = output_dir / f"{video_path.stem}.srt"
    with open(srt_file, "w", encoding="utf-8") as f:
        f.write(srt_content)
    print(f"📄 SRT 字幕已保存: {srt_file}")

    # 清理临时片段
    segments_dir = output_dir / "segments"
    if segments_dir.exists():
        shutil.rmtree(segments_dir)
        print("🧹 已清理临时片段文件")

    return full_text, all_subtitles


# ---------------------------------------------------------------------------
# 主处理流程
# ---------------------------------------------------------------------------

def process_ocr(
    share_text: str,
    output_dir: str,
    *,
    local_file: bool = False,
    info_only: bool = False,
    subtitle_area: Optional[Tuple[float, float, float, float]] = None,
    keep_files: bool = False,
) -> None:
    """OCR 完整处理流程。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if local_file:
        video_path = Path(share_text)
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        print(f"📹 处理本地视频: {video_path}")
        full_text, all_subtitles = process_video_ocr(video_path, out, subtitle_area)

        text_file = out / f"{video_path.stem}_ocr.txt"
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

    # 2. 下载视频（OCR 必须本地处理）
    print("\n📥 启动本地 OCR 处理模式...")
    video_path = download_video(
        video_info["url"], out, f"{video_info['video_id']}.mp4"
    )

    full_text, all_subtitles = process_video_ocr(video_path, out, subtitle_area)

    # 保存结果
    result_data = {
        "status": "success",
        "video_id": video_info["video_id"],
        "title": video_info["title"],
        "text": full_text,
        "method": "local_ocr",
        "output_dir": str(out),
    }

    result_file = out / f"{video_info['video_id']}_ocr_result.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    text_file = out / f"{video_info['video_id']}_ocr.txt"
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(f"标题: {video_info['title']}\n")
        f.write(f"视频ID: {video_info['video_id']}\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(full_text)

    _print_result(full_text, text_file)


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

def _parse_subtitle_area(value: str) -> Tuple[float, float, float, float]:
    """解析字幕区域参数，格式: y_start,y_end,x_start,x_end"""
    parts = [float(x.strip()) for x in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("字幕区域格式: y_start,y_end,x_start,x_end（如 0.7,0.95,0.05,0.95）")
    return (parts[0], parts[1], parts[2], parts[3])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="视频 OCR 硬字幕提取工具 — 使用 RapidOCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
使用示例:
  uv run video_ocr.py "抖音分享链接" -o ./output
  uv run video_ocr.py --local /path/to/video.mp4 -o ./output
  uv run video_ocr.py "链接" --subtitle-area 0.5,0.8,0.1,0.9 -o ./output

无需 API 密钥，纯本地处理。
RapidOCR 默认支持中英文混合识别。
""",
    )

    parser.add_argument("input", nargs="?", help="抖音分享链接或本地视频路径")
    parser.add_argument("--local", "-l", action="store_true", help="处理本地视频文件")
    parser.add_argument("--output", "-o", type=str, default="./output", help="输出目录（默认 ./output）")
    parser.add_argument("--info-only", action="store_true", help="仅获取视频信息")
    parser.add_argument("--keep-files", "-k", action="store_true", help="保留中间文件")
    parser.add_argument(
        "--subtitle-area",
        type=_parse_subtitle_area,
        default=None,
        metavar="y1,y2,x1,x2",
        help="自定义字幕区域（默认 0.7,0.95,0.05,0.95），格式: y_start,y_end,x_start,x_end",
    )

    args = parser.parse_args()

    if not args.input:
        parser.print_help()
        sys.exit(1)

    try:
        process_ocr(
            args.input,
            args.output,
            local_file=args.local,
            info_only=args.info_only,
            subtitle_area=args.subtitle_area,
            keep_files=args.keep_files,
        )
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
