#!/usr/bin/env python3
"""
抖音长视频本地处理工具

功能：
1. 解析抖音分享链接获取无水印视频
2. 下载视频到本地
3. 长视频自动切分为片段
4. 支持两种文案提取方式：
   - ASR 语音识别（默认）
   - OCR 硬字幕提取（更准确，需要视频有字幕）
5. 合并所有识别结果

使用方法：
    python douyin_video_processor.py "抖音分享链接" [选项]
    python douyin_video_processor.py "抖音分享链接" --ocr  # 使用OCR提取字幕

环境变量：
    DASHSCOPE_API_KEY: 阿里云百炼API密钥（ASR模式需要）
"""

from __future__ import annotations

import subprocess
import sys

# ============================================================
# 依赖自动检测与安装
# ============================================================
def ensure_dependencies(use_ocr=False):
    """检查并安装缺失的依赖（首次运行时自动安装）"""
    # 基础依赖（ASR 模式）
    REQUIRED = {
        "dashscope": "dashscope",
        "requests": "requests",
    }

    # OCR 模式额外依赖
    if use_ocr:
        REQUIRED.update({
            "cv2": "opencv-python",
            "numpy": "numpy",
            "paddleocr": "paddleocr",
            "paddle": "paddlepaddle",
        })

    missing = []
    for module, package in REQUIRED.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"📦 正在安装缺失的依赖: {missing}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q"] + missing,
        )
        print("✅ 依赖安装完成\n")

# 检查是否使用 OCR 模式（在 argparse 之前简单检查）
_use_ocr = "--ocr" in sys.argv
ensure_dependencies(use_ocr=_use_ocr)
# ============================================================

import os
import re
import json
import argparse
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from urllib import request
from http import HTTPStatus
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from difflib import SequenceMatcher
from multiprocessing import Manager

# 现在可以安全导入所有依赖
import requests
import dashscope

# OCR 相关依赖延迟导入
if _use_ocr:
    import cv2
    import numpy as np
    from paddleocr import PaddleOCR

# 请求头，模拟移动端访问
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) EdgiOS/121.0.2277.107 Version/17.0 Mobile/15E148 Safari/604.1'
}

# 默认配置
DEFAULT_MODEL = "paraformer-v2"
DEFAULT_SEGMENT_DURATION = 300  # 默认5分钟一个片段
MAX_PARALLEL_TRANSCRIPTIONS = 3  # 最大并行转录数

# OCR 配置
OCR_FRAME_INTERVAL = 1.0  # 每隔多少秒采样一帧
OCR_SIMILARITY_THRESHOLD = 0.85  # 文本相似度阈值，用于去重
OCR_SEGMENT_DURATION = 60  # OCR 模式视频切分时长（秒）
OCR_PARALLEL_WORKERS = 4  # OCR 并行进程数


def _process_segment_ocr(args):
    """
    处理单个视频片段的 OCR（用于多进程）

    Args:
        args: (segment_path, segment_index, time_offset, lang, subtitle_area)

    Returns:
        (segment_index, subtitles_list)
    """
    segment_path, segment_index, time_offset, lang, subtitle_area = args

    # 每个进程独立创建 OCR 实例
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang=lang)

    # 打开视频
    cap = cv2.VideoCapture(str(segment_path))
    if not cap.isOpened():
        return (segment_index, [])

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    # 字幕区域
    sub_area = subtitle_area or (0.70, 0.95, 0.05, 0.95)

    def extract_text(frame):
        """从帧中提取文字"""
        h, w = frame.shape[:2]
        # 裁剪字幕区域
        y1, y2 = int(h * sub_area[0]), int(h * sub_area[1])
        x1, x2 = int(w * sub_area[2]), int(w * sub_area[3])
        region = frame[y1:y2, x1:x2]

        # 预处理
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, binary = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY)
        processed = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        # OCR
        result = ocr.predict(processed)
        if not result or len(result) == 0:
            return ""

        res = result[0]
        texts = [t for t, s in zip(res.get('rec_texts', []), res.get('rec_scores', [])) if s > 0.7]
        return " ".join(texts)

    def is_similar(t1, t2):
        if not t1 or not t2:
            return False
        return SequenceMatcher(None, t1, t2).ratio() >= OCR_SIMILARITY_THRESHOLD

    # 采样处理
    subtitles = []
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
                    "text": current_text
                })
            current_text = text
            current_start = t
        elif not text and current_text:
            subtitles.append({
                "start_time": time_offset + current_start,
                "end_time": time_offset + t,
                "text": current_text
            })
            current_text = ""

        t += OCR_FRAME_INTERVAL

    if current_text:
        subtitles.append({
            "start_time": time_offset + current_start,
            "end_time": time_offset + duration,
            "text": current_text
        })

    cap.release()
    print(f"   ✅ 片段 {segment_index + 1} 完成，提取 {len(subtitles)} 条字幕")
    return (segment_index, subtitles)


class SubtitleOCR:
    """视频硬字幕 OCR 提取器"""

    def __init__(
        self,
        use_gpu: bool = False,
        lang: str = 'ch',
        subtitle_area: Optional[Tuple[float, float, float, float]] = None
    ):
        """
        初始化 OCR 提取器

        Args:
            use_gpu: 是否使用 GPU 加速
            lang: OCR 语言，'ch' 中文, 'en' 英文
            subtitle_area: 字幕区域 (y_start_ratio, y_end_ratio, x_start_ratio, x_end_ratio)
                          例如 (0.75, 0.95, 0.1, 0.9) 表示画面下方 75%-95% 高度，左右各留 10%
        """
        # PaddleOCR 3.x API 变化：移除了 show_log, use_angle_cls, use_gpu 等参数
        # GPU 使用由 PaddlePaddle 框架自动检测
        self.ocr = PaddleOCR(lang=lang)
        # 默认字幕区域：画面底部 70%-95%，左右各留 5%
        self.subtitle_area = subtitle_area or (0.70, 0.95, 0.05, 0.95)

    def _crop_subtitle_area(self, frame: np.ndarray) -> np.ndarray:
        """裁剪字幕区域"""
        h, w = frame.shape[:2]
        y_start = int(h * self.subtitle_area[0])
        y_end = int(h * self.subtitle_area[1])
        x_start = int(w * self.subtitle_area[2])
        x_end = int(w * self.subtitle_area[3])
        return frame[y_start:y_end, x_start:x_end]

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """预处理帧以提高 OCR 准确率"""
        # 转灰度
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # 增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 二值化（针对白色字幕）
        _, binary = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY)

        # 转回 BGR（PaddleOCR 需要）
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def _extract_text_from_frame(self, frame: np.ndarray) -> str:
        """从单帧提取文字"""
        # 裁剪字幕区域
        subtitle_region = self._crop_subtitle_area(frame)

        # 预处理
        processed = self._preprocess_frame(subtitle_region)

        # OCR 识别 (PaddleOCR 3.x 使用 predict)
        result = self.ocr.predict(processed)

        if not result or len(result) == 0:
            return ""

        # PaddleOCR 3.x 返回格式: [{'rec_texts': [...], 'rec_scores': [...]}]
        res = result[0]
        rec_texts = res.get('rec_texts', [])
        rec_scores = res.get('rec_scores', [])

        # 提取置信度高的文字
        texts = []
        for text, score in zip(rec_texts, rec_scores):
            if score > 0.7:  # 只保留置信度高的
                texts.append(text)

        return " ".join(texts)

    def _is_similar(self, text1: str, text2: str, threshold: float = OCR_SIMILARITY_THRESHOLD) -> bool:
        """判断两个文本是否相似"""
        if not text1 or not text2:
            return False
        ratio = SequenceMatcher(None, text1, text2).ratio()
        return ratio >= threshold

    def extract_subtitles(
        self,
        video_path: Path,
        frame_interval: float = OCR_FRAME_INTERVAL,
        progress_callback: Optional[callable] = None
    ) -> List[Dict]:
        """
        从视频中提取字幕

        Args:
            video_path: 视频文件路径
            frame_interval: 帧采样间隔（秒）
            progress_callback: 进度回调函数

        Returns:
            字幕列表，每项包含 start_time, end_time, text
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        # 计算需要处理的时间点
        sample_times = []
        t = 0.0
        while t < duration:
            sample_times.append(t)
            t += frame_interval

        total_samples = len(sample_times)
        print(f"📊 视频信息: {duration:.1f}秒, {fps:.1f}fps, 共{total_samples}帧待处理")

        subtitles = []
        current_text = ""
        current_start = 0.0

        for i, current_time in enumerate(sample_times):
            # 直接跳转到目标时间点（毫秒），避免逐帧解码
            cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
            ret, frame = cap.read()

            if not ret:
                continue

            text = self._extract_text_from_frame(frame)

            if progress_callback:
                progress_callback(i + 1, total_samples)

            # 文本变化检测
            if text and not self._is_similar(text, current_text):
                if current_text:
                    subtitles.append({
                        "start_time": current_start,
                        "end_time": current_time,
                        "text": current_text
                    })
                current_text = text
                current_start = current_time
            elif not text and current_text:
                subtitles.append({
                    "start_time": current_start,
                    "end_time": current_time,
                    "text": current_text
                })
                current_text = ""

        if current_text:
            subtitles.append({
                "start_time": current_start,
                "end_time": duration,
                "text": current_text
            })

        cap.release()

        print(f"\n✅ 识别完成，共处理 {len(sample_times)} 帧，提取 {len(subtitles)} 条字幕")

        return subtitles

    def subtitles_to_text(self, subtitles: List[Dict]) -> str:
        """将字幕列表合并为纯文本"""
        # 去重并合并
        unique_texts = []
        for sub in subtitles:
            text = sub["text"].strip()
            if text and (not unique_texts or not self._is_similar(text, unique_texts[-1])):
                unique_texts.append(text)

        return "\n".join(unique_texts)

    def subtitles_to_srt(self, subtitles: List[Dict]) -> str:
        """将字幕列表转换为 SRT 格式"""
        srt_lines = []

        for i, sub in enumerate(subtitles, 1):
            start = self._format_srt_time(sub["start_time"])
            end = self._format_srt_time(sub["end_time"])
            text = sub["text"]

            srt_lines.append(f"{i}")
            srt_lines.append(f"{start} --> {end}")
            srt_lines.append(text)
            srt_lines.append("")

        return "\n".join(srt_lines)

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """格式化为 SRT 时间格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


class DouyinVideoProcessor:
    """抖音长视频处理器"""

    def __init__(
        self,
        api_key: str = "",
        model: str = DEFAULT_MODEL,
        output_dir: Optional[str] = None,
        segment_duration: int = DEFAULT_SEGMENT_DURATION,
        use_ocr: bool = False,
        ocr_gpu: bool = False
    ):
        self.api_key = api_key
        self.model = model
        self.segment_duration = segment_duration
        self.use_ocr = use_ocr
        self.ocr_gpu = ocr_gpu

        # 设置输出目录
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = Path(tempfile.mkdtemp(prefix="douyin_"))

        # 设置阿里云百炼API密钥（ASR 模式需要）
        if api_key:
            dashscope.api_key = api_key

        # 初始化 OCR（延迟加载）
        self._ocr_engine = None

        print(f"📁 工作目录: {self.output_dir}")
        if use_ocr:
            print("🔤 提取模式: OCR 硬字幕识别")
        else:
            print("🎤 提取模式: ASR 语音识别")

    @property
    def ocr_engine(self) -> SubtitleOCR:
        """延迟加载 OCR 引擎"""
        if self._ocr_engine is None:
            self._ocr_engine = SubtitleOCR(use_gpu=self.ocr_gpu)
        return self._ocr_engine

    def parse_share_url(self, share_text: str) -> Dict:
        """从分享文本中提取无水印视频链接"""
        print("🔍 正在解析抖音分享链接...")

        # 提取分享链接
        urls = re.findall(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            share_text
        )
        if not urls:
            raise ValueError("未找到有效的分享链接")

        share_url = urls[0]
        share_response = requests.get(share_url, headers=HEADERS)
        video_id = share_response.url.split("?")[0].strip("/").split("/")[-1]
        share_url = f'https://www.iesdouyin.com/share/video/{video_id}'

        # 获取视频页面内容
        response = requests.get(share_url, headers=HEADERS)
        response.raise_for_status()

        pattern = re.compile(
            pattern=r"window\._ROUTER_DATA\s*=\s*(.*?)</script>",
            flags=re.DOTALL,
        )
        find_res = pattern.search(response.text)

        if not find_res or not find_res.group(1):
            raise ValueError("从HTML中解析视频信息失败")

        # 解析JSON数据
        json_data = json.loads(find_res.group(1).strip())
        VIDEO_ID_PAGE_KEY = "video_(id)/page"
        NOTE_ID_PAGE_KEY = "note_(id)/page"

        if VIDEO_ID_PAGE_KEY in json_data["loaderData"]:
            original_video_info = json_data["loaderData"][VIDEO_ID_PAGE_KEY]["videoInfoRes"]
        elif NOTE_ID_PAGE_KEY in json_data["loaderData"]:
            original_video_info = json_data["loaderData"][NOTE_ID_PAGE_KEY]["videoInfoRes"]
        else:
            raise Exception("无法从JSON中解析视频或图集信息")

        data = original_video_info["item_list"][0]

        # 获取视频信息
        video_url = data["video"]["play_addr"]["url_list"][0].replace("playwm", "play")
        desc = data.get("desc", "").strip() or f"douyin_{video_id}"

        # 替换文件名中的非法字符
        desc = re.sub(r'[\\/:*?"<>|]', '_', desc)

        result = {
            "url": video_url,
            "title": desc,
            "video_id": video_id
        }

        print(f"✅ 解析成功: {desc}")
        print(f"   视频ID: {video_id}")

        return result

    def download_video(self, video_info: Dict, filename: Optional[str] = None) -> Path:
        """下载视频到本地"""
        if filename:
            filepath = self.output_dir / filename
        else:
            filepath = self.output_dir / f"{video_info['video_id']}.mp4"

        print(f"⬇️  正在下载视频: {video_info['title']}")

        response = requests.get(video_info['url'], headers=HEADERS, stream=True)
        response.raise_for_status()

        # 获取文件大小
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = downloaded / total_size * 100
                        print(f"\r   下载进度: {progress:.1f}%", end="", flush=True)

        print(f"\n✅ 视频下载完成: {filepath}")
        return filepath

    def get_video_duration(self, video_path: Path) -> float:
        """获取视频时长（秒）"""
        cmd = [
            'ffprobe', '-v', 'error', '-show_entries',
            'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())

    def split_video(self, video_path: Path) -> List[Path]:
        """将长视频切分为多个片段"""
        duration = self.get_video_duration(video_path)
        print(f"📊 视频总时长: {duration:.1f}秒 ({duration/60:.1f}分钟)")

        # 如果视频时长小于切分阈值，直接返回原视频
        if duration <= self.segment_duration:
            print("   视频较短，无需切分")
            return [video_path]

        # 计算需要切分的片段数
        num_segments = int(duration / self.segment_duration) + 1
        print(f"✂️  正在将视频切分为 {num_segments} 个片段...")

        segments_dir = self.output_dir / "segments"
        segments_dir.mkdir(exist_ok=True)

        segment_paths = []

        for i in range(num_segments):
            start_time = i * self.segment_duration
            segment_filename = f"segment_{i:03d}.mp4"
            segment_path = segments_dir / segment_filename

            cmd = [
                'ffmpeg', '-y', '-i', str(video_path),
                '-ss', str(start_time),
                '-t', str(self.segment_duration),
                '-c', 'copy',  # 无损切分
                str(segment_path)
            ]

            subprocess.run(cmd, capture_output=True, check=True)

            # 验证片段是否有效
            if segment_path.exists() and segment_path.stat().st_size > 0:
                segment_paths.append(segment_path)
                print(f"   片段 {i+1}/{num_segments}: {segment_filename}")

        print(f"✅ 切分完成，共 {len(segment_paths)} 个片段")
        return segment_paths

    def extract_audio(self, video_path: Path) -> Path:
        """从视频文件中提取音频"""
        audio_path = video_path.with_suffix('.mp3')

        cmd = [
            'ffmpeg', '-y', '-i', str(video_path),
            '-vn',  # 不要视频
            '-acodec', 'libmp3lame',
            '-q:a', '0',  # 最高质量
            str(audio_path)
        ]

        subprocess.run(cmd, capture_output=True, check=True)
        return audio_path

    def upload_file_for_transcription(self, file_path: Path) -> str:
        """上传文件获取可访问的URL（使用阿里云OSS临时链接）"""
        # 阿里云百炼支持直接上传本地文件
        # 这里使用 file:// 协议或直接传文件路径
        return f"file://{file_path.absolute()}"

    def transcribe_audio(self, audio_path: Path, segment_index: int = 0) -> Dict:
        """转录单个音频文件"""
        print(f"🎤 正在识别片段 {segment_index + 1}: {audio_path.name}")

        try:
            # 对于本地文件，需要先读取并上传
            # 阿里云百炼支持直接传本地文件路径
            task_response = dashscope.audio.asr.Transcription.async_call(
                model=self.model,
                file_urls=[str(audio_path.absolute())],
                language_hints=['zh', 'en']
            )

            # 等待转录完成
            transcription_response = dashscope.audio.asr.Transcription.wait(
                task=task_response.output.task_id
            )

            if transcription_response.status_code == HTTPStatus.OK:
                results = transcription_response.output.get('results', [])

                for transcription in results:
                    if 'transcription_url' in transcription:
                        url = transcription['transcription_url']
                        result = json.loads(request.urlopen(url).read().decode('utf8'))

                        if 'transcripts' in result and len(result['transcripts']) > 0:
                            text = result['transcripts'][0]['text']
                            return {
                                "segment_index": segment_index,
                                "status": "success",
                                "text": text,
                                "file": str(audio_path)
                            }

                return {
                    "segment_index": segment_index,
                    "status": "success",
                    "text": "",
                    "file": str(audio_path)
                }
            else:
                return {
                    "segment_index": segment_index,
                    "status": "error",
                    "text": "",
                    "error": transcription_response.output.get('message', '未知错误'),
                    "file": str(audio_path)
                }

        except Exception as e:
            return {
                "segment_index": segment_index,
                "status": "error",
                "text": "",
                "error": str(e),
                "file": str(audio_path)
            }

    def transcribe_video_url(self, video_url: str) -> str:
        """直接通过视频URL进行转录（适用于短视频）"""
        print("🎤 正在进行语音识别...")

        try:
            task_response = dashscope.audio.asr.Transcription.async_call(
                model=self.model,
                file_urls=[video_url],
                language_hints=['zh', 'en']
            )

            transcription_response = dashscope.audio.asr.Transcription.wait(
                task=task_response.output.task_id
            )

            if transcription_response.status_code == HTTPStatus.OK:
                for transcription in transcription_response.output['results']:
                    url = transcription['transcription_url']
                    result = json.loads(request.urlopen(url).read().decode('utf8'))

                    if 'transcripts' in result and len(result['transcripts']) > 0:
                        return result['transcripts'][0]['text']

                return "未识别到文本内容"
            else:
                raise Exception(f"转录失败: {transcription_response.output.message}")

        except Exception as e:
            raise Exception(f"语音识别失败: {str(e)}")

    def split_video_for_ocr(self, video_path: Path) -> List[Tuple[Path, float]]:
        """将视频切分为多个片段用于 OCR 并行处理

        Returns:
            [(segment_path, time_offset), ...]
        """
        duration = self.get_video_duration(video_path)
        print(f"📊 视频总时长: {duration:.1f}秒 ({duration/60:.1f}分钟)")

        if duration <= OCR_SEGMENT_DURATION:
            print("   视频较短，无需切分")
            return [(video_path, 0.0)]

        num_segments = int(duration / OCR_SEGMENT_DURATION) + 1
        print(f"✂️  正在将视频切分为 {num_segments} 个片段（每段 {OCR_SEGMENT_DURATION} 秒）...")

        segments_dir = self.output_dir / "ocr_segments"
        segments_dir.mkdir(exist_ok=True)

        segment_info = []
        for i in range(num_segments):
            start_time = i * OCR_SEGMENT_DURATION
            segment_path = segments_dir / f"segment_{i:03d}.mp4"

            cmd = [
                'ffmpeg', '-y', '-i', str(video_path),
                '-ss', str(start_time),
                '-t', str(OCR_SEGMENT_DURATION),
                '-c', 'copy',
                str(segment_path)
            ]
            subprocess.run(cmd, capture_output=True, check=True)

            if segment_path.exists() and segment_path.stat().st_size > 0:
                segment_info.append((segment_path, start_time))
                print(f"   片段 {i+1}/{num_segments}: {segment_path.name}")

        print(f"✅ 切分完成，共 {len(segment_info)} 个片段")
        return segment_info

    def process_video_ocr(self, video_path: Path) -> Tuple[str, List[Dict]]:
        """使用 OCR 提取视频硬字幕（支持并行处理）"""
        print("\n🔤 正在使用 OCR 提取硬字幕...")

        # 1. 切分视频
        segments = self.split_video_for_ocr(video_path)

        # 2. 准备并行任务参数
        tasks = [
            (str(seg_path), idx, time_offset, 'ch', self.ocr_engine.subtitle_area if self._ocr_engine else None)
            for idx, (seg_path, time_offset) in enumerate(segments)
        ]

        # 3. 并行处理
        num_workers = min(OCR_PARALLEL_WORKERS, len(segments))
        print(f"\n🚀 使用 {num_workers} 进程并行处理...")

        all_subtitles = []
        if len(segments) == 1:
            # 单片段直接处理
            _, subs = _process_segment_ocr(tasks[0])
            all_subtitles = subs
        else:
            # 多片段并行处理
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                results = list(executor.map(_process_segment_ocr, tasks))

            # 按片段顺序合并
            results.sort(key=lambda x: x[0])
            for _, subs in results:
                all_subtitles.extend(subs)

        print(f"\n✅ 识别完成，共提取 {len(all_subtitles)} 条字幕")

        # 4. 转换为纯文本
        full_text = self.ocr_engine.subtitles_to_text(all_subtitles)

        # 5. 保存 SRT 文件
        srt_content = self.ocr_engine.subtitles_to_srt(all_subtitles)
        srt_file = self.output_dir / f"{video_path.stem}.srt"
        with open(srt_file, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        print(f"📄 SRT 字幕已保存: {srt_file}")

        # 清理临时片段
        segments_dir = self.output_dir / "ocr_segments"
        if segments_dir.exists():
            shutil.rmtree(segments_dir)
            print("🧹 已清理临时片段文件")

        # 构建结果
        results = [{
            "segment_index": 0,
            "status": "success",
            "text": full_text,
            "subtitles": all_subtitles,
            "file": str(video_path)
        }]

        return full_text, results

    def process_long_video(
        self,
        video_path: Path,
        parallel: bool = True
    ) -> Tuple[str, List[Dict]]:
        """处理长视频：切分、提取音频/字幕、识别、合并"""

        # OCR 模式：直接处理整个视频
        if self.use_ocr:
            return self.process_video_ocr(video_path)

        # ASR 模式：切分、提取音频、语音识别
        # 1. 切分视频
        segments = self.split_video(video_path)

        # 2. 提取每个片段的音频
        print("\n🔊 正在提取音频...")
        audio_files = []
        for i, segment in enumerate(segments):
            audio_path = self.extract_audio(segment)
            audio_files.append((i, audio_path))
            print(f"   音频 {i+1}/{len(segments)}: {audio_path.name}")

        # 3. 并行或串行转录
        print("\n🎤 正在进行语音识别...")
        results = []

        if parallel and len(audio_files) > 1:
            with ThreadPoolExecutor(max_workers=MAX_PARALLEL_TRANSCRIPTIONS) as executor:
                futures = {
                    executor.submit(self.transcribe_audio, audio_path, idx): idx
                    for idx, audio_path in audio_files
                }

                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    status = "✅" if result["status"] == "success" else "❌"
                    print(f"   {status} 片段 {result['segment_index'] + 1} 识别完成")
        else:
            for idx, audio_path in audio_files:
                result = self.transcribe_audio(audio_path, idx)
                results.append(result)
                status = "✅" if result["status"] == "success" else "❌"
                print(f"   {status} 片段 {idx + 1} 识别完成")

        # 4. 按顺序合并结果
        results.sort(key=lambda x: x["segment_index"])

        full_text_parts = []
        for result in results:
            if result["status"] == "success" and result["text"]:
                full_text_parts.append(result["text"])

        full_text = "\n\n".join(full_text_parts)

        return full_text, results

    def process(
        self,
        share_link: str,
        force_local: bool = False,
        keep_files: bool = False
    ) -> Dict:
        """完整处理流程"""

        # 1. 解析分享链接
        video_info = self.parse_share_url(share_link)

        # 2. OCR 模式必须本地处理
        if self.use_ocr:
            force_local = True

        # 3. 判断是否需要本地处理
        # 先尝试直接URL识别（适用于短视频，仅ASR模式）
        if not force_local and not self.use_ocr:
            try:
                print("\n🚀 尝试直接在线识别...")
                text = self.transcribe_video_url(video_info['url'])

                return {
                    "status": "success",
                    "video_id": video_info["video_id"],
                    "title": video_info["title"],
                    "text": text,
                    "method": "online_asr",
                    "segments": []
                }
            except Exception as e:
                print(f"⚠️  在线识别失败: {e}")
                print("   切换到本地处理模式...")

        # 4. 本地处理流程
        method = "local_ocr" if self.use_ocr else "local_asr"
        print(f"\n📥 启动本地处理模式 ({method})...")

        # 下载视频
        video_path = self.download_video(video_info)

        # 处理长视频
        full_text, segment_results = self.process_long_video(video_path)

        # 保存结果
        result = {
            "status": "success",
            "video_id": video_info["video_id"],
            "title": video_info["title"],
            "text": full_text,
            "method": "local",
            "segments": segment_results,
            "output_dir": str(self.output_dir)
        }

        # 保存到文件
        result_file = self.output_dir / f"{video_info['video_id']}_result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        text_file = self.output_dir / f"{video_info['video_id']}_transcript.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(f"标题: {video_info['title']}\n")
            f.write(f"视频ID: {video_info['video_id']}\n")
            f.write(f"{'='*50}\n\n")
            f.write(full_text)

        print(f"\n📝 转录结果已保存: {text_file}")

        # 清理临时文件
        if not keep_files:
            segments_dir = self.output_dir / "segments"
            if segments_dir.exists():
                shutil.rmtree(segments_dir)
                print("🧹 已清理临时片段文件")

        return result


def process_local_video(
    video_path: str,
    api_key: str = "",
    model: str = DEFAULT_MODEL,
    output_dir: Optional[str] = None,
    segment_duration: int = DEFAULT_SEGMENT_DURATION,
    use_ocr: bool = False,
    ocr_gpu: bool = False
) -> Dict:
    """处理本地视频文件"""

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    processor = DouyinVideoProcessor(
        api_key=api_key,
        model=model,
        output_dir=output_dir or str(video_path.parent),
        segment_duration=segment_duration,
        use_ocr=use_ocr,
        ocr_gpu=ocr_gpu
    )

    print(f"📹 处理本地视频: {video_path}")

    full_text, segment_results = processor.process_long_video(video_path)

    result = {
        "status": "success",
        "source": str(video_path),
        "text": full_text,
        "segments": segment_results,
        "output_dir": str(processor.output_dir)
    }

    # 保存结果
    result_file = processor.output_dir / f"{video_path.stem}_result.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    text_file = processor.output_dir / f"{video_path.stem}_transcript.txt"
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(f"源文件: {video_path}\n")
        f.write(f"{'='*50}\n\n")
        f.write(full_text)

    print(f"\n📝 转录结果已保存: {text_file}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="抖音长视频处理工具 - 支持 ASR 语音识别和 OCR 硬字幕提取",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # ASR 模式（默认）- 语音识别
  python douyin_video_processor.py "抖音分享链接"

  # OCR 模式 - 提取视频硬字幕（更准确，需要视频有字幕）
  python douyin_video_processor.py "抖音分享链接" --ocr

  # OCR 模式 + GPU 加速
  python douyin_video_processor.py "抖音分享链接" --ocr --ocr-gpu

  # 强制使用本地模式处理（ASR模式）
  python douyin_video_processor.py "抖音分享链接" --force-local

  # 处理本地视频文件
  python douyin_video_processor.py --local /path/to/video.mp4
  python douyin_video_processor.py --local /path/to/video.mp4 --ocr

  # 指定输出目录
  python douyin_video_processor.py "抖音分享链接" -o ./output

  # 自定义切分时长（秒，仅ASR模式）
  python douyin_video_processor.py "抖音分享链接" --segment-duration 180

环境变量:
  DASHSCOPE_API_KEY: 阿里云百炼API密钥（ASR模式需要，OCR模式不需要）

模式对比:
  ASR 模式: 语音转文字，适用于所有视频，需要 API 密钥
  OCR 模式: 硬字幕识别，更准确但需要视频有字幕，无需 API 密钥
        """
    )

    parser.add_argument(
        "input",
        nargs="?",
        help="抖音分享链接或本地视频路径"
    )

    parser.add_argument(
        "--local", "-l",
        action="store_true",
        help="处理本地视频文件"
    )

    parser.add_argument(
        "--ocr",
        action="store_true",
        help="使用 OCR 模式提取硬字幕（需要视频有字幕，无需API密钥）"
    )

    parser.add_argument(
        "--ocr-gpu",
        action="store_true",
        help="OCR 模式使用 GPU 加速（需要安装 paddlepaddle-gpu）"
    )

    parser.add_argument(
        "--force-local", "-f",
        action="store_true",
        help="强制使用本地下载模式（跳过在线识别，仅ASR模式）"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出目录（默认使用临时目录）"
    )

    parser.add_argument(
        "--segment-duration", "-s",
        type=int,
        default=DEFAULT_SEGMENT_DURATION,
        help=f"视频切分时长（秒），默认 {DEFAULT_SEGMENT_DURATION}，仅ASR模式"
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"语音识别模型，默认 {DEFAULT_MODEL}，仅ASR模式"
    )

    parser.add_argument(
        "--keep-files", "-k",
        action="store_true",
        help="保留中间文件（切分的片段和音频）"
    )

    parser.add_argument(
        "--info-only",
        action="store_true",
        help="仅获取视频信息，不进行下载和识别"
    )

    args = parser.parse_args()

    # 检查API密钥（仅ASR模式需要）
    api_key = os.getenv('DASHSCOPE_API_KEY', '')
    if not args.ocr and not api_key and not args.info_only:
        print("❌ 错误: ASR 模式需要设置环境变量 DASHSCOPE_API_KEY")
        print("   请设置阿里云百炼API密钥:")
        print("   export DASHSCOPE_API_KEY='your-api-key'")
        print("")
        print("   或者使用 OCR 模式（无需API密钥）:")
        print("   python douyin_video_processor.py \"链接\" --ocr")
        sys.exit(1)

    if not args.input:
        parser.print_help()
        sys.exit(1)

    try:
        if args.local:
            # 处理本地视频
            result = process_local_video(
                video_path=args.input,
                api_key=api_key,
                model=args.model,
                output_dir=args.output,
                segment_duration=args.segment_duration,
                use_ocr=args.ocr,
                ocr_gpu=args.ocr_gpu
            )
        else:
            processor = DouyinVideoProcessor(
                api_key=api_key,
                model=args.model,
                output_dir=args.output,
                segment_duration=args.segment_duration,
                use_ocr=args.ocr,
                ocr_gpu=args.ocr_gpu
            )

            if args.info_only:
                # 仅获取视频信息
                video_info = processor.parse_share_url(args.input)
                print("\n📋 视频信息:")
                print(json.dumps(video_info, ensure_ascii=False, indent=2))
            else:
                # 完整处理流程
                result = processor.process(
                    share_link=args.input,
                    force_local=args.force_local or args.ocr,
                    keep_files=args.keep_files
                )

                print("\n" + "="*60)
                print("📜 识别结果:")
                print("="*60)
                print(result["text"])
                print("="*60)

        print("\n✅ 处理完成!")

    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
