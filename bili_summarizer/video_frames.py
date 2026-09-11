#!/usr/bin/env python3
"""视频截帧与网格图拼接（参考 BiliNote video_reader.py）

功能:
  - extract_frames:     按固定间隔截取视频帧
  - build_grid_image:   将多帧拼接为网格图（3x3 等）
  - capture_screenshot: 在指定时间点截取单帧（关键截图模式）
  - image_to_base64:    图片转 base64（用于视觉模型 API 传输）

用法:
  python video_frames.py extract video.mp4 --interval 10 --out frames/
  python video_frames.py grid video.mp4 --interval 10 --cols 3 --rows 3
  python video_frames.py shot video.mp4 --ts 120.5 --out shot.jpg
"""
import argparse
import base64
import os
import subprocess

from PIL import Image


# ========== FFmpeg 封装 ==========

def get_video_duration(video_path: str) -> float:
    """获取视频时长（秒），使用 ffprobe"""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        import json
        data = json.loads(result.stdout)
        return float(data.get("format", {}).get("duration", 0))
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, ValueError):
        # 兜底: 尝试 imageio-ffmpeg 提供的 ffprobe（处理未加入 PATH 的情况）
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            probe = os.path.join(os.path.dirname(exe), "ffprobe")
            if os.path.exists(probe):
                result = subprocess.run(
                    [probe, "-v", "quiet", "-print_format", "json",
                     "-show_format", video_path],
                    capture_output=True, text=True, check=True,
                )
                import json as _json
                data = _json.loads(result.stdout)
                return float(data.get("format", {}).get("duration", 0))
        except Exception:
            pass
        return 0.0


def _ffmpeg(args: list[str], timeout: int = 120) -> None:
    """执行 ffmpeg，自动处理未加入 PATH 的情况"""
    try:
        subprocess.run(["ffmpeg"] + args, check=True, capture_output=True,
                       timeout=timeout)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # 兜底: imageio-ffmpeg 提供的 ffmpeg
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            subprocess.run([exe] + args, check=True, capture_output=True,
                           timeout=timeout)
        except Exception as e:
            raise RuntimeError(f"ffmpeg 执行失败: {e}") from e


# ========== 核心功能 ==========

def extract_frames(video_path: str, interval: int = 10,
                   width: int = 640, height: int = 360,
                   output_dir: str = "frames") -> list[str]:
    """按固定间隔截取视频帧，返回图片路径列表

    参考 BiliNote: timestamps = range(0, int(duration), frame_interval)
    """
    os.makedirs(output_dir, exist_ok=True)
    duration = get_video_duration(video_path)
    if duration <= 0:
        raise RuntimeError(f"无法获取视频时长: {video_path}")

    frames = []
    timestamps = range(0, int(duration), interval)
    for i, ts in enumerate(timestamps):
        output = os.path.join(output_dir, f"frame_{i:03d}.jpg")
        _ffmpeg([
            "-ss", str(ts), "-i", video_path,
            "-frames:v", "1", "-vf", f"scale={width}:{height}",
            "-q:v", "2", output,
        ])
        if os.path.exists(output):
            frames.append(output)
    return frames


def build_grid_image(frames: list[str], cols: int = 3, rows: int = 3,
                     cell_w: int = 640, cell_h: int = 360,
                     output: str = "grid_visual.jpg") -> str:
    """将多帧拼接为网格图（参考 BiliNote video_reader.py:73-96）

    不足 cols*rows 的组会被丢弃，保证每张网格图完整。
    """
    group_size = cols * rows
    if not frames:
        raise ValueError("没有可用的帧")

    # 取前 group_size 张（或少于 group_size 但至少有 1 张）
    images = frames[:group_size]

    grid = Image.new("RGB", (cell_w * cols, cell_h * rows))
    for i, frame_path in enumerate(images):
        img = Image.open(frame_path).convert("RGB")
        img = img.resize((cell_w, cell_h), Image.LANCZOS)
        x = (i % cols) * cell_w
        y = (i // cols) * cell_h
        grid.paste(img, (x, y))

    grid.save(output, quality=85)
    return output


def capture_screenshot(video_path: str, timestamp: float,
                       output_path: str, width: int = 1280) -> str:
    """在指定时间点截取单帧（关键截图模式）— 本地文件"""
    _ffmpeg([
        "-ss", str(timestamp), "-i", video_path,
        "-frames:v", "1", "-vf", f"scale={width}:-1",
        "-q:v", "2", output_path,
    ])
    return output_path


def _build_ffmpeg_headers(headers: dict | None = None) -> str:
    """构建 ffmpeg -headers 参数（B站 CDN 需要 Referer）"""
    base = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.bilibili.com",
    }
    if headers:
        base.update(headers)
    return "".join(f"{k}: {v}\r\n" for k, v in base.items())


def capture_screenshot_from_stream(
    stream_url: str, timestamp: float, output_path: str,
    width: int = 1280, headers: dict | None = None, timeout: int = 30,
) -> str:
    """流式截帧：直接从视频流 URL 截取指定时间点的帧，无需下载完整视频

    利用 B站 CDN 支持的 HTTP Range Request，ffmpeg -ss 会 seek 到目标时间点
    附近的关键帧，只下载该位置的数据，大幅减少网络传输量。

    Args:
        stream_url: 视频流地址（DASH video baseUrl）
        timestamp: 截图时间点（秒）
        output_path: 输出图片路径
        width: 输出图片宽度（高度自动等比缩放）
        headers: 额外 HTTP 头（如 Cookie）
        timeout: 单帧截取超时（秒）
    """
    hdr = _build_ffmpeg_headers(headers)
    _ffmpeg([
        "-headers", hdr,
        "-ss", str(timestamp),
        "-i", stream_url,
        "-frames:v", "1",
        "-vf", f"scale={width}:-1",
        "-q:v", "2",
        output_path,
    ], timeout=timeout)
    return output_path


def extract_frames_from_stream(
    stream_url: str, duration: float, interval: int = 10,
    width: int = 640, height: int = 360,
    output_dir: str = "frames",
    headers: dict | None = None, timeout_per_frame: int = 30,
) -> list[str]:
    """流式截帧：按固定间隔从视频流 URL 截取多帧，无需下载完整视频

    Args:
        stream_url: 视频流地址
        duration: 视频时长（秒），从 get_video_info 获取
        interval: 截帧间隔（秒）
        width/height: 输出帧尺寸
        output_dir: 输出目录
        headers: 额外 HTTP 头
        timeout_per_frame: 每帧截取超时（秒）
    Returns:
        生成的图片路径列表
    """
    os.makedirs(output_dir, exist_ok=True)
    if duration <= 0:
        raise RuntimeError(f"无效的视频时长: {duration}")

    hdr = _build_ffmpeg_headers(headers)
    frames = []
    timestamps = range(0, int(duration), interval)
    for i, ts in enumerate(timestamps):
        output = os.path.join(output_dir, f"frame_{i:03d}.jpg")
        try:
            _ffmpeg([
                "-headers", hdr,
                "-ss", str(ts),
                "-i", stream_url,
                "-frames:v", "1",
                "-vf", f"scale={width}:{height}",
                "-q:v", "2",
                output,
            ], timeout=timeout_per_frame)
            if os.path.exists(output):
                frames.append(output)
        except RuntimeError:
            # 单帧失败不中断，继续下一帧
            pass
    return frames


def image_to_base64(image_path: str) -> str:
    """图片转 base64（用于 API 传输）"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ========== CLI ==========

def main():
    parser = argparse.ArgumentParser(description="视频截帧与网格图拼接")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # extract 子命令
    p_extract = sub.add_parser("extract", help="按间隔截取帧")
    p_extract.add_argument("video_path")
    p_extract.add_argument("--interval", type=int, default=10)
    p_extract.add_argument("--width", type=int, default=640)
    p_extract.add_argument("--height", type=int, default=360)
    p_extract.add_argument("--out", default="frames")

    # grid 子命令
    p_grid = sub.add_parser("grid", help="生成网格图")
    p_grid.add_argument("video_path")
    p_grid.add_argument("--interval", type=int, default=10)
    p_grid.add_argument("--cols", type=int, default=3)
    p_grid.add_argument("--rows", type=int, default=3)
    p_grid.add_argument("--out", default="grid_visual.jpg")

    # shot 子命令
    p_shot = sub.add_parser("shot", help="指定时间点截图")
    p_shot.add_argument("video_path")
    p_shot.add_argument("--ts", type=float, required=True)
    p_shot.add_argument("--out", default="screenshot.jpg")

    args = parser.parse_args()

    if args.cmd == "extract":
        frames = extract_frames(args.video_path, args.interval,
                                args.width, args.height, args.out)
        print(f"提取 {len(frames)} 帧到 {args.out}/")
    elif args.cmd == "grid":
        frames = extract_frames(args.video_path, args.interval,
                                out_dir="frames_tmp")
        grid_path = build_grid_image(frames, args.cols, args.rows,
                                     output=args.out)
        print(f"网格图已生成: {grid_path}")
    elif args.cmd == "shot":
        path = capture_screenshot(args.video_path, args.ts, args.out)
        print(f"截图已生成: {path}")


if __name__ == "__main__":
    main()
