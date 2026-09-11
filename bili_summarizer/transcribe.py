#!/usr/bin/env python3
"""Whisper 语音转写 - 支持断点续传 + GPU 自动加速 + 自动降级

设备策略 (--device 默认 auto):
  - 检测到 CUDA GPU -> device=cuda, compute_type=float16 (快 ~10-30x)
  - 无 GPU           -> device=cpu,  compute_type=int8
  - 若 GPU 推理在运行时失败 (如缺 cublas/cudnn dll)，自动降级到 cpu 重跑，不中断任务

断点续传:
  - 每转写完一个 segment，立即写入 {output}.cache 文件
  - 转写完成后合并为最终 .txt 文件，删除 .cache
  - 如果中途中断 (OOM/超时)，.cache 保留已转写的段落
  - 下次运行时检测 .cache，跳过已转写的部分，从断点继续

用法:
  python transcribe.py audio.mp3 --model base --language zh
  python transcribe.py audio.mp3 --model small --device cuda --vad
  python transcribe.py audio.mp3 --beam 5        # 求准
  python transcribe.py audio.mp3 --resume        # 从上次中断处继续
"""
import argparse
import json
import os
import sys

# HuggingFace 国内镜像 + 禁用 Xet 协议（国内网络优化）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from faster_whisper import WhisperModel


def load_cache(cache_path: str) -> list[dict]:
    """加载已转写的段落缓存"""
    if not os.path.exists(cache_path):
        return []
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def save_cache(cache_path: str, segments: list[dict]):
    """保存已转写的段落到缓存文件"""
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)


def _cuda_available() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False


def pick_device(requested: str | None) -> tuple[str, str]:
    """自动选择 device 和 compute_type。返回 (device, compute_type)"""
    if requested == "cpu":
        return "cpu", "int8"
    if requested == "cuda":
        if _cuda_available():
            return "cuda", "float16"
        print("[device] 请求 cuda 但不可用，回退 CPU", file=sys.stderr)
        return "cpu", "int8"
    # 自动
    if _cuda_available():
        print("[device] 检测到 CUDA GPU，使用 cuda/float16", file=sys.stderr)
        return "cuda", "float16"
    print("[device] 未检测到 GPU，使用 cpu/int8", file=sys.stderr)
    return "cpu", "int8"


def _is_cuda_init_failure(exc: Exception) -> bool:
    """判断异常是否为 CUDA 运行时库缺失（cublas/cudnn dll 找不到）"""
    msg = str(exc)
    low = msg.lower()
    keys = ("cublas", "cudnn", "not found or cannot be loaded",
            "cublas64", "libcublas", "libcudnn", "dll")
    return any(k in low or k in msg for k in keys)


def _ensure_cuda_dll_path() -> None:
    """把 pip 安装的 NVIDIA CUDA 运行库 DLL 目录自动补进 PATH。

    修复 ctranslate2 报 "Library cublas64_12.dll is not found"：缺少 CUDA 12 运行库时，
    从 site-packages/nvidia/<pkg>/bin 自动发现（nvidia-cublas-cu12 / nvidia-cudnn-cu12），
    免去手工配置 PATH。未安装这些包则静默跳过（交由 CPU(int8) 兜底）。
    """
    import glob as _glob
    dll_dirs = []
    for base in sys.path:
        if not base:
            continue
        for d in _glob.glob(os.path.join(base, "nvidia", "*", "bin")):
            if _glob.glob(os.path.join(d, "cublas64_*.dll")) or \
               _glob.glob(os.path.join(d, "cudnn64_*.dll")):
                dll_dirs.append(d)
    if dll_dirs:
        cur = os.environ.get("PATH", "")
        os.environ["PATH"] = os.pathsep.join(dll_dirs + ([cur] if cur else []))
        print(f"[device] 已注入 CUDA DLL 目录: {dll_dirs}", file=sys.stderr)


def transcribe_once(audio_path: str, model_name: str, device: str,
                    compute_type: str, language: str, beam: int, vad: bool) -> list[dict]:
    """加载模型并转写，返回 segment dict 列表（实现在一个函数内，便于降级重跑）。"""
    _ensure_cuda_dll_path()  # GPU 走 cu12 前确保 cublas/cudnn DLL 可发现
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        audio_path, language=language, beam_size=beam, vad_filter=vad)
    # 惰性迭代：预先生成全部段落才能捕捉运行时 CUDA 错误
    out = []
    for seg in segments:
        out.append({"start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": seg.text.strip()})
    return out


def main():
    parser = argparse.ArgumentParser(description="Whisper 语音转写（GPU 自动加速 + 支持断点续传）")
    parser.add_argument("audio_path", help="音频文件路径")
    parser.add_argument("--model", default="small", help="模型大小 (base/small/medium/large-v3)")
    parser.add_argument("--language", default="zh", help="语言代码 (zh/en)")
    parser.add_argument("--device", default=None,
                        help="推理设备: auto(默认)/cuda/cpu")
    parser.add_argument("--beam", type=int, default=1,
                        help="beam_size: 1=快(贪心), 5=准(慢)。默认 1 追求速度")
    parser.add_argument("--vad", action="store_true",
                        help="开启 VAD 静音过滤（跳过大段沉默，通常更省时间）")
    parser.add_argument("--output", default=None, help="输出文件路径（默认同名.txt）")
    parser.add_argument("--resume", action="store_true",
                        help="从上次中断处继续（自动检测.cache文件）")

    args = parser.parse_args()

    # 确定输出路径
    if args.output is None:
        output_path = f"{args.audio_path.rsplit('.', 1)[0]}.txt"
    else:
        output_path = args.output

    cache_path = f"{output_path}.cache"

    # 加载缓存（断点续传）
    cached_segments = []
    if args.resume and os.path.exists(cache_path):
        cached_segments = load_cache(cache_path)
        if cached_segments:
            last_end = cached_segments[-1]["end"]
            print(f"[Resume] 检测到缓存: {len(cached_segments)} 段, 已转写到 {last_end:.1f}s")

    device, compute_type = pick_device(args.device)

    # 转写（GPU 失败自动降级 CPU）
    new_segments = []
    try:
        print(f"[device] model='{args.model}' device={device} compute_type={compute_type}")
        print(f"Transcribing '{args.audio_path}' (beam={args.beam}, vad={args.vad})...")
        new_segments = transcribe_once(args.audio_path, args.model, device,
                                       compute_type, args.language, args.beam, args.vad)
        print(f"Detected segments: {len(new_segments)}")
    except Exception as e:
        if device == "cuda" and _is_cuda_init_failure(e):
            print(f"[device] GPU 推理失败: {e}", file=sys.stderr)
            print("[device] 自动降级到 CPU(int8) 重跑...", file=sys.stderr)
            device, compute_type = "cpu", "int8"
            new_segments = transcribe_once(args.audio_path, args.model, device,
                                           compute_type, args.language, args.beam, args.vad)
        else:
            raise

    # 合并缓存 + 新段落
    all_segments = list(cached_segments)
    for seg in new_segments:
        if cached_segments and seg["start"] < cached_segments[-1]["end"]:
            continue
        all_segments.append(seg)

    # 存缓存 + 写最终文件（保留原有断点续传语义）
    save_cache(cache_path, all_segments)
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in all_segments:
            f.write(seg["text"] + "\n")
    if os.path.exists(cache_path):
        os.remove(cache_path)

    print(f"\nTranscription complete. Saved to '{output_path}'")
    print(f"Segments: {len(all_segments)} (device={device})")


if __name__ == "__main__":
    main()
