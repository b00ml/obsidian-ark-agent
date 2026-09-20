#!/usr/bin/env python3
"""视觉模型 API 调用（兼容 OpenAI/DeepSeek/Qwen 接口）

功能:
  - analyze_with_visual:       字幕文本 + 网格图 → 视觉模型 → 增强笔记
  - generate_screenshot_markers: LLM 分析字幕 → 决定截图时间点
  - should_use_visual:         智能调度，根据内容特征判断是否需要视觉分析
  - load_visual_config:        加载 config/visual_models.json 配置

参考设计: DESIGN-MULTIMODAL.md (Phase 2)
"""
import argparse
import json
import os
import re
import sys

import requests

from prompt_loader import load_prompt, render
from provider_client import ProviderError, TextModelClient

# 配置文件默认路径（相对本文件）
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config", "visual_models.json"
)


def load_visual_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """加载视觉模型配置（JSON）"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"视觉模型配置文件不存在: {config_path}\n"
            "请复制 visual_models.json.example 为 visual_models.json 并填入 API Key"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _chat_completion(config: dict, messages: list, max_tokens: int,
                     timeout: int, image_b64: str | None = None,
                     client: TextModelClient | None = None) -> dict:
    """统一的 chat/completions 调用（支持可选的图片输入）

    config: {"api_base", "api_key", "model"}
    返回: {"content", "model", "usage"}
    """
    model_client = client or TextModelClient(
        api_base=config["api_base"], api_key=config.get("api_key", ""),
        model=config["model"], timeout=timeout, transport=requests.post,
    )
    try:
        return model_client.chat(messages[0]["content"], max_tokens=max_tokens,
                                 image_b64=image_b64,
                                 detail=config.get("detail", "high")).to_dict()
    except ProviderError as exc:
        raise RuntimeError(str(exc)) from exc


def analyze_with_visual(transcript: str, grid_image_b64: str,
                        title: str, config: dict) -> dict:
    """将字幕文本 + 网格图一起传给视觉模型，生成增强笔记

    config: {"api_base", "api_key", "model", ...}
    返回: {"content", "model", "usage"}
    """
    prompt = render(
        load_prompt("visual-analyze-user"),
        {"title": title, "transcript": transcript[:4000]},
    )

    return _chat_completion(config, [{"role": "user", "content": prompt}],
                            max_tokens=2000, timeout=120,
                            image_b64=grid_image_b64)


def analyze_grid_card(title: str, grid_image_b64: str, config: dict) -> dict:
    """逐张网格图 → 只产"画面事实短卡"(screen-only)，禁产整篇（OPT-092 责任分层）。

    修复：旧实现把整段字幕喂进每张图并要求产一篇完整笔记 → N 张图就 N 份整篇反复堆叠
    （实测 10 张图 = 71KB/1676 行）。现改为每张仅抽取画面客观事实，字幕与整篇放到聚合层。
    """
    prompt = render(load_prompt("visual-grid-card-user"), {"title": title})
    return _chat_completion(config, [{"role": "user", "content": prompt}],
                            max_tokens=600, timeout=120,
                            image_b64=grid_image_b64)


def synthesize_visual_note(title: str, cards: dict[int, str], transcript: str,
                           config: dict, max_cards: int = 60) -> dict:
    """聚合全部画面短卡 + 字幕 → 只产【一遍】结构化画面笔记（OPT-092/093）。

    - 内建去重与"帧间不一致需人工核对"标注（一致性，不擅自 pick）。
    - 角色分流：只保留画面相关，字幕听力内容归独立《总结》。
    cards: {帧序号: 短卡文本}
    """
    lines = []
    for idx in sorted(cards)[:max_cards]:
        lines.append(f"### 网格图 {idx}\n{cards[idx]}")
    cardtext = "\n\n".join(lines)[:12000]
    prompt = render(
        load_prompt("visual-synth-user"),
        {"title": title, "cards": cardtext, "transcript": transcript[:20000]},
    )
    return _chat_completion(config, [{"role": "user", "content": prompt}],
                            max_tokens=3000, timeout=180)


def summarize_transcript(title: str, transcript: str, config: dict) -> dict:
    """字幕 → 独立《总结》笔记（听力内容归此，与画面笔记分工，OPT-094 三件套）。"""
    prompt = render(
        load_prompt("bili-summary-user"),
        {"title": title, "transcript": transcript[:20000]},
    )
    return _chat_completion(config, [{"role": "user", "content": prompt}],
                            max_tokens=2500, timeout=120)


# 逐字稿高频专名/噪声清洗（OPT-096）。仅收编高置信度、低误伤的正则替换：
# 例：ASR 把"千问 3B/8B"听成"千问或三八"。注意中文相邻字符间没有 \b 边界，
# 故不能用 \b 锚定，改用负向前瞻排除"三八妇女节/三八节"等常见歧义。
_CLEAN_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"三八(?![妇女]|街|级|神)"), "3B/8B"),  # 本地大模型尺寸误写（千问 3B/8B）
    (re.compile(r"[ \t]{2,}"), " "),          # 多余空白
    (re.compile(r"\.{2,}"), "。"),            # ASR 停顿拍点（连续句号）归一
]


def clean_transcript(text: str) -> str:
    """对逐字稿做保守的专名/噪声清洗（只改高置信项，不猜不删内容）。"""
    if not text:
        return text
    out = text
    for pat, repl in _CLEAN_RULES:
        out = pat.sub(repl, out)
    return out


def generate_screenshot_markers(transcript: str, title: str,
                                config: dict, max_count: int = 5) -> list[dict]:
    """让 LLM 分析字幕，决定哪些时间点需要截图

    返回: [{"timestamp": 120.5, "reason": "展示代码示例"}, ...]
    """
    prompt = render(
        load_prompt("screenshot-markers-user"),
        {"title": title, "transcript": transcript[:4000], "max_count": max_count},
    )

    result = _chat_completion(config, [{"role": "user", "content": prompt}],
                              max_tokens=500, timeout=30)

    text = result["content"].strip().strip("```")
    text = re.sub(r"^json\s*", "", text, flags=re.IGNORECASE)

    try:
        markers = json.loads(text)
        if not isinstance(markers, list):
            return []
        # 规范化: 只保留含 timestamp 的项
        return [m for m in markers
                if isinstance(m, dict) and "timestamp" in m][:max_count]
    except json.JSONDecodeError:
        # 从非严格 JSON 中提取 [mm:ss] 时间戳
        return _extract_timestamps_from_text(text, max_count)


def _extract_timestamps_from_text(text: str, max_count: int) -> list[dict]:
    """容错: 从模型返回的文本中提取时间戳（支持 [M:SS] / [H:MM:SS] 格式）"""
    markers = []
    pattern = r"\[(\d+):(\d{2})(?::(\d{2}))?\]"
    for m in re.finditer(pattern, text):
        if m.group(3):
            ts = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
        else:
            ts = int(m.group(1)) * 60 + int(m.group(2))
        markers.append({"timestamp": float(ts), "reason": text[max(0, m.start()-20):m.end()]})
        if len(markers) >= max_count:
            break
    return markers


def should_use_visual(transcript: str, title: str) -> bool:
    """智能调度: 根据内容特征判断是否需要视觉分析（成本控制）

    视觉关键词密度高 + 视频不长 → 启用视觉模式。
    """
    visual_keywords = ["演示", "代码", "图表", "操作", "画面", "展示",
                       "教程", "安装", "配置", "效果", "对比"]
    text_len = len(transcript)
    keyword_hits = sum(1 for kw in visual_keywords if kw in transcript)

    # 短视频 + 高视觉关键词密度 → 启用视觉
    if text_len < 2000 and keyword_hits >= 2:
        return True
    # 中等长度 + 高视觉关键词密度 → 启用视觉
    if text_len < 5000 and keyword_hits >= 3:
        return True
    return False


# ========== CLI ==========

def main():
    parser = argparse.ArgumentParser(description="视觉模型分析工具")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # analyze 子命令（无图片，用于测试连通性）
    p_analyze = sub.add_parser("analyze", help="文本分析（测试API连通性）")
    p_analyze.add_argument("--text", required=True, help="输入文本")
    p_analyze.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="配置文件路径")
    p_analyze.add_argument("--model", default="visual", help="配置中使用的模型键 (default|visual)")

    # markers 子命令（分析字幕生成截图时间点）
    p_markers = sub.add_parser("markers", help="生成截图时间点")
    p_markers.add_argument("transcript_file", help="字幕文件路径")
    p_markers.add_argument("--title", default="未知视频", help="视频标题")
    p_markers.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="配置文件路径")
    p_markers.add_argument("--model", default="default", help="配置中使用的模型键 (default|visual)")
    p_markers.add_argument("--max-count", type=int, default=5, help="最大截图数")

    args = parser.parse_args()

    try:
        config = load_visual_config(args.config)
        model_config = config.get(args.model, config.get("default", {}))

        if args.cmd == "analyze":
            result = _chat_completion(
                model_config,
                [{"role": "user", "content": args.text}],
                max_tokens=2000, timeout=120,
            )
            print(result["content"])
        elif args.cmd == "markers":
            with open(args.transcript_file, "r", encoding="utf-8") as f:
                transcript = f.read()
            markers = generate_screenshot_markers(
                transcript, args.title, model_config, args.max_count)
            print(json.dumps(markers, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
