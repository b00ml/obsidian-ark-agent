#!/usr/bin/env python3
"""B站视频字幕提取工具 - 3层降级策略

输入 BV号或 B站视频链接，输出带时间戳的字幕文本。

策略:
  1. B站 API 直取字幕 (秒级)       BV -> cid -> subtitle_url -> JSON
  2. yt-dlp 下载字幕 (秒级)        yt-dlp --write-subs --skip-download
  3. 音频下载 + Whisper 转写(分钟级) yt-dlp -x -> transcribe.py

频道级批量摄取 (设计文档4.0 执行线#4):
  --channel / --batch-file 枚举视频清单 → 写入 inbox/queue.jsonl
  (枚举入队，不直接连跑；单视频三层降级流水线不变)

参考项目:
- BiliNote (Pipeline 状态机设计)
- astrbot_plugin_biliVideo (Cookie回退链、下载质量选项)
- bili-transcript (Skill集成方式)
- yt-dlp 实战指南 (412处理、Windows文件名安全)
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

from vault_io import controlled_write  # P0-01 受控写入：路径守卫+原子写+审计

# yt-dlp Python API（频道级批量枚举用；单视频下载仍走 YT_DLP_CMD 子进程）。
# 顶部导入便于测试 mock（patch bt.yt_dlp.YoutubeDL）；
# 缺失时仅频道枚举不可用，纯文本三层降级流水线不受影响。
try:
    import yt_dlp
except ImportError:
    yt_dlp = None

# 多模态功能（可选导入，缺依赖时降级为纯文本模式）
try:
    import video_frames as vf
    import visual_analyzer as va
    _VISUAL_AVAILABLE = True
except ImportError:
    vf = va = None
    _VISUAL_AVAILABLE = False

# ========== 国内网络优化 ==========
# HuggingFace 国内镜像（模型下载时）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 禁用 xet 协议（该协议在部分镜像站返回 401）
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# 提高网络容错，减少超时导致的失败
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")


def ensure_ffmpeg() -> None:
    """确保 ffmpeg 可用：系统 ffmpeg 优先，否则使用 imageio-ffmpeg 静态二进制
    （参考 astrbot: 系统已装则优先系统版，未装则用内置）
    """
    if shutil.which("ffmpeg"):
        return
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_dir = os.path.dirname(ffmpeg_exe)
        if ffmpeg_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
            print(f"[ffmpeg] 使用内置 ffmpeg: {os.path.basename(ffmpeg_exe)}")
    except ImportError:
        print("[ffmpeg] 警告: 未找到 ffmpeg，音频下载/合并可能失败")
        print("[ffmpeg] 安装: pip install imageio-ffmpeg 或 winget install ffmpeg")

# ========== 常量 ==========

# 字幕语言优先级（分两级: 人工CC优先 > AI字幕降级）
# AI 字幕默认启用，但必须经 validate_subtitle 把关（不匹配再降级）；
# 参考 BBDown 也支持 --skip-ai，但本项目默认行为相反：有就用，验证不过才不用。
CC_LANG_PRIORITY = ["zh-Hans", "zh-CN", "zh", "zh-Hant"]   # 人工字幕（可信）
AI_LANG_PRIORITY = ["ai-zh"]                                # AI字幕（默认启用，需 validate_subtitle 校验）
# 兼容旧引用
SUBTITLE_LANG_PRIORITY = CC_LANG_PRIORITY + AI_LANG_PRIORITY

# 下载质量: Whisper 转写 32k 音质足够，无需高音质浪费带宽
DOWNLOAD_QUALITY = {
    "fast": "7",   # 32k
    "medium": "5", # 64k
    "high": "0",   # 128k
}

# quality -> Whisper 模型: 原来固定 base。GPU 下 base->small 更快更准，high->medium 求更准。
# CPU 用户可设环境变量 BILI_WHISPER_MODEL=base 回调。
WHISPER_MODEL_MAP = {
    "fast": "small",
    "medium": "small",
    "high": "medium",
}
def resolve_whisper_model(quality: str) -> str:
    return os.environ.get("BILI_WHISPER_MODEL") or WHISPER_MODEL_MAP.get(quality, "small")

# 推理设备: None=自动检测(有GPU用cuda否则cpu)。可设 BILI_WHISPER_DEVICE=cuda/cpu
WHISPER_DEVICE = os.environ.get("BILI_WHISPER_DEVICE", None)
# beam: 1=贪心求快(默认)，5=求准。可设 BILI_WHISPER_BEAM=5
WHISPER_BEAM = int(os.environ.get("BILI_WHISPER_BEAM", "1"))
# VAD 静音过滤: 默认开，跳过大段沉默更省时间。设 BILI_WHISPER_VAD=0 关闭
WHISPER_VAD = os.environ.get("BILI_WHISPER_VAD", "1") == "1"



# yt-dlp 调用方式: 用 python -m yt_dlp 而非裸命令，
# 保证 venv 环境（PATH 不含 Scripts 目录）也能正常工作
YT_DLP_CMD = [sys.executable, "-m", "yt_dlp"]

BILI_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.bilibili.com",
}

# yt-dlp 公共参数（参考 yt-dlp 实战指南踩坑经验）
# --restrict-filenames : Windows 文件名安全（替换空格/特殊字符）
# --no-playlist        : 避免误下播放列表
# --add-header         : 添加 Referer 头解决 412 反爬
YT_DLP_BASE_ARGS = [
    "--restrict-filenames",
    "--no-playlist",
    "--add-header", "Referer:https://www.bilibili.com",
]

# 频道级批量摄取: 默认任务队列路径（仓库根/inbox/queue.jsonl，相对本文件定位）
DEFAULT_QUEUE_PATH = str(Path(__file__).resolve().parent.parent / "inbox" / "queue.jsonl")


# ========== 工具函数 ==========

# b23.tv 短链解析超时（秒）：短链 302 一跳即达 /video/BVxxx，超时快速失败（OPT-115）
SHORT_LINK_TIMEOUT = 10.0
# B站官方短域：不透明短码（如 b23.tv/GwA3BUs）需跟随重定向解析出 BV 号
_SHORT_LINK_RE = re.compile(r"https?://(?:b23\.tv|bili2233\.cn)/[\w./?#=&-]+", re.I)


def resolve_short_link(url: str, timeout: float = SHORT_LINK_TIMEOUT) -> str:
    """跟随 B站短链重定向，从最终地址提取 BV 号（OPT-115）。

    b23.tv 短链 302 一跳即达 /video/BVxxx，无需搜索引擎/外部 agent 反查；
    网络失败或解析不出时抛 ValueError 带原因（调用方按工具错误呈现给模型）。
    """
    try:
        resp = requests.get(url, headers=BILI_HEADERS, timeout=timeout, allow_redirects=True)
    except requests.RequestException as e:
        raise ValueError(f"短链解析失败（网络）: {url}: {e}") from e
    # 先看最终 URL，再退回响应 HTML 头部（极端情况落在中间页）
    for text in (str(resp.url), resp.text[:20000]):
        m = re.search(r"(BV\w{10})", text or "")
        if m:
            return m.group(1)
    raise ValueError(f"短链未解析出 BV 号: {url} → {resp.url}")


def extract_bvid(input_str: str) -> str:
    """从用户输入中提取BV号
    支持: BV1xx411x7xx / https://www.bilibili.com/video/BV1xx411x7xx/
          b23.tv / bili2233.cn 短链（不透明短码跟随重定向解析，OPT-115）
    """
    match = re.search(r"(BV\w{10})", input_str)
    if match:
        return match.group(1)
    short = _SHORT_LINK_RE.search(input_str)
    if short:
        return resolve_short_link(short.group(0))
    raise ValueError(f"无法从输入中提取BV号: {input_str}")


def format_timestamp(seconds: float) -> str:
    """秒数 → [H:MM:SS] 或 [M:SS] 格式（支持长视频）"""
    total = int(seconds)
    hours = total // 3600
    mins = (total % 3600) // 60
    secs = total % 60
    if hours > 0:
        return f"[{hours}:{mins:02d}:{secs:02d}]"
    return f"[{mins}:{secs:02d}]"


def load_cookie(cookie_path: str = "bilibili_cookie.json") -> dict:
    """加载B站Cookie（三级回退: 文件 → 浏览器 → 游客）"""
    if not os.path.exists(cookie_path):
        print("[Cookie] 未找到 bilibili_cookie.json，将尝试浏览器Cookie")
        print("[Cookie] 获取方式: 浏览器登录bilibili.com → F12 → Application → Cookies")
        return {}
    try:
        with open(cookie_path, "r", encoding="utf-8") as f:
            cookie = json.load(f)
        print("[Cookie] 已加载 bilibili_cookie.json")
        return cookie
    except json.JSONDecodeError as e:
        print(f"[Cookie] 配置文件格式错误: {e}")
        return {}


def refresh_cookie(cookie: dict, cookie_path: str = "bilibili_cookie.json") -> dict:
    """刷新B站Cookie（参考 BiliNote Cookie 自动刷新机制）

    当 SESSDATA 过期时，使用 bili_jct + ac_time_value 刷新登录态。
    刷新成功后自动回写到 cookie 文件。

    需要 cookie 中包含: SESSDATA, BILI_JCT, ac_time_value
    返回: 刷新后的 cookie dict（刷新失败则返回原 cookie）
    """
    if not cookie.get("SESSDATA") or not cookie.get("BILI_JCT"):
        return cookie

    ac_time_value = cookie.get("ac_time_value", "")
    if not ac_time_value:
        # 没有 ac_time_value，无法自动刷新
        return cookie

    headers = {
        "User-Agent": BILI_HEADERS["User-Agent"],
        "Referer": "https://www.bilibili.com",
        "Cookie": cookie_to_header(cookie),
    }

    try:
        # 直接尝试刷新（不预检查，刷新失败说明 Cookie 仍有效或无法刷新）
        refresh_url = "https://api.bilibili.com/x/correspond/1/refresh"
        refresh_data = {
            "csrf": cookie["BILI_JCT"],
            "correspond_one": ac_time_value,
        }
        resp = requests.post(refresh_url, headers=headers, data=refresh_data, timeout=10)
        result = resp.json()

        if result.get("code") == 0 and result.get("data", {}).get("cookie_info"):
            cookie_info = result["data"]["cookie_info"]
            # 更新 Cookie 字段
            for item in cookie_info.get("cookies", []):
                if item["name"] in ("SESSDATA", "BILI_JCT", "ac_time_value", "BUVID3"):
                    cookie[item["name"]] = item["value"]

            # 回写到文件
            with open(cookie_path, "w", encoding="utf-8") as f:
                json.dump(cookie, f, ensure_ascii=False, indent=2)

            print("[Cookie] Cookie 已自动刷新并保存")
            return cookie
        else:
            # code != 0 说明 Cookie 仍有效或刷新不可用
            return cookie

    except Exception as e:
        print(f"[Cookie] 刷新异常: {e}，使用现有Cookie继续")
        return cookie


def cookie_to_header(cookie: dict) -> str:
    """Cookie字典 → HTTP Header格式"""
    return "; ".join(f"{k}={v}" for k, v in cookie.items() if v)


def print_status(state: str, msg: str):
    """Pipeline 状态输出（参考 BiliNote 状态机）"""
    print(f"[{state}] {msg}")


def _silent_file_cookie(path: str = "bilibili_cookie.json") -> dict:
    """静默读取文件 Cookie（结构非法/缺失一律返回 {}），供 yt-dlp 层复用，不打日志。"""
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}
    return {}


def _write_cookies_file(cookie: dict) -> str:
    """把 Cookie 字典写成 Netscape 格式 cookies.txt，返回路径。

    理由：yt-dlp 传真实会话必须走 `--cookies <file>`；`--add-header Cookie:` 已废弃
    且会被 B 站 412 反爬拦截（对比实测：--cookies 成功、--add-header 失败）。
    """
    lines = ["# Netscape HTTP Cookie File",
             "# This file was generated for yt-dlp by bili_transcript.py"]
    for k, v in cookie.items():
        if isinstance(v, str) and v:
            lines.append("\t".join([".bilibili.com", "TRUE", "/", "TRUE",
                                    "2147483647", k, v]))
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "temp_cookies.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def run_ytdlp(args: list, timeout: int, check: bool = True) -> subprocess.CompletedProcess:
    """执行 yt-dlp 命令，带 Cookie 降级重试
    三级回退: Cookie文件 → 浏览器Cookie → 游客模式
    浏览器Cookie 依次尝试 edge → chrome（用户常用 Edge，也兼容 Chrome；均失败才降级游客）
    args: 不含 yt-dlp 本体，如 ["--get-title", url]
    注意: 中文 Windows 下 yt-dlp 输出为 GBK，需 errors="replace" 容错
    """
    cmd = YT_DLP_CMD + args
    # 优先级1: 文件 Cookie（bilibili_cookie.json）转成 cookies.txt 给 yt-dlp
    # 游客模式无 Cookie/buvid 访问部分视频会被 B 站 412 反爬拦截，带上真实会话更稳妥
    file_cookie = _silent_file_cookie()
    if file_cookie:
        try:
            cf = _write_cookies_file(file_cookie)
            proc = subprocess.run(cmd + ["--cookies", cf],
                                  capture_output=True, text=True,
                                  timeout=timeout, check=False,
                                  encoding="utf-8", errors="replace")
            if proc.returncode == 0:
                return proc
        except subprocess.TimeoutExpired:
            raise
        except subprocess.SubprocessError:
            pass  # 文件 Cookie 层异常，继续降级
    # 优先级2: 浏览器Cookie（Edge 优先，失败再试 Chrome）
    for browser in ("edge", "chrome"):
        try:
            proc = subprocess.run(cmd + ["--cookies-from-browser", browser],
                                  capture_output=True, text=True,
                                  timeout=timeout, check=False,
                                  encoding="utf-8", errors="replace")
            if proc.returncode == 0:
                return proc
        except subprocess.TimeoutExpired:
            raise
        except subprocess.SubprocessError:
            continue  # 该浏览器 Cookie 取不到，试下一个
    # 优先级3: 浏览器Cookie均失败 → 游客模式重试
    print("[Cookie] 浏览器Cookie获取失败，降级为游客模式")
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout, check=check,
                          encoding="utf-8", errors="replace")


# ========== 策略 1: B站 API ==========

def get_video_info(bvid: str, cookie: dict) -> dict:
    """Step A: BV号 → cid + 视频标题"""
    url = f"https://api.bilibili.com/x/player/pagelist?bvid={bvid}"
    headers = dict(BILI_HEADERS)
    if cookie:
        headers["Cookie"] = cookie_to_header(cookie)

    resp = requests.get(url, headers=headers, timeout=10)
    data = resp.json()

    if data["code"] != 0:
        raise RuntimeError(f"API错误: {data.get('message', 'unknown')}")

    page = data["data"][0]
    return {"cid": page["cid"], "title": page["part"], "bvid": bvid,
            "duration": page.get("duration", 0)}


def get_video_meta(bvid: str, cookie: dict) -> dict:
    """获取视频完整元数据（标题/作者/发布日期/时长），用于生成笔记 YAML frontmatter。

    用 web-interface/view API（get_video_info 的 pagelist 不返回 author/pubdate）。
    失败时返回含 bvid 的最小 meta，不抛异常——元数据获取失败不应阻断字幕流程。
    """
    import datetime as _dt
    url = f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
    headers = dict(BILI_HEADERS)
    if cookie:
        headers["Cookie"] = cookie_to_header(cookie)
    fallback = {
        "title": bvid, "author": "", "date": "",
        "duration": 0, "bvid": bvid,
        "source_url": f"https://www.bilibili.com/video/{bvid}",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        data = resp.json()
        if data.get("code") != 0:
            return fallback
        d = data["data"]
        return {
            "title": d.get("title", bvid),
            "author": d.get("owner", {}).get("name", ""),
            "date": _dt.datetime.fromtimestamp(d["pubdate"]).strftime("%Y-%m-%d")
                    if d.get("pubdate") else "",
            "duration": d.get("duration", 0),
            "bvid": bvid,
            "source_url": f"https://www.bilibili.com/video/{bvid}",
        }
    except Exception as e:
        print_status("META", f"元数据获取失败，使用最小 meta: {e}")
        return fallback


def get_stream_url(bvid: str, cid: int, cookie: dict, qn: int = 64) -> str | None:
    """通过 playurl API 获取视频流地址（DASH 格式），用于流式截帧

    无需下载完整视频，ffmpeg 直接从流 URL seek 截帧。
    Args:
        bvid: BV号
        cid: 视频 cid（从 get_video_info 获取）
        cookie: Cookie dict
        qn: 画质代码（64=720p, 80=1080p）。截图用 720p 足够
    Returns:
        视频流 URL（DASH video baseUrl），失败返回 None
    """
    url = (f"https://api.bilibili.com/x/player/playurl"
           f"?bvid={bvid}&cid={cid}&qn={qn}&fnval=16&fourk=0")
    headers = dict(BILI_HEADERS)
    if cookie:
        headers["Cookie"] = cookie_to_header(cookie)

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        data = resp.json()
        if data["code"] != 0:
            print(f"[STREAM] playurl API 错误: {data.get('message', 'unknown')}")
            return None

        dash = data.get("data", {}).get("dash", {})
        videos = dash.get("video", [])
        if not videos:
            return None

        # 选最高可用画质（按 id 降序）
        videos.sort(key=lambda v: v.get("id", 0), reverse=True)
        stream_url = videos[0].get("baseUrl") or videos[0].get("base_url")
        if not stream_url:
            return None

        # 补全 https
        if stream_url.startswith("//"):
            stream_url = "https:" + stream_url
        return stream_url
    except (requests.RequestException, KeyError, ValueError) as e:
        print(f"[STREAM] 获取流地址失败: {e}")
        return None


def _pick_subtitle_url(subtitles: list[dict]) -> tuple[str, bool] | None:
    """从 subtitles 列表选出最合适的字幕。返回 (subtitle_url, is_ai) 或 None。

    分级：人工CC（可信，直接信任）> AI字幕（不可信，需上层验证）> 兜底第一个（视作AI）。
    """
    if not subtitles:
        return None
    # 第一级：人工CC字幕（可信）
    for lang in CC_LANG_PRIORITY:
        for sub in subtitles:
            if sub.get("lan") == lang:
                return sub["subtitle_url"], False
    # 第二级：AI字幕（不可信，需后续验证）
    for lang in AI_LANG_PRIORITY:
        for sub in subtitles:
            if sub.get("lan") == lang:
                return sub["subtitle_url"], True
    # 兜底：取第一个（标记为AI，需验证）
    return subtitles[0]["subtitle_url"], True


def get_subtitle_url(cid: int, bvid: str, cookie: dict) -> tuple[str, bool] | None:
    """Step B（传统路径）: cid → 字幕URL + 是否为AI字幕

    用无签名的 /x/player/v2 请求。B站风控逐步要求 w_rid/wts 后，
    该接口可能返回空——故上方先走 WBI 签名路径（get_subtitle_url_wbi）。
    返回: (subtitle_url, is_ai) 或 None
    """
    url = f"https://api.bilibili.com/x/player/v2?cid={cid}&bvid={bvid}"
    headers = dict(BILI_HEADERS)
    if cookie:
        headers["Cookie"] = cookie_to_header(cookie)

    resp = requests.get(url, headers=headers, timeout=10)
    data = resp.json()

    subtitles = data.get("data", {}).get("subtitle", {}).get("subtitles", [])
    return _pick_subtitle_url(subtitles)


def get_subtitle_url_wbi(cid: int, bvid: str, cookie: dict) -> tuple[str, bool] | None:
    """Step B（WBI 签名路径）: 用 bilibili-api-python 稳定取字幕

    复用库内 WBI 签名算法请求 /x/player/wbi/v2，规避无签名接口被风控返回空，
    也避免“登录态缓存串台”（返回其他视频字幕）——这正是之前 AI 字幕不匹配的根因之一。
    返回: (subtitle_url, is_ai) 或 None；库缺失 / 请求失败返回 None（由上层回退传统路径）。
    """
    try:
        from bilibili_api import video, Credential, sync
    except Exception:
        print("[WBI] bilibili-api-python 未安装，回退传统路径")
        return None
    try:
        cred = None
        if cookie and cookie.get("SESSDATA"):
            cred = Credential(sessdata=cookie.get("SESSDATA"),
                              bili_jct=cookie.get("BILI_JCT") or None)
        # sync() 把异步协程在同步上下文跑起来（本流水线为同步 requests 风格）
        v = video.Video(bvid=bvid, credential=cred)
        sub = sync(v.get_subtitle(cid=cid)) or {}
        subtitles = (sub or {}).get("subtitles") or []
        if not subtitles and cred is not None:
            # OPT-110 诊断：带登录态却拿不到字幕 → 大概率 SESSDATA 过期（B站把 AI 字幕
            # 藏在登录态后，过期 cookie 等同游客）。显式告警，不再静默降级成音转错字。
            import requests as _rq
            try:
                nav = _rq.get("https://api.bilibili.com/x/web-interface/nav",
                              headers={"User-Agent": "Mozilla/5.0",
                                       "Cookie": f"SESSDATA={cred.sessdata}"}, timeout=10).json()
                if nav.get("code") == -101 or not (nav.get("data") or {}).get("isLogin"):
                    print("[SUBTITLE] ⚠️ 登录态已过期（SESSDATA 失效），AI 字幕不可用 → "
                          "将降级音频转写（同音错字概率高）。请更新 bilibili_cookie.json 的 SESSDATA。")
            except Exception:
                pass
        return _pick_subtitle_url(subtitles)
    except Exception as e:
        print(f"[WBI] 签名请求字幕失败: {e}，回退传统路径")
        return None


def download_subtitle(subtitle_url: str) -> str:
    """Step C: 下载字幕JSON → 格式化文本"""
    if not subtitle_url.startswith("http"):
        subtitle_url = f"https:{subtitle_url}"

    resp = requests.get(subtitle_url, headers=BILI_HEADERS, timeout=10)
    subtitle_json = resp.json()

    lines = []
    for item in subtitle_json.get("body", []):
        timestamp = format_timestamp(item["from"])
        lines.append(f"{timestamp} {item['content']}")

    return "\n".join(lines)


def validate_subtitle(title: str, transcript: str, duration: int = 0) -> bool:
    """验证字幕内容与视频标题是否匹配

    B站 AI 字幕存在缓存错误问题：登录态下 API 可能返回其他视频的字幕。
    通过以下维度检测不匹配：
    1. 时间戳跨度 vs 视频时长（字幕末尾时间戳不应远超视频时长）
    2. 标题关键词在字幕中是否有出现（至少匹配 1 个关键词）

    title: 视频标题
    transcript: 字幕文本（带 [M:SS] 时间戳）
    duration: 视频时长（秒），0 表示不检查
    """
    # 维度 1: 时间戳跨度检查
    if duration > 0:
        timestamps = re.findall(r"\[(\d+):(\d+)(?::(\d+))?\]", transcript)
        if timestamps:
            last_ts = timestamps[-1]
            if last_ts[2]:  # H:MM:SS
                total_secs = int(last_ts[0]) * 3600 + int(last_ts[1]) * 60 + int(last_ts[2])
            else:  # M:SS
                total_secs = int(last_ts[0]) * 60 + int(last_ts[1])
            # 字幕末尾时间戳超过视频时长 2 倍 → 可疑
            if total_secs > duration * 2:
                print_status("VALIDATE", f"字幕时间戳({total_secs}s)远超视频时长({duration}s)，可疑")
                return False

    # 维度 2: 标题关键词匹配
    # 从标题提取关键词（中文 2 字以上，英文 3 字母以上）
    keywords = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", title)
    if not keywords:
        return True  # 无法提取关键词，跳过检查

    # 取前 5 个关键词，检查字幕中是否至少出现 1 个
    transcript_lower = transcript.lower()
    matched = 0
    for kw in keywords[:5]:
        if kw.lower() in transcript_lower:
            matched += 1

    if matched == 0:
        print_status("VALIDATE", f"标题关键词({keywords[:5]})在字幕中均未出现，可疑")
        return False

    return True


def strategy_api(bvid: str, cookie: dict, trust_ai: bool = False) -> tuple[str, str] | None:
    """策略1: 通过B站API获取字幕

    - 人工CC字幕：直接信任，无需验证
    - AI字幕：默认启用并下载，但必须通过 validate_subtitle 把关（不匹配再降级/重试）
      —— 解决“之前 B站 AI字幕 与视频不匹配”问题：用 WBI 签名接口取对应视频字幕，
      再用 validate_subtitle 二次校验，失败降级到策略2/3。
      保留 trust_ai 参数仅为向后兼容，不再作为“是否使用 AI 字幕”的开关。

    签名路径：先 WBI（稳定），失败回退传统无签名路径。
    返回: (title, transcript_text) 或 None
    """
    try:
        print_status("PARSING", f"获取视频信息: {bvid}")
        info = get_video_info(bvid, cookie)

        print_status("SUBTITLE", "尝试B站API获取字幕...")
        sub_info = get_subtitle_url_wbi(info["cid"], bvid, cookie)
        if sub_info is None:
            # WBI 签名路径不可用/失败 → 回退传统无签名路径
            sub_info = get_subtitle_url(info["cid"], bvid, cookie)
        if not sub_info:
            print_status("SUBTITLE", "视频无字幕，降级到策略2")
            return None

        subtitle_url, is_ai = sub_info

        # 人工CC字幕：直接信任
        if not is_ai:
            transcript = download_subtitle(subtitle_url)
            if not transcript.strip():
                print_status("SUBTITLE", "字幕内容为空，降级到策略2")
                return None
            print_status("DONE", f"成功获取字幕 (API-CC, {len(transcript)}字)")
            return info["title"], transcript

        # AI字幕：默认启用，下载并 validate_subtitle 把关，最多重试2次
        max_retries = 2
        for attempt in range(max_retries):
            transcript = download_subtitle(subtitle_url)
            if not transcript.strip():
                print_status("SUBTITLE", f"字幕内容为空 (尝试 {attempt+1}/{max_retries})")
                continue

            if validate_subtitle(info["title"], transcript, info.get("duration", 0)):
                print_status("DONE", f"成功获取字幕 (API-AI, {len(transcript)}字, 尝试 {attempt+1})")
                return info["title"], transcript

            print_status("SUBTITLE", f"AI字幕内容不匹配 (尝试 {attempt+1}/{max_retries})")
            # 重试时重新取字幕URL（服务器可能返回不同结果）
            if attempt < max_retries - 1:
                print_status("SUBTITLE", "重试获取AI字幕...")
                new_sub = get_subtitle_url_wbi(info["cid"], bvid, cookie)
                if new_sub is None:
                    new_sub = get_subtitle_url(info["cid"], bvid, cookie)
                if new_sub:
                    subtitle_url = new_sub[0]

        print_status("SUBTITLE", f"AI字幕{max_retries}次验证均失败，降级到策略2")
        return None
    except Exception as e:
        print_status("SUBTITLE", f"策略1失败: {e}，降级到策略2")
        return None


# ========== 字幕缓存复用 ==========

def check_transcript_cache(bvid: str) -> tuple[str, str] | None:
    """命中 `transcript_{bvid}.txt` 缓存 → 直接复用，跳过三层降级流水线。

    同一 BV 二次处理直接走缓存（秒级），避免重复下载/转写。title 从 `meta_{bvid}.json` 读取。
    返回 (title, transcript) 或 None。
    """
    tpath = f"transcript_{bvid}.txt"
    if not os.path.exists(tpath):
        return None
    try:
        with open(tpath, "r", encoding="utf-8") as f:
            transcript = f.read()
    except OSError:
        return None
    if not transcript.strip():
        return None
    title = bvid
    mpath = f"meta_{bvid}.json"
    if os.path.exists(mpath):
        try:
            with open(mpath, "r", encoding="utf-8") as f:
                meta = json.load(f)
            title = meta.get("title") or bvid
        except (OSError, json.JSONDecodeError):
            pass
    return title, transcript


# ========== 策略 2: yt-dlp 字幕 ==========

def get_title_via_ytdlp(bvid: str) -> str | None:
    """通过 yt-dlp 获取视频标题"""
    url = f"https://www.bilibili.com/video/{bvid}"
    try:
        result = run_ytdlp(["--get-title"] + YT_DLP_BASE_ARGS + [url], timeout=30)
        title = result.stdout.strip()
        return title or None
    except Exception:
        return None


def strategy_ytdlp_subtitle(bvid: str) -> tuple[str, str] | None:
    """策略2: 用yt-dlp下载字幕"""
    url = f"https://www.bilibili.com/video/{bvid}"
    try:
        title = get_title_via_ytdlp(bvid)

        print_status("SUBTITLE", "尝试yt-dlp下载字幕...")
        result = run_ytdlp(
            ["--write-subs", "--skip-download",
             "--sub-lang", "zh-Hans,zh-CN,zh,ai-zh"]
            + YT_DLP_BASE_ARGS + ["-o", "temp_subtitle", url],
            timeout=60
        )

        # 查找下载的字幕文件
        for ext in [".zh-Hans.vtt", ".zh-CN.vtt", ".zh.vtt", ".ai-zh.vtt",
                    ".zh-Hans.srt", ".zh.srt", ".ai-zh.srt"]:
            sub_file = f"temp_subtitle{ext}"
            if os.path.exists(sub_file):
                with open(sub_file, "r", encoding="utf-8") as f:
                    transcript = f.read()
                os.remove(sub_file)
                print_status("DONE", f"成功获取字幕 (yt-dlp, {len(transcript)}字)")
                return title or bvid, transcript

        print_status("SUBTITLE", "无字幕文件，降级到策略3")
        return None
    except Exception as e:
        print_status("SUBTITLE", f"策略2失败: {e}，降级到策略3")
        return None


# ========== 策略 3: 音频下载 + Whisper ==========

def download_audio(url: str, audio_quality: str) -> str:
    """下载音频，返回本地文件路径

    方式1: -x --audio-format mp3 (需系统完整 ffmpeg，含 ffprobe)
    方式2: 直接下载 bestaudio (m4a)，无需 ffmpeg 后处理
            faster-whisper (PyAV) 原生支持 m4a 解码
    """
    # 方式1: 转 mp3
    try:
        run_ytdlp(
            ["-x", "--audio-format", "mp3",
             "--audio-quality", audio_quality]
            + YT_DLP_BASE_ARGS + ["-o", "temp_audio.%(ext)s", url],
            timeout=300
        )
        if os.path.exists("temp_audio.mp3"):
            return "temp_audio.mp3"
        if os.path.exists("temp_audio.m4a"):
            print("[Audio] ffmpeg后处理失败（可能缺ffprobe），使用原始m4a")
            return "temp_audio.m4a"
        raise FileNotFoundError("音频未生成")
    except subprocess.CalledProcessError:
        # 方式2: 直接下载原始音频
        if os.path.exists("temp_audio.m4a"):
            print("[Audio] 转mp3失败，使用原始m4a")
            return "temp_audio.m4a"
        run_ytdlp(
            ["-f", "bestaudio", "-o", "temp_audio.%(ext)s"]
            + YT_DLP_BASE_ARGS + [url],
            timeout=300
        )
        for name in os.listdir("."):
            if name.startswith("temp_audio.") and not name.endswith(".txt"):
                return name
        raise FileNotFoundError("音频下载失败")


def strategy_whisper(bvid: str, quality: str = "fast",
                     whisper_model: str | None = None,
                     device: str | None = None,
                     beam: int | None = None,
                     vad: bool | None = None) -> tuple[str, str] | None:
    """策略3: 下载音频 + Whisper转写（支持断点续传 + GPU 自动加速）

    转写中断后再次运行同一BV号时，自动从断点继续。
    whisper_model/device/beam/vad 均可显式传入；None 时按环境变量或默认值。
    """
    url = f"https://www.bilibili.com/video/{bvid}"
    try:
        title = get_title_via_ytdlp(bvid)

        # 下载音频
        audio_quality = DOWNLOAD_QUALITY.get(quality, "7")
        print_status("TRANSCRIBING", f"下载音频中 (quality={quality})...")
        audio_file = download_audio(url, audio_quality)

        # 转写（传递 --resume 支持断点续传）
        transcript_path = f"{os.path.splitext(audio_file)[0]}.txt"
        cache_path = f"{transcript_path}.cache"

        if os.path.exists(cache_path):
            print_status("TRANSCRIBING", "检测到上次转写缓存，断点续传中...")
        else:
            print_status("TRANSCRIBING", "Whisper转写中（可能需要几分钟）...")

        # 解析推理参数：显式参数优先，其次环境变量/映射，最后默认值
        model = whisper_model or resolve_whisper_model(quality)
        dev = device if device is not None else WHISPER_DEVICE
        b = beam if beam is not None else WHISPER_BEAM
        use_vad = vad if vad is not None else WHISPER_VAD

        script_dir = os.path.dirname(os.path.abspath(__file__))
        transcribe_script = os.path.join(script_dir, "transcribe.py")
        cmd = [sys.executable, transcribe_script, audio_file,
               "--model", str(model), "--language", "zh", "--beam", str(b),
               "--resume"]
        if dev:
            cmd += ["--device", dev]
        if use_vad:
            cmd += ["--vad"]
        print_status("TRANSCRIBING",
                     f"Whisper model={model} device={'auto' if not dev else dev} beam={b} vad={use_vad}")
        subprocess.run(cmd, check=True, timeout=600)

        # 读取转写结果
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = f.read()

        # 清理临时文件（含缓存）
        for f in ["temp_audio.mp3", "temp_audio.m4a", "temp_audio.txt",
                  "temp_audio.txt.cache"]:
            if os.path.exists(f):
                os.remove(f)

        print_status("DONE", f"成功获取字幕 (Whisper, {len(transcript)}字)")
        return title or bvid, transcript
    except Exception as e:
        # 转写失败时保留缓存，下次可断点续传
        print_status("TRANSCRIBING", f"策略3失败: {e}")
        print_status("TRANSCRIBING", "缓存已保留，下次运行将断点续传")
        return None


# ========== 多模态: 视觉分析 (Phase 2-4) ==========

def resolve_output_dirs(vault_path: str | None, safe_title: str):
    """根据是否指定 Vault 路径，决定输出目录

    vault_path 提供时:
      截图 → {vault}/raw/screenshots/{safe_title}/
      笔记 → {vault}/Inbox/
    否则 (本地模式):
      截图 → ./screenshots/{safe_title}/
      笔记 → ./ 当前目录

    返回: (screenshot_dir, note_dir, in_vault)
    """
    if vault_path:
        shot_dir = os.path.join(vault_path, "raw", "screenshots", safe_title)
        note_dir = os.path.join(vault_path, "Inbox")
    else:
        shot_dir = os.path.join("screenshots", safe_title)
        note_dir = "."
    os.makedirs(shot_dir, exist_ok=True)
    if vault_path:
        os.makedirs(note_dir, exist_ok=True)
    return shot_dir, note_dir, bool(vault_path)


def download_video(bvid: str) -> str:
    """下载完整视频（用于截帧/截图），返回本地文件路径

    格式: bestvideo+bestaudio 合并为 mp4（需 ffmpeg）
    """
    url = f"https://www.bilibili.com/video/{bvid}"
    print_status("VISUAL", "下载视频中（用于画面分析）...")
    result = run_ytdlp(
        ["-f", "bestvideo+bestaudio/best", "--merge-output-format", "mp4",
         "-o", "temp_video.%(ext)s"] + YT_DLP_BASE_ARGS + [url],
        timeout=600
    )
    for name in ["temp_video.mp4", "temp_video.mkv", "temp_video.webm"]:
        if os.path.exists(name):
            print_status("VISUAL", f"视频已就绪: {name}")
            return name
    raise FileNotFoundError("视频下载失败")


def run_screenshot_mode(bvid: str, title: str, transcript: str,
                        model_config: dict, screenshot_cfg: dict,
                        vault_path: str | None = None,
                        cookie: dict | None = None) -> str | None:
    """模式A: 关键截图模式（流式截帧优先，下载兜底）

    LLM 分析字幕决定截图时间点 → 优先流式截帧（不下载视频）
    → 流地址获取失败时降级为下载完整视频
    vault_path 提供时截图存入 {vault}/raw/screenshots/，笔记写入 {vault}/Inbox/
    返回: 生成的 Markdown 笔记路径（未生成则 None）
    """
    max_count = screenshot_cfg.get("max_count", 5)
    width = screenshot_cfg.get("width", 1280)

    print_status("VISUAL", "LLM 分析字幕，决定截图时间点...")
    markers = va.generate_screenshot_markers(transcript, title,
                                             model_config, max_count)
    if not markers:
        print_status("VISUAL", "未生成截图标记，跳过截图模式")
        return None

    # 确定输出目录
    safe_title = re.sub(r'[\\/:*?"<>|]', "_", title)
    shot_dir, note_dir, in_vault = resolve_output_dirs(vault_path, safe_title)

    # 优先：流式截帧（不下载视频）
    stream_url = None
    video_path = None
    if cookie:
        try:
            info = get_video_info(bvid, cookie)
            stream_url = get_stream_url(bvid, info["cid"], cookie)
        except Exception:
            pass

    if stream_url:
        print_status("VISUAL", f"流式截帧模式（无需下载视频）")
        stream_headers = {}
        if cookie.get("SESSDATA"):
            stream_headers["Cookie"] = cookie_to_header(cookie)
        shot_paths = []
        for i, marker in enumerate(markers):
            ts = float(marker["timestamp"])
            out = os.path.join(shot_dir, f"screenshot_{i+1:03d}.jpg")
            print_status("VISUAL", f"截图 [{ts:.0f}s]: {marker.get('reason', '')[:30]}")
            try:
                vf.capture_screenshot_from_stream(
                    stream_url, ts, out, width=width,
                    headers=stream_headers, timeout=30)
            except RuntimeError as e:
                print(f"  截帧失败: {e}")
            if os.path.exists(out):
                shot_paths.append((os.path.basename(out), marker.get("reason", "")))
    else:
        # 降级：下载完整视频
        print_status("VISUAL", "流地址获取失败，降级为下载视频...")
        video_path = download_video(bvid)
        shot_paths = []
        for i, marker in enumerate(markers):
            ts = float(marker["timestamp"])
            out = os.path.join(shot_dir, f"screenshot_{i+1:03d}.jpg")
            print_status("VISUAL", f"截图 [{ts:.0f}s]: {marker.get('reason', '')[:30]}")
            vf.capture_screenshot(video_path, ts, out, width=width)
            if os.path.exists(out):
                shot_paths.append((os.path.basename(out), marker.get("reason", "")))

    # 清理视频临时文件
    for name in ["temp_video.mp4", "temp_video.mkv", "temp_video.webm"]:
        if os.path.exists(name):
            os.remove(name)

    # 生成含截图的 Markdown 笔记（Obsidian 嵌入语法）
    note = [f"# {title}", "", "## 🎬 视频信息", f"- 标题: {title}",
            f"- 链接: https://www.bilibili.com/video/{bvid}", "",
            "## 🖼️ 关键截图", ""]
    for i, (fname, reason) in enumerate(shot_paths, 1):
        note.append(f"### 截图 {i}")
        if reason:
            note.append(f"**原因**: {reason}")
        note.append(f"![[{fname}]]")
        note.append("")
    note.append("## 📝 逐字稿")
    note.append(f"[[{title}-逐字稿]]")
    note.append("")

    note_path = os.path.join(note_dir, f"{safe_title}-screenshots.md")
    controlled_write(note_path, "\n".join(note),
                     vault_root=vault_path, overwrite=True, actor="bili-screenshot")
    print_status("DONE", f"截图笔记已生成: {note_path} ({len(shot_paths)}张)")
    return note_path


def run_visual_mode(bvid: str, title: str, transcript: str,
                    visual_config: dict, grid_cfg: dict,
                    vault_path: str | None = None,
                    cookie: dict | None = None) -> str | None:
    """模式B: 网格图理解模式（流式截帧优先，下载兜底）

    优先流式截帧（不下载视频） → ffmpeg 按间隔截帧 → PIL 拼接 3x3 网格图
    → 网格图 + 字幕文本 → 视觉模型 → 生成增强笔记
    vault_path 提供时网格图存入 {vault}/raw/screenshots/，笔记写入 {vault}/Inbox/
    返回: 生成的 Markdown 笔记路径（未生成则 None）
    """
    cols = grid_cfg.get("cols", 3)
    rows = grid_cfg.get("rows", 3)
    cell_w = grid_cfg.get("cell_width", 640)
    cell_h = grid_cfg.get("cell_height", 360)
    interval = grid_cfg.get("frame_interval", 10)
    max_grid = grid_cfg.get("max_grid_images", 10)

    # 优先：流式截帧（不下载视频）
    stream_url = None
    duration = 0
    if cookie:
        try:
            info = get_video_info(bvid, cookie)
            duration = info.get("duration", 0)
            stream_url = get_stream_url(bvid, info["cid"], cookie)
        except Exception:
            pass

    frames_dir = "frames_tmp"
    if stream_url and duration > 0:
        print_status("VISUAL", f"流式截帧模式（无需下载视频，时长 {duration}s）")
        stream_headers = {}
        if cookie.get("SESSDATA"):
            stream_headers["Cookie"] = cookie_to_header(cookie)
        frames = vf.extract_frames_from_stream(
            stream_url, duration, interval=interval,
            width=cell_w, height=cell_h,
            output_dir=frames_dir, headers=stream_headers,
            timeout_per_frame=30)
    else:
        # 降级：下载完整视频
        print_status("VISUAL", "流地址获取失败，降级为下载视频...")
        video_path = download_video(bvid)
        frames = vf.extract_frames(video_path, interval=interval,
                                   width=cell_w, height=cell_h,
                                   output_dir=frames_dir)

    if not frames:
        print_status("VISUAL", "截帧失败，跳过视觉模式")
        return None

    # 确定输出目录
    safe_title = re.sub(r'[\\/:*?"<>|]', "_", title)
    grid_dir, note_dir, in_vault = resolve_output_dirs(vault_path, safe_title)

    # 分组拼接网格图（不足一组丢弃）
    group_size = cols * rows
    grid_paths = []
    for i in range(0, len(frames), group_size):
        group = frames[i:i + group_size]
        if len(group) < group_size:
            break
        out = os.path.join(grid_dir, f"grid_{len(grid_paths)+1:03d}.jpg")
        vf.build_grid_image(group, cols=cols, rows=rows,
                            cell_w=cell_w, cell_h=cell_h, output=out)
        grid_paths.append(out)
        if len(grid_paths) >= max_grid:
            break

    # 逐张分析网格图 → 画面事实短卡（screen-only，禁整篇；OPT-092 责任分层）
    cards: dict[int, str] = {}
    for i, g in enumerate(grid_paths, 1):
        print_status("VISUAL", f"视觉模型分析网格图 {i}/{len(grid_paths)}...")
        b64 = vf.image_to_base64(g)
        result = va.analyze_grid_card(title, b64, visual_config)
        cards[i] = (result.get("content") or "").strip()

    # 清理临时文件
    for name in ["temp_video.mp4", "temp_video.mkv", "temp_video.webm"]:
        if os.path.exists(name):
            os.remove(name)
    if os.path.exists(frames_dir):
        for f in os.listdir(frames_dir):
            os.remove(os.path.join(frames_dir, f))
        os.rmdir(frames_dir)

    # 生成增强笔记：视频信息 + 逐帧短卡 + 单遍聚合（含一致性标注）
    note = [f"# {title}", "", "## 🎬 视频信息", f"- 标题: {title}",
            f"- 链接: https://www.bilibili.com/video/{bvid}", "",
            "## 🖼️ 画面分析（逐帧事实卡）", ""]
    for i in sorted(cards):
        note.append(f"### 网格图 {i}")
        note.append(f"![[{os.path.basename(grid_paths[i-1])}]]")
        note.append("")
        note.append(cards[i])
        note.append("")

    # 单遍聚合：合并全部短卡 + 字幕 → 只产一遍结构化画面笔记（去重+帧间不一致标注）
    if cards:
        try:
            synth = (va.synthesize_visual_note(title, cards, transcript, visual_config)
                     .get("content") or "").strip()
        except Exception as e:
            print_status("VISUAL", f"聚合失败，回退为逐卡展示: {e}")
            synth = ""
        if synth:
            note.append("## 📋 结构化画面总结（单遍聚合）")
            note.append("")
            note.append(synth)
            note.append("")

    note.append("## 📝 关联笔记")
    note.append(f"- [[{safe_title}-总结]]")
    note.append(f"- [[{safe_title}-逐字稿]]")
    note.append("")

    note_path = os.path.join(note_dir, f"{safe_title}-visual.md")
    controlled_write(note_path, "\n".join(note),
                     vault_root=vault_path, overwrite=True, actor="bili-visual")

    # 三件套补全：逐字稿 + 总结（角色分流；已存在则不覆盖）
    write_transcript_note(note_dir, safe_title, title, bvid, transcript, vault_root=vault_path)
    summary_path = write_summary_note(note_dir, safe_title, title, bvid, transcript,
                                      visual_config, vault_root=vault_path)

    # 知识编译（优化设计文档4.0 执行线#2，best-effort）：总结落盘后
    # LLM 提取概念/实体 → 增量更新 vault 的 wiki 概念/实体页。
    # 任何异常只打 [COMPILE] 警告，不影响三件套主流程。
    if KNOWLEDGE_COMPILE_ENABLED and vault_path:
        compile_knowledge_for_note(vault_path, summary_path,
                                   f"{safe_title}-总结", visual_config)

    # 死链检查：只扫本次刚写入的三件套，避免反复重扫既有笔记的噪音（OPT-095）
    written = {f"{safe_title}-visual", f"{safe_title}-总结", f"{safe_title}-逐字稿"}
    for m in check_note_links(note_dir, only=written):
        print_status("LINK", f"死链提醒: {m}")

    print_status("DONE", f"视觉分析笔记已生成: {note_path} ({len(grid_paths)}张网格图)")
    return note_path


def write_transcript_note(note_dir: str, safe_title: str, title: str,
                          bvid: str, transcript: str,
                          vault_root: str | None = None) -> str:
    """写入/复用《逐字稿》笔记（已存在则不覆盖；写入前做保守专名清洗 OPT-095/096）。"""
    target = os.path.join(note_dir, f"{safe_title}-逐字稿.md")
    if not os.path.exists(target):
        body = (
            f"---\ntitle: \"{title} - 逐字稿\"\ntype: video-transcript\n"
            f"source: \"https://www.bilibili.com/video/{bvid}\"\n"
            f"tags: [video, transcript]\n---\n\n"
            f"> 视频总结见 [[{safe_title}-总结]]\n\n\n"
            f"{va.clean_transcript(transcript)}\n"
        )
        controlled_write(target, body, vault_root=vault_root,
                         overwrite=True, actor="bili-transcript")
    return target


def write_summary_note(note_dir: str, safe_title: str, title: str,
                       bvid: str, transcript: str, config: dict,
                       vault_root: str | None = None) -> str:
    """生成《总结》笔记（听力内容归此，与画面笔记分工；OP-094 三件套）。"""
    target = os.path.join(note_dir, f"{safe_title}-总结.md")
    if os.path.exists(target):
        return target
    body = (f"# {title}\n\n## 🎬 视频信息\n- 标题: {title}\n"
            f"- 链接: https://www.bilibili.com/video/{bvid}\n\n")
    try:
        content = (va.summarize_transcript(title, transcript, config)
                   .get("content") or "").strip()
    except Exception as e:
        print_status("SUMMARY", f"总结生成失败，落占位: {e}")
        content = "> ⚠️ 总结生成失败，请核对画面笔记与逐字稿。"
    if not content:
        content = "> ⚠️ 总结为空，请核对画面笔记与逐字稿。"
    body += content + "\n\n## 📝 关联\n- [[{safe_title}-visual]]\n- [[{safe_title}-逐字稿]]\n".format(
        safe_title=safe_title)
    controlled_write(target, body, vault_root=vault_root,
                     overwrite=True, actor="bili-summary")
    return target


# 知识编译开关（优化设计文档4.0 执行线#2）：--vault 下总结笔记落盘后，
# LLM 提取概念/实体 → 增量合并/新建 wiki 页。编译为 best-effort，
# 任何异常只打 [COMPILE] 警告，绝不影响主笔记产出。
KNOWLEDGE_COMPILE_ENABLED = True


def compile_knowledge_for_note(vault_path: str, note_path: str, note_title: str,
                               model_config: dict) -> None:
    """总结笔记落盘后的知识编译入口（best-effort，异常只警告不中断主流程）。

    复用既有 --model-config 客户端（visual_analyzer._chat_completion）包装
    knowledge_compiler 所需的 llm_call；wiki 落在 {vault_path}/wiki/。
    """
    try:
        from knowledge_compiler import compile_note

        def llm_call(prompt: str) -> str:
            return va._chat_completion(
                model_config, [{"role": "user", "content": prompt}],
                max_tokens=800, timeout=60)["content"]

        with open(note_path, "r", encoding="utf-8") as f:
            note_content = f.read()
        result = compile_note(vault_path, note_path, note_title, note_content, llm_call)
        print_status("COMPILE", f"知识编译完成: 新建{len(result['created'])} "
                                f"合并{len(result['updated'])} "
                                f"冲突{len(result['conflicts'])} "
                                f"跳过{result['skipped']}")
    except Exception as e:
        print_status("COMPILE", f"知识编译跳过（不影响主笔记）: {e}")


def check_note_links(note_dir: str, only: set[str] | None = None) -> list[str]:
    """扫 <note_dir> 下各笔记的 [[target]]，校验缺失并汇总列表（OPT-095 死链检查）。

    非致命：只汇总提醒，不中断流水线。以 note_dir 为作用域（三件套互相引用即满足）。
    only: 传入本次刚写入的笔记 basename 集合时，只检查这些文件，避免每次全量重扫既有笔记产生噪音。
    [[*]]（含 ![[*]]）若带文件扩展名视为媒体/附件嵌入（Obsidian 从全库检索解析，
    网格图等存 raw/screenshots 不在 note_dir），不计入笔记死链。仅追踪「笔记」交叉引用。
    """
    import glob as _glob
    known = set()
    for p in _glob.glob(os.path.join(note_dir, "*.md")):
        known.add(os.path.splitext(os.path.basename(p))[0])
    media_ext = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
                 ".mp4", ".webm", ".mov", ".pdf", ".mp3", ".wav", ".md"}
    missing = []
    for p in _glob.glob(os.path.join(note_dir, "*.md")):
        base = os.path.splitext(os.path.basename(p))[0]
        if only is not None and base not in only:
            continue
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        for m in re.finditer(r"\[\[([^\]|#]+)", text):
            core = m.group(1).strip().split("#")[0].strip()
            if not core or os.path.splitext(core)[1].lower() in media_ext:
                continue  # 图片/音视频等媒体嵌入不算笔记死链
            if core not in known:
                missing.append(f"{os.path.basename(p)} → [[{m.group(1).strip()}]]")
    return sorted(set(missing))


# ========== 频道级批量摄取 (设计文档4.0 执行线#4) ==========

def enumerate_channel(url: str, max_items: int | None = None) -> list[dict]:
    """枚举频道/合集/播放列表下的视频清单（yt-dlp flat 模式，不解析单视频详情）

    批量摄取第一步：枚举结果交由 enqueue_bili_batch 写入 inbox 队列，
    由既有队列消费端逐条处理（枚举入队，不直接连跑）。

    Args:
        url: 频道/合集/播放列表 URL（yt-dlp 支持的 space/合集/播放列表页）
        max_items: 枚举条数上限；None 不限制
    Returns:
        [{"bvid": ..., "title": ...}, ...]，无法提取BV号的条目跳过并告警
    Raises:
        RuntimeError: yt_dlp 未安装
        其他异常（网络/解析）原样上抛，由调用方处理
    """
    if yt_dlp is None:
        raise RuntimeError("yt_dlp 未安装，无法枚举频道: pip install -U yt-dlp")
    opts = {
        "extract_flat": "in_playlist",  # 只取清单，不逐个解析视频详情
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }
    print_status("CHANNEL", f"开始枚举: {url}")
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    entries = (info or {}).get("entries") or []

    results: list[dict] = []
    for entry in entries:
        # bvid 优先取 entry.id，缺失时回退 entry.url（flat 条目两种形态都存在）
        candidates = []
        if isinstance(entry, dict):
            candidates = [str(entry[k]) for k in ("id", "url") if entry.get(k)]
        elif entry:
            candidates = [str(entry)]
        bvid = None
        for raw in candidates:
            try:
                bvid = extract_bvid(raw)
                break
            except ValueError:
                continue
        if not bvid:
            print_status("CHANNEL", f"跳过无法提取BV号的条目: {candidates or entry}")
            continue
        title = str(entry.get("title") or "") if isinstance(entry, dict) else ""
        results.append({"bvid": bvid, "title": title})
        print_status("CHANNEL", f"  [{len(results)}] {bvid} {title}")
        if max_items is not None and len(results) >= max_items:
            print_status("CHANNEL", f"已达上限 {max_items} 条，停止枚举")
            break
    print_status("CHANNEL", f"枚举完成: {len(results)} 条")
    return results


def parse_batch_file(path: str) -> list[str]:
    """解析批量文件：每行一个 BV号或URL，空行与 # 开头注释忽略

    用 extract_bvid 归一；无法提取BV号的行跳过并告警（不中断整批）。
    """
    bvids: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                bvids.append(extract_bvid(line))
            except ValueError:
                print_status("BATCH", f"第{lineno}行无法提取BV号，跳过: {line}")
    return bvids


def enqueue_bili_batch(queue_path: str, items: list[str],
                       source: str = "bili_channel") -> int:
    """把 BV 号列表逐条写入 inbox SQLite 队列。

    task 结构与 inbox_poll.poll 产出的任务同构，供队列消费端统一处理。
    dedupe_key 使用 inbox_poll.fingerprint 的 bili:{BV} 规则；跨次调用
    重复 BV 会被数据库唯一约束跳过。

    Args:
        queue_path: 兼容的队列路径；同名 .db 文件为 SQLite 队列
        items: BV 号列表
        source: 任务来源标记（写入 task.source）
    Returns:
        实际入队条数
    """
    if not items:
        print_status("BATCH", "无可入队任务（0 条）")
        return 0

    # 复用接收层的队列写入：把 ../inbox_collector 加入 sys.path 后导入 inbox_poll
    inbox_collector_dir = str(Path(__file__).resolve().parent.parent / "inbox_collector")
    if inbox_collector_dir not in sys.path:
        sys.path.insert(0, inbox_collector_dir)
    import inbox_poll
    from queue_store import InboxQueueStore

    queue_path = os.path.abspath(queue_path)  # 避免 enqueue 对裸文件名 makedirs("") 报错
    db_path = os.path.splitext(queue_path)[0] + ".db"
    legacy_seen = (os.path.join(os.path.dirname(queue_path), "seen.txt")
                   if os.path.basename(queue_path) == "queue.jsonl" else None)
    store = InboxQueueStore(db_path, queue_path=queue_path, seen_path=legacy_seen)
    store.migrate_legacy()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    received_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    count = 0
    for i, bvid in enumerate(items, 1):
        task = {
            "id": f"{ts}-{i:03d}",
            "type": "bili",
            "url": f"https://www.bilibili.com/video/{bvid}",
            "status": "pending",
            "retry_count": 0,
            "received_at": received_at,
            "email_subject": "channel-batch",
            "source": source,
            "error": None,
        }
        dedupe_key = inbox_poll.fingerprint(task["url"], "bili")
        if store.enqueue(task, dedupe_key=dedupe_key, markers=[dedupe_key]):
            count += 1
            print_status("BATCH", f"已入队 [{i}/{len(items)}] {bvid}")
        else:
            print_status("BATCH", f"重复跳过 [{i}/{len(items)}] {bvid}")
    print_status("BATCH", f"入队完成: {count} 条 → {db_path}")
    return count


# ========== 主流程 ==========

def build_arg_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数：单视频三层降级 + 频道级批量摄取（--channel/--batch-file）两条入口"""
    parser = argparse.ArgumentParser(description="B站视频字幕提取工具")
    parser.add_argument("input", nargs="?", default=None,
                        help="BV号或B站视频URL（--channel/--batch-file 批量模式下可省略）")
    parser.add_argument("--output", "-o", default=None, help="输出文件路径")
    parser.add_argument("--cookie", default="bilibili_cookie.json",
                        help="Cookie文件路径 (默认: bilibili_cookie.json)")
    parser.add_argument("--quality", default="fast", choices=["fast", "medium", "high"],
                        help="音频下载质量 (默认: fast=32k, 转写足够)")
    parser.add_argument("--trust-ai", action="store_true",
                        help="（保留兼容，不再作为开关）信任AI字幕。AI字幕现默认启用但经过内容校验")
    parser.add_argument("--no-cache", action="store_true",
                        help="忽略已生成的 transcript_<bvid>.txt 缓存，强制重新爬取/转写")
    parser.add_argument("--whisper-model", default=None,
                        help="Whisper模型 (base/small/medium/large-v3)；默认按 quality 自动映射")
    parser.add_argument("--whisper-device", default=None,
                        help="推理设备 auto/cuda/cpu；默认自动检测")
    parser.add_argument("--beam", type=int, default=None,
                        help="beam_size: 1=快(默认), 5=准")
    parser.add_argument("--no-vad", action="store_true",
                        help="关闭 VAD 静音过滤（默认开启）")
    parser.add_argument("--screenshot", action="store_true",
                        help="模式A: 关键截图（LLM决定截图时间点，需下载视频）")
    parser.add_argument("--visual", action="store_true",
                        help="模式B: 网格图理解（视觉模型分析画面，需下载视频）")
    parser.add_argument("--model-config", default=va.DEFAULT_CONFIG_PATH if va else "config/visual_models.json",
                        help="视觉模型配置文件路径 (默认: config/visual_models.json)")
    parser.add_argument("--vault", default=None,
                        help="Obsidian Vault 路径，如 C:/path/to/your/obsidian-vault（提供后截图存入 raw/screenshots/，笔记写入 Inbox/）")
    # ---- 频道级批量摄取（设计文档4.0 执行线#4）----
    parser.add_argument("--channel", default=None,
                        help="频道/合集/播放列表URL：枚举视频清单后批量入队（不直接连跑）")
    parser.add_argument("--batch-file", default=None,
                        help="批量文件路径：每行一个BV号或URL，空行与 # 开头注释忽略")
    parser.add_argument("--queue-path", default=DEFAULT_QUEUE_PATH,
                        help="任务队列路径 (默认: 仓库根 inbox/queue.jsonl)")
    parser.add_argument("--limit", type=int, default=None,
                        help="频道枚举条数上限（默认不限制）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印将入队的任务，不写队列")
    return parser


def main():
    # Windows 终端默认 GBK，Whisper 转写可能含无法编码的字符
    import sys as _sys
    if hasattr(_sys.stdout, "reconfigure"):
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = build_arg_parser()
    args = parser.parse_args()

    # ========== 频道级批量摄取模式（--channel / --batch-file）==========
    # 枚举入队，不直接连跑：视频逐条交给既有队列消费端处理
    if args.channel or args.batch_file:
        try:
            planned: list[str] = []
            if args.channel:
                planned.extend(e["bvid"] for e in
                               enumerate_channel(args.channel, max_items=args.limit))
            if args.batch_file:
                planned.extend(parse_batch_file(args.batch_file))
            # 同一次运行内保序去重；跨次重复由 SQLite dedupe_key 约束跳过
            deduped: list[str] = []
            for bvid in planned:
                if bvid not in deduped:
                    deduped.append(bvid)
            if not deduped:
                print_status("BATCH", "计划入队 0 条，无需写入队列")
                return
            print_status("BATCH", f"计划入队 {len(deduped)} 条，队列路径: {args.queue_path}")
            if args.dry_run:
                for bvid in deduped:
                    print_status("BATCH", f"[dry-run] https://www.bilibili.com/video/{bvid}")
                print_status("BATCH", "dry-run 模式：未写入队列")
                return
            count = enqueue_bili_batch(args.queue_path, deduped)
            print_status("DONE", f"批量入队完成: {count} 条 → {args.queue_path}")
        except Exception as e:
            print_status("BATCH", f"批量模式失败: {e}")
            sys.exit(1)
        return

    if not args.input:
        parser.error("需要提供 BV号/URL，或使用 --channel / --batch-file 批量模式")

    # 确保 ffmpeg 可用（策略3 需要）
    ensure_ffmpeg()

    # 提取BV号
    try:
        bvid = extract_bvid(args.input)
    except ValueError as e:
        print(f"错误: {e}")
        sys.exit(1)
    print(f"BV号: {bvid}")

    # 加载Cookie
    cookie = load_cookie(args.cookie)

    # 优先复用字幕缓存（同一BV二次处理直接秒级返回，跳过三层降级）
    result = None
    if not args.no_cache:
        result = check_transcript_cache(bvid)
        if result:
            title, transcript = result
            print_status("CACHE", f"复用已有字幕缓存 transcript_{bvid}.txt ({len(transcript)}字)，跳过爬取/转写")
            meta = get_video_meta(bvid, cookie)
            meta_path = f"meta_{bvid}.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        else:
            print_status("CACHE", "未命中字幕缓存，进入三层降级流水线")

    # 未命中缓存 → 3层降级策略
    if not result:
        result = strategy_api(bvid, cookie, trust_ai=args.trust_ai)

        # 策略1失败时尝试刷新Cookie并重试一次
        if not result and cookie:
            print_status("COOKIE", "尝试刷新Cookie后重试...")
            new_cookie = refresh_cookie(cookie, args.cookie)
            if new_cookie != cookie:
                cookie = new_cookie
                result = strategy_api(bvid, cookie, trust_ai=args.trust_ai)

        if not result:
            result = strategy_ytdlp_subtitle(bvid)
        if not result:
            result = strategy_whisper(
                bvid, args.quality,
                whisper_model=args.whisper_model,
                device=args.whisper_device,
                beam=args.beam,
                vad=(not args.no_vad),
            )

    if not result:
        print("\n所有策略均失败，请检查:")
        print("  1. BV号是否正确")
        print("  2. 是否已配置 bilibili_cookie.json 或浏览器已登录bilibili")
        print("  3. yt-dlp 和 ffmpeg 是否已安装: pip install -U yt-dlp")
        print("  4. yt-dlp 版本是否过旧: yt-dlp --version (需 2024.01+)")
        sys.exit(1)

    title, transcript = result

    # 输出字幕文本（默认文件名用 BV 号，避免中文标题在 Windows/GBK 终端乱码）
    output_path = args.output or f"transcript_{bvid}.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(transcript)

    # 同时输出元数据 JSON（含 author/date，供笔记生成直接读取，免去二次调 API）
    meta = get_video_meta(bvid, cookie)
    meta_path = f"meta_{bvid}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*40}")
    print(f"标题: {title}")
    print(f"字幕已保存到: {output_path}")
    print(f"元数据已保存到: {meta_path}")
    print(f"字数: {len(transcript)}")

    # 多模态增强（可选）
    if (args.screenshot or args.visual) and not _VISUAL_AVAILABLE:
        print("\n[VISUAL] 警告: 视觉功能依赖未安装，跳过")
        print("[VISUAL] 安装: pip install pillow")
    elif args.screenshot or args.visual:
        try:
            full_config = va.load_visual_config(args.model_config)
            model_config = full_config.get("default", {})
            visual_config = full_config.get("visual", model_config)
            grid_cfg = full_config.get("grid", {})
            screenshot_cfg = full_config.get("screenshot", {})

            print(f"\n{'='*40}")
            if args.screenshot:
                note = run_screenshot_mode(bvid, title, transcript,
                                           model_config, screenshot_cfg,
                                           args.vault, cookie)
                if note:
                    print(f"截图笔记: {note}")
            if args.visual:
                note = run_visual_mode(bvid, title, transcript,
                                       visual_config, grid_cfg,
                                       args.vault, cookie)
                if note:
                    print(f"视觉分析笔记: {note}")
        except Exception as e:
            print(f"\n[VISUAL] 多模态分析失败: {e}")
            print("[VISUAL] 字幕已正常生成，可忽略此错误")


if __name__ == "__main__":
    main()
