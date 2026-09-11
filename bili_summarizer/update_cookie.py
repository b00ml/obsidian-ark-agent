"""B站 Cookie 一键更新工具（交互式）。

用法：浏览器 F12 → 网络/应用面板复制完整 Cookie 串（或 SESSDATA/bili_jct 等键值对所在行），
运行 `python update_cookie.py` 后粘贴整行回车——自动解析并写入 bilibili_cookie.json。
所需键：SESSDATA（必需）/ bili_jct / buvid3 / DedeUserID / ac_time_value（可选）。
凭据只落本地 bilibili_cookie.json（已在 .gitignore），不要粘贴到任何对话或线上服务。
"""
import json
import re
from pathlib import Path

COOKIE_FILE = Path(__file__).parent / "bilibili_cookie.json"


def parse_cookie_header(raw: str) -> dict:
    """把浏览器复制的整行 Cookie 串解析成 {name: value}。"""
    out: dict = {}
    for part in raw.replace("\n", "").split(";"):
        if "=" not in part:
            continue
        k, _, v = part.strip().partition("=")
        if k:
            out[k.strip()] = v.strip()
    return out


def main() -> None:
    print("粘贴完整 Cookie 串（F12 → 应用/网络 → 复制 Cookie 值，整行），回车结束：")
    chunks = []
    while True:
        line = input()
        if not line.strip():
            break
        chunks.append(line)
    raw = "".join(chunks)
    if not raw.strip():
        print("未输入任何内容，退出。")
        return

    parsed = parse_cookie_header(raw)
    data = {}
    if COOKIE_FILE.exists():
        try:
            data = json.loads(COOKIE_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}

    mapping = {"SESSDATA": "SESSDATA", "bili_jct": "BILI_JCT",
               "buvid3": "BUVID3", "DedeUserID": "DedeUserID",
               "ac_time_value": "ac_time_value"}
    updated = []
    for src, dst in mapping.items():
        v = parsed.get(src)
        if v:
            data[dst] = v
            updated.append(dst)
    if "SESSDATA" not in data:
        print("⚠️ 未解析到 SESSDATA——确认复制的是登录后的 B站 Cookie。")
        return
    COOKIE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写入 {COOKIE_FILE.name}，更新字段: {updated}")
    print("SESSDATA 前缀:", data["SESSDATA"][:12] + "***")
    print("验证：跑一次 bili_transcript 或等下一次对话，观察是否还报 [SUBTITLE] 登录态过期。")


if __name__ == "__main__":
    main()
