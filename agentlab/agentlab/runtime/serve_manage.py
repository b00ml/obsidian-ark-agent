"""serve 生命周期管理：单实例守护（OPT-091）。

补「serve 是独立 daemon，需有人管理」的短板：
- start：pidfile 单实例。端口已被活进程占 → 报「已在运行」退出（fail-closed，
  不让两个实例抢端口）；pidfile 陈旧（死进程遗留）→ 继承清理再起。
- stop：读 pidfile + 按端口反查归属进程，`taskkill /F /T` 连子树一起清——
  避免「shell 停了、python 子进程仍占端口」的孤儿。
- status：pid / 端口监听 / /health 三态，供人类与监控判断。

纯函数（接受 cfg 对象）便于单测，不启动真实服务即可覆盖孤儿/端口/pidfile 逻辑。
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def _base_dir(cfg) -> Path:
    """pidfile / serve.log 落点：与 sessions 同级（trace_dir 的父目录）。"""
    return Path(getattr(cfg, "trace_dir", "logs/trace")).parent


def pid_file(cfg) -> Path:
    return _base_dir(cfg) / "serve.pid"


def log_file(cfg) -> Path:
    return _base_dir(cfg) / "serve.log"


def _listen(host: str, port: int) -> tuple[str, int]:
    host = host or "127.0.0.1"
    return host, int(port)


def port_in_use(host: str, port: int) -> bool:
    host, port = _listen(host, port)
    with socket.socket() as s:
        s.settimeout(0.3)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False


def _run_text(cmd: list[str], timeout: float):
    """运行外部命令并安全取回文本。netstat/tasklist/taskkill 输出为系统 ANSI 编码
    （中文 Windows 为 GBK），在 UTF-8 模式（PYTHONUTF8=1 / 系统级 UTF-8）下
    text=True 按默认编码解码会崩掉读线程 → stdout=None；errors="replace"
    保证解码永不失败（判活只匹配 LISTENING / PID 数字等 ASCII 子串）。"""
    return subprocess.run(cmd, capture_output=True, timeout=timeout,
                          encoding="utf-8", errors="replace")


def _port_owner(host: str, port: int) -> int | None:
    """windows 下按 `netstat -ano` 反查监听 <port> 的 PID；跨平台均可安全调用。"""
    _, port = _listen(host, port)
    try:
        r = _run_text(["netstat", "-ano"], timeout=5)
    except Exception:
        return None
    for line in (r.stdout or "").splitlines():
        if "LISTENING" not in line:
            continue
        parts = line.split()
        # 本地地址在 parts[1]（IPv4）或 parts[1]/equv 处，含 "[::]:8643" 形式；取尾 PID
        addrs = (parts[1], parts[2]) if len(parts) >= 3 else (parts[1],)
        for addr in addrs:
            if addr.endswith(f":{port}"):
                try:
                    return int(parts[-1])
                except ValueError:
                    continue
    return None


def alive(pid: int | None) -> bool:
    """pid 是否存活。windows 用 tasklist 判定（os.kill(pid,0) 在 win 上对信号 0 不可靠）。"""
    if not pid:
        return False
    is_win = sys.platform == "win32"
    try:
        if is_win:
            r = _run_text(["tasklist", "/FI", f"PID eq {pid}"], timeout=5)
            return str(pid) in (r.stdout or "")
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def read_pid(cfg) -> int | None:
    p = pid_file(cfg)
    if not p.exists():
        return None
    try:
        return int(p.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return None


def write_pid(cfg, pid: int | None) -> None:
    p = pid_file(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    if pid is None:
        p.unlink(missing_ok=True)
    else:
        p.write_text(str(pid), encoding="utf-8")


def _health_ok(host: str, port: int, token: str = "") -> bool:
    url = f"http://{host}:{port}/health"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    try:
        with urllib.request.urlopen(req, timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def cmd_status(cfg) -> dict:
    sc = getattr(cfg, "serve", None)
    host, port = _listen(getattr(sc, "host", "127.0.0.1"), getattr(sc, "port", 8643))
    pid = read_pid(cfg)
    pid_alive = alive(pid)
    listening = port_in_use(host, port)
    owner = _port_owner(host, port) if listening else None
    # 权威状态以端口为准：能监听即有实例在服务
    running = listening
    return {
        "running": running,
        "pid": owner if running else (pid if pid_alive else None),
        "port": port,
        "listening": listening,
        "pidfile_pid": pid,
        "pid_alive": pid_alive,
        "health": _health_ok(host, port) if running else False,
        "token": getattr(sc, "token", ""),
    }


def cmd_start(cfg, config_arg: str | None = None, wait_ok: float = 2.0) -> dict:
    """单实例启动。端口被活进程占 → 报错退出；死 pidfile → 继承清理。"""
    sc = getattr(cfg, "serve", None)
    host, port = _listen(getattr(sc, "host", "127.0.0.1"), getattr(sc, "port", 8643))
    # fail-closed：无 token 拒绝（与 serve.start 一致）
    if not getattr(sc, "token", ""):
        raise SystemExit("[SERVE] token 未配置（fail-closed）。请在 config.json serve.token 指定钥匙。")

    if port_in_use(host, port):
        owner = _port_owner(host, port)
        if owner and alive(owner):
            raise SystemExit(
                f"[SERVE] 已在运行 pid={owner} @ {host}:{port}。"
                "如需重启：先 `serve-manage stop` 再 `start`。"
            )
        raise SystemExit(
            f"[SERVE] 端口 {host}:{port} 被进程 pid={owner} 占用但判活失败，"
            "请先手工核对后 `serve-manage stop`。"
        )

    stale = read_pid(cfg)
    if stale and not alive(stale):
        # 陈旧 pidfile（死进程遗留）→ 继承清理后接手端口
        print(f"[SERVE] 清理陈旧 pidfile（pid={stale} 已退出）")
        write_pid(cfg, None)

    _base_dir(cfg).mkdir(parents=True, exist_ok=True)
    stdout = stderr = log_file(cfg).open("a", encoding="utf-8")
    argv = ["-m", "agentlab.runtime.serve"]
    if config_arg:
        argv += ["-c", str(config_arg)]
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    child = subprocess.Popen(
        [sys.executable, *argv],
        cwd=str(Path.cwd()),
        stdout=stdout,
        stderr=stderr,
        creationflags=flags,
    )
    write_pid(cfg, child.pid)
    ok = _health_ok(host, port)
    if not ok:
        deadline = time.time() + max(wait_ok, 0)
        while time.time() < deadline and not ok:
            time.sleep(0.3)
            ok = _health_ok(host, port)
    if ok:
        print(f"[SERVE] 已启动 pid={child.pid} @ http://{host}:{port}")
    else:
        print(f"[SERVE] 已启动 pid={child.pid}，但 /health 未就绪（日志见 {log_file(cfg)}）")
    return {"pid": child.pid, "port": port, "ready": ok, "log": str(log_file(cfg))}


def cmd_stop(cfg) -> dict:
    """连子树终止：pidfile pid 与端口归属进程一并清，杜绝孤儿。"""
    sc = getattr(cfg, "serve", None)
    host, port = _listen(getattr(sc, "host", "127.0.0.1"), getattr(sc, "port", 8643))
    targets: set[int] = set()
    pid = read_pid(cfg)
    if pid and alive(pid):
        targets.add(pid)
    if port_in_use(host, port):
        owner = _port_owner(host, port)
        if owner:
            targets.add(owner)
    if not targets:
        write_pid(cfg, None)
        return {"stopped": False, "detail": "无运行中的 serve（端口空闲 / pid 无效）"}

    killed = []
    for t in targets:
        if sys.platform == "win32":
            _run_text(["taskkill", "/PID", str(t), "/F", "/T"], timeout=10)
        else:
            subprocess.run(["kill", "-TERM", str(t)], capture_output=True, timeout=5)
        killed.append(t)
    write_pid(cfg, None)
    # 端口应释放；短暂等待后校验
    deadline = time.time() + 3
    while time.time() < deadline and port_in_use(host, port):
        time.sleep(0.2)
    return {"stopped": True, "pids": killed,
            "port_released": not port_in_use(host, port)}


def main(argv: list[str] | None = None) -> int:
    import argparse

    from agentlab.runtime.config import load_config, DEFAULT_CONFIG

    p = argparse.ArgumentParser(prog="serve-manage", description="serve 生命周期管理（单实例守护）")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("start", "stop", "status"):
        sp = sub.add_parser(name)
        sp.add_argument("-c", "--config", default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config) if args.config else (load_config(DEFAULT_CONFIG)
                                                        if Path(DEFAULT_CONFIG).exists() else load_config(None))
    cfg_path = args.config or (DEFAULT_CONFIG if Path(DEFAULT_CONFIG).exists() else None)

    if args.cmd == "status":
        import json
        print(json.dumps(cmd_status(cfg), ensure_ascii=False, indent=2))
    elif args.cmd == "start":
        cmd_start(cfg, config_arg=cfg_path)
    else:
        d = cmd_stop(cfg)
        print(f"[SERVE] stop: {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())